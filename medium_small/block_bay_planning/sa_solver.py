from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .area_plan_solver import AreaPlanSolverMixin, MediumAssignment
from .models import ProblemData, SAConfig, SolveResult


class SmallPlanInfeasible(ValueError):
    def __init__(self, message: str, voyage_id: str, status: str, port: str, size: str, area_no: str) -> None:
        super().__init__(message)
        self.voyage_id = voyage_id
        self.status = status
        self.port = port
        self.size = size
        self.area_no = area_no


class SimulatedAnnealingSolver(AreaPlanSolverMixin):
    """Medium-plan SA with small-plan proxy scoring and constructive bay output."""

    def __init__(self, problem: ProblemData, config: SAConfig | None = None) -> None:
        """预计算箱区、贝位和小计划连续 6 小贝索引。"""
        self.problem = problem
        self.config = config or SAConfig()
        self.random = random.Random(self.config.seed)
        self.bays = problem.bays
        self.energy_history: list[dict] = []
        self.bays_by_area: dict[str, list[str]] = defaultdict(list)
        self.bays_by_block: dict[str, list[str]] = defaultdict(list)
        self.blocks_by_area: dict[str, list[str]] = defaultdict(list)
        self.block_area: dict[str, str] = {}
        self.block_bays: dict[str, tuple[str, ...]] = {}
        self.block_boundary_adjusted: dict[str, bool] = {}
        self.block_order: dict[str, int] = {}
        self.area_edge_blocks: dict[str, set[str]] = {}
        self.area_edge_bays: dict[str, set[str]] = {}
        self.area_edge_size_cap: Counter[tuple[str, str]] = Counter()
        self.area_six_block_count: Counter[str] = Counter()
        self.small_plan_area_feedback: Counter[tuple[str, str, str, str, str]] = Counter()
        self.small_plan_strict_area_feedback: Counter[tuple[str, str, str, str, str]] = Counter()
        self.area_total_cap: Counter[str] = Counter()
        self.area_size_cap: Counter[tuple[str, str]] = Counter()
        self.area_size_height_cap: Counter[tuple[str, str, str]] = Counter()
        self.block_total_cap: Counter[str] = Counter()
        self.block_size_cap: Counter[tuple[str, str]] = Counter()
        self._candidate_area_cache: dict[str, set[str]] = {}
        self.best_small_feasible_assignment: MediumAssignment | None = None
        self.best_small_feasible_iteration: int | None = None
        self.best_small_feasible_energy: float | None = None
        self.used_small_feasible_fallback: bool = False
        self.final_small_plan_failure: str = ""
        self.medium_initial_assignment_diagnostics: dict = {}
        self.small_unplaced_by_group: Counter[str] = Counter()
        self.small_plan_construction_mode: str = ""
        self.small_plan_medium_repair_added_boxes: int = 0
        self.medium_small_learned_area_size_caps: dict[tuple[str, str, str, str], int] = {}
        self.medium_small_feedback_rounds: list[dict] = []

        for key, bay in problem.bays.items():
            self.bays_by_area[bay.area_no].append(key)
            self.bays_by_block[bay.block_id].append(key)
            self.block_area[bay.block_id] = bay.area_no
            self.block_bays[bay.block_id] = bay.block_bays
            self.block_boundary_adjusted[bay.block_id] = bay.block_boundary_adjusted
            self.area_total_cap[bay.area_no] += bay.physical_capacity
            self.block_total_cap[bay.block_id] += bay.physical_capacity
            for size_mode, cap in bay.cap_by_size.items():
                self.area_size_cap[(bay.area_no, size_mode)] += cap
                self.block_size_cap[(bay.block_id, size_mode)] += cap

        for keys in self.bays_by_area.values():
            keys.sort(key=lambda k: problem.bays[k].bay_order)
        for block_id, keys in self.bays_by_block.items():
            keys.sort(key=lambda k: problem.bays[k].bay_order)
            self.block_order[block_id] = min(problem.bays[k].bay_order for k in keys)
        for block_id in sorted(self.bays_by_block, key=lambda b: (self.block_area[b], self.block_order[b])):
            self.blocks_by_area[self.block_area[block_id]].append(block_id)
        for area_no, block_ids in self.blocks_by_area.items():
            if block_ids:
                self.area_edge_blocks[area_no] = {block_ids[0], block_ids[-1]}
        for area_no, bay_keys in self.bays_by_area.items():
            if bay_keys:
                edge_keys = {bay_keys[0], bay_keys[-1]}
                last_key = bay_keys[-1]
                for bay_key in bay_keys:
                    if problem.bays[bay_key].large_bay_partner_key == last_key:
                        edge_keys.add(bay_key)
                self.area_edge_bays[area_no] = edge_keys
                for bay_key in self.area_edge_bays[area_no]:
                    for size_mode, cap in problem.bays[bay_key].cap_by_size.items():
                        self.area_edge_size_cap[(area_no, size_mode)] += cap
        for area_no in self.bays_by_area:
            self.area_six_block_count[area_no] = len(self._six_bay_blocks_by_area(area_no))
        self._build_small_proxy_height_capacity()
        self._candidate_area_cache = {
            group.group_id: self._compute_candidate_areas_for_group(group)
            for group in self.problem.groups
        }

    def solve(self) -> SolveResult:
        """依次求解中计划和小计划，并生成 CSV 行。"""
        small_plan_failures: list[str] = []
        max_retries = max(0, self.config.max_small_plan_retries)
        self.used_small_feasible_fallback = False
        self.final_small_plan_failure = ""
        for retry in range(max_retries + 1):
            self.random = random.Random(self.config.seed + retry)
            self.energy_history = []
            medium_assignment = self._solve_medium()
            try:
                medium_rows, small_rows = self._make_outputs(medium_assignment)
                break
            except ValueError as exc:
                if isinstance(exc, SmallPlanInfeasible):
                    self.small_plan_area_feedback[(exc.voyage_id, exc.status, exc.port, exc.size, exc.area_no)] += 1
                    self.small_plan_area_feedback[(exc.voyage_id, exc.status, "*", exc.size, exc.area_no)] += 1
                small_plan_failures.append(str(exc))
                self.final_small_plan_failure = str(exc)
                if self.best_small_feasible_assignment is not None:
                    medium_assignment = self._copy_assignment(self.best_small_feasible_assignment)
                    medium_rows, small_rows = self._make_outputs(medium_assignment)
                    self.used_small_feasible_fallback = True
                    break
                if retry == max_retries:
                    raise RuntimeError(
                        "small plan is infeasible under the medium-plan area allocation "
                        f"after {max_retries + 1} simulated-annealing attempt(s): {exc}"
                    ) from exc
        medium_energy = self.medium_energy(medium_assignment)
        medium_unplaced = self._medium_unplaced_by_group(medium_assignment)
        medium_small_shortage = self._medium_small_coarse_shortage()
        big_plan_inheritance = self._medium_big_plan_inheritance_stats(medium_rows)
        final_medium_counter = self._medium_counter_from_rows(medium_rows)
        final_medium_output_energy_components = self._medium_small_feedback_energy_components(final_medium_counter)
        diagnostics = {
            "iterations": self.config.iterations,
            "user_area_constraints": getattr(self.problem, "user_area_constraint_summary", {}),
            "attribute_rules": self.problem.attribute_rules.as_dict() if hasattr(self.problem.attribute_rules, "as_dict") else {},
            "medium_initial_assignment": getattr(self, "medium_initial_assignment_diagnostics", {}),
            "medium_initial_assignment_attempts": self.config.medium_initial_assignment_attempts,
            "medium_initial_assignment_method": getattr(
                self,
                "medium_initial_assignment_diagnostics",
                {},
            ).get("method"),
            "medium_small_feedback_iterations": getattr(
                self.config,
                "medium_small_feedback_iterations",
                0,
            ),
            "medium_small_feedback_rounds": self.medium_small_feedback_rounds,
            "medium_small_feedback_learned_cap_count": len(self.medium_small_learned_area_size_caps),
            "small_plan_retry_count": len(small_plan_failures),
            "small_plan_retry_failures": small_plan_failures,
            "used_small_feasible_fallback": self.used_small_feasible_fallback,
            "small_feasible_fallback_iteration": self.best_small_feasible_iteration,
            "small_feasible_fallback_energy": (
                round(self.best_small_feasible_energy, 4)
                if self.best_small_feasible_energy is not None
                else None
            ),
            "final_small_plan_failure_before_fallback": self.final_small_plan_failure,
            "small_plan_area_feedback": {
                "|".join(key): count for key, count in sorted(self.small_plan_area_feedback.items())
            },
            "small_plan_strict_area_feedback": {
                "|".join(key): count for key, count in sorted(self.small_plan_strict_area_feedback.items())
            },
            "sa_energy": round(medium_energy, 4),
            "medium_energy": round(medium_energy, 4),
            "final_medium_output_energy_components": final_medium_output_energy_components,
            "medium_big_plan_inheritance": big_plan_inheritance,
            "user_required_area_usage": self._user_required_area_usage(medium_rows, small_rows),
            "user_area_constraint_violations": self._user_area_constraint_violations(medium_rows, small_rows),
            "small_plan_proxy_energy": (
                round(self._small_plan_proxy_energy_from_assignment(medium_assignment), 4)
                if getattr(self.config, "small_plan_proxy_every", 0) > 0
                else 0.0
            ),
            "medium_box_count": sum(group.demand for group in self.problem.groups),
            "medium_decision_group_count": len(self.problem.groups),
            "medium_doc_floor_added_boxes": getattr(self.problem, "medium_doc_floor_added_boxes", 0),
            "medium_doc_floor_added_groups": getattr(self.problem, "medium_doc_floor_added_groups", 0),
            "medium_doc_floor_shifted_boxes": getattr(self.problem, "medium_doc_floor_shifted_boxes", 0),
            "medium_doc_floor_shifted_groups": getattr(self.problem, "medium_doc_floor_shifted_groups", 0),
            "medium_doc_floor_by_coarse_group": getattr(self.problem, "medium_doc_floor_by_coarse_group", {}),
            "medium_doc_floor_added_by_coarse_group": getattr(
                self.problem,
                "medium_doc_floor_added_by_coarse_group",
                {},
            ),
            "medium_doc_floor_shifted_by_coarse_group": getattr(
                self.problem,
                "medium_doc_floor_shifted_by_coarse_group",
                {},
            ),
            "medium_small_coarse_floor_shortage_boxes": sum(medium_small_shortage.values()),
            "medium_small_coarse_floor_shortage_by_group": {
                key: qty for key, qty in sorted(medium_small_shortage.items()) if qty > 0
            },
            "group_count": len(self.problem.groups),
            "bay_count": len(self.problem.bays),
            "area_count": len(self.bays_by_area),
            "berth_distance_count": len(self.problem.berth_distances),
            "medium_row_count": len(medium_rows),
            "small_row_count": len(small_rows),
            "small_plan_used_six_bay_block_count": len(
                {row.get("six_bay_block_id") for row in small_rows if row.get("six_bay_block_id")}
            ),
            "small_plan_construction_mode": self.small_plan_construction_mode,
            "small_plan_medium_repair_added_boxes": self.small_plan_medium_repair_added_boxes,
            "small_doc_group_count": len(self.problem.small_groups),
            "medium_unplaced_boxes": sum(medium_unplaced.values()),
            "medium_unplaced_by_group": {key: qty for key, qty in sorted(medium_unplaced.items()) if qty > 0},
            "medium_unplaced_group_details": self._medium_unplaced_group_details(medium_unplaced),
            "small_unplaced_boxes": sum(self.small_unplaced_by_group.values()),
            "small_unplaced_by_group": {
                key: qty for key, qty in sorted(self.small_unplaced_by_group.items()) if qty > 0
            },
            "small_unplaced_group_details": self._small_unplaced_group_details(self.small_unplaced_by_group),
            "planning_hierarchy": "simulated annealing assigns the medium plan to yard areas; small plan uses doc containers and assigns fine groups to bays by a post-SA constructive heuristic",
            "medium_plan_space_policy": "medium plan assigns only voyage/flow/discharge-port/size quantities by yard area; bays and six-small-bay groups belong only to the small plan",
            "large_box_edge_policy": "small-plan hard constraint: the first and last bay of an area cannot receive 20ft boxes",
            "size_overlap_policy": "20/40/45 capacities are size-specific by YBY_ENABLECSIZECD; total bay load also cannot exceed physical empty-slot capacity",
            "size_modes": ["20", "40", "45"],
            "size_policy": "20ft, 40ft, and 45ft are planned and capacity-checked separately using YBY_ENABLECSIZECD",
            "area_size_quota_count": len(self.problem.area_size_quota),
            "big_plan_volume_policy": (
                "big plan volume is treated as the full predicted area/size pattern; "
                "medium demand is calculated separately by the rolling planning-time ratio and distributed by that pattern"
            ),
            "voyage_windows": {
                voyage_id: {
                    "window_start": window[0].isoformat(sep=" "),
                    "window_end": window[1].isoformat(sep=" "),
                }
                for voyage_id, window in self.problem.voyage_windows.items()
            },
            "operation_conflict_policy": (
                "medium-plan soft constraint: avoid areas with loading during the voyage window; "
                "prefer areas with loading in the 24h after the voyage window"
            ),
            "berth_distance_policy": "medium-plan soft constraint: prefer yard areas closer to the voyage berth",
            "attribute_balance_policy": "medium-plan soft constraint: keep the aggregate voyage/flow/size area pattern close to the big plan; large coarse groups balance across used areas",
            "medium_concentrated_group_threshold": self.config.medium_concentrated_group_threshold,
            "medium_large_group_min_area_boxes": self.config.medium_large_group_min_area_boxes,
            "medium_small_group_policy": (
                "medium-plan coarse groups at or below the threshold prefer concentrated yard-area assignment; "
                "larger groups balance boxes across actually used yard areas and penalize small area fragments"
            ),
            "concentration_penalties": {
                "fine_group_area": self.config.small_plan_group_area_split_penalty,
                "medium_concentrated_group_threshold": self.config.medium_concentrated_group_threshold,
                "medium_small_group_area_split": self.config.medium_small_group_area_split_penalty,
                "medium_small_group_fragment": self.config.medium_small_group_fragment_penalty,
                "medium_large_group_min_area_boxes": self.config.medium_large_group_min_area_boxes,
                "medium_large_group_small_area": self.config.medium_large_group_small_area_penalty,
            },
            "inheritance_penalties": {
                "big_plan_area_deviation": self.config.big_plan_area_deviation_penalty,
                "big_plan_fallback_tier": self.config.big_plan_fallback_tier_penalty,
                "medium_small_feedback_cap": self.config.medium_small_feedback_cap_penalty,
            },
            "operation_penalties": {
                "berth_distance": self.config.berth_distance_penalty,
                "active_loading_area": self.config.active_loading_area_penalty,
                "post_window_loading_area_reward": self.config.post_window_loading_area_reward,
            },
            "big_plan_area_policy": (
                "medium-plan hard candidates are big-plan-used areas plus OF-function yard areas "
                "with available capacity; big-plan voyage/flow/area/size new_qty is a hard upper "
                "bound, and fallback areas are used in ordered inheritance tiers"
            ),
            "tops_policy": "TOPS bay ranges for non-target voyages are closed before medium and small planning; TOPS records for target voyages are ignored",
            "small_plan_six_bay_policy": (
                "small plan prefers dynamic six-small-bay continuous blocks first for one fine group "
                "and then for the same voyage/flow/discharge-port/size coarse group; different sizes "
                "may share that preferred block but cannot share the same bay"
            ),
            "small_plan_proxy_policy": (
                "optional lightweight small-plan proxy score for doc-container bay capacity, "
                "size/height bay capacity, 45ft edge capacity, and six-small-bay block affinity; "
                "disabled when small_plan_proxy_every=0"
            ),
            "small_plan_edge_policy": (
                "hard constraints: 20ft cannot use area edge bays; 45ft must use area edge bays; "
                "if an area has 45ft doc boxes, its edge bays are reserved for 45ft"
            ),
            "small_plan_height_policy": "hard constraint: boxes with different heights cannot share the same bay",
            "small_plan_prestow_policy": "pre-stow isolation soft preference is implemented through a reserved pre_stow flag; current data has no pre-stow marker",
            "small_plan_special_stow_policy": (
                "small-plan soft constraint is reserved for explicit fixed special-stow markers; "
                "current data has no such marker, so no special-stow category is inferred"
            ),
            "tops_reserved_slot_count": self.problem.tops_reserved_slot_count,
            "tops_closed_bay_count": self.problem.tops_closed_bay_count,
            "misplaced_bay_exclusion_ratio": self.problem.misplaced_bay_exclusion_ratio,
            "misplaced_excluded_bay_count": self.problem.misplaced_excluded_bay_count,
            "berth_by_voyage": self.problem.berth_by_voyage,
            "business_special_codes": sorted(self.problem.business_special_codes),
            "planning_time": self.problem.planning_time.isoformat(sep=" "),
            "horizon_hours": self.problem.horizon_hours,
            "target_voyages": self.problem.target_voyages,
        }
        diagnostics["energy_record_every"] = self.config.progress_every
        diagnostics["log_every"] = self.config.log_every
        diagnostics["small_plan_check_every"] = self.config.small_plan_check_every
        diagnostics["small_plan_proxy_every"] = self.config.small_plan_proxy_every
        diagnostics["energy_record_count"] = len(self.energy_history)
        return SolveResult(
            medium_assignment,
            diagnostics["medium_energy"],
            medium_rows,
            small_rows,
            diagnostics,
            self.energy_history,
        )

    def _medium_big_plan_inheritance_stats(self, medium_rows: list[dict]) -> dict[str, float | int]:
        actual: Counter[tuple[str, str, str, str]] = Counter()
        for row in medium_rows:
            voyage_id = str(row.get("voyage_id", ""))
            flow = str(row.get("flow", ""))
            area_no = str(row.get("area_no", ""))
            size = str(row.get("size", ""))
            big_size = "40" if size == "45" else size if size in {"20", "40"} else "40"
            qty = int(row.get("planned_boxes", 0) or 0)
            if qty > 0:
                actual[(voyage_id, flow, area_no, big_size)] += qty

        quota = Counter({key: int(qty) for key, qty in self.problem.area_size_quota.items() if qty > 0})
        total = sum(actual.values())
        inherited = sum(min(qty, quota.get(key, 0)) for key, qty in actual.items())
        deviated = max(0, total - inherited)
        return {
            "total_boxes": total,
            "inherited_boxes": inherited,
            "deviated_boxes": deviated,
            "inheritance_ratio": round(inherited / total, 6) if total else 1.0,
            "deviation_ratio": round(deviated / total, 6) if total else 0.0,
        }

    def _temperature(self, iteration: int) -> float:
        ratio = iteration / max(1, self.config.iterations)
        return self.config.initial_temperature * (
            self.config.final_temperature / self.config.initial_temperature
        ) ** ratio

    def _should_record_energy(self, iteration: int) -> bool:
        """?????????????????"""
        if iteration == 0 or iteration == self.config.iterations:
            return True
        return self.config.progress_every > 0 and iteration % self.config.progress_every == 0

    def _record_energy_history(
        self,
        phase: str,
        iteration: int,
        current_energy: float,
        best_energy: float,
        temperature: float,
        accepted_count: int,
    ) -> None:
        """?????????????"""
        self.energy_history.append(
            {
                "phase": phase,
                "iteration": iteration,
                "current_energy": round(current_energy, 6),
                "best_energy": round(best_energy, 6),
                "temperature": round(temperature, 6),
                "accepted_count": accepted_count,
            }
        )

    def _is_area_edge_block(self, block_id: str) -> bool:
        area_no = self.block_area.get(block_id, "")
        return block_id in self.area_edge_blocks.get(area_no, set())

    def _is_area_edge_bay(self, bay_key: str) -> bool:
        bay = self.bays[bay_key]
        return bay_key in self.area_edge_bays.get(bay.area_no, set())

    def _bay_footprint_keys(self, bay_key: str, size: str) -> tuple[str, ...]:
        bay = self.bays[bay_key]
        if size in {"40", "45"}:
            if not bay.large_bay_partner_key:
                return ()
            return (bay_key, bay.large_bay_partner_key)
        return (bay_key,)

    def _initial_medium_assignment(self) -> MediumAssignment:
        feedback_rounds = max(0, int(getattr(self.config, "medium_small_feedback_iterations", 0) or 0))
        if feedback_rounds <= 0:
            return super()._initial_medium_assignment()

        self.medium_small_learned_area_size_caps = {}
        self.medium_small_feedback_rounds = []
        best_assignment: MediumAssignment | None = None
        best_score: tuple[int, int, int, int, int] | None = None
        best_round = 0

        for round_index in range(feedback_rounds + 1):
            assignment = super()._initial_medium_assignment()
            medium_counter = self._medium_counter_from_assignment(assignment)
            try:
                small_rows = self._make_feasible_doc_small_rows(medium_counter)
                repaired_counter = self._repair_medium_counter_to_cover_small_rows(medium_counter, small_rows)
            except ValueError as exc:
                self.medium_small_feedback_rounds.append(
                    {
                        "round": round_index,
                        "status": "small_plan_failed",
                        "error": str(exc),
                        "learned_cap_count": len(self.medium_small_learned_area_size_caps),
                    }
                )
                if best_assignment is None:
                    best_assignment = self._copy_assignment(assignment)
                break

            score = self._medium_small_feedback_score(repaired_counter)
            energy_components = self._medium_small_feedback_energy_components(repaired_counter)
            learned_caps = self._learn_medium_small_area_size_caps(medium_counter, repaired_counter)
            changed = (
                self._merge_medium_small_learned_caps(learned_caps)
                if round_index < feedback_rounds
                else False
            )
            stats = self._medium_big_plan_inheritance_stats(self._medium_rows_from_counter(repaired_counter))
            self.medium_small_feedback_rounds.append(
                {
                    "round": round_index,
                    "status": "ok",
                    "score": list(score),
                    "inheritance_ratio": stats["inheritance_ratio"],
                    "inherited_boxes": stats["inherited_boxes"],
                    "deviated_boxes": stats["deviated_boxes"],
                    "new_cap_count": len(learned_caps),
                    "learned_cap_count": len(self.medium_small_learned_area_size_caps),
                    "energy_components": energy_components,
                    "changed": changed,
                }
            )
            self._log_medium_small_feedback_round(self.medium_small_feedback_rounds[-1])
            if best_score is None or score < best_score:
                best_score = score
                best_assignment = self._copy_assignment(assignment)
                best_round = round_index
            if round_index >= feedback_rounds or not changed:
                break

        diagnostics = dict(getattr(self, "medium_initial_assignment_diagnostics", {}))
        diagnostics["medium_small_feedback_iterations"] = feedback_rounds
        diagnostics["medium_small_feedback_selected_round"] = best_round
        diagnostics["medium_small_feedback_learned_cap_count"] = len(self.medium_small_learned_area_size_caps)
        diagnostics["medium_small_feedback_rounds"] = self.medium_small_feedback_rounds
        self.medium_initial_assignment_diagnostics = diagnostics
        return best_assignment if best_assignment is not None else super()._initial_medium_assignment()

    def _medium_small_feedback_energy_components(self, medium_counter: Counter[tuple]) -> dict[str, float]:
        assignment = self._assignment_from_medium_counter(medium_counter)
        if assignment is None:
            return {}
        components = self.medium_energy_components(
            assignment,
            include_small_proxy=getattr(self.config, "small_plan_proxy_every", 0) > 0,
        )
        return {
            key: round(float(value), 4)
            for key, value in sorted(components.items())
            if abs(float(value)) > 1e-9 or key == "total"
        }

    @staticmethod
    def _log_medium_small_feedback_round(round_info: dict) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        components = round_info.get("energy_components", {})
        parts = [
            f"[{stamp}] medium-small feedback round={round_info.get('round')} "
            f"inherit={round_info.get('inherited_boxes')}/{round_info.get('inherited_boxes', 0) + round_info.get('deviated_boxes', 0)}",
            f"ratio={round_info.get('inheritance_ratio')}",
            f"deviated={round_info.get('deviated_boxes')}",
            f"learned_caps={round_info.get('learned_cap_count')}",
            f"total={components.get('total')}",
            f"fallback={components.get('big_plan_fallback_tier', 0.0)}",
            f"learned_cap={components.get('medium_small_learned_cap', 0.0)}",
            f"big_dev={components.get('big_plan_deviation', 0.0)}",
            f"shape={components.get('group_area_shape', 0.0)}",
            f"split={components.get('group_area_split', 0.0)}",
        ]
        if "small_plan_proxy" in components:
            parts.append(f"small_proxy={components.get('small_plan_proxy', 0.0)}")
        print(" ".join(parts), flush=True)

    def _learn_medium_small_area_size_caps(
        self,
        medium_counter: Counter[tuple],
        repaired_counter: Counter[tuple],
    ) -> dict[tuple[str, str, str, str], int]:
        medium_usage = self._area_size_usage_from_medium_counter(medium_counter)
        repaired_usage = self._area_size_usage_from_medium_counter(repaired_counter)
        learned: dict[tuple[str, str, str, str], int] = {}
        for key, qty in medium_usage.items():
            quota = int(self.problem.area_size_quota.get(key, 0) or 0)
            if quota <= 0:
                continue
            repaired_qty = int(repaired_usage.get(key, 0) or 0)
            if qty > repaired_qty and repaired_qty < quota:
                learned[key] = max(0, repaired_qty)
        return learned

    def _merge_medium_small_learned_caps(self, learned_caps: dict[tuple[str, str, str, str], int]) -> bool:
        changed = False
        for key, cap in learned_caps.items():
            old = self.medium_small_learned_area_size_caps.get(key)
            if old is None or cap < old:
                self.medium_small_learned_area_size_caps[key] = cap
                changed = True
        return changed

    def _area_size_usage_from_medium_counter(self, medium_counter: Counter[tuple]) -> Counter[tuple[str, str, str, str]]:
        usage: Counter[tuple[str, str, str, str]] = Counter()
        for (voyage_id, flow, _port, size, area_no), qty in medium_counter.items():
            if qty <= 0:
                continue
            usage[(voyage_id, flow, area_no, self._big_plan_size(size))] += int(qty)
        return usage

    def _medium_small_feedback_score(self, medium_counter: Counter[tuple]) -> tuple[int, int, int, int, int]:
        usage = self._area_size_usage_from_medium_counter(medium_counter)
        tier_boxes: Counter[int] = Counter()
        all_big_areas = {row.area_no for row in self.problem.big_plan if row.planned_boxes > 0}
        same_voyage_size_areas: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
        for (voyage_id, flow, area_no, size), quota in self.problem.area_size_quota.items():
            if quota > 0:
                same_voyage_size_areas[(voyage_id, flow, size)].add(area_no)
        for (voyage_id, flow, area_no, size), qty in usage.items():
            inherited = min(qty, int(self.problem.area_size_quota.get((voyage_id, flow, area_no, size), 0) or 0))
            if inherited > 0:
                tier_boxes[0] += inherited
            rest = qty - inherited
            if rest <= 0:
                continue
            if area_no in same_voyage_size_areas.get((voyage_id, flow, size), set()):
                tier_boxes[1] += rest
            elif area_no in all_big_areas:
                tier_boxes[2] += rest
            else:
                tier_boxes[3] += rest
        return (
            -tier_boxes[0],
            tier_boxes[3],
            tier_boxes[2],
            tier_boxes[1],
            sum(usage.values()),
        )

    def _make_outputs(self, medium_assignment: MediumAssignment) -> tuple[list[dict], list[dict]]:
        """汇总中计划和小计划输出行。"""
        medium_counter = self._medium_counter_from_assignment(medium_assignment)

        self.small_unplaced_by_group = Counter()
        small_rows = self._make_feasible_doc_small_rows(medium_counter)
        repaired_medium_counter = self._repair_medium_counter_to_cover_small_rows(medium_counter, small_rows)
        medium_rows = self._medium_rows_from_counter(repaired_medium_counter)
        return medium_rows, small_rows

    def _medium_rows_from_counter(self, medium_counter: Counter[tuple]) -> list[dict]:
        medium_rows = []
        for key, count in sorted(medium_counter.items()):
            if count <= 0:
                continue
            voyage_id, flow, port, size, area_no = key
            window_start, window_end = self.problem.voyage_windows[voyage_id]
            active_loading = self._area_has_loading_during_window(voyage_id, area_no)
            post_loading = self._area_has_loading_after_window(voyage_id, area_no)
            medium_rows.append(
                {
                    "plan_level": "medium",
                    "voyage_id": voyage_id,
                    "flow": flow,
                    "port": port,
                    "size": size,
                    "window_start": window_start.isoformat(sep=" "),
                    "window_end": window_end.isoformat(sep=" "),
                    "area_loading_during_window": active_loading,
                    "area_loading_after_window_24h": post_loading,
                    "area_no": area_no,
                    "planned_boxes": count,
                }
            )
        return medium_rows

    def _make_feasible_doc_small_rows(self, medium_counter: Counter[tuple]) -> list[dict]:
        try:
            rows = self._make_doc_small_rows(medium_counter, strict=True)
            self.small_plan_construction_mode = "strict_medium_area_inheritance"
            return rows
        except SmallPlanInfeasible as exc:
            self.final_small_plan_failure = str(exc)

        try:
            rows = self._make_direct_small_rows(medium_counter)
        except SmallPlanInfeasible as exc:
            self.small_plan_area_feedback[(exc.voyage_id, exc.status, exc.port, exc.size, exc.area_no)] += 1
            self.small_plan_area_feedback[(exc.voyage_id, exc.status, "*", exc.size, exc.area_no)] += 1
            self.final_small_plan_failure = str(exc)
            rows = self._make_doc_small_rows(medium_counter, strict=False)
            self.small_plan_construction_mode = "partial_strict_medium_area_inheritance"
            return rows
        self.small_plan_construction_mode = "direct_bay_repair"
        return rows

    def _medium_unplaced_by_group(self, medium_assignment: MediumAssignment) -> Counter[str]:
        unplaced: Counter[str] = Counter()
        groups_by_id = {group.group_id: group for group in self.problem.groups}
        for group_id, group in groups_by_id.items():
            assigned = sum(medium_assignment.get(group_id, Counter()).values())
            shortage = max(0, group.demand - assigned)
            if shortage:
                unplaced[group_id] = shortage
        return unplaced

    def _medium_small_coarse_shortage(self) -> Counter[str]:
        medium_by_key: Counter[tuple[str, str, str, str]] = Counter()
        for group in self.problem.groups:
            medium_by_key[(group.voyage_id, group.status, group.port, group.size_mode)] += group.demand
        small_by_key: Counter[tuple[str, str, str, str]] = Counter()
        for group in self.problem.small_groups:
            small_by_key[(group.voyage_id, group.status, group.port, group.size)] += group.demand
        shortage: Counter[str] = Counter()
        for key, small_qty in small_by_key.items():
            missing = small_qty - medium_by_key.get(key, 0)
            if missing > 0:
                shortage["|".join(key)] = missing
        return shortage

    def _medium_counter_from_assignment(self, medium_assignment: MediumAssignment) -> Counter[tuple]:
        medium_counter: Counter[tuple] = Counter()
        groups_by_id = {group.group_id: group for group in self.problem.groups}

        for group_id, area_counts in medium_assignment.items():
            group = groups_by_id[group_id]
            attrs = self._medium_attrs(group)
            for area_no, qty in area_counts.items():
                if qty > 0:
                    medium_counter[attrs + (area_no,)] += qty
        return medium_counter

    def _medium_counter_from_rows(self, medium_rows: list[dict]) -> Counter[tuple]:
        medium_counter: Counter[tuple] = Counter()
        for row in medium_rows:
            qty = int(row.get("planned_boxes", 0) or 0)
            if qty <= 0:
                continue
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            medium_counter[key] += qty
        return medium_counter

    def _user_required_area_usage(self, medium_rows: list[dict], small_rows: list[dict]) -> dict:
        requirements = getattr(self.problem, "user_voyage_area_requirements", {})
        if not requirements:
            return {}
        medium_used: Counter[tuple[str, str]] = Counter()
        small_used: Counter[tuple[str, str]] = Counter()
        for row in medium_rows:
            key = (str(row.get("voyage_id", "")), str(row.get("area_no", "")))
            medium_used[key] += int(row.get("planned_boxes", 0) or 0)
        for row in small_rows:
            key = (str(row.get("voyage_id", "")), str(row.get("area_no", "")))
            small_used[key] += int(row.get("planned_boxes", 0) or 0)
        out = {}
        for voyage_id, areas in sorted(requirements.items()):
            voyage_usage = {}
            for area_no in sorted(areas):
                medium_boxes = int(medium_used[(voyage_id, area_no)])
                small_boxes = int(small_used[(voyage_id, area_no)])
                voyage_usage[area_no] = {
                    "medium_boxes": medium_boxes,
                    "small_boxes": small_boxes,
                    "satisfied": medium_boxes + small_boxes > 0,
                }
            out[voyage_id] = voyage_usage
        return out

    def _user_area_constraint_violations(self, medium_rows: list[dict], small_rows: list[dict]) -> dict:
        required_usage = self._user_required_area_usage(medium_rows, small_rows)
        unmet_required = []
        for voyage_id, area_usage in required_usage.items():
            for area_no, usage in area_usage.items():
                if not usage.get("satisfied", False):
                    unmet_required.append(
                        {
                            "voyage_id": voyage_id,
                            "area_no": area_no,
                            "reason": "required_area_not_used",
                        }
                    )

        allowlist = getattr(self.problem, "user_voyage_area_allowlist", {})
        blocklist = getattr(self.problem, "user_voyage_area_blocklist", {})
        forbidden_usage: Counter[tuple[str, str, str]] = Counter()
        outside_only_usage: Counter[tuple[str, str, str]] = Counter()
        for plan_level, rows in (("medium", medium_rows), ("small", small_rows)):
            for row in rows:
                qty = int(row.get("planned_boxes", 0) or 0)
                if qty <= 0:
                    continue
                voyage_id = str(row.get("voyage_id", ""))
                area_no = str(row.get("area_no", ""))
                if area_no in blocklist.get(voyage_id, set()):
                    forbidden_usage[(voyage_id, area_no, plan_level)] += qty
                allowed = allowlist.get(voyage_id, set())
                if allowed and area_no not in allowed:
                    outside_only_usage[(voyage_id, area_no, plan_level)] += qty

        forbidden = [
            {"voyage_id": voyage_id, "area_no": area_no, "plan_level": level, "boxes": qty}
            for (voyage_id, area_no, level), qty in sorted(forbidden_usage.items())
        ]
        outside_only = [
            {"voyage_id": voyage_id, "area_no": area_no, "plan_level": level, "boxes": qty}
            for (voyage_id, area_no, level), qty in sorted(outside_only_usage.items())
        ]
        return {
            "has_violations": bool(unmet_required or forbidden or outside_only),
            "unmet_required_areas": unmet_required,
            "forbidden_area_usage": forbidden,
            "outside_only_area_usage": outside_only,
        }

    def _medium_unplaced_group_details(self, unplaced: Counter[str]) -> list[dict]:
        groups_by_id = {group.group_id: group for group in self.problem.groups}
        details = []
        for group_id, qty in sorted(unplaced.items()):
            if qty <= 0:
                continue
            group = groups_by_id.get(group_id)
            if group is None:
                details.append({"group_id": group_id, "unplaced_boxes": int(qty)})
                continue
            details.append(
                {
                    "group_id": group_id,
                    "voyage_id": group.voyage_id,
                    "flow": group.status,
                    "port": group.port,
                    "size": group.size,
                    "big_plan_size_mode": group.big_plan_size_mode,
                    "demand": int(group.demand),
                    "unplaced_boxes": int(qty),
                }
            )
        return details

    def _small_unplaced_group_details(self, unplaced: Counter[str]) -> list[dict]:
        groups_by_id = {group.group_id: group for group in self.problem.small_groups}
        details = []
        for group_id, qty in sorted(unplaced.items()):
            if qty <= 0:
                continue
            group = groups_by_id.get(group_id)
            if group is None:
                details.append({"group_id": group_id, "unplaced_boxes": int(qty)})
                continue
            details.append(
                {
                    "group_id": group_id,
                    "voyage_id": group.voyage_id,
                    "flow": group.status,
                    "port": group.port,
                    "size": group.size,
                    "height": group.height,
                    "weight_class": group.weight_class,
                    "special_stow_code": group.special_stow_code,
                    "demand": int(group.demand),
                    "unplaced_boxes": int(qty),
                }
            )
        return details

    def _check_small_plan_feedback(self, medium_assignment: MediumAssignment, iteration: int) -> tuple[bool, MediumAssignment | None]:
        """Run a periodic small-plan feasibility probe and add feedback immediately.

        The first probe keeps the original strict medium-area inheritance.  If
        that fails, the final-output direct bay repair is tried and, when it is
        feasible, its area pattern is converted back into a medium assignment
        candidate for the SA loop.
        """
        medium_counter = self._medium_counter_from_assignment(medium_assignment)
        try:
            self._make_doc_small_rows(medium_counter)
        except SmallPlanInfeasible as exc:
            strict_exc = exc
            self._add_small_plan_strict_feedback(strict_exc)
            try:
                repair_rows = self._make_direct_small_rows(medium_counter)
            except SmallPlanInfeasible as repair_exc:
                exc = repair_exc
                self._log_small_plan_check(
                    iteration,
                    strict_feasible=False,
                    repair_feasible=False,
                    detail=(
                        f"strict_failed=voyage={strict_exc.voyage_id},flow={strict_exc.status},"
                        f"port={strict_exc.port},size={strict_exc.size},area={strict_exc.area_no}; "
                        f"repair_failed=voyage={repair_exc.voyage_id},flow={repair_exc.status},"
                        f"port={repair_exc.port},size={repair_exc.size},area={repair_exc.area_no}"
                    ),
                )
                key = (exc.voyage_id, exc.status, exc.port, exc.size, exc.area_no)
                broad_key = (exc.voyage_id, exc.status, "*", exc.size, exc.area_no)
                self.small_plan_area_feedback[key] += 1
                self.small_plan_area_feedback[broad_key] += 1
                return True, None
            else:
                repaired_assignment = self._repair_assignment_to_cover_small_rows(medium_assignment, repair_rows)
                feed_method = "in_place_capacity_checked"
                if repaired_assignment is None:
                    repaired_counter = self._repair_medium_counter_to_cover_small_rows(medium_counter, repair_rows)
                    repaired_assignment = self._assignment_from_medium_counter(repaired_counter)
                    feed_method = "counter_projection"
                if repaired_assignment is not None:
                    medium_hard_feasible = self._check_medium_hard_constraints(repaired_assignment)
                    self._log_small_plan_check(
                        iteration,
                        strict_feasible=False,
                        repair_feasible=True,
                        detail=(
                            f"strict_failed=voyage={strict_exc.voyage_id},flow={strict_exc.status},"
                            f"port={strict_exc.port},size={strict_exc.size},area={strict_exc.area_no}; "
                            f"repair_fed_back_to_medium=yes method={feed_method} "
                            f"medium_hard_feasible={medium_hard_feasible}"
                        ),
                    )
                    try:
                        self._make_doc_small_rows(self._medium_counter_from_assignment(repaired_assignment))
                    except SmallPlanInfeasible:
                        return True, repaired_assignment
                    self._remember_small_feasible_assignment(repaired_assignment, iteration)
                    return True, repaired_assignment
                self._log_small_plan_check(
                    iteration,
                    strict_feasible=False,
                    repair_feasible=True,
                    detail=(
                        f"strict_failed=voyage={strict_exc.voyage_id},flow={strict_exc.status},"
                        f"port={strict_exc.port},size={strict_exc.size},area={strict_exc.area_no}; "
                        "repair_fed_back_to_medium=no"
                    ),
                )
                self._remember_small_feasible_assignment(medium_assignment, iteration)
                return False, None
        self._log_small_plan_check(iteration, strict_feasible=True, repair_feasible=True, detail="repair_not_needed")
        self._remember_small_feasible_assignment(medium_assignment, iteration)
        return False, None

    def _add_small_plan_strict_feedback(self, exc: SmallPlanInfeasible) -> None:
        key = (exc.voyage_id, exc.status, exc.port, exc.size, exc.area_no)
        broad_key = (exc.voyage_id, exc.status, "*", exc.size, exc.area_no)
        self.small_plan_strict_area_feedback[key] += 1
        self.small_plan_strict_area_feedback[broad_key] += 1

    def _is_strict_small_plan_feasible(self, medium_assignment: MediumAssignment) -> bool:
        try:
            self._make_doc_small_rows(self._medium_counter_from_assignment(medium_assignment))
        except SmallPlanInfeasible:
            return False
        return True

    @staticmethod
    def _log_small_plan_check(
        iteration: int,
        strict_feasible: bool,
        repair_feasible: bool,
        detail: str,
    ) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{stamp}] small-plan check iter={iteration}: "
            f"strict_feasible={strict_feasible} repair_feasible={repair_feasible} {detail}",
            flush=True,
        )

    def _remember_small_feasible_assignment(self, medium_assignment: MediumAssignment, iteration: int) -> None:
        energy = self.medium_energy(medium_assignment)
        if self.best_small_feasible_energy is not None and energy >= self.best_small_feasible_energy:
            return
        self.best_small_feasible_assignment = self._copy_assignment(medium_assignment)
        self.best_small_feasible_iteration = iteration
        self.best_small_feasible_energy = energy

    def _make_doc_small_rows(self, medium_counter: Counter[tuple], strict: bool = True) -> list[dict]:
        coarse_weights: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        for key, count in medium_counter.items():
            voyage_id, flow, port, size, area_no = key
            coarse_weights[(voyage_id, flow, port, size)][area_no] += count

        group_area_allocations = self._small_group_area_allocations(coarse_weights, strict=strict)
        area_has_45 = self._areas_with_45(group_area_allocations)
        bay_load: Counter[str] = Counter()
        bay_size_load: Counter[tuple[str, str]] = Counter()
        bay_port_size_load: Counter[tuple[str, str, str]] = Counter()
        bay_stack_used: Counter[tuple[str, str]] = Counter()
        bay_used_size: dict[str, str] = {}
        bay_used_height: dict[str, str] = {}
        bay_affinity: dict[str, tuple] = {}
        bay_coarse_affinity: dict[str, tuple] = {}
        block_affinity: dict[str, tuple] = {}
        block_coarse_affinity: dict[str, tuple] = {}
        small_counter: Counter[tuple] = Counter()
        block_members_by_id: dict[str, tuple[str, ...]] = {}
        remaining_big_plan_quota: Counter[tuple[str, str, str, str]] = Counter(self.problem.area_size_quota)

        for group in self.problem.small_groups:
            area_allocations = group_area_allocations[group.group_id]
            for area_no, qty in area_allocations.items():
                remaining = qty
                for bay_key in self._candidate_bays_for_small_group(
                    group,
                    area_no,
                    bay_load,
                    bay_used_size,
                    bay_used_height,
                    bay_affinity,
                    bay_coarse_affinity,
                    block_affinity,
                    block_coarse_affinity,
                    area_has_45,
                ):
                    bay = self.bays[bay_key]
                    footprint = self._bay_footprint_keys(bay_key, group.size)
                    if not footprint:
                        continue
                    free_total = min(self.bays[key].physical_capacity - bay_load[key] for key in footprint)
                    free_size = bay.cap_by_size.get(group.size, 0) - bay_size_load[(bay_key, group.size)]
                    free_stack = self._remaining_stack_capacity_for_small_group(
                        group,
                        bay_key,
                        bay_port_size_load,
                        bay_stack_used,
                    )
                    big_quota_key = (group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
                    quota_room = (
                        remaining_big_plan_quota[big_quota_key]
                        if self.problem.area_size_quota.get(big_quota_key, 0) > 0
                        else remaining
                    )
                    take = min(remaining, free_total, free_size, free_stack, quota_room)
                    if take <= 0:
                        continue
                    block_id = self._small_block_id_for_bay(area_no, bay_key)
                    small_counter[
                        (
                            group.voyage_id,
                            group.group_id,
                            group.status,
                            group.port,
                            group.size,
                            group.height,
                            group.weight_class,
                            group.special_stow_code,
                            area_no,
                            bay_key,
                            bay.bay_no,
                            block_id,
                        )
                    ] += take
                    for key in footprint:
                        bay_load[key] += take
                        bay_used_size[key] = group.size
                        bay_used_height[key] = group.height
                    bay_size_load[(bay_key, group.size)] += take
                    self._apply_stack_usage_for_small_group(group, bay_key, take, bay_port_size_load, bay_stack_used)
                    bay_affinity.setdefault(bay_key, self._small_affinity(group))
                    bay_coarse_affinity.setdefault(bay_key, self._small_coarse_affinity(group))
                    if block_id:
                        block_affinity.setdefault(block_id, self._small_affinity(group))
                        block_coarse_affinity.setdefault(block_id, self._small_coarse_affinity(group))
                        block_members_by_id.setdefault(block_id, self._small_block_bay_nos(area_no, block_id))
                    if self.problem.area_size_quota.get(big_quota_key, 0) > 0:
                        remaining_big_plan_quota[big_quota_key] -= take
                    remaining -= take
                    if remaining == 0:
                        break
                if remaining > 0:
                    if not strict:
                        self.small_unplaced_by_group[group.group_id] += remaining
                        continue
                    raise SmallPlanInfeasible(
                        (
                            f"small plan cannot place {remaining} boxes for "
                            f"voyage={group.voyage_id}, flow={group.status}, port={group.port}, size={group.size}, "
                            f"height={group.height}, weight={group.weight_class}, area={area_no}"
                        ),
                        group.voyage_id,
                        group.status,
                        group.port,
                        group.size,
                        area_no,
                    )

        rows: list[dict] = []
        block_total: Counter[str] = Counter()
        for key, count in small_counter.items():
            (
                _voyage_id,
                group_id,
                _flow,
                _port,
                size,
                _height,
                _weight_class,
                _special_stow_code,
                _area_no,
                _bay_key,
                bay_no,
                block_id,
            ) = key
            if not block_id:
                continue
            block_total[block_id] += count

        for key, count in sorted(small_counter.items()):
            (
                voyage_id,
                group_id,
                flow,
                port,
                size,
                height,
                weight_class,
                special_stow_code,
                area_no,
                _bay_key,
                bay_no,
                block_id,
            ) = key
            rows.append(
                {
                    "plan_level": "small",
                    "voyage_id": voyage_id,
                    "group_id": group_id,
                    "flow": flow,
                    "port": port,
                    "size": size,
                    "height": height,
                    "weight_class": weight_class,
                    "special_stow": bool(special_stow_code),
                    "special_stow_code": special_stow_code or "NORMAL",
                    "area_no": area_no,
                    "bay_no": bay_no,
                    "six_bay_block_id": block_id,
                    "six_bay_block_bays": "|".join(block_members_by_id.get(block_id, ())) if block_id else "",
                    "six_bay_block_total_boxes": block_total.get(block_id, 0) if block_id else 0,
                    "planned_boxes": count,
                }
            )
        return rows

    def _make_direct_small_rows(self, medium_counter: Counter[tuple]) -> list[dict]:
        order_strategies = ("difficulty", "large_demand_first", "original")
        last_error: SmallPlanInfeasible | None = None
        best_rows: list[dict] | None = None
        best_score: tuple[int, int, int, int, int] | None = None
        for strategy in order_strategies:
            try:
                rows = self._make_direct_small_rows_with_order(medium_counter, strategy)
            except SmallPlanInfeasible as exc:
                last_error = exc
                self.small_plan_area_feedback[(exc.voyage_id, exc.status, exc.port, exc.size, exc.area_no)] += 1
                self.small_plan_area_feedback[(exc.voyage_id, exc.status, "*", exc.size, exc.area_no)] += 1
                continue
            score = self._direct_small_rows_inheritance_score(rows)
            if best_score is None or score < best_score:
                best_score = score
                best_rows = rows
        if best_rows is not None:
            return best_rows
        if last_error is not None:
            raise last_error
        return []

    def _direct_small_rows_inheritance_score(self, rows: list[dict]) -> tuple[int, int, int, int, int]:
        tier_boxes: Counter[int] = Counter()
        for row in rows:
            qty = int(row.get("planned_boxes", 0) or 0)
            if qty <= 0:
                continue
            voyage_id = str(row.get("voyage_id", ""))
            flow = str(row.get("flow", ""))
            area_no = str(row.get("area_no", ""))
            size = str(row.get("size", ""))
            big_size = self._big_plan_size(size)
            quota_key = (voyage_id, flow, area_no, big_size)
            if self.problem.area_size_quota.get(quota_key, 0) > 0:
                tier = 0
            elif area_no in self.problem.assigned_areas.get((voyage_id, flow), set()):
                tier = 1
            elif self._is_any_big_plan_area(area_no):
                tier = 2
            else:
                tier = 3
            tier_boxes[tier] += qty
        return (
            tier_boxes[3],
            tier_boxes[2],
            tier_boxes[1],
            -tier_boxes[0],
            len(rows),
        )

    def _make_direct_small_rows_with_order(self, medium_counter: Counter[tuple], order_strategy: str) -> list[dict]:
        remaining_exact_quota: Counter[tuple[str, str, str, str, str]] = Counter(medium_counter)
        remaining_broad_quota: Counter[tuple[str, str, str, str]] = Counter()
        for (voyage_id, flow, _port, size, area_no), qty in medium_counter.items():
            remaining_broad_quota[(voyage_id, flow, size, area_no)] += qty
        remaining_big_plan_quota: Counter[tuple[str, str, str, str]] = Counter(self.problem.area_size_quota)

        area_has_45: set[str] = set()
        bay_load: Counter[str] = Counter()
        bay_size_load: Counter[tuple[str, str]] = Counter()
        bay_port_size_load: Counter[tuple[str, str, str]] = Counter()
        bay_stack_used: Counter[tuple[str, str]] = Counter()
        bay_used_size: dict[str, str] = {}
        bay_used_height: dict[str, str] = {}
        bay_affinity: dict[str, tuple] = {}
        bay_coarse_affinity: dict[str, tuple] = {}
        block_affinity: dict[str, tuple] = {}
        block_coarse_affinity: dict[str, tuple] = {}
        small_counter: Counter[tuple] = Counter()
        block_members_by_id: dict[str, tuple[str, ...]] = {}

        for group in self._ordered_small_groups_for_direct_plan(order_strategy):
            remaining = group.demand
            while remaining > 0:
                placed_this_round = 0
                candidates = self._candidate_bays_for_direct_small_group(
                    group,
                    remaining,
                    remaining_exact_quota,
                    remaining_broad_quota,
                    remaining_big_plan_quota,
                    bay_load,
                    bay_size_load,
                    bay_used_size,
                    bay_used_height,
                    bay_affinity,
                    bay_coarse_affinity,
                    block_affinity,
                    block_coarse_affinity,
                    area_has_45,
                )
                for bay_key in candidates:
                    bay = self.bays[bay_key]
                    area_no = bay.area_no
                    footprint = self._bay_footprint_keys(bay_key, group.size)
                    if not footprint:
                        continue
                    free_total = min(self.bays[key].physical_capacity - bay_load[key] for key in footprint)
                    free_size = bay.cap_by_size.get(group.size, 0) - bay_size_load[(bay_key, group.size)]
                    free_stack = self._remaining_stack_capacity_for_small_group(
                        group,
                        bay_key,
                        bay_port_size_load,
                        bay_stack_used,
                    )
                    big_quota_key = (group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
                    quota_room = (
                        remaining_big_plan_quota[big_quota_key]
                        if self.problem.area_size_quota.get(big_quota_key, 0) > 0
                        else remaining
                    )
                    take = min(remaining, free_total, free_size, free_stack, quota_room)
                    if take <= 0:
                        continue

                    block_id = self._small_block_id_for_bay(area_no, bay_key)
                    small_counter[
                        (
                            group.voyage_id,
                            group.group_id,
                            group.status,
                            group.port,
                            group.size,
                            group.height,
                            group.weight_class,
                            group.special_stow_code,
                            area_no,
                            bay_key,
                            bay.bay_no,
                            block_id,
                        )
                    ] += take
                    for key in footprint:
                        bay_load[key] += take
                        bay_used_size[key] = group.size
                        bay_used_height[key] = group.height
                    bay_size_load[(bay_key, group.size)] += take
                    self._apply_stack_usage_for_small_group(group, bay_key, take, bay_port_size_load, bay_stack_used)
                    bay_affinity.setdefault(bay_key, self._small_affinity(group))
                    bay_coarse_affinity.setdefault(bay_key, self._small_coarse_affinity(group))
                    if block_id:
                        block_affinity.setdefault(block_id, self._small_affinity(group))
                        block_coarse_affinity.setdefault(block_id, self._small_coarse_affinity(group))
                        block_members_by_id.setdefault(block_id, self._small_block_bay_nos(area_no, block_id))
                    if group.size == "45":
                        area_has_45.add(area_no)
                    remaining_exact_quota[(group.voyage_id, group.status, group.port, group.size, area_no)] -= take
                    remaining_broad_quota[(group.voyage_id, group.status, group.size, area_no)] -= take
                    big_quota_key = (group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
                    if self.problem.area_size_quota.get(big_quota_key, 0) > 0:
                        remaining_big_plan_quota[big_quota_key] -= take
                    remaining -= take
                    placed_this_round += take
                    if remaining == 0:
                        break

                if placed_this_round == 0:
                    area_no = self._best_direct_failure_area(group, remaining_broad_quota)
                    raise SmallPlanInfeasible(
                        (
                            f"small direct repair cannot place {remaining} boxes for "
                            f"voyage={group.voyage_id}, flow={group.status}, port={group.port}, size={group.size}, "
                            f"height={group.height}, weight={group.weight_class}, area={area_no}"
                        ),
                        group.voyage_id,
                        group.status,
                        group.port,
                        group.size,
                        area_no,
                    )

        self._improve_direct_small_inheritance(
            small_counter,
            remaining_big_plan_quota,
            block_members_by_id,
        )
        return self._small_rows_from_counter(small_counter, block_members_by_id)

    def _improve_direct_small_inheritance(
        self,
        small_counter: Counter[tuple],
        remaining_big_plan_quota: Counter[tuple[str, str, str, str]],
        block_members_by_id: dict[str, tuple[str, ...]],
    ) -> None:
        groups_by_id = {group.group_id: group for group in self.problem.small_groups}
        moved = True
        while moved:
            moved = False
            (
                bay_load,
                bay_size_load,
                bay_port_size_load,
                bay_stack_used,
                bay_used_size,
                bay_used_height,
                area_has_45,
            ) = self._direct_small_usage_state(small_counter)
            source_keys = [
                key
                for key, qty in small_counter.items()
                if qty > 0 and self._direct_small_source_can_improve(groups_by_id.get(key[1]), key[8])
            ]
            source_keys.sort(
                key=lambda key: (
                    self._direct_small_area_tier(groups_by_id[key[1]], key[8]),
                    -small_counter[key],
                    key[0],
                    key[1],
                    key[8],
                ),
                reverse=True,
            )
            for source_key in source_keys:
                source_qty = small_counter.get(source_key, 0)
                if source_qty <= 0:
                    continue
                group = groups_by_id.get(source_key[1])
                if group is None:
                    continue
                target = self._best_direct_small_inherited_target(
                    group,
                    source_key[8],
                    source_qty,
                    remaining_big_plan_quota,
                    bay_load,
                    bay_size_load,
                    bay_port_size_load,
                    bay_stack_used,
                    bay_used_size,
                    bay_used_height,
                    area_has_45,
                )
                if target is None:
                    continue
                target_area, target_bay_key, take = target
                if take <= 0:
                    continue
                target_bay = self.bays[target_bay_key]
                block_id = self._small_block_id_for_bay(target_area, target_bay_key)
                target_key = (
                    source_key[0],
                    source_key[1],
                    source_key[2],
                    source_key[3],
                    source_key[4],
                    source_key[5],
                    source_key[6],
                    source_key[7],
                    target_area,
                    target_bay_key,
                    target_bay.bay_no,
                    block_id,
                )
                small_counter[source_key] -= take
                if small_counter[source_key] <= 0:
                    del small_counter[source_key]
                small_counter[target_key] += take
                target_quota_key = (
                    group.voyage_id,
                    group.status,
                    target_area,
                    self._big_plan_size(group.size),
                )
                remaining_big_plan_quota[target_quota_key] -= take
                source_quota_key = (
                    group.voyage_id,
                    group.status,
                    source_key[8],
                    self._big_plan_size(group.size),
                )
                if self.problem.area_size_quota.get(source_quota_key, 0) > 0:
                    remaining_big_plan_quota[source_quota_key] += take
                if block_id:
                    block_members_by_id.setdefault(block_id, self._small_block_bay_nos(target_area, block_id))
                moved = True
                break

    def _direct_small_usage_state(
        self,
        small_counter: Counter[tuple],
    ) -> tuple[
        Counter[str],
        Counter[tuple[str, str]],
        Counter[tuple[str, str, str]],
        Counter[tuple[str, str]],
        dict[str, str],
        dict[str, str],
        set[str],
    ]:
        bay_load: Counter[str] = Counter()
        bay_size_load: Counter[tuple[str, str]] = Counter()
        bay_port_size_load: Counter[tuple[str, str, str]] = Counter()
        bay_stack_used: Counter[tuple[str, str]] = Counter()
        bay_used_size: dict[str, str] = {}
        bay_used_height: dict[str, str] = {}
        area_has_45: set[str] = set()
        for key, qty in small_counter.items():
            if qty <= 0:
                continue
            size = key[4]
            height = key[5]
            area_no = key[8]
            bay_key = key[9]
            footprint = self._bay_footprint_keys(bay_key, size)
            for footprint_key in footprint:
                bay_load[footprint_key] += qty
                bay_used_size[footprint_key] = size
                bay_used_height[footprint_key] = height
            bay_size_load[(bay_key, size)] += qty
            group = type(
                "SmallStackGroup",
                (),
                {"size": size, "port": key[3]},
            )()
            self._apply_stack_usage_for_small_group(group, bay_key, qty, bay_port_size_load, bay_stack_used)
            if size == "45":
                area_has_45.add(area_no)
        return bay_load, bay_size_load, bay_port_size_load, bay_stack_used, bay_used_size, bay_used_height, area_has_45

    def _direct_small_source_can_improve(self, group, area_no: str) -> bool:
        if group is None:
            return False
        return self._direct_small_area_tier(group, area_no) > 0

    def _direct_small_area_tier(self, group, area_no: str) -> int:
        quota_key = (group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
        if self.problem.area_size_quota.get(quota_key, 0) > 0:
            return 0
        if area_no in self.problem.assigned_areas.get((group.voyage_id, group.status), set()):
            return 1
        if self._is_any_big_plan_area(area_no):
            return 2
        return 3

    def _best_direct_small_inherited_target(
        self,
        group,
        source_area: str,
        source_qty: int,
        remaining_big_plan_quota: Counter[tuple[str, str, str, str]],
        bay_load: Counter[str],
        bay_size_load: Counter[tuple[str, str]],
        bay_port_size_load: Counter[tuple[str, str, str]],
        bay_stack_used: Counter[tuple[str, str]],
        bay_used_size: dict[str, str],
        bay_used_height: dict[str, str],
        area_has_45: set[str],
    ) -> tuple[str, str, int] | None:
        big_size = self._big_plan_size(group.size)
        target_areas = [
            area_no
            for (voyage_id, flow, area_no, size), quota in self.problem.area_size_quota.items()
            if voyage_id == group.voyage_id
            and flow == group.status
            and size == big_size
            and area_no != source_area
            and quota > 0
            and remaining_big_plan_quota[(voyage_id, flow, area_no, size)] > 0
        ]
        target_areas.sort(
            key=lambda area_no: (
                -remaining_big_plan_quota[(group.voyage_id, group.status, area_no, big_size)],
                area_no,
            )
        )
        best: tuple[str, str, int] | None = None
        for area_no in target_areas:
            quota_left = remaining_big_plan_quota[(group.voyage_id, group.status, area_no, big_size)]
            if quota_left <= 0:
                continue
            for bay_key in self.bays_by_area.get(area_no, []):
                if not self._small_bay_hard_feasible(
                    group,
                    area_no,
                    bay_key,
                    bay_used_size,
                    bay_used_height,
                    area_has_45,
                ):
                    continue
                footprint = self._bay_footprint_keys(bay_key, group.size)
                if not footprint:
                    continue
                free_total = min(self.bays[key].physical_capacity - bay_load[key] for key in footprint)
                free_size = self.bays[bay_key].cap_by_size.get(group.size, 0) - bay_size_load[(bay_key, group.size)]
                free_stack = self._remaining_stack_capacity_for_small_group(
                    group,
                    bay_key,
                    bay_port_size_load,
                    bay_stack_used,
                )
                take = min(source_qty, quota_left, free_total, free_size, free_stack)
                if take <= 0:
                    continue
                candidate = (area_no, bay_key, take)
                if best is None or (
                    take,
                    quota_left,
                    -self.bays[bay_key].bay_order,
                ) > (
                    best[2],
                    remaining_big_plan_quota[(group.voyage_id, group.status, best[0], big_size)],
                    -self.bays[best[1]].bay_order,
                ):
                    best = candidate
        return best

    def _ordered_small_groups_for_direct_plan(self, order_strategy: str) -> list:
        groups = list(self.problem.small_groups)
        if order_strategy == "original":
            return groups
        if order_strategy == "large_demand_first":
            return sorted(groups, key=lambda group: (self._small_group_difficulty_key(group)[:3], -group.demand, group.group_id))
        return sorted(groups, key=self._small_group_difficulty_key)

    def _candidate_bays_for_direct_small_group(
        self,
        group,
        remaining: int,
        remaining_exact_quota: Counter[tuple[str, str, str, str, str]],
        remaining_broad_quota: Counter[tuple[str, str, str, str]],
        remaining_big_plan_quota: Counter[tuple[str, str, str, str]],
        bay_load: Counter[str],
        bay_size_load: Counter[tuple[str, str]],
        bay_used_size: dict[str, str],
        bay_used_height: dict[str, str],
        bay_affinity: dict[str, tuple],
        bay_coarse_affinity: dict[str, tuple],
        block_affinity: dict[str, tuple],
        block_coarse_affinity: dict[str, tuple],
        area_has_45: set[str],
    ) -> list[str]:
        candidates: list[str] = []
        for area_no in self._direct_small_candidate_areas(group):
            for bay_key in self.bays_by_area.get(area_no, []):
                if not self._small_bay_hard_feasible(
                    group,
                    area_no,
                    bay_key,
                    bay_used_size,
                    bay_used_height,
                    area_has_45,
                ):
                    continue
                footprint = self._bay_footprint_keys(bay_key, group.size)
                if not footprint:
                    continue
                free_total = min(self.bays[key].physical_capacity - bay_load[key] for key in footprint)
                free_size = self.bays[bay_key].cap_by_size.get(group.size, 0) - bay_size_load[(bay_key, group.size)]
                big_quota_key = (group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
                quota_room = (
                    remaining_big_plan_quota[big_quota_key]
                    if self.problem.area_size_quota.get(big_quota_key, 0) > 0
                    else group.demand
                )
                if min(free_total, free_size, quota_room) <= 0:
                    continue
                candidates.append(bay_key)
        return sorted(
            candidates,
            key=lambda bay_key: self._direct_small_bay_sort_key(
                group,
                bay_key,
                remaining,
                remaining_exact_quota,
                remaining_broad_quota,
                remaining_big_plan_quota,
                bay_load,
                bay_size_load,
                bay_affinity,
                bay_coarse_affinity,
                block_affinity,
                block_coarse_affinity,
            ),
        )

    def _direct_small_bay_sort_key(
        self,
        group,
        bay_key: str,
        remaining: int,
        remaining_exact_quota: Counter[tuple[str, str, str, str, str]],
        remaining_broad_quota: Counter[tuple[str, str, str, str]],
        remaining_big_plan_quota: Counter[tuple[str, str, str, str]],
        bay_load: Counter[str],
        bay_size_load: Counter[tuple[str, str]],
        bay_affinity: dict[str, tuple],
        bay_coarse_affinity: dict[str, tuple],
        block_affinity: dict[str, tuple],
        block_coarse_affinity: dict[str, tuple],
    ) -> tuple:
        bay = self.bays[bay_key]
        area_no = bay.area_no
        exact_left = remaining_exact_quota[(group.voyage_id, group.status, group.port, group.size, area_no)]
        broad_left = remaining_broad_quota[(group.voyage_id, group.status, group.size, area_no)]
        big_quota_key = (group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
        big_quota_left = (
            remaining_big_plan_quota[big_quota_key]
            if self.problem.area_size_quota.get(big_quota_key, 0) > 0
            else 0
        )
        quota_bucket = self._area_fallback_tier_for_group(group, area_no)
        if quota_bucket >= 2 and self._foreign_exact_quota_left(
            group,
            area_no,
            self._big_plan_size(group.size),
            remaining_big_plan_quota,
        ) > 0:
            quota_bucket = 4
        footprint = self._bay_footprint_keys(bay_key, group.size)
        free_total = min(self.bays[key].physical_capacity - bay_load[key] for key in footprint)
        free_size = bay.cap_by_size.get(group.size, 0) - bay_size_load[(bay_key, group.size)]
        usable = min(free_total, free_size, big_quota_left if big_quota_left > 0 else max(free_total, free_size))
        return (
            quota_bucket,
            0 if exact_left > 0 else 1 if broad_left > 0 else 2,
            0 if usable >= remaining else 1,
            self._small_block_score(group, area_no, bay_key, block_affinity, block_coarse_affinity),
            self._bay_affinity_score(group, bay_key, bay_affinity, bay_coarse_affinity),
            bay.is_fallback_bay,
            0 if group.port in bay.existing_ports else 1,
            -max(big_quota_left, exact_left, broad_left, 0),
            bay_load[bay_key],
            bay.bay_order,
        )

    def _foreign_exact_quota_left(
        self,
        group,
        area_no: str,
        big_size: str,
        remaining_big_plan_quota: Counter[tuple[str, str, str, str]],
    ) -> int:
        total = 0
        for (voyage_id, flow, quota_area, size), quota in self.problem.area_size_quota.items():
            if quota_area != area_no or size != big_size or quota <= 0:
                continue
            if voyage_id == group.voyage_id and flow == group.status:
                continue
            total += max(0, int(remaining_big_plan_quota[(voyage_id, flow, quota_area, size)]))
        return total

    def _direct_small_candidate_areas(self, group) -> list[str]:
        return [
            area_no
            for area_no in self.bays_by_area
            if self._user_area_policy_allows(group.voyage_id, area_no)
            and (
                self._area_supports_group_flow(group, area_no)
                or self._user_area_policy_forces_support(group.voyage_id, area_no)
            )
        ]

    def _best_direct_failure_area(
        self,
        group,
        remaining_broad_quota: Counter[tuple[str, str, str, str]],
    ) -> str:
        candidates = [
            area_no
            for area_no in self._direct_small_candidate_areas(group)
            if self.area_size_height_cap[(area_no, group.size, group.height)] > 0
        ]
        if not candidates:
            return ""
        return min(
            candidates,
            key=lambda area_no: (
                0 if remaining_broad_quota[(group.voyage_id, group.status, group.size, area_no)] > 0 else 1,
                -self.area_size_height_cap[(area_no, group.size, group.height)],
                area_no,
            ),
        )

    def _small_rows_from_counter(
        self,
        small_counter: Counter[tuple],
        block_members_by_id: dict[str, tuple[str, ...]],
    ) -> list[dict]:
        rows: list[dict] = []
        block_total: Counter[str] = Counter()
        for key, count in small_counter.items():
            block_id = key[-1]
            if block_id:
                block_total[block_id] += count

        for key, count in sorted(small_counter.items()):
            (
                voyage_id,
                group_id,
                flow,
                port,
                size,
                height,
                weight_class,
                special_stow_code,
                area_no,
                _bay_key,
                bay_no,
                block_id,
            ) = key
            rows.append(
                {
                    "plan_level": "small",
                    "voyage_id": voyage_id,
                    "group_id": group_id,
                    "flow": flow,
                    "port": port,
                    "size": size,
                    "height": height,
                    "weight_class": weight_class,
                    "special_stow": bool(special_stow_code),
                    "special_stow_code": special_stow_code or "NORMAL",
                    "area_no": area_no,
                    "bay_no": bay_no,
                    "six_bay_block_id": block_id,
                    "six_bay_block_bays": "|".join(block_members_by_id.get(block_id, ())) if block_id else "",
                    "six_bay_block_total_boxes": block_total.get(block_id, 0) if block_id else 0,
                    "planned_boxes": count,
                }
            )
        return rows

    def _repair_medium_counter_to_cover_small_rows(
        self,
        medium_counter: Counter[tuple],
        small_rows: list[dict],
    ) -> Counter[tuple]:
        target_by_coarse: Counter[tuple[str, str, str, str]] = Counter()
        for voyage_id, flow, port, size, _area_no in medium_counter:
            target_by_coarse[(voyage_id, flow, port, size)] += medium_counter[
                (voyage_id, flow, port, size, _area_no)
            ]

        small_usage: Counter[tuple[str, str, str, str, str]] = Counter()
        for row in small_rows:
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            small_usage[key] += int(row.get("planned_boxes", 0) or 0)

        repaired = Counter({key: qty for key, qty in small_usage.items() if qty > 0})
        area_load: Counter[str] = Counter()
        area_size_load: Counter[tuple[str, str]] = Counter()
        quota_load: Counter[tuple[str, str, str, str]] = Counter()
        for (voyage_id, flow, _port, size, area_no), qty in repaired.items():
            if qty <= 0:
                continue
            size_mode = size if size in {"20", "40", "45"} else "40"
            big_size = self._big_plan_size(size)
            area_load[area_no] += qty
            area_size_load[(area_no, size_mode)] += qty
            quota_load[(voyage_id, flow, area_no, big_size)] += qty

        representatives: dict[tuple[str, str, str, str], object] = {}
        for group in self.problem.groups:
            representatives.setdefault(self._medium_attrs(group), group)

        small_by_coarse: Counter[tuple[str, str, str, str]] = Counter()
        for key, qty in small_usage.items():
            small_by_coarse[key[:4]] += qty

        for coarse_key, target_qty in sorted(target_by_coarse.items()):
            remaining = int(target_qty) - int(small_by_coarse.get(coarse_key, 0))
            if remaining <= 0:
                continue
            group = representatives.get(coarse_key)
            if group is None:
                continue
            allocation = self._allocate_medium_remainder_by_inheritance_tiers(
                group,
                remaining,
                medium_counter,
                area_load,
                area_size_load,
                quota_load,
            )
            for area_no, qty in allocation.items():
                if qty <= 0:
                    continue
                repaired[coarse_key + (area_no,)] += qty
                area_load[area_no] += qty
                area_size_load[(area_no, group.size_mode)] += qty
                quota_load[(group.voyage_id, group.status, area_no, group.big_plan_size_mode)] += qty

        medium_total = sum(qty for qty in medium_counter.values() if qty > 0)
        repaired_total = sum(qty for qty in repaired.values() if qty > 0)
        self.small_plan_medium_repair_added_boxes = max(0, repaired_total - medium_total)
        return Counter({key: qty for key, qty in repaired.items() if qty > 0})

    def _allocate_medium_remainder_by_inheritance_tiers(
        self,
        group,
        demand: int,
        medium_counter: Counter[tuple],
        area_load: Counter[str],
        area_size_load: Counter[tuple[str, str]],
        quota_load: Counter[tuple[str, str, str, str]],
    ) -> Counter[str]:
        allocation: Counter[str] = Counter()
        remaining = max(0, int(demand))
        if remaining <= 0:
            return allocation

        medium_weights: Counter[str] = Counter()
        for (voyage_id, flow, port, size, area_no), qty in medium_counter.items():
            if (voyage_id, flow, port, size) == self._medium_attrs(group) and qty > 0:
                medium_weights[area_no] += qty
        big_weights = Counter(self._fallback_area_weights(group.voyage_id, group.status, group.big_plan_size_mode))

        def original_quota(area_no: str) -> int:
            key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
            return int(self.problem.area_size_quota.get(key, 0) or 0)

        def tier_for_area(area_no: str) -> int:
            quota = original_quota(area_no)
            key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
            if quota > 0 and quota_load[key] + allocation[area_no] < quota:
                return 0
            if quota > 0:
                return 99
            if self._is_big_plan_area_for_group(group, area_no):
                return 1
            if self._is_any_big_plan_area(area_no):
                return 2
            return 3

        def free_for_area(area_no: str) -> int:
            total_free = self.area_total_cap[area_no] - area_load[area_no] - allocation[area_no]
            size_key = (area_no, group.size_mode)
            size_free = self.area_size_cap[size_key] - area_size_load[size_key] - allocation[area_no]
            free = min(total_free, size_free)
            quota = original_quota(area_no)
            if quota > 0:
                quota_key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
                free = min(free, quota - quota_load[quota_key] - allocation[area_no])
            return max(0, int(free))

        candidates = [area_no for area_no in self._candidate_areas_for_group(group) if free_for_area(area_no) > 0]
        for tier in (0, 1, 2, 3):
            while remaining > 0:
                tier_candidates = [area_no for area_no in candidates if tier_for_area(area_no) == tier and free_for_area(area_no) > 0]
                if not tier_candidates:
                    break
                tier_candidates.sort(
                    key=lambda area_no: (
                        0 if free_for_area(area_no) >= remaining else 1,
                        -int(medium_weights.get(area_no, 0)),
                        -int(big_weights.get(area_no, 0)),
                        -free_for_area(area_no),
                        area_no,
                    )
                )
                progressed = False
                for area_no in tier_candidates:
                    take = min(remaining, free_for_area(area_no))
                    if take <= 0:
                        continue
                    allocation[area_no] += take
                    remaining -= take
                    progressed = True
                    if remaining <= 0:
                        break
                if not progressed:
                    break
            if remaining <= 0:
                break
        return allocation

    def _repair_assignment_to_cover_small_rows(
        self,
        medium_assignment: MediumAssignment,
        small_rows: list[dict],
    ) -> MediumAssignment | None:
        repaired = self._copy_assignment(medium_assignment)
        groups_by_id = {group.group_id: group for group in self.problem.groups}
        groups_by_attrs: defaultdict[tuple[str, str, str, str], list] = defaultdict(list)
        for group in self.problem.groups:
            groups_by_attrs[self._medium_attrs(group)].append(group)

        small_usage: Counter[tuple[str, str, str, str, str]] = Counter()
        for row in small_rows:
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            small_usage[key] += int(row.get("planned_boxes", 0) or 0)

        current_usage: Counter[tuple[str, str, str, str, str]] = Counter()
        for group_id, area_counts in repaired.items():
            group = groups_by_id[group_id]
            attrs = self._medium_attrs(group)
            for area_no, qty in area_counts.items():
                if qty > 0:
                    current_usage[attrs + (area_no,)] += qty

        area_load, area_size_load, quota_load = self._medium_loads(repaired)
        for target_key, required_qty in sorted(small_usage.items()):
            missing = required_qty - current_usage.get(target_key, 0)
            if missing <= 0:
                continue
            attrs = target_key[:4]
            target_area = target_key[4]
            groups = groups_by_attrs.get(attrs, [])
            if not groups:
                return None
            donor_keys = [
                key
                for key, qty in current_usage.items()
                if key[:4] == attrs
                and key[4] != target_area
                and qty > small_usage.get(key, 0)
            ]
            donor_keys.sort(key=lambda key: (-(current_usage[key] - small_usage.get(key, 0)), key[4]))
            for donor_key in donor_keys:
                if missing <= 0:
                    break
                donor_area = donor_key[4]
                surplus = current_usage[donor_key] - small_usage.get(donor_key, 0)
                if surplus <= 0:
                    continue
                for group in sorted(groups, key=lambda item: -repaired.get(item.group_id, Counter()).get(donor_area, 0)):
                    if missing <= 0 or surplus <= 0:
                        break
                    available = repaired.get(group.group_id, Counter()).get(donor_area, 0)
                    if available <= 0:
                        continue
                    movable = min(missing, surplus, available)
                    while movable > 0:
                        if self._medium_feasible_after(
                            repaired,
                            group,
                            donor_area,
                            target_area,
                            movable,
                            area_load,
                            area_size_load,
                            quota_load,
                        ):
                            break
                        movable -= 1
                    if movable <= 0:
                        continue
                    group_assignment = repaired[group.group_id]
                    group_assignment[donor_area] -= movable
                    if group_assignment[donor_area] <= 0:
                        del group_assignment[donor_area]
                    group_assignment[target_area] += movable
                    area_load[donor_area] -= movable
                    area_load[target_area] += movable
                    area_size_load[(donor_area, group.size_mode)] -= movable
                    area_size_load[(target_area, group.size_mode)] += movable
                    quota_load[(group.voyage_id, group.status, donor_area, group.big_plan_size_mode)] -= movable
                    quota_load[(group.voyage_id, group.status, target_area, group.big_plan_size_mode)] += movable
                    current_usage[donor_key] -= movable
                    current_usage[target_key] += movable
                    missing -= movable
                    surplus -= movable
            if missing > 0:
                return None

        return repaired

    def _assignment_from_medium_counter(self, medium_counter: Counter[tuple]) -> MediumAssignment | None:
        groups_by_attrs: defaultdict[tuple[str, str, str, str], list] = defaultdict(list)
        for group in self.problem.groups:
            groups_by_attrs[self._medium_attrs(group)].append(group)

        area_counts_by_attrs: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        for key, qty in medium_counter.items():
            if qty <= 0:
                continue
            voyage_id, flow, port, size, area_no = key
            area_counts_by_attrs[(voyage_id, flow, port, size)][area_no] += int(qty)

        assignment: MediumAssignment = {group.group_id: Counter() for group in self.problem.groups}
        for attrs, area_counts in area_counts_by_attrs.items():
            groups = sorted(groups_by_attrs.get(attrs, []), key=lambda group: (-group.demand, group.group_id))
            if not groups:
                return None
            total_demand = sum(group.demand for group in groups)
            if sum(area_counts.values()) != total_demand:
                return None
            remaining_by_area = Counter(area_counts)
            area_order = sorted(remaining_by_area, key=lambda area_no: (-remaining_by_area[area_no], area_no))
            for group in groups:
                remaining = group.demand
                for area_no in area_order:
                    if remaining <= 0:
                        break
                    take = min(remaining, remaining_by_area[area_no])
                    if take <= 0:
                        continue
                    assignment[group.group_id][area_no] += take
                    remaining_by_area[area_no] -= take
                    remaining -= take
                if remaining > 0:
                    return None

        for attrs, groups in groups_by_attrs.items():
            if attrs in area_counts_by_attrs:
                continue
            if any(group.demand > 0 for group in groups):
                return None
        return assignment

    def make_small_rows_from_medium_rows(self, medium_rows: list[dict]) -> list[dict]:
        """Construct the small plan from externally supplied medium-plan rows.

        The expected row format is the existing medium output CSV schema:
        voyage_id, flow, port, size, area_no, planned_boxes. Extra columns are
        ignored.
        """
        _medium_rows, small_rows = self.make_small_and_repaired_medium_rows_from_medium_rows(medium_rows)
        return small_rows

    def make_small_and_repaired_medium_rows_from_medium_rows(
        self,
        medium_rows: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Construct a small plan and the medium rows that cover the repaired placement."""
        medium_counter = self._medium_counter_from_external_rows(medium_rows)
        self.small_unplaced_by_group = Counter()
        self.small_plan_construction_mode = ""
        self.final_small_plan_failure = ""
        small_rows = self._make_feasible_doc_small_rows(medium_counter)
        repaired_medium_counter = self._repair_medium_counter_to_cover_small_rows(medium_counter, small_rows)
        return self._medium_rows_from_counter(repaired_medium_counter), small_rows

    def _medium_counter_from_external_rows(self, medium_rows: list[dict]) -> Counter[tuple]:
        self.small_unplaced_by_group = Counter()
        self.small_plan_construction_mode = ""
        self.final_small_plan_failure = ""
        medium_counter: Counter[tuple] = Counter()
        required = {"voyage_id", "flow", "port", "size", "area_no", "planned_boxes"}
        for index, row in enumerate(medium_rows, start=1):
            missing = [column for column in required if column not in row]
            if missing:
                raise ValueError(f"medium plan row {index} missing columns: {missing}")
            try:
                planned_boxes = int(round(float(row.get("planned_boxes") or 0)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"medium plan row {index} has invalid planned_boxes: {row.get('planned_boxes')!r}") from exc
            if planned_boxes <= 0:
                continue
            medium_counter[
                (
                    str(row.get("voyage_id", "")).strip(),
                    str(row.get("flow", "")).strip(),
                    str(row.get("port", "")).strip(),
                    str(row.get("size", "")).strip(),
                    str(row.get("area_no", "")).strip(),
                )
            ] += planned_boxes
        return medium_counter

    def _small_plan_proxy_energy_from_assignment(self, medium_assignment: MediumAssignment) -> float:
        coarse_weights: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        groups_by_id = {group.group_id: group for group in self.problem.groups}
        for group_id, area_counts in medium_assignment.items():
            group = groups_by_id[group_id]
            for area_no, qty in area_counts.items():
                if qty > 0:
                    coarse_weights[(group.voyage_id, group.status, group.port, group.size)][area_no] += qty
        return self._small_plan_proxy_energy(coarse_weights)

    def _small_plan_proxy_energy(self, coarse_weights: dict[tuple[str, str, str, str], Counter[str]]) -> float:
        """Cheap small-plan quality proxy used inside medium-plan SA.

        This does not assign individual bays. It projects document-container fine
        groups onto the medium-plan area pattern and scores bay-level risk signals
        that are cheap to compute: size capacity, 45ft edge capacity, six-small-bay
        affinity pressure.
        """
        if not self.problem.small_groups:
            return 0.0

        try:
            allocations = self._small_group_area_allocations(coarse_weights)
        except SmallPlanInfeasible:
            return self.config.small_plan_feedback_penalty * 1000.0
        area_total: Counter[str] = Counter()
        area_size: Counter[tuple[str, str]] = Counter()
        area_size_height: Counter[tuple[str, str, str]] = Counter()
        area_affinity: defaultdict[tuple[str, str], set[tuple]] = defaultdict(set)
        area_special_codes: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        energy = 0.0

        groups_by_id = {group.group_id: group for group in self.problem.small_groups}
        for group_id, area_counts in allocations.items():
            group = groups_by_id[group_id]
            affinity = self._small_affinity(group)
            used_area_count = sum(1 for qty in area_counts.values() if qty > 0)
            energy += self.config.small_plan_group_area_split_penalty * max(0, used_area_count - 1)
            for area_no, qty in area_counts.items():
                if qty <= 0:
                    continue
                area_total[area_no] += qty
                area_size[(area_no, group.size)] += qty
                area_size_height[(area_no, group.size, group.height)] += qty
                area_affinity[(group.voyage_id, area_no)].add(affinity)
                if group.special_stow_code:
                    area_special_codes[(group.voyage_id, area_no)].add(group.special_stow_code)

        for area_no, qty in area_total.items():
            overflow = max(0, qty - self.area_total_cap[area_no])
            energy += self.config.small_plan_proxy_capacity_penalty * overflow
        for (area_no, size), qty in area_size.items():
            overflow = max(0, qty - self.area_size_cap[(area_no, size)])
            energy += self.config.small_plan_proxy_capacity_penalty * overflow
            if size == "45":
                edge_overflow = max(0, qty - self.area_edge_size_cap[(area_no, "45")])
                energy += self.config.small_plan_proxy_capacity_penalty * edge_overflow
        for (area_no, size, height), qty in area_size_height.items():
            overflow = max(0, qty - self.area_size_height_cap[(area_no, size, height)])
            energy += self.config.small_plan_proxy_height_capacity_penalty * overflow

        for (voyage_id, area_no), affinities in area_affinity.items():
            preferred_blocks = self.area_six_block_count[area_no]
            if preferred_blocks > 0:
                overflow = max(0, len(affinities) - preferred_blocks)
                energy += self.config.small_plan_proxy_six_block_conflict_penalty * overflow
            special_variety = len(area_special_codes.get((voyage_id, area_no), set()))
            energy += self.config.special_stow_isolation_penalty * max(0, special_variety - 1)
        return energy

    def _small_group_area_allocations(
        self,
        coarse_weights: dict[tuple[str, str, str, str], Counter[str]],
        strict: bool = True,
    ) -> dict[str, dict[str, int]]:
        allocations: dict[str, dict[str, int]] = {}
        voyage_flow_size_weights: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        for (voyage_id, flow, _port, size), weights in coarse_weights.items():
            voyage_flow_size_weights[(voyage_id, flow, size)].update(weights)
        remaining_inherited_by_key: dict[tuple[str, str, str], Counter[str]] = {
            key: Counter(weights)
            for key, weights in voyage_flow_size_weights.items()
        }
        for (voyage_id, flow, size), weights in remaining_inherited_by_key.items():
            big_size = self._big_plan_size(size)
            for area_no in list(weights):
                quota_key = (voyage_id, flow, area_no, big_size)
                quota = self.problem.area_size_quota.get(quota_key, 0)
                if quota > 0:
                    weights[area_no] = min(weights[area_no], quota)
                    if weights[area_no] <= 0:
                        del weights[area_no]

        groups_by_key: defaultdict[tuple[str, str, str, str], list] = defaultdict(list)
        for group in self.problem.small_groups:
            groups_by_key[(group.voyage_id, group.status, group.port, group.size)].append(group)

        remaining_height_cap: Counter[tuple[str, str, str]] = Counter(self.area_size_height_cap)
        for key, groups in sorted(groups_by_key.items()):
            voyage_id, flow, port, size = key
            inherited_key = (voyage_id, flow, size)
            # Strict inheritance is enforced at (voyage, flow, size, area).
            # Port-level medium weights are not an upper bound because predicted
            # ports and document ports may differ materially.
            remaining_inherited_cap = remaining_inherited_by_key.get(inherited_key)
            weights = remaining_inherited_cap or Counter()
            if not weights:
                weights = self._fallback_area_weights(voyage_id, flow, size)
                remaining_inherited_cap = remaining_inherited_by_key.setdefault(inherited_key, Counter(weights))
            if remaining_inherited_cap is None:
                remaining_inherited_cap = remaining_inherited_by_key.setdefault(inherited_key, Counter())
            remaining_pool: Counter[str] = Counter(
                {
                    area: qty
                    for area, qty in remaining_inherited_cap.items()
                    if qty > 0
                }
            )
            preferred_areas: Counter[str] = Counter()
            for group in sorted(groups, key=self._small_group_difficulty_key):
                group_alloc = self._allocate_small_group_to_concentrated_areas(
                    group,
                    remaining_pool,
                    weights,
                    remaining_height_cap,
                    remaining_inherited_cap,
                    preferred_areas,
                )
                if sum(group_alloc.values()) != group.demand:
                    if not strict:
                        allocations[group.group_id] = group_alloc
                        self.small_unplaced_by_group[group.group_id] += group.demand - sum(group_alloc.values())
                        for area_no, qty in group_alloc.items():
                            remaining_pool[area_no] -= qty
                            remaining_inherited_cap[area_no] -= qty
                            remaining_height_cap[(area_no, group.size, group.height)] -= qty
                            preferred_areas[area_no] += qty
                        continue
                    area_no = next((area for area, qty in weights.items() if qty > 0), "")
                    raise SmallPlanInfeasible(
                        (
                            f"small plan cannot inherit enough medium quota for "
                            f"voyage={group.voyage_id}, flow={group.status}, port={group.port}, "
                            f"size={group.size}, demand={group.demand}, allocated={sum(group_alloc.values())}"
                        ),
                        group.voyage_id,
                        group.status,
                        group.port,
                        group.size,
                        area_no,
                    )
                allocations[group.group_id] = group_alloc
                for area_no, qty in group_alloc.items():
                    remaining_pool[area_no] -= qty
                    remaining_inherited_cap[area_no] -= qty
                    remaining_height_cap[(area_no, group.size, group.height)] -= qty
                    preferred_areas[area_no] += qty
        return allocations

    def _allocate_small_group_to_concentrated_areas(
        self,
        group,
        remaining_pool: Counter[str],
        weights: Counter[str],
        remaining_height_cap: Counter[tuple[str, str, str]],
        remaining_inherited_cap: Counter[str],
        preferred_areas: Counter[str] | None = None,
    ) -> dict[str, int]:
        allocation: Counter[str] = Counter()
        remaining = group.demand
        preferred_areas = preferred_areas or Counter()
        candidates = [
            area_no
            for area_no, qty in weights.items()
            if (
                qty > 0
                and remaining_inherited_cap[area_no] > 0
                and self.area_total_cap[area_no] > 0
                and self.area_size_cap[(area_no, group.size)] > 0
                and remaining_height_cap[(area_no, group.size, group.height)] > 0
            )
        ]
        candidates.sort(
            key=lambda area_no: (
                0 if not preferred_areas or preferred_areas[area_no] > 0 else 1,
                0 if remaining_pool[area_no] >= group.demand else 1,
                -preferred_areas[area_no],
                -remaining_height_cap[(area_no, group.size, group.height)],
                -remaining_pool[area_no],
                -weights[area_no],
                area_no,
            )
        )
        for area_no in candidates:
            take = min(
                remaining,
                max(0, remaining_pool[area_no]),
                max(0, remaining_inherited_cap[area_no] - allocation[area_no]),
                max(0, remaining_height_cap[(area_no, group.size, group.height)] - allocation[area_no]),
            )
            if take <= 0:
                continue
            allocation[area_no] += take
            remaining -= take
            if remaining == 0:
                break
        if remaining > 0:
            for area_no in candidates:
                take = min(
                    remaining,
                    max(0, remaining_inherited_cap[area_no] - allocation[area_no]),
                    max(0, remaining_height_cap[(area_no, group.size, group.height)] - allocation[area_no]),
                )
                if take <= 0:
                    continue
                allocation[area_no] += take
                remaining -= take
                if remaining == 0:
                    break
        return dict(allocation)

    def _areas_with_45(self, allocations: dict[str, dict[str, int]]) -> set[str]:
        group_by_id = {group.group_id: group for group in self.problem.small_groups}
        areas: set[str] = set()
        for group_id, area_counts in allocations.items():
            group = group_by_id[group_id]
            if group.size != "45":
                continue
            areas.update(area_no for area_no, qty in area_counts.items() if qty > 0)
        return areas

    def _fallback_area_weights(self, voyage_id: str, flow: str, size: str) -> Counter[str]:
        weights: Counter[str] = Counter()
        for (v, f, area_no, quota_size), qty in self.problem.area_size_quota.items():
            if v == voyage_id and f == flow and quota_size == size:
                weights[area_no] += qty
        return weights

    def _big_plan_size(self, size: str) -> str:
        return "40" if size == "45" else size if size in {"20", "40"} else "40"

    def _small_group_difficulty_key(self, group) -> tuple[int, int, int, int, str, str, str]:
        feasible_areas = [
            area_no
            for (area_no, size, height), cap in self.area_size_height_cap.items()
            if size == group.size and height == group.height and cap > 0
        ]
        total_height_capacity = sum(
            self.area_size_height_cap[(area_no, group.size, group.height)]
            for area_no in feasible_areas
        )
        size_priority = {"45": 0, "20": 1, "40": 2}.get(group.size, 3)
        return (
            size_priority,
            len(feasible_areas),
            total_height_capacity,
            -group.demand,
            group.height,
            group.weight_class,
            group.group_id,
        )

    def _build_small_proxy_height_capacity(self) -> None:
        heights_by_size: defaultdict[str, set[str]] = defaultdict(set)
        for group in self.problem.small_groups:
            heights_by_size[group.size].add(group.height)
        for bay_key, bay in self.bays.items():
            is_edge = self._is_area_edge_bay(bay_key)
            for size, heights in heights_by_size.items():
                cap = bay.cap_by_size.get(size, 0)
                if cap <= 0:
                    continue
                if size in {"40", "45"} and not self._bay_footprint_keys(bay_key, size):
                    continue
                if size == "20" and is_edge:
                    continue
                if size == "45" and not is_edge:
                    continue
                footprint = self._bay_footprint_keys(bay_key, size)
                existing_sizes = set().union(*(self.bays[key].existing_size_modes for key in footprint))
                if existing_sizes and existing_sizes != {size}:
                    continue
                existing_heights = set().union(*(self.bays[key].existing_heights for key in footprint))
                for height in heights:
                    if existing_heights and existing_heights != {height}:
                        continue
                    self.area_size_height_cap[(bay.area_no, size, height)] += cap

    def _bay_no_mix_attrs(self) -> tuple[str, ...]:
        rules = getattr(self.problem, "attribute_rules", None)
        attrs = getattr(rules, "bay_no_mix_attributes", ("size", "height"))
        return tuple(str(attr) for attr in attrs if str(attr))

    def _row_no_mix_attrs(self) -> tuple[str, ...]:
        rules = getattr(self.problem, "attribute_rules", None)
        attrs = getattr(rules, "row_no_mix_attributes", ("port",))
        return tuple(str(attr) for attr in attrs if str(attr))

    @staticmethod
    def _group_attr_value(group, attr: str) -> str:
        if attr == "flow":
            attr = "status"
        value = getattr(group, attr, "")
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def _row_mix_key_for_group(self, group) -> str:
        return "|".join(f"{attr}={self._group_attr_value(group, attr)}" for attr in self._row_no_mix_attrs()) or "__all__"

    def _bay_no_mix_signature(self, group) -> tuple[str, ...]:
        return tuple(self._group_attr_value(group, attr) for attr in self._bay_no_mix_attrs())

    def _row_existing_attrs_allow_group(self, bay: Bay, row_no: str, group) -> bool:
        row_attrs = getattr(bay, "existing_attrs_by_row", {}).get(str(row_no), {})
        for attr in self._row_no_mix_attrs():
            values = set(row_attrs.get(attr, set()))
            if values and self._group_attr_value(group, attr) not in values:
                return False
        return True

    def _bay_existing_attrs_allow_group(self, group, footprint: tuple[str, ...]) -> bool:
        for key in footprint:
            bay = self.bays[key]
            existing_attrs = getattr(bay, "existing_attrs", {})
            for attr in self._bay_no_mix_attrs():
                values = set(existing_attrs.get(attr, set()))
                if values and values != {self._group_attr_value(group, attr)}:
                    return False
        return True

    def _row_stack_capacities_for_group(self, bay_key: str, size: str, group) -> list[int]:
        bay = self.bays[bay_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or {}
        if not row_caps and bay.row_physical_capacity:
            row_caps = bay.row_physical_capacity
        has_row_caps = bool(row_caps)
        capacities: list[int] = []
        for row_no, cap in row_caps.items():
            if not self._row_existing_attrs_allow_group(bay, str(row_no), group):
                continue
            cap_int = int(cap)
            if cap_int > 0:
                capacities.append(cap_int)
        if has_row_caps:
            return capacities
        fallback = int(bay.cap_by_size.get(size, 0) or bay.physical_capacity)
        return [fallback] if fallback > 0 else []

    def _stack_count_for_group(self, bay_key: str, size: str, group) -> int:
        return len(self._row_stack_capacities_for_group(bay_key, size, group))

    def _stack_count_for_bay_size(self, bay_key: str, size: str) -> int:
        bay = self.bays[bay_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or bay.row_physical_capacity
        if row_caps:
            return sum(1 for cap in row_caps.values() if int(cap) > 0)
        return 1 if int(bay.cap_by_size.get(size, 0) or bay.physical_capacity) > 0 else 0

    def _stack_unit_capacity_for_group(self, bay_key: str, size: str, group) -> int:
        capacities = self._row_stack_capacities_for_group(bay_key, size, group)
        return max(capacities) if capacities else 0

    def _stack_units_for_quantity(self, bay_key: str, size: str, group, quantity: int) -> int:
        if quantity <= 0:
            return 0
        unit_capacity = self._stack_unit_capacity_for_group(bay_key, size, group)
        if unit_capacity <= 0:
            return 10**9
        return int(math.ceil(quantity / unit_capacity))

    def _remaining_stack_capacity_for_small_group(
        self,
        group,
        bay_key: str,
        bay_port_size_load: Counter[tuple[str, str, str]],
        bay_stack_used: Counter[tuple[str, str]],
    ) -> int:
        capacity = 10**9
        row_mix_key = self._row_mix_key_for_group(group)
        for footprint_key in self._bay_footprint_keys(bay_key, group.size):
            unit_capacity = self._stack_unit_capacity_for_group(footprint_key, group.size, group)
            port_stack_count = self._stack_count_for_group(footprint_key, group.size, group)
            total_stack_count = self._stack_count_for_bay_size(footprint_key, group.size)
            if unit_capacity <= 0 or port_stack_count <= 0 or total_stack_count <= 0:
                return 0
            port_key = (footprint_key, row_mix_key, group.size)
            total_key = (footprint_key, group.size)
            current_load = bay_port_size_load[port_key]
            current_units = self._stack_units_for_quantity(footprint_key, group.size, group, current_load)
            other_units = bay_stack_used[total_key] - current_units
            capacity = min(capacity, port_stack_count * unit_capacity - current_load)
            capacity = min(capacity, max(0, total_stack_count - other_units) * unit_capacity - current_load)
        return max(0, int(capacity))

    def _apply_stack_usage_for_small_group(
        self,
        group,
        bay_key: str,
        quantity: int,
        bay_port_size_load: Counter[tuple[str, str, str]],
        bay_stack_used: Counter[tuple[str, str]],
    ) -> None:
        row_mix_key = self._row_mix_key_for_group(group)
        for footprint_key in self._bay_footprint_keys(bay_key, group.size):
            port_key = (footprint_key, row_mix_key, group.size)
            total_key = (footprint_key, group.size)
            before = bay_port_size_load[port_key]
            before_units = self._stack_units_for_quantity(footprint_key, group.size, group, before)
            after = before + quantity
            after_units = self._stack_units_for_quantity(footprint_key, group.size, group, after)
            bay_port_size_load[port_key] = after
            bay_stack_used[total_key] += after_units - before_units

    def _candidate_bays_for_small_group(
        self,
        group,
        area_no: str,
        bay_load: Counter[str],
        bay_used_size: dict[str, str],
        bay_used_height: dict[str, str],
        bay_affinity: dict[str, tuple],
        bay_coarse_affinity: dict[str, tuple],
        block_affinity: dict[str, tuple],
        block_coarse_affinity: dict[str, tuple],
        area_has_45: set[str],
    ) -> list[str]:
        candidates = [
            bay_key for bay_key in self.bays_by_area.get(area_no, [])
            if self._small_bay_hard_feasible(
                group,
                area_no,
                bay_key,
                bay_used_size,
                bay_used_height,
                bay_affinity,
                area_has_45,
            )
        ]
        return sorted(
            candidates,
            key=lambda bay_key: (
                self._small_block_score(group, area_no, bay_key, block_affinity, block_coarse_affinity),
                self._bay_affinity_score(group, bay_key, bay_affinity, bay_coarse_affinity),
                self.bays[bay_key].is_fallback_bay,
                0 if group.port in self.bays[bay_key].existing_ports else 1,
                bay_load[bay_key],
                self.bays[bay_key].bay_order,
            ),
        )

    def _small_bay_hard_feasible(
        self,
        group,
        area_no: str,
        bay_key: str,
        bay_used_size: dict[str, str],
        bay_used_height: dict[str, str],
        bay_affinity: dict[str, tuple],
        area_has_45: set[str],
    ) -> bool:
        bay = self.bays[bay_key]
        if bay.cap_by_size.get(group.size, 0) <= 0:
            return False
        footprint = self._bay_footprint_keys(bay_key, group.size)
        if not footprint:
            return False
        if not self._bay_existing_attrs_allow_group(group, footprint):
            return False
        existing_signature = bay_affinity.get(bay_key)
        if existing_signature is not None and existing_signature != self._bay_no_mix_signature(group):
            return False
        existing_sizes = set().union(*(self.bays[key].existing_size_modes for key in footprint))
        if "size" in self._bay_no_mix_attrs() and existing_sizes and existing_sizes != {group.size}:
            return False
        if "size" in self._bay_no_mix_attrs() and any(bay_used_size.get(key) is not None and bay_used_size[key] != group.size for key in footprint):
            return False
        existing_heights = set().union(*(self.bays[key].existing_heights for key in footprint))
        if "height" in self._bay_no_mix_attrs() and existing_heights and existing_heights != {group.height}:
            return False
        if "height" in self._bay_no_mix_attrs() and any(bay_used_height.get(key) is not None and bay_used_height[key] != group.height for key in footprint):
            return False
        is_edge = self._is_area_edge_bay(bay_key)
        if group.size == "20" and is_edge:
            return False
        if group.size == "45" and not is_edge:
            return False
        if is_edge and area_no in area_has_45 and group.size != "45":
            return False
        return True

    def _small_affinity(self, group) -> tuple:
        return self._bay_no_mix_signature(group)

    def _small_coarse_affinity(self, group) -> tuple:
        return (
            group.voyage_id,
            group.status,
            group.port,
            group.size,
        )

    def _bay_affinity_score(
        self,
        group,
        bay_key: str,
        bay_affinity: dict[str, tuple],
        bay_coarse_affinity: dict[str, tuple],
    ) -> int:
        affinity = bay_affinity.get(bay_key)
        group_is_isolated = group.pre_stow or group.special_stow
        if affinity is None:
            return 0 if group_is_isolated else 2
        if affinity == self._small_affinity(group):
            return 0
        affinity_is_isolated = bool(affinity[-2] or affinity[-1])
        if group_is_isolated or affinity_is_isolated:
            return self.config.special_stow_isolation_penalty
        if bay_coarse_affinity.get(bay_key) == self._small_coarse_affinity(group):
            return 1
        return 5

    def _small_block_score(
        self,
        group,
        area_no: str,
        bay_key: str,
        block_affinity: dict[str, tuple],
        block_coarse_affinity: dict[str, tuple],
    ) -> int:
        block_id = self._small_block_id_for_bay(area_no, bay_key)
        if not block_id:
            return 6
        affinity = block_affinity.get(block_id)
        group_is_isolated = group.pre_stow or group.special_stow
        if affinity is None:
            return 0 if group_is_isolated else 1
        if affinity == self._small_affinity(group):
            return 0
        affinity_is_isolated = bool(affinity[-2] or affinity[-1])
        if group_is_isolated or affinity_is_isolated:
            return max(30, int(self.config.special_stow_isolation_penalty * 2))
        if block_coarse_affinity.get(block_id) == self._small_coarse_affinity(group):
            return 0
        return max(15, int(self.config.small_plan_proxy_six_block_conflict_penalty))

    def _small_block_id_for_bay(self, area_no: str, bay_key: str) -> str:
        for block_id, members in self._six_bay_blocks_by_area(area_no).items():
            if bay_key in members:
                return block_id
        return ""

    def _small_block_bay_nos(self, area_no: str, block_id: str) -> tuple[str, ...]:
        members = self._six_bay_blocks_by_area(area_no).get(block_id, ())
        return tuple(self.bays[bay_key].bay_no for bay_key in members)

    def _six_bay_blocks_by_area(self, area_no: str) -> dict[str, tuple[str, ...]]:
        bay_keys = self.bays_by_area.get(area_no, [])
        blocks: dict[str, tuple[str, ...]] = {}
        start = 0
        block_index = 1
        while start <= len(bay_keys) - 6:
            members = tuple(bay_keys[start : start + 6])
            if self._is_preferred_six_bay_block(members):
                blocks[f"{area_no}-SB{block_index:02d}"] = members
                block_index += 1
                start += 6
                continue
            start += 1
        return blocks

    def _is_preferred_six_bay_block(self, bay_keys: tuple[str, ...]) -> bool:
        if len(bay_keys) != 6:
            return False
        member_set = set(bay_keys)
        large_starts = [
            key for key in bay_keys
            if (
                (self.bays[key].cap_by_size.get("40", 0) > 0 or self.bays[key].cap_by_size.get("45", 0) > 0)
                and self.bays[key].large_bay_partner_key in member_set
            )
        ]
        for left_index, left in enumerate(large_starts):
            left_pair = {left, self.bays[left].large_bay_partner_key}
            for right in large_starts[left_index + 1:]:
                right_pair = {right, self.bays[right].large_bay_partner_key}
                if left_pair & right_pair:
                    continue
                remaining = [key for key in bay_keys if key not in left_pair and key not in right_pair]
                if sum(1 for key in remaining if self.bays[key].cap_by_size.get("20", 0) > 0) >= 2:
                    return True
        return False

    @staticmethod
    def _allocate_by_weights(weights: dict[str, int], total: int) -> dict[str, int]:
        items = [(key, value) for key, value in sorted(weights.items()) if value > 0]
        if not items or total <= 0:
            return {}
        source_total = sum(value for _, value in items)
        raw = [value * total / source_total for _, value in items]
        base = [int(value) for value in raw]
        remain = total - sum(base)
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
        for idx in order[:remain]:
            base[idx] += 1
        return {key: qty for (key, _), qty in zip(items, base) if qty > 0}

    def _medium_attrs(self, group) -> tuple:
        return (
            group.voyage_id,
            group.status,
            group.port,
            group.size,
        )

    def _area_has_loading_during_window(self, voyage_id: str, area_no: str) -> bool:
        window_start, window_end = self.problem.voyage_windows[voyage_id]
        return any(
            op.voyage_id != voyage_id and _time_windows_overlap(op.start_time, op.end_time, window_start, window_end)
            for op in self.problem.area_operations.get(area_no, [])
        )

    def _area_has_loading_after_window(self, voyage_id: str, area_no: str) -> bool:
        _, window_end = self.problem.voyage_windows[voyage_id]
        prefer_end = window_end + timedelta(hours=24)
        return any(
            op.voyage_id != voyage_id and _time_windows_overlap(op.start_time, op.end_time, window_end, prefer_end)
            for op in self.problem.area_operations.get(area_no, [])
        )


def write_rows(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _time_windows_overlap(start_a, end_a, start_b, end_b) -> bool:
    return start_a < end_b and start_b < end_a

