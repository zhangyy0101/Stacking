from __future__ import annotations

import csv
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
                self.area_edge_bays[area_no] = {bay_keys[0], bay_keys[-1]}
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
        diagnostics = {
            "iterations": self.config.iterations,
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
            "sa_energy": round(medium_energy, 4),
            "medium_energy": round(medium_energy, 4),
            "small_plan_proxy_energy": round(self._small_plan_proxy_energy_from_assignment(medium_assignment), 4),
            "medium_box_count": sum(group.demand for group in self.problem.groups),
            "medium_decision_group_count": len(self.problem.groups),
            "group_count": len(self.problem.groups),
            "bay_count": len(self.problem.bays),
            "area_count": len(self.bays_by_area),
            "medium_row_count": len(medium_rows),
            "small_row_count": len(small_rows),
            "small_plan_used_six_bay_block_count": len(
                {row.get("six_bay_block_id") for row in small_rows if row.get("six_bay_block_id")}
            ),
            "small_doc_group_count": len(self.problem.small_groups),
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
            "attribute_balance_policy": "medium-plan soft constraint: distribute each flow/discharge-port/size attribute group across areas following the big-plan area pattern",
            "medium_concentrated_group_threshold": self.config.medium_concentrated_group_threshold,
            "medium_small_group_policy": (
                "medium-plan coarse groups at or below the threshold prefer concentrated yard-area assignment; "
                "larger groups keep proportional area balancing"
            ),
            "big_plan_area_policy": (
                "big-plan voyage/flow/area/size quantities are strict upper bounds; "
                "medium-plan SA may rebalance within those inherited bounds but cannot use "
                "area-size combinations absent from the big plan"
            ),
            "tops_policy": "TOPS bay ranges for non-target voyages are closed before medium and small planning; TOPS records for target voyages are ignored",
            "small_plan_six_bay_policy": (
                "small plan prefers dynamic six-small-bay continuous blocks first for one fine group "
                "and then for the same voyage/flow/discharge-port/size coarse group; different sizes "
                "may share that preferred block but cannot share the same bay"
            ),
            "small_plan_proxy_policy": (
                "medium-plan simulated annealing includes a lightweight small-plan proxy score "
                "for doc-container bay capacity, size/height bay capacity, 45ft edge capacity, "
                "and six-small-bay block affinity"
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

    def _make_outputs(self, medium_assignment: MediumAssignment) -> tuple[list[dict], list[dict]]:
        """汇总中计划和小计划输出行。"""
        medium_counter = self._medium_counter_from_assignment(medium_assignment)

        medium_rows = []
        for key, count in sorted(medium_counter.items()):
            (
                voyage_id,
                flow,
                port,
                size,
                area_no,
            ) = key
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

        small_rows = self._make_doc_small_rows(medium_counter)
        return medium_rows, small_rows

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

    def _check_small_plan_feedback(self, medium_assignment: MediumAssignment, iteration: int) -> bool:
        """Run a periodic small-plan feasibility probe and add feedback immediately."""
        try:
            self._make_doc_small_rows(self._medium_counter_from_assignment(medium_assignment))
        except SmallPlanInfeasible as exc:
            key = (exc.voyage_id, exc.status, exc.port, exc.size, exc.area_no)
            broad_key = (exc.voyage_id, exc.status, "*", exc.size, exc.area_no)
            self.small_plan_area_feedback[key] += 1
            self.small_plan_area_feedback[broad_key] += 1
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{stamp}] small-plan feedback iter={iteration}: "
                f"voyage={exc.voyage_id} flow={exc.status} port={exc.port} size={exc.size} "
                f"area={exc.area_no} count={self.small_plan_area_feedback[key]}",
                flush=True,
            )
            return True
        self._remember_small_feasible_assignment(medium_assignment, iteration)
        return False

    def _remember_small_feasible_assignment(self, medium_assignment: MediumAssignment, iteration: int) -> None:
        energy = self.medium_energy(medium_assignment)
        if self.best_small_feasible_energy is not None and energy >= self.best_small_feasible_energy:
            return
        self.best_small_feasible_assignment = self._copy_assignment(medium_assignment)
        self.best_small_feasible_iteration = iteration
        self.best_small_feasible_energy = energy

    def _make_doc_small_rows(self, medium_counter: Counter[tuple]) -> list[dict]:
        coarse_weights: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        for key, count in medium_counter.items():
            voyage_id, flow, port, size, area_no = key
            coarse_weights[(voyage_id, flow, port, size)][area_no] += count

        group_area_allocations = self._small_group_area_allocations(coarse_weights)
        area_has_45 = self._areas_with_45(group_area_allocations)
        bay_load: Counter[str] = Counter()
        bay_size_load: Counter[tuple[str, str]] = Counter()
        bay_used_size: dict[str, str] = {}
        bay_used_height: dict[str, str] = {}
        bay_affinity: dict[str, tuple] = {}
        bay_coarse_affinity: dict[str, tuple] = {}
        block_affinity: dict[str, tuple] = {}
        block_coarse_affinity: dict[str, tuple] = {}
        small_counter: Counter[tuple] = Counter()
        block_members_by_id: dict[str, tuple[str, ...]] = {}

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
                    free_total = bay.physical_capacity - bay_load[bay_key]
                    free_size = bay.cap_by_size.get(group.size, 0) - bay_size_load[(bay_key, group.size)]
                    take = min(remaining, free_total, free_size)
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
                    bay_load[bay_key] += take
                    bay_size_load[(bay_key, group.size)] += take
                    bay_used_size[bay_key] = group.size
                    bay_used_height[bay_key] = group.height
                    bay_affinity.setdefault(bay_key, self._small_affinity(group))
                    bay_coarse_affinity.setdefault(bay_key, self._small_coarse_affinity(group))
                    if block_id:
                        block_affinity.setdefault(block_id, self._small_affinity(group))
                        block_coarse_affinity.setdefault(block_id, self._small_coarse_affinity(group))
                        block_members_by_id.setdefault(block_id, self._small_block_bay_nos(area_no, block_id))
                    remaining -= take
                    if remaining == 0:
                        break
                if remaining > 0:
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

    def make_small_rows_from_medium_rows(self, medium_rows: list[dict]) -> list[dict]:
        """Construct the small plan from externally supplied medium-plan rows.

        The expected row format is the existing medium output CSV schema:
        voyage_id, flow, port, size, area_no, planned_boxes. Extra columns are
        ignored.
        """
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
        return self._make_doc_small_rows(medium_counter)

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
    ) -> dict[str, dict[str, int]]:
        allocations: dict[str, dict[str, int]] = {}
        voyage_flow_size_weights: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        for (voyage_id, flow, _port, size), weights in coarse_weights.items():
            voyage_flow_size_weights[(voyage_id, flow, size)].update(weights)
        remaining_inherited_by_key: dict[tuple[str, str, str], Counter[str]] = {
            key: Counter(weights)
            for key, weights in voyage_flow_size_weights.items()
        }

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
                if size == "20" and is_edge:
                    continue
                if size == "45" and not is_edge:
                    continue
                existing_sizes = set(bay.existing_size_modes)
                if existing_sizes and existing_sizes != {size}:
                    continue
                existing_heights = set(bay.existing_heights)
                for height in heights:
                    if existing_heights and existing_heights != {height}:
                        continue
                    self.area_size_height_cap[(bay.area_no, size, height)] += cap

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
        area_has_45: set[str],
    ) -> bool:
        bay = self.bays[bay_key]
        if bay.cap_by_size.get(group.size, 0) <= 0:
            return False
        existing_sizes = set(bay.existing_size_modes)
        if existing_sizes and existing_sizes != {group.size}:
            return False
        used_size = bay_used_size.get(bay_key)
        if used_size is not None and used_size != group.size:
            return False
        existing_heights = set(bay.existing_heights)
        if existing_heights and existing_heights != {group.height}:
            return False
        used_height = bay_used_height.get(bay_key)
        if used_height is not None and used_height != group.height:
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
        # Same fine attribute excluding size may share a six-bay block; same
        # size is still required for sharing an individual bay by hard rule.
        return (
            group.voyage_id,
            group.status,
            group.port,
            group.height,
            group.weight_class,
            group.pre_stow,
            group.special_stow_code,
        )

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
        big_count = sum(
            1 for key in bay_keys
            if self.bays[key].cap_by_size.get("40", 0) > 0 or self.bays[key].cap_by_size.get("45", 0) > 0
        )
        small_flags = [self.bays[key].cap_by_size.get("20", 0) > 0 for key in bay_keys]
        has_two_consecutive_small = any(a and b for a, b in zip(small_flags, small_flags[1:]))
        return big_count >= 2 and sum(small_flags) >= 2 and has_two_consecutive_small

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

