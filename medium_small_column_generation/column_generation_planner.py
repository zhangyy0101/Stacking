from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from medium_small_column_generation.block_bay_planning.models import Bay, ProblemData, SmallBoxGroup


SIZE_ORDER = {"45": 0, "20": 1, "40": 2}
EXPORT_FLOWS = frozenset({"OF"})


@dataclass(frozen=True)
class PlacementColumn:
    column_id: str
    group_id: str
    demand_source: str
    voyage_id: str
    flow: str
    port: str
    size: str
    big_plan_size: str
    height: str
    weight_class: str
    special_stow_code: str
    area_no: str
    bay_key: str
    bay_no: str
    block_id: str
    block_bays: tuple[str, ...]
    quantity: int
    stack_units: int
    quota_key: tuple[str, str, str, str]
    coarse_key: tuple[str, str, str, str]
    intrinsic_cost: float


@dataclass
class ColumnGenerationConfig:
    max_iterations: int = 30
    columns_per_iteration: int = 2500
    stalled_pricing_columns: int = 500
    primal_expansion_columns: int = 800
    max_primal_expansion_rounds: int = 3
    primal_expansion_reduced_cost_limit: float = 1_000_000.0
    initial_columns_per_group: int = 16
    max_candidate_bays_per_group: int = 500
    mip_time_limit: float = 120.0
    mip_gap: float = 0.01
    verbose: bool = True
    use_gurobi: bool = True
    full_column_pool: bool = False
    demand_mode: str = "original"
    medium_plan_quota: dict[tuple[str, str, str, str, str], int] | None = None
    medium_plan_bay_quota: dict[tuple[str, str, str, str, str, str], int] | None = None
    unplaced_penalty: float = 1_000_000.0
    group_area_balance_penalty: float = 18.0
    medium_concentrated_group_threshold: int = 26
    medium_small_group_area_split_penalty: float = 900.0
    medium_small_group_fragment_penalty: float = 50.0
    medium_large_group_min_area_boxes: int = 10
    medium_large_group_small_area_penalty: float = 900.0
    big_plan_area_deviation_penalty: float = 8.0
    big_plan_fallback_tier_penalty: float = 120.0
    small_plan_group_area_split_penalty: float = 80.0
    small_plan_group_block_split_penalty: float = 35.0
    small_plan_group_bay_split_penalty: float = 8.0
    small_plan_coarse_area_block_split_penalty: float = 24.0
    small_plan_coarse_area_bay_split_penalty: float = 2.5
    berth_distance_penalty: float = 0.02
    active_loading_area_penalty: float = 16.0
    post_window_loading_area_reward: float = 5.0
    fallback_bay_penalty: float = 4.0
    non_preferred_block_penalty: float = 6.0
    port_mismatch_penalty: float = 1.0


@dataclass
class ColumnGenerationResult:
    medium_rows: list[dict]
    small_rows: list[dict]
    diagnostics: dict
    columns: list[PlacementColumn] = field(default_factory=list)


class ColumnGenerationPlanner:
    """Small-plan-first column generation.

    A column is a feasible placement pattern for one document-container fine
    group in one bay with a fixed quantity. The restricted master covers all
    fine groups and respects bay exclusivity/capacity. Big-plan area/size
    quantities are soft targets, and fallback areas remain available with a
    tiered inheritance penalty.
    """

    def __init__(self, problem: ProblemData, config: ColumnGenerationConfig | None = None) -> None:
        self.problem = problem
        self.config = config or ColumnGenerationConfig()
        self.group_source: dict[str, str] = {}
        self.demand_stats: dict[str, int | str] = {}
        self.groups = sorted(self._build_planning_groups(), key=self._group_sort_key)
        self.groups_by_id = {group.group_id: group for group in self.groups}
        self.bays = problem.bays
        self.attribute_rules = getattr(problem, "attribute_rules", None)
        self.bays_by_area: dict[str, list[str]] = defaultdict(list)
        self.area_edge_bays: dict[str, set[str]] = defaultdict(set)
        self.block_members_by_area: dict[str, dict[str, tuple[str, ...]]] = {}
        self.block_by_bay: dict[tuple[str, str], str] = {}
        self.block_bay_nos: dict[str, tuple[str, ...]] = {}
        self.area_size_height_cap: Counter[tuple[str, str, str]] = Counter()
        self.quota_by_key: Counter[tuple[str, str, str, str]] = Counter()
        self.group_demand = {group.group_id: int(group.demand) for group in self.groups}
        self.coarse_demand: Counter[tuple[str, str, str, str]] = Counter()
        self.voyage_flow_size_demand: Counter[tuple[str, str, str]] = Counter()
        self._columns: list[PlacementColumn] = []
        self._column_keys: set[tuple[str, str, int]] = set()
        self._candidate_cache: dict[tuple[str, str], list[tuple[str, int, float]]] = {}
        self._candidate_scope = "stage0"
        self._master_seed_selected: Counter[int] = Counter()
        self._master_seed_unplaced: Counter[str] = Counter()
        self._master_start_selected: Counter[int] = Counter()
        self._master_start_unplaced: Counter[str] = Counter()
        self._primal_expansion_rounds = 0
        self._prepare_yard_indexes()
        self._prepare_quota()
        for group in self.groups:
            self.coarse_demand[self._coarse_key(group)] += group.demand
            self.voyage_flow_size_demand[(group.voyage_id, group.status, self._big_plan_size(group.size))] += group.demand

    @property
    def columns(self) -> list[PlacementColumn]:
        return self._columns

    def _build_planning_groups(self) -> list[SmallBoxGroup]:
        mode = (self.config.demand_mode or "original").strip().lower().replace("_", "-")
        if mode not in {"original", "medium", "medium-with-doc-floor", "doc-only"}:
            raise ValueError(
                "demand_mode must be one of: original, medium, medium-with-doc-floor, doc-only"
            )

        source_doc_groups = list(self.problem.small_groups)
        source_doc_boxes = sum(group.demand for group in source_doc_groups)
        if mode == "doc-only":
            self.group_source = {group.group_id: "document" for group in source_doc_groups}
            self.demand_stats = {
                "demand_mode": mode,
                "source_doc_group_count": len(source_doc_groups),
                "source_doc_box_count": source_doc_boxes,
                "medium_target_group_count": len(self.problem.groups),
                "medium_target_box_count": sum(group.demand for group in self.problem.groups),
                "original_small_output_box_count": source_doc_boxes,
                "original_medium_output_box_count": 0,
                "forecast_fallback_group_count": 0,
                "forecast_fallback_box_count": 0,
                "dropped_doc_box_count": 0,
                "doc_boxes_outside_medium_target": 0,
                "planned_box_count": source_doc_boxes,
            }
            return source_doc_groups

        remaining: Counter[tuple[str, str, str, str]] = Counter()
        for group in self.problem.groups:
            remaining[(group.voyage_id, group.status, group.port, group.size)] += group.demand

        if mode == "original":
            return self._build_original_planning_groups(source_doc_groups, source_doc_boxes, remaining)

        planning_groups: list[SmallBoxGroup] = []
        dropped_doc_boxes = 0
        height_weights = self._forecast_height_weights(source_doc_groups)
        for group in sorted(source_doc_groups, key=lambda g: (g.voyage_id, g.status, g.port, SIZE_ORDER.get(g.size, 3), g.group_id)):
            key = self._small_group_coarse_key(group)
            if mode == "medium":
                take = min(group.demand, remaining.get(key, 0))
                if take <= 0:
                    dropped_doc_boxes += group.demand
                    continue
                if take < group.demand:
                    dropped_doc_boxes += group.demand - take
                planning_group = self._copy_group_with_demand(group, take)
                planning_groups.append(planning_group)
                self.group_source[planning_group.group_id] = "document"
                remaining[key] -= take
            else:
                planning_groups.append(group)
                self.group_source[group.group_id] = "document"
                remaining[key] -= group.demand

        forecast_group_count = 0
        forecast_box_count = 0
        for (voyage_id, flow, port, size), qty in sorted(remaining.items()):
            if qty <= 0:
                continue
            for height, height_qty in self._split_forecast_by_height(
                voyage_id,
                flow,
                port,
                size,
                int(qty),
                height_weights,
            ):
                forecast_group_count += 1
                group = SmallBoxGroup(
                    group_id=f"{voyage_id}_F{forecast_group_count:03d}",
                    voyage_id=voyage_id,
                    status=flow,
                    port=port,
                    size=size,
                    height=height,
                    weight_class="UNK",
                    demand=int(height_qty),
                    pre_stow=False,
                    special_stow=False,
                    special_stow_code="",
                )
                planning_groups.append(group)
                self.group_source[group.group_id] = "forecast_fallback"
                forecast_box_count += int(height_qty)

        self.demand_stats = {
            "demand_mode": mode,
            "source_doc_group_count": len(source_doc_groups),
            "source_doc_box_count": source_doc_boxes,
            "medium_target_group_count": len(self.problem.groups),
            "medium_target_box_count": sum(group.demand for group in self.problem.groups),
            "original_small_output_box_count": (
                source_doc_boxes if mode == "medium-with-doc-floor" else source_doc_boxes - dropped_doc_boxes
            ),
            "original_medium_output_box_count": sum(group.demand for group in self.problem.groups),
            "forecast_fallback_group_count": forecast_group_count,
            "forecast_fallback_box_count": forecast_box_count,
            "dropped_doc_box_count": dropped_doc_boxes,
            "doc_boxes_outside_medium_target": 0,
            "planned_box_count": sum(group.demand for group in planning_groups),
        }
        return planning_groups

    def _build_original_planning_groups(
        self,
        source_doc_groups: list[SmallBoxGroup],
        source_doc_boxes: int,
        remaining_medium: Counter[tuple[str, str, str, str]],
    ) -> list[SmallBoxGroup]:
        """Build the same demand scopes as the SA + heuristic pipeline.

        The original pipeline writes the medium plan from ``problem.groups`` and
        writes the small plan from all current document groups.  Document boxes
        consume only the exact medium coarse target with the same voyage, flow,
        port, and true size; the data loader lifts medium demand beforehand when
        the document floor is higher than the forecast medium target.
        """
        planning_groups = list(source_doc_groups)
        self.group_source = {group.group_id: "document" for group in source_doc_groups}
        doc_boxes_outside_medium_target = 0

        for group in sorted(source_doc_groups, key=lambda g: (g.voyage_id, g.status, g.port, SIZE_ORDER.get(g.size, 3), g.group_id)):
            doc_boxes_outside_medium_target += self._consume_medium_target_for_document_group(
                remaining_medium,
                group,
            )

        height_weights = self._forecast_height_weights(source_doc_groups)
        forecast_group_count = 0
        forecast_box_count = 0
        for (voyage_id, flow, port, size), qty in sorted(remaining_medium.items()):
            if qty <= 0:
                continue
            for height, height_qty in self._split_forecast_by_height(
                voyage_id,
                flow,
                port,
                size,
                int(qty),
                height_weights,
            ):
                forecast_group_count += 1
                group = SmallBoxGroup(
                    group_id=f"{voyage_id}_F{forecast_group_count:03d}",
                    voyage_id=voyage_id,
                    status=flow,
                    port=port,
                    size=size,
                    height=height,
                    weight_class="UNK",
                    demand=int(height_qty),
                    pre_stow=False,
                    special_stow=False,
                    special_stow_code="",
                )
                planning_groups.append(group)
                self.group_source[group.group_id] = "forecast_fallback"
                forecast_box_count += int(height_qty)

        self.demand_stats = {
            "demand_mode": "original",
            "source_doc_group_count": len(source_doc_groups),
            "source_doc_box_count": source_doc_boxes,
            "medium_target_group_count": len(self.problem.groups),
            "medium_target_box_count": sum(group.demand for group in self.problem.groups),
            "original_small_output_box_count": source_doc_boxes,
            "original_medium_output_box_count": sum(group.demand for group in self.problem.groups),
            "forecast_fallback_group_count": forecast_group_count,
            "forecast_fallback_box_count": forecast_box_count,
            "dropped_doc_box_count": 0,
            "doc_boxes_outside_medium_target": doc_boxes_outside_medium_target,
            "planned_box_count": sum(group.demand for group in planning_groups),
        }
        return planning_groups

    @staticmethod
    def _consume_medium_target_for_document_group(
        remaining: Counter[tuple[str, str, str, str]],
        group: SmallBoxGroup,
    ) -> int:
        need = int(group.demand)
        exact_key = (group.voyage_id, group.status, group.port, group.size)
        take = min(need, max(0, remaining.get(exact_key, 0)))
        if take > 0:
            remaining[exact_key] -= take
            need -= take
        if need <= 0:
            return 0
        return need

    @staticmethod
    def _copy_group_with_demand(group: SmallBoxGroup, demand: int) -> SmallBoxGroup:
        return SmallBoxGroup(
            group_id=group.group_id,
            voyage_id=group.voyage_id,
            status=group.status,
            port=group.port,
            size=group.size,
            height=group.height,
            weight_class=group.weight_class,
            demand=int(demand),
            pre_stow=group.pre_stow,
            special_stow=group.special_stow,
            special_stow_code=group.special_stow_code,
        )

    @staticmethod
    def _small_group_coarse_key(group: SmallBoxGroup) -> tuple[str, str, str, str]:
        return group.voyage_id, group.status, group.port, group.size

    @staticmethod
    def _forecast_height_weights(
        groups: list[SmallBoxGroup],
    ) -> dict[tuple, Counter[str]]:
        weights: dict[tuple, Counter[str]] = defaultdict(Counter)
        for group in groups:
            height = group.height or "HQ"
            keys = [
                (group.voyage_id, group.status, group.port, group.size),
                (group.voyage_id, group.status, group.size),
                (group.status, group.size),
                (group.size,),
                ("*",),
            ]
            for key in keys:
                weights[key][height] += group.demand
        return weights

    @classmethod
    def _split_forecast_by_height(
        cls,
        voyage_id: str,
        flow: str,
        port: str,
        size: str,
        qty: int,
        height_weights: dict[tuple, Counter[str]],
    ) -> list[tuple[str, int]]:
        if qty <= 0:
            return []
        for key in (
            (voyage_id, flow, port, size),
            (voyage_id, flow, size),
            (flow, size),
            (size,),
            ("*",),
        ):
            weights = height_weights.get(key)
            if weights:
                return cls._allocate_integer_by_weights(weights, qty)
        return [("HQ", qty)]

    @staticmethod
    def _allocate_integer_by_weights(weights: Counter[str], total: int) -> list[tuple[str, int]]:
        items = [(key, value) for key, value in sorted(weights.items()) if value > 0]
        if not items or total <= 0:
            return []
        source_total = sum(value for _key, value in items)
        raw = [value * total / source_total for _key, value in items]
        base = [int(value) for value in raw]
        remain = total - sum(base)
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
        for idx in order[:remain]:
            base[idx] += 1
        return [(key, qty) for (key, _value), qty in zip(items, base) if qty > 0]

    def solve(self) -> ColumnGenerationResult:
        self._build_initial_columns()
        base_initial_column_count = len(self._columns)
        seed_stats = self._seed_restricted_master_columns()
        if self.config.full_column_pool:
            before_full_pool = len(self._columns)
            self._expand_all_candidate_columns()
            full_pool_added_columns = len(self._columns) - before_full_pool
        else:
            full_pool_added_columns = 0
        diagnostics: dict = {
            "algorithm": "small_plan_first_column_generation",
            "target_voyages": self.problem.target_voyages,
            "user_area_constraints": getattr(self.problem, "user_area_constraint_summary", {}),
            "attribute_rules": self.attribute_rules.as_dict() if hasattr(self.attribute_rules, "as_dict") else {},
            "small_doc_group_count": int(self.demand_stats.get("source_doc_group_count", 0)),
            "small_doc_box_count": int(self.demand_stats.get("source_doc_box_count", 0)),
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "berth_distance_count": len(self.problem.berth_distances),
            "berth_by_voyage": self.problem.berth_by_voyage,
            "demand_mode": self.config.demand_mode,
            "demand_alignment": self.demand_stats,
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
            "initial_column_count": len(self._columns),
            "base_initial_column_count": base_initial_column_count,
            "full_column_pool": bool(self.config.full_column_pool),
            "full_pool_added_columns": full_pool_added_columns,
            **seed_stats,
            "pricing_iterations": [],
            "gurobi_available": False,
            "used_greedy_fallback": False,
            "concentration_penalties": {
                "fine_group_area": self.config.small_plan_group_area_split_penalty,
                "fine_group_block": self.config.small_plan_group_block_split_penalty,
                "fine_group_bay": self.config.small_plan_group_bay_split_penalty,
                "coarse_area_block": self.config.small_plan_coarse_area_block_split_penalty,
                "coarse_area_bay": self.config.small_plan_coarse_area_bay_split_penalty,
                "medium_concentrated_group_threshold": self.config.medium_concentrated_group_threshold,
                "medium_small_group_area_split": self.config.medium_small_group_area_split_penalty,
                "medium_small_group_fragment": self.config.medium_small_group_fragment_penalty,
                "medium_large_group_min_area_boxes": self.config.medium_large_group_min_area_boxes,
                "medium_large_group_small_area": self.config.medium_large_group_small_area_penalty,
            },
            "inheritance_penalties": {
                "big_plan_area_deviation": self.config.big_plan_area_deviation_penalty,
                "big_plan_fallback_tier": self.config.big_plan_fallback_tier_penalty,
            },
        }

        selected: Counter[int]
        unplaced: Counter[str]
        try:
            if not self.config.use_gurobi:
                raise RuntimeError("Gurobi disabled by config")
            selected, unplaced, master_stats = self._solve_by_column_generation()
            diagnostics.update(master_stats)
        except Exception as exc:
            diagnostics["used_greedy_fallback"] = True
            diagnostics["gurobi_failure"] = f"{type(exc).__name__}: {exc}"
            selected, unplaced = self._greedy_fallback()

        selected, unplaced, repair_stats = self._repair_or_replace_unplaced_solution(selected, unplaced)
        diagnostics.update(repair_stats)

        if self._uses_original_output_scope():
            small_rows = self._make_small_rows(selected, allowed_sources={"document"})
        else:
            small_rows = self._make_small_rows(selected)
        medium_rows = self._make_medium_rows_from_selected_columns(selected, plan_level="medium")
        consistency_stats = self._small_medium_consistency_stats(small_rows, medium_rows)
        bay_consistency_stats = self._small_medium_bay_consistency_stats(small_rows, medium_rows)
        diagnostics.update(
            {
                "final_column_count": len(self._columns),
                "selected_column_count": sum(1 for qty in selected.values() if qty > 0),
                "medium_plan_granularity": "bay",
                "medium_plan_source": "selected_columns",
                "small_row_count": len(small_rows),
                "medium_row_count": len(medium_rows),
                "planned_small_boxes": sum(int(row["planned_boxes"]) for row in small_rows),
                "planned_medium_boxes": sum(int(row["planned_boxes"]) for row in medium_rows),
                "planned_medium_by_source": self._planned_medium_by_source(medium_rows),
                "medium_area_rows_below_min_boxes": self._count_medium_area_rows_below_min(medium_rows),
                "medium_big_plan_inheritance": self._medium_big_plan_inheritance_stats(medium_rows),
                "final_medium_inheritance_energy_components": self._medium_inheritance_energy_components(medium_rows),
                "user_required_area_usage": self._user_required_area_usage(medium_rows, small_rows),
                "user_area_constraint_violations": self._user_area_constraint_violations(medium_rows, small_rows),
                "unplaced_boxes": sum(unplaced.values()),
                "unplaced_by_group": {key: qty for key, qty in sorted(unplaced.items()) if qty > 0},
                "unplaced_group_details": self._unplaced_group_details(unplaced),
                **consistency_stats,
                **bay_consistency_stats,
            }
        )
        return ColumnGenerationResult(medium_rows=medium_rows, small_rows=small_rows, diagnostics=diagnostics, columns=self._columns)

    def _repair_or_replace_unplaced_solution(
        self,
        selected: Counter[int],
        unplaced: Counter[str],
    ) -> tuple[Counter[int], Counter[str], dict]:
        initial_unplaced = sum(unplaced.values())
        stats: dict = {
            "pre_repair_unplaced_boxes": initial_unplaced,
            "post_repair_unplaced_boxes": initial_unplaced,
            "used_unplaced_repair": False,
            "unplaced_repair_method": "",
        }
        if initial_unplaced <= 0:
            return selected, unplaced, stats

        staged_selected, staged_unplaced, stage_stats = self._staged_repair_selected_solution(selected)
        candidates = [
            ("master_incumbent", selected, unplaced),
            ("staged_fallback_repair", staged_selected, staged_unplaced),
        ]
        method, best_selected, best_unplaced = min(
            candidates,
            key=lambda item: self._solution_rank(item[1], item[2]),
        )
        stats.update(
            {
                "used_unplaced_repair": method != "master_incumbent",
                "unplaced_repair_method": method,
                "staged_repair_unplaced_boxes": sum(staged_unplaced.values()),
                "staged_repair_iterations": stage_stats,
                "post_repair_unplaced_boxes": sum(best_unplaced.values()),
            }
        )
        return best_selected, best_unplaced, stats

    def _solution_rank(self, selected: Counter[int], unplaced: Counter[str]) -> tuple[int, float, int]:
        return (
            sum(unplaced.values()),
            self._selected_solution_energy(selected, unplaced),
            sum(1 for qty in selected.values() if qty > 0),
        )

    def _selected_solution_energy(self, selected: Counter[int], unplaced: Counter[str]) -> float:
        energy = float(self.config.unplaced_penalty) * sum(max(0, int(qty)) for qty in unplaced.values())
        actual_quota: Counter[tuple[str, str, str, str]] = Counter()
        actual_coarse_area: Counter[tuple[str, str, str, str, str]] = Counter()
        used_group_area: set[tuple[str, str]] = set()
        used_group_block: set[tuple[str, str]] = set()
        used_coarse_area_block: set[tuple[str, str, str, str, str, str]] = set()
        used_coarse_area_bay: set[tuple[str, str, str, str, str, str]] = set()
        used_voyage_area: set[tuple[str, str]] = set()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            multiplier = int(chosen)
            energy += (col.intrinsic_cost + self.config.small_plan_group_bay_split_penalty) * multiplier
            qty = col.quantity * multiplier
            actual_quota[col.quota_key] += qty
            actual_coarse_area[col.coarse_key + (col.area_no,)] += qty
            used_group_area.add((col.group_id, col.area_no))
            if col.block_id:
                used_group_block.add((col.group_id, col.block_id))
                used_coarse_area_block.add(col.coarse_key + (col.area_no, col.block_id))
            used_coarse_area_bay.add(col.coarse_key + (col.area_no, col.bay_key))
            used_voyage_area.add((col.voyage_id, col.area_no))

        energy += self.config.small_plan_group_area_split_penalty * len(used_group_area)
        energy += self.config.small_plan_group_block_split_penalty * len(used_group_block)
        energy += self.config.small_plan_coarse_area_block_split_penalty * len(used_coarse_area_block)
        energy += self.config.small_plan_coarse_area_bay_split_penalty * len(used_coarse_area_bay)
        energy += sum(self._voyage_area_cost(voyage_id, area_no) for voyage_id, area_no in used_voyage_area)

        target_keys = set(actual_quota)
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                target_keys.add(key)
        for voyage_id, flow, area_no, big_size in target_keys:
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            energy += self.config.big_plan_area_deviation_penalty * abs(actual_quota.get((voyage_id, flow, area_no, big_size), 0) - target)

        by_coarse: defaultdict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        for key, qty in actual_coarse_area.items():
            by_coarse[key[:4]].append(float(qty))
        for coarse_key, quantities in by_coarse.items():
            demand = max(1, int(self.coarse_demand.get(coarse_key, sum(quantities))))
            if self._prefers_concentrated_coarse_key(coarse_key):
                if quantities:
                    energy += self.config.medium_small_group_area_split_penalty * max(0, len(quantities) - 1)
                    energy -= self.config.medium_small_group_fragment_penalty * max(quantities)
            else:
                min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
                if min_boxes > 0:
                    small_area_penalty = self.config.medium_large_group_small_area_penalty / max(1.0, min_boxes)
                    energy += sum(small_area_penalty * max(0.0, min_boxes - qty) for qty in quantities if qty > 0)
                if len(quantities) > 1:
                    pair_penalty = self.config.group_area_balance_penalty / max(1.0, demand) / max(1, len(quantities) - 1)
                    for left_index, left in enumerate(quantities):
                        for right in quantities[left_index + 1 :]:
                            energy += pair_penalty * abs(left - right)
        return float(energy)

    def _repair_from_column_priority(self, column_values: dict[int, float]) -> tuple[Counter[int], Counter[str]]:
        selected: Counter[int] = Counter()
        state = self._empty_selection_state()
        placed: Counter[str] = Counter()
        priority = sorted(
            ((float(value), idx) for idx, value in column_values.items() if value > 1e-6),
            reverse=True,
        )
        for _value, idx in priority:
            if idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            remaining = int(self.group_demand.get(col.group_id, 0)) - int(placed.get(col.group_id, 0))
            if remaining <= 0 or col.quantity > remaining:
                continue
            if not self._column_fits_state(col, state, remaining):
                continue
            self._apply_column_to_state(col, state)
            selected[idx] = 1
            placed[col.group_id] += col.quantity
        return self._repair_selected_solution(selected)

    def _seed_restricted_master_columns(self) -> dict[str, int]:
        before = len(self._columns)
        selected, unplaced = self._repair_selected_solution(Counter())
        self._master_seed_selected = selected
        self._master_seed_unplaced = unplaced
        self._master_start_selected = selected
        self._master_start_unplaced = unplaced
        return {
            "master_seed_added_columns": len(self._columns) - before,
            "master_seed_selected_columns": sum(1 for qty in selected.values() if qty > 0),
            "master_seed_unplaced_boxes": sum(unplaced.values()),
        }

    def _uses_original_output_scope(self) -> bool:
        return (self.config.demand_mode or "original").strip().lower().replace("_", "-") == "original"

    def _count_medium_area_rows_below_min(self, medium_rows: list[dict]) -> int:
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        if min_boxes <= 1:
            return 0
        area_counter: Counter[tuple[str, str, str, str, str]] = Counter()
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
            area_counter[key] += qty
        return sum(
            1
            for qty in area_counter.values()
            if 0 < qty < min_boxes
        )

    @staticmethod
    def _small_medium_consistency_stats(small_rows: list[dict], medium_rows: list[dict]) -> dict[str, int]:
        small_counter: Counter[tuple[str, str, str, str, str]] = Counter()
        medium_counter: Counter[tuple[str, str, str, str, str]] = Counter()
        for row in small_rows:
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            small_counter[key] += int(row.get("planned_boxes", 0) or 0)
        for row in medium_rows:
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                str(row.get("area_no", "")),
            )
            medium_counter[key] += int(row.get("planned_boxes", 0) or 0)
        violations = 0
        shortage = 0
        for key, qty in small_counter.items():
            excess = qty - medium_counter.get(key, 0)
            if excess > 0:
                violations += 1
                shortage += excess
        return {
            "small_medium_consistency_violations": violations,
            "small_medium_consistency_shortage_boxes": shortage,
        }

    @staticmethod
    def _small_medium_bay_consistency_stats(small_rows: list[dict], medium_rows: list[dict]) -> dict[str, int]:
        small_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
        medium_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
        for row in small_rows:
            area_no = str(row.get("area_no", ""))
            bay_key = str(row.get("bay_key") or f"{area_no}-{row.get('bay_no', '')}" if row.get("bay_no") else "")
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                area_no,
                bay_key,
            )
            small_counter[key] += int(row.get("planned_boxes", 0) or 0)
        for row in medium_rows:
            area_no = str(row.get("area_no", ""))
            bay_key = str(row.get("bay_key") or f"{area_no}-{row.get('bay_no', '')}" if row.get("bay_no") else "")
            key = (
                str(row.get("voyage_id", "")),
                str(row.get("flow", "")),
                str(row.get("port", "")),
                str(row.get("size", "")),
                area_no,
                bay_key,
            )
            medium_counter[key] += int(row.get("planned_boxes", 0) or 0)
        violations = 0
        shortage = 0
        for key, qty in small_counter.items():
            excess = qty - medium_counter.get(key, 0)
            if excess > 0:
                violations += 1
                shortage += excess
        return {
            "small_medium_bay_consistency_violations": violations,
            "small_medium_bay_consistency_shortage_boxes": shortage,
        }

    def _solve_by_column_generation(self) -> tuple[Counter[int], Counter[str], dict]:
        import gurobipy as gp
        from gurobipy import GRB

        stats = {"gurobi_available": True, "pricing_iterations": []}
        final_lp_bound = None
        best_start_source = "greedy_seed"
        best_start_selected = Counter(self._master_seed_selected)
        best_start_unplaced = Counter(self._master_seed_unplaced)
        pricing_iterations = 0 if self.config.full_column_pool else self.config.max_iterations
        for iteration in range(pricing_iterations):
            lp_model, lp_vars, lp_constraints = self._build_restricted_master(gp, GRB, relax=True)
            lp_model.optimize()
            if lp_model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
                raise RuntimeError(f"restricted master LP status={lp_model.Status}")
            lp_unplaced = self._gurobi_unplaced_values(lp_vars)
            lp_column_values = self._gurobi_column_values(lp_vars)
            lp_repair_selected, lp_repair_unplaced = self._repair_from_column_priority(
                lp_column_values
            )
            if self._solution_rank(lp_repair_selected, lp_repair_unplaced) < self._solution_rank(
                best_start_selected,
                best_start_unplaced,
            ):
                best_start_source = f"lp_guided_iter_{iteration}"
                best_start_selected = lp_repair_selected
                best_start_unplaced = lp_repair_unplaced
            pricing_stats = self._price_columns(lp_constraints, iteration, lp_unplaced, lp_column_values)
            new_count = int(pricing_stats.get("new_columns", 0) or 0)
            stats["pricing_iterations"].append(
                {
                    "iteration": iteration,
                    "columns": len(self._columns),
                    "lp_objective": float(lp_model.ObjVal),
                    "lp_unplaced_boxes": sum(lp_unplaced.values()),
                    "lp_guided_repair_unplaced_boxes": sum(lp_repair_unplaced.values()),
                    "lp_guided_repair_columns": sum(1 for qty in lp_repair_selected.values() if qty > 0),
                    **pricing_stats,
                }
            )
            if self.config.verbose:
                print(
                    f"[column-generation] iter={iteration} lp={lp_model.ObjVal:.3f} "
                    f"columns={len(self._columns)} new={new_count}",
                    flush=True,
                )
            if new_count == 0:
                break

        final_lp_model, _final_lp_vars, _final_lp_constraints = self._build_restricted_master(gp, GRB, relax=True)
        final_lp_model.optimize()
        if final_lp_model.Status in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
            final_lp_bound = float(final_lp_model.ObjVal)

        self._master_start_selected = best_start_selected
        self._master_start_unplaced = best_start_unplaced
        stats.update(
            {
                "master_mip_start_source": best_start_source,
                "master_mip_start_repaired_columns": sum(1 for qty in best_start_selected.values() if qty > 0),
                "master_mip_start_repaired_unplaced_boxes": sum(best_start_unplaced.values()),
                "restricted_master_lp_bound": final_lp_bound,
            }
        )

        mip_model, mip_vars, _mip_constraints = self._build_restricted_master(gp, GRB, relax=False)
        mip_model.Params.TimeLimit = self.config.mip_time_limit
        mip_model.Params.MIPGap = self.config.mip_gap
        self._apply_gurobi_mip_start(mip_vars)
        mip_model.optimize()
        if mip_model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT} or mip_model.SolCount <= 0:
            raise RuntimeError(f"integer master status={mip_model.Status}, solutions={mip_model.SolCount}")

        selected: Counter[int] = Counter()
        for idx, var in mip_vars["column"].items():
            if var.X > 0.5:
                selected[idx] = 1
        unplaced = Counter(
            {
                group_id: int(round(var.X))
                for group_id, var in mip_vars["unplaced"].items()
                if var.X > 1e-6
            }
        )
        stats.update(
            {
                "master_status": int(mip_model.Status),
                "master_objective": float(mip_model.ObjVal),
                "master_primal_bound": float(getattr(mip_model, "ObjVal", float("nan"))),
                "master_dual_bound": float(getattr(mip_model, "ObjBound", float("nan"))),
                "master_mip_gap": float(getattr(mip_model, "MIPGap", 0.0)),
                "master_mip_gap_is_reliable": float(getattr(mip_model, "MIPGap", 0.0)) < 1e10,
                "restricted_master_lp_gap": self._relative_gap(float(mip_model.ObjVal), final_lp_bound),
                "master_mip_start_columns": sum(1 for qty in self._master_start_selected.values() if qty > 0),
                "master_mip_start_unplaced_boxes": sum(self._master_start_unplaced.values()),
            }
        )
        return selected, unplaced, stats

    @staticmethod
    def _gurobi_unplaced_values(lp_vars) -> Counter[str]:
        return Counter(
            {
                group_id: int(round(var.X))
                for group_id, var in lp_vars["unplaced"].items()
                if var.X > 1e-6
            }
        )

    @staticmethod
    def _relative_gap(primal: float | None, bound: float | None) -> float | None:
        if primal is None or bound is None:
            return None
        try:
            primal_value = float(primal)
            bound_value = float(bound)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(primal_value) or not math.isfinite(bound_value):
            return None
        return max(0.0, (primal_value - bound_value) / max(1.0, abs(primal_value)))

    @staticmethod
    def _gurobi_column_values(lp_vars) -> dict[int, float]:
        return {
            idx: float(var.X)
            for idx, var in lp_vars["column"].items()
            if var.X > 1e-6
        }

    def _apply_gurobi_mip_start(self, mip_vars) -> None:
        if not self._master_start_selected and not self._master_start_unplaced:
            return
        for idx, var in mip_vars["column"].items():
            var.Start = 1.0 if self._master_start_selected.get(idx, 0) > 0 else 0.0
        for group_id, var in mip_vars["unplaced"].items():
            var.Start = float(self._master_start_unplaced.get(group_id, 0))

    def _build_restricted_master(self, gp, GRB, relax: bool):
        model = gp.Model("yard_small_plan_column_generation")
        model.Params.OutputFlag = 1 if self.config.verbose else 0
        column_vtype = GRB.CONTINUOUS if relax else GRB.BINARY
        columns = {
            idx: model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=column_vtype,
                obj=col.intrinsic_cost + self.config.small_plan_group_bay_split_penalty,
                name=f"col_{idx}",
            )
            for idx, col in enumerate(self._columns)
        }
        unplaced = {
            group.group_id: model.addVar(
                lb=0.0,
                ub=group.demand,
                vtype=GRB.CONTINUOUS if relax else GRB.INTEGER,
                obj=self.config.unplaced_penalty,
                name=f"unplaced_{group.group_id}",
            )
            for group in self.groups
        }

        group_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_capacity_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_size_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_port_size_cols: defaultdict[tuple[str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        group_bay_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        bay_attr_choice_cols: defaultdict[tuple[str, str, str], list[int]] = defaultdict(list)
        coarse_area_cols: defaultdict[tuple[str, str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        area_size_cols: defaultdict[tuple[str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        group_area_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        group_block_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        coarse_area_block_cols: defaultdict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(list)
        coarse_area_bay_cols: defaultdict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(list)
        voyage_area_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        edge_45_cols: defaultdict[str, list[int]] = defaultdict(list)
        edge_non45_cols: defaultdict[str, list[int]] = defaultdict(list)
        for idx, col in enumerate(self._columns):
            group_cols[col.group_id].append((idx, col))
            for footprint_key in self._placement_footprint_keys(col.bay_key, col.size):
                bay_capacity_cols[footprint_key].append((idx, col))
                bay_port_size_cols[(footprint_key, self._row_mix_key_for_column(col), col.size)].append((idx, col))
                for attr in self._bay_no_mix_attrs():
                    bay_attr_choice_cols[(footprint_key, attr, self._column_attr_value(col, attr))].append(idx)
            bay_size_capacity_cols[(col.bay_key, col.size)].append((idx, col))
            group_bay_cols[(col.group_id, col.bay_key)].append(idx)
            coarse_area_cols[col.coarse_key + (col.area_no,)].append((idx, col))
            area_size_cols[col.quota_key].append((idx, col))
            group_area_cols[(col.group_id, col.area_no)].append(idx)
            if col.block_id:
                group_block_cols[(col.group_id, col.block_id)].append(idx)
                coarse_area_block_cols[col.coarse_key + (col.area_no, col.block_id)].append(idx)
            coarse_area_bay_cols[col.coarse_key + (col.area_no, col.bay_key)].append(idx)
            voyage_area_cols[(col.voyage_id, col.area_no)].append(idx)
            if col.bay_key in self.area_edge_bays.get(col.area_no, set()):
                if col.size == "45":
                    edge_45_cols[col.area_no].append(idx)
                else:
                    edge_non45_cols[col.area_no].append(idx)

        group_cover = {}
        for group in self.groups:
            expr = gp.quicksum(col.quantity * columns[idx] for idx, col in group_cols.get(group.group_id, []))
            group_cover[group.group_id] = model.addConstr(expr + unplaced[group.group_id] == group.demand, name=f"cover_{group.group_id}")

        required_area_limit = {}
        for voyage_id, areas in sorted(getattr(self.problem, "user_voyage_area_requirements", {}).items()):
            for area_no in sorted(areas):
                indices = voyage_area_cols.get((voyage_id, area_no), [])
                if not indices:
                    continue
                required_area_limit[(voyage_id, area_no)] = model.addConstr(
                    gp.quicksum(self._columns[idx].quantity * columns[idx] for idx in indices) >= 1.0,
                    name=f"user_required_area_{len(required_area_limit)}",
                )

        bay_capacity_limit = {}
        for bay_key, items in bay_capacity_cols.items():
            bay_capacity_limit[bay_key] = model.addConstr(
                gp.quicksum(col.quantity * columns[idx] for idx, col in items) <= self.bays[bay_key].physical_capacity,
                name=f"bay_cap_{bay_key}",
            )
        bay_size_limit = {}
        for key, items in bay_size_capacity_cols.items():
            bay_key, size = key
            bay_size_limit[key] = model.addConstr(
                gp.quicksum(col.quantity * columns[idx] for idx, col in items) <= self.bays[bay_key].cap_by_size.get(size, 0),
                name=f"bay_size_{bay_key}_{size}",
            )
        bay_port_stack_link = {}
        bay_port_stack_limit = {}
        bay_stack_total_limit = {}
        bay_stack_vars = {}
        stack_vtype = GRB.CONTINUOUS if relax else GRB.INTEGER
        for key, items in sorted(bay_port_size_cols.items()):
            bay_key, port, size = key
            sample_group = self.groups_by_id.get(items[0][1].group_id) if items else None
            if sample_group is None:
                continue
            stack_count = self._stack_count_for_group(bay_key, size, sample_group)
            unit_capacity = self._stack_unit_capacity_for_group(bay_key, size, sample_group)
            if stack_count <= 0 or unit_capacity <= 0:
                continue
            stack_var = model.addVar(lb=0.0, ub=stack_count, vtype=stack_vtype, name=f"stack_{bay_key}_{port}_{size}")
            bay_stack_vars[key] = stack_var
            load = gp.quicksum(col.quantity * columns[idx] for idx, col in items)
            bay_port_stack_link[key] = model.addConstr(load <= unit_capacity * stack_var, name=f"stack_load_{bay_key}_{port}_{size}")
            bay_port_stack_limit[key] = model.addConstr(stack_var <= stack_count, name=f"stack_port_cap_{bay_key}_{port}_{size}")
        stack_vars_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for (bay_key, _port, size), stack_var in bay_stack_vars.items():
            stack_vars_by_bay_size[(bay_key, size)].append(stack_var)
        for (bay_key, size), stack_vars in stack_vars_by_bay_size.items():
            stack_count = self._stack_count_for_bay_size(bay_key, size)
            if stack_count > 0:
                bay_stack_total_limit[(bay_key, size)] = model.addConstr(
                    gp.quicksum(stack_vars) <= stack_count,
                    name=f"stack_total_{bay_key}_{size}",
                )
        group_bay_limit = {
            key: model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= 1.0, name=f"group_bay_{key[0]}_{key[1]}")
            for key, indices in group_bay_cols.items()
        }
        quota_limit = {}
        medium_plan_quota_limit = {}
        for key, items in area_size_cols.items():
            cap = int(self.quota_by_key.get(key, 0))
            if cap <= 0:
                continue
            quota_limit[key] = model.addConstr(
                gp.quicksum(col.quantity * columns[idx] for idx, col in items) <= cap,
                name=f"big_quota_{len(quota_limit)}",
            )
        if self.config.medium_plan_quota is not None:
            medium_plan_quota = Counter(self.config.medium_plan_quota)
            for key, items in coarse_area_cols.items():
                cap = int(medium_plan_quota.get(key, 0))
                medium_plan_quota_limit[key] = model.addConstr(
                    gp.quicksum(col.quantity * columns[idx] for idx, col in items) <= cap,
                    name=f"medium_quota_{len(medium_plan_quota_limit)}",
                )
        if self.config.medium_plan_bay_quota is not None:
            medium_plan_bay_quota = Counter(self.config.medium_plan_bay_quota)
            for key, indices in coarse_area_bay_cols.items():
                cap = int(medium_plan_bay_quota.get(key, 0))
                medium_plan_quota_limit[key] = model.addConstr(
                    gp.quicksum(self._columns[idx].quantity * columns[idx] for idx in indices) <= cap,
                    name=f"medium_bay_quota_{len(medium_plan_quota_limit)}",
                )

        seed_unplaced_limit = None
        if not relax and (self._master_seed_selected or self._master_seed_unplaced):
            seed_unplaced_limit = model.addConstr(
                gp.quicksum(unplaced.values()) <= sum(self._master_seed_unplaced.values()),
                name="seed_unplaced_cap",
            )

        relaxed_objective_constraints = {}
        if relax:
            relaxed_objective_constraints = self._add_relaxed_master_objectives(
                gp,
                model,
                columns,
                area_size_cols,
                group_area_cols,
                group_block_cols,
                coarse_area_block_cols,
                coarse_area_bay_cols,
                voyage_area_cols,
            )
        else:
            self._add_integer_master_objectives(
                gp,
                GRB,
                model,
                columns,
                coarse_area_cols,
                area_size_cols,
                group_area_cols,
                group_block_cols,
                coarse_area_block_cols,
                coarse_area_bay_cols,
                voyage_area_cols,
                edge_45_cols,
                edge_non45_cols,
                bay_attr_choice_cols,
            )
        model.update()
        return model, {"column": columns, "unplaced": unplaced}, {
            "group_cover": group_cover,
            "bay_capacity_limit": bay_capacity_limit,
            "bay_size_limit": bay_size_limit,
            "bay_port_stack_link": bay_port_stack_link,
            "bay_port_stack_limit": bay_port_stack_limit,
            "bay_stack_total_limit": bay_stack_total_limit,
            "group_bay_limit": group_bay_limit,
            "quota_limit": quota_limit,
            "medium_plan_quota_limit": medium_plan_quota_limit,
            "required_area_limit": required_area_limit,
            "seed_unplaced_limit": seed_unplaced_limit,
            **relaxed_objective_constraints,
        }

    def _add_relaxed_master_objectives(
        self,
        gp,
        model,
        columns,
        area_size_cols,
        group_area_cols,
        group_block_cols,
        coarse_area_block_cols,
        coarse_area_bay_cols,
        voyage_area_cols,
    ) -> dict[str, dict]:
        area_size_keys = set(area_size_cols)
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                area_size_keys.add(key)

        big_plan_deviation_balance = {}
        for key in sorted(area_size_keys):
            items = area_size_cols.get(key, [])
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            pos = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty)
            neg = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty)
            actual = gp.quicksum(col.quantity * columns[idx] for idx, col in items)
            big_plan_deviation_balance[key] = model.addConstr(actual - target == pos - neg)

        fixed_use_constraints = {}
        for key, indices in group_area_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_group_area_split_penalty)
            fixed_use_constraints[("group_area",) + key] = model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in group_block_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_group_block_split_penalty)
            fixed_use_constraints[("group_block",) + key] = model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_block_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_coarse_area_block_split_penalty)
            fixed_use_constraints[("coarse_area_block",) + key] = model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_bay_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_coarse_area_bay_split_penalty)
            fixed_use_constraints[("coarse_area_bay",) + key] = model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for (voyage_id, area_no), indices in voyage_area_cols.items():
            cost = self._voyage_area_cost(voyage_id, area_no)
            if abs(cost) <= 1e-9:
                continue
            use = model.addVar(lb=0.0, ub=1.0, obj=cost)
            total = gp.quicksum(columns[idx] for idx in indices)
            fixed_use_constraints[("voyage_area", voyage_id, area_no)] = model.addConstr(total <= len(indices) * use)
            if cost < 0:
                fixed_use_constraints[("voyage_area_reward", voyage_id, area_no)] = model.addConstr(use <= total)
        return {
            "big_plan_deviation_balance": big_plan_deviation_balance,
            "fixed_use_objective_limit": fixed_use_constraints,
        }

    def _add_integer_master_objectives(
        self,
        gp,
        GRB,
        model,
        columns,
        coarse_area_cols,
        area_size_cols,
        group_area_cols,
        group_block_cols,
        coarse_area_block_cols,
        coarse_area_bay_cols,
        voyage_area_cols,
        edge_45_cols,
        edge_non45_cols,
        bay_attr_choice_cols,
    ) -> None:
        coarse_area_keys = set(coarse_area_cols)
        for coarse_key in self.coarse_demand:
            voyage_id, flow, _port, size = coarse_key
            big_size = self._big_plan_size(size)
            for (v, f, area_no, s), qty in self.quota_by_key.items():
                if v == voyage_id and f == flow and s == big_size and qty > 0:
                    coarse_area_keys.add(coarse_key + (area_no,))
        self._add_coarse_group_area_objectives(gp, GRB, model, columns, coarse_area_keys, coarse_area_cols)

        area_size_keys = set(area_size_cols)
        for key, qty in self.quota_by_key.items():
            voyage_id, flow, _area_no, big_size = key
            if qty > 0 and self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0:
                area_size_keys.add(key)
        for key in sorted(area_size_keys):
            items = area_size_cols.get(key, [])
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            pos = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty, name=f"big_pos_{len(model.getVars())}")
            neg = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty, name=f"big_neg_{len(model.getVars())}")
            actual = gp.quicksum(col.quantity * columns[idx] for idx, col in items)
            model.addConstr(actual - target == pos - neg)

        for (group_id, area_no), indices in group_area_cols.items():
            use = model.addVar(vtype=GRB.BINARY, obj=self.config.small_plan_group_area_split_penalty, name=f"use_ga_{group_id}_{area_no}")
            model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for (group_id, block_id), indices in group_block_cols.items():
            use = model.addVar(vtype=GRB.BINARY, obj=self.config.small_plan_group_block_split_penalty, name=f"use_gb_{group_id}_{block_id}")
            model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_block_cols.items():
            voyage_id, flow, port, size, area_no, block_id = key
            use = model.addVar(
                vtype=GRB.BINARY,
                obj=self.config.small_plan_coarse_area_block_split_penalty,
                name=f"use_cab_{voyage_id}_{flow}_{size}_{area_no}_{block_id}",
            )
            model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_bay_cols.items():
            voyage_id, flow, port, size, area_no, bay_key = key
            use = model.addVar(
                vtype=GRB.BINARY,
                obj=self.config.small_plan_coarse_area_bay_split_penalty,
                name=f"use_cay_{voyage_id}_{flow}_{size}_{area_no}_{bay_key}",
            )
            model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)

        for (voyage_id, area_no), indices in voyage_area_cols.items():
            cost = self._voyage_area_cost(voyage_id, area_no)
            use = model.addVar(vtype=GRB.BINARY, obj=cost, name=f"use_va_{voyage_id}_{area_no}")
            total = gp.quicksum(columns[idx] for idx in indices)
            model.addConstr(total <= len(indices) * use)
            if cost < 0:
                model.addConstr(use <= total)

        for area_no in set(edge_45_cols) | set(edge_non45_cols):
            has45 = model.addVar(vtype=GRB.BINARY, name=f"area_has45_{area_no}")
            if edge_45_cols.get(area_no):
                model.addConstr(gp.quicksum(columns[idx] for idx in edge_45_cols[area_no]) <= len(edge_45_cols[area_no]) * has45)
            if edge_non45_cols.get(area_no):
                model.addConstr(gp.quicksum(columns[idx] for idx in edge_non45_cols[area_no]) <= len(edge_non45_cols[area_no]) * (1 - has45))

        self._add_bay_compatibility_constraints(
            gp,
            GRB,
            model,
            columns,
            bay_attr_choice_cols,
        )

    def _add_coarse_group_area_objectives(
        self,
        gp,
        GRB,
        model,
        columns,
        coarse_area_keys: set[tuple[str, str, str, str, str]],
        coarse_area_cols: dict[tuple[str, str, str, str, str], list[tuple[int, PlacementColumn]]],
    ) -> None:
        area_keys_by_coarse: defaultdict[tuple[str, str, str, str], list[tuple[str, str, str, str, str]]] = defaultdict(list)
        for key in coarse_area_keys:
            voyage_id, flow, port, size, _area_no = key
            coarse_key = (voyage_id, flow, port, size)
            area_keys_by_coarse[coarse_key].append(key)

        for coarse_key, area_keys in sorted(area_keys_by_coarse.items()):
            demand = self.coarse_demand[coarse_key]
            if demand <= 0:
                continue
            if self._prefers_concentrated_coarse_key(coarse_key):
                self._add_concentrated_coarse_group_objective(
                    gp,
                    GRB,
                    model,
                    columns,
                    coarse_key,
                    area_keys,
                    coarse_area_cols,
                    demand,
                )
            else:
                self._add_large_coarse_group_balance_objective(
                    gp,
                    GRB,
                    model,
                    columns,
                    coarse_key,
                    area_keys,
                    coarse_area_cols,
                    demand,
                )

    def _add_concentrated_coarse_group_objective(
        self,
        gp,
        GRB,
        model,
        columns,
        coarse_key: tuple[str, str, str, str],
        area_keys: list[tuple[str, str, str, str, str]],
        coarse_area_cols: dict[tuple[str, str, str, str, str], list[tuple[int, PlacementColumn]]],
        demand: int,
    ) -> None:
        largest = model.addVar(
            lb=0.0,
            ub=float(demand),
            obj=-self.config.medium_small_group_fragment_penalty,
            name=f"coarse_largest_{'_'.join(coarse_key)}",
        )
        primary_vars = []
        for key in sorted(area_keys):
            *_, area_no = key
            items = coarse_area_cols.get(key, [])
            actual = gp.quicksum(col.quantity * columns[idx] for idx, col in items)
            use = model.addVar(
                vtype=GRB.BINARY,
                obj=self.config.medium_small_group_area_split_penalty,
                name=f"use_conc_area_{'_'.join(coarse_key)}_{area_no}",
            )
            primary = model.addVar(
                vtype=GRB.BINARY,
                obj=-self.config.medium_small_group_area_split_penalty,
                name=f"primary_conc_area_{'_'.join(coarse_key)}_{area_no}",
            )
            model.addConstr(actual <= demand * use)
            model.addConstr(primary <= use)
            model.addConstr(largest <= actual + demand * (1 - primary))
            primary_vars.append(primary)
        if primary_vars:
            primary_sum = gp.quicksum(primary_vars)
            model.addConstr(primary_sum <= 1)
            model.addConstr(largest <= demand * primary_sum)

    def _add_large_coarse_group_balance_objective(
        self,
        gp,
        GRB,
        model,
        columns,
        coarse_key: tuple[str, str, str, str],
        area_keys: list[tuple[str, str, str, str, str]],
        coarse_area_cols: dict[tuple[str, str, str, str, str], list[tuple[int, PlacementColumn]]],
        demand: int,
    ) -> None:
        area_terms = []
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        small_area_penalty = self.config.medium_large_group_small_area_penalty / max(1.0, min_boxes)
        for key in sorted(area_keys):
            *_, area_no = key
            items = coarse_area_cols.get(key, [])
            actual = gp.quicksum(col.quantity * columns[idx] for idx, col in items)
            use = model.addVar(vtype=GRB.BINARY, name=f"use_bal_area_{'_'.join(coarse_key)}_{area_no}")
            model.addConstr(actual <= demand * use)
            model.addConstr(actual >= use)
            if min_boxes > 0:
                shortage = model.addVar(
                    lb=0.0,
                    obj=small_area_penalty,
                    name=f"small_bal_area_{'_'.join(coarse_key)}_{area_no}",
                )
                model.addConstr(shortage >= min_boxes * use - actual)
            area_terms.append((area_no, actual, use))

        if len(area_terms) <= 1:
            return
        pair_penalty = self.config.group_area_balance_penalty / max(1.0, demand) / max(1, len(area_terms) - 1)
        for left_index, (_left_area, left_actual, left_use) in enumerate(area_terms):
            for _right_area, right_actual, right_use in area_terms[left_index + 1 :]:
                diff = model.addVar(lb=0.0, obj=pair_penalty, name=f"bal_diff_{len(model.getVars())}")
                inactive = demand * (2 - left_use - right_use)
                model.addConstr(diff >= left_actual - right_actual - inactive)
                model.addConstr(diff >= right_actual - left_actual - inactive)

    def _add_bay_compatibility_constraints(
        self,
        gp,
        GRB,
        model,
        columns,
        bay_attr_choice_cols: dict[tuple[str, str, str], list[int]],
    ) -> None:
        use_by_bay_attr: defaultdict[tuple[str, str], list] = defaultdict(list)
        for (bay_key, attr, value), indices in sorted(bay_attr_choice_cols.items()):
            use = model.addVar(vtype=GRB.BINARY, name=f"bay_use_{attr}_{bay_key}_{value}")
            model.addConstr(gp.quicksum(columns[idx] for idx in indices) <= len(indices) * use)
            use_by_bay_attr[(bay_key, attr)].append(use)
        for (bay_key, attr), uses in use_by_bay_attr.items():
            model.addConstr(gp.quicksum(uses) <= 1, name=f"bay_one_{attr}_{bay_key}")

    def _price_columns(
        self,
        lp_constraints: dict,
        iteration: int,
        lp_unplaced: Counter[str] | None = None,
        lp_column_values: dict[int, float] | None = None,
    ) -> dict:
        group_dual = {group_id: constr.Pi for group_id, constr in lp_constraints["group_cover"].items()}
        group_dual = self._effective_group_duals(group_dual, lp_unplaced or Counter())
        bay_capacity_dual = {bay_key: constr.Pi for bay_key, constr in lp_constraints["bay_capacity_limit"].items()}
        bay_size_dual = {key: constr.Pi for key, constr in lp_constraints["bay_size_limit"].items()}
        bay_port_stack_dual = {
            key: constr.Pi for key, constr in lp_constraints.get("bay_port_stack_link", {}).items()
        }
        group_bay_dual = {key: constr.Pi for key, constr in lp_constraints["group_bay_limit"].items()}
        quota_dual = {key: constr.Pi for key, constr in lp_constraints.get("quota_limit", {}).items()}
        medium_plan_quota_dual = {
            key: constr.Pi for key, constr in lp_constraints.get("medium_plan_quota_limit", {}).items()
        }
        big_plan_deviation_dual = {
            key: constr.Pi for key, constr in lp_constraints.get("big_plan_deviation_balance", {}).items()
        }
        fixed_use_dual = {
            key: constr.Pi for key, constr in lp_constraints.get("fixed_use_objective_limit", {}).items()
        }
        lp_column_values = lp_column_values or {}
        lp_quota_actual = self._column_values_quota_actual(lp_column_values)
        lp_coarse_area_actual = self._column_values_coarse_area_actual(lp_column_values)
        candidates: list[tuple[float, SmallBoxGroup, str, int, float]] = []
        negative_candidates: list[tuple[float, SmallBoxGroup, str, int, float]] = []
        primal_candidates: list[tuple[tuple, float, SmallBoxGroup, str, int, float]] = []
        scanned = 0
        for group in self.groups:
            for bay_key, max_qty, base_cost in self._candidate_bays_for_group(group):
                for qty in self._quantity_options(group, max_qty):
                    key = (group.group_id, bay_key, qty)
                    if key in self._column_keys:
                        continue
                    scanned += 1
                    area_no = self.bays[bay_key].area_no
                    coarse_area_key = self._coarse_key(group) + (area_no,)
                    block_id = self.block_by_bay.get((area_no, bay_key), "")
                    coarse_key = self._coarse_key(group)
                    group_area_key = ("group_area", group.group_id, area_no)
                    group_block_key = ("group_block", group.group_id, block_id)
                    coarse_area_bay_key = ("coarse_area_bay",) + coarse_key + (area_no, bay_key)
                    coarse_area_block_key = ("coarse_area_block",) + coarse_key + (area_no, block_id)
                    voyage_area_key = ("voyage_area", group.voyage_id, area_no)
                    reduced = (
                        base_cost
                        + self.config.small_plan_group_bay_split_penalty
                        - group_dual.get(group.group_id, 0.0) * qty
                        - bay_capacity_dual.get(bay_key, 0.0) * qty
                        - bay_size_dual.get((bay_key, group.size), 0.0) * qty
                        - sum(
                            bay_port_stack_dual.get((footprint_key, self._row_mix_key_for_group(group), group.size), 0.0) * qty
                            for footprint_key in self._placement_footprint_keys(bay_key, group.size)
                        )
                        - group_bay_dual.get((group.group_id, bay_key), 0.0)
                        - quota_dual.get(self._quota_key(group, area_no), 0.0) * qty
                        - medium_plan_quota_dual.get(coarse_area_key, 0.0) * qty
                        - medium_plan_quota_dual.get(coarse_area_key + (bay_key,), 0.0) * qty
                        - big_plan_deviation_dual.get(self._quota_key(group, area_no), 0.0) * qty
                        + self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            group_area_key,
                            self.config.small_plan_group_area_split_penalty,
                        )
                        + self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            coarse_area_bay_key,
                            self.config.small_plan_coarse_area_bay_split_penalty,
                        )
                        + self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            voyage_area_key,
                            self._voyage_area_cost(group.voyage_id, area_no),
                        )
                    )
                    if block_id:
                        reduced += self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            group_block_key,
                            self.config.small_plan_group_block_split_penalty,
                        )
                        reduced += self._fixed_use_reduced_adjustment(
                            fixed_use_dual,
                            coarse_area_block_key,
                            self.config.small_plan_coarse_area_block_split_penalty,
                        )
                    candidate = (reduced, group, bay_key, qty, base_cost)
                    candidates.append(candidate)
                    primal_candidates.append(
                        (
                            self._primal_expansion_score(
                                group,
                                bay_key,
                                qty,
                                base_cost,
                                lp_quota_actual,
                                lp_coarse_area_actual,
                            ),
                            reduced,
                            group,
                            bay_key,
                            qty,
                            base_cost,
                        )
                    )
                    if reduced < -1e-6:
                        negative_candidates.append(candidate)
        candidates.sort(key=lambda item: item[0])
        negative_candidates.sort(key=lambda item: item[0])
        if negative_candidates:
            selected_candidates = negative_candidates[: self.config.columns_per_iteration]
            mode = "negative_reduced_cost"
        elif iteration == 0:
            limit = max(0, int(getattr(self.config, "stalled_pricing_columns", 0) or 0))
            selected_candidates = candidates[:limit]
            mode = "stalled_best_reduced_cost"
        elif self._primal_expansion_rounds < max(0, int(self.config.max_primal_expansion_rounds or 0)):
            limit = max(0, int(getattr(self.config, "primal_expansion_columns", 0) or 0))
            reduced_limit = float(getattr(self.config, "primal_expansion_reduced_cost_limit", 0.0) or 0.0)
            if reduced_limit > 0:
                primal_candidates = [
                    item for item in primal_candidates
                    if item[1] <= reduced_limit
                ]
            primal_candidates.sort(key=lambda item: item[0])
            selected_candidates = [
                (reduced, group, bay_key, qty, base_cost)
                for _score, reduced, group, bay_key, qty, base_cost in primal_candidates[:limit]
            ]
            if selected_candidates:
                self._primal_expansion_rounds += 1
                mode = "integer_primal_expansion"
            else:
                mode = "no_primal_candidate_under_limit"
        else:
            selected_candidates = []
            mode = "no_improving_column"
        added = 0
        for _reduced, group, bay_key, qty, base_cost in selected_candidates:
            if self._add_column(group, bay_key, qty, base_cost):
                added += 1
        return {
            "new_columns": added,
            "pricing_mode": mode,
            "scanned_candidates": scanned,
            "negative_reduced_candidates": len(negative_candidates),
            "primal_expansion_rounds_used": self._primal_expansion_rounds,
            "primal_expansion_reduced_cost_limit": self.config.primal_expansion_reduced_cost_limit,
            "best_reduced_cost": round(candidates[0][0], 6) if candidates else None,
            "worst_added_reduced_cost": round(selected_candidates[-1][0], 6) if selected_candidates else None,
        }

    @staticmethod
    def _fixed_use_reduced_adjustment(fixed_use_dual: dict, key: tuple, fixed_cost: float) -> float:
        if key in fixed_use_dual:
            return -float(fixed_use_dual.get(key, 0.0))
        return float(fixed_cost)

    def _column_values_quota_actual(self, column_values: dict[int, float]) -> Counter[tuple[str, str, str, str]]:
        actual: Counter[tuple[str, str, str, str]] = Counter()
        for idx, value in column_values.items():
            if value <= 1e-9 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            actual[col.quota_key] += col.quantity * float(value)
        return actual

    def _column_values_coarse_area_actual(self, column_values: dict[int, float]) -> Counter[tuple[str, str, str, str, str]]:
        actual: Counter[tuple[str, str, str, str, str]] = Counter()
        for idx, value in column_values.items():
            if value <= 1e-9 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            actual[col.coarse_key + (col.area_no,)] += col.quantity * float(value)
        return actual

    def _primal_expansion_score(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        qty: int,
        base_cost: float,
        lp_quota_actual: Counter[tuple[str, str, str, str]],
        lp_coarse_area_actual: Counter[tuple[str, str, str, str, str]],
    ) -> tuple:
        bay = self.bays[bay_key]
        area_no = bay.area_no
        quota_key = self._quota_key(group, area_no)
        target = self._area_size_target(group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
        quota_gap = target - lp_quota_actual.get(quota_key, 0.0)
        coarse_area_key = self._coarse_key(group) + (area_no,)
        existing_area_qty = lp_coarse_area_actual.get(coarse_area_key, 0.0)
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        if self._prefers_concentrated_coarse_key(self._coarse_key(group)):
            shape_score = (0 if existing_area_qty > 1e-6 else 1, -existing_area_qty)
        else:
            after_qty = existing_area_qty + qty
            shortage_after = max(0.0, float(min_boxes) - after_qty) if after_qty > 1e-6 else 0.0
            shape_score = (0 if existing_area_qty > 1e-6 else 1, shortage_after)
        return (
            0 if quota_gap > 1e-6 else 1,
            -min(float(qty), max(0.0, quota_gap)),
            shape_score,
            self._voyage_area_cost(group.voyage_id, area_no),
            base_cost,
            -qty,
            bay.bay_order,
        )

    def _effective_group_duals(self, group_dual: dict[str, float], lp_unplaced: Counter[str]) -> dict[str, float]:
        out = dict(group_dual)
        threshold = float(self.config.unplaced_penalty) * 0.1
        for group_id, qty in lp_unplaced.items():
            if qty > 0 and out.get(group_id, 0.0) < threshold:
                out[group_id] = float(self.config.unplaced_penalty)
        return out

    def _build_initial_columns(self) -> None:
        for group in self.groups:
            candidates = self._candidate_bays_for_group(group)
            for bay_key, max_qty, base_cost in candidates[: self.config.initial_columns_per_group]:
                for qty in self._quantity_options(group, max_qty):
                    self._add_column(group, bay_key, qty, base_cost)

    def _expand_all_candidate_columns(self) -> None:
        for group in self.groups:
            for bay_key, max_qty, base_cost in self._candidate_bays_for_group(group):
                for qty in self._quantity_options(group, max_qty):
                    self._add_column(group, bay_key, qty, base_cost)

    def _bay_no_mix_attrs(self) -> tuple[str, ...]:
        attrs = getattr(self.attribute_rules, "bay_no_mix_attributes", ("size", "height"))
        return tuple(str(attr) for attr in attrs if str(attr))

    def _row_no_mix_attrs(self) -> tuple[str, ...]:
        attrs = getattr(self.attribute_rules, "row_no_mix_attributes", ("port",))
        return tuple(str(attr) for attr in attrs if str(attr))

    @staticmethod
    def _group_attr_value(group: SmallBoxGroup, attr: str) -> str:
        if attr == "flow":
            attr = "status"
        value = getattr(group, attr, "")
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    @staticmethod
    def _column_attr_value(col: PlacementColumn, attr: str) -> str:
        if attr == "flow":
            attr = "status"
        if attr == "status":
            value = col.flow
        else:
            value = getattr(col, attr, "")
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def _row_mix_key_for_group(self, group: SmallBoxGroup) -> str:
        return "|".join(f"{attr}={self._group_attr_value(group, attr)}" for attr in self._row_no_mix_attrs()) or "__all__"

    def _row_mix_key_for_column(self, col: PlacementColumn) -> str:
        return "|".join(f"{attr}={self._column_attr_value(col, attr)}" for attr in self._row_no_mix_attrs()) or "__all__"

    def _row_existing_attrs_allow_group(self, bay: Bay, row_no: str, group: SmallBoxGroup) -> bool:
        row_attrs = getattr(bay, "existing_attrs_by_row", {}).get(str(row_no), {})
        for attr in self._row_no_mix_attrs():
            values = set(row_attrs.get(attr, set()))
            if values and self._group_attr_value(group, attr) not in values:
                return False
        return True

    def _bay_existing_attrs_allow_group(self, group: SmallBoxGroup, footprint: tuple[str, ...]) -> bool:
        for key in footprint:
            bay = self.bays[key]
            existing_attrs = getattr(bay, "existing_attrs", {})
            for attr in self._bay_no_mix_attrs():
                values = set(existing_attrs.get(attr, set()))
                if values and values != {self._group_attr_value(group, attr)}:
                    return False
        return True

    def _bay_state_attrs_allow_group(self, group: SmallBoxGroup, footprint: tuple[str, ...], state: dict) -> bool:
        used_attrs = state.setdefault("bay_used_attrs", {})
        for key in footprint:
            for attr in self._bay_no_mix_attrs():
                state_key = (key, attr)
                value = self._group_attr_value(group, attr)
                if used_attrs.get(state_key, value) != value:
                    return False
        return True

    def _row_stack_capacities_for_group(self, bay_key: str, size: str, group: SmallBoxGroup) -> list[int]:
        return [cap for _row_no, cap in self._row_stack_capacity_items_for_group(bay_key, size, group)]

    def _row_stack_capacity_items_for_group(self, bay_key: str, size: str, group: SmallBoxGroup) -> list[tuple[str, int]]:
        bay = self.bays[bay_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or {}
        if not row_caps and bay.row_physical_capacity:
            row_caps = bay.row_physical_capacity
        has_row_caps = bool(row_caps)
        capacities: list[tuple[str, int]] = []
        for row_no, cap in row_caps.items():
            if not self._row_existing_attrs_allow_group(bay, str(row_no), group):
                continue
            cap_int = int(cap)
            if cap_int > 0:
                capacities.append((str(row_no), cap_int))
        if has_row_caps:
            return capacities
        fallback = int(bay.cap_by_size.get(size, 0) or bay.physical_capacity)
        return [("__bay__", fallback)] if fallback > 0 else []

    def _stack_count_for_group(self, bay_key: str, size: str, group: SmallBoxGroup) -> int:
        return len(self._row_stack_capacities_for_group(bay_key, size, group))

    def _stack_count_for_bay_size(self, bay_key: str, size: str) -> int:
        bay = self.bays[bay_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or bay.row_physical_capacity
        if row_caps:
            return sum(1 for cap in row_caps.values() if int(cap) > 0)
        return 1 if int(bay.cap_by_size.get(size, 0) or bay.physical_capacity) > 0 else 0

    def _stack_unit_capacity_for_group(self, bay_key: str, size: str, group: SmallBoxGroup) -> int:
        capacities = self._row_stack_capacities_for_group(bay_key, size, group)
        return max(capacities) if capacities else 0

    def _stack_units_for_quantity(self, bay_key: str, size: str, group: SmallBoxGroup, quantity: int) -> int:
        if quantity <= 0:
            return 0
        unit_capacity = self._stack_unit_capacity_for_group(bay_key, size, group)
        if unit_capacity <= 0:
            return 10**9
        return int(math.ceil(quantity / unit_capacity))

    def _column_stack_units(self, group: SmallBoxGroup, bay_key: str, quantity: int) -> int:
        units = [
            self._stack_units_for_quantity(key, group.size, group, quantity)
            for key in self._placement_footprint_keys(bay_key, group.size)
        ]
        return max(units) if units else 10**9

    def _remaining_stack_capacity_for_group_bay(self, group: SmallBoxGroup, bay_key: str, state: dict) -> int:
        capacity = 10**9
        row_mix_key = self._row_mix_key_for_group(group)
        for footprint_key in self._placement_footprint_keys(bay_key, group.size):
            unit_capacity = self._stack_unit_capacity_for_group(footprint_key, group.size, group)
            port_stack_count = self._stack_count_for_group(footprint_key, group.size, group)
            total_stack_count = self._stack_count_for_bay_size(footprint_key, group.size)
            if unit_capacity <= 0 or port_stack_count <= 0 or total_stack_count <= 0:
                return 0
            port_key = (footprint_key, row_mix_key, group.size)
            total_key = (footprint_key, group.size)
            current_load = state["bay_port_size_load"][port_key]
            current_units = self._stack_units_for_quantity(footprint_key, group.size, group, current_load)
            other_units = state["bay_stack_used"][total_key] - current_units
            capacity = min(capacity, port_stack_count * unit_capacity - current_load)
            capacity = min(capacity, max(0, total_stack_count - other_units) * unit_capacity - current_load)
        return max(0, int(capacity))

    def _apply_stack_usage_to_state(self, group: SmallBoxGroup, bay_key: str, quantity: int, state: dict) -> None:
        row_mix_key = self._row_mix_key_for_group(group)
        for footprint_key in self._placement_footprint_keys(bay_key, group.size):
            port_key = (footprint_key, row_mix_key, group.size)
            total_key = (footprint_key, group.size)
            before = state["bay_port_size_load"][port_key]
            before_units = self._stack_units_for_quantity(footprint_key, group.size, group, before)
            after = before + quantity
            after_units = self._stack_units_for_quantity(footprint_key, group.size, group, after)
            state["bay_port_size_load"][port_key] = after
            state["bay_stack_used"][total_key] += after_units - before_units

    def _add_column(self, group: SmallBoxGroup, bay_key: str, quantity: int, base_cost: float) -> bool:
        if quantity <= 0:
            return False
        stack_units = self._column_stack_units(group, bay_key, quantity)
        if stack_units >= 10**9:
            return False
        key = (group.group_id, bay_key, quantity)
        if key in self._column_keys:
            return False
        bay = self.bays[bay_key]
        block_id = self.block_by_bay.get((bay.area_no, bay_key), "")
        column_id = f"C{len(self._columns) + 1:07d}"
        col = PlacementColumn(
            column_id=column_id,
            group_id=group.group_id,
            demand_source=self.group_source.get(group.group_id, "document"),
            voyage_id=group.voyage_id,
            flow=group.status,
            port=group.port,
            size=group.size,
            big_plan_size=self._big_plan_size(group.size),
            height=group.height,
            weight_class=group.weight_class,
            special_stow_code=group.special_stow_code,
            area_no=bay.area_no,
            bay_key=bay_key,
            bay_no=bay.bay_no,
            block_id=block_id,
            block_bays=self.block_bay_nos.get(block_id, ()),
            quantity=quantity,
            stack_units=stack_units,
            quota_key=self._quota_key(group, bay.area_no),
            coarse_key=self._coarse_key(group),
            intrinsic_cost=base_cost,
        )
        self._columns.append(col)
        self._column_keys.add(key)
        return True

    def _greedy_fallback(self) -> tuple[Counter[int], Counter[str]]:
        selected, unplaced, _stats = self._staged_repair_selected_solution(Counter())
        return selected, unplaced

    def _repair_selected_solution(self, selected: Counter[int]) -> tuple[Counter[int], Counter[str]]:
        repaired, state, placed = self._selection_state(selected)
        unplaced: Counter[str] = Counter()
        for group in self.groups:
            remaining = int(group.demand) - int(placed.get(group.group_id, 0))
            while remaining > 0:
                choice = self._best_repair_column(group, state, remaining, repaired)
                if choice is None:
                    break
                idx, col = choice
                self._apply_column_to_state(col, state)
                repaired[idx] = 1
                placed[group.group_id] += col.quantity
                remaining -= col.quantity
            if remaining > 0:
                unplaced[group.group_id] = remaining
        return repaired, unplaced

    def _staged_repair_selected_solution(
        self,
        selected: Counter[int],
    ) -> tuple[Counter[int], Counter[str], list[dict]]:
        repaired, state, placed = self._selection_state(selected)
        stages = [
            ("stage1a", True),
            ("stage1b", False),
            ("stage2", False),
            ("stage3", False),
        ]
        stats: list[dict] = []
        for stage, enforce_quota in stages:
            before_unplaced = sum(max(0, int(group.demand) - int(placed.get(group.group_id, 0))) for group in self.groups)
            if before_unplaced <= 0:
                break
            before_columns = len(self._columns)
            before_selected = sum(1 for qty in repaired.values() if qty > 0)
            for group in self.groups:
                remaining = int(group.demand) - int(placed.get(group.group_id, 0))
                while remaining > 0:
                    choice = self._best_repair_column(
                        group,
                        state,
                        remaining,
                        repaired,
                        stage=stage,
                        enforce_quota=enforce_quota,
                    )
                    if choice is None:
                        break
                    idx, col = choice
                    self._apply_column_to_state(col, state)
                    repaired[idx] = 1
                    placed[group.group_id] += col.quantity
                    remaining -= col.quantity
            after_unplaced = sum(max(0, int(group.demand) - int(placed.get(group.group_id, 0))) for group in self.groups)
            stats.append(
                {
                    "stage": stage,
                    "enforce_big_plan_quota": enforce_quota,
                    "before_unplaced_boxes": before_unplaced,
                    "after_unplaced_boxes": after_unplaced,
                    "placed_boxes": before_unplaced - after_unplaced,
                    "added_columns": len(self._columns) - before_columns,
                    "added_selected_columns": sum(1 for qty in repaired.values() if qty > 0) - before_selected,
                }
            )
        unplaced = Counter(
            {
                group.group_id: int(group.demand) - int(placed.get(group.group_id, 0))
                for group in self.groups
                if int(group.demand) - int(placed.get(group.group_id, 0)) > 0
            }
        )
        return repaired, unplaced, stats

    def _selection_state(self, selected: Counter[int]) -> tuple[Counter[int], dict, Counter[str]]:
        repaired: Counter[int] = Counter()
        placed: Counter[str] = Counter()
        state = self._empty_selection_state()
        for idx, chosen in sorted(selected.items()):
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            group = self.groups_by_id.get(col.group_id)
            if group is None:
                continue
            remaining = int(group.demand) - int(placed.get(group.group_id, 0))
            if remaining <= 0 or col.quantity > remaining:
                continue
            if not self._column_fits_state(col, state, remaining):
                continue
            self._apply_column_to_state(col, state)
            repaired[idx] = 1
            placed[group.group_id] += col.quantity
        return repaired, state, placed

    @staticmethod
    def _empty_selection_state() -> dict:
        return {
            "bay_load": Counter(),
            "bay_size_load": Counter(),
            "bay_port_size_load": Counter(),
            "bay_stack_used": Counter(),
            "bay_used_size": {},
            "bay_used_height": {},
            "bay_used_attrs": {},
            "group_bay_used": set(),
            "area_edge_has45": set(),
            "area_edge_has_non45": set(),
            "big_plan_quota_used": Counter(),
            "medium_plan_quota_used": Counter(),
            "medium_plan_bay_quota_used": Counter(),
        }

    def _ensure_repair_columns(self, group: SmallBoxGroup, state: dict, remaining: int) -> None:
        for bay_key, _max_qty, base_cost in self._candidate_bays_for_group(group):
            capacity = self._remaining_capacity_for_group_bay(group, bay_key, state, remaining)
            if capacity <= 0:
                continue
            self._add_column(group, bay_key, min(remaining, capacity), base_cost)

    def _best_repair_column(
        self,
        group: SmallBoxGroup,
        state: dict,
        remaining: int,
        selected: Counter[int],
        stage: str | None = None,
        enforce_quota: bool = True,
    ) -> tuple[int, PlacementColumn] | None:
        best: tuple[tuple[float, int, int, str], str, int, float] | None = None
        for bay_key, _max_qty, base_cost in self._candidate_bays_for_group(group, scope=stage):
            capacity = self._remaining_capacity_for_group_bay(
                group,
                bay_key,
                state,
                remaining,
                enforce_quota=enforce_quota,
            )
            if capacity <= 0:
                continue
            qty = min(remaining, capacity)
            bay = self.bays[bay_key]
            score = (
                base_cost
                + self._voyage_area_cost(group.voyage_id, bay.area_no),
                0 if qty >= remaining else 1,
                -qty,
                bay_key,
            )
            candidate = (score, bay_key, qty, base_cost)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            return None
        _score, bay_key, qty, base_cost = best
        idx = self._column_index_for(group.group_id, bay_key, qty)
        if idx is None:
            self._add_column(group, bay_key, qty, base_cost)
            idx = len(self._columns) - 1
        if selected.get(idx, 0) > 0:
            return None
        col = self._columns[idx]
        if not self._column_fits_state(col, state, remaining, enforce_quota=enforce_quota):
            return None
        return idx, col

    def _column_index_for(self, group_id: str, bay_key: str, quantity: int) -> int | None:
        for idx, col in enumerate(self._columns):
            if col.group_id == group_id and col.bay_key == bay_key and col.quantity == quantity:
                return idx
        return None

    def _column_fits_state(
        self,
        col: PlacementColumn,
        state: dict,
        remaining: int,
        enforce_quota: bool = True,
    ) -> bool:
        group = self.groups_by_id.get(col.group_id)
        if group is None or col.quantity <= 0 or col.quantity > remaining:
            return False
        return col.quantity <= self._remaining_capacity_for_group_bay(
            group,
            col.bay_key,
            state,
            remaining,
            enforce_quota=enforce_quota,
        )

    def _remaining_capacity_for_group_bay(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        state: dict,
        remaining: int,
        enforce_quota: bool = True,
    ) -> int:
        if (group.group_id, bay_key) in state["group_bay_used"]:
            return 0
        bay = self.bays[bay_key]
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        if not self._bay_state_attrs_allow_group(group, footprint, state):
            return 0
        is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
        if is_edge and group.size == "45" and bay.area_no in state["area_edge_has_non45"]:
            return 0
        if is_edge and group.size != "45" and bay.area_no in state["area_edge_has45"]:
            return 0
        capacity = int(remaining)
        for key in footprint:
            capacity = min(capacity, self.bays[key].physical_capacity - state["bay_load"][key])
        capacity = min(capacity, bay.cap_by_size.get(group.size, 0) - state["bay_size_load"][(bay_key, group.size)])
        capacity = min(capacity, self._remaining_stack_capacity_for_group_bay(group, bay_key, state))
        quota_key = self._quota_key(group, bay.area_no)
        quota = self.quota_by_key.get(quota_key, 0)
        if enforce_quota and quota > 0:
            capacity = min(capacity, quota - state["big_plan_quota_used"][quota_key])
        if self.config.medium_plan_quota is not None:
            medium_plan_quota = Counter(self.config.medium_plan_quota)
            coarse_area_key = self._coarse_key(group) + (bay.area_no,)
            capacity = min(capacity, medium_plan_quota[coarse_area_key] - state["medium_plan_quota_used"][coarse_area_key])
        if self.config.medium_plan_bay_quota is not None:
            medium_plan_bay_quota = Counter(self.config.medium_plan_bay_quota)
            coarse_bay_key = self._coarse_key(group) + (bay.area_no, bay_key)
            capacity = min(capacity, medium_plan_bay_quota[coarse_bay_key] - state["medium_plan_bay_quota_used"][coarse_bay_key])
        return max(0, int(capacity))

    def _apply_column_to_state(self, col: PlacementColumn, state: dict) -> None:
        footprint = self._placement_footprint_keys(col.bay_key, col.size)
        for key in footprint:
            state["bay_load"][key] += col.quantity
            state["bay_used_size"][key] = col.size
            state["bay_used_height"][key] = col.height
            for attr in self._bay_no_mix_attrs():
                state.setdefault("bay_used_attrs", {})[(key, attr)] = self._column_attr_value(col, attr)
        state["bay_size_load"][(col.bay_key, col.size)] += col.quantity
        group = self.groups_by_id.get(col.group_id)
        if group is not None:
            self._apply_stack_usage_to_state(group, col.bay_key, col.quantity, state)
        state["group_bay_used"].add((col.group_id, col.bay_key))
        if col.bay_key in self.area_edge_bays.get(col.area_no, set()):
            if col.size == "45":
                state["area_edge_has45"].add(col.area_no)
            else:
                state["area_edge_has_non45"].add(col.area_no)
        state["big_plan_quota_used"][col.quota_key] += col.quantity
        state["medium_plan_quota_used"][col.coarse_key + (col.area_no,)] += col.quantity
        state["medium_plan_bay_quota_used"][col.coarse_key + (col.area_no, col.bay_key)] += col.quantity

    def _legacy_greedy_fallback(self) -> tuple[Counter[int], Counter[str]]:
        selected: Counter[int] = Counter()
        unplaced: Counter[str] = Counter()
        bay_load: Counter[str] = Counter()
        bay_size_load: Counter[tuple[str, str]] = Counter()
        bay_used_size: dict[str, str] = {}
        bay_used_height: dict[str, str] = {}
        bay_used_attrs: dict[tuple[str, str], str] = {}
        group_bay_used: set[tuple[str, str]] = set()
        area_edge_has45: set[str] = set()
        area_edge_has_non45: set[str] = set()
        medium_plan_quota = Counter(self.config.medium_plan_quota or {})
        medium_plan_bay_quota = Counter(self.config.medium_plan_bay_quota or {})
        big_plan_quota_used: Counter[tuple[str, str, str, str]] = Counter()
        medium_plan_quota_used: Counter[tuple[str, str, str, str, str]] = Counter()
        medium_plan_bay_quota_used: Counter[tuple[str, str, str, str, str, str]] = Counter()
        columns_by_group: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        for idx, col in enumerate(self._columns):
            columns_by_group[col.group_id].append((idx, col))
        for group in self.groups:
            remaining = group.demand
            for idx, col in sorted(columns_by_group[group.group_id], key=lambda item: (item[1].intrinsic_cost, -item[1].quantity)):
                if remaining <= 0:
                    break
                bay = self.bays[col.bay_key]
                if (col.group_id, col.bay_key) in group_bay_used:
                    continue
                footprint = self._placement_footprint_keys(col.bay_key, col.size)
                if not footprint:
                    continue
                if any(
                    bay_used_attrs.get((key, attr), self._column_attr_value(col, attr)) != self._column_attr_value(col, attr)
                    for key in footprint
                    for attr in self._bay_no_mix_attrs()
                ):
                    continue
                if any(bay_load[key] + col.quantity > self.bays[key].physical_capacity for key in footprint):
                    continue
                if bay_size_load[(col.bay_key, col.size)] + col.quantity > bay.cap_by_size.get(col.size, 0):
                    continue
                quota = self.quota_by_key.get(col.quota_key, 0)
                if quota > 0 and big_plan_quota_used[col.quota_key] + col.quantity > quota:
                    continue
                coarse_area_key = col.coarse_key + (col.area_no,)
                if self.config.medium_plan_quota is not None and (
                    medium_plan_quota_used[coarse_area_key] + col.quantity > medium_plan_quota[coarse_area_key]
                ):
                    continue
                coarse_bay_key = col.coarse_key + (col.area_no, col.bay_key)
                if self.config.medium_plan_bay_quota is not None and (
                    medium_plan_bay_quota_used[coarse_bay_key] + col.quantity > medium_plan_bay_quota[coarse_bay_key]
                ):
                    continue
                if col.quantity > remaining:
                    continue
                is_edge = col.bay_key in self.area_edge_bays.get(col.area_no, set())
                if is_edge and col.size == "45" and col.area_no in area_edge_has_non45:
                    continue
                if is_edge and col.size != "45" and col.area_no in area_edge_has45:
                    continue
                selected[idx] = 1
                for key in footprint:
                    bay_load[key] += col.quantity
                    bay_used_size[key] = col.size
                    bay_used_height[key] = col.height
                    for attr in self._bay_no_mix_attrs():
                        bay_used_attrs[(key, attr)] = self._column_attr_value(col, attr)
                bay_size_load[(col.bay_key, col.size)] += col.quantity
                group_bay_used.add((col.group_id, col.bay_key))
                if is_edge and col.size == "45":
                    area_edge_has45.add(col.area_no)
                elif is_edge:
                    area_edge_has_non45.add(col.area_no)
                big_plan_quota_used[col.quota_key] += col.quantity
                medium_plan_quota_used[coarse_area_key] += col.quantity
                medium_plan_bay_quota_used[coarse_bay_key] += col.quantity
                remaining -= col.quantity
            if remaining > 0:
                unplaced[group.group_id] = remaining
        return selected, unplaced

    def _candidate_bays_for_group(self, group: SmallBoxGroup, scope: str | None = None) -> list[tuple[str, int, float]]:
        scope = scope or self._candidate_scope
        cache_key = (scope, group.group_id)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        out: list[tuple[str, int, float]] = []
        for area_no in self._candidate_areas_for_group(group, scope=scope):
            for bay_key in self.bays_by_area.get(area_no, []):
                max_qty = self._max_quantity_in_bay(group, bay_key)
                if max_qty <= 0:
                    continue
                cost = self._column_base_cost(group, bay_key)
                out.append((bay_key, min(max_qty, group.demand), cost))
        if self._prefers_concentrated_coarse_key(self._coarse_key(group)):
            out.sort(
                key=lambda item: (
                    self._area_fallback_tier_for_group(group, self.bays[item[0]].area_no),
                    self._concentrated_area_sort_key(group, self.bays[item[0]].area_no),
                    item[2],
                    -item[1],
                    self.bays[item[0]].bay_order,
                )
            )
        else:
            out.sort(
                key=lambda item: (
                    self._area_fallback_tier_for_group(group, self.bays[item[0]].area_no),
                    item[2],
                    -item[1],
                    self.bays[item[0]].area_no,
                    self.bays[item[0]].bay_order,
                )
            )
        out = self._limit_candidate_bays(group, out)
        self._candidate_cache[cache_key] = out
        return out

    def _candidate_areas_for_group(self, group: SmallBoxGroup, scope: str | None = None) -> list[str]:
        scope = scope or self._candidate_scope
        big_size = self._big_plan_size(group.size)

        def in_scope(area_no: str) -> bool:
            if scope in {"stage0", "stage1a"}:
                return self.quota_by_key.get((group.voyage_id, group.status, area_no, big_size), 0) > 0
            if scope == "stage1b":
                return self._is_big_plan_area_for_group(group, area_no)
            if scope == "stage2":
                return self._is_any_big_plan_area(area_no)
            return True

        return sorted(
            [
                area_no
                for area_no in self.bays_by_area
                if in_scope(area_no)
                and self._user_area_policy_allows(group.voyage_id, area_no)
                and (
                    self._area_supports_group_flow(group, area_no)
                    or self._user_area_policy_forces_support(group.voyage_id, area_no)
                )
            ],
            key=lambda area_no: (self._area_fallback_tier_for_group(group, area_no), area_no),
        )

    def _user_area_policy_allows(self, voyage_id: str, area_no: str) -> bool:
        allow = getattr(self.problem, "user_voyage_area_allowlist", {}).get(voyage_id, set())
        block = getattr(self.problem, "user_voyage_area_blocklist", {}).get(voyage_id, set())
        if area_no in block:
            return False
        if allow and area_no not in allow:
            return False
        return True

    def _user_area_policy_forces_support(self, voyage_id: str, area_no: str) -> bool:
        allow = getattr(self.problem, "user_voyage_area_allowlist", {}).get(voyage_id, set())
        required = getattr(self.problem, "user_voyage_area_requirements", {}).get(voyage_id, set())
        return area_no in allow or area_no in required

    def _area_supports_group_flow(self, group: SmallBoxGroup, area_no: str) -> bool:
        if self._is_big_plan_area_for_group(group, area_no):
            return True
        functions = self.problem.area_functions.get(area_no, set())
        if group.status in EXPORT_FLOWS:
            return "OF" in functions
        return group.status in functions

    def _limit_candidate_bays(
        self,
        group: SmallBoxGroup,
        candidates: list[tuple[str, int, float]],
    ) -> list[tuple[str, int, float]]:
        limit = max(0, int(self.config.max_candidate_bays_per_group or 0))
        if limit <= 0 or len(candidates) <= limit:
            return candidates
        preferred_areas = set(self._area_weights(group))
        if not preferred_areas:
            return candidates[:limit]
        preferred = [item for item in candidates if self.bays[item[0]].area_no in preferred_areas]
        fallback = [item for item in candidates if self.bays[item[0]].area_no not in preferred_areas]
        if not fallback:
            return preferred[:limit]
        fallback_limit = min(len(fallback), max(1, limit // 4))
        preferred_limit = max(0, limit - fallback_limit)
        return preferred[:preferred_limit] + fallback[:fallback_limit]

    def _max_quantity_in_bay(self, group: SmallBoxGroup, bay_key: str) -> int:
        bay = self.bays[bay_key]
        if bay.cap_by_size.get(group.size, 0) <= 0:
            return 0
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return 0
        if not self._bay_existing_attrs_allow_group(group, footprint):
            return 0
        is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
        if group.size == "20" and is_edge:
            return 0
        if group.size == "45" and not is_edge:
            return 0
        footprint_capacity = min(self.bays[key].physical_capacity for key in footprint)
        stack_capacity = min(
            self._stack_count_for_group(key, group.size, group) * self._stack_unit_capacity_for_group(key, group.size, group)
            for key in footprint
        )
        return max(0, min(group.demand, footprint_capacity, bay.cap_by_size.get(group.size, 0), stack_capacity))

    def _placement_footprint_keys(self, bay_key: str, size: str) -> tuple[str, ...]:
        bay = self.bays[bay_key]
        if size in {"40", "45"}:
            if not bay.large_bay_partner_key:
                return ()
            return (bay_key, bay.large_bay_partner_key)
        return (bay_key,)

    def _column_base_cost(self, group: SmallBoxGroup, bay_key: str) -> float:
        bay = self.bays[bay_key]
        cost = 0.0
        if not self.block_by_bay.get((bay.area_no, bay_key)):
            cost += self.config.non_preferred_block_penalty
        if group.port not in bay.existing_ports:
            cost += self.config.port_mismatch_penalty
        if bay.is_fallback_bay:
            cost += self.config.fallback_bay_penalty
        if group.special_stow or group.pre_stow:
            cost -= 1.0
        return cost

    def _area_fallback_tier_penalty(self, group: SmallBoxGroup, area_no: str) -> float:
        tier = self._area_fallback_tier_for_group(group, area_no)
        if tier == 0:
            return 0.0
        return float(self.config.big_plan_fallback_tier_penalty) * tier

    def _is_big_plan_area_for_group(self, group: SmallBoxGroup, area_no: str) -> bool:
        if area_no in self.problem.assigned_areas.get((group.voyage_id, group.status), set()):
            return True
        if group.status in EXPORT_FLOWS:
            return any(area_no in self.problem.assigned_areas.get((group.voyage_id, flow), set()) for flow in EXPORT_FLOWS)
        return False

    def _is_any_big_plan_area(self, area_no: str) -> bool:
        return any(row.area_no == area_no and row.planned_boxes > 0 for row in self.problem.big_plan)

    def _area_fallback_tier_for_group(self, group: SmallBoxGroup, area_no: str) -> int:
        return self._area_fallback_tier_for_attrs(
            group.voyage_id,
            group.status,
            area_no,
            self._big_plan_size(group.size),
        )

    def _area_fallback_tier_for_attrs(self, voyage_id: str, flow: str, area_no: str, big_size: str) -> int:
        if self.quota_by_key.get((voyage_id, flow, area_no, big_size), 0) > 0:
            return 0
        if area_no in self.problem.assigned_areas.get((voyage_id, flow), set()):
            return 1
        if flow in EXPORT_FLOWS and any(
            area_no in self.problem.assigned_areas.get((voyage_id, compatible_flow), set())
            for compatible_flow in EXPORT_FLOWS
        ):
            return 1
        if self._is_any_big_plan_area(area_no):
            return 2
        return 3

    def _quantity_options(self, group: SmallBoxGroup, max_qty: int) -> list[int]:
        values = {1, max_qty, min(group.demand, max_qty)}
        if group.demand % max_qty:
            values.add(group.demand % max_qty)
        values.add(max(1, max_qty // 2))
        return sorted(qty for qty in values if 0 < qty <= max_qty)

    def _prepare_yard_indexes(self) -> None:
        for key, bay in self.bays.items():
            self.bays_by_area[bay.area_no].append(key)
        for keys in self.bays_by_area.values():
            keys.sort(key=lambda bay_key: self.bays[bay_key].bay_order)
        for area_no, keys in self.bays_by_area.items():
            if keys:
                edge_keys = {keys[0], keys[-1]}
                last_key = keys[-1]
                for bay_key in keys:
                    if self.bays[bay_key].large_bay_partner_key == last_key:
                        edge_keys.add(bay_key)
                self.area_edge_bays[area_no] = edge_keys
            blocks = self._six_bay_blocks_for_area(area_no, keys)
            self.block_members_by_area[area_no] = blocks
            for block_id, members in blocks.items():
                self.block_bay_nos[block_id] = tuple(self.bays[bay_key].bay_no for bay_key in members)
                for bay_key in members:
                    self.block_by_bay[(area_no, bay_key)] = block_id
        heights_by_size: defaultdict[str, set[str]] = defaultdict(set)
        for group in self.groups:
            heights_by_size[group.size].add(group.height)
        for bay_key, bay in self.bays.items():
            is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
            for size, heights in heights_by_size.items():
                if size == "20" and is_edge:
                    continue
                if size == "45" and not is_edge:
                    continue
                cap = bay.cap_by_size.get(size, 0)
                if cap <= 0:
                    continue
                footprint = self._placement_footprint_keys(bay_key, size)
                if not footprint:
                    continue
                for height in heights:
                    existing_heights = set().union(*(self.bays[key].existing_heights for key in footprint))
                    if existing_heights and existing_heights != {height}:
                        continue
                    self.area_size_height_cap[(bay.area_no, size, height)] += cap

    def _prepare_quota(self) -> None:
        requested_keys = {
            (group.voyage_id, group.status, self._big_plan_size(group.size))
            for group in self.groups
        }
        for voyage_id, flow, big_size in requested_keys:
            compatible = EXPORT_FLOWS if flow in EXPORT_FLOWS else {flow}
            for row in self.problem.big_plan:
                row_size = self._big_plan_size(row.size_mode)
                if (
                    row.voyage_id == voyage_id
                    and row.flow in compatible
                    and (row_size == big_size or row.size_mode == "ALL")
                ):
                    self.quota_by_key[(voyage_id, flow, row.area_no, big_size)] += row.planned_boxes

    def _area_weights(self, group: SmallBoxGroup) -> Counter[str]:
        weights: Counter[str] = Counter()
        big_size = self._big_plan_size(group.size)
        for (voyage_id, flow, area_no, size), qty in self.quota_by_key.items():
            if voyage_id == group.voyage_id and flow == group.status and size == big_size and qty > 0:
                weights[area_no] += qty
        return weights

    def _quota_key(self, group: SmallBoxGroup, area_no: str) -> tuple[str, str, str, str]:
        return group.voyage_id, group.status, area_no, self._big_plan_size(group.size)

    def _coarse_key(self, group: SmallBoxGroup) -> tuple[str, str, str, str]:
        return group.voyage_id, group.status, group.port, group.size

    def _big_plan_size(self, size: str) -> str:
        return "40" if size == "45" else size if size in {"20", "40"} else "40"

    def _prefers_concentrated_coarse_key(self, coarse_key: tuple[str, str, str, str]) -> bool:
        threshold = int(self.config.medium_concentrated_group_threshold or 0)
        return threshold > 0 and self.coarse_demand[coarse_key] <= threshold

    def _concentrated_area_sort_key(self, group: SmallBoxGroup, area_no: str) -> tuple[int, int, int, int, str]:
        coarse_key = self._coarse_key(group)
        demand = max(int(self.coarse_demand.get(coarse_key, group.demand)), int(group.demand))
        quota = self.quota_by_key.get(self._quota_key(group, area_no), 0)
        height_cap = self.area_size_height_cap.get((area_no, group.size, group.height), 0)
        in_big_plan = self._is_big_plan_area_for_group(group, area_no)
        useful_cap = min(quota, height_cap) if in_big_plan and quota > 0 else height_cap
        return (0 if in_big_plan else 1, 0 if useful_cap >= demand else 1, -useful_cap, -quota, area_no)

    def _coarse_area_target(self, coarse_key: tuple[str, str, str, str], area_no: str) -> float:
        voyage_id, flow, _port, size = coarse_key
        big_size = self._big_plan_size(size)
        total = sum(qty for (v, f, _a, s), qty in self.quota_by_key.items() if v == voyage_id and f == flow and s == big_size)
        if total <= 0:
            return 0.0
        quota = self.quota_by_key.get((voyage_id, flow, area_no, big_size), 0)
        return self.coarse_demand[coarse_key] * quota / total

    def _area_size_target(self, voyage_id: str, flow: str, area_no: str, big_size: str) -> float:
        total = sum(qty for (v, f, _a, s), qty in self.quota_by_key.items() if v == voyage_id and f == flow and s == big_size)
        demand = self.voyage_flow_size_demand[(voyage_id, flow, big_size)]
        if total <= 0:
            return 0.0
        return demand * self.quota_by_key.get((voyage_id, flow, area_no, big_size), 0) / total

    def _voyage_area_cost(self, voyage_id: str, area_no: str) -> float:
        cost = 0.0
        if area_no in getattr(self.problem, "user_voyage_area_requirements", {}).get(voyage_id, set()):
            cost -= max(1000.0, float(self.config.unplaced_penalty) * 0.25)
        berth = self.problem.berth_by_voyage.get(voyage_id, "")
        if berth:
            distance = self.problem.berth_distances.get((area_no, berth))
            if distance is not None:
                cost += self.config.berth_distance_penalty * distance / 100.0
        if self._area_has_loading_during_window(voyage_id, area_no):
            cost += self.config.active_loading_area_penalty
        if self._area_has_loading_after_window(voyage_id, area_no):
            cost -= self.config.post_window_loading_area_reward
        return cost

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

    def _unplaced_group_details(self, unplaced: Counter[str]) -> list[dict]:
        details = []
        for group_id, qty in sorted(unplaced.items()):
            if qty <= 0:
                continue
            group = self.groups_by_id.get(group_id)
            if group is None:
                details.append({"group_id": group_id, "unplaced_boxes": int(qty)})
                continue
            details.append(
                {
                    "group_id": group_id,
                    "demand_source": self.group_source.get(group_id, "document"),
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

    def _area_has_loading_during_window(self, voyage_id: str, area_no: str) -> bool:
        window_start, window_end = self.problem.voyage_windows[voyage_id]
        return any(
            op.voyage_id != voyage_id and _time_windows_overlap(op.start_time, op.end_time, window_start, window_end)
            for op in self.problem.area_operations.get(area_no, [])
        )

    def _area_has_loading_after_window(self, voyage_id: str, area_no: str) -> bool:
        _window_start, window_end = self.problem.voyage_windows[voyage_id]
        prefer_end = window_end + timedelta(hours=24)
        return any(
            op.voyage_id != voyage_id and _time_windows_overlap(op.start_time, op.end_time, window_end, prefer_end)
            for op in self.problem.area_operations.get(area_no, [])
        )

    def _medium_big_plan_inheritance_stats(self, medium_rows: list[dict]) -> dict[str, float | int]:
        actual = self._medium_area_size_counter(medium_rows)
        total = sum(actual.values())
        inherited = sum(min(qty, int(self.quota_by_key.get(key, 0) or 0)) for key, qty in actual.items())
        deviated = max(0, total - inherited)
        return {
            "total_boxes": total,
            "inherited_boxes": inherited,
            "deviated_boxes": deviated,
            "inheritance_ratio": round(inherited / total, 6) if total else 1.0,
            "deviation_ratio": round(deviated / total, 6) if total else 0.0,
        }

    def _medium_inheritance_energy_components(self, medium_rows: list[dict]) -> dict[str, float]:
        actual = self._medium_area_size_counter(medium_rows)
        components = {
            "big_plan_fallback_tier": 0.0,
            "big_plan_deviation": 0.0,
        }
        for (voyage_id, flow, area_no, big_size), qty in actual.items():
            tier = self._area_fallback_tier_for_attrs(voyage_id, flow, area_no, big_size)
            if tier > 0:
                components["big_plan_fallback_tier"] += self.config.big_plan_fallback_tier_penalty * tier * qty

        targets = self._effective_big_plan_area_size_targets()
        for key in set(actual) | set(targets):
            components["big_plan_deviation"] += self.config.big_plan_area_deviation_penalty * abs(
                actual.get(key, 0) - targets.get(key, 0.0)
            )
        components["total"] = sum(components.values())
        return {key: round(value, 4) for key, value in components.items()}

    def _medium_area_size_counter(self, medium_rows: list[dict]) -> Counter[tuple[str, str, str, str]]:
        actual: Counter[tuple[str, str, str, str]] = Counter()
        for row in medium_rows:
            qty = int(row.get("planned_boxes", 0) or 0)
            if qty <= 0:
                continue
            voyage_id = str(row.get("voyage_id", ""))
            flow = str(row.get("flow", "OF") or "OF")
            area_no = str(row.get("area_no", ""))
            big_size = self._big_plan_size(str(row.get("size", "")))
            actual[(voyage_id, flow, area_no, big_size)] += qty
        return actual

    @staticmethod
    def _planned_medium_by_source(medium_rows: list[dict]) -> dict[str, int]:
        out: Counter[str] = Counter()
        for row in medium_rows:
            document_boxes = int(row.get("document_boxes", 0) or 0)
            forecast_boxes = int(row.get("forecast_fallback_boxes", 0) or 0)
            known = document_boxes + forecast_boxes
            if document_boxes > 0:
                out["document"] += document_boxes
            if forecast_boxes > 0:
                out["forecast_fallback"] += forecast_boxes
            remainder = int(row.get("planned_boxes", 0) or 0) - known
            if remainder > 0:
                out["unknown"] += remainder
        return {key: int(value) for key, value in sorted(out.items())}

    def _effective_big_plan_area_size_targets(self) -> dict[tuple[str, str, str, str], float]:
        targets: dict[tuple[str, str, str, str], float] = {}
        keys = {
            (voyage_id, flow, big_size)
            for voyage_id, flow, _area_no, big_size in self.quota_by_key
            if self.voyage_flow_size_demand[(voyage_id, flow, big_size)] > 0
        }
        for voyage_id, flow, big_size in keys:
            total = sum(
                qty
                for (v, f, _area_no, s), qty in self.quota_by_key.items()
                if v == voyage_id and f == flow and s == big_size
            )
            demand = self.voyage_flow_size_demand[(voyage_id, flow, big_size)]
            if total <= 0 or demand <= 0:
                continue
            target_total = min(demand, total)
            for (v, f, area_no, s), quota in self.quota_by_key.items():
                if v == voyage_id and f == flow and s == big_size and quota > 0:
                    targets[(v, f, area_no, s)] = target_total * quota / total
        return targets

    def _make_small_rows(self, selected: Counter[int], allowed_sources: set[str] | None = None) -> list[dict]:
        counter: Counter[tuple] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0:
                continue
            col = self._columns[idx]
            source = self.group_source.get(col.group_id, "document")
            if allowed_sources is not None and source not in allowed_sources:
                continue
            counter[
                (
                    col.voyage_id,
                    col.group_id,
                    col.flow,
                    col.port,
                    col.size,
                    col.height,
                    col.weight_class,
                    col.special_stow_code,
                    col.area_no,
                    col.bay_no,
                    col.block_id,
                    col.block_bays,
                )
            ] += col.quantity * chosen
        block_total: Counter[str] = Counter()
        for key, qty in counter.items():
            block_id = key[10]
            if block_id:
                block_total[block_id] += qty
        rows: list[dict] = []
        for key, qty in sorted(counter.items()):
            voyage_id, group_id, flow, port, size, height, weight_class, special_code, area_no, bay_no, block_id, block_bays = key
            rows.append(
                {
                    "plan_level": "small",
                    "voyage_id": voyage_id,
                    "group_id": group_id,
                    "demand_source": self.group_source.get(group_id, "document"),
                    "flow": flow,
                    "port": port,
                    "size": size,
                    "height": height,
                    "weight_class": weight_class,
                    "special_stow": bool(special_code),
                    "special_stow_code": special_code or "NORMAL",
                    "area_no": area_no,
                    "bay_no": bay_no,
                    "six_bay_block_id": block_id,
                    "six_bay_block_bays": "|".join(block_bays) if block_id else "",
                    "six_bay_block_total_boxes": block_total.get(block_id, 0) if block_id else 0,
                    "planned_boxes": qty,
                }
            )
        return rows

    def _medium_output_row(
        self,
        plan_level: str,
        voyage_id: str,
        flow: str,
        port: str,
        size: str,
        area_no: str,
        bay_key: str,
        bay_no: str,
        block_id: str,
        block_bays: tuple[str, ...],
        qty: int,
        source_counts: Counter[str] | None = None,
    ) -> dict:
        source_counts = Counter(source_counts or {})
        return {
            "plan_level": plan_level,
            "voyage_id": voyage_id,
            "flow": flow,
            "port": port,
            "size": size,
            "area_no": area_no,
            "bay_no": bay_no,
            "six_bay_block_id": block_id,
            "six_bay_block_bays": "|".join(block_bays) if block_id else "",
            "planned_boxes": qty,
            "document_boxes": int(source_counts.get("document", 0)),
            "forecast_fallback_boxes": int(source_counts.get("forecast_fallback", 0)),
        }

    def _make_medium_rows(self, small_rows: list[dict]) -> list[dict]:
        counter: Counter[tuple[str, str, str, str, str, str, str, str, tuple[str, ...]]] = Counter()
        for row in small_rows:
            area_no = str(row["area_no"])
            bay_no = str(row.get("bay_no", ""))
            bay_key = str(row.get("bay_key") or f"{area_no}-{bay_no}" if bay_no else "")
            block_id = str(row.get("six_bay_block_id", ""))
            block_bays = tuple(str(row.get("six_bay_block_bays", "")).split("|")) if row.get("six_bay_block_bays") else ()
            key = (
                str(row["voyage_id"]),
                str(row["flow"]),
                str(row["port"]),
                str(row["size"]),
                area_no,
                bay_key,
                bay_no,
                block_id,
                block_bays,
            )
            counter[key] += int(row["planned_boxes"])
        rows: list[dict] = []
        for (voyage_id, flow, port, size, area_no, bay_key, bay_no, block_id, block_bays), qty in sorted(counter.items()):
            rows.append(self._medium_output_row("medium_from_small", voyage_id, flow, port, size, area_no, bay_key, bay_no, block_id, block_bays, qty))
        return rows

    def _make_medium_rows_from_selected_columns(self, selected: Counter[int], plan_level: str = "medium") -> list[dict]:
        counter: Counter[tuple[str, str, str, str, str, str, str, str, tuple[str, ...]]] = Counter()
        source_counter: defaultdict[tuple, Counter[str]] = defaultdict(Counter)
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            qty = col.quantity * int(chosen)
            key = (
                col.voyage_id,
                col.flow,
                col.port,
                col.size,
                col.area_no,
                col.bay_key,
                col.bay_no,
                col.block_id,
                col.block_bays,
            )
            counter[key] += qty
            source_counter[key][self.group_source.get(col.group_id, "document")] += qty
        rows: list[dict] = []
        for (voyage_id, flow, port, size, area_no, bay_key, bay_no, block_id, block_bays), qty in sorted(counter.items()):
            if qty > 0:
                key = (voyage_id, flow, port, size, area_no, bay_key, bay_no, block_id, block_bays)
                rows.append(
                    self._medium_output_row(
                        plan_level,
                        voyage_id,
                        flow,
                        port,
                        size,
                        area_no,
                        bay_key,
                        bay_no,
                        block_id,
                        block_bays,
                        qty,
                        source_counter[key],
                    )
                )
        return rows

    def _make_original_medium_rows(self, selected: Counter[int]) -> list[dict]:
        return self._make_medium_rows_from_selected_columns(selected, plan_level="medium")

    def _make_original_medium_area_rows(self, selected: Counter[int]) -> list[dict]:
        selected_coarse_area_weights: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        selected_area_weights: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        small_lower_by_coarse_area: Counter[tuple[str, str, str, str, str]] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0:
                continue
            col = self._columns[idx]
            selected_coarse_area_weights[col.coarse_key][col.area_no] += col.quantity * chosen
            selected_area_weights[(col.voyage_id, col.flow, col.big_plan_size)][col.area_no] += col.quantity * chosen
            if self.group_source.get(col.group_id, "document") == "document":
                lower_key = col.coarse_key + (col.area_no,)
                lower_qty = col.quantity * chosen
                small_lower_by_coarse_area[lower_key] += lower_qty

        remaining_quota: Counter[tuple[str, str, str, str]] = Counter(self.quota_by_key)
        counter: Counter[tuple[str, str, str, str, str]] = Counter()
        medium_remaining_by_coarse: Counter[tuple[str, str, str, str]] = Counter()
        representative_by_coarse: dict[tuple[str, str, str, str], object] = {}
        for group in self.problem.groups:
            coarse_key = (group.voyage_id, group.status, group.port, group.size)
            medium_remaining_by_coarse[coarse_key] += int(group.demand)
            representative_by_coarse.setdefault(coarse_key, group)

        for (voyage_id, flow, port, size, area_no), qty in small_lower_by_coarse_area.items():
            if qty <= 0:
                continue
            counter[(voyage_id, flow, port, size, area_no)] += qty
            remaining_quota[(voyage_id, flow, area_no, self._big_plan_size(size))] -= qty
            self._consume_medium_remaining_for_small_lower(
                medium_remaining_by_coarse,
                voyage_id,
                flow,
                port,
                size,
                int(qty),
            )

        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        concentration_threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))
        sorted_groups = sorted(
            representative_by_coarse.values(),
            key=lambda g: (
                g.voyage_id,
                g.status,
                SIZE_ORDER.get(g.size, 3),
                0 if concentration_threshold > 0 and int(medium_remaining_by_coarse[(g.voyage_id, g.status, g.port, g.size)]) <= concentration_threshold else 1,
                (
                    int(medium_remaining_by_coarse[(g.voyage_id, g.status, g.port, g.size)])
                    if concentration_threshold > 0 and int(medium_remaining_by_coarse[(g.voyage_id, g.status, g.port, g.size)]) <= concentration_threshold
                    else -int(medium_remaining_by_coarse[(g.voyage_id, g.status, g.port, g.size)])
                ),
                g.port,
                g.group_id,
            ),
        )
        for group in sorted_groups:
            big_size = self._big_plan_size(group.size)
            coarse_key = (group.voyage_id, group.status, group.port, group.size)
            remaining_target = int(medium_remaining_by_coarse.get(coarse_key, 0))
            if remaining_target <= 0:
                continue
            quota_key_prefix = (group.voyage_id, group.status)
            caps = Counter(
                {
                    area_no: qty
                    for (voyage_id, flow, area_no, size), qty in remaining_quota.items()
                    if voyage_id == quota_key_prefix[0] and flow == quota_key_prefix[1] and size == big_size and qty > 0
                }
            )
            weights = Counter(
                {
                    area_no: qty
                    for area_no, qty in selected_coarse_area_weights.get(coarse_key, Counter()).items()
                    if caps.get(area_no, 0) > 0
                }
            )
            if not weights:
                weights = Counter(
                    {
                        area_no: qty
                        for area_no, qty in selected_area_weights.get((group.voyage_id, group.status, big_size), Counter()).items()
                        if caps.get(area_no, 0) > 0
                    }
                )
            if not weights:
                weights = Counter(caps)
            if not weights:
                weights = Counter(selected_coarse_area_weights.get(coarse_key, Counter()))
            if not weights:
                weights = Counter(selected_area_weights.get((group.voyage_id, group.status, big_size), Counter()))
            allocation = self._allocate_area_quantity(
                remaining_target,
                weights,
                self._caps_with_overflow(caps, weights, remaining_target),
                min_boxes=min_boxes,
                concentrate=(concentration_threshold > 0 and remaining_target <= concentration_threshold),
            )
            for area_no, qty in allocation.items():
                if qty <= 0:
                    continue
                counter[(group.voyage_id, group.status, group.port, group.size, area_no)] += qty
                remaining_quota[(group.voyage_id, group.status, area_no, big_size)] -= qty

        rows: list[dict] = []
        for (voyage_id, flow, port, size, area_no), qty in sorted(counter.items()):
            window_start, window_end = self.problem.voyage_windows[voyage_id]
            rows.append(
                {
                    "plan_level": "medium",
                    "voyage_id": voyage_id,
                    "flow": flow,
                    "port": port,
                    "size": size,
                    "window_start": window_start.isoformat(sep=" "),
                    "window_end": window_end.isoformat(sep=" "),
                    "area_loading_during_window": self._area_has_loading_during_window(voyage_id, area_no),
                    "area_loading_after_window_24h": self._area_has_loading_after_window(voyage_id, area_no),
                    "area_no": area_no,
                    "planned_boxes": qty,
                }
            )
        return rows

    def _consume_medium_remaining_for_small_lower(
        self,
        medium_remaining_by_coarse: Counter[tuple[str, str, str, str]],
        voyage_id: str,
        flow: str,
        port: str,
        size: str,
        qty: int,
    ) -> int:
        need = max(0, int(qty))
        if need <= 0:
            return 0
        exact_key = (voyage_id, flow, port, size)
        take = min(need, max(0, medium_remaining_by_coarse.get(exact_key, 0)))
        if take > 0:
            medium_remaining_by_coarse[exact_key] -= take
            need -= take
        return need

    @staticmethod
    def _caps_with_overflow(caps: Counter[str], weights: Counter[str], total: int) -> Counter[str]:
        adjusted = Counter({area: int(qty) for area, qty in caps.items() if qty > 0})
        missing = max(0, int(total) - sum(adjusted.values()))
        if missing <= 0:
            return adjusted
        overflow_areas = [area for area, qty in sorted(weights.items()) if qty > 0]
        if not overflow_areas:
            overflow_areas = sorted(adjusted)
        if not overflow_areas:
            return adjusted
        overflow_weights = Counter({area: max(1, int(weights.get(area, 0))) for area in overflow_areas})
        for area, qty in ColumnGenerationPlanner._allocate_integer_by_weights(overflow_weights, missing):
            adjusted[area] += qty
        return adjusted

    @classmethod
    def _allocate_area_quantity(
        cls,
        total: int,
        weights: Counter[str],
        caps: Counter[str],
        min_boxes: int = 0,
        concentrate: bool = False,
    ) -> Counter[str]:
        allocation: Counter[str] = Counter()
        remaining = max(0, int(total))
        available = Counter({area: int(cap) for area, cap in caps.items() if cap > 0})
        if remaining <= 0 or not available:
            return allocation
        if concentrate or (min_boxes > 1 and remaining < min_boxes):
            for area in sorted(
                available,
                key=lambda item: (
                    0 if available[item] >= remaining else 1,
                    -int(weights.get(item, 0)),
                    -available[item],
                    item,
                ),
            ):
                if remaining <= 0:
                    break
                take = min(remaining, available[area])
                if take <= 0:
                    continue
                allocation[area] += take
                remaining -= take
            return allocation

        while remaining > 0 and available:
            candidate_weights = Counter({area: max(0, int(weights.get(area, 0))) for area in available})
            if sum(candidate_weights.values()) <= 0:
                candidate_weights = Counter(available)
            if min_boxes > 1 and total >= min_boxes:
                filtered = Counter(
                    {
                        area: weight
                        for area, weight in candidate_weights.items()
                        if available[area] >= min_boxes
                    }
                )
                if filtered:
                    candidate_weights = filtered
                candidate_weights = cls._remove_tiny_weighted_areas(candidate_weights, remaining, min_boxes)
            raw_allocation = Counter(dict(cls._allocate_integer_by_weights(candidate_weights, remaining)))
            progress = 0
            overflow = 0
            for area in list(available):
                take = min(available[area], raw_allocation.get(area, 0))
                if take > 0:
                    allocation[area] += take
                    available[area] -= take
                    progress += take
                    if available[area] <= 0:
                        del available[area]
                overflow += max(0, raw_allocation.get(area, 0) - take)
            remaining -= progress
            if remaining <= 0:
                break
            if progress == 0:
                area = max(available, key=lambda key: (available[key], weights.get(key, 0), key))
                take = min(remaining, available[area])
                allocation[area] += take
                available[area] -= take
                remaining -= take
                if available[area] <= 0:
                    del available[area]
            elif overflow <= 0 and progress < remaining:
                weights = Counter(available)
        return cls._merge_small_fragments(allocation, caps, weights, min_boxes)

    @classmethod
    def _remove_tiny_weighted_areas(
        cls,
        weights: Counter[str],
        total: int,
        min_boxes: int,
    ) -> Counter[str]:
        if min_boxes <= 1 or total < min_boxes or len(weights) <= 1:
            return weights
        active = Counter({area: weight for area, weight in weights.items() if weight > 0})
        if len(active) <= 1:
            return weights
        while len(active) > 1:
            trial = Counter(dict(cls._allocate_integer_by_weights(active, total)))
            tiny_areas = [
                area
                for area, qty in trial.items()
                if 0 < qty < min_boxes
            ]
            if not tiny_areas or len(tiny_areas) >= len(active):
                break
            for area in tiny_areas:
                del active[area]
        return active or weights

    @staticmethod
    def _merge_small_fragments(
        allocation: Counter[str],
        caps: Counter[str],
        weights: Counter[str],
        min_boxes: int,
    ) -> Counter[str]:
        if min_boxes <= 1 or sum(allocation.values()) < min_boxes:
            return allocation
        merged = Counter({area: qty for area, qty in allocation.items() if qty > 0})
        for area in sorted(list(merged), key=lambda key: (merged[key], int(weights.get(key, 0)), key)):
            qty = merged.get(area, 0)
            if qty <= 0 or qty >= min_boxes:
                continue
            recipients = [
                target
                for target, target_qty in merged.items()
                if target != area and target_qty > 0 and int(caps.get(target, 0)) - target_qty >= qty
            ]
            if not recipients:
                continue
            target = sorted(
                recipients,
                key=lambda key: (
                    0 if merged[key] >= min_boxes else 1,
                    -int(weights.get(key, 0)),
                    -merged[key],
                    -(int(caps.get(key, 0)) - merged[key]),
                    key,
                ),
            )[0]
            merged[target] += qty
            del merged[area]
        while True:
            small_areas = [
                area
                for area, qty in merged.items()
                if 0 < qty < min_boxes
            ]
            total_small = sum(merged[area] for area in small_areas)
            if len(small_areas) <= 1 or total_small < min_boxes:
                break
            candidates: list[str] = []
            for area, cap in caps.items():
                current = merged.get(area, 0)
                if area in small_areas:
                    room = int(cap) - current
                    if current + room >= total_small:
                        candidates.append(area)
                elif current > 0:
                    if int(cap) - current >= total_small:
                        candidates.append(area)
                elif int(cap) >= total_small:
                    candidates.append(area)
            if not candidates:
                break
            target = sorted(
                candidates,
                key=lambda key: (
                    0 if merged.get(key, 0) >= min_boxes else 1,
                    0 if key in small_areas else 1,
                    -int(weights.get(key, 0)),
                    -int(caps.get(key, 0)),
                    key,
                ),
            )[0]
            for area in list(small_areas):
                if area == target:
                    continue
                merged[target] += merged[area]
                del merged[area]
            if merged[target] > int(caps.get(target, 0)):
                break
        return merged

    def _six_bay_blocks_for_area(self, area_no: str, bay_keys: list[str]) -> dict[str, tuple[str, ...]]:
        blocks: dict[str, tuple[str, ...]] = {}
        start = 0
        block_index = 1
        while start <= len(bay_keys) - 6:
            members = tuple(bay_keys[start : start + 6])
            if self._is_preferred_six_bay_block(members):
                blocks[f"{area_no}-SB{block_index:02d}"] = members
                block_index += 1
                start += 6
            else:
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

    def _group_sort_key(self, group: SmallBoxGroup) -> tuple[int, int, int, str, str, str]:
        return (
            SIZE_ORDER.get(group.size, 3),
            len(self._area_weights(group)) if hasattr(self, "quota_by_key") else 0,
            -group.demand,
            group.voyage_id,
            group.group_id,
            group.port,
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


def write_columns(path: str | Path, columns: Iterable[PlacementColumn]) -> None:
    rows = [
        {
            "column_id": col.column_id,
            "group_id": col.group_id,
            "demand_source": col.demand_source,
            "voyage_id": col.voyage_id,
            "flow": col.flow,
            "port": col.port,
            "size": col.size,
            "height": col.height,
            "area_no": col.area_no,
            "bay_no": col.bay_no,
            "six_bay_block_id": col.block_id,
            "quantity": col.quantity,
            "stack_units": col.stack_units,
            "intrinsic_cost": round(col.intrinsic_cost, 6),
        }
        for col in columns
    ]
    write_rows(path, rows)


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _time_windows_overlap(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    return start_a < end_b and start_b < end_a
