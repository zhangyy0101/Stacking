from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Iterable

from medium_small_raw.block_bay_planning.models import Bay, ProblemData, SmallBoxGroup


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
    row_allocation: tuple[tuple[str, str, int], ...]
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
    stage0_closure_enabled: bool = True
    stage0_closure_max_extra_columns: int = 1500
    stage0_min_unplaced_time_limit: float = 12.0
    initial_columns_per_group: int = 16
    max_candidate_bays_per_group: int = 500
    mip_time_limit: float = 120.0
    mip_gap: float = 0.01
    diving_max_steps: int = 200
    diving_fractional_tolerance: float = 1e-5
    diving_fix_batch_size: int = 8
    diving_max_no_improve_steps: int = 5
    diving_improvement_rounds: int = 6
    diving_improvement_time_limit: float = 6.0
    diving_improvement_max_groups: int = 14
    diving_improvement_max_no_improve_rounds: int = 2
    repair_lns_rounds: int = 2
    repair_lns_time_limit: float = 3.0
    repair_lns_max_groups: int = 16
    repair_lns_max_no_improve_rounds: int = 2
    coarse_compaction_lns_rounds: int = 0
    coarse_compaction_lns_time_limit: float = 4.0
    coarse_compaction_lns_max_groups: int = 16
    coarse_compaction_lns_max_no_improve_rounds: int = 1
    post_repair_area_relayout_enabled: bool = True
    post_repair_area_relayout_max_patterns: int = 1
    diving_price_columns: bool = False
    diving_stop_on_lp_unplaced: bool = True
    diving_skip_when_lp_unplaced: bool = True
    verbose: bool = True
    use_scip: bool = True
    full_column_pool: bool = False
    demand_mode: str = "original"
    medium_plan_quota: dict[tuple[str, str, str, str, str], int] | None = None
    medium_plan_bay_quota: dict[tuple[str, str, str, str, str, str], int] | None = None
    unplaced_penalty: float = 100_000.0
    required_area_reward: float = 1_000.0
    group_area_balance_penalty: float = 18.0
    medium_concentrated_group_threshold: int = 26
    medium_small_group_area_split_penalty: float = 2400.0
    medium_small_group_fragment_penalty: float = 90.0
    medium_large_group_min_area_boxes: int = 10
    medium_large_group_small_area_penalty: float = 900.0
    medium_large_group_area_open_penalty: float = 0.0
    medium_large_group_target_area_boxes: int = 60
    big_plan_area_deviation_penalty: float = 3.0
    big_plan_fallback_tier_penalty: float = 20.0
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
        self._expand_user_bay_adjust_rules()
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

    def _expand_user_bay_adjust_rules(self) -> None:
        rules = getattr(self.problem, "user_bay_adjust_rules", []) or []
        if not rules:
            return
        requirements = {
            str(group_id): set(values)
            for group_id, values in getattr(self.problem, "user_group_bay_requirements", {}).items()
        }
        blocklist = {
            str(group_id): set(values)
            for group_id, values in getattr(self.problem, "user_group_bay_blocklist", {}).items()
        }
        matched = 0
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            voyage_id = str(rule.get("voyage_id", ""))
            attributes = rule.get("attributes", {}) if isinstance(rule.get("attributes", {}), dict) else {}
            required_bays = set(rule.get("required_bays", set()) or set())
            blocked_bays = set(rule.get("blocked_bays", set()) or set())
            for group in self.groups:
                if voyage_id and str(group.voyage_id) != voyage_id:
                    continue
                if not self._group_matches_bay_adjust_attributes(group, attributes):
                    continue
                matched += 1
                if required_bays:
                    requirements.setdefault(group.group_id, set()).update(required_bays)
                if blocked_bays:
                    blocklist.setdefault(group.group_id, set()).update(blocked_bays)
        for group_id, blocked in blocklist.items():
            if blocked and group_id in requirements:
                requirements[group_id].difference_update(blocked)
        self.problem.user_group_bay_requirements = {group_id: values for group_id, values in requirements.items() if values}
        self.problem.user_group_bay_blocklist = {group_id: values for group_id, values in blocklist.items() if values}
        summary = dict(getattr(self.problem, "user_bay_constraint_summary", {}) or {})
        summary["expanded_matched_planning_groups"] = matched
        summary["expanded_required_group_count"] = len(self.problem.user_group_bay_requirements)
        summary["expanded_blocked_group_count"] = len(self.problem.user_group_bay_blocklist)
        self.problem.user_bay_constraint_summary = summary

    def _group_matches_bay_adjust_attributes(self, group: SmallBoxGroup, attributes: dict) -> bool:
        for attr, expected in attributes.items():
            if self._group_attr_value(group, str(attr)) != str(expected):
                return False
        return True

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
            "scip_available": False,
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
                "medium_large_group_area_open": self.config.medium_large_group_area_open_penalty,
                "medium_large_group_target_area_boxes": self.config.medium_large_group_target_area_boxes,
            },
            "inheritance_penalties": {
                "unplaced": self.config.unplaced_penalty,
                "required_area_reward": self.config.required_area_reward,
                "big_plan_area_deviation": self.config.big_plan_area_deviation_penalty,
                "big_plan_fallback_tier": self.config.big_plan_fallback_tier_penalty,
            },
        }

        selected: Counter[int]
        unplaced: Counter[str]
        try:
            if not self.config.use_scip:
                raise RuntimeError("SCIP disabled by config")
            selected, unplaced, master_stats = self._solve_by_column_generation()
            diagnostics.update(master_stats)
        except Exception as exc:
            diagnostics["used_greedy_fallback"] = True
            diagnostics["scip_failure"] = f"{type(exc).__name__}: {exc}"
            selected, unplaced = self._greedy_fallback()

        selected, unplaced, repair_stats = self._repair_or_replace_unplaced_solution(selected, unplaced)
        diagnostics.update(repair_stats)
        selected, relayout_stats = self._post_repair_area_relayout(selected, unplaced)
        diagnostics.update(relayout_stats)

        if self._uses_original_output_scope():
            small_rows = self._make_small_rows(selected, allowed_sources={"document"})
        else:
            small_rows = self._make_small_rows(selected)
        medium_rows = self._make_medium_rows_from_selected_columns(selected, plan_level="medium")
        consistency_stats = self._small_medium_consistency_stats(small_rows, medium_rows)
        bay_consistency_stats = self._small_medium_bay_consistency_stats(small_rows, medium_rows)
        medium_fragmentation = self._medium_fragmentation_stats(medium_rows)
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
                "medium_fragmentation": medium_fragmentation,
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
        lns_selected, lns_unplaced, lns_stats = self._run_repair_lns_rounds(
            best_selected,
            best_unplaced,
            unplaced,
        )
        if self._solution_rank(lns_selected, lns_unplaced) < self._solution_rank(best_selected, best_unplaced):
            method = "staged_fallback_repair_lns"
            best_selected = lns_selected
            best_unplaced = lns_unplaced
        compaction_selected, compaction_unplaced, compaction_stats = self._run_coarse_compaction_lns_rounds(
            best_selected,
            best_unplaced,
        )
        if self._solution_rank(compaction_selected, compaction_unplaced) < self._solution_rank(best_selected, best_unplaced):
            method = "staged_fallback_repair_lns_compaction"
            best_selected = compaction_selected
            best_unplaced = compaction_unplaced
        stats.update(
            {
                "used_unplaced_repair": method != "master_incumbent",
                "unplaced_repair_method": method,
                "staged_repair_unplaced_boxes": sum(staged_unplaced.values()),
                "staged_repair_iterations": stage_stats.get("iterations", []),
                "staged_repair_candidates": stage_stats.get("candidates", []),
                "staged_repair_selected_candidate": stage_stats.get("selected_candidate", ""),
                "post_repair_unplaced_boxes": sum(best_unplaced.values()),
                **lns_stats,
                **compaction_stats,
            }
        )
        return best_selected, best_unplaced, stats

    def _post_repair_area_relayout(
        self,
        selected: Counter[int],
        unplaced: Counter[str],
    ) -> tuple[Counter[int], dict]:
        stats: dict = {
            "post_repair_area_relayout_enabled": bool(getattr(self.config, "post_repair_area_relayout_enabled", True)),
            "post_repair_area_relayout_used": False,
            "post_repair_area_relayout_accepted": False,
            "post_repair_area_relayout_skip_reason": "",
        }
        if not stats["post_repair_area_relayout_enabled"]:
            stats["post_repair_area_relayout_skip_reason"] = "disabled"
            return selected, stats
        if sum(unplaced.values()) > 0:
            stats["post_repair_area_relayout_skip_reason"] = "has_unplaced_boxes"
            return selected, stats
        if self.config.medium_plan_bay_quota is not None:
            stats["post_repair_area_relayout_skip_reason"] = "fixed_medium_bay_quota"
            return selected, stats

        target_group_area = self._selected_group_area_quantities(selected)
        if not target_group_area:
            stats["post_repair_area_relayout_skip_reason"] = "empty_solution"
            return selected, stats

        before_metrics = self._area_relayout_concentration_metrics(selected)
        candidate, candidate_stats = self._build_area_relayout_solution(target_group_area, selected)
        stats.update(candidate_stats)
        stats["post_repair_area_relayout_before"] = before_metrics
        if candidate is None:
            stats["post_repair_area_relayout_skip_reason"] = candidate_stats.get("post_repair_area_relayout_failure_reason", "failed")
            return selected, stats
        if self._selected_group_area_quantities(candidate) != target_group_area:
            stats["post_repair_area_relayout_skip_reason"] = "area_quantities_changed"
            return selected, stats

        after_metrics = self._area_relayout_concentration_metrics(candidate)
        stats["post_repair_area_relayout_after"] = after_metrics
        stats["post_repair_area_relayout_delta_score"] = round(before_metrics["score"] - after_metrics["score"], 6)
        stats["post_repair_area_relayout_used"] = True
        if after_metrics["score"] + 1e-6 < before_metrics["score"]:
            stats["post_repair_area_relayout_accepted"] = True
            return candidate, stats
        stats["post_repair_area_relayout_skip_reason"] = "no_concentration_improvement"
        return selected, stats

    def _selected_group_area_quantities(self, selected: Counter[int]) -> Counter[tuple[str, str]]:
        quantities: Counter[tuple[str, str]] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            quantities[(col.group_id, col.area_no)] += int(chosen) * int(col.quantity)
        return quantities

    def _area_relayout_concentration_metrics(self, selected: Counter[int]) -> dict[str, float | int]:
        fine_area_bays: set[tuple[str, str, str]] = set()
        coarse_area_bays: Counter[tuple[str, str, str, str, str, str]] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            qty = int(chosen) * int(col.quantity)
            fine_area_bays.add((col.group_id, col.area_no, col.bay_key))
            coarse_area_bays[col.coarse_key + (col.area_no, col.bay_key)] += qty

        fine_area_pairs = {(group_id, area_no) for group_id, area_no, _bay_key in fine_area_bays}
        coarse_area_pairs = {key[:5] for key in coarse_area_bays}
        fine_excess_bays = max(0, len(fine_area_bays) - len(fine_area_pairs))
        coarse_excess_bays = max(0, len(coarse_area_bays) - len(coarse_area_pairs))
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        coarse_tail_boxes = 0
        if min_boxes > 0:
            for qty in coarse_area_bays.values():
                if 0 < qty < min_boxes:
                    coarse_tail_boxes += min_boxes - int(qty)
        score = (
            1000.0 * fine_excess_bays
            + 360.0 * coarse_excess_bays
            + 24.0 * coarse_tail_boxes
            + 10.0 * len(fine_area_bays)
            + 4.0 * len(coarse_area_bays)
        )
        return {
            "score": round(score, 6),
            "fine_area_bays": len(fine_area_bays),
            "fine_area_pairs": len(fine_area_pairs),
            "fine_excess_bays": fine_excess_bays,
            "coarse_area_bays": len(coarse_area_bays),
            "coarse_area_pairs": len(coarse_area_pairs),
            "coarse_excess_bays": coarse_excess_bays,
            "coarse_tail_boxes": coarse_tail_boxes,
        }

    def _build_area_relayout_solution(
        self,
        target_group_area: Counter[tuple[str, str]],
        original_selected: Counter[int],
    ) -> tuple[Counter[int] | None, dict]:
        selected: Counter[int] = Counter()
        state = self._empty_selection_state()
        relayout_state = {
            "coarse_area_bay_load": Counter(),
            "group_area_bay_load": Counter(),
        }
        stats: dict = {
            "post_repair_area_relayout_group_area_pairs": len(target_group_area),
            "post_repair_area_relayout_selected_columns": 0,
            "post_repair_area_relayout_failure_reason": "",
            "post_repair_area_relayout_areas_attempted": 0,
            "post_repair_area_relayout_areas_relaid": 0,
            "post_repair_area_relayout_areas_kept_original": 0,
        }

        coarse_area_total: Counter[tuple[str, str, str, str, str]] = Counter()
        for (group_id, area_no), qty in target_group_area.items():
            group = self.groups_by_id.get(group_id)
            if group is None:
                stats["post_repair_area_relayout_failure_reason"] = "missing_group"
                stats["post_repair_area_relayout_failed_group"] = group_id
                return None, stats
            coarse_area_total[self._coarse_key(group) + (area_no,)] += int(qty)

        by_area_coarse: defaultdict[tuple[str, tuple[str, str, str, str]], list[tuple[SmallBoxGroup, str, int]]] = defaultdict(list)
        target_areas: set[str] = set()
        for (group_id, area_no), qty in target_group_area.items():
            qty = int(qty)
            if qty <= 0:
                continue
            target_areas.add(area_no)
            group = self.groups_by_id[group_id]
            by_area_coarse[(area_no, self._coarse_key(group))].append((group, area_no, qty))

        original_by_area: defaultdict[str, list[tuple[int, PlacementColumn, int]]] = defaultdict(list)
        for idx, chosen in original_selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            original_by_area[col.area_no].append((idx, col, int(chosen)))

        for area_no in sorted(target_areas):
            stats["post_repair_area_relayout_areas_attempted"] += 1
            trial_selected = Counter(selected)
            trial_state = self._copy_selection_state_for_relayout(state)
            trial_relayout_state = self._copy_area_relayout_state(relayout_state)
            area_success = True
            area_coarse_keys = sorted(
                [key for key in by_area_coarse if key[0] == area_no],
                key=lambda key: (
                    0 if coarse_area_total[key[1] + (key[0],)] <= self.config.medium_concentrated_group_threshold else 1,
                    coarse_area_total[key[1] + (key[0],)] if coarse_area_total[key[1] + (key[0],)] <= self.config.medium_concentrated_group_threshold else -coarse_area_total[key[1] + (key[0],)],
                    key[1],
                ),
            )
            for area_coarse_key in area_coarse_keys:
                entries = sorted(
                    by_area_coarse[area_coarse_key],
                    key=lambda item: (-int(item[2]), self._group_sort_key(item[0])),
                )
                for group, entry_area_no, target_qty in entries:
                    remaining = int(target_qty)
                    while remaining > 0:
                        choice = self._best_area_relayout_column(group, entry_area_no, remaining, trial_state, trial_relayout_state)
                        if choice is None:
                            area_success = False
                            stats["post_repair_area_relayout_failure_reason"] = "partial_area_fallback"
                            stats["post_repair_area_relayout_last_failed_group"] = group.group_id
                            stats["post_repair_area_relayout_last_failed_area"] = entry_area_no
                            stats["post_repair_area_relayout_last_failed_remaining"] = remaining
                            break
                        idx, col = choice
                        self._apply_column_to_state(col, trial_state)
                        self._apply_column_to_area_relayout_state(col, trial_relayout_state)
                        trial_selected[idx] += 1
                        remaining -= int(col.quantity)
                    if not area_success:
                        break
                if not area_success:
                    break
            if area_success:
                selected = trial_selected
                state = trial_state
                relayout_state = trial_relayout_state
                stats["post_repair_area_relayout_areas_relaid"] += 1
            else:
                stats["post_repair_area_relayout_areas_kept_original"] += 1
                for idx, col, chosen in original_by_area.get(area_no, []):
                    for _ in range(chosen):
                        self._apply_column_to_state(col, state)
                        self._apply_column_to_area_relayout_state(col, relayout_state)
                        selected[idx] += 1

        stats["post_repair_area_relayout_selected_columns"] = sum(1 for qty in selected.values() if qty > 0)
        return selected, stats

    @staticmethod
    def _copy_selection_state_for_relayout(state: dict) -> dict:
        copied = {}
        for key, value in state.items():
            if isinstance(value, Counter):
                copied[key] = Counter(value)
            elif isinstance(value, set):
                copied[key] = set(value)
            elif isinstance(value, dict):
                copied[key] = dict(value)
            else:
                copied[key] = value
        return copied

    @staticmethod
    def _copy_area_relayout_state(state: dict) -> dict:
        return {key: Counter(value) for key, value in state.items()}

    def _best_area_relayout_column(
        self,
        group: SmallBoxGroup,
        area_no: str,
        remaining: int,
        state: dict,
        relayout_state: dict,
    ) -> tuple[int, PlacementColumn] | None:
        best: tuple[tuple[float, int, int, str], int, PlacementColumn] | None = None
        for bay_key in self.bays_by_area.get(area_no, []):
            if self._max_quantity_in_bay(group, bay_key) <= 0:
                continue
            base_cost = self._column_base_cost(group, bay_key)
            capacity = self._remaining_capacity_for_group_bay(group, bay_key, state, remaining, enforce_quota=False)
            if capacity <= 0:
                continue
            qty = min(int(remaining), int(capacity))
            patterns = self._row_allocation_patterns_for_column(
                group,
                bay_key,
                qty,
                state=state,
                max_patterns=max(1, int(getattr(self.config, "post_repair_area_relayout_max_patterns", 8) or 8)),
            )
            for pattern in patterns:
                idx = self._ensure_area_relayout_column(group, bay_key, qty, base_cost, pattern)
                if idx is None:
                    continue
                col = self._columns[idx]
                if not self._column_fits_state(col, state, remaining, enforce_quota=False):
                    continue
                score = (
                    self._area_relayout_column_score(col, relayout_state, remaining),
                    0 if int(col.quantity) >= int(remaining) else 1,
                    -int(col.quantity),
                    col.bay_key,
                )
                candidate = (score, idx, col)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return None
        return best[1], best[2]

    def _ensure_area_relayout_column(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        quantity: int,
        base_cost: float,
        row_allocation: tuple[tuple[str, str, int], ...],
    ) -> int | None:
        signature = self._row_allocation_signature(row_allocation)
        for idx, col in enumerate(self._columns):
            if (
                col.group_id == group.group_id
                and col.bay_key == bay_key
                and int(col.quantity) == int(quantity)
                and col.row_allocation == signature
            ):
                return idx
        before = len(self._columns)
        self._add_column(group, bay_key, int(quantity), base_cost, row_allocation=signature)
        for idx in range(before, len(self._columns)):
            col = self._columns[idx]
            if (
                col.group_id == group.group_id
                and col.bay_key == bay_key
                and int(col.quantity) == int(quantity)
                and col.row_allocation == signature
            ):
                return idx
        return None

    def _area_relayout_column_score(self, col: PlacementColumn, relayout_state: dict, remaining: int) -> float:
        group_bay_key = (col.group_id, col.area_no, col.bay_key)
        coarse_bay_key = col.coarse_key + (col.area_no, col.bay_key)
        existing_group_bay = int(relayout_state["group_area_bay_load"][group_bay_key])
        existing_coarse_bay = int(relayout_state["coarse_area_bay_load"][coarse_bay_key])
        score = 0.0
        if existing_group_bay <= 0:
            score += 1000.0
        else:
            score -= min(300.0, float(existing_group_bay))
        if existing_coarse_bay <= 0:
            score += 360.0
        else:
            score -= min(180.0, float(existing_coarse_bay))
        score -= 12.0 * int(col.quantity)
        if int(remaining) - int(col.quantity) > 0:
            score += 80.0
        score += 0.01 * self.bays[col.bay_key].bay_order
        return score

    def _apply_column_to_area_relayout_state(self, col: PlacementColumn, relayout_state: dict) -> None:
        group_bay_key = (col.group_id, col.area_no, col.bay_key)
        coarse_bay_key = col.coarse_key + (col.area_no, col.bay_key)
        relayout_state["group_area_bay_load"][group_bay_key] += int(col.quantity)
        relayout_state["coarse_area_bay_load"][coarse_bay_key] += int(col.quantity)

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
            energy += self._area_fallback_tier_penalty_for_column(col) * qty
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
                energy += self.config.medium_large_group_area_open_penalty * max(
                    0,
                    len(quantities) - self._target_large_group_area_count(coarse_key, demand),
                )
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

    def _area_fallback_tier_penalty_for_column(self, col: PlacementColumn) -> float:
        tier = self._area_fallback_tier_for_attrs(
            col.voyage_id,
            col.flow,
            col.area_no,
            col.big_plan_size,
        )
        return float(self.config.big_plan_fallback_tier_penalty) * tier

    def _repair_from_column_priority(
        self,
        column_values: dict[int, float],
        allow_new_columns: bool = True,
    ) -> tuple[Counter[int], Counter[str]]:
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
        return self._repair_selected_solution(selected, allow_new_columns=allow_new_columns)

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

    def _medium_fragmentation_stats(self, medium_rows: list[dict]) -> dict[str, int | float]:
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        concentration_threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))
        by_coarse_area: Counter[tuple[str, str, str, str, str]] = Counter()
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
            by_coarse_area[key] += qty
        by_coarse: defaultdict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        for (*coarse_key, _area_no), qty in by_coarse_area.items():
            by_coarse[tuple(coarse_key)].append(qty)
        tiny_rows = 0
        small_groups = 0
        small_multi_area_groups = 0
        large_groups = 0
        large_tiny_rows = 0
        max_area_count = 0
        for quantities in by_coarse.values():
            total = sum(quantities)
            area_count = len(quantities)
            max_area_count = max(max_area_count, area_count)
            tiny_count = sum(1 for qty in quantities if min_boxes > 1 and 0 < qty < min_boxes)
            tiny_rows += tiny_count
            if concentration_threshold > 0 and total <= concentration_threshold:
                small_groups += 1
                if area_count > 1:
                    small_multi_area_groups += 1
            else:
                large_groups += 1
                large_tiny_rows += tiny_count
        return {
            "coarse_group_count": len(by_coarse),
            "tiny_area_rows": tiny_rows,
            "small_coarse_group_count": small_groups,
            "small_coarse_multi_area_groups": small_multi_area_groups,
            "large_coarse_group_count": large_groups,
            "large_coarse_tiny_area_rows": large_tiny_rows,
            "max_area_count_per_coarse_group": max_area_count,
            "average_area_count_per_coarse_group": round(
                sum(len(quantities) for quantities in by_coarse.values()) / max(1, len(by_coarse)),
                4,
            ),
        }

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
        from pyscipopt import Model, quicksum

        stats = {"scip_available": True, "pricing_iterations": []}
        final_lp_bound = None
        best_start_source = "greedy_seed"
        best_start_selected = Counter(self._master_seed_selected)
        best_start_unplaced = Counter(self._master_seed_unplaced)
        last_lp_unplaced: Counter[str] = Counter(self._master_seed_unplaced)
        pricing_iterations = 0 if self.config.full_column_pool else self.config.max_iterations
        for iteration in range(pricing_iterations):
            iteration_start = perf_counter()
            if self.config.verbose:
                print(
                    f"[column-generation-scip] building LP iter={iteration} columns={len(self._columns)}",
                    flush=True,
                )
            lp_model, lp_vars, lp_constraints = self._build_restricted_master(Model, quicksum, relax=True)
            self._set_scip_param(lp_model, "limits/time", float(self.config.mip_time_limit))
            if self.config.verbose:
                print(f"[column-generation-scip] solving LP iter={iteration}", flush=True)
            lp_model.optimize()
            lp_status = self._scip_status_name(lp_model)
            if lp_status not in {"optimal"}:
                raise RuntimeError(f"restricted master LP status={lp_status}")
            lp_objective = self._scip_objective_value(lp_model)
            lp_unplaced = self._scip_unplaced_values(lp_model, lp_vars)
            last_lp_unplaced = Counter(lp_unplaced)
            lp_column_values = self._scip_column_values(lp_model, lp_vars)
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
            pricing_stats = self._price_columns(lp_model, lp_constraints, iteration, lp_unplaced, lp_column_values)
            new_count = int(pricing_stats.get("new_columns", 0) or 0)
            stats["pricing_iterations"].append(
                {
                    "iteration": iteration,
                    "columns": len(self._columns),
                    "lp_objective": lp_objective,
                    "lp_unplaced_boxes": sum(lp_unplaced.values()),
                    "lp_guided_repair_unplaced_boxes": sum(lp_repair_unplaced.values()),
                    "lp_guided_repair_columns": sum(1 for qty in lp_repair_selected.values() if qty > 0),
                    **pricing_stats,
                }
            )
            if self.config.verbose:
                print(
                    f"[column-generation-scip] iter={iteration} lp={lp_objective:.3f} "
                    f"columns={len(self._columns)} new={new_count} elapsed={perf_counter() - iteration_start:.1f}s",
                    flush=True,
                )
            self._free_scip_model(lp_model)
            if new_count == 0:
                break

        closure_source_unplaced = Counter(best_start_unplaced)
        for group_id, qty in last_lp_unplaced.items():
            closure_source_unplaced[group_id] = max(int(closure_source_unplaced.get(group_id, 0)), int(qty))
        closure_stats = self._stage0_unplaced_column_closure(closure_source_unplaced)
        stats.update(closure_stats)
        augmented_selected, augmented_unplaced = self._repair_selected_solution(best_start_selected)
        augmented_stats = self._stage0_repaired_candidate_stats(
            "stage0_augmented_repair",
            best_start_selected,
            best_start_unplaced,
            augmented_selected,
            augmented_unplaced,
        )
        stats.update(augmented_stats)
        if augmented_stats.get("stage0_augmented_repair_accepted"):
            best_start_source = "stage0_augmented_repair"
            best_start_selected = augmented_selected
            best_start_unplaced = augmented_unplaced

        min_stage0_selected, min_stage0_unplaced, min_stage0_stats = self._solve_stage0_min_unplaced_master(Model, quicksum)
        stats.update(min_stage0_stats)
        stage0_min_proven = (
            bool(min_stage0_stats.get("stage0_min_unplaced_has_solution"))
            and str(min_stage0_stats.get("stage0_min_unplaced_status", "")).lower() == "optimal"
        )
        if stage0_min_proven:
            stage0_unplaced_cap = sum(min_stage0_unplaced.values())
            stage0_unplaced_cap_source = "stage0_min_unplaced_optimal"
        elif sum(best_start_unplaced.values()) <= 0:
            stage0_unplaced_cap = 0
            stage0_unplaced_cap_source = "zero_unplaced_incumbent"
        else:
            stage0_unplaced_cap = None
            stage0_unplaced_cap_source = ""
        if stage0_min_proven and self._solution_rank(
            min_stage0_selected,
            min_stage0_unplaced,
        ) < self._solution_rank(best_start_selected, best_start_unplaced):
            best_start_source = "stage0_min_unplaced_master"
            best_start_selected = min_stage0_selected
            best_start_unplaced = min_stage0_unplaced
        elif min_stage0_stats.get("stage0_min_unplaced_has_solution"):
            candidate_stats = self._stage0_repaired_candidate_stats(
                "stage0_min_unplaced_candidate",
                best_start_selected,
                best_start_unplaced,
                min_stage0_selected,
                min_stage0_unplaced,
            )
            stats.update(candidate_stats)
            if candidate_stats.get("stage0_min_unplaced_candidate_accepted"):
                best_start_source = "stage0_min_unplaced_candidate"
                best_start_selected = min_stage0_selected
                best_start_unplaced = min_stage0_unplaced

        final_lp_model, _final_lp_vars, _final_lp_constraints = self._build_restricted_master(
            Model,
            quicksum,
            relax=True,
            unplaced_cap=stage0_unplaced_cap,
        )
        self._set_scip_param(final_lp_model, "limits/time", min(float(self.config.mip_time_limit), 30.0))
        final_lp_model.optimize()
        if self._scip_status_name(final_lp_model) == "optimal":
            final_lp_bound = self._scip_objective_value(final_lp_model)
        self._free_scip_model(final_lp_model)

        self._master_start_selected = best_start_selected
        self._master_start_unplaced = best_start_unplaced
        stats.update(
            {
                "master_mip_start_source": best_start_source,
                "master_mip_start_repaired_columns": sum(1 for qty in best_start_selected.values() if qty > 0),
                "master_mip_start_repaired_unplaced_boxes": sum(best_start_unplaced.values()),
                "stage0_min_unplaced_cap_enforced": stage0_unplaced_cap is not None,
                "stage0_unplaced_cap": stage0_unplaced_cap,
                "stage0_unplaced_cap_source": stage0_unplaced_cap_source,
                "restricted_master_lp_bound": final_lp_bound,
            }
        )

        selected, unplaced, diving_stats = self._solve_master_by_diving(
            Model,
            quicksum,
            final_lp_bound,
            best_start_source,
            best_start_selected,
            best_start_unplaced,
            stage0_unplaced_cap,
        )
        stats.update(diving_stats)
        return selected, unplaced, stats

    def _stage0_repaired_candidate_stats(
        self,
        label: str,
        incumbent_selected: Counter[int],
        incumbent_unplaced: Counter[str],
        candidate_selected: Counter[int],
        candidate_unplaced: Counter[str],
    ) -> dict:
        before_columns = len(self._columns)
        incumbent_final_selected, incumbent_final_unplaced, _incumbent_stage_stats = self._staged_repair_selected_solution(
            incumbent_selected,
            allow_new_columns=True,
        )
        candidate_final_selected, candidate_final_unplaced, _candidate_stage_stats = self._staged_repair_selected_solution(
            candidate_selected,
            allow_new_columns=True,
        )
        incumbent_rank = self._solution_rank(incumbent_final_selected, incumbent_final_unplaced)
        candidate_rank = self._solution_rank(candidate_final_selected, candidate_final_unplaced)
        accepted = candidate_rank < incumbent_rank
        return {
            f"{label}_pre_unplaced_boxes": sum(candidate_unplaced.values()),
            f"{label}_final_unplaced_boxes": sum(candidate_final_unplaced.values()),
            f"{label}_final_objective": round(self._selected_solution_energy(candidate_final_selected, candidate_final_unplaced), 6),
            f"{label}_incumbent_final_unplaced_boxes": sum(incumbent_final_unplaced.values()),
            f"{label}_incumbent_final_objective": round(self._selected_solution_energy(incumbent_final_selected, incumbent_final_unplaced), 6),
            f"{label}_accepted": accepted,
            f"{label}_comparison_added_columns": len(self._columns) - before_columns,
        }

    def _solve_stage0_min_unplaced_master(self, Model, quicksum) -> tuple[Counter[int], Counter[str], dict]:
        stats = {
            "stage0_min_unplaced_has_solution": False,
            "stage0_min_unplaced_status": "",
            "stage0_min_unplaced_boxes": None,
            "stage0_min_unplaced_columns": 0,
            "stage0_min_unplaced_seconds": 0.0,
            "stage0_min_unplaced_gap": None,
            "stage0_min_unplaced_objective": None,
        }
        start = perf_counter()
        model, vars_by_name, _constraints = self._build_restricted_master(
            Model,
            quicksum,
            relax=False,
            objective_mode="min_unplaced",
        )
        time_limit = float(getattr(self.config, "stage0_min_unplaced_time_limit", 0.0) or 0.0)
        if time_limit > 0:
            self._set_scip_param(model, "limits/time", time_limit)
        self._set_scip_param(model, "limits/gap", 0.0)
        self._add_scip_mip_start(model, vars_by_name)
        model.optimize()
        status = self._scip_status_name(model)
        has_solution = self._scip_solution_count(model) > 0
        selected = Counter(self._master_seed_selected)
        unplaced = Counter(self._master_seed_unplaced)
        if has_solution:
            selected, unplaced = self._solution_from_scip_vars(model, vars_by_name)
        stats.update(
            {
                "stage0_min_unplaced_has_solution": has_solution,
                "stage0_min_unplaced_status": status,
                "stage0_min_unplaced_boxes": sum(unplaced.values()) if has_solution else None,
                "stage0_min_unplaced_columns": sum(1 for qty in selected.values() if qty > 0) if has_solution else 0,
                "stage0_min_unplaced_seconds": round(perf_counter() - start, 3),
                "stage0_min_unplaced_gap": self._scip_gap(model) if has_solution else None,
                "stage0_min_unplaced_objective": self._scip_objective_value(model) if has_solution else None,
            }
        )
        self._free_scip_model(model)
        return selected, unplaced, stats

    def _stage0_unplaced_column_closure(self, unplaced: Counter[str]) -> dict:
        stats = {
            "stage0_closure_enabled": bool(getattr(self.config, "stage0_closure_enabled", False)),
            "stage0_closure_source_unplaced_boxes": sum(qty for qty in unplaced.values() if qty > 0),
            "stage0_closure_group_count": sum(1 for qty in unplaced.values() if qty > 0),
            "stage0_closure_added_columns": 0,
            "stage0_closure_hit_limit": False,
        }
        if not stats["stage0_closure_enabled"] or not unplaced:
            return stats
        limit = max(0, int(getattr(self.config, "stage0_closure_max_extra_columns", 0) or 0))
        added = 0
        groups = [
            self.groups_by_id[group_id]
            for group_id, qty in sorted(unplaced.items(), key=lambda item: (-int(item[1]), item[0]))
            if qty > 0 and group_id in self.groups_by_id
        ]
        for group in groups:
            unplaced_qty = max(1, int(unplaced.get(group.group_id, 0)))
            for bay_key, max_qty, base_cost in self._candidate_bays_for_group(group, scope="stage0"):
                for qty in self._stage0_closure_quantity_options(group, max_qty, unplaced_qty):
                    if self._add_column(group, bay_key, qty, base_cost):
                        added += 1
                        if limit > 0 and added >= limit:
                            stats["stage0_closure_added_columns"] = added
                            stats["stage0_closure_hit_limit"] = True
                            return stats
        stats["stage0_closure_added_columns"] = added
        return stats

    def _stage0_closure_quantity_options(self, group: SmallBoxGroup, max_qty: int, unplaced_qty: int) -> list[int]:
        max_qty = max(0, min(int(max_qty), int(group.demand)))
        if max_qty <= 0:
            return []
        values = set(self._quantity_options(group, max_qty))
        values.update(
            qty
            for qty in (
                1,
                2,
                3,
                4,
                5,
                max_qty // 3,
                max_qty // 2,
                (2 * max_qty) // 3,
                max_qty,
                min(unplaced_qty, max_qty),
                unplaced_qty % max_qty if max_qty > 0 else 0,
            )
            if qty > 0
        )
        if max_qty <= 16:
            values.update(range(1, max_qty + 1))
        else:
            values.update(range(1, min(10, max_qty) + 1))
            values.update(qty for qty in range(15, max_qty + 1, 5))
        return sorted(qty for qty in values if 0 < qty <= max_qty)

    def _solve_master_by_diving(
        self,
        Model,
        quicksum,
        final_lp_bound: float | None,
        best_start_source: str,
        best_start_selected: Counter[int],
        best_start_unplaced: Counter[str],
        stage0_unplaced_cap: int | None = None,
    ) -> tuple[Counter[int], Counter[str], dict]:
        solve_start = perf_counter()
        max_steps = max(1, int(getattr(self.config, "diving_max_steps", 200) or 200))
        tolerance = max(1e-9, float(getattr(self.config, "diving_fractional_tolerance", 1e-5) or 1e-5))
        fix_batch_size = max(1, int(getattr(self.config, "diving_fix_batch_size", 8) or 8))
        max_no_improve_steps = max(0, int(getattr(self.config, "diving_max_no_improve_steps", 0) or 0))
        current_fix_batch_size = fix_batch_size
        price_during_diving = bool(getattr(self.config, "diving_price_columns", False))
        stop_on_lp_unplaced = bool(getattr(self.config, "diving_stop_on_lp_unplaced", False))
        fixed_columns: dict[int, int] = {}
        last_zero_unplaced_fixed_columns: dict[int, int] = {}
        rollback_fixed_columns: dict[int, int] | None = None
        best_source = best_start_source
        best_selected = Counter(best_start_selected)
        best_unplaced = Counter(best_start_unplaced)
        diving_iterations: list[dict] = []
        last_improvement_step = -1
        last_lp_objective: float | None = None
        status = "diving_step_limit"
        min_lp_time = 2.0
        if (
            bool(getattr(self.config, "diving_skip_when_lp_unplaced", False))
            and stage0_unplaced_cap is None
            and sum(best_start_unplaced.values()) > 0
        ):
            objective = self._selected_solution_energy(best_selected, best_unplaced)
            return best_selected, best_unplaced, {
                "master_algorithm": "column_generation_diving",
                "master_status": "diving_skipped_lp_unplaced",
                "master_objective": objective,
                "master_primal_bound": objective,
                "master_dual_bound": final_lp_bound,
                "master_mip_gap": self._relative_gap(objective, final_lp_bound),
                "master_mip_gap_is_reliable": False,
                "restricted_master_lp_gap": self._relative_gap(objective, final_lp_bound),
                "master_solve_seconds": round(perf_counter() - solve_start, 3),
                "master_mip_start_added": False,
                "master_mip_start_columns": sum(1 for qty in self._master_start_selected.values() if qty > 0),
                "master_mip_start_unplaced_boxes": sum(self._master_start_unplaced.values()),
                "master_stage0_unplaced_cap": stage0_unplaced_cap,
                "diving_incumbent_source": best_source,
                "diving_max_steps": max_steps,
                "diving_fractional_tolerance": tolerance,
                "diving_fix_batch_size": fix_batch_size,
                "diving_max_no_improve_steps": max_no_improve_steps,
                "diving_final_fix_batch_size": current_fix_batch_size,
                "diving_price_columns": price_during_diving,
                "diving_stop_on_lp_unplaced": stop_on_lp_unplaced,
                "diving_skip_when_lp_unplaced": True,
                "diving_step_count": 0,
                "diving_fixed_one_columns": 0,
                "diving_fixed_zero_columns": 0,
                "diving_final_repair_added_columns": 0,
                "diving_final_repair_total_added_columns": 0,
                "diving_final_repair_unplaced_boxes": 0,
                "diving_chosen_final_repair_unplaced_boxes": 0,
                "diving_final_repair_candidates": [],
                "diving_improvement_rounds_requested": 0,
                "diving_improvement_time_limit": 0.0,
                "diving_improvement_max_groups": 0,
                "diving_improvement_max_no_improve_rounds": 0,
                "diving_improvement_rounds_run": 0,
                "diving_improvement_improvements": 0,
                "diving_improvement_incumbent_source": "",
                "diving_improvement_iterations": [],
                "diving_improvement_stop_reason": "skipped_lp_unplaced",
                "diving_iterations": [],
            }

        for step in range(max_steps):
            elapsed = perf_counter() - solve_start
            remaining_time = float(self.config.mip_time_limit) - elapsed
            if self.config.mip_time_limit > 0 and remaining_time <= min_lp_time:
                status = "diving_time_limit"
                break

            if self.config.verbose:
                print(
                    "[column-generation-scip] diving LP "
                    f"step={step} columns={len(self._columns)} "
                    f"fixed1={sum(1 for value in fixed_columns.values() if value == 1)} "
                    f"fixed0={sum(1 for value in fixed_columns.values() if value == 0)}",
                    flush=True,
                )
            lp_model, lp_vars, lp_constraints = self._build_restricted_master(
                Model,
                quicksum,
                relax=True,
                fixed_column_values=fixed_columns,
                unplaced_cap=stage0_unplaced_cap,
            )
            if self.config.mip_time_limit > 0:
                self._set_scip_param(lp_model, "limits/time", max(1.0, min(30.0, remaining_time)))
            lp_model.optimize()
            lp_status = self._scip_status_name(lp_model)
            if lp_status != "optimal":
                self._free_scip_model(lp_model)
                status = f"diving_lp_{lp_status}"
                break

            lp_objective = self._scip_objective_value(lp_model)
            last_lp_objective = lp_objective
            lp_unplaced_float = self._scip_unplaced_float_values(lp_model, lp_vars)
            lp_unplaced = Counter(
                {
                    group_id: int(round(value))
                    for group_id, value in lp_unplaced_float.items()
                    if value > tolerance
                }
            )
            lp_unplaced_boxes = sum(lp_unplaced.values())
            lp_column_values = self._scip_column_values(lp_model, lp_vars)
            fractional_values = self._fractional_column_values(lp_column_values, fixed_columns, tolerance)
            before_repair_columns = len(self._columns)
            lp_repair_selected, lp_repair_unplaced = self._repair_from_column_priority(
                lp_column_values,
                allow_new_columns=False,
            )
            repair_added_columns = len(self._columns) - before_repair_columns
            if self._solution_rank(lp_repair_selected, lp_repair_unplaced) < self._solution_rank(
                best_selected,
                best_unplaced,
            ):
                best_source = f"diving_lp_repair_step_{step}"
                best_selected = lp_repair_selected
                best_unplaced = lp_repair_unplaced
                last_improvement_step = step

            pricing_stats: dict = {}
            new_count = 0
            if price_during_diving and not self.config.full_column_pool:
                pricing_stats = self._price_columns(
                    lp_model,
                    lp_constraints,
                    self.config.max_iterations + step,
                    lp_unplaced,
                    lp_column_values,
                )
                new_count = int(pricing_stats.get("new_columns", 0) or 0)
            else:
                pricing_stats = {
                    "new_columns": 0,
                    "pricing_mode": "diving_pricing_disabled",
                }

            record = {
                "step": step,
                "columns": len(self._columns),
                "lp_objective": lp_objective,
                "lp_unplaced_boxes": lp_unplaced_boxes,
                "fixed_one_columns": sum(1 for value in fixed_columns.values() if value == 1),
                "fixed_zero_columns": sum(1 for value in fixed_columns.values() if value == 0),
                "fractional_column_count": len(fractional_values),
                "largest_fractional_column_value": round(max(fractional_values.values()), 6) if fractional_values else None,
                "lp_guided_repair_unplaced_boxes": sum(lp_repair_unplaced.values()),
                "lp_guided_repair_added_columns": repair_added_columns,
                **pricing_stats,
            }
            self._free_scip_model(lp_model)

            if lp_unplaced_boxes > 0:
                rollback_fixed_columns = dict(last_zero_unplaced_fixed_columns)
                record["lp_unplaced_policy"] = "stop_or_reduce" if stop_on_lp_unplaced else "continue"
                record["rollback_fixed_one_columns"] = sum(1 for value in rollback_fixed_columns.values() if value == 1)
                record["rollback_fixed_zero_columns"] = sum(1 for value in rollback_fixed_columns.values() if value == 0)
                if stop_on_lp_unplaced:
                    record["decision"] = "stop_on_lp_unplaced"
                    record["previous_fix_batch_size"] = current_fix_batch_size
                    fixed_columns = dict(rollback_fixed_columns)
                    if current_fix_batch_size > 1:
                        current_fix_batch_size = max(1, current_fix_batch_size // 2)
                        record["decision"] = "rollback_reduce_batch_after_lp_unplaced"
                        record["next_fix_batch_size"] = current_fix_batch_size
                        diving_iterations.append(record)
                        status = "diving_batch_reduced_after_lp_unplaced"
                        continue
                    diving_iterations.append(record)
                    status = "diving_lp_unplaced_stop"
                    break
            else:
                last_zero_unplaced_fixed_columns = dict(fixed_columns)

            if new_count > 0:
                record["decision"] = "reprice_after_new_columns"
                diving_iterations.append(record)
                if self.config.verbose:
                    print(
                        f"[column-generation-scip] diving step={step} lp={lp_objective:.3f} "
                        f"new={new_count}",
                        flush=True,
                    )
                continue

            if self._lp_solution_is_integral(lp_column_values, lp_unplaced_float, tolerance):
                lp_selected = Counter(
                    {
                        idx: 1
                        for idx, value in lp_column_values.items()
                        if value >= 1.0 - tolerance
                    }
                )
                for idx, value in fixed_columns.items():
                    if value == 1:
                        lp_selected[idx] = 1
                integral_selected, integral_unplaced = self._repair_selected_solution(lp_selected)
                if self._solution_rank(integral_selected, integral_unplaced) < self._solution_rank(
                    best_selected,
                    best_unplaced,
                ):
                    best_source = f"diving_integral_lp_step_{step}"
                    best_selected = integral_selected
                    best_unplaced = integral_unplaced
                    last_improvement_step = step
                record["decision"] = "integral_lp_solution"
                diving_iterations.append(record)
                status = "diving_integral_lp"
                break

            decisions = self._choose_diving_fixes(fractional_values, fixed_columns, current_fix_batch_size)
            if not decisions:
                record["decision"] = "no_fractional_column"
                diving_iterations.append(record)
                status = "diving_no_fractional_column"
                break
            for idx, fixed_value, _reason in decisions:
                fixed_columns[idx] = fixed_value
            first_idx, first_fixed_value, first_reason = decisions[0]
            record.update(
                {
                    "decision": "fix_columns_batch",
                    "decision_count": len(decisions),
                    "decision_fix_batch_size": current_fix_batch_size,
                    "decision_column": first_idx,
                    "decision_value": first_fixed_value,
                    "decision_reason": first_reason,
                    "decision_lp_value": round(float(lp_column_values.get(first_idx, 0.0)), 6),
                    "decision_reason_counts": dict(Counter(reason for _idx, _fixed_value, reason in decisions)),
                    "decision_fixed_one_count": sum(1 for _idx, fixed_value, _reason in decisions if fixed_value == 1),
                    "decision_fixed_zero_count": sum(1 for _idx, fixed_value, _reason in decisions if fixed_value == 0),
                }
            )
            diving_iterations.append(record)
            if max_no_improve_steps > 0 and step - last_improvement_step >= max_no_improve_steps:
                record["early_stop_reason"] = "no_incumbent_improvement"
                record["diving_steps_since_improvement"] = step - last_improvement_step
                status = "diving_no_incumbent_improvement"
                break
            if self.config.verbose:
                print(
                    f"[column-generation-scip] diving step={step} lp={lp_objective:.3f} "
                    f"fix_count={len(decisions)} first_col={first_idx} "
                    f"value={first_fixed_value} reason={first_reason}",
                    flush=True,
                )

        final_repair_added_columns = 0
        final_repair_unplaced_boxes: int | None = None
        chosen_final_repair_added_columns = 0
        chosen_final_repair_unplaced_boxes: int | None = None
        final_repair_candidate_stats: list[dict] = []
        final_repair_candidates = [("diving_fixed_column_repair", fixed_columns)]
        if rollback_fixed_columns is not None and rollback_fixed_columns != fixed_columns:
            final_repair_candidates.append(("diving_rollback_fixed_column_repair", rollback_fixed_columns))
        for source, candidate_fixed_columns in final_repair_candidates:
            fixed_selected = Counter({idx: 1 for idx, value in candidate_fixed_columns.items() if value == 1})
            before_final_repair_columns = len(self._columns)
            fixed_repair_selected, fixed_repair_unplaced = self._repair_selected_solution(fixed_selected)
            added_columns = len(self._columns) - before_final_repair_columns
            candidate_unplaced_boxes = sum(fixed_repair_unplaced.values())
            final_repair_added_columns += added_columns
            final_repair_unplaced_boxes = (
                candidate_unplaced_boxes
                if final_repair_unplaced_boxes is None
                else min(final_repair_unplaced_boxes, candidate_unplaced_boxes)
            )
            candidate_objective = self._selected_solution_energy(fixed_repair_selected, fixed_repair_unplaced)
            final_repair_candidate_stats.append(
                {
                    "source": source,
                    "fixed_one_columns": sum(1 for value in candidate_fixed_columns.values() if value == 1),
                    "fixed_zero_columns": sum(1 for value in candidate_fixed_columns.values() if value == 0),
                    "added_columns": added_columns,
                    "selected_columns": sum(1 for qty in fixed_repair_selected.values() if qty > 0),
                    "unplaced_boxes": candidate_unplaced_boxes,
                    "objective": candidate_objective,
                }
            )
            if self._solution_rank(fixed_repair_selected, fixed_repair_unplaced) < self._solution_rank(
                best_selected,
                best_unplaced,
            ):
                best_source = source
                best_selected = fixed_repair_selected
                best_unplaced = fixed_repair_unplaced
                chosen_final_repair_added_columns = added_columns
                chosen_final_repair_unplaced_boxes = candidate_unplaced_boxes

        improvement_selected, improvement_unplaced, improvement_stats = self._run_diving_improvement_rounds(
            Model,
            quicksum,
            best_selected,
            best_unplaced,
            solve_start,
            stage0_unplaced_cap,
        )
        if self._solution_rank(improvement_selected, improvement_unplaced) < self._solution_rank(
            best_selected,
            best_unplaced,
        ):
            best_source = str(improvement_stats.get("diving_improvement_incumbent_source", best_source))
            best_selected = improvement_selected
            best_unplaced = improvement_unplaced
            if status in {"diving_no_incumbent_improvement", "diving_step_limit", "diving_time_limit"}:
                status = "diving_improvement_found"

        objective = self._selected_solution_energy(best_selected, best_unplaced)
        dual_bound = final_lp_bound if final_lp_bound is not None else last_lp_objective
        return best_selected, best_unplaced, {
            "master_algorithm": "column_generation_diving",
            "master_status": status,
            "master_objective": objective,
            "master_primal_bound": objective,
            "master_dual_bound": dual_bound,
            "master_mip_gap": self._relative_gap(objective, dual_bound),
            "master_mip_gap_is_reliable": False,
            "restricted_master_lp_gap": self._relative_gap(objective, final_lp_bound),
            "master_solve_seconds": round(perf_counter() - solve_start, 3),
            "master_mip_start_added": False,
            "master_mip_start_columns": sum(1 for qty in self._master_start_selected.values() if qty > 0),
            "master_mip_start_unplaced_boxes": sum(self._master_start_unplaced.values()),
            "master_stage0_unplaced_cap": stage0_unplaced_cap,
            "diving_incumbent_source": best_source,
            "diving_max_steps": max_steps,
            "diving_fractional_tolerance": tolerance,
            "diving_fix_batch_size": fix_batch_size,
            "diving_max_no_improve_steps": max_no_improve_steps,
            "diving_final_fix_batch_size": current_fix_batch_size,
            "diving_price_columns": price_during_diving,
            "diving_stop_on_lp_unplaced": stop_on_lp_unplaced,
            "diving_skip_when_lp_unplaced": bool(getattr(self.config, "diving_skip_when_lp_unplaced", False)),
            "diving_step_count": len(diving_iterations),
            "diving_fixed_one_columns": sum(1 for value in fixed_columns.values() if value == 1),
            "diving_fixed_zero_columns": sum(1 for value in fixed_columns.values() if value == 0),
            "diving_final_repair_added_columns": chosen_final_repair_added_columns,
            "diving_final_repair_total_added_columns": final_repair_added_columns,
            "diving_final_repair_unplaced_boxes": final_repair_unplaced_boxes or 0,
            "diving_chosen_final_repair_unplaced_boxes": chosen_final_repair_unplaced_boxes or 0,
            "diving_final_repair_candidates": final_repair_candidate_stats,
            **improvement_stats,
            "diving_iterations": diving_iterations,
        }

    def _run_diving_improvement_rounds(
        self,
        Model,
        quicksum,
        incumbent_selected: Counter[int],
        incumbent_unplaced: Counter[str],
        solve_start: float,
        stage0_unplaced_cap: int | None = None,
    ) -> tuple[Counter[int], Counter[str], dict]:
        max_rounds = max(0, int(getattr(self.config, "diving_improvement_rounds", 0) or 0))
        per_round_time = max(0.0, float(getattr(self.config, "diving_improvement_time_limit", 0.0) or 0.0))
        max_groups = max(1, int(getattr(self.config, "diving_improvement_max_groups", 1) or 1))
        max_no_improve_rounds = max(
            0,
            int(getattr(self.config, "diving_improvement_max_no_improve_rounds", 0) or 0),
        )
        stats = {
            "diving_improvement_rounds_requested": max_rounds,
            "diving_improvement_time_limit": per_round_time,
            "diving_improvement_max_groups": max_groups,
            "diving_improvement_max_no_improve_rounds": max_no_improve_rounds,
            "diving_improvement_rounds_run": 0,
            "diving_improvement_improvements": 0,
            "diving_improvement_incumbent_source": "",
            "diving_improvement_iterations": [],
        }
        if max_rounds <= 0 or per_round_time <= 0:
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats

        best_selected = Counter(incumbent_selected)
        best_unplaced = Counter(incumbent_unplaced)
        best_rank = self._solution_rank(best_selected, best_unplaced)
        neighborhoods = self._diving_improvement_neighborhoods(best_selected, max_rounds, max_groups)
        stats["diving_improvement_candidate_neighborhoods"] = len(neighborhoods)
        if not neighborhoods:
            return best_selected, best_unplaced, stats

        min_round_time = 1.0
        no_improve_rounds = 0
        for round_no, release_group_ids in enumerate(neighborhoods[:max_rounds]):
            elapsed = perf_counter() - solve_start
            remaining_total = float(self.config.mip_time_limit) - elapsed
            if self.config.mip_time_limit > 0 and remaining_total <= min_round_time:
                break
            round_time = per_round_time
            if self.config.mip_time_limit > 0:
                round_time = max(min_round_time, min(per_round_time, remaining_total))
            fixed_values = self._fixed_selected_columns_for_released_groups(best_selected, release_group_ids)
            if len(fixed_values) >= len(self._columns):
                continue

            if self.config.verbose:
                print(
                    "[column-generation-scip] diving improvement "
                    f"round={round_no} release_groups={len(release_group_ids)} "
                    f"fixed={len(fixed_values)} time_limit={round_time:.1f}s",
                    flush=True,
                )
            before_rank = best_rank
            model, vars_by_name, _constraints = self._build_restricted_master(
                Model,
                quicksum,
                relax=False,
                fixed_column_values=fixed_values,
                unplaced_cap=stage0_unplaced_cap,
            )
            self._set_scip_param(model, "limits/time", float(round_time))
            self._set_scip_param(model, "limits/gap", float(self.config.mip_gap))
            previous_start_selected = self._master_start_selected
            previous_start_unplaced = self._master_start_unplaced
            self._master_start_selected = best_selected
            self._master_start_unplaced = best_unplaced
            mip_start_added = self._add_scip_mip_start(model, vars_by_name)
            self._master_start_selected = previous_start_selected
            self._master_start_unplaced = previous_start_unplaced
            model.optimize()
            status = self._scip_status_name(model)
            has_solution = self._scip_solution_count(model) > 0
            candidate_selected = best_selected
            candidate_unplaced = best_unplaced
            candidate_objective = self._selected_solution_energy(candidate_selected, candidate_unplaced)
            if has_solution:
                candidate_selected, candidate_unplaced = self._solution_from_scip_vars(model, vars_by_name)
                candidate_objective = self._selected_solution_energy(candidate_selected, candidate_unplaced)
                candidate_rank = self._solution_rank(candidate_selected, candidate_unplaced)
                if candidate_rank < best_rank:
                    best_selected = candidate_selected
                    best_unplaced = candidate_unplaced
                    best_rank = candidate_rank
                    stats["diving_improvement_improvements"] += 1
                    stats["diving_improvement_incumbent_source"] = f"diving_improvement_round_{round_no}"
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
            else:
                no_improve_rounds += 1
            stats["diving_improvement_iterations"].append(
                {
                    "round": round_no,
                    "status": status,
                    "has_solution": has_solution,
                    "mip_start_added": mip_start_added,
                    "released_group_count": len(release_group_ids),
                    "fixed_column_count": len(fixed_values),
                    "objective": candidate_objective,
                    "unplaced_boxes": sum(candidate_unplaced.values()),
                    "improved": best_rank < before_rank,
                }
            )
            stats["diving_improvement_rounds_run"] += 1
            self._free_scip_model(model)
            if max_no_improve_rounds > 0 and no_improve_rounds >= max_no_improve_rounds:
                stats["diving_improvement_stop_reason"] = "no_incumbent_improvement"
                break
        return best_selected, best_unplaced, stats

    def _fixed_columns_for_released_groups(
        self,
        selected: Counter[int],
        release_group_ids: set[str],
    ) -> dict[int, int]:
        fixed: dict[int, int] = {}
        for idx, col in enumerate(self._columns):
            if col.group_id in release_group_ids:
                continue
            fixed[idx] = 1 if selected.get(idx, 0) > 0 else 0
        return fixed

    def _solution_from_scip_vars(self, model, vars_by_name) -> tuple[Counter[int], Counter[str]]:
        selected = Counter(
            {
                idx: 1
                for idx, var in vars_by_name["column"].items()
                if self._scip_value(model, var) > 0.5
            }
        )
        unplaced = Counter(
            {
                group_id: int(round(self._scip_value(model, var)))
                for group_id, var in vars_by_name["unplaced"].items()
                if self._scip_value(model, var) > 1e-6
            }
        )
        return selected, unplaced

    def _diving_improvement_neighborhoods(
        self,
        selected: Counter[int],
        max_rounds: int,
        max_groups: int,
    ) -> list[set[str]]:
        by_coarse: defaultdict[tuple[str, str, str, str], dict] = defaultdict(
            lambda: {"groups": Counter(), "areas": Counter(), "bays": set(), "blocks": set()}
        )
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            qty = int(col.quantity) * int(chosen)
            data = by_coarse[col.coarse_key]
            data["groups"][col.group_id] += qty
            data["areas"][col.area_no] += qty
            data["bays"].add(col.bay_key)
            if col.block_id:
                data["blocks"].add(col.block_id)

        ranked: list[tuple[float, tuple[str, str, str, str], set[str]]] = []
        for coarse_key, data in by_coarse.items():
            groups = data["groups"]
            if not groups:
                continue
            area_count = sum(1 for qty in data["areas"].values() if qty > 0)
            bay_count = len(data["bays"])
            block_count = len(data["blocks"])
            demand = max(1, int(self.coarse_demand.get(coarse_key, sum(groups.values()))))
            if self._prefers_concentrated_coarse_key(coarse_key):
                score = 1000.0 * max(0, area_count - 1) + 20.0 * max(0, bay_count - 1)
            else:
                tiny_area_boxes = sum(1 for qty in data["areas"].values() if 0 < qty < self.config.medium_large_group_min_area_boxes)
                score = 600.0 * tiny_area_boxes + 10.0 * max(0, bay_count - max(1, demand // 20))
            score += 40.0 * max(0, block_count - 1) + 0.01 * demand
            release_groups = {
                group_id
                for group_id, _qty in groups.most_common(max_groups)
            }
            if release_groups:
                ranked.append((score, coarse_key, release_groups))

        neighborhoods: list[set[str]] = []
        seen: set[tuple[str, ...]] = set()
        for _score, _coarse_key, group_ids in sorted(ranked, reverse=True):
            signature = tuple(sorted(group_ids))
            if signature in seen:
                continue
            seen.add(signature)
            neighborhoods.append(group_ids)
            if len(neighborhoods) >= max_rounds:
                break
        return neighborhoods

    def _choose_diving_fixes(
        self,
        fractional_column_values: dict[int, float],
        fixed_columns: dict[int, int],
        limit: int,
    ) -> list[tuple[int, int, str]]:
        fixed_selected = Counter({idx: 1 for idx, value in fixed_columns.items() if value == 1})
        _repaired, state, placed = self._selection_state(fixed_selected)
        candidates = []
        for idx, value in fractional_column_values.items():
            if idx in fixed_columns or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            group = self.groups_by_id.get(col.group_id)
            if group is None:
                continue
            remaining = int(self.group_demand.get(col.group_id, 0)) - int(placed.get(col.group_id, 0))
            repair_score = float("inf")
            if remaining > 0 and col.quantity <= remaining and float(value) > 0.5:
                base_cost = col.intrinsic_cost
                repair_score = self._repair_column_score(group, col.bay_key, col.quantity, base_cost, state)
            candidates.append((0 if float(value) > 0.5 else 1, -float(value), repair_score, -int(col.quantity), idx))
        decisions: list[tuple[int, int, str]] = []
        for _tier, _neg_value, _repair_score, _neg_qty, idx in sorted(candidates):
            if len(decisions) >= limit:
                break
            col = self._columns[idx]
            remaining = int(self.group_demand.get(col.group_id, 0)) - int(placed.get(col.group_id, 0))
            if remaining <= 0 or col.quantity > remaining:
                decisions.append((idx, 0, "group_already_satisfied"))
                continue
            value = float(fractional_column_values.get(idx, 0.0))
            if value <= 0.5:
                decisions.append((idx, 0, "fractional_round_down"))
                continue
            if not self._column_fits_state(col, state, remaining):
                decisions.append((idx, 0, "infeasible_with_fixed_columns"))
                continue
            decisions.append((idx, 1, "fractional_round_up"))
            self._apply_column_to_state(col, state)
            placed[col.group_id] += col.quantity
        return decisions

    @staticmethod
    def _fractional_column_values(
        column_values: dict[int, float],
        fixed_columns: dict[int, int],
        tolerance: float,
    ) -> dict[int, float]:
        return {
            idx: float(value)
            for idx, value in column_values.items()
            if idx not in fixed_columns
            and tolerance < float(value) < 1.0 - tolerance
        }

    @staticmethod
    def _lp_solution_is_integral(
        column_values: dict[int, float],
        unplaced_values: dict[str, float],
        tolerance: float,
    ) -> bool:
        for value in column_values.values():
            if abs(float(value) - round(float(value))) > tolerance:
                return False
        for value in unplaced_values.values():
            if abs(float(value) - round(float(value))) > tolerance:
                return False
        return True

    def _scip_unplaced_values(self, model, lp_vars) -> Counter[str]:
        return Counter(
            {
                group_id: int(round(self._scip_value(model, var)))
                for group_id, var in lp_vars["unplaced"].items()
                if self._scip_value(model, var) > 1e-6
            }
        )

    def _scip_unplaced_float_values(self, model, lp_vars) -> dict[str, float]:
        return {
            group_id: self._scip_value(model, var)
            for group_id, var in lp_vars["unplaced"].items()
            if self._scip_value(model, var) > 1e-9
        }

    def _scip_column_values(self, model, lp_vars) -> dict[int, float]:
        return {
            idx: self._scip_value(model, var)
            for idx, var in lp_vars["column"].items()
            if self._scip_value(model, var) > 1e-6
        }

    def _add_scip_mip_start(self, model, mip_vars) -> bool:
        if not self._master_start_selected and not self._master_start_unplaced:
            return False
        try:
            creator = getattr(model, "createPartialSol", None) or getattr(model, "createSol")
            sol = creator()
            for idx, var in mip_vars["column"].items():
                model.setSolVal(sol, var, 1.0 if self._master_start_selected.get(idx, 0) > 0 else 0.0)
            for group_id, var in mip_vars["unplaced"].items():
                model.setSolVal(sol, var, float(self._master_start_unplaced.get(group_id, 0)))
            try:
                return bool(model.addSol(sol, free=True))
            except Exception:
                return bool(
                    model.trySol(
                        sol,
                        printreason=False,
                        completely=False,
                        checkbounds=True,
                        checkintegrality=True,
                        checklprows=True,
                        free=True,
                    )
                )
        except Exception:
            return False

    def _configure_scip_output(self, model) -> None:
        if not self.config.verbose:
            try:
                model.hideOutput()
                return
            except Exception:
                pass
            self._set_scip_param(model, "display/verblevel", 0)

    @staticmethod
    def _set_scip_param(model, name: str, value: object) -> None:
        setters = (getattr(model, "setParam", None), getattr(model, "setRealParam", None), getattr(model, "setIntParam", None))
        last_error: Exception | None = None
        for setter in setters:
            if setter is None:
                continue
            try:
                setter(name, value)
                return
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error

    @staticmethod
    def _scip_status_name(model) -> str:
        return str(model.getStatus()).lower()

    @staticmethod
    def _scip_solution_count(model) -> int:
        try:
            return int(model.getNSols())
        except Exception:
            try:
                return 1 if model.getBestSol() is not None else 0
            except Exception:
                return 0

    @staticmethod
    def _scip_objective_value(model) -> float:
        try:
            return float(model.getObjVal())
        except Exception:
            return float("nan")

    @staticmethod
    def _scip_gap(model) -> float:
        try:
            return float(model.getGap())
        except Exception:
            return 0.0

    @staticmethod
    def _scip_primal_bound(model) -> float:
        for method_name in ("getPrimalbound", "getPrimalBound"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                return float(method())
            except Exception:
                continue
        return float("nan")

    @staticmethod
    def _scip_dual_bound(model) -> float:
        for method_name in ("getDualbound", "getDualBound"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                return float(method())
            except Exception:
                continue
        return float("nan")

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
    def _scip_value(model, var) -> float:
        return float(model.getVal(var))

    @staticmethod
    def _scip_dual(model, constr) -> float:
        for method_name in ("getDualsolLinear", "getDualsol"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                return float(method(constr))
            except Exception:
                continue
        return 0.0

    @staticmethod
    def _free_scip_model(model) -> None:
        for method_name in ("freeTransform", "freeProb"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                method()
                return
            except Exception:
                continue

    def _build_restricted_master(
        self,
        Model,
        quicksum,
        relax: bool,
        fixed_column_values: dict[int, int] | None = None,
        unplaced_cap: int | None = None,
        objective_mode: str = "full",
    ):
        model = Model("yard_small_plan_column_generation_scip")
        self._configure_scip_output(model)
        try:
            model.setMinimize()
        except Exception:
            pass
        column_vtype = "C" if relax else "B"
        fixed_column_values = fixed_column_values or {}
        columns = {
            idx: model.addVar(
                lb=float(fixed_column_values[idx]) if idx in fixed_column_values else 0.0,
                ub=float(fixed_column_values[idx]) if idx in fixed_column_values else 1.0,
                vtype=column_vtype,
                obj=0.0 if objective_mode == "min_unplaced" else col.intrinsic_cost + self.config.small_plan_group_bay_split_penalty,
                name=f"col_{idx}",
            )
            for idx, col in enumerate(self._columns)
        }
        unplaced = {
            group.group_id: model.addVar(
                lb=0.0,
                ub=group.demand,
                vtype="C" if relax else "I",
                obj=1.0 if objective_mode == "min_unplaced" else self.config.unplaced_penalty,
                name=f"unplaced_{group.group_id}",
            )
            for group in self.groups
        }

        group_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_capacity_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_size_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_port_size_cols: defaultdict[tuple[str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        row_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        row_size_capacity_cols: defaultdict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        row_attr_choice_cols: defaultdict[tuple[str, str, str, str], list[int]] = defaultdict(list)
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
                for attr in self._bay_no_mix_attrs(col.voyage_id):
                    bay_attr_choice_cols[(footprint_key, attr, self._column_attr_value(col, attr))].append(idx)
            for footprint_key, row_no, qty in col.row_allocation:
                row_capacity_cols[(footprint_key, row_no)].append((idx, int(qty)))
                row_size_capacity_cols[(footprint_key, row_no, col.size)].append((idx, int(qty)))
                for attr in self._row_no_mix_attrs(col.voyage_id):
                    row_attr_choice_cols[(footprint_key, row_no, attr, self._column_attr_value(col, attr))].append(idx)
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
            expr = quicksum(col.quantity * columns[idx] for idx, col in group_cols.get(group.group_id, []))
            group_cover[group.group_id] = model.addCons(expr + unplaced[group.group_id] == group.demand, name=f"cover_{group.group_id}")

        required_area_limit = {}
        for voyage_id, areas in sorted(getattr(self.problem, "user_voyage_area_requirements", {}).items()):
            for area_no in sorted(areas):
                indices = voyage_area_cols.get((voyage_id, area_no), [])
                if not indices:
                    continue
                required_area_limit[(voyage_id, area_no)] = model.addCons(
                    quicksum(self._columns[idx].quantity * columns[idx] for idx in indices) >= 1.0,
                    name=f"user_required_area_{len(required_area_limit)}",
                )

        required_group_bay_limit = {}
        for group_id, bay_keys in sorted(getattr(self.problem, "user_group_bay_requirements", {}).items()):
            for bay_key in sorted(bay_keys):
                indices = group_bay_cols.get((group_id, bay_key), [])
                required_group_bay_limit[(group_id, bay_key)] = model.addCons(
                    quicksum(self._columns[idx].quantity * columns[idx] for idx in indices) >= 1.0,
                    name=f"user_required_bay_{len(required_group_bay_limit)}",
                )

        bay_capacity_limit = {}
        for bay_key, items in bay_capacity_cols.items():
            bay_capacity_limit[bay_key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items) <= self.bays[bay_key].physical_capacity,
                name=f"bay_cap_{bay_key}",
            )
        bay_size_limit = {}
        for key, items in bay_size_capacity_cols.items():
            bay_key, size = key
            bay_size_limit[key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items) <= self.bays[bay_key].cap_by_size.get(size, 0),
                name=f"bay_size_{bay_key}_{size}",
            )
        row_capacity_limit = {}
        for key, items in row_capacity_cols.items():
            bay_key, row_no = key
            cap = int(self.bays[bay_key].row_physical_capacity.get(row_no, self.bays[bay_key].physical_capacity))
            row_capacity_limit[key] = model.addCons(
                quicksum(qty * columns[idx] for idx, qty in items) <= cap,
                name=f"row_cap_{bay_key}_{row_no}",
            )
        row_size_limit = {}
        for key, items in row_size_capacity_cols.items():
            bay_key, row_no, size = key
            cap = int(self.bays[bay_key].row_cap_by_size.get(size, {}).get(row_no, self.bays[bay_key].cap_by_size.get(size, 0)))
            row_size_limit[key] = model.addCons(
                quicksum(qty * columns[idx] for idx, qty in items) <= cap,
                name=f"row_size_{bay_key}_{row_no}_{size}",
            )
        bay_port_stack_link = {}
        bay_port_stack_limit = {}
        bay_stack_total_limit = {}
        bay_stack_vars = {}
        stack_vtype = "C" if relax else "I"
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
            load = quicksum(col.quantity * columns[idx] for idx, col in items)
            bay_port_stack_link[key] = model.addCons(load <= unit_capacity * stack_var, name=f"stack_load_{bay_key}_{port}_{size}")
            bay_port_stack_limit[key] = model.addCons(stack_var <= stack_count, name=f"stack_port_cap_{bay_key}_{port}_{size}")
        stack_vars_by_bay_size: defaultdict[tuple[str, str], list] = defaultdict(list)
        for (bay_key, _port, size), stack_var in bay_stack_vars.items():
            stack_vars_by_bay_size[(bay_key, size)].append(stack_var)
        for (bay_key, size), stack_vars in stack_vars_by_bay_size.items():
            stack_count = self._stack_count_for_bay_size(bay_key, size)
            if stack_count > 0:
                bay_stack_total_limit[(bay_key, size)] = model.addCons(
                    quicksum(stack_vars) <= stack_count,
                    name=f"stack_total_{bay_key}_{size}",
                )
        group_bay_limit = {
            key: model.addCons(quicksum(columns[idx] for idx in indices) <= 1.0, name=f"group_bay_{key[0]}_{key[1]}")
            for key, indices in group_bay_cols.items()
        }
        quota_limit = {}
        medium_plan_quota_limit = {}
        for key, items in area_size_cols.items():
            cap = int(self.quota_by_key.get(key, 0))
            if cap <= 0:
                continue
            quota_limit[key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items) <= cap,
                name=f"big_quota_{len(quota_limit)}",
            )
        if self.config.medium_plan_quota is not None:
            medium_plan_quota = Counter(self.config.medium_plan_quota)
            for key, items in coarse_area_cols.items():
                cap = int(medium_plan_quota.get(key, 0))
                medium_plan_quota_limit[key] = model.addCons(
                    quicksum(col.quantity * columns[idx] for idx, col in items) <= cap,
                    name=f"medium_quota_{len(medium_plan_quota_limit)}",
                )
        if self.config.medium_plan_bay_quota is not None:
            medium_plan_bay_quota = Counter(self.config.medium_plan_bay_quota)
            for key, indices in coarse_area_bay_cols.items():
                cap = int(medium_plan_bay_quota.get(key, 0))
                medium_plan_quota_limit[key] = model.addCons(
                    quicksum(self._columns[idx].quantity * columns[idx] for idx in indices) <= cap,
                    name=f"medium_bay_quota_{len(medium_plan_quota_limit)}",
                )

        seed_unplaced_limit = None
        if not relax and (self._master_seed_selected or self._master_seed_unplaced):
            seed_unplaced_limit = model.addCons(
                quicksum(unplaced.values()) <= sum(self._master_seed_unplaced.values()),
                name="seed_unplaced_cap",
            )
        stage0_unplaced_limit = None
        if unplaced_cap is not None:
            stage0_unplaced_limit = model.addCons(
                quicksum(unplaced.values()) <= int(unplaced_cap),
                name="stage0_unplaced_cap",
            )

        relaxed_objective_constraints = {}
        if objective_mode == "min_unplaced":
            pass
        elif relax:
            relaxed_objective_constraints = self._add_relaxed_master_objectives(
                quicksum,
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
                quicksum,
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
                row_attr_choice_cols,
            )
        return model, {"column": columns, "unplaced": unplaced}, {
            "group_cover": group_cover,
            "bay_capacity_limit": bay_capacity_limit,
            "bay_size_limit": bay_size_limit,
            "row_capacity_limit": row_capacity_limit,
            "row_size_limit": row_size_limit,
            "bay_port_stack_link": bay_port_stack_link,
            "bay_port_stack_limit": bay_port_stack_limit,
            "bay_stack_total_limit": bay_stack_total_limit,
            "group_bay_limit": group_bay_limit,
            "quota_limit": quota_limit,
            "medium_plan_quota_limit": medium_plan_quota_limit,
            "required_area_limit": required_area_limit,
            "required_group_bay_limit": required_group_bay_limit,
            "seed_unplaced_limit": seed_unplaced_limit,
            "stage0_unplaced_limit": stage0_unplaced_limit,
            **relaxed_objective_constraints,
        }

    def _add_relaxed_master_objectives(
        self,
        quicksum,
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
            pos = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty, name=f"lp_big_pos_{len(model.getVars())}")
            neg = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty, name=f"lp_big_neg_{len(model.getVars())}")
            actual = quicksum(col.quantity * columns[idx] for idx, col in items)
            big_plan_deviation_balance[key] = model.addCons(actual - target == pos - neg)

        fixed_use_constraints = {}
        for key, indices in group_area_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_group_area_split_penalty)
            fixed_use_constraints[("group_area",) + key] = model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in group_block_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_group_block_split_penalty)
            fixed_use_constraints[("group_block",) + key] = model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_block_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_coarse_area_block_split_penalty)
            fixed_use_constraints[("coarse_area_block",) + key] = model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_bay_cols.items():
            use = model.addVar(lb=0.0, ub=1.0, obj=self.config.small_plan_coarse_area_bay_split_penalty)
            fixed_use_constraints[("coarse_area_bay",) + key] = model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for (voyage_id, area_no), indices in voyage_area_cols.items():
            cost = self._voyage_area_cost(voyage_id, area_no)
            if abs(cost) <= 1e-9:
                continue
            use = model.addVar(lb=0.0, ub=1.0, obj=cost)
            total = quicksum(columns[idx] for idx in indices)
            fixed_use_constraints[("voyage_area", voyage_id, area_no)] = model.addCons(total <= len(indices) * use)
            if cost < 0:
                fixed_use_constraints[("voyage_area_reward", voyage_id, area_no)] = model.addCons(use <= total)
        return {
            "big_plan_deviation_balance": big_plan_deviation_balance,
            "fixed_use_objective_limit": fixed_use_constraints,
        }

    def _add_integer_master_objectives(
        self,
        quicksum,
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
        row_attr_choice_cols,
    ) -> None:
        coarse_area_keys = set(coarse_area_cols)
        self._add_coarse_group_area_objectives(quicksum, model, columns, coarse_area_keys, coarse_area_cols)

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
            actual = quicksum(col.quantity * columns[idx] for idx, col in items)
            model.addCons(actual - target == pos - neg)

        for (group_id, area_no), indices in group_area_cols.items():
            use = model.addVar(vtype="B", obj=self.config.small_plan_group_area_split_penalty, name=f"use_ga_{group_id}_{area_no}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for (group_id, block_id), indices in group_block_cols.items():
            use = model.addVar(vtype="B", obj=self.config.small_plan_group_block_split_penalty, name=f"use_gb_{group_id}_{block_id}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_block_cols.items():
            voyage_id, flow, port, size, area_no, block_id = key
            use = model.addVar(
                vtype="B",
                obj=self.config.small_plan_coarse_area_block_split_penalty,
                name=f"use_cab_{voyage_id}_{flow}_{size}_{area_no}_{block_id}",
            )
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
        for key, indices in coarse_area_bay_cols.items():
            voyage_id, flow, port, size, area_no, bay_key = key
            use = model.addVar(
                vtype="B",
                obj=self.config.small_plan_coarse_area_bay_split_penalty,
                name=f"use_cay_{voyage_id}_{flow}_{size}_{area_no}_{bay_key}",
            )
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)

        for (voyage_id, area_no), indices in voyage_area_cols.items():
            cost = self._voyage_area_cost(voyage_id, area_no)
            use = model.addVar(vtype="B", obj=cost, name=f"use_va_{voyage_id}_{area_no}")
            total = quicksum(columns[idx] for idx in indices)
            model.addCons(total <= len(indices) * use)
            if cost < 0:
                model.addCons(use <= total)

        for area_no in set(edge_45_cols) | set(edge_non45_cols):
            has45 = model.addVar(vtype="B", name=f"area_has45_{area_no}")
            if edge_45_cols.get(area_no):
                model.addCons(quicksum(columns[idx] for idx in edge_45_cols[area_no]) <= len(edge_45_cols[area_no]) * has45)
            if edge_non45_cols.get(area_no):
                model.addCons(quicksum(columns[idx] for idx in edge_non45_cols[area_no]) <= len(edge_non45_cols[area_no]) * (1 - has45))

        self._add_bay_compatibility_constraints(
            quicksum,
            model,
            columns,
            bay_attr_choice_cols,
        )
        self._add_row_compatibility_constraints(
            quicksum,
            model,
            columns,
            row_attr_choice_cols,
        )

    def _add_coarse_group_area_objectives(
        self,
        quicksum,
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
            actual_by_area = self._add_coarse_area_actual_variables(
                quicksum,
                model,
                columns,
                coarse_key,
                area_keys,
                coarse_area_cols,
                demand,
            )
            if self._prefers_concentrated_coarse_key(coarse_key):
                self._add_concentrated_coarse_group_objective(
                    model,
                    coarse_key,
                    area_keys,
                    actual_by_area,
                    demand,
                )
            else:
                self._add_large_coarse_group_balance_objective(
                    model,
                    coarse_key,
                    area_keys,
                    actual_by_area,
                    demand,
                )

    def _add_coarse_area_actual_variables(
        self,
        quicksum,
        model,
        columns,
        coarse_key: tuple[str, str, str, str],
        area_keys: list[tuple[str, str, str, str, str]],
        coarse_area_cols: dict[tuple[str, str, str, str, str], list[tuple[int, PlacementColumn]]],
        demand: int,
    ) -> dict[tuple[str, str, str, str, str], object]:
        actual_by_area = {}
        for key in sorted(area_keys):
            *_, area_no = key
            items = coarse_area_cols.get(key, [])
            actual = model.addVar(
                lb=0.0,
                ub=float(demand),
                name=f"coarse_actual_{'_'.join(coarse_key)}_{area_no}",
            )
            model.addCons(actual == quicksum(col.quantity * columns[idx] for idx, col in items))
            actual_by_area[key] = actual
        return actual_by_area

    def _add_concentrated_coarse_group_objective(
        self,
        model,
        coarse_key: tuple[str, str, str, str],
        area_keys: list[tuple[str, str, str, str, str]],
        actual_by_area: dict[tuple[str, str, str, str, str], object],
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
            actual = actual_by_area[key]
            use = model.addVar(
                vtype="B",
                obj=self.config.medium_small_group_area_split_penalty,
                name=f"use_conc_area_{'_'.join(coarse_key)}_{area_no}",
            )
            primary = model.addVar(
                vtype="B",
                obj=-self.config.medium_small_group_area_split_penalty,
                name=f"primary_conc_area_{'_'.join(coarse_key)}_{area_no}",
            )
            model.addCons(actual <= demand * use)
            model.addCons(primary <= use)
            model.addCons(largest <= actual + demand * (1 - primary))
            primary_vars.append(primary)
        if primary_vars:
            primary_sum = sum(primary_vars)
            model.addCons(primary_sum <= 1)
            model.addCons(largest <= demand * primary_sum)

    def _add_large_coarse_group_balance_objective(
        self,
        model,
        coarse_key: tuple[str, str, str, str],
        area_keys: list[tuple[str, str, str, str, str]],
        actual_by_area: dict[tuple[str, str, str, str, str], object],
        demand: int,
    ) -> None:
        area_terms = []
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        small_area_penalty = self.config.medium_large_group_small_area_penalty / max(1.0, min_boxes)
        use_vars = []
        for key in sorted(area_keys):
            *_, area_no = key
            actual = actual_by_area[key]
            use = model.addVar(vtype="B", name=f"use_bal_area_{'_'.join(coarse_key)}_{area_no}")
            model.addCons(actual <= demand * use)
            model.addCons(actual >= use)
            use_vars.append(use)
            if min_boxes > 0:
                shortage = model.addVar(
                    lb=0.0,
                    obj=small_area_penalty,
                    name=f"small_bal_area_{'_'.join(coarse_key)}_{area_no}",
                )
                model.addCons(shortage >= min_boxes * use - actual)
            area_terms.append((area_no, actual, use))

        target_area_count = self._target_large_group_area_count(coarse_key, demand)
        if use_vars and self.config.medium_large_group_area_open_penalty > 0:
            extra_areas = model.addVar(
                lb=0.0,
                obj=self.config.medium_large_group_area_open_penalty,
                name=f"extra_bal_area_{'_'.join(coarse_key)}",
            )
            model.addCons(extra_areas >= sum(use_vars) - target_area_count)

        if len(area_terms) <= 1:
            return
        pair_penalty = self.config.group_area_balance_penalty / max(1.0, demand) / max(1, len(area_terms) - 1)
        for left_index, (_left_area, left_actual, left_use) in enumerate(area_terms):
            for _right_area, right_actual, right_use in area_terms[left_index + 1 :]:
                diff = model.addVar(lb=0.0, obj=pair_penalty, name=f"bal_diff_{len(model.getVars())}")
                inactive = demand * (2 - left_use - right_use)
                model.addCons(diff >= left_actual - right_actual - inactive)
                model.addCons(diff >= right_actual - left_actual - inactive)

    def _add_bay_compatibility_constraints(
        self,
        quicksum,
        model,
        columns,
        bay_attr_choice_cols: dict[tuple[str, str, str], list[int]],
    ) -> None:
        use_by_bay_attr: defaultdict[tuple[str, str], list] = defaultdict(list)
        for (bay_key, attr, value), indices in sorted(bay_attr_choice_cols.items()):
            use = model.addVar(vtype="B", name=f"bay_use_{attr}_{bay_key}_{value}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
            use_by_bay_attr[(bay_key, attr)].append(use)
        for (bay_key, attr), uses in use_by_bay_attr.items():
            model.addCons(quicksum(uses) <= 1, name=f"bay_one_{attr}_{bay_key}")

    def _add_row_compatibility_constraints(
        self,
        quicksum,
        model,
        columns,
        row_attr_choice_cols: dict[tuple[str, str, str, str], list[int]],
    ) -> None:
        use_by_row_attr: defaultdict[tuple[str, str, str], list] = defaultdict(list)
        for (bay_key, row_no, attr, value), indices in sorted(row_attr_choice_cols.items()):
            use = model.addVar(vtype="B", name=f"row_use_{attr}_{bay_key}_{row_no}_{value}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
            use_by_row_attr[(bay_key, row_no, attr)].append(use)
        for (bay_key, row_no, attr), uses in use_by_row_attr.items():
            model.addCons(quicksum(uses) <= 1, name=f"row_one_{attr}_{bay_key}_{row_no}")

    def _price_columns(
        self,
        lp_model,
        lp_constraints: dict,
        iteration: int,
        lp_unplaced: Counter[str] | None = None,
        lp_column_values: dict[int, float] | None = None,
    ) -> dict:
        group_dual = {group_id: self._scip_dual(lp_model, constr) for group_id, constr in lp_constraints["group_cover"].items()}
        group_dual = self._effective_group_duals(group_dual, lp_unplaced or Counter())
        bay_capacity_dual = {bay_key: self._scip_dual(lp_model, constr) for bay_key, constr in lp_constraints["bay_capacity_limit"].items()}
        bay_size_dual = {key: self._scip_dual(lp_model, constr) for key, constr in lp_constraints["bay_size_limit"].items()}
        bay_port_stack_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("bay_port_stack_link", {}).items()
        }
        group_bay_dual = {key: self._scip_dual(lp_model, constr) for key, constr in lp_constraints["group_bay_limit"].items()}
        quota_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("quota_limit", {}).items()
        }
        medium_plan_quota_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("medium_plan_quota_limit", {}).items()
        }
        big_plan_deviation_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("big_plan_deviation_balance", {}).items()
        }
        fixed_use_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("fixed_use_objective_limit", {}).items()
        }
        lp_column_values = lp_column_values or {}
        lp_quota_actual = self._column_values_quota_actual(lp_column_values)
        lp_coarse_area_actual = self._column_values_coarse_area_actual(lp_column_values)
        lp_unplaced = lp_unplaced or Counter()
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

    def _bay_no_mix_attrs(self, voyage_id: object = None) -> tuple[str, ...]:
        if self.attribute_rules is not None and voyage_id is not None and hasattr(self.attribute_rules, "bay_no_mix_for"):
            attrs = self.attribute_rules.bay_no_mix_for(voyage_id)
        else:
            attrs = getattr(self.attribute_rules, "bay_no_mix_attributes", ("size", "height"))
        return tuple(str(attr) for attr in attrs if str(attr))

    def _row_no_mix_attrs(self, voyage_id: object = None) -> tuple[str, ...]:
        if self.attribute_rules is not None and voyage_id is not None and hasattr(self.attribute_rules, "row_no_mix_for"):
            attrs = self.attribute_rules.row_no_mix_for(voyage_id)
        else:
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
        return "|".join(f"{attr}={self._group_attr_value(group, attr)}" for attr in self._row_no_mix_attrs(group.voyage_id)) or "__all__"

    def _row_mix_key_for_column(self, col: PlacementColumn) -> str:
        return "|".join(f"{attr}={self._column_attr_value(col, attr)}" for attr in self._row_no_mix_attrs(col.voyage_id)) or "__all__"

    def _row_existing_attrs_allow_group(self, bay: Bay, row_no: str, group: SmallBoxGroup) -> bool:
        row_attrs = getattr(bay, "existing_attrs_by_row", {}).get(str(row_no), {})
        for attr in self._row_no_mix_attrs(group.voyage_id):
            values = set(row_attrs.get(attr, set()))
            if values and self._group_attr_value(group, attr) not in values:
                return False
        return True

    def _bay_existing_attrs_allow_group(self, group: SmallBoxGroup, footprint: tuple[str, ...]) -> bool:
        for key in footprint:
            bay = self.bays[key]
            existing_attrs = getattr(bay, "existing_attrs", {})
            for attr in self._bay_no_mix_attrs(group.voyage_id):
                values = set(existing_attrs.get(attr, set()))
                if values and values != {self._group_attr_value(group, attr)}:
                    return False
        return True

    def _bay_state_attrs_allow_group(self, group: SmallBoxGroup, footprint: tuple[str, ...], state: dict) -> bool:
        used_attrs = state.setdefault("bay_used_attrs", {})
        for key in footprint:
            for attr in self._bay_no_mix_attrs(group.voyage_id):
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

    def _row_capacity_items_for_group(
        self,
        footprint_key: str,
        size: str,
        group: SmallBoxGroup,
        state: dict | None = None,
    ) -> list[tuple[str, int]]:
        bay = self.bays[footprint_key]
        row_caps = bay.row_cap_by_size.get(size, {}) or {}
        if not row_caps and bay.row_physical_capacity:
            row_caps = bay.row_physical_capacity
        if row_caps:
            out: list[tuple[str, int]] = []
            for row_no, raw_cap in row_caps.items():
                row_no = str(row_no)
                if not self._row_existing_attrs_allow_group(bay, row_no, group):
                    continue
                cap = int(raw_cap)
                if state is not None:
                    row_key = (footprint_key, row_no)
                    row_size_key = (footprint_key, row_no, size)
                    cap = min(
                        cap - int(state["row_load"][row_key]),
                        int(bay.row_physical_capacity.get(row_no, raw_cap)) - int(state["row_load"][row_key]),
                        int(raw_cap) - int(state["row_size_load"][row_size_key]),
                    )
                    for attr in self._row_no_mix_attrs(group.voyage_id):
                        value = self._group_attr_value(group, attr)
                        used = state["row_used_attrs"].get((footprint_key, row_no, attr), value)
                        if used != value:
                            cap = 0
                            break
                if cap > 0:
                    out.append((row_no, int(cap)))
            return sorted(out, key=lambda item: self._row_sort_key(item[0]))
        cap = int(bay.cap_by_size.get(size, 0) or bay.physical_capacity)
        if state is not None:
            cap = min(
                cap - int(state["bay_load"][footprint_key]),
                int(bay.physical_capacity) - int(state["bay_load"][footprint_key]),
                int(bay.cap_by_size.get(size, cap)) - int(state["bay_size_load"][(footprint_key, size)]),
            )
        return [("__bay__", cap)] if cap > 0 else []

    @staticmethod
    def _row_sort_key(row_no: str) -> tuple[int, str]:
        try:
            return int(row_no), row_no
        except ValueError:
            return 10**9, row_no

    @staticmethod
    def _row_allocation_signature(row_allocation: tuple[tuple[str, str, int], ...]) -> tuple[tuple[str, str, int], ...]:
        return tuple(sorted((str(bay_key), str(row_no), int(qty)) for bay_key, row_no, qty in row_allocation if int(qty) > 0))

    def _row_allocation_patterns_for_column(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        quantity: int,
        state: dict | None = None,
        max_patterns: int = 6,
    ) -> list[tuple[tuple[str, str, int], ...]]:
        if quantity <= 0:
            return []
        footprint = self._placement_footprint_keys(bay_key, group.size)
        if not footprint:
            return []
        per_bay = {
            key: dict(self._row_capacity_items_for_group(key, group.size, group, state=state))
            for key in footprint
        }
        if any(not caps for caps in per_bay.values()):
            return []
        common_rows = set.intersection(*(set(caps) for caps in per_bay.values()))
        if not common_rows:
            return []
        row_caps = {
            row_no: min(per_bay[key][row_no] for key in footprint)
            for row_no in common_rows
        }
        row_caps = {row_no: cap for row_no, cap in row_caps.items() if cap > 0}
        if sum(row_caps.values()) < quantity:
            return []

        ordered_rows = sorted(row_caps, key=self._row_sort_key)
        row_orders: list[list[str]] = [ordered_rows, list(reversed(ordered_rows))]
        row_orders.extend([[row_no] + [other for other in ordered_rows if other != row_no] for row_no in ordered_rows if row_caps[row_no] >= quantity])

        patterns: list[tuple[tuple[str, str, int], ...]] = []
        seen: set[tuple[tuple[str, str, int], ...]] = set()
        for order in row_orders:
            remaining = int(quantity)
            by_row: list[tuple[str, int]] = []
            for row_no in order:
                if remaining <= 0:
                    break
                take = min(remaining, row_caps[row_no])
                if take > 0:
                    by_row.append((row_no, take))
                    remaining -= take
            if remaining > 0:
                continue
            allocation = self._row_allocation_signature(
                tuple((footprint_key, row_no, take) for row_no, take in by_row for footprint_key in footprint)
            )
            if allocation in seen:
                continue
            seen.add(allocation)
            patterns.append(allocation)
            if len(patterns) >= max_patterns:
                break
        return patterns

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

    def _add_column(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        quantity: int,
        base_cost: float,
        row_allocation: tuple[tuple[str, str, int], ...] | None = None,
        state: dict | None = None,
    ) -> bool:
        if quantity <= 0:
            return False
        if not self._user_bay_policy_allows(group, bay_key):
            return False
        stack_units = self._column_stack_units(group, bay_key, quantity)
        if stack_units >= 10**9:
            return False
        allocations = [self._row_allocation_signature(row_allocation)] if row_allocation else self._row_allocation_patterns_for_column(group, bay_key, quantity, state=state)
        allocations = [allocation for allocation in allocations if allocation]
        if not allocations:
            return False
        bay = self.bays[bay_key]
        block_id = self.block_by_bay.get((bay.area_no, bay_key), "")
        added = False
        for allocation in allocations:
            key = (group.group_id, bay_key, quantity, allocation)
            if key in self._column_keys:
                continue
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
                row_allocation=allocation,
                quota_key=self._quota_key(group, bay.area_no),
                coarse_key=self._coarse_key(group),
                intrinsic_cost=base_cost,
            )
            self._columns.append(col)
            self._column_keys.add(key)
            added = True
        return added

    def _greedy_fallback(self) -> tuple[Counter[int], Counter[str]]:
        selected, unplaced, _stats = self._staged_repair_selected_solution(Counter())
        return selected, unplaced

    def _repair_selected_solution(
        self,
        selected: Counter[int],
        allow_new_columns: bool = True,
    ) -> tuple[Counter[int], Counter[str]]:
        repaired, state, placed = self._selection_state(selected)
        unplaced: Counter[str] = Counter()
        for group in self.groups:
            remaining = int(group.demand) - int(placed.get(group.group_id, 0))
            while remaining > 0:
                choice = self._best_repair_column(
                    group,
                    state,
                    remaining,
                    repaired,
                    allow_new_columns=allow_new_columns,
                )
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
        allow_new_columns: bool = True,
    ) -> tuple[Counter[int], Counter[str], dict]:
        candidates: list[tuple[str, Counter[int], Counter[str], list[dict]]] = []
        for label, group_order in self._staged_repair_group_orders():
            repaired, unplaced, stats = self._staged_repair_selected_solution_for_order(
                selected,
                group_order,
                allow_new_columns=allow_new_columns,
            )
            candidates.append((label, repaired, unplaced, stats))
        label, best_selected, best_unplaced, best_stats = min(
            candidates,
            key=lambda item: self._solution_rank(item[1], item[2]),
        )
        return best_selected, best_unplaced, {
            "selected_candidate": label,
            "iterations": best_stats,
            "candidates": [
                {
                    "candidate": candidate_label,
                    "unplaced_boxes": sum(candidate_unplaced.values()),
                    "objective": round(self._selected_solution_energy(candidate_selected, candidate_unplaced), 6),
                    "selected_columns": sum(1 for qty in candidate_selected.values() if qty > 0),
                }
                for candidate_label, candidate_selected, candidate_unplaced, _candidate_stats in candidates
            ],
        }

    def _run_repair_lns_rounds(
        self,
        incumbent_selected: Counter[int],
        incumbent_unplaced: Counter[str],
        seed_unplaced: Counter[str],
    ) -> tuple[Counter[int], Counter[str], dict]:
        max_rounds = max(0, int(getattr(self.config, "repair_lns_rounds", 0) or 0))
        per_round_time = max(0.0, float(getattr(self.config, "repair_lns_time_limit", 0.0) or 0.0))
        max_groups = max(1, int(getattr(self.config, "repair_lns_max_groups", 1) or 1))
        max_no_improve_rounds = max(
            0,
            int(getattr(self.config, "repair_lns_max_no_improve_rounds", 0) or 0),
        )
        stats = {
            "repair_lns_rounds_requested": max_rounds,
            "repair_lns_time_limit": per_round_time,
            "repair_lns_max_groups": max_groups,
            "repair_lns_max_no_improve_rounds": max_no_improve_rounds,
            "repair_lns_candidate_neighborhoods": 0,
            "repair_lns_rounds_run": 0,
            "repair_lns_improvements": 0,
            "repair_lns_added_columns": 0,
            "repair_lns_initial_unplaced_boxes": sum(incumbent_unplaced.values()),
            "repair_lns_seed_unplaced_boxes": sum(seed_unplaced.values()),
            "repair_lns_initial_objective": round(self._selected_solution_energy(incumbent_selected, incumbent_unplaced), 6),
            "repair_lns_final_objective": round(self._selected_solution_energy(incumbent_selected, incumbent_unplaced), 6),
            "repair_lns_iterations": [],
            "repair_lns_stop_reason": "",
        }
        if max_rounds <= 0 or per_round_time <= 0:
            stats["repair_lns_stop_reason"] = "disabled"
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats
        if not any(qty > 0 for qty in seed_unplaced.values()) and not any(qty > 0 for qty in incumbent_unplaced.values()):
            stats["repair_lns_stop_reason"] = "no_seed_unplaced"
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats

        try:
            from pyscipopt import Model, quicksum
        except Exception as exc:
            stats["repair_lns_stop_reason"] = f"scip_unavailable:{type(exc).__name__}"
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats

        neighborhoods = self._repair_lns_neighborhoods(incumbent_selected, seed_unplaced, max_rounds * 2, max_groups)
        stats["repair_lns_candidate_neighborhoods"] = len(neighborhoods)
        if not neighborhoods:
            stats["repair_lns_stop_reason"] = "no_neighborhood"
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats

        best_selected = Counter(incumbent_selected)
        best_unplaced = Counter(incumbent_unplaced)
        best_rank = self._solution_rank(best_selected, best_unplaced)
        no_improve_rounds = 0
        for round_no, (label, release_group_ids) in enumerate(neighborhoods[:max_rounds]):
            before_columns = len(self._columns)
            self._ensure_repair_lns_columns(release_group_ids, best_selected)
            added_columns = len(self._columns) - before_columns
            stats["repair_lns_added_columns"] += added_columns
            if self.config.verbose:
                print(
                    "[column-generation-scip] repair LNS "
                    f"round={round_no} label={label} release_groups={len(release_group_ids)} "
                    f"added_columns={added_columns} time_limit={per_round_time:.1f}s",
                    flush=True,
                )

            (
                candidate_selected,
                candidate_unplaced,
                sub_stats,
            ) = self._solve_repair_lns_subproblem(
                Model,
                quicksum,
                best_selected,
                best_unplaced,
                release_group_ids,
                per_round_time,
            )
            status = str(sub_stats.get("status", ""))
            has_solution = bool(sub_stats.get("has_solution", False))
            mip_start_added = bool(sub_stats.get("mip_start_added", False))
            candidate_rank = best_rank
            candidate_objective = self._selected_solution_energy(candidate_selected, candidate_unplaced)
            round_improved = False
            if has_solution:
                candidate_rank = self._solution_rank(candidate_selected, candidate_unplaced)
                candidate_objective = self._selected_solution_energy(candidate_selected, candidate_unplaced)
                if candidate_rank < best_rank:
                    best_selected = candidate_selected
                    best_unplaced = candidate_unplaced
                    best_rank = candidate_rank
                    stats["repair_lns_improvements"] += 1
                    round_improved = True
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
            else:
                no_improve_rounds += 1
            stats["repair_lns_iterations"].append(
                {
                    "round": round_no,
                    "label": label,
                    "status": status,
                    "has_solution": has_solution,
                    "mip_start_added": mip_start_added,
                    "hard_no_unplaced": bool(sub_stats.get("hard_no_unplaced", False)),
                    "unplaced_nonworsening_cap": sub_stats.get("unplaced_nonworsening_cap"),
                    "released_group_count": len(release_group_ids),
                    "candidate_column_count": sub_stats.get("candidate_column_count", 0),
                    "fixed_selected_column_count": sub_stats.get("fixed_selected_column_count", 0),
                    "added_columns": added_columns,
                    "objective": round(candidate_objective, 6),
                    "unplaced_boxes": sum(candidate_unplaced.values()),
                    "improved": round_improved,
                }
            )
            stats["repair_lns_rounds_run"] += 1
            if max_no_improve_rounds > 0 and no_improve_rounds >= max_no_improve_rounds:
                stats["repair_lns_stop_reason"] = "no_incumbent_improvement"
                break
        stats["repair_lns_final_objective"] = round(self._selected_solution_energy(best_selected, best_unplaced), 6)
        stats["repair_lns_final_unplaced_boxes"] = sum(best_unplaced.values())
        if not stats["repair_lns_stop_reason"]:
            stats["repair_lns_stop_reason"] = "completed"
        return best_selected, best_unplaced, stats

    def _run_coarse_compaction_lns_rounds(
        self,
        incumbent_selected: Counter[int],
        incumbent_unplaced: Counter[str],
    ) -> tuple[Counter[int], Counter[str], dict]:
        max_rounds = max(0, int(getattr(self.config, "coarse_compaction_lns_rounds", 0) or 0))
        per_round_time = max(0.0, float(getattr(self.config, "coarse_compaction_lns_time_limit", 0.0) or 0.0))
        max_groups = max(1, int(getattr(self.config, "coarse_compaction_lns_max_groups", 1) or 1))
        max_no_improve_rounds = max(
            0,
            int(getattr(self.config, "coarse_compaction_lns_max_no_improve_rounds", 0) or 0),
        )
        stats = {
            "coarse_compaction_lns_rounds_requested": max_rounds,
            "coarse_compaction_lns_time_limit": per_round_time,
            "coarse_compaction_lns_max_groups": max_groups,
            "coarse_compaction_lns_max_no_improve_rounds": max_no_improve_rounds,
            "coarse_compaction_lns_candidate_neighborhoods": 0,
            "coarse_compaction_lns_rounds_run": 0,
            "coarse_compaction_lns_improvements": 0,
            "coarse_compaction_lns_added_columns": 0,
            "coarse_compaction_lns_initial_objective": round(self._selected_solution_energy(incumbent_selected, incumbent_unplaced), 6),
            "coarse_compaction_lns_final_objective": round(self._selected_solution_energy(incumbent_selected, incumbent_unplaced), 6),
            "coarse_compaction_lns_iterations": [],
            "coarse_compaction_lns_stop_reason": "",
        }
        if max_rounds <= 0 or per_round_time <= 0:
            stats["coarse_compaction_lns_stop_reason"] = "disabled"
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats
        if sum(incumbent_unplaced.values()) > 0:
            stats["coarse_compaction_lns_stop_reason"] = "incumbent_has_unplaced"
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats
        try:
            from pyscipopt import Model, quicksum
        except Exception as exc:
            stats["coarse_compaction_lns_stop_reason"] = f"scip_unavailable:{type(exc).__name__}"
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats

        neighborhoods = self._coarse_compaction_neighborhoods(incumbent_selected, max_rounds * 2, max_groups)
        stats["coarse_compaction_lns_candidate_neighborhoods"] = len(neighborhoods)
        if not neighborhoods:
            stats["coarse_compaction_lns_stop_reason"] = "no_fragmented_coarse_group"
            return Counter(incumbent_selected), Counter(incumbent_unplaced), stats

        best_selected = Counter(incumbent_selected)
        best_unplaced = Counter(incumbent_unplaced)
        best_rank = self._solution_rank(best_selected, best_unplaced)
        no_improve_rounds = 0
        for round_no, (label, release_group_ids) in enumerate(neighborhoods[:max_rounds]):
            before_columns = len(self._columns)
            self._ensure_repair_lns_columns(release_group_ids, best_selected)
            added_columns = len(self._columns) - before_columns
            stats["coarse_compaction_lns_added_columns"] += added_columns
            if self.config.verbose:
                print(
                    "[column-generation-scip] coarse compaction LNS "
                    f"round={round_no} label={label} release_groups={len(release_group_ids)} "
                    f"added_columns={added_columns} time_limit={per_round_time:.1f}s",
                    flush=True,
                )
            candidate_selected, candidate_unplaced, sub_stats = self._solve_repair_lns_subproblem(
                Model,
                quicksum,
                best_selected,
                best_unplaced,
                release_group_ids,
                per_round_time,
            )
            has_solution = bool(sub_stats.get("has_solution", False))
            candidate_rank = best_rank
            candidate_objective = self._selected_solution_energy(candidate_selected, candidate_unplaced)
            round_improved = False
            if has_solution:
                candidate_rank = self._solution_rank(candidate_selected, candidate_unplaced)
                candidate_objective = self._selected_solution_energy(candidate_selected, candidate_unplaced)
                if candidate_rank < best_rank:
                    best_selected = candidate_selected
                    best_unplaced = candidate_unplaced
                    best_rank = candidate_rank
                    stats["coarse_compaction_lns_improvements"] += 1
                    round_improved = True
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
            else:
                no_improve_rounds += 1
            stats["coarse_compaction_lns_iterations"].append(
                {
                    "round": round_no,
                    "label": label,
                    "status": str(sub_stats.get("status", "")),
                    "has_solution": has_solution,
                    "mip_start_added": bool(sub_stats.get("mip_start_added", False)),
                    "hard_no_unplaced": bool(sub_stats.get("hard_no_unplaced", False)),
                    "unplaced_nonworsening_cap": sub_stats.get("unplaced_nonworsening_cap"),
                    "released_group_count": len(release_group_ids),
                    "candidate_column_count": sub_stats.get("candidate_column_count", 0),
                    "fixed_selected_column_count": sub_stats.get("fixed_selected_column_count", 0),
                    "added_columns": added_columns,
                    "objective": round(candidate_objective, 6),
                    "unplaced_boxes": sum(candidate_unplaced.values()),
                    "improved": round_improved,
                }
            )
            stats["coarse_compaction_lns_rounds_run"] += 1
            if max_no_improve_rounds > 0 and no_improve_rounds >= max_no_improve_rounds:
                stats["coarse_compaction_lns_stop_reason"] = "no_incumbent_improvement"
                break
        stats["coarse_compaction_lns_final_objective"] = round(self._selected_solution_energy(best_selected, best_unplaced), 6)
        stats["coarse_compaction_lns_final_unplaced_boxes"] = sum(best_unplaced.values())
        if not stats["coarse_compaction_lns_stop_reason"]:
            stats["coarse_compaction_lns_stop_reason"] = "completed"
        return best_selected, best_unplaced, stats

    def _solve_repair_lns_subproblem(
        self,
        Model,
        quicksum,
        incumbent_selected: Counter[int],
        incumbent_unplaced: Counter[str],
        release_group_ids: set[str],
        time_limit: float,
    ) -> tuple[Counter[int], Counter[str], dict]:
        fixed_selected = Counter(
            {
                idx: qty
                for idx, qty in incumbent_selected.items()
                if qty > 0 and 0 <= idx < len(self._columns) and self._columns[idx].group_id not in release_group_ids
            }
        )
        _fixed_repaired, fixed_state, fixed_placed = self._selection_state(fixed_selected)
        hard_no_unplaced = sum(incumbent_unplaced.values()) <= 0
        released_remaining = {
            group_id: max(0, int(self.groups_by_id[group_id].demand) - int(fixed_placed.get(group_id, 0)))
            for group_id in release_group_ids
            if group_id in self.groups_by_id
        }
        candidate_indices: list[int] = []
        for idx, col in enumerate(self._columns):
            if col.group_id not in release_group_ids:
                continue
            group = self.groups_by_id.get(col.group_id)
            if group is None:
                continue
            remaining = int(group.demand) - int(fixed_placed.get(group.group_id, 0))
            if remaining <= 0 or col.quantity > remaining:
                continue
            if self._column_fits_state(col, fixed_state, remaining, enforce_quota=False):
                candidate_indices.append(idx)

        stats = {
            "status": "",
            "has_solution": False,
            "mip_start_added": False,
            "candidate_column_count": len(candidate_indices),
            "fixed_selected_column_count": sum(1 for qty in fixed_selected.values() if qty > 0),
            "hard_no_unplaced": hard_no_unplaced,
        }
        if not candidate_indices:
            if hard_no_unplaced and any(qty > 0 for qty in released_remaining.values()):
                stats["status"] = "no_candidate_columns_hard_no_unplaced"
                return Counter(incumbent_selected), Counter(incumbent_unplaced), stats
            unplaced = Counter(
                {
                    group_id: qty
                    for group_id, qty in released_remaining.items()
                    if qty > 0
                }
            )
            for group_id, qty in incumbent_unplaced.items():
                if group_id not in release_group_ids and qty > 0:
                    unplaced[group_id] += int(qty)
            stats["status"] = "no_candidate_columns"
            return Counter(fixed_selected), unplaced, stats

        model = Model("yard_repair_lns_residual_scip")
        self._configure_scip_output(model)
        try:
            model.setMinimize()
        except Exception:
            pass
        self._set_scip_param(model, "limits/time", float(time_limit))
        self._set_scip_param(model, "limits/gap", float(self.config.mip_gap))

        column_vars = {}
        for idx in candidate_indices:
            col = self._columns[idx]
            obj = (
                col.intrinsic_cost
                + self.config.small_plan_group_bay_split_penalty
                + self._area_fallback_tier_penalty_for_column(col) * col.quantity
            )
            column_vars[idx] = model.addVar(vtype="B", obj=float(obj), name=f"lns_col_{idx}")

        unplaced_vars = {}
        for group_id in sorted(release_group_ids):
            group = self.groups_by_id.get(group_id)
            if group is None:
                continue
            remaining = int(released_remaining.get(group_id, 0))
            unplaced_vars[group_id] = model.addVar(
                lb=0.0,
                ub=0.0 if hard_no_unplaced else float(remaining),
                vtype="I",
                obj=float(self.config.unplaced_penalty),
                name=f"lns_unplaced_{group_id}",
            )

        fixed_unplaced_total = sum(
            int(qty)
            for group_id, qty in incumbent_unplaced.items()
            if group_id not in release_group_ids and qty > 0
        )
        incumbent_unplaced_total = sum(int(qty) for qty in incumbent_unplaced.values() if qty > 0)
        released_unplaced_cap = max(0, incumbent_unplaced_total - fixed_unplaced_total)
        if unplaced_vars:
            model.addCons(
                quicksum(unplaced_vars.values()) <= released_unplaced_cap,
                name="lns_unplaced_nonworsening_cap",
            )
        stats["unplaced_nonworsening_cap"] = released_unplaced_cap

        group_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_capacity_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_size_cols: defaultdict[tuple[str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_port_size_cols: defaultdict[tuple[str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        row_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        row_size_cols: defaultdict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        row_attr_choice_cols: defaultdict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        group_bay_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        group_area_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        group_block_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        coarse_area_bay_cols: defaultdict[tuple[str, str, str, str, str, str], list[int]] = defaultdict(list)
        coarse_area_cols: defaultdict[tuple[str, str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        voyage_area_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        area_size_cols: defaultdict[tuple[str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        medium_quota_cols: defaultdict[tuple[str, str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        medium_bay_quota_cols: defaultdict[tuple[str, str, str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_attr_choice_cols: defaultdict[tuple[str, str, str], list[int]] = defaultdict(list)
        edge_45_cols: defaultdict[str, list[int]] = defaultdict(list)
        edge_non45_cols: defaultdict[str, list[int]] = defaultdict(list)
        for idx in candidate_indices:
            col = self._columns[idx]
            group_cols[col.group_id].append((idx, col))
            for footprint_key in self._placement_footprint_keys(col.bay_key, col.size):
                bay_capacity_cols[footprint_key].append((idx, col))
                bay_port_size_cols[(footprint_key, self._row_mix_key_for_column(col), col.size)].append((idx, col))
                for attr in self._bay_no_mix_attrs(col.voyage_id):
                    bay_attr_choice_cols[(footprint_key, attr, self._column_attr_value(col, attr))].append(idx)
            for footprint_key, row_no, qty in col.row_allocation:
                row_capacity_cols[(footprint_key, row_no)].append((idx, int(qty)))
                row_size_cols[(footprint_key, row_no, col.size)].append((idx, int(qty)))
                for attr in self._row_no_mix_attrs(col.voyage_id):
                    row_attr_choice_cols[(footprint_key, row_no, attr, self._column_attr_value(col, attr))].append(idx)
            bay_size_cols[(col.bay_key, col.size)].append((idx, col))
            group_bay_cols[(col.group_id, col.bay_key)].append(idx)
            group_area_cols[(col.group_id, col.area_no)].append(idx)
            if col.block_id:
                group_block_cols[(col.group_id, col.block_id)].append(idx)
            coarse_area_bay_cols[col.coarse_key + (col.area_no, col.bay_key)].append(idx)
            coarse_area_cols[col.coarse_key + (col.area_no,)].append((idx, col))
            voyage_area_cols[(col.voyage_id, col.area_no)].append(idx)
            area_size_cols[col.quota_key].append((idx, col))
            medium_quota_cols[col.coarse_key + (col.area_no,)].append((idx, col))
            medium_bay_quota_cols[col.coarse_key + (col.area_no, col.bay_key)].append((idx, col))
            if col.bay_key in self.area_edge_bays.get(col.area_no, set()):
                if col.size == "45":
                    edge_45_cols[col.area_no].append(idx)
                else:
                    edge_non45_cols[col.area_no].append(idx)

        for group_id in sorted(release_group_ids):
            group = self.groups_by_id.get(group_id)
            if group is None:
                continue
            remaining = max(0, int(group.demand) - int(fixed_placed.get(group_id, 0)))
            expr = quicksum(col.quantity * column_vars[idx] for idx, col in group_cols.get(group_id, []))
            model.addCons(expr + unplaced_vars[group_id] == remaining, name=f"lns_cover_{group_id}")

        for bay_key, items in bay_capacity_cols.items():
            residual = self.bays[bay_key].physical_capacity - fixed_state["bay_load"][bay_key]
            model.addCons(quicksum(col.quantity * column_vars[idx] for idx, col in items) <= residual)
        for (bay_key, size), items in bay_size_cols.items():
            residual = self.bays[bay_key].cap_by_size.get(size, 0) - fixed_state["bay_size_load"][(bay_key, size)]
            model.addCons(quicksum(col.quantity * column_vars[idx] for idx, col in items) <= residual)
        for (bay_key, row_no), items in row_capacity_cols.items():
            residual = int(self.bays[bay_key].row_physical_capacity.get(row_no, self.bays[bay_key].physical_capacity)) - fixed_state["row_load"][(bay_key, row_no)]
            model.addCons(quicksum(qty * column_vars[idx] for idx, qty in items) <= residual)
        for (bay_key, row_no, size), items in row_size_cols.items():
            residual = int(self.bays[bay_key].row_cap_by_size.get(size, {}).get(row_no, self.bays[bay_key].cap_by_size.get(size, 0))) - fixed_state["row_size_load"][(bay_key, row_no, size)]
            model.addCons(quicksum(qty * column_vars[idx] for idx, qty in items) <= residual)
        for (group_id, bay_key), indices in group_bay_cols.items():
            model.addCons(quicksum(column_vars[idx] for idx in indices) <= 1)

        stack_vars_by_bay_size: defaultdict[tuple[str, str], list[tuple[object, int]]] = defaultdict(list)
        for key, items in sorted(bay_port_size_cols.items()):
            bay_key, _port, size = key
            sample_group = self.groups_by_id.get(items[0][1].group_id) if items else None
            if sample_group is None:
                continue
            unit_capacity = self._stack_unit_capacity_for_group(bay_key, size, sample_group)
            port_stack_count = self._stack_count_for_group(bay_key, size, sample_group)
            if unit_capacity <= 0 or port_stack_count <= 0:
                continue
            fixed_load = fixed_state["bay_port_size_load"][key]
            fixed_units = self._stack_units_for_quantity(bay_key, size, sample_group, fixed_load)
            stack_var = model.addVar(lb=float(fixed_units), ub=float(port_stack_count), vtype="I", name=f"lns_stack_{bay_key}_{size}_{len(stack_vars_by_bay_size)}")
            load = quicksum(col.quantity * column_vars[idx] for idx, col in items)
            model.addCons(fixed_load + load <= unit_capacity * stack_var)
            stack_vars_by_bay_size[(bay_key, size)].append((stack_var, fixed_units))
        for bay_size_key, stack_items in stack_vars_by_bay_size.items():
            bay_key, size = bay_size_key
            total_stack_count = self._stack_count_for_bay_size(bay_key, size)
            fixed_total_units = fixed_state["bay_stack_used"][bay_size_key]
            model.addCons(quicksum(stack_var - fixed_units for stack_var, fixed_units in stack_items) <= max(0, total_stack_count - fixed_total_units))

        self._add_local_use_objectives(model, quicksum, column_vars, group_area_cols, fixed_state["used_group_area"], self.config.small_plan_group_area_split_penalty)
        self._add_local_use_objectives(model, quicksum, column_vars, group_block_cols, fixed_state["used_group_block"], self.config.small_plan_group_block_split_penalty)
        self._add_local_use_objectives(model, quicksum, column_vars, coarse_area_bay_cols, fixed_state["used_coarse_area_bay"], self.config.small_plan_coarse_area_bay_split_penalty)
        self._add_local_use_objectives(model, quicksum, column_vars, voyage_area_cols, fixed_state["used_voyage_area"], 0.0, voyage_area_cost=True)
        self._add_local_coarse_area_distribution_objectives(
            model,
            quicksum,
            column_vars,
            coarse_area_cols,
            fixed_state["coarse_area_used"],
            release_group_ids,
        )

        for key, items in area_size_cols.items():
            voyage_id, flow, area_no, big_size = key
            target = self._area_size_target(voyage_id, flow, area_no, big_size)
            fixed_actual = fixed_state["big_plan_quota_used"][key]
            pos = model.addVar(lb=0.0, obj=float(self.config.big_plan_area_deviation_penalty), name=f"lns_big_pos_{len(model.getVars())}")
            neg = model.addVar(lb=0.0, obj=float(self.config.big_plan_area_deviation_penalty), name=f"lns_big_neg_{len(model.getVars())}")
            actual = fixed_actual + quicksum(col.quantity * column_vars[idx] for idx, col in items)
            model.addCons(actual - target == pos - neg)

        if self.config.medium_plan_quota is not None:
            medium_plan_quota = Counter(self.config.medium_plan_quota)
            for key, items in medium_quota_cols.items():
                residual = medium_plan_quota[key] - fixed_state["medium_plan_quota_used"][key]
                model.addCons(quicksum(col.quantity * column_vars[idx] for idx, col in items) <= residual)
        if self.config.medium_plan_bay_quota is not None:
            medium_plan_bay_quota = Counter(self.config.medium_plan_bay_quota)
            for key, items in medium_bay_quota_cols.items():
                residual = medium_plan_bay_quota[key] - fixed_state["medium_plan_bay_quota_used"][key]
                model.addCons(quicksum(col.quantity * column_vars[idx] for idx, col in items) <= residual)

        self._add_bay_compatibility_constraints(quicksum, model, column_vars, bay_attr_choice_cols)
        self._add_row_compatibility_constraints(quicksum, model, column_vars, row_attr_choice_cols)
        for area_no in set(edge_45_cols) | set(edge_non45_cols):
            has45 = model.addVar(vtype="B", name=f"lns_area_has45_{area_no}")
            if edge_45_cols.get(area_no):
                model.addCons(quicksum(column_vars[idx] for idx in edge_45_cols[area_no]) <= len(edge_45_cols[area_no]) * has45)
            if edge_non45_cols.get(area_no):
                model.addCons(quicksum(column_vars[idx] for idx in edge_non45_cols[area_no]) <= len(edge_non45_cols[area_no]) * (1 - has45))

        fixed_voyage_area_qty: Counter[tuple[str, str]] = Counter()
        for idx, qty in fixed_selected.items():
            if qty > 0 and 0 <= idx < len(self._columns):
                col = self._columns[idx]
                fixed_voyage_area_qty[(col.voyage_id, col.area_no)] += int(qty) * int(col.quantity)
        for voyage_id, areas in sorted(getattr(self.problem, "user_voyage_area_requirements", {}).items()):
            for area_no in sorted(areas):
                if fixed_voyage_area_qty[(voyage_id, area_no)] >= 1:
                    continue
                indices = voyage_area_cols.get((voyage_id, area_no), [])
                if indices:
                    model.addCons(quicksum(self._columns[idx].quantity * column_vars[idx] for idx in indices) >= 1)

        fixed_group_bay_qty: Counter[tuple[str, str]] = Counter()
        for idx, qty in fixed_selected.items():
            if qty > 0 and 0 <= idx < len(self._columns):
                col = self._columns[idx]
                fixed_group_bay_qty[(col.group_id, col.bay_key)] += int(qty) * int(col.quantity)
        for group_id, bay_keys in sorted(getattr(self.problem, "user_group_bay_requirements", {}).items()):
            for bay_key in sorted(bay_keys):
                if fixed_group_bay_qty[(group_id, bay_key)] >= 1:
                    continue
                indices = group_bay_cols.get((group_id, bay_key), [])
                model.addCons(quicksum(self._columns[idx].quantity * column_vars[idx] for idx in indices) >= 1)

        mip_start_added = self._add_repair_lns_start(model, column_vars, unplaced_vars, incumbent_selected, incumbent_unplaced)
        stats["mip_start_added"] = mip_start_added
        model.optimize()
        stats["status"] = self._scip_status_name(model)
        stats["has_solution"] = self._scip_solution_count(model) > 0
        selected = Counter(fixed_selected)
        unplaced = Counter(
            {
                group_id: int(qty)
                for group_id, qty in incumbent_unplaced.items()
                if group_id not in release_group_ids and qty > 0
            }
        )
        if stats["has_solution"]:
            for idx, var in column_vars.items():
                if self._scip_value(model, var) > 0.5:
                    selected[idx] = 1
            for group_id, var in unplaced_vars.items():
                value = int(round(self._scip_value(model, var)))
                if value > 0:
                    unplaced[group_id] += value
        else:
            selected = Counter(incumbent_selected)
            unplaced = Counter(incumbent_unplaced)
        self._free_scip_model(model)
        return selected, unplaced, stats

    def _add_repair_lns_start(
        self,
        model,
        column_vars: dict[int, object],
        unplaced_vars: dict[str, object],
        selected: Counter[int],
        unplaced: Counter[str],
    ) -> bool:
        try:
            creator = getattr(model, "createPartialSol", None) or getattr(model, "createSol")
            sol = creator()
            for idx, var in column_vars.items():
                model.setSolVal(sol, var, 1.0 if selected.get(idx, 0) > 0 else 0.0)
            for group_id, var in unplaced_vars.items():
                model.setSolVal(sol, var, float(unplaced.get(group_id, 0)))
            return bool(model.addSol(sol, free=True))
        except Exception:
            return False

    def _add_local_use_objectives(
        self,
        model,
        quicksum,
        column_vars: dict[int, object],
        grouped_indices: dict,
        already_used: set,
        penalty: float,
        voyage_area_cost: bool = False,
    ) -> None:
        for key, indices in grouped_indices.items():
            if key in already_used:
                continue
            cost = self._voyage_area_cost(*key) if voyage_area_cost else penalty
            if abs(float(cost)) <= 1e-9:
                continue
            use = model.addVar(vtype="B", obj=float(cost), name=f"lns_use_{len(model.getVars())}")
            model.addCons(quicksum(column_vars[idx] for idx in indices) <= len(indices) * use)
            if cost < 0:
                model.addCons(use <= quicksum(column_vars[idx] for idx in indices))

    def _add_local_coarse_area_distribution_objectives(
        self,
        model,
        quicksum,
        column_vars: dict[int, object],
        coarse_area_cols: dict[tuple[str, str, str, str, str], list[tuple[int, PlacementColumn]]],
        fixed_coarse_area_used: Counter[tuple[str, str, str, str, str]],
        release_group_ids: set[str],
    ) -> None:
        release_coarse_keys = {
            self._coarse_key(group)
            for group_id in release_group_ids
            if (group := self.groups_by_id.get(group_id)) is not None
        }
        if not release_coarse_keys:
            return
        area_keys_by_coarse: defaultdict[tuple[str, str, str, str], set[tuple[str, str, str, str, str]]] = defaultdict(set)
        for key in coarse_area_cols:
            if key[:4] in release_coarse_keys:
                area_keys_by_coarse[key[:4]].add(key)
        for key, qty in fixed_coarse_area_used.items():
            if qty > 0 and key[:4] in release_coarse_keys:
                area_keys_by_coarse[key[:4]].add(key)

        for coarse_key, area_keys in sorted(area_keys_by_coarse.items()):
            demand = max(1, int(self.coarse_demand.get(coarse_key, 0)))
            if demand <= 0:
                continue
            actual_by_area = {}
            for key in sorted(area_keys):
                fixed_qty = float(fixed_coarse_area_used.get(key, 0))
                items = coarse_area_cols.get(key, [])
                actual = model.addVar(
                    lb=0.0,
                    ub=float(demand),
                    name=f"lns_coarse_actual_{'_'.join(coarse_key)}_{key[4]}",
                )
                model.addCons(actual == fixed_qty + quicksum(col.quantity * column_vars[idx] for idx, col in items))
                actual_by_area[key] = actual
            if not actual_by_area:
                continue
            if self._prefers_concentrated_coarse_key(coarse_key):
                self._add_concentrated_coarse_group_objective(
                    model,
                    coarse_key,
                    sorted(actual_by_area),
                    actual_by_area,
                    demand,
                )
            else:
                self._add_large_coarse_group_balance_objective(
                    model,
                    coarse_key,
                    sorted(actual_by_area),
                    actual_by_area,
                    demand,
                )

    def _repair_lns_neighborhoods(
        self,
        selected: Counter[int],
        seed_unplaced: Counter[str],
        max_neighborhoods: int,
        max_groups: int,
    ) -> list[tuple[str, set[str]]]:
        selected_group_qty: Counter[str] = Counter()
        coarse_to_groups: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        area_to_groups: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        fallback_groups: Counter[str] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            qty = int(chosen) * int(col.quantity)
            selected_group_qty[col.group_id] += qty
            coarse_to_groups[col.coarse_key][col.group_id] += qty
            area_to_groups[(col.voyage_id, col.size, col.area_no)][col.group_id] += qty
            group = self.groups_by_id.get(col.group_id)
            if group is not None and self._area_fallback_tier_for_group(group, col.area_no) > 0:
                fallback_groups[col.group_id] += qty

        neighborhoods: list[tuple[str, set[str]]] = []
        seen: set[tuple[str, ...]] = set()

        def add(label: str, group_ids: Iterable[str]) -> None:
            cleaned = [group_id for group_id in group_ids if group_id in self.groups_by_id]
            if not cleaned:
                return
            ordered = sorted(
                set(cleaned),
                key=lambda group_id: (
                    -int(seed_unplaced.get(group_id, 0)),
                    -int(selected_group_qty.get(group_id, 0)),
                    group_id,
                ),
            )[:max_groups]
            signature = tuple(sorted(ordered))
            if not signature or signature in seen:
                return
            seen.add(signature)
            neighborhoods.append((label, set(ordered)))

        for group_id, _qty in seed_unplaced.most_common():
            group = self.groups_by_id.get(group_id)
            if group is None:
                continue
            coarse_key = self._coarse_key(group)
            add(f"seed_coarse_{group_id}", [group_id] + [gid for gid, _ in coarse_to_groups.get(coarse_key, Counter()).most_common()])
            for (voyage_id, size, _area_no), area_groups in sorted(area_to_groups.items()):
                if voyage_id == group.voyage_id and size == group.size and group_id in area_groups:
                    add(f"seed_area_{group_id}", [group_id] + [gid for gid, _ in area_groups.most_common()])

        for group_id, _qty in fallback_groups.most_common(max_neighborhoods):
            group = self.groups_by_id.get(group_id)
            if group is None:
                continue
            add(
                f"fallback_coarse_{group_id}",
                [group_id] + [gid for gid, _ in coarse_to_groups.get(self._coarse_key(group), Counter()).most_common()],
            )

        for index, group_ids in enumerate(self._diving_improvement_neighborhoods(selected, max_neighborhoods, max_groups)):
            add(f"fragmentation_{index}", group_ids)
            if len(neighborhoods) >= max_neighborhoods:
                break
        return neighborhoods[:max_neighborhoods]

    def _coarse_compaction_neighborhoods(
        self,
        selected: Counter[int],
        max_neighborhoods: int,
        max_groups: int,
    ) -> list[tuple[str, set[str]]]:
        by_coarse: defaultdict[tuple[str, str, str, str], dict] = defaultdict(
            lambda: {"groups": Counter(), "areas": Counter(), "area_groups": defaultdict(Counter), "bays": set()}
        )
        area_groups: defaultdict[str, Counter[str]] = defaultdict(Counter)
        selected_group_qty: Counter[str] = Counter()
        for idx, chosen in selected.items():
            if chosen <= 0 or idx < 0 or idx >= len(self._columns):
                continue
            col = self._columns[idx]
            qty = int(chosen) * int(col.quantity)
            selected_group_qty[col.group_id] += qty
            data = by_coarse[col.coarse_key]
            data["groups"][col.group_id] += qty
            data["areas"][col.area_no] += qty
            data["area_groups"][col.area_no][col.group_id] += qty
            data["bays"].add(col.bay_key)
            area_groups[col.area_no][col.group_id] += qty

        ranked: list[tuple[float, tuple[str, str, str, str], set[str]]] = []
        min_boxes = max(1, int(self.config.medium_large_group_min_area_boxes or 1))
        for coarse_key, data in by_coarse.items():
            groups: Counter[str] = data["groups"]
            areas: Counter[str] = data["areas"]
            if not groups or not areas:
                continue
            demand = max(1, int(self.coarse_demand.get(coarse_key, sum(groups.values()))))
            area_count = sum(1 for qty in areas.values() if qty > 0)
            tiny_count = sum(1 for qty in areas.values() if 0 < qty < min_boxes)
            target_count = self._target_large_group_area_count(coarse_key, demand)
            if self._prefers_concentrated_coarse_key(coarse_key):
                score = 3000.0 * max(0, area_count - 1) + 200.0 * tiny_count + 0.01 * demand
            else:
                score = 1800.0 * tiny_count + 250.0 * max(0, area_count - target_count) + 0.01 * demand
            if score <= 0:
                continue

            release = {group_id for group_id, _qty in groups.most_common(max_groups)}
            remaining_slots = max(0, max_groups - len(release))
            if remaining_slots > 0:
                tiny_or_small_areas = sorted(
                    [area_no for area_no, qty in areas.items() if qty < min_boxes],
                    key=lambda area_no: (areas[area_no], area_no),
                )
                if not tiny_or_small_areas:
                    tiny_or_small_areas = [area_no for area_no, _qty in areas.most_common()]
                related: Counter[str] = Counter()
                for area_no in tiny_or_small_areas:
                    related.update(area_groups.get(area_no, Counter()))
                for group_id, _qty in related.most_common():
                    if len(release) >= max_groups:
                        break
                    if group_id not in release:
                        release.add(group_id)
            if release:
                ranked.append((score, coarse_key, release))

        neighborhoods: list[tuple[str, set[str]]] = []
        seen: set[tuple[str, ...]] = set()
        for score, coarse_key, group_ids in sorted(ranked, reverse=True):
            ordered = sorted(
                group_ids,
                key=lambda group_id: (-selected_group_qty.get(group_id, 0), group_id),
            )[:max_groups]
            signature = tuple(sorted(ordered))
            if not signature or signature in seen:
                continue
            seen.add(signature)
            label = f"coarse_compact_{'_'.join(coarse_key)}_{int(score)}"
            neighborhoods.append((label, set(ordered)))
            if len(neighborhoods) >= max_neighborhoods:
                break
        return neighborhoods

    def _fixed_selected_columns_for_released_groups(
        self,
        selected: Counter[int],
        release_group_ids: set[str],
    ) -> dict[int, int]:
        return {
            idx: 1
            for idx, qty in selected.items()
            if qty > 0 and 0 <= idx < len(self._columns) and self._columns[idx].group_id not in release_group_ids
        }

    def _ensure_repair_lns_columns(self, release_group_ids: set[str], selected: Counter[int]) -> None:
        fixed_selected = Counter(
            {
                idx: qty
                for idx, qty in selected.items()
                if qty > 0 and 0 <= idx < len(self._columns) and self._columns[idx].group_id not in release_group_ids
            }
        )
        _fixed_selected, state, placed = self._selection_state(fixed_selected)
        stages = [
            ("stage1a", True),
            ("stage1b", False),
            ("stage2", False),
            ("stage3", False),
        ]
        for group_id in sorted(release_group_ids):
            group = self.groups_by_id.get(group_id)
            if group is None:
                continue
            remaining = int(group.demand) - int(placed.get(group.group_id, 0))
            if remaining <= 0:
                continue
            for stage, enforce_quota in stages:
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
                    for qty in self._quantity_options(group, min(remaining, capacity)):
                        self._add_column(group, bay_key, qty, base_cost)

    def _staged_repair_group_orders(self) -> list[tuple[str, list[SmallBoxGroup]]]:
        groups = list(self.groups)
        orders: list[tuple[str, list[SmallBoxGroup]]] = [
            ("default", groups),
            ("coarse_concentration_first", self._coarse_concentration_repair_order(groups)),
        ]
        unique: list[tuple[str, list[SmallBoxGroup]]] = []
        seen: set[tuple[str, ...]] = set()
        for label, ordered_groups in orders:
            signature = tuple(group.group_id for group in ordered_groups)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append((label, ordered_groups))
        return unique

    def _coarse_concentration_repair_order(self, groups: list[SmallBoxGroup]) -> list[SmallBoxGroup]:
        by_coarse: defaultdict[tuple[str, str, str, str], list[SmallBoxGroup]] = defaultdict(list)
        for group in groups:
            by_coarse[self._coarse_key(group)].append(group)

        threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))

        def coarse_rank(coarse_key: tuple[str, str, str, str]) -> tuple:
            demand = int(self.coarse_demand.get(coarse_key, 0))
            small_group = threshold > 0 and demand <= threshold
            voyage_id, flow, port, size = coarse_key
            return (
                0 if small_group else 1,
                demand if small_group else -demand,
                SIZE_ORDER.get(size, 3),
                sum(
                    1
                    for (quota_voyage, quota_flow, _area_no, quota_size), quota in self.quota_by_key.items()
                    if quota_voyage == voyage_id
                    and quota_flow == flow
                    and quota_size == self._big_plan_size(size)
                    and quota > 0
                ),
                voyage_id,
                flow,
                port,
                size,
            )

        ordered: list[SmallBoxGroup] = []
        for coarse_key in sorted(by_coarse, key=coarse_rank):
            ordered.extend(
                sorted(
                    by_coarse[coarse_key],
                    key=lambda group: (
                        -int(group.demand),
                        group.height,
                        group.weight_class,
                        group.group_id,
                    ),
                )
            )
        return ordered

    def _staged_repair_selected_solution_for_order(
        self,
        selected: Counter[int],
        group_order: list[SmallBoxGroup],
        allow_new_columns: bool = True,
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
            for group in group_order:
                remaining = int(group.demand) - int(placed.get(group.group_id, 0))
                while remaining > 0:
                    choice = self._best_repair_column(
                        group,
                        state,
                        remaining,
                        repaired,
                        allow_new_columns=allow_new_columns,
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
            "row_load": Counter(),
            "row_size_load": Counter(),
            "row_used_attrs": {},
            "bay_used_size": {},
            "bay_used_height": {},
            "bay_used_attrs": {},
            "group_bay_used": set(),
            "used_group_area": set(),
            "used_group_block": set(),
            "used_coarse_area_block": set(),
            "used_coarse_area_bay": set(),
            "used_voyage_area": set(),
            "coarse_area_used": Counter(),
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
            self._add_column(group, bay_key, min(remaining, capacity), base_cost, state=state)

    def _best_repair_column(
        self,
        group: SmallBoxGroup,
        state: dict,
        remaining: int,
        selected: Counter[int],
        allow_new_columns: bool = True,
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
            score = (
                self._repair_column_score(group, bay_key, qty, base_cost, state),
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
        idx = self._column_index_for(group.group_id, bay_key, qty, state=state)
        if idx is None:
            if not allow_new_columns:
                return None
            self._add_column(group, bay_key, qty, base_cost, state=state)
            idx = self._column_index_for(group.group_id, bay_key, qty, state=state)
            if idx is None:
                return None
        if selected.get(idx, 0) > 0:
            return None
        col = self._columns[idx]
        if not self._column_fits_state(col, state, remaining, enforce_quota=enforce_quota):
            return None
        return idx, col

    def _repair_column_score(
        self,
        group: SmallBoxGroup,
        bay_key: str,
        qty: int,
        base_cost: float,
        state: dict,
    ) -> float:
        bay = self.bays[bay_key]
        area_no = bay.area_no
        block_id = self.block_by_bay.get((area_no, bay_key), "")
        quota_key = self._quota_key(group, area_no)
        coarse_key = self._coarse_key(group)
        coarse_area_key = coarse_key + (area_no,)

        score = (
            base_cost
            + self.config.small_plan_group_bay_split_penalty
        )

        if (group.group_id, area_no) not in state["used_group_area"]:
            score += self.config.small_plan_group_area_split_penalty
        if block_id and (group.group_id, block_id) not in state["used_group_block"]:
            score += self.config.small_plan_group_block_split_penalty
        if block_id and coarse_key + (area_no, block_id) not in state["used_coarse_area_block"]:
            score += self.config.small_plan_coarse_area_block_split_penalty
        if coarse_key + (area_no, bay_key) not in state["used_coarse_area_bay"]:
            score += self.config.small_plan_coarse_area_bay_split_penalty
        if (group.voyage_id, area_no) not in state["used_voyage_area"]:
            score += self._voyage_area_cost(group.voyage_id, area_no)

        before_quota = state["big_plan_quota_used"][quota_key]
        target = self._area_size_target(group.voyage_id, group.status, area_no, self._big_plan_size(group.size))
        score += self.config.big_plan_area_deviation_penalty * (
            abs(before_quota + qty - target) - abs(before_quota - target)
        )
        score += self._coarse_area_incremental_repair_cost(coarse_area_key, qty, state["coarse_area_used"])
        return float(score)

    def _coarse_area_incremental_repair_cost(
        self,
        coarse_area_key: tuple[str, str, str, str, str],
        qty: int,
        coarse_area_used: Counter[tuple[str, str, str, str, str]],
    ) -> float:
        coarse_key = coarse_area_key[:4]
        area_no = coarse_area_key[4]
        quantities = {
            key[4]: float(value)
            for key, value in coarse_area_used.items()
            if key[:4] == coarse_key and value > 0
        }
        before = self._coarse_area_distribution_energy(coarse_key, quantities)
        quantities[area_no] = quantities.get(area_no, 0.0) + float(qty)
        after = self._coarse_area_distribution_energy(coarse_key, quantities)
        return after - before

    def _coarse_area_distribution_energy(
        self,
        coarse_key: tuple[str, str, str, str],
        quantities_by_area: dict[str, float],
    ) -> float:
        quantities = [qty for qty in quantities_by_area.values() if qty > 0]
        if not quantities:
            return 0.0
        if self._prefers_concentrated_coarse_key(coarse_key):
            return (
                self.config.medium_small_group_area_split_penalty * max(0, len(quantities) - 1)
                - self.config.medium_small_group_fragment_penalty * max(quantities)
            )

        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        energy = 0.0
        demand = max(1, int(self.coarse_demand.get(coarse_key, sum(quantities))))
        energy += self.config.medium_large_group_area_open_penalty * max(
            0,
            len(quantities) - self._target_large_group_area_count(coarse_key, demand),
        )
        if min_boxes > 0:
            small_area_penalty = self.config.medium_large_group_small_area_penalty / max(1.0, min_boxes)
            energy += sum(small_area_penalty * max(0.0, min_boxes - qty) for qty in quantities if qty > 0)

        if len(quantities) > 1:
            pair_penalty = self.config.group_area_balance_penalty / max(1.0, demand) / max(1, len(quantities) - 1)
            for left_index, left in enumerate(quantities):
                for right in quantities[left_index + 1 :]:
                    energy += pair_penalty * abs(left - right)
        return float(energy)

    def _column_index_for(self, group_id: str, bay_key: str, quantity: int, state: dict | None = None) -> int | None:
        for idx, col in enumerate(self._columns):
            if col.group_id == group_id and col.bay_key == bay_key and col.quantity == quantity:
                if state is not None and not self._column_row_allocation_fits_state(col, state):
                    continue
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
        if not self._column_row_allocation_fits_state(col, state):
            return False
        return col.quantity <= self._remaining_capacity_for_group_bay(
            group,
            col.bay_key,
            state,
            remaining,
            enforce_quota=enforce_quota,
        )

    def _column_row_allocation_fits_state(self, col: PlacementColumn, state: dict) -> bool:
        group = self.groups_by_id.get(col.group_id)
        if group is None:
            return False
        by_footprint: Counter[str] = Counter()
        for footprint_key, row_no, qty in col.row_allocation:
            qty = int(qty)
            if qty <= 0:
                continue
            bay = self.bays[footprint_key]
            row_key = (footprint_key, row_no)
            row_size_key = (footprint_key, row_no, col.size)
            physical_cap = int(bay.row_physical_capacity.get(row_no, bay.physical_capacity))
            size_cap = int(bay.row_cap_by_size.get(col.size, {}).get(row_no, bay.cap_by_size.get(col.size, 0)))
            if state["row_load"][row_key] + qty > physical_cap:
                return False
            if state["row_size_load"][row_size_key] + qty > size_cap:
                return False
            if not self._row_existing_attrs_allow_group(bay, row_no, group):
                return False
            for attr in self._row_no_mix_attrs(group.voyage_id):
                value = self._column_attr_value(col, attr)
                used = state["row_used_attrs"].get((footprint_key, row_no, attr), value)
                if used != value:
                    return False
            by_footprint[footprint_key] += qty
        return bool(by_footprint) and all(qty == col.quantity for qty in by_footprint.values())

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
        if not self._user_bay_policy_allows(group, bay_key):
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
        row_patterns = self._row_allocation_patterns_for_column(group, bay_key, capacity, state=state, max_patterns=1)
        if not row_patterns:
            row_capacity = 0
            for qty in range(capacity - 1, 0, -1):
                if self._row_allocation_patterns_for_column(group, bay_key, qty, state=state, max_patterns=1):
                    row_capacity = qty
                    break
            capacity = min(capacity, row_capacity)
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
            for attr in self._bay_no_mix_attrs(col.voyage_id):
                state.setdefault("bay_used_attrs", {})[(key, attr)] = self._column_attr_value(col, attr)
        state["bay_size_load"][(col.bay_key, col.size)] += col.quantity
        group = self.groups_by_id.get(col.group_id)
        if group is not None:
            self._apply_stack_usage_to_state(group, col.bay_key, col.quantity, state)
        for footprint_key, row_no, qty in col.row_allocation:
            qty = int(qty)
            if qty <= 0:
                continue
            state["row_load"][(footprint_key, row_no)] += qty
            state["row_size_load"][(footprint_key, row_no, col.size)] += qty
            for attr in self._row_no_mix_attrs(col.voyage_id):
                state["row_used_attrs"][(footprint_key, row_no, attr)] = self._column_attr_value(col, attr)
        state["group_bay_used"].add((col.group_id, col.bay_key))
        state["used_group_area"].add((col.group_id, col.area_no))
        if col.block_id:
            state["used_group_block"].add((col.group_id, col.block_id))
            state["used_coarse_area_block"].add(col.coarse_key + (col.area_no, col.block_id))
        state["used_coarse_area_bay"].add(col.coarse_key + (col.area_no, col.bay_key))
        state["used_voyage_area"].add((col.voyage_id, col.area_no))
        state["coarse_area_used"][col.coarse_key + (col.area_no,)] += col.quantity
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
                if not self._user_bay_policy_allows(group, col.bay_key):
                    continue
                footprint = self._placement_footprint_keys(col.bay_key, col.size)
                if not footprint:
                    continue
                if any(
                    bay_used_attrs.get((key, attr), self._column_attr_value(col, attr)) != self._column_attr_value(col, attr)
                    for key in footprint
                    for attr in self._bay_no_mix_attrs(col.voyage_id)
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
                    for attr in self._bay_no_mix_attrs(col.voyage_id):
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
                if not self._user_bay_policy_allows(group, bay_key):
                    continue
                max_qty = self._max_quantity_in_bay(group, bay_key)
                if max_qty <= 0:
                    continue
                cost = self._column_base_cost(group, bay_key)
                out.append((bay_key, min(max_qty, group.demand), cost))
        if self._prefers_concentrated_coarse_key(self._coarse_key(group)):
            out.sort(
                key=lambda item: (
                    0 if self._user_bay_policy_requires(group, item[0]) else 1,
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
                    0 if self._user_bay_policy_requires(group, item[0]) else 1,
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

    def _user_bay_policy_allows(self, group: SmallBoxGroup, bay_key: str) -> bool:
        blocked = getattr(self.problem, "user_group_bay_blocklist", {}).get(group.group_id, set())
        return bay_key not in blocked

    def _user_bay_policy_requires(self, group: SmallBoxGroup, bay_key: str) -> bool:
        required = getattr(self.problem, "user_group_bay_requirements", {}).get(group.group_id, set())
        return bay_key in required

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
        required_bays = getattr(self.problem, "user_group_bay_requirements", {}).get(group.group_id, set())
        required = [item for item in candidates if item[0] in required_bays]
        remaining_limit = max(0, limit - len(required))
        preferred_areas = set(self._area_weights(group))
        if not preferred_areas:
            return required + [item for item in candidates if item[0] not in required_bays][:remaining_limit]
        preferred = [item for item in candidates if item[0] not in required_bays and self.bays[item[0]].area_no in preferred_areas]
        fallback = [item for item in candidates if item[0] not in required_bays and self.bays[item[0]].area_no not in preferred_areas]
        if not fallback:
            return required + preferred[:remaining_limit]
        fallback_limit = min(len(fallback), max(1, remaining_limit // 4)) if remaining_limit > 0 else 0
        preferred_limit = max(0, remaining_limit - fallback_limit)
        return required + preferred[:preferred_limit] + fallback[:fallback_limit]

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
        if self._user_bay_policy_requires(group, bay_key):
            cost -= float(self.config.required_area_reward)
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

    def _target_large_group_area_count(self, coarse_key: tuple[str, str, str, str], demand: int | None = None) -> int:
        if self._prefers_concentrated_coarse_key(coarse_key):
            return 1
        target_boxes = max(1, int(self.config.medium_large_group_target_area_boxes or 1))
        total = max(1, int(self.coarse_demand.get(coarse_key, demand or 1) if demand is None else demand))
        return max(1, math.ceil(total / target_boxes))

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
            cost -= float(self.config.required_area_reward)
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
        bay_requirements = getattr(self.problem, "user_group_bay_requirements", {})
        bay_blocklist = getattr(self.problem, "user_group_bay_blocklist", {})
        forbidden_usage: Counter[tuple[str, str, str]] = Counter()
        outside_only_usage: Counter[tuple[str, str, str]] = Counter()
        required_bay_usage: Counter[tuple[str, str]] = Counter()
        forbidden_bay_usage: Counter[tuple[str, str, str]] = Counter()
        for plan_level, rows in (("medium", medium_rows), ("small", small_rows)):
            for row in rows:
                qty = int(row.get("planned_boxes", 0) or 0)
                if qty <= 0:
                    continue
                voyage_id = str(row.get("voyage_id", ""))
                area_no = str(row.get("area_no", ""))
                group_id = str(row.get("group_id", ""))
                bay_key = str(row.get("bay_key", ""))
                if area_no in blocklist.get(voyage_id, set()):
                    forbidden_usage[(voyage_id, area_no, plan_level)] += qty
                allowed = allowlist.get(voyage_id, set())
                if allowed and area_no not in allowed:
                    outside_only_usage[(voyage_id, area_no, plan_level)] += qty
                if group_id and bay_key in bay_requirements.get(group_id, set()):
                    required_bay_usage[(group_id, bay_key)] += qty
                if group_id and bay_key in bay_blocklist.get(group_id, set()):
                    forbidden_bay_usage[(group_id, bay_key, plan_level)] += qty

        forbidden = [
            {"voyage_id": voyage_id, "area_no": area_no, "plan_level": level, "boxes": qty}
            for (voyage_id, area_no, level), qty in sorted(forbidden_usage.items())
        ]
        outside_only = [
            {"voyage_id": voyage_id, "area_no": area_no, "plan_level": level, "boxes": qty}
            for (voyage_id, area_no, level), qty in sorted(outside_only_usage.items())
        ]
        unmet_required_bays = [
            {"group_id": group_id, "bay_key": bay_key, "reason": "required_bay_not_used"}
            for group_id, bay_keys in sorted(bay_requirements.items())
            for bay_key in sorted(bay_keys)
            if required_bay_usage[(group_id, bay_key)] <= 0
        ]
        forbidden_bays = [
            {"group_id": group_id, "bay_key": bay_key, "plan_level": level, "boxes": qty}
            for (group_id, bay_key, level), qty in sorted(forbidden_bay_usage.items())
        ]
        return {
            "has_violations": bool(unmet_required or forbidden or outside_only or unmet_required_bays or forbidden_bays),
            "unmet_required_areas": unmet_required,
            "forbidden_area_usage": forbidden,
            "outside_only_area_usage": outside_only,
            "unmet_required_bays": unmet_required_bays,
            "forbidden_bay_usage": forbidden_bays,
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
            qty_by_row: Counter[str] = Counter()
            for _footprint_key, row_no, qty in col.row_allocation:
                qty_by_row[str(row_no)] = max(qty_by_row[str(row_no)], int(qty))
            for row_no, row_qty in qty_by_row.items():
                row_specific_allocation = self._format_row_allocation(
                    tuple(item for item in col.row_allocation if str(item[1]) == row_no)
                )
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
                        row_no,
                        col.block_id,
                        col.block_bays,
                        row_specific_allocation,
                    )
                ] += row_qty * chosen
        block_total: Counter[str] = Counter()
        for key, qty in counter.items():
            block_id = key[11]
            if block_id:
                block_total[block_id] += qty
        rows: list[dict] = []
        for key, qty in sorted(counter.items()):
            voyage_id, group_id, flow, port, size, height, weight_class, special_code, area_no, bay_no, row_no, block_id, block_bays, row_allocation = key
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
                    "row_no": row_no,
                    "row_allocation": row_allocation,
                    "six_bay_block_id": block_id,
                    "six_bay_block_bays": "|".join(block_bays) if block_id else "",
                    "six_bay_block_total_boxes": block_total.get(block_id, 0) if block_id else 0,
                    "planned_boxes": qty,
                }
            )
        return rows

    @staticmethod
    def _format_row_allocation(row_allocation: tuple[tuple[str, str, int], ...]) -> str:
        return "|".join(f"{bay_key}:{row_no}:{int(qty)}" for bay_key, row_no, qty in row_allocation if int(qty) > 0)

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

        counter = self._compact_medium_counter(
            counter,
            small_lower_by_coarse_area,
            representative_by_coarse,
            selected_coarse_area_weights,
            selected_area_weights,
        )

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

    def _compact_medium_counter(
        self,
        counter: Counter[tuple[str, str, str, str, str]],
        small_lower_by_coarse_area: Counter[tuple[str, str, str, str, str]],
        representative_by_coarse: dict[tuple[str, str, str, str], object],
        selected_coarse_area_weights: dict[tuple[str, str, str, str], Counter[str]],
        selected_area_weights: dict[tuple[str, str, str], Counter[str]],
    ) -> Counter[tuple[str, str, str, str, str]]:
        compacted: Counter[tuple[str, str, str, str, str]] = Counter()
        by_coarse: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        for (*coarse_key, area_no), qty in counter.items():
            if qty > 0:
                by_coarse[tuple(coarse_key)][area_no] += int(qty)

        for coarse_key, quantities in by_coarse.items():
            lower = Counter(
                {
                    area_no: int(small_lower_by_coarse_area.get(coarse_key + (area_no,), 0))
                    for area_no in quantities
                    if small_lower_by_coarse_area.get(coarse_key + (area_no,), 0) > 0
                }
            )
            group = representative_by_coarse.get(coarse_key)
            weights = Counter(selected_coarse_area_weights.get(coarse_key, Counter()))
            quota_caps: Counter[str] = Counter()
            if group is not None:
                big_size = self._big_plan_size(group.size)
                weights.update(selected_area_weights.get((group.voyage_id, group.status, big_size), Counter()))
                preferred_areas = set()
                for (voyage_id, flow, area_no, size), quota in self.quota_by_key.items():
                    if voyage_id == group.voyage_id and flow == group.status and size == big_size and quota > 0:
                        weights[area_no] += int(quota)
                        quota_caps[area_no] += int(quota)
                        preferred_areas.add(area_no)
            else:
                preferred_areas = set()
            if not weights:
                weights = Counter(quantities)

            new_quantities = self._compact_coarse_medium_distribution(
                coarse_key,
                quantities,
                lower,
                weights,
                preferred_areas,
                quota_caps,
            )
            for area_no, qty in new_quantities.items():
                if qty > 0:
                    compacted[coarse_key + (area_no,)] += int(qty)
        return compacted

    def _compact_coarse_medium_distribution(
        self,
        coarse_key: tuple[str, str, str, str],
        quantities: Counter[str],
        lower: Counter[str],
        weights: Counter[str],
        preferred_areas: set[str],
        quota_caps: Counter[str],
    ) -> Counter[str]:
        total = int(sum(quantities.values()))
        if total <= 0:
            return Counter()
        lower = Counter({area: min(int(qty), int(quantities.get(area, qty))) for area, qty in lower.items() if qty > 0})
        lower_total = int(sum(lower.values()))
        movable = max(0, total - lower_total)
        if movable <= 0:
            return Counter({area: qty for area, qty in lower.items() if qty > 0})

        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        concentration_threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))
        candidates = set(quantities) | set(lower) | set(quota_caps) | {area for area, qty in weights.items() if qty > 0}
        if not candidates:
            return Counter({area: qty for area, qty in quantities.items() if qty > 0})

        def area_sort_key(area_no: str) -> tuple[int, int, int, int, str]:
            return (
                0 if area_no in preferred_areas else 1,
                0 if lower.get(area_no, 0) > 0 else 1,
                -int(weights.get(area_no, 0)),
                -int(quantities.get(area_no, 0)),
                area_no,
            )

        if concentration_threshold > 0 and total <= concentration_threshold:
            primary = sorted(candidates, key=area_sort_key)[0]
            out = Counter(lower)
            out[primary] += movable
            return Counter({area: qty for area, qty in out.items() if qty > 0})

        target_count = max(
            1,
            len([area for area, qty in lower.items() if qty > 0]),
            self._target_large_group_area_count(coarse_key, total),
        )
        target_areas = {area for area, qty in lower.items() if qty > 0}
        target_areas.update(
            area
            for area, qty in quantities.items()
            if qty > 0 and area in preferred_areas
        )
        for area_no in sorted(candidates, key=area_sort_key):
            selected_cap = sum(max(int(lower.get(area, 0)), int(quota_caps.get(area, 0))) for area in target_areas)
            if len(target_areas) >= target_count and selected_cap >= total:
                break
            target_areas.add(area_no)
        if not target_areas:
            target_areas.add(sorted(candidates, key=area_sort_key)[0])

        out = Counter(lower)
        remaining = movable
        if min_boxes > 1:
            for area_no in sorted(
                target_areas,
                key=lambda area: (out.get(area, 0) <= 0, out.get(area, 0), -weights.get(area, 0), area),
            ):
                if remaining <= 0:
                    break
                current = int(out.get(area_no, 0))
                if current <= 0 or current >= min_boxes:
                    continue
                add = min(remaining, min_boxes - current)
                out[area_no] += add
                remaining -= add

        if remaining > 0:
            target_weights = Counter(
                {
                    area: max(1, int(weights.get(area, 0) or quantities.get(area, 0) or 1))
                    for area in target_areas
                }
            )
            movable_caps = Counter(
                {
                    area: max(0, max(int(quota_caps.get(area, 0)), int(lower.get(area, 0))) - int(out.get(area, 0)))
                    for area in target_areas
                }
            )
            if sum(movable_caps.values()) < remaining:
                movable_caps = self._caps_with_overflow(movable_caps, target_weights, remaining)
            for area_no, qty in self._allocate_area_quantity(remaining, target_weights, movable_caps).items():
                out[area_no] += qty

        return self._merge_movable_medium_fragments(out, lower, weights, min_boxes)

    @staticmethod
    def _merge_movable_medium_fragments(
        quantities: Counter[str],
        lower: Counter[str],
        weights: Counter[str],
        min_boxes: int,
    ) -> Counter[str]:
        if min_boxes <= 1:
            return Counter({area: qty for area, qty in quantities.items() if qty > 0})
        out = Counter({area: int(qty) for area, qty in quantities.items() if qty > 0})
        for area_no in sorted(list(out), key=lambda area: (out[area], int(weights.get(area, 0)), area)):
            qty = out.get(area_no, 0)
            if qty <= 0 or qty >= min_boxes:
                continue
            movable = qty - int(lower.get(area_no, 0))
            if movable <= 0:
                continue
            recipients = [area for area, target_qty in out.items() if area != area_no and target_qty > 0]
            if not recipients:
                continue
            target = sorted(
                recipients,
                key=lambda area: (
                    0 if out[area] >= min_boxes else 1,
                    -int(weights.get(area, 0)),
                    -out[area],
                    area,
                ),
            )[0]
            out[area_no] -= movable
            out[target] += movable
            if out[area_no] <= 0:
                del out[area_no]
        return Counter({area: qty for area, qty in out.items() if qty > 0})

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
            "row_allocation": ColumnGenerationPlanner._format_row_allocation(col.row_allocation),
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
