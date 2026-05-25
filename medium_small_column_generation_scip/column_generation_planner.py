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

from medium_small_column_generation_scip.block_bay_planning.models import Bay, ProblemData, SmallBoxGroup


SIZE_ORDER = {"45": 0, "20": 1, "40": 2}
EXPORT_FLOWS = frozenset({"OF", "OZ"})


@dataclass(frozen=True)
class PlacementColumn:
    column_id: str
    group_id: str
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
    quota_key: tuple[str, str, str, str]
    coarse_key: tuple[str, str, str, str]
    intrinsic_cost: float


@dataclass
class ColumnGenerationConfig:
    max_iterations: int = 30
    columns_per_iteration: int = 2500
    initial_columns_per_group: int = 16
    max_candidate_bays_per_group: int = 500
    mip_time_limit: float = 120.0
    mip_gap: float = 0.01
    verbose: bool = True
    use_scip: bool = True
    demand_mode: str = "original"
    medium_plan_quota: dict[tuple[str, str, str, str, str], int] | None = None
    unplaced_penalty: float = 1_000_000.0
    group_area_balance_penalty: float = 18.0
    medium_concentrated_group_threshold: int = 26
    medium_small_group_area_split_penalty: float = 500.0
    medium_small_group_fragment_penalty: float = 20.0
    medium_large_group_min_area_boxes: int = 5
    medium_large_group_small_area_penalty: float = 120.0
    big_plan_area_deviation_penalty: float = 1.5
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
    fine groups, respects bay exclusivity and inherited big-plan quota, then
    the final integer master adds the existing medium/small soft objectives.
    """

    def __init__(self, problem: ProblemData, config: ColumnGenerationConfig | None = None) -> None:
        self.problem = problem
        self.config = config or ColumnGenerationConfig()
        self.group_source: dict[str, str] = {}
        self.demand_stats: dict[str, int | str] = {}
        self.groups = sorted(self._build_planning_groups(), key=self._group_sort_key)
        self.groups_by_id = {group.group_id: group for group in self.groups}
        self.bays = problem.bays
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
        self._candidate_cache: dict[str, list[tuple[str, int, float]]] = {}
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
        writes the small plan from all current document groups.  Document ports
        are allowed to inherit medium quota at the broader voyage/flow/size
        level, mirroring ``_small_group_area_allocations`` in the heuristic.
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

        compatible_keys = [
            key
            for key, qty in remaining.items()
            if (
                qty > 0
                and key[0] == group.voyage_id
                and key[1] == group.status
                and key[3] == group.size
                and key != exact_key
            )
        ]
        for key in sorted(compatible_keys):
            take = min(need, remaining[key])
            if take <= 0:
                continue
            remaining[key] -= take
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
        diagnostics: dict = {
            "algorithm": "small_plan_first_column_generation",
            "target_voyages": self.problem.target_voyages,
            "small_doc_group_count": int(self.demand_stats.get("source_doc_group_count", 0)),
            "small_doc_box_count": int(self.demand_stats.get("source_doc_box_count", 0)),
            "planned_group_count": len(self.groups),
            "planned_box_count": sum(group.demand for group in self.groups),
            "demand_mode": self.config.demand_mode,
            "demand_alignment": self.demand_stats,
            "initial_column_count": len(self._columns),
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
            self._expand_all_candidate_columns()
            selected, unplaced = self._greedy_fallback()

        if self._uses_original_output_scope():
            small_rows = self._make_small_rows(selected, allowed_sources={"document"})
            medium_rows = self._make_original_medium_rows(selected)
        else:
            small_rows = self._make_small_rows(selected)
            medium_rows = self._make_medium_rows(small_rows)
        consistency_stats = self._small_medium_consistency_stats(small_rows, medium_rows)
        diagnostics.update(
            {
                "final_column_count": len(self._columns),
                "selected_column_count": sum(1 for qty in selected.values() if qty > 0),
                "small_row_count": len(small_rows),
                "medium_row_count": len(medium_rows),
                "planned_small_boxes": sum(int(row["planned_boxes"]) for row in small_rows),
                "planned_medium_boxes": sum(int(row["planned_boxes"]) for row in medium_rows),
                "medium_area_rows_below_min_boxes": self._count_medium_area_rows_below_min(medium_rows),
                "unplaced_boxes": sum(unplaced.values()),
                "unplaced_by_group": {key: qty for key, qty in sorted(unplaced.items()) if qty > 0},
                **consistency_stats,
            }
        )
        return ColumnGenerationResult(medium_rows=medium_rows, small_rows=small_rows, diagnostics=diagnostics, columns=self._columns)

    def _uses_original_output_scope(self) -> bool:
        return (self.config.demand_mode or "original").strip().lower().replace("_", "-") == "original"

    def _count_medium_area_rows_below_min(self, medium_rows: list[dict]) -> int:
        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        if min_boxes <= 1:
            return 0
        return sum(
            1
            for row in medium_rows
            if 0 < int(row.get("planned_boxes", 0) or 0) < min_boxes
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

    def _solve_by_column_generation(self) -> tuple[Counter[int], Counter[str], dict]:
        from pyscipopt import Model, quicksum

        stats = {"scip_available": True, "pricing_iterations": []}
        for iteration in range(self.config.max_iterations):
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
            new_count = self._price_columns(lp_model, lp_constraints, iteration)
            stats["pricing_iterations"].append(
                {
                    "iteration": iteration,
                    "columns": len(self._columns),
                    "lp_objective": lp_objective,
                    "new_columns": new_count,
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

        mip_build_start = perf_counter()
        if self.config.verbose:
            print(
                f"[column-generation-scip] building final integer master columns={len(self._columns)}",
                flush=True,
            )
        mip_model, mip_vars, _mip_constraints = self._build_restricted_master(Model, quicksum, relax=False)
        if self.config.verbose:
            print(
                f"[column-generation-scip] final integer master built in {perf_counter() - mip_build_start:.1f}s",
                flush=True,
            )
        self._set_scip_param(mip_model, "limits/time", float(self.config.mip_time_limit))
        self._set_scip_param(mip_model, "limits/gap", float(self.config.mip_gap))
        mip_solve_start = perf_counter()
        if self.config.verbose:
            print(
                f"[column-generation-scip] solving final integer master time_limit={self.config.mip_time_limit}s",
                flush=True,
            )
        mip_model.optimize()
        mip_solve_elapsed = perf_counter() - mip_solve_start
        mip_status = self._scip_status_name(mip_model)
        solution_count = self._scip_solution_count(mip_model)
        if mip_status not in {"optimal", "timelimit", "gaplimit", "bestsollimit", "userinterrupt"} or solution_count <= 0:
            raise RuntimeError(f"integer master status={mip_status}, solutions={solution_count}")

        selected: Counter[int] = Counter()
        for idx, var in mip_vars["column"].items():
            if self._scip_value(mip_model, var) > 0.5:
                selected[idx] = 1
        unplaced = Counter(
            {
                group_id: int(round(self._scip_value(mip_model, var)))
                for group_id, var in mip_vars["unplaced"].items()
                if self._scip_value(mip_model, var) > 1e-6
            }
        )
        stats.update(
            {
                "master_status": mip_status,
                "master_objective": self._scip_objective_value(mip_model),
                "master_mip_gap": self._scip_gap(mip_model),
                "master_solve_seconds": round(mip_solve_elapsed, 3),
            }
        )
        self._free_scip_model(mip_model)
        return selected, unplaced, stats

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
    def _scip_value(model, var) -> float:
        return float(model.getVal(var))

    @staticmethod
    def _scip_dual(model, constr) -> float:
        try:
            return float(model.getDualsol(constr))
        except Exception:
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

    def _build_restricted_master(self, Model, quicksum, relax: bool):
        model = Model("yard_small_plan_column_generation_scip")
        self._configure_scip_output(model)
        try:
            model.setMinimize()
        except Exception:
            pass
        column_vtype = "C" if relax else "B"
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
                vtype="C" if relax else "I",
                obj=self.config.unplaced_penalty,
                name=f"unplaced_{group.group_id}",
            )
            for group in self.groups
        }

        group_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_capacity_cols: defaultdict[str, list[tuple[int, PlacementColumn]]] = defaultdict(list)
        bay_size_capacity_cols: defaultdict[tuple[str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
        group_bay_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        bay_size_choice_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        bay_height_choice_cols: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
        quota_cols: defaultdict[tuple[str, str, str, str], list[tuple[int, PlacementColumn]]] = defaultdict(list)
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
            bay_capacity_cols[col.bay_key].append((idx, col))
            bay_size_capacity_cols[(col.bay_key, col.size)].append((idx, col))
            group_bay_cols[(col.group_id, col.bay_key)].append(idx)
            bay_size_choice_cols[(col.bay_key, col.size)].append(idx)
            bay_height_choice_cols[(col.bay_key, col.height)].append(idx)
            quota_cols[col.quota_key].append((idx, col))
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
        group_bay_limit = {
            key: model.addCons(quicksum(columns[idx] for idx in indices) <= 1.0, name=f"group_bay_{key[0]}_{key[1]}")
            for key, indices in group_bay_cols.items()
        }
        quota_limit = {}
        for quota_key, items in quota_cols.items():
            cap = self.quota_by_key.get(quota_key, 0)
            quota_limit[quota_key] = model.addCons(
                quicksum(col.quantity * columns[idx] for idx, col in items) <= cap,
                name=f"quota_{'|'.join(quota_key)}",
            )
        medium_plan_quota_limit = {}
        if self.config.medium_plan_quota is not None:
            medium_plan_quota = Counter(self.config.medium_plan_quota)
            for key, items in coarse_area_cols.items():
                cap = int(medium_plan_quota.get(key, 0))
                medium_plan_quota_limit[key] = model.addCons(
                    quicksum(col.quantity * columns[idx] for idx, col in items) <= cap,
                    name=f"medium_quota_{len(medium_plan_quota_limit)}",
                )

        if not relax:
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
                bay_size_choice_cols,
                bay_height_choice_cols,
            )
        return model, {"column": columns, "unplaced": unplaced}, {
            "group_cover": group_cover,
            "bay_capacity_limit": bay_capacity_limit,
            "bay_size_limit": bay_size_limit,
            "group_bay_limit": group_bay_limit,
            "quota_limit": quota_limit,
            "medium_plan_quota_limit": medium_plan_quota_limit,
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
        bay_size_choice_cols,
        bay_height_choice_cols,
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
            pos = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty / max(1.0, target), name=f"big_pos_{len(model.getVars())}")
            neg = model.addVar(lb=0.0, obj=self.config.big_plan_area_deviation_penalty / max(1.0, target), name=f"big_neg_{len(model.getVars())}")
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
            bay_size_choice_cols,
            bay_height_choice_cols,
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
        for key in sorted(area_keys):
            *_, area_no = key
            actual = actual_by_area[key]
            use = model.addVar(vtype="B", name=f"use_bal_area_{'_'.join(coarse_key)}_{area_no}")
            model.addCons(actual <= demand * use)
            model.addCons(actual >= use)
            if min_boxes > 0:
                shortage = model.addVar(
                    lb=0.0,
                    obj=small_area_penalty,
                    name=f"small_bal_area_{'_'.join(coarse_key)}_{area_no}",
                )
                model.addCons(shortage >= min_boxes * use - actual)
            area_terms.append((area_no, actual, use))

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
        bay_size_choice_cols: dict[tuple[str, str], list[int]],
        bay_height_choice_cols: dict[tuple[str, str], list[int]],
    ) -> None:
        size_use_by_bay: defaultdict[str, list] = defaultdict(list)
        for (bay_key, size), indices in sorted(bay_size_choice_cols.items()):
            use = model.addVar(vtype="B", name=f"bay_use_size_{bay_key}_{size}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
            size_use_by_bay[bay_key].append(use)
        for bay_key, uses in size_use_by_bay.items():
            model.addCons(quicksum(uses) <= 1, name=f"bay_one_size_{bay_key}")

        height_use_by_bay: defaultdict[str, list] = defaultdict(list)
        for (bay_key, height), indices in sorted(bay_height_choice_cols.items()):
            use = model.addVar(vtype="B", name=f"bay_use_height_{bay_key}_{height}")
            model.addCons(quicksum(columns[idx] for idx in indices) <= len(indices) * use)
            height_use_by_bay[bay_key].append(use)
        for bay_key, uses in height_use_by_bay.items():
            model.addCons(quicksum(uses) <= 1, name=f"bay_one_height_{bay_key}")

    def _price_columns(self, lp_model, lp_constraints: dict, iteration: int) -> int:
        group_dual = {group_id: self._scip_dual(lp_model, constr) for group_id, constr in lp_constraints["group_cover"].items()}
        bay_capacity_dual = {bay_key: self._scip_dual(lp_model, constr) for bay_key, constr in lp_constraints["bay_capacity_limit"].items()}
        bay_size_dual = {key: self._scip_dual(lp_model, constr) for key, constr in lp_constraints["bay_size_limit"].items()}
        group_bay_dual = {key: self._scip_dual(lp_model, constr) for key, constr in lp_constraints["group_bay_limit"].items()}
        quota_dual = {quota_key: self._scip_dual(lp_model, constr) for quota_key, constr in lp_constraints["quota_limit"].items()}
        medium_plan_quota_dual = {
            key: self._scip_dual(lp_model, constr)
            for key, constr in lp_constraints.get("medium_plan_quota_limit", {}).items()
        }
        candidates: list[tuple[float, SmallBoxGroup, str, int, float]] = []
        for group in self.groups:
            for bay_key, max_qty, base_cost in self._candidate_bays_for_group(group):
                for qty in self._quantity_options(group, max_qty):
                    key = (group.group_id, bay_key, qty)
                    if key in self._column_keys:
                        continue
                    quota_key = self._quota_key(group, self.bays[bay_key].area_no)
                    coarse_area_key = self._coarse_key(group) + (self.bays[bay_key].area_no,)
                    reduced = (
                        base_cost
                        + self.config.small_plan_group_bay_split_penalty
                        - group_dual.get(group.group_id, 0.0) * qty
                        - bay_capacity_dual.get(bay_key, 0.0) * qty
                        - bay_size_dual.get((bay_key, group.size), 0.0) * qty
                        - group_bay_dual.get((group.group_id, bay_key), 0.0)
                        - quota_dual.get(quota_key, 0.0) * qty
                        - medium_plan_quota_dual.get(coarse_area_key, 0.0) * qty
                    )
                    if reduced < -1e-6:
                        candidates.append((reduced, group, bay_key, qty, base_cost))
        candidates.sort(key=lambda item: item[0])
        added = 0
        for _reduced, group, bay_key, qty, base_cost in candidates[: self.config.columns_per_iteration]:
            if self._add_column(group, bay_key, qty, base_cost):
                added += 1
        return added

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

    def _add_column(self, group: SmallBoxGroup, bay_key: str, quantity: int, base_cost: float) -> bool:
        if quantity <= 0:
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
            quota_key=self._quota_key(group, bay.area_no),
            coarse_key=self._coarse_key(group),
            intrinsic_cost=base_cost,
        )
        self._columns.append(col)
        self._column_keys.add(key)
        return True

    def _greedy_fallback(self) -> tuple[Counter[int], Counter[str]]:
        selected: Counter[int] = Counter()
        unplaced: Counter[str] = Counter()
        bay_load: Counter[str] = Counter()
        bay_size_load: Counter[tuple[str, str]] = Counter()
        bay_used_size: dict[str, str] = {}
        bay_used_height: dict[str, str] = {}
        group_bay_used: set[tuple[str, str]] = set()
        area_edge_has45: set[str] = set()
        area_edge_has_non45: set[str] = set()
        quota_used: Counter[tuple[str, str, str, str]] = Counter()
        medium_plan_quota = Counter(self.config.medium_plan_quota or {})
        medium_plan_quota_used: Counter[tuple[str, str, str, str, str]] = Counter()
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
                if bay_used_size.get(col.bay_key, col.size) != col.size:
                    continue
                if bay_used_height.get(col.bay_key, col.height) != col.height:
                    continue
                if bay_load[col.bay_key] + col.quantity > bay.physical_capacity:
                    continue
                if bay_size_load[(col.bay_key, col.size)] + col.quantity > bay.cap_by_size.get(col.size, 0):
                    continue
                if quota_used[col.quota_key] + col.quantity > self.quota_by_key[col.quota_key]:
                    continue
                coarse_area_key = col.coarse_key + (col.area_no,)
                if self.config.medium_plan_quota is not None and (
                    medium_plan_quota_used[coarse_area_key] + col.quantity > medium_plan_quota[coarse_area_key]
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
                bay_load[col.bay_key] += col.quantity
                bay_size_load[(col.bay_key, col.size)] += col.quantity
                bay_used_size[col.bay_key] = col.size
                bay_used_height[col.bay_key] = col.height
                group_bay_used.add((col.group_id, col.bay_key))
                if is_edge and col.size == "45":
                    area_edge_has45.add(col.area_no)
                elif is_edge:
                    area_edge_has_non45.add(col.area_no)
                quota_used[col.quota_key] += col.quantity
                medium_plan_quota_used[coarse_area_key] += col.quantity
                remaining -= col.quantity
            if remaining > 0:
                unplaced[group.group_id] = remaining
        return selected, unplaced

    def _candidate_bays_for_group(self, group: SmallBoxGroup) -> list[tuple[str, int, float]]:
        cached = self._candidate_cache.get(group.group_id)
        if cached is not None:
            return cached
        out: list[tuple[str, int, float]] = []
        for area_no, quota in self._area_weights(group).items():
            if quota <= 0 or group.status not in self.problem.area_functions.get(area_no, set()):
                continue
            for bay_key in self.bays_by_area.get(area_no, []):
                max_qty = self._max_quantity_in_bay(group, bay_key)
                if max_qty <= 0:
                    continue
                cost = self._column_base_cost(group, bay_key)
                out.append((bay_key, min(max_qty, group.demand), cost))
        if self._prefers_concentrated_coarse_key(self._coarse_key(group)):
            out.sort(
                key=lambda item: (
                    self._concentrated_area_sort_key(group, self.bays[item[0]].area_no),
                    item[2],
                    -item[1],
                    self.bays[item[0]].bay_order,
                )
            )
        else:
            out.sort(key=lambda item: (item[2], -item[1], self.bays[item[0]].area_no, self.bays[item[0]].bay_order))
        out = out[: self.config.max_candidate_bays_per_group]
        self._candidate_cache[group.group_id] = out
        return out

    def _max_quantity_in_bay(self, group: SmallBoxGroup, bay_key: str) -> int:
        bay = self.bays[bay_key]
        if bay.cap_by_size.get(group.size, 0) <= 0:
            return 0
        if bay.existing_size_modes and bay.existing_size_modes != {group.size}:
            return 0
        if bay.existing_heights and bay.existing_heights != {group.height}:
            return 0
        is_edge = bay_key in self.area_edge_bays.get(bay.area_no, set())
        if group.size == "20" and is_edge:
            return 0
        if group.size == "45" and not is_edge:
            return 0
        quota_free = self.quota_by_key.get(self._quota_key(group, bay.area_no), 0)
        return max(0, min(group.demand, bay.physical_capacity, bay.cap_by_size.get(group.size, 0), quota_free))

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
                self.area_edge_bays[area_no] = {keys[0], keys[-1]}
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
                for height in heights:
                    if bay.existing_heights and bay.existing_heights != {height}:
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

    def _concentrated_area_sort_key(self, group: SmallBoxGroup, area_no: str) -> tuple[int, int, int, str]:
        coarse_key = self._coarse_key(group)
        demand = max(int(self.coarse_demand.get(coarse_key, group.demand)), int(group.demand))
        quota = self.quota_by_key.get(self._quota_key(group, area_no), 0)
        height_cap = self.area_size_height_cap.get((area_no, group.size, group.height), 0)
        useful_cap = min(quota, height_cap)
        return (0 if useful_cap >= demand else 1, -useful_cap, -quota, area_no)

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

    def _make_medium_rows(self, small_rows: list[dict]) -> list[dict]:
        counter: Counter[tuple[str, str, str, str, str]] = Counter()
        for row in small_rows:
            key = (
                str(row["voyage_id"]),
                str(row["flow"]),
                str(row["port"]),
                str(row["size"]),
                str(row["area_no"]),
            )
            counter[key] += int(row["planned_boxes"])
        rows: list[dict] = []
        for (voyage_id, flow, port, size, area_no), qty in sorted(counter.items()):
            window_start, window_end = self.problem.voyage_windows[voyage_id]
            rows.append(
                {
                    "plan_level": "medium_from_small",
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

    def _make_original_medium_rows(self, selected: Counter[int]) -> list[dict]:
        selected_coarse_area_weights: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        selected_area_weights: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        small_lower_by_coarse_area: Counter[tuple[str, str, str, str, str]] = Counter()
        small_lower_by_coarse: Counter[tuple[str, str, str, str]] = Counter()
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
                small_lower_by_coarse[col.coarse_key] += lower_qty

        remaining_quota: Counter[tuple[str, str, str, str]] = Counter(self.quota_by_key)
        counter: Counter[tuple[str, str, str, str, str]] = Counter()
        for (voyage_id, flow, port, size, area_no), qty in small_lower_by_coarse_area.items():
            if qty <= 0:
                continue
            counter[(voyage_id, flow, port, size, area_no)] += qty
            remaining_quota[(voyage_id, flow, area_no, self._big_plan_size(size))] -= qty

        medium_target_by_coarse: Counter[tuple[str, str, str, str]] = Counter()
        representative_by_coarse: dict[tuple[str, str, str, str], object] = {}
        for group in self.problem.groups:
            coarse_key = (group.voyage_id, group.status, group.port, group.size)
            medium_target_by_coarse[coarse_key] += int(group.demand)
            representative_by_coarse.setdefault(coarse_key, group)

        min_boxes = max(0, int(self.config.medium_large_group_min_area_boxes or 0))
        concentration_threshold = max(0, int(self.config.medium_concentrated_group_threshold or 0))
        sorted_groups = sorted(
            representative_by_coarse.values(),
            key=lambda g: (
                g.voyage_id,
                g.status,
                SIZE_ORDER.get(g.size, 3),
                0 if concentration_threshold > 0 and int(medium_target_by_coarse[(g.voyage_id, g.status, g.port, g.size)]) <= concentration_threshold else 1,
                (
                    int(medium_target_by_coarse[(g.voyage_id, g.status, g.port, g.size)])
                    if concentration_threshold > 0 and int(medium_target_by_coarse[(g.voyage_id, g.status, g.port, g.size)]) <= concentration_threshold
                    else -int(medium_target_by_coarse[(g.voyage_id, g.status, g.port, g.size)])
                ),
                g.port,
                g.group_id,
            ),
        )
        for group in sorted_groups:
            big_size = self._big_plan_size(group.size)
            coarse_key = (group.voyage_id, group.status, group.port, group.size)
            lower_total = int(small_lower_by_coarse.get(coarse_key, 0))
            target_total = max(int(medium_target_by_coarse.get(coarse_key, 0)), lower_total)
            remaining_target = target_total - lower_total
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
            allocation = self._allocate_area_quantity(
                remaining_target,
                weights,
                caps,
                min_boxes=min_boxes,
                concentrate=(concentration_threshold > 0 and target_total <= concentration_threshold),
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
        big_count = sum(1 for key in bay_keys if self.bays[key].cap_by_size.get("40", 0) > 0 or self.bays[key].cap_by_size.get("45", 0) > 0)
        small_flags = [self.bays[key].cap_by_size.get("20", 0) > 0 for key in bay_keys]
        return big_count >= 2 and sum(small_flags) >= 2 and any(a and b for a, b in zip(small_flags, small_flags[1:]))

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
            "voyage_id": col.voyage_id,
            "flow": col.flow,
            "port": col.port,
            "size": col.size,
            "height": col.height,
            "area_no": col.area_no,
            "bay_no": col.bay_no,
            "six_bay_block_id": col.block_id,
            "quantity": col.quantity,
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
