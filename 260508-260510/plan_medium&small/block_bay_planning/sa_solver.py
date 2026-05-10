from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from .bay_plan_solver import BayPlanSolverMixin
from .block_plan_solver import BlockPlanSolverMixin
from .models import Bay, ProblemData, SAConfig, SolveResult, Unit


class SimulatedAnnealingSolver(BlockPlanSolverMixin, BayPlanSolverMixin):
    """中计划/小计划两阶段求解器。

    第一阶段调用 `BlockPlanSolverMixin`，在大计划给定箱区内把箱子分到区块；
    第二阶段调用 `BayPlanSolverMixin`，在中计划选出的区块内把大属性组分到贝位。

    这个类本身主要负责公共索引、求解编排、结果输出和诊断信息。
    """

    def __init__(self, problem: ProblemData, config: SAConfig | None = None) -> None:
        """预计算区块/贝位索引。

        模拟退火会频繁查询“某箱区有哪些区块”“某区块有哪些贝位”“某区块的
        20/40 容量是多少”，所以初始化时先把这些关系整理成字典，避免
        在每个邻域动作里反复扫描全量贝位。
        """
        self.problem = problem
        self.config = config or SAConfig()
        self.random = random.Random(self.config.seed)
        self.units = problem.units
        self.bays = problem.bays
        self.bays_by_area: dict[str, list[str]] = defaultdict(list)
        self.bays_by_block: dict[str, list[str]] = defaultdict(list)
        self.blocks_by_area: dict[str, list[str]] = defaultdict(list)
        self.block_area: dict[str, str] = {}
        self.block_bays: dict[str, tuple[str, ...]] = {}
        self.block_boundary_adjusted: dict[str, bool] = {}
        self.block_order: dict[str, int] = {}
        self.area_edge_blocks: dict[str, set[str]] = {}
        self.area_edge_bays: dict[str, set[str]] = {}
        self.block_total_cap: Counter[str] = Counter()
        self.block_size_cap: Counter[tuple[str, str]] = Counter()

        for key, bay in problem.bays.items():
            self.bays_by_area[bay.area_no].append(key)
            self.bays_by_block[bay.block_id].append(key)
            self.block_area[bay.block_id] = bay.area_no
            self.block_bays[bay.block_id] = bay.block_bays
            self.block_boundary_adjusted[bay.block_id] = bay.block_boundary_adjusted
            self.block_total_cap[bay.block_id] += bay.physical_capacity
            for size_mode, cap in bay.cap_by_size.items():
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

    def solve(self) -> SolveResult:
        """依次求解中计划和小计划，并生成 CSV 行。"""
        medium_assignment = self._solve_medium()
        medium_assignment, small_assignment = self._solve_small(medium_assignment)
        medium_rows, small_rows = self._make_outputs(medium_assignment, small_assignment)
        diagnostics = {
            "iterations": self.config.iterations,
            "medium_energy": round(self.medium_energy(medium_assignment), 4),
            "small_energy": round(self.small_energy(medium_assignment, small_assignment), 4),
            "unit_count": len(self.units),
            "group_count": len(self.problem.groups),
            "bay_count": len(self.problem.bays),
            "block_count": len(self.bays_by_block),
            "medium_row_count": len(medium_rows),
            "small_row_count": len(small_rows),
            "planning_hierarchy": "medium plan assigns attributes to yard blocks first; small plan assigns large attributes to bays inside selected blocks",
            "block_policy": "preferred six-20ft-small-bay blocks; boundaries are adjusted to avoid splitting inferred 40ft bay pairs; block concentration is a soft preference",
            "large_box_edge_policy": "small-plan hard constraint: the first and last bay of an area cannot receive 20ft boxes",
            "size_overlap_policy": "20/40 capacities are size-specific; 45ft boxes consume 40ft capacity; total bay load also cannot exceed unsplit physical capacity",
            "size_modes": ["20", "40"],
            "big_plan_size_modes": ["20", "40"],
            "big_plan_size_policy": "45ft boxes consume the 40ft quota in both medium and small plans",
            "area_size_quota_count": len(self.problem.area_size_quota),
            "small_plan_threshold": self.problem.small_plan_threshold,
            "big_plan_volume_policy": "medium and small plans use 100% of the big-plan volume; no 70% rescaling is applied",
            "voyage_windows": {
                voyage_id: {
                    "window_start": window[0].isoformat(sep=" "),
                    "window_end": window[1].isoformat(sep=" "),
                }
                for voyage_id, window in self.problem.voyage_windows.items()
            },
            "operation_conflict_policy": (
                "medium-plan soft constraint: avoid blocks in areas with loading during the voyage window; "
                "prefer areas with loading in the 24h after the voyage window"
            ),
            "business_special_codes": sorted(self.problem.business_special_codes),
            "planning_time": self.problem.planning_time.isoformat(sep=" "),
            "horizon_hours": self.problem.horizon_hours,
            "target_voyages": self.problem.target_voyages,
        }
        return SolveResult(small_assignment, diagnostics["medium_energy"] + diagnostics["small_energy"], medium_rows, small_rows, diagnostics)

    def _temperature(self, iteration: int) -> float:
        ratio = iteration / max(1, self.config.iterations)
        return self.config.initial_temperature * (
            self.config.final_temperature / self.config.initial_temperature
        ) ** ratio

    def _is_area_edge_block(self, block_id: str) -> bool:
        area_no = self.block_area.get(block_id, "")
        return block_id in self.area_edge_blocks.get(area_no, set())

    def _is_area_edge_bay(self, bay_key: str) -> bool:
        bay = self.bays[bay_key]
        return bay_key in self.area_edge_bays.get(bay.area_no, set())

    def _make_outputs(self, medium_assignment: list[str], small_assignment: list[str]) -> tuple[list[dict], list[dict]]:
        """汇总中计划和小计划输出行。

        输出中同时保留两个尺寸字段：
        - `size`：箱明细原始尺寸，可能为 20/40/45；
        - `big_plan_size`：求解尺寸口径，只会是 20 或 40，45 会显示为 40。
        这样可以一眼核对中小计划是否严格消耗了大计划的尺寸配额。
        """
        medium_counter: Counter[tuple] = Counter()
        small_counter: Counter[tuple] = Counter()
        group_map = {g.group_id: g for g in self.problem.groups}
        voyage_area_blocks: defaultdict[tuple[str, str], set[int]] = defaultdict(set)

        for unit, block_id in zip(self.units, medium_assignment):
            group = unit.group
            area_no = self.block_area[block_id]
            voyage_area_blocks[(group.voyage_id, area_no)].add(self._block_index(block_id))
            attrs = self._attrs(group)
            medium_counter[attrs + (area_no, block_id)] += 1

        for unit, block_id, bay_key in zip(self.units, medium_assignment, small_assignment):
            group = unit.group
            if not group.requires_small_plan(self.problem.small_plan_threshold):
                continue
            area_no = self.block_area[block_id]
            bay = self.bays[bay_key]
            attrs = self._attrs(group)
            small_counter[attrs + (area_no, block_id, bay.bay_no)] += 1

        medium_rows = []
        for key, count in sorted(medium_counter.items()):
            (
                voyage_id,
                group_id,
                size,
                big_plan_size,
                height,
                status,
                port,
                operator,
                ctype,
                special,
                business_special,
                is_large_attribute,
                area_no,
                block_id,
            ) = key
            block_metrics = self._voyage_area_block_metrics(voyage_area_blocks[(voyage_id, area_no)])
            window_start, window_end = self.problem.voyage_windows[voyage_id]
            active_loading = self._area_has_loading_during_window(voyage_id, area_no)
            post_loading = self._area_has_loading_after_window(voyage_id, area_no)
            medium_rows.append(
                {
                    "plan_level": "medium",
                    "voyage_id": voyage_id,
                    "group_id": group_id,
                    "size": size,
                    "big_plan_size": big_plan_size,
                    "height": height,
                    "status": status,
                    "port": port,
                    "operator": operator,
                    "ctype": ctype,
                    "special": special,
                    "business_special": business_special,
                    "is_large_attribute": is_large_attribute,
                    "attribute_quantity_threshold": self.problem.small_plan_threshold,
                    "window_start": window_start.isoformat(sep=" "),
                    "window_end": window_end.isoformat(sep=" "),
                    "area_loading_during_window": active_loading,
                    "area_loading_after_window_24h": post_loading,
                    "area_no": area_no,
                    "yard_block": block_id,
                    "yard_block_bays": self._block_bay_label(block_id),
                    "yard_block_bay_count": len(self.block_bays.get(block_id, ())),
                    "yard_block_boundary_adjusted": self.block_boundary_adjusted.get(block_id, False),
                    "voyage_area_block_count": block_metrics["count"],
                    "voyage_area_block_span": block_metrics["span"],
                    "voyage_area_block_gap_count": block_metrics["gaps"],
                    "bay_no": "",
                    "is_fallback_bay": "",
                    "fallback_reason": "",
                    "plan_reason": self._plan_reason(is_large_attribute, special, "medium"),
                    "planned_boxes": count,
                    "group_demand": group_map[group_id].demand,
                }
            )

        small_rows = []
        for key, count in sorted(small_counter.items()):
            (
                voyage_id,
                group_id,
                size,
                big_plan_size,
                height,
                status,
                port,
                operator,
                ctype,
                special,
                business_special,
                is_large_attribute,
                area_no,
                block_id,
                bay_no,
            ) = key
            bay = self.bays[f"{area_no}-{bay_no}"]
            block_metrics = self._voyage_area_block_metrics(voyage_area_blocks[(voyage_id, area_no)])
            window_start, window_end = self.problem.voyage_windows[voyage_id]
            active_loading = self._area_has_loading_during_window(voyage_id, area_no)
            post_loading = self._area_has_loading_after_window(voyage_id, area_no)
            small_rows.append(
                {
                    "plan_level": "small",
                    "voyage_id": voyage_id,
                    "group_id": group_id,
                    "size": size,
                    "big_plan_size": big_plan_size,
                    "height": height,
                    "status": status,
                    "port": port,
                    "operator": operator,
                    "ctype": ctype,
                    "special": special,
                    "business_special": business_special,
                    "is_large_attribute": is_large_attribute,
                    "attribute_quantity_threshold": self.problem.small_plan_threshold,
                    "window_start": window_start.isoformat(sep=" "),
                    "window_end": window_end.isoformat(sep=" "),
                    "area_loading_during_window": active_loading,
                    "area_loading_after_window_24h": post_loading,
                    "area_no": area_no,
                    "yard_block": block_id,
                    "yard_block_bays": self._block_bay_label(block_id),
                    "yard_block_bay_count": len(self.block_bays.get(block_id, ())),
                    "yard_block_boundary_adjusted": self.block_boundary_adjusted.get(block_id, False),
                    "voyage_area_block_count": block_metrics["count"],
                    "voyage_area_block_span": block_metrics["span"],
                    "voyage_area_block_gap_count": block_metrics["gaps"],
                    "bay_no": bay_no,
                    "is_fallback_bay": bay.is_fallback_bay,
                    "fallback_reason": "|".join(sorted(bay.fallback_reasons)),
                    "plan_reason": self._plan_reason(is_large_attribute, special, "small"),
                    "planned_boxes": count,
                    "group_demand": group_map[group_id].demand,
                }
            )
        return medium_rows, small_rows

    def _attrs(self, group) -> tuple:
        is_large_attribute = group.requires_small_plan(self.problem.small_plan_threshold)
        return (
            group.voyage_id,
            group.group_id,
            group.size,
            group.big_plan_size_mode,
            group.height,
            group.status,
            group.port,
            group.operator,
            group.ctype,
            group.special_signature,
            group.business_special_signature,
            is_large_attribute,
        )

    def _block_bay_label(self, block_id: str) -> str:
        return ",".join(self.block_bays.get(block_id, ()))

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

    @staticmethod
    def _voyage_area_block_metrics(blocks: set[int]) -> dict[str, int]:
        if not blocks:
            return {"count": 0, "span": 0, "gaps": 0}
        span = max(blocks) - min(blocks) + 1
        return {"count": len(blocks), "span": span, "gaps": span - len(blocks)}

    def _plan_reason(self, is_large_attribute: bool, special: str, level: str) -> str:
        reasons = ["large_attribute_to_bay" if level == "small" else "attribute_to_yard_block"]
        if is_large_attribute:
            reasons.append("large_attribute")
        if special != "NORMAL":
            reasons.append("special_stow_soft_preference")
        return ";".join(reasons)

    @staticmethod
    def _block_index(block_id: str) -> int:
        try:
            return int(block_id.rsplit("B", 1)[1])
        except (IndexError, ValueError):
            return 0


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
