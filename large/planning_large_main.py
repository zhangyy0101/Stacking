from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from planning_large_solver import (
    DailyRollingYardPlanningData,
    DailyRollingYardPlanningSolution,
    YardPlanningWeights,
    solve_daily_rolling_yard_plan,
)


DEFAULT_PLANNING_TIME = "2026-05-19 09:30:00"
DEFAULT_EXPORT_VESSELS = None
DEFAULT_IMPORT_VESSELS = None
KNOWN_EXPORT_SNAPSHOT_FLOWS = {"OF", "OZ", "T"}
UNKNOWN_EXPORT_SNAPSHOT_FLOW_FALLBACK = "OF"
BERTH_CONFLICT_THRESHOLD_HOURS = 2.0

# 进口资料箱中存在 IE/RF/RE，但当前箱区功能表没有这些功能。
# 这里默认将它们映射为 OZ，与 flat_full_yard_plan_1.4_scip 的大计划口径保持一致。
DEFAULT_FLOW_ALIASES = {
    "IE": "OZ",
    "RF": "OZ",
    "RE": "OZ",
}


@dataclass(frozen=True)
class PlanningInputArtifacts:
    """参数构造层：包含求解器输入、滚动状态、诊断信息和辅助业务信息。"""

    data: DailyRollingYardPlanningData
    planning_time: pd.Timestamp
    export_vessels: list[str]
    import_vessels: list[str]
    area_functions: dict[str, set[str]]
    berth_by_vessel: dict[str, str]
    previous_plan_rows: pd.DataFrame
    diagnostics: dict[str, Any]


class RollingPlanningState:
    """
    滚动规划状态管理器。

    当前用 CSV 保存每次规划输出的箱区级总计划。下一天滚动规划时，
    这里会读取上一版计划并构造模型中的 P20/P40/O：
    - P：上一规划节点的总计划；
    - O：是否旧航次，旧航次会进入调整幅度惩罚。
    """

    def __init__(self, state_dir: Path) -> None:
        """
        功能：
            初始化滚动规划状态管理器，并确定历史计划 CSV 的存储路径。

        参数：
            state_dir: 状态文件所在目录。

        返回：
            无。
        """
        self.state_dir = state_dir
        self.plan_history_path = state_dir / "plan_history.csv"

    def read_history(self) -> pd.DataFrame:
        """
        功能：
            读取历史规划状态表；若状态文件不存在，则返回包含固定列名的空表。

        参数：
            无。

        返回：
            历史计划 DataFrame，包含规划时间、航次、流向、箱区、箱型和计划箱量等字段。

        异常：
            pandas 读取 CSV 失败时会透传相应异常。
        """

        if not self.plan_history_path.exists():
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "planning_time",
                    "voy_id",
                    "flow",
                    "area_no",
                    "size",
                    "planned_qty",
                    "status_name",
                    "objective_value",
                ]
            )
        history = pd.read_csv(self.plan_history_path)
        if "planning_time" in history.columns:
            history["planning_time"] = pd.to_datetime(history["planning_time"], errors="coerce")
        return history

    def latest_previous_plan(
        self,
        planning_time: pd.Timestamp,
        vessels: Sequence[str],
    ) -> pd.DataFrame:
        """
        功能：
            读取每个航次在当前规划时间之前的最新一版历史计划。

        参数：
            planning_time: 当前规划节点时间。
            vessels: 需要查询历史计划的航次集合。

        返回：
            每个航次在 ``planning_time`` 之前最新规划时间对应的计划记录。

        异常：
            pandas 分组、时间比较或历史文件读取失败时会透传相应异常。
        """

        history = self.read_history()
        if history.empty:
            return history
        history = history[history["voy_id"].map(normalize_code).isin(set(vessels))].copy()
        history = history[history["planning_time"] < planning_time].copy()
        if history.empty:
            return history
        latest_by_vessel = history.groupby("voy_id")["planning_time"].transform("max")
        return history[history["planning_time"] == latest_by_vessel].copy()

    def build_previous_plan_params(
        self,
        planning_time: pd.Timestamp,
        vessels: Sequence[str],
    ) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str, str], float], dict[str, int], pd.DataFrame]:
        """
        功能：
            将历史计划表转换成求解器参数 P20、P40 和 O。

        参数：
            planning_time: 当前规划节点时间。
            vessels: 需要构造历史计划参数的航次集合。

        返回：
            四元组 ``(p20, p40, old_flags, previous)``。其中 ``p20`` 和 ``p40``
            的键为 ``(voy_id, flow, area_no)``，``old_flags`` 表示航次是否存在
            上一版计划，``previous`` 为用于构造参数的历史计划明细。

        异常：
            历史计划读取、字段转换或 pandas 分组失败时会透传相应异常。
        """

        previous = self.latest_previous_plan(planning_time, vessels)
        p20: dict[tuple[str, str, str], float] = {}
        p40: dict[tuple[str, str, str], float] = {}
        old_flags = {v: 0 for v in vessels}
        if previous.empty:
            return p20, p40, old_flags, previous

        previous = previous.copy()
        previous["voy_id"] = previous["voy_id"].map(normalize_code)
        previous["flow"] = previous["flow"].map(normalize_code)
        previous["area_no"] = previous["area_no"].map(normalize_code)
        previous["size"] = previous["size"].map(normalize_size)
        previous["planned_qty"] = pd.to_numeric(previous["planned_qty"], errors="coerce").fillna(0.0)

        for vessel in previous["voy_id"].dropna().unique():
            old_flags[vessel] = 1

        grouped = previous.groupby(["voy_id", "flow", "area_no", "size"], dropna=False)["planned_qty"].sum()
        for (vessel, flow, area, size), qty in grouped.items():
            if not vessel or not flow or not area:
                continue
            key = (str(vessel), str(flow), str(area))
            if size == "20":
                p20[key] = float(qty)
            elif size == "40":
                p40[key] = float(qty)
        return p20, p40, old_flags, previous

    def append_solution(
        self,
        planning_time: pd.Timestamp,
        solution: DailyRollingYardPlanningSolution,
    ) -> pd.DataFrame:
        """
        功能：
            将本次求解得到的非零总计划写入滚动状态表。

        参数：
            planning_time: 当前规划节点时间。
            solution: 求解器返回的规划结果。

        返回：
            本次追加到状态表的新计划记录。若求解结果没有非零计划量，则返回空表。

        异常：
            创建目录、读取历史状态或写入 CSV 失败时会透传相应异常。
        """

        rows = []
        run_id = uuid.uuid4().hex[:12]
        for size, values in (("20", solution.x20), ("40", solution.x40)):
            for (vessel, flow, area), qty in values.items():
                if qty <= 0:
                    continue
                rows.append(
                    {
                        "run_id": run_id,
                        "planning_time": planning_time.isoformat(),
                        "voy_id": vessel,
                        "flow": flow,
                        "area_no": area,
                        "size": size,
                        "planned_qty": int(qty),
                        "status_name": solution.status_name,
                        "objective_value": solution.objective_value,
                    }
                )
        new_rows = pd.DataFrame(rows)
        if new_rows.empty:
            return new_rows
        self.state_dir.mkdir(parents=True, exist_ok=True)
        old = self.read_history()
        if not old.empty:
            old["planning_time"] = pd.to_datetime(old["planning_time"], errors="coerce")
            replacing_vessels = set(new_rows["voy_id"].map(normalize_code))
            old = old[
                ~(
                    old["planning_time"].eq(planning_time)
                    & old["voy_id"].map(normalize_code).isin(replacing_vessels)
                )
            ].copy()
        combined = pd.concat([old, new_rows], ignore_index=True)
        combined.to_csv(self.plan_history_path, index=False, encoding="utf-8-sig")
        return new_rows


def build_current_case_data(
    *,
    base_dir: Path,
    data_dir: Optional[Path],
    planning_time: pd.Timestamp,
    export_vessels: Optional[Sequence[str]],
    import_vessels: Optional[Sequence[str]],
    state: RollingPlanningState,
    flow_aliases: Optional[Mapping[str, str]] = None,
    user_design: bool = False,
    user_design_large_plan_area: Optional[Sequence[str]] = None,
    adjust_plan_info: Optional[Mapping[str, Any]] = None,
) -> PlanningInputArtifacts:
    """
    功能：
        构造当前规划节点的完整模型输入，是 pipeline 的主入口。

    参数：
        base_dir: 项目或脚本所在基础目录。
        data_dir: 业务数据目录；为 ``None`` 时会自动发现默认数据目录。
        planning_time: 当前规划节点时间。
        export_vessels: 本次参与规划的出口航次集合。
        import_vessels: 本次参与规划的进口航次集合。
        state: 滚动规划状态管理器，用于读取 P/O 历史参数。
        flow_aliases: 作业流向别名映射，例如将 IE/RF/RE 映射为 IF。

    返回：
        ``PlanningInputArtifacts``，包含求解器输入、规划时间、航次集合、
        箱区功能、泊位信息、上一版计划和诊断信息。

    异常：
        FileNotFoundError: 数据目录、箱区功能表或距离矩阵未找到时抛出。
        KeyError: 必需业务字段、泊位、箱区或距离矩阵列缺失时抛出。
        pandas 读取 parquet、CSV、Excel 或数据转换失败时会透传相应异常。
    """

    data_dir = discover_data_dir(base_dir, data_dir)
    flow_aliases = {normalize_code(k): normalize_code(v) for k, v in (flow_aliases or {}).items()}

    # 当前要规划的航次集合。出口和进口在数据来源上不同，但求解器中统一进入 V。
    if export_vessels is None:
        export_vessels = discover_export_vessels(data_dir)
    if import_vessels is None:
        import_vessels = discover_import_vessels(data_dir)
    export_vessels = normalize_vessel_list(export_vessels)
    import_vessels = normalize_vessel_list(import_vessels)
    all_vessels = export_vessels + import_vessels

    # 基础静态参数：泊位距离、箱区功能、箱区作业能力。
    vessel_info = read_vessel_info(data_dir / "vessel_berth_info_new.csv")
    area_file = discover_area_function_file(data_dir)
    areas, area_functions, load_capacity = read_area_functions(area_file)
    distance = read_distance_matrix(
        discover_distance_matrix(base_dir),
        areas,
        read_berths_for_vessels(vessel_info, all_vessels),
    )
    berth_by_vessel = read_berths_for_vessels(vessel_info, all_vessels)
    allowed_areas_by_vessel, required_areas_by_vessel, area_control_diagnostics = build_large_area_controls(
        vessels=all_vessels,
        areas=areas,
        user_design=user_design,
        user_design_large_plan_area=user_design_large_plan_area or [],
        adjust_plan_info=adjust_plan_info or {},
    )
    user_design_active = bool(area_control_diagnostics["user_design_active"])

    # 堆场快照：识别当前规划航次已经在场的箱子。
    snapshot = read_snapshot(data_dir / "bay_slots_detail.parquet")
    current_snapshot = extract_current_snapshot_rows(
        snapshot,
        export_vessels=export_vessels,
        import_vessels=import_vessels,
        flow_aliases=flow_aliases,
    )

    # 异常贝位：
    # 1. 当前出口航次已在场箱若流向与箱区功能不匹配，则记为异常关联箱；
    # 2. 若某贝位的异常关联箱占该贝位总箱位行数超过 2/3，则关闭整个贝位；
    # 3. L/Q 按箱号去重：流向匹配箱区功能的进入 L，不匹配的进入 Q。
    bay_total_slots = build_bay_total_slot_counts(snapshot, areas)
    bad_bays = identify_bad_bays(current_snapshot, area_functions, bay_total_slots)
    l20, l40, q20, q40 = build_snapshot_count_params(current_snapshot, area_functions, set(areas))
    if user_design_active:
        departure_deductions = {area: 0.0 for area in areas}
        departure_avoidance = {
            "active_export_voyages": [],
            "counts_by_voyage_area": [],
            "disabled_reason": "user_design_large_plan_area",
        }
        effective_load_capacity = dict(load_capacity)
        close_berth_pairs: list[tuple[str, str]] = []
        close_berth_diagnostics: list[dict[str, Any]] = []
    else:
        departure_deductions, departure_avoidance = compute_departure_operation_deductions(
            vessel_info=vessel_info,
            snapshot=snapshot,
            planning_time=planning_time,
            
            areas=areas,
        )
        effective_load_capacity = {
            area: max(0.0, float(load_capacity.get(area, 0.0)) - float(departure_deductions.get(area, 0.0)))
            for area in areas
        }
        
        close_berth_pairs, close_berth_diagnostics = build_close_export_berth_pairs(
            vessel_info=vessel_info,
            export_vessels=export_vessels,
            threshold_hours=BERTH_CONFLICT_THRESHOLD_HOURS,
        )

    # 容量统计：
    # 只读取未拆分 bay_slots_detail.parquet。C20 使用未拆分物理箱位行数口径；
    # C20Direct 使用 YBY_ENABLECSIZECD 包含 20 的空箱位；
    # C40 使用 YBY_ENABLECSIZECD 包含 40/45 的空箱位，45 按 40 处理。
    # 三者都会剔除已占用箱位和达到 2/3 阈值后关闭的异常贝位。
    available_slots = prepare_slot_frame(snapshot, areas, bad_bays)
    bay20_equiv = available_slots
    bay20_direct = available_slots[available_slots["enable_20"]].copy()
    bay40 = available_slots[available_slots["enable_40"]].copy()
    c20 = count_slots_by_area(bay20_equiv, areas)
    c20_direct = count_slots_by_area(bay20_direct, areas)
    c40 = count_slots_by_area(bay40, areas)

    # TOPS 扣减：只看 SPL_STDATE/SPL_EDDATE 判断计划是否生效。
    # 若生效 TOPS 的 SPL_CONDITIONCODE 不是当前待规划航次，则扣除 SPR_STBAY~SPR_EDBAY
    # 覆盖的完整贝位。为与容量口径一致，分别扣 C20、C20Direct、C40。
    tops = pd.read_parquet(data_dir / "tops_plan_info.parquet")
    tops20, tops20_direct, tops40, active_tops_count = compute_tops_capacity_deductions(
        tops,
        planning_time,
        all_vessels,
        bay20_equiv,
        bay20_direct,
        bay40,
    )
    cbar20 = {(v, a): max(0.0, c20.get(a, 0.0) - tops20.get((v, a), 0.0)) for v in all_vessels for a in areas}
    cbar20_direct = {
        (v, a): max(0.0, c20_direct.get(a, 0.0) - tops20_direct.get((v, a), 0.0))
        for v in all_vessels
        for a in areas
    }
    cbar40 = {(v, a): max(0.0, c40.get(a, 0.0) - tops40.get((v, a), 0.0)) for v in all_vessels for a in areas}

    # 需求 D：出口 = 资料箱/快照箱去重 + 预估超出部分补 OF；进口 = 资料箱 + 快照补全。
    d20, d40, demand_diagnostics = build_demand_params(
        data_dir,
        export_vessels=export_vessels,
        import_vessels=import_vessels,
        current_snapshot=current_snapshot,
        model_areas=set(areas),
        flow_aliases=flow_aliases,
    )
    of_work_lanes = {
        vessel: read_prediction_work_lanes(data_dir / f"predict_data_{vessel}.xlsx")
        if (data_dir / f"predict_data_{vessel}.xlsx").exists()
        else 0.0
        for vessel in export_vessels
    }
    of_work_lanes.update({vessel: 0.0 for vessel in import_vessels})

    # 流向集合 F：取箱区功能、需求、快照箱流向的并集。
    flows = sorted(
        {
            flow
            for funcs in area_functions.values()
            for flow in funcs
        }
        | {flow for _, flow in d20}
        | {flow for _, flow in d40}
        | {flow for _, flow, _ in l20}
        | {flow for _, flow, _ in l40}
        | {flow for _, flow, _ in q20}
        | {flow for _, flow, _ in q40}
    )

    # U/E：箱区用途和可用性。
    # U 只看功能是否允许；E 还会检查 TOPS 后容量是否为正。
    u = {(a, f): int(area_allows_flow(a, f, area_functions)) for a in areas for f in flows}
    e20, e40 = build_availability_flags(
        vessels=all_vessels,
        flows=flows,
        areas=areas,
        area_functions=area_functions,
        cbar20=cbar20,
        cbar20_direct=cbar20_direct,
        cbar40=cbar40,
    )

    # 历史计划 P 和新旧航次标记 O。
    p20, p40, old_flags, previous_rows = state.build_previous_plan_params(planning_time, all_vessels)

    # 汇总所有集合和参数，交给 solver 建模。
    model_data = DailyRollingYardPlanningData(
        V=all_vessels,
        F=flows,
        A=areas,
        D20=d20,
        D40=d40,
        L20=l20,
        L40=l40,
        Q20=q20,
        Q40=q40,
        C20=c20,
        C20Direct=c20_direct,
        C40=c40,
        Cbar20=cbar20,
        Cbar20Direct=cbar20_direct,
        Cbar40=cbar40,
        H=effective_load_capacity,
        distance=distance,
        U=u,
        E20=e20,
        E40=e40,
        P20=p20,
        P40=p40,
        O=old_flags,
        OFWorkLanes=of_work_lanes,
        berth_conflict_pairs=close_berth_pairs,
        allowed_areas_by_vessel=allowed_areas_by_vessel,
        required_areas_by_vessel=required_areas_by_vessel,
        weights=YardPlanningWeights(
            miss=100.0,
            operation=50.0,
            of_area=40.0,
            distance=30.0,
            share=20.0,
            berth_conflict=25.0,
            adjustment=10.0,
            balance=1.0,
        ),
        allow_unmet_demand=True,
        strict_validation=True,
    )

    diagnostics = {
        "data_dir": str(data_dir),
        "area_count": len(areas),
        "flows": flows,
        "flow_aliases": flow_aliases,
        **area_control_diagnostics,
        "bad_bay_count": len(bad_bays),
        "bad_bay_sample": sorted(list(bad_bays))[:20],
        "current_snapshot_rows": int(len(current_snapshot)),
        "active_tops_rows": int(active_tops_count),
        "capacity20_total": float(sum(c20.values())),
        "capacity20_direct_total": float(sum(c20_direct.values())),
        "capacity40_total": float(sum(c40.values())),
        "capacity20_if_total": float(sum(c20[a] for a in areas if "IF" in area_functions.get(a, set()))),
        "capacity20_direct_if_total": float(sum(c20_direct[a] for a in areas if "IF" in area_functions.get(a, set()))),
        "capacity40_if_total": float(sum(c40[a] for a in areas if "IF" in area_functions.get(a, set()))),
        "load_capacity_original_total": float(sum(load_capacity.values())),
        "load_capacity_effective_total": float(sum(effective_load_capacity.values())),
        "departure_operation_deduction_total": float(sum(departure_deductions.values())),
        "departure_operation_deductions_by_area": departure_deductions,
        "departure_operation_avoidance": departure_avoidance,
        "close_berth_conflict_threshold_hours": BERTH_CONFLICT_THRESHOLD_HOURS,
        "close_berth_conflict_pairs": [list(pair) for pair in close_berth_pairs],
        "close_berth_conflict_pair_details": close_berth_diagnostics,
        "cbar20_min": float(min(cbar20.values()) if cbar20 else 0.0),
        "cbar20_direct_min": float(min(cbar20_direct.values()) if cbar20_direct else 0.0),
        "cbar40_min": float(min(cbar40.values()) if cbar40 else 0.0),
        "old_vessels": sorted([v for v, flag in old_flags.items() if flag]),
        "of_work_lanes": of_work_lanes,
        "of_area_limits": {vessel: 2.0 * lanes for vessel, lanes in of_work_lanes.items()},
        "demand": demand_diagnostics,
    }
    return PlanningInputArtifacts(
        data=model_data,
        planning_time=planning_time,
        export_vessels=export_vessels,
        import_vessels=import_vessels,
        area_functions=area_functions,
        berth_by_vessel=berth_by_vessel,
        previous_plan_rows=previous_rows,
        diagnostics=diagnostics,
    )


def project_root_from(base_dir: Path) -> Path:
    """
    Return the repository root for both the old flat layout and the new large/
    subdirectory layout.
    """

    base_dir = base_dir.resolve()
    if (base_dir / "large").is_dir() or (base_dir / "medium_small").is_dir():
        return base_dir
    if (base_dir.parent / "large").is_dir() or (base_dir.parent / "medium_small").is_dir():
        return base_dir.parent
    return base_dir


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _search_roots(base_dir: Path) -> list[Path]:
    project_root = project_root_from(base_dir)
    return _unique_paths([base_dir, project_root, project_root / "large", Path.cwd()])


def resolve_input_path(path: Path, base_dir: Path) -> Path:
    """
    Resolve a user supplied input path against cwd, script dir, and repo root.
    """

    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    candidates = _unique_paths([Path.cwd() / path, *[root / path for root in _search_roots(base_dir)]])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(path: Path, base_dir: Path) -> Path:
    """
    Keep relative outputs under the large/ script directory.
    """

    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _looks_like_planning_data_dir(path: Path) -> bool:
    core_required = [
        "vessel_berth_info_new.csv",
        "tops_plan_info.parquet",
        "bay_slots_detail.parquet",
    ]
    has_area_workbook = any(
        p.suffix.lower() in {".xlsx", ".xls"}
        and not p.name.startswith("~")
        and "predict" not in p.name.lower()
        for p in path.iterdir()
    )
    if all((path / name).exists() for name in core_required) and has_area_workbook:
        return True
    required = [
        "vessel_berth_info_new.csv",
        "tops_plan_info.parquet",
        "bay_slots_detail.parquet",
        "箱区功能.xlsx",
    ]
    return all((path / name).exists() for name in required)


def discover_data_dir(base_dir: Path, data_dir: Optional[Path]) -> Path:
    """
    功能：
        确定本次规划使用的数据目录。

    参数：
        base_dir: 自动搜索数据目录时使用的基础目录。
        data_dir: 用户显式指定的数据目录；不为空时直接返回该目录的绝对路径。

    返回：
        数据目录路径。

    异常：
        FileNotFoundError: 未显式指定目录且无法唯一找到包含 ``20260508`` 的数据目录时抛出。
    """
    if data_dir is not None:
        resolved = resolve_input_path(data_dir, base_dir)
        if not resolved.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {resolved}")
        return resolved

    candidates: list[Path] = []
    for root in _search_roots(base_dir):
        if not root.is_dir():
            continue
        candidates.extend(p for p in root.iterdir() if p.is_dir() and re.search(r"20\d{6}", p.name))
    candidates = _unique_paths(candidates)
    usable = [p for p in candidates if _looks_like_planning_data_dir(p)]
    if len(usable) == 1:
        return usable[0]
    if len(usable) > 1:
        return sorted(usable, key=lambda p: p.name)[-1]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"Expected one planning data directory, found: {candidates}")


def normalize_vessel_list(values: Optional[Sequence[str]]) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        vessel = normalize_code(value)
        if not vessel or vessel in seen:
            continue
        seen.add(vessel)
        out.append(vessel)
    return out


def normalize_area_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    seen: set[str] = set()
    try:
        iterator = list(values)
    except TypeError:
        iterator = [values]
    for value in iterator:
        area = normalize_code(value)
        if not area or area in seen:
            continue
        seen.add(area)
        out.append(area)
    return out


def large_plan_adjust_entry(adjust_plan_info: Mapping[str, Any] | None, vessel: str) -> Mapping[str, Any]:
    if not isinstance(adjust_plan_info, Mapping):
        return {}
    source = adjust_plan_info.get("adjust_plan_info", adjust_plan_info)
    if not isinstance(source, Mapping):
        return {}
    large_plan = source.get("large_plan", source)
    if not isinstance(large_plan, Mapping):
        return {}
    if "add" in large_plan or "remove" in large_plan:
        return large_plan
    return (
        large_plan.get(vessel)
        or large_plan.get(str(vessel))
        or large_plan.get(normalize_code(vessel))
        or {}
    )


def build_large_area_controls(
    *,
    vessels: Sequence[str],
    areas: Sequence[str],
    user_design: bool,
    user_design_large_plan_area: Sequence[str] | None,
    adjust_plan_info: Mapping[str, Any] | None,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
    area_set = set(areas)
    design_areas_raw = normalize_area_list(user_design_large_plan_area)
    design_areas = [area for area in design_areas_raw if area in area_set]
    active_user_design = bool(user_design and design_areas)

    allowed_by_vessel: dict[str, list[str]] = {}
    required_by_vessel: dict[str, list[str]] = {}
    per_vessel: dict[str, dict[str, Any]] = {}
    unknown_areas: dict[str, list[str]] = {}

    for vessel in vessels:
        entry = large_plan_adjust_entry(adjust_plan_info, vessel)
        add_raw = normalize_area_list(entry.get("add")) if isinstance(entry, Mapping) else []
        remove_raw = normalize_area_list(entry.get("remove")) if isinstance(entry, Mapping) else []
        add = [area for area in add_raw if area in area_set]
        remove = [area for area in remove_raw if area in area_set]
        unknown = sorted((set(add_raw) | set(remove_raw)) - area_set)
        if unknown:
            unknown_areas[vessel] = unknown

        allowed = set(design_areas if active_user_design else areas)
        allowed.update(add)
        allowed.difference_update(remove)
        required = sorted(set(add) & allowed)

        allowed_by_vessel[vessel] = sorted(allowed)
        if required:
            required_by_vessel[vessel] = required
        per_vessel[vessel] = {
            "allowed_count": len(allowed),
            "required_areas": required,
            "removed_areas": sorted(set(remove)),
        }

    diagnostics = {
        "user_design_requested": bool(user_design),
        "user_design_active": active_user_design,
        "user_design_large_plan_area": design_areas,
        "user_design_large_plan_area_ignored": sorted(set(design_areas_raw) - area_set),
        "adjust_plan_info_large_plan_present": bool(
            isinstance(adjust_plan_info, Mapping) and adjust_plan_info.get("large_plan", adjust_plan_info)
        ),
        "adjust_plan_unknown_areas": unknown_areas,
        "area_controls_by_vessel": per_vessel,
    }
    return allowed_by_vessel, required_by_vessel, diagnostics


def read_adjust_plan_info(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        return {}
    adjust_plan_info = data.get("adjust_plan_info", data)
    return adjust_plan_info if isinstance(adjust_plan_info, dict) else {}


def discover_export_vessels(data_dir: Path) -> list[str]:
    vessels: set[str] = set()
    for path in data_dir.glob("predict_data_*.xlsx"):
        match = re.fullmatch(r"predict_data_(.+)\.xlsx", path.name, flags=re.IGNORECASE)
        if match:
            vessel = normalize_code(match.group(1))
            if vessel:
                vessels.add(vessel)
    for path in data_dir.glob("container_info_*.parquet"):
        if path.name.lower().startswith("container_info_import"):
            continue
        match = re.fullmatch(r"container_info_(.+)\.parquet", path.name, flags=re.IGNORECASE)
        if match:
            vessel = normalize_code(match.group(1))
            if vessel:
                vessels.add(vessel)
    return sorted(vessels)


def discover_import_container_dir(data_dir: Path) -> Path:
    candidates: list[Path] = []
    for path in data_dir.iterdir():
        if not path.is_dir():
            continue
        if any(path.glob("container_info_import*.parquet")):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Could not find import container directory under {data_dir}.")
    return sorted(candidates, key=lambda p: p.name)[0]


def discover_import_vessels(data_dir: Path) -> list[str]:
    import_dir = discover_import_container_dir(data_dir)
    vessels: set[str] = set()
    for path in import_dir.glob("container_info_import*.parquet"):
        match = re.fullmatch(r"container_info_import(?:_voy)?_(.+)\.parquet", path.name, flags=re.IGNORECASE)
        if match:
            vessel = normalize_code(match.group(1))
            if vessel:
                vessels.add(vessel)
    return sorted(vessels)


def resolve_import_doc_path(import_dir: Path, vessel: str) -> Path:
    direct_candidates = [
        import_dir / f"container_info_import_{vessel}.parquet",
        import_dir / f"container_info_import_voy_{vessel}.parquet",
    ]
    for path in direct_candidates:
        if path.exists():
            return path
    for path in import_dir.glob("container_info_import*.parquet"):
        match = re.fullmatch(r"container_info_import(?:_voy)?_(.+)\.parquet", path.name, flags=re.IGNORECASE)
        if match and normalize_code(match.group(1)) == vessel:
            return path
    raise FileNotFoundError(f"Missing import container info for voyage {vessel} in {import_dir}.")


def discover_area_function_file(data_dir: Path) -> Path:
    """
    功能：
        在数据目录中自动发现箱区功能 Excel 文件。

    参数：
        data_dir: 业务数据目录。

    返回：
        箱区功能表文件路径。

    异常：
        FileNotFoundError: 未找到或找到多个候选 Excel 文件时抛出。
    """
    candidates = [
        p
        for p in data_dir.iterdir()
        if p.suffix.lower() in {".xlsx", ".xls"}
        and not p.name.startswith("~")
        and "predict" not in p.name.lower()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one area function workbook, found: {candidates}")
    return candidates[0]


def discover_distance_matrix(base_dir: Path) -> Path:
    """
    功能：
        在基础目录中自动发现泊位到箱区的距离矩阵工作簿。

    参数：
        base_dir: 自动搜索距离矩阵时使用的基础目录。
        
    返回：
        距离矩阵 Excel 文件路径。

    异常：
        FileNotFoundError: 未找到符合工作表数量特征的距离矩阵工作簿时抛出。
    """
    for root in _search_roots(base_dir):
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.suffix.lower() not in {".xlsx", ".xls"} or path.name.startswith("~"):
                continue
            try:
                xls = pd.ExcelFile(path)
            except Exception:
                continue
            if len(xls.sheet_names) >= 5:
                return path
    raise FileNotFoundError("Could not find the berth-area distance matrix workbook.")


def normalize_code(value: Any) -> Optional[str]:
    """
    功能：
        将业务编码统一清洗为去空格的大写字符串，并处理 Excel 数字编码的 ``.0`` 后缀。

    参数：
        value: 待清洗的原始值。

    返回：
        清洗后的编码；空值、NaN 或空字符串返回 ``None``。
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_size(value: Any) -> str:
    """
    功能：
        将箱尺寸编码标准化为模型使用的 20/40 两类。

    参数：
        value: 原始箱尺寸编码。

    返回：
        ``"20"``、``"40"`` 或空字符串。45 尺箱按 40 尺口径处理。
        空字符串在后续汇总中按非 20 尺处理，进入 40 尺分支。
    """
    code = normalize_code(value)
    if not code:
        return ""
    if code.startswith("20"):
        return "20"
    if code.startswith(("40", "45")):
        return "40"
    return ""


def normalize_flow(value: Any, aliases: Mapping[str, str]) -> Optional[str]:
    """
    功能：
        清洗作业流向编码，并应用流向别名映射。

    参数：
        value: 原始作业流向编码。
        aliases: 流向别名映射，键和值均为标准化后的流向编码。

    返回：
        映射后的作业流向；空值返回 ``None``。
    """
    flow = normalize_code(value)
    if not flow:
        return None
    return aliases.get(flow, flow)


def normalize_export_snapshot_flow(value: Any, aliases: Mapping[str, str]) -> Optional[str]:
    flow = normalize_flow(value, aliases)
    if flow and flow not in KNOWN_EXPORT_SNAPSHOT_FLOWS:
        return UNKNOWN_EXPORT_SNAPSHOT_FLOW_FALLBACK
    return flow


def medium_small_area_flow(flow: Any) -> str:
    normalized = normalize_flow(flow, {})
    if not normalized:
        return "OF"
    if normalized == "OF":
        return "OF"
    if normalized in {"IF", "IZ", "T"}:
        return normalized
    return "OZ"


def area_allows_flow(area: Any, flow: Any, area_functions: Mapping[str, set[str]]) -> bool:
    area_code = normalize_code(area)
    flow_code = normalize_code(flow)
    if not area_code or not flow_code:
        return False
    return flow_code in area_functions.get(area_code, set())


def normalize_bay(value: Any) -> Optional[str]:
    """
    功能：
        清洗贝位、排号或层号编码，并将纯数字编码补齐为两位字符串。

    参数：
        value: 原始贝位、排号或层号。

    返回：
        标准化后的编码；空值返回 ``None``。
    """
    code = normalize_code(value)
    if not code:
        return None
    if code.isdigit():
        return f"{int(code):02d}"
    return code


def normalize_int_string(value: Any) -> Optional[str]:
    """
    功能：
        将整数型业务编码标准化为不带前导零和小数后缀的字符串。

    参数：
        value: 原始整数型编码。

    返回：
        标准化后的字符串；空值返回 ``None``。
    """
    code = normalize_code(value)
    if not code:
        return None
    if re.fullmatch(r"\d+", code):
        return str(int(code))
    return code


def parse_enable_size_flags(value: Any) -> tuple[bool, bool]:
    """
    功能：
        解析箱位适放箱型字段，判断该箱位是否适放 20 尺和 40 尺箱。

    参数：
        value: ``YBY_ENABLECSIZECD`` 原始值。

    返回：
        二元组 ``(enable_20, enable_40)``。其中 45 尺按 40 尺处理。
    """

    if value is None or pd.isna(value):
        return True, True
    tokens = re.findall(r"\d+", str(value))
    if not tokens:
        return True, True
    sizes = {normalize_size(token) for token in tokens}
    return "20" in sizes, "40" in sizes


def read_vessel_info(path: Path) -> pd.DataFrame:
    """
    功能：
        读取并清洗船舶靠泊信息。

    参数：
        path: 船舶靠泊信息 CSV 路径。

    返回：
        清洗后的 DataFrame，新增航次号、计划泊位、实际泊位、距离矩阵泊位、
        开港时间和关港时间字段。

    异常：
        FileNotFoundError: 文件不存在时由 pandas 抛出。
        KeyError: 必需列缺失时抛出。
        pandas 读取或时间转换失败时会透传相应异常。
    """

    df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore").copy()
    df["voy_id"] = df["VOY_ID"].map(normalize_code)
    df["voyage_direction"] = df["VOY_IEFG"].map(normalize_code)
    df["planned_berth"] = df["VBT_BTH_PBTHNO"].map(normalize_int_string)
    df["actual_berth"] = df["VBT_BTH_ABTHNO"].map(normalize_int_string)
    df["berth"] = df["planned_berth"].fillna(df["actual_berth"]).map(lambda x: f"B{x}" if x else None)
    df["open_time"] = pd.to_datetime(df["SCD_RCVSTDT"], errors="coerce")
    df["close_time"] = pd.to_datetime(df["SCD_RCVEDDT"], errors="coerce")
    df["planned_berth_time"] = pd.to_datetime(df["VBT_PBTHDT"], errors="coerce")
    df["planned_departure_time"] = pd.to_datetime(df["VBT_PDPTDT"], errors="coerce")
    return df


def read_berths_for_vessels(vessel_info: pd.DataFrame, vessels: Sequence[str]) -> dict[str, str]:
    """
    功能：
        为当前规划航次提取泊位号，用于读取泊位到箱区的距离。

    参数：
        vessel_info: 已清洗的船舶靠泊信息 DataFrame。
        vessels: 当前规划航次集合。

    返回：
        航次到泊位编码的映射，例如 ``{"453334": "B1"}``。

    异常：
        KeyError: 航次缺少靠泊信息或泊位为空时抛出。
    """

    info = vessel_info.drop_duplicates("voy_id").set_index("voy_id")
    result: dict[str, str] = {}
    for vessel in vessels:
        if vessel not in info.index:
            raise KeyError(f"Missing vessel berth info for voyage {vessel}.")
        berth = info.loc[vessel, "berth"]
        if not berth:
            raise KeyError(f"Missing berth for voyage {vessel}.")
        result[vessel] = str(berth)
    return result


def read_area_functions(path: Path) -> tuple[list[str], dict[str, set[str]], dict[str, float]]:
    """
    功能：
        读取箱区功能表，并构造箱区集合、功能集合和作业能力参数。

    参数：
        path: 箱区功能 Excel 文件路径。

    返回：
        三元组 ``(areas, area_functions, load_capacity)``。``areas`` 是可参与规划
        的箱区集合，``area_functions`` 表示每个箱区允许的流向，``load_capacity``
        表示箱区每日作业能力 H。

    异常：
        KeyError: 缺少 ``load_capacity`` 或 ``load_campacity`` 列时抛出。
        pandas 读取 Excel 或数值转换失败时会透传相应异常。
    """

    df = pd.read_excel(path).copy()
    load_col = "load_capacity" if "load_capacity" in df.columns else "load_campacity"
    if load_col not in df.columns:
        raise KeyError("Area function workbook must contain load_capacity or load_campacity.")
    df["area_no"] = df["area_no"].map(normalize_code)
    df = df[df["area_no"].notna()].drop_duplicates("area_no")
    areas = df["area_no"].tolist()

    area_functions: dict[str, set[str]] = {}
    load_capacity: dict[str, float] = {}
    for _, row in df.iterrows():
        area = row["area_no"]
        funcs = {
            normalize_code(part)
            for part in str(row["cntr_type"]).split(",")
            if normalize_code(part)
        }
        area_functions[area] = set(funcs)
        load_capacity[area] = float(pd.to_numeric(row[load_col], errors="coerce") or 0.0)
    return areas, area_functions, load_capacity


def read_distance_matrix(path: Path, areas: Sequence[str], berth_by_vessel: Mapping[str, str]) -> dict[tuple[str, str], float]:
    """
    功能：
        读取距离矩阵工作表，并构造求解器使用的 ``distance[v,a]`` 参数。

    参数：
        path: 距离矩阵 Excel 文件路径。
        areas: 当前规划箱区集合。
        berth_by_vessel: 航次到泊位编码的映射。

    返回：
        键为 ``(voy_id, area_no)``、值为距离的字典。

    异常：
        KeyError: 距离矩阵缺少泊位列或箱区行时抛出。
        pandas 读取 Excel 或数值转换失败时会透传相应异常。
    """

    xls = pd.ExcelFile(path)
    matrix = pd.read_excel(path, sheet_name=xls.sheet_names[3]).copy()
    matrix["area_no"] = matrix["area_no"].map(normalize_code)
    matrix = matrix.dropna(subset=["area_no"]).set_index("area_no")
    result: dict[tuple[str, str], float] = {}
    for vessel, berth in berth_by_vessel.items():
        if berth not in matrix.columns:
            raise KeyError(f"Distance matrix does not contain berth column {berth}.")
        for area in areas:
            if area not in matrix.index:
                raise KeyError(f"Distance matrix does not contain area {area}.")
            result[(vessel, area)] = float(matrix.loc[area, berth])
    return result


def read_snapshot(path: Path) -> pd.DataFrame:
    """
    功能：
        读取并清洗未拆分堆场快照 ``bay_slots_detail.parquet``。

    参数：
        path: 未拆分堆场快照 parquet 文件路径。

    返回：
        清洗后的快照 DataFrame，新增箱区、贝位、排、层、箱号、尺寸、流向、
        进出口航次和是否占用字段。

    异常：
        FileNotFoundError: 文件不存在时由 pandas 抛出。
        KeyError: 必需列缺失时抛出。
        pandas 读取 parquet 或字段转换失败时会透传相应异常。
    """

    df = pd.read_parquet(path).copy()
    df["area_no"] = df["YAA_AREANO"].map(normalize_code)
    df["bay_no"] = df["YBY_BAYNO"].map(normalize_bay)
    df["row_no"] = df["YST_ROWNO"].map(normalize_bay)
    df["tier_no"] = df["YST_TIERNO"].map(normalize_bay)
    df["slot_no"] = df["YST_SLOTNO"].map(normalize_code)
    df["cntr_id"] = df["IYC_CNTRID"].map(normalize_code)
    df["size"] = df["IYC_CSZ_CSIZECD"].map(normalize_size)
    df["raw_flow"] = df["IYC_STS_CSTATUSCD"].map(normalize_code)
    df["e_voy"] = df["IYC_EVOY_ID"].map(normalize_code)
    df["i_voy"] = df["IYC_IVOY_ID"].map(normalize_code)
    df["has_container"] = pd.to_numeric(df["HAS_CONTAINER"], errors="coerce").fillna(0).astype(int)
    enable_flags = df["YBY_ENABLECSIZECD"].map(parse_enable_size_flags)
    df["enable_20"] = enable_flags.map(lambda flags: flags[0])
    df["enable_40"] = enable_flags.map(lambda flags: flags[1])
    df["slot_uid"] = list(zip(df["area_no"], df["bay_no"], df["row_no"], df["tier_no"], df["slot_no"]))
    return df


def extract_current_snapshot_rows(
    snapshot: pd.DataFrame,
    *,
    export_vessels: Sequence[str],
    import_vessels: Sequence[str],
    flow_aliases: Mapping[str, str],
) -> pd.DataFrame:
    """
    功能：
        从堆场快照中筛选当前待规划航次已经在场的箱。

    参数：
        snapshot: 已清洗的堆场快照 DataFrame。
        export_vessels: 当前待规划出口航次集合。
        import_vessels: 当前待规划进口航次集合。
        flow_aliases: 作业流向别名映射。

    返回：
        当前规划航次的在场箱 DataFrame，新增 ``voy_id``、``direction`` 和
        映射后的 ``flow`` 字段。
    """

    export_set = set(export_vessels)
    import_set = set(import_vessels)
    occupied = snapshot[snapshot["has_container"].eq(1)].copy()
    rows = []
    for _, row in occupied.iterrows():
        vessel = None
        direction = None
        if row["e_voy"] in export_set:
            vessel = row["e_voy"]
            direction = "E"
            flow = normalize_export_snapshot_flow(row["raw_flow"], flow_aliases)
        elif row["i_voy"] in import_set:
            vessel = row["i_voy"]
            direction = "I"
            flow = normalize_flow(row["raw_flow"], flow_aliases)
        if not vessel:
            continue
        rows.append(
            {
                **row.to_dict(),
                "voy_id": vessel,
                "direction": direction,
                "flow": flow,
            }
        )
    return pd.DataFrame(rows)


def build_bay_total_slot_counts(snapshot: pd.DataFrame, areas: Sequence[str]) -> dict[tuple[str, str], int]:
    """
    Count the unsplit slot rows in each area/bay.
    """

    if snapshot.empty:
        return {}
    df = snapshot[snapshot["area_no"].isin(set(areas))].copy()
    grouped = df.groupby(["area_no", "bay_no"], dropna=False)["slot_uid"].nunique()
    return {
        (area, bay): int(qty)
        for (area, bay), qty in grouped.items()
        if area and bay
    }


def identify_bad_bays(
    current_snapshot: pd.DataFrame,
    area_functions: Mapping[str, set[str]],
    bay_total_slots: Mapping[tuple[str, str], int],
) -> set[tuple[str, str]]:
    """
    功能：
        识别需要关闭新增容量的异常贝位集合。

    参数：
        current_snapshot: 当前规划航次的在场箱 DataFrame。
        area_functions: 箱区到允许作业流向集合的映射。

    返回：
        异常贝位集合，元素为 ``(area_no, bay_no)``。
    """

    bad_bays: set[tuple[str, str]] = set()
    if current_snapshot.empty:
        return bad_bays

    rows = current_snapshot.copy()
    rows = rows[
        rows["direction"].eq("E")
        & rows["cntr_id"].notna()
        & ~rows["cntr_id"].isin({"", "-1"})
    ].copy()
    if rows.empty:
        return bad_bays

    rows["is_bad_flow"] = rows.apply(
        lambda row: not area_allows_flow(row.get("area_no"), row.get("flow"), area_functions),
        axis=1,
    )
    bad_rows = rows[rows["is_bad_flow"]].copy()
    if bad_rows.empty:
        return bad_bays

    bad_rows = bad_rows.drop_duplicates(["cntr_id", "area_no", "bay_no"])
    grouped = bad_rows.groupby(["area_no", "bay_no"], dropna=False)["cntr_id"].nunique()
    for (area, bay), bad_count in grouped.items():
        if not area or not bay:
            continue
        total_slots = bay_total_slots.get((area, bay), 0)
        if total_slots > 0 and float(bad_count) > (2.0 / 3.0) * float(total_slots):
            bad_bays.add((area, bay))
    return bad_bays


def build_snapshot_count_params(
    current_snapshot: pd.DataFrame,
    area_functions: Mapping[str, set[str]],
    model_areas: set[str],
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
]:
    """
    功能：
        根据当前在场箱和异常贝位构造 L20、L40、Q20、Q40 快照参数。

    参数：
        current_snapshot: 当前规划航次的在场箱 DataFrame。
        bad_bays: 异常贝位集合。
        model_areas: 当前模型允许参与规划的箱区集合。

    返回：
        四元组 ``(l20, l40, q20, q40)``。字典键均为
        ``(voy_id, flow, area_no)``，值为按箱号去重后的箱量。
    """

    l20: dict[tuple[str, str, str], float] = {}
    l40: dict[tuple[str, str, str], float] = {}
    q20: dict[tuple[str, str, str], float] = {}
    q40: dict[tuple[str, str, str], float] = {}
    if current_snapshot.empty:
        return l20, l40, q20, q40

    unique_containers = current_snapshot.copy()
    unique_containers = unique_containers[unique_containers["cntr_id"].notna()].copy()
    unique_containers = unique_containers.sort_values(["cntr_id", "area_no", "bay_no"]).drop_duplicates("cntr_id", keep="first")

    for _, row in unique_containers.iterrows():
        vessel = row.get("voy_id")
        flow = row.get("flow")
        area = row.get("area_no")
        size = row.get("size")
        if str(row.get("cntr_id")) == "-1":
            continue
        if not vessel or not flow or not area or area not in model_areas:
            continue
        flow_matches_area = area_allows_flow(area, flow, area_functions)
        if flow_matches_area and size == "20":
            target = l20
        elif flow_matches_area and size == "40":
            target = l40
        elif size == "20":
            target = q20
        else:
            target = q40
        key = (vessel, flow, area)
        target[key] = target.get(key, 0.0) + 1.0
    return l20, l40, q20, q40


def prepare_slot_frame(raw: pd.DataFrame, areas: Sequence[str], bad_bays: set[tuple[str, str]]) -> pd.DataFrame:
    """
    功能：
        将原始箱位表整理成可统计剩余容量的空箱位表。

    参数：
        raw: 原始箱位 DataFrame。
        areas: 当前模型允许参与规划的箱区集合。
        bad_bays: 异常贝位集合，需要从新增容量中剔除。

    返回：
        只包含模型箱区、空箱位和非异常贝位的 DataFrame，并新增 ``slot_uid``
        唯一箱位标识。

    异常：
        KeyError: 原始箱位表缺少必需列时抛出。
    """

    df = raw.copy()
    df["area_no"] = df["YAA_AREANO"].map(normalize_code)
    df["bay_no"] = df["YBY_BAYNO"].map(normalize_bay)
    df["row_no"] = df["YST_ROWNO"].map(normalize_bay)
    df["tier_no"] = df["YST_TIERNO"].map(normalize_bay)
    df["slot_no"] = df["YST_SLOTNO"].map(normalize_code)
    df["has_container"] = pd.to_numeric(df["HAS_CONTAINER"], errors="coerce").fillna(0).astype(int)
    df = df[df["area_no"].isin(set(areas))].copy()
    df = df[df["has_container"].eq(0)].copy()
    if bad_bays:
        bad_index = pd.MultiIndex.from_tuples(sorted(bad_bays), names=["area_no", "bay_no"])
        row_index = pd.MultiIndex.from_frame(df[["area_no", "bay_no"]])
        df = df[~row_index.isin(bad_index)].copy()
    df["slot_uid"] = list(zip(df["area_no"], df["bay_no"], df["row_no"], df["tier_no"], df["slot_no"]))
    return df


def count_slots_by_area(slots: pd.DataFrame, areas: Sequence[str]) -> dict[str, float]:
    """
    Count remaining slot rows by area for one capacity view.
    """

    if slots.empty:
        return {area: 0.0 for area in areas}
    counts = slots.groupby("area_no")["slot_uid"].nunique()
    return {area: float(counts.get(area, 0.0)) for area in areas}


def build_demand_params(
    data_dir: Path,
    *,
    export_vessels: Sequence[str],
    import_vessels: Sequence[str],
    current_snapshot: pd.DataFrame,
    model_areas: set[str],
    flow_aliases: Mapping[str, str],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, Any]]:
    """
    功能：
        构造求解器最终需求参数 D20 和 D40。

    参数：
        data_dir: 业务数据目录。
        export_vessels: 当前待规划出口航次集合。
        import_vessels: 当前待规划进口航次集合。
        current_snapshot: 当前规划航次的在场箱 DataFrame。
        flow_aliases: 作业流向别名映射。

    返回：
        三元组 ``(d20, d40, diagnostics)``。``d20`` 和 ``d40`` 的键为
        ``(voy_id, flow)``，``diagnostics`` 记录每个航次的资料箱、快照箱、
        预估箱和补量情况。

    异常：
        FileNotFoundError: 资料箱或预估文件不存在时由 pandas 抛出。
        StopIteration: 未在数据目录中找到进口资料箱子目录时抛出。
        KeyError: 资料箱、快照或预估文件缺少必需列时抛出。
        pandas 读取 parquet、Excel 或数据转换失败时会透传相应异常。
    """

    d20: dict[tuple[str, str], float] = {}
    d40: dict[tuple[str, str], float] = {}
    diagnostics: dict[str, Any] = {}

    for vessel in export_vessels:
        doc_path = data_dir / f"container_info_{vessel}.parquet"
        predict_path = data_dir / f"predict_data_{vessel}.xlsx"
        if doc_path.exists():
            doc = normalize_container_frame(pd.read_parquet(doc_path), flow_aliases)
            doc = doc[doc["e_voy"].eq(vessel)].copy()
        else:
            doc = empty_normalized_container_frame()
        snap_all = current_snapshot[current_snapshot["voy_id"].eq(vessel)].copy()
        snap = snap_all[snap_all["area_no"].isin(model_areas)].copy()
        existing_ids = valid_container_ids(snap_all)
        if existing_ids:
            doc = doc[~doc["cntr_id"].isin(existing_ids)].copy()
        merged = merge_snapshot_and_doc(doc, snap)
        add_grouped_demand(merged, vessel, d20, d40)

        pred20, pred40 = read_prediction_counts(predict_path) if predict_path.exists() else (0.0, 0.0)
        known_for_prediction = merge_snapshot_and_doc(
            normalize_export_doc_for_prediction(doc_path, vessel, flow_aliases),
            snap_all,
        )
        detail20, detail40 = count_size_totals(known_for_prediction)
        extra20 = max(0.0, pred20 - detail20)
        extra40 = max(0.0, pred40 - detail40)
        if extra20:
            d20[(vessel, "OF")] = d20.get((vessel, "OF"), 0.0) + extra20
        if extra40:
            d40[(vessel, "OF")] = d40.get((vessel, "OF"), 0.0) + extra40
        diagnostics[vessel] = {
            "type": "export",
            "doc_path": str(doc_path) if doc_path.exists() else None,
            "predict_path": str(predict_path) if predict_path.exists() else None,
            "doc_rows": int(len(doc)),
            "snapshot_rows": int(len(snap_all)),
            "snapshot_rows_in_model_areas": int(len(snap)),
            "dedup_rows": int(len(merged)),
            "prediction20": float(pred20),
            "prediction40": float(pred40),
            "detail20_before_prediction_extra": float(detail20),
            "detail40_before_prediction_extra": float(detail40),
            "extra_prediction20_to_OF": float(extra20),
            "extra_prediction40_to_OF": float(extra40),
        }

    import_dir = discover_import_container_dir(data_dir)
    for vessel in import_vessels:
        doc_path = resolve_import_doc_path(import_dir, vessel)
        doc = normalize_container_frame(pd.read_parquet(doc_path), flow_aliases)
        doc = doc[doc["i_voy"].eq(vessel)].copy()
        snap_all = current_snapshot[current_snapshot["voy_id"].eq(vessel)].copy()
        snap = snap_all[snap_all["area_no"].isin(model_areas)].copy()
        existing_ids = valid_container_ids(snap_all)
        if existing_ids:
            doc = doc[~doc["cntr_id"].isin(existing_ids)].copy()
        merged = merge_snapshot_and_doc(doc, snap)
        add_grouped_demand(merged, vessel, d20, d40)
        diagnostics[vessel] = {
            "type": "import",
            "doc_path": str(doc_path),
            "doc_rows": int(len(doc)),
            "snapshot_rows": int(len(snap_all)),
            "snapshot_rows_in_model_areas": int(len(snap)),
            "dedup_rows": int(len(merged)),
        }
    return d20, d40, diagnostics


def empty_normalized_container_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["cntr_id", "e_voy", "i_voy", "size", "raw_flow", "flow"]
    )


def normalize_export_doc_for_prediction(
    doc_path: Path,
    vessel: str,
    flow_aliases: Mapping[str, str],
) -> pd.DataFrame:
    if not doc_path.exists():
        return empty_normalized_container_frame()
    doc = normalize_container_frame(pd.read_parquet(doc_path), flow_aliases)
    doc = doc[doc["e_voy"].eq(vessel)].copy()
    doc["flow"] = "OF"
    return doc


def valid_container_ids(rows: pd.DataFrame) -> set[str]:
    if rows.empty or "cntr_id" not in rows.columns:
        return set()
    values = rows["cntr_id"].dropna().map(str)
    return {value for value in values if value and value != "-1"}


def count_size_totals(rows: pd.DataFrame) -> tuple[float, float]:
    if rows.empty or "size" not in rows.columns:
        return 0.0, 0.0
    counts = rows.groupby("size").size()
    return float(counts.get("20", 0.0)), float(counts.get("40", 0.0))


def normalize_container_frame(df: pd.DataFrame, flow_aliases: Mapping[str, str]) -> pd.DataFrame:
    """
    功能：
        统一清洗资料箱字段，包括箱号、进出口航次、尺寸和作业流向。

    参数：
        df: 原始资料箱 DataFrame。
        flow_aliases: 作业流向别名映射。

    返回：
        新增 ``cntr_id``、``e_voy``、``i_voy``、``size``、``raw_flow`` 和
        ``flow`` 字段的 DataFrame。

    异常：
        KeyError: 原始资料箱缺少必需列时抛出。
    """

    df = df.copy()
    df["cntr_id"] = df["IYC_CNTRID"].map(normalize_code)
    df["e_voy"] = df["IYC_EVOY_ID"].map(normalize_code)
    df["i_voy"] = df["IYC_IVOY_ID"].map(normalize_code)
    df["size"] = df["IYC_CSZ_CSIZECD"].map(normalize_size)
    df["raw_flow"] = df["IYC_STS_CSTATUSCD"].map(normalize_code)
    df["flow"] = df["raw_flow"].map(lambda value: medium_small_area_flow(normalize_flow(value, flow_aliases)))
    return df


def merge_snapshot_and_doc(doc: pd.DataFrame, snapshot_rows: pd.DataFrame) -> pd.DataFrame:
    """
    功能：
        合并资料箱和快照箱，并按箱号去重。

    参数：
        doc: 已清洗的资料箱 DataFrame。
        snapshot_rows: 当前航次相关的快照箱 DataFrame。

    返回：
        按箱号去重后的箱明细 DataFrame。同一箱号同时存在于资料箱和快照中时，
        优先保留快照行。

    异常：
        KeyError: 输入 DataFrame 缺少 ``cntr_id``、``flow`` 或 ``size`` 列时抛出。
    """

    doc_part = doc[["cntr_id", "flow", "size"]].copy()
    doc_part["source_rank"] = 1
    snap_part = snapshot_rows[["cntr_id", "flow", "size"]].copy() if not snapshot_rows.empty else doc_part.iloc[0:0].copy()
    snap_part["source_rank"] = 0
    merged = pd.concat([snap_part, doc_part], ignore_index=True)
    merged = merged[merged["cntr_id"].notna() & merged["flow"].notna() & merged["size"].notna()].copy()
    merged = merged.sort_values("source_rank").drop_duplicates("cntr_id", keep="first")
    return merged


def add_grouped_demand(
    rows: pd.DataFrame,
    vessel: str,
    d20: dict[tuple[str, str], float],
    d40: dict[tuple[str, str], float],
) -> None:
    """
    功能：
        将去重后的箱明细按流向和箱型汇总，并累加写入 D20/D40 需求字典。

    参数：
        rows: 已去重的箱明细 DataFrame。
        vessel: 当前航次号。
        d20: 20 尺箱需求字典，函数会原地更新。
        d40: 40 尺箱需求字典，函数会原地更新。

    返回：
        无。

    异常：
        KeyError: ``rows`` 缺少 ``flow`` 或 ``size`` 列时抛出。
    """

    if rows.empty:
        return
    grouped = rows.groupby(["flow", "size"]).size()
    for (flow, size), qty in grouped.items():
        target = d20 if size == "20" else d40
        target[(vessel, flow)] = target.get((vessel, flow), 0.0) + float(qty)


def read_prediction_counts(path: Path) -> tuple[float, float]:
    """
    功能：
        读取出口预估箱量，并将 45 尺箱统一并入 40 尺口径。

    参数：
        path: 出口预估箱量 Excel 文件路径。

    返回：
        二元组 ``(total20, total40)``，分别表示 20 尺和 40 尺预估箱量。

    异常：
        FileNotFoundError: 文件不存在时由 pandas 抛出。
        KeyError: 缺少箱型或数量列时抛出。
        pandas 读取 Excel 或数值转换失败时会透传相应异常。
    """

    xls = pd.ExcelFile(path)
    df = pd.read_excel(path, sheet_name=xls.sheet_names[0]).copy()
    size_col = "IYC_CSZ_CSIZECD"
    count_col = "count"
    total20 = 0.0
    total40 = 0.0
    for _, row in df.iterrows():
        size = normalize_size(row[size_col])
        count = float(pd.to_numeric(row[count_col], errors="coerce") or 0.0)
        if size == "20":
            total20 += count
        elif size == "40":
            total40 += count
    return total20, total40


def read_prediction_work_lanes(path: Path) -> float:
    """
    Read the export vessel work-lane count from the prediction workbook.

    Some current files store the lane count as the second column header of the
    "作业路" sheet (for example columns ["作业路数", 3]) and have no data rows.
    This reader therefore checks both column headers and cell values.
    """

    xls = pd.ExcelFile(path)
    robust_sheet_name = next((name for name in xls.sheet_names if "\u4f5c\u4e1a\u8def" in str(name)), None)
    if robust_sheet_name is None:
        for name in xls.sheet_names:
            preview = pd.read_excel(path, sheet_name=name, nrows=2)
            columns = {str(col).strip() for col in preview.columns}
            if "IYC_CSZ_CSIZECD" in columns or "IYC_POT_UNLDPORT" in columns:
                continue
            if len(columns) <= 3:
                robust_sheet_name = name
                break
    if robust_sheet_name is not None:
        lane_df = pd.read_excel(path, sheet_name=robust_sheet_name).copy()
        lane_candidates: list[float] = []

        def collect_lane_numeric(value: Any) -> None:
            if value is None or pd.isna(value):
                return
            text = str(value).strip()
            if not text or text == "\u4f5c\u4e1a\u8def\u6570":
                return
            numeric = pd.to_numeric(value, errors="coerce")
            if pd.notna(numeric):
                lane_candidates.append(float(numeric))

        for column in lane_df.columns:
            collect_lane_numeric(column)
        for value in lane_df.to_numpy().ravel():
            collect_lane_numeric(value)
        positive_lane_candidates = [value for value in lane_candidates if value > 0]
        if positive_lane_candidates:
            return positive_lane_candidates[0]
    sheet_name = "作业路" if "作业路" in xls.sheet_names else None
    if sheet_name is None:
        sheet_name = next((name for name in xls.sheet_names if "作业" in str(name) and "路" in str(name)), None)
    if sheet_name is None:
        raise KeyError(f"Prediction workbook {path} is missing the 作业路 sheet.")

    df = pd.read_excel(path, sheet_name=sheet_name).copy()
    candidates: list[float] = []

    def collect_numeric(value: Any) -> None:
        if value is None or pd.isna(value):
            return
        text = str(value).strip()
        if not text or text == "作业路数":
            return
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            candidates.append(float(numeric))

    for column in df.columns:
        collect_numeric(column)
    for value in df.to_numpy().ravel():
        collect_numeric(value)

    positive_candidates = [value for value in candidates if value > 0]
    if not positive_candidates:
        return 0.0
    return positive_candidates[0]


def compute_departure_operation_deductions(
    *,
    vessel_info: pd.DataFrame,
    snapshot: pd.DataFrame,
    planning_time: pd.Timestamp,
    areas: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    planning_date = planning_time.date()
    info = vessel_info.copy()
    info = info[
        info["voyage_direction"].eq("E")
        & info["planned_berth_time"].notna()
        & info["planned_departure_time"].notna()
    ].copy()
    if info.empty:
        return {area: 0.0 for area in areas}, {"active_export_voyages": [], "counts_by_voyage_area": []}

    berth_dates = info["planned_berth_time"].dt.date
    departure_dates = info["planned_departure_time"].dt.date
    active = info[(berth_dates <= planning_date) & (planning_date <= departure_dates)].copy()
    active = active.drop_duplicates("voy_id")
    if active.empty:
        return {area: 0.0 for area in areas}, {"active_export_voyages": [], "counts_by_voyage_area": []}

    days_remaining = {
        row["voy_id"]: max(1, (row["planned_departure_time"].date() - planning_date).days + 1)
        for _, row in active.iterrows()
        if row.get("voy_id")
    }
    active_voyages = set(days_remaining)
    occupied = snapshot[
        snapshot["has_container"].eq(1)
        & snapshot["e_voy"].isin(active_voyages)
        & snapshot["area_no"].isin(set(areas))
        & snapshot["cntr_id"].notna()
    ].copy()
    occupied = occupied[~occupied["cntr_id"].isin({"", "-1"})].copy()

    deductions = {area: 0.0 for area in areas}
    count_rows: list[dict[str, Any]] = []
    if not occupied.empty:
        unique_containers = occupied.sort_values(["cntr_id", "area_no"]).drop_duplicates("cntr_id", keep="first")
        grouped = unique_containers.groupby(["e_voy", "area_no"], dropna=False)["cntr_id"].nunique()
        for (voyage, area), count in grouped.items():
            if not voyage or not area or area not in deductions:
                continue
            days = days_remaining.get(voyage, 1)
            deduction = float(math.ceil(float(count) / float(days))) if count else 0.0
            deductions[area] += deduction
            count_rows.append(
                {
                    "voy_id": voyage,
                    "area_no": area,
                    "yard_container_count": int(count),
                    "days_until_planned_departure_inclusive": int(days),
                    "daily_departure_operation_deduction": deduction,
                }
            )

    diagnostics = {
        "active_export_voyages": sorted(active_voyages),
        "counts_by_voyage_area": sorted(count_rows, key=lambda row: (row["voy_id"], row["area_no"])),
    }
    return deductions, diagnostics


def build_close_export_berth_pairs(
    *,
    vessel_info: pd.DataFrame,
    export_vessels: Sequence[str],
    threshold_hours: float,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    export_set = set(export_vessels)
    info = vessel_info[vessel_info["voy_id"].isin(export_set)].drop_duplicates("voy_id").copy()
    berth_time_by_voyage = {
        row["voy_id"]: row["planned_berth_time"]
        for _, row in info.iterrows()
        if row.get("voy_id") and pd.notna(row.get("planned_berth_time"))
    }

    pairs: list[tuple[str, str]] = []
    details: list[dict[str, Any]] = []
    for left, right in combinations(export_vessels, 2):
        left_time = berth_time_by_voyage.get(left)
        right_time = berth_time_by_voyage.get(right)
        if left_time is None or right_time is None:
            continue
        if left_time.date() != right_time.date():
            continue
        delta_hours = abs((left_time - right_time).total_seconds()) / 3600.0
        if delta_hours <= threshold_hours:
            pairs.append((left, right))
            details.append(
                {
                    "left_voy_id": left,
                    "right_voy_id": right,
                    "left_planned_berth_time": left_time.isoformat(),
                    "right_planned_berth_time": right_time.isoformat(),
                    "delta_hours": float(delta_hours),
                }
            )
    return pairs, details


def compute_tops_capacity_deductions(
    tops: pd.DataFrame,
    planning_time: pd.Timestamp,
    vessels: Sequence[str],
    bay20_equiv: pd.DataFrame,
    bay20_direct: pd.DataFrame,
    bay40: pd.DataFrame,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    int,
]:
    """
    功能：
        计算 TOPS 对各航次、各箱区容量的扣减量。

    参数：
        tops: TOPS 计划 DataFrame。
        planning_time: 当前规划节点时间。
        vessels: 当前规划航次集合。
        bay20_equiv: 20 尺等价物理容量口径的空箱位表。
        bay20_direct: 真实适放 20 尺箱口径的空箱位表。
        bay40: 真实适放 40 尺箱口径的空箱位表。

    返回：
        四元组 ``(tops20, tops20_direct, tops40, active_tops_count)``，前三个字典
        的键为 ``(voy_id, area_no)``，值为应扣除的箱位数量。

    异常：
        KeyError: TOPS 表缺少 ``SPL_STDATE``、``SPL_EDDATE``、``SPL_CONDITIONCODE``、
            ``SPR_STBAY`` 或 ``SPR_EDBAY`` 时抛出。
        pandas 时间转换或筛选失败时会透传相应异常。
    """

    active = tops.copy()
    active["condition_vessel"] = active["SPL_CONDITIONCODE"].map(normalize_code)
    active["start_time"] = parse_tops_time(active["SPL_STDATE"])
    active["end_time"] = parse_tops_time(active["SPL_EDDATE"])
    if "SPL_ISVALID" in active.columns:
        active = active[active["SPL_ISVALID"].astype(str).str.upper().eq("Y")].copy()
    if "SPR_ISVALID" in active.columns:
        active = active[active["SPR_ISVALID"].astype(str).str.upper().eq("Y")].copy()
    active = active[(active["start_time"] <= planning_time) & (planning_time <= active["end_time"])].copy()

    tops20: dict[tuple[str, str], float] = {}
    tops20_direct: dict[tuple[str, str], float] = {}
    tops40: dict[tuple[str, str], float] = {}
    for vessel in vessels:
        relevant = active[active["condition_vessel"] != vessel].copy()
        tops20.update(count_tops_blocked_slots(relevant, bay20_equiv, vessel))
        tops20_direct.update(count_tops_blocked_slots(relevant, bay20_direct, vessel))
        tops40.update(count_tops_blocked_slots(relevant, bay40, vessel))
    return tops20, tops20_direct, tops40, int(len(active))


def parse_tops_time(series: pd.Series) -> pd.Series:
    """
    功能：
        将 TOPS 时间字段解析为 pandas 时间序列，兼容 datetime 字符串和 Unix 秒时间戳。

    参数：
        series: TOPS 起止时间字段。

    返回：
        转换后的 ``datetime64`` Series，无法解析的值为 ``NaT``。
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def count_tops_blocked_slots(tops_rows: pd.DataFrame, slots: pd.DataFrame, vessel: str) -> dict[tuple[str, str], float]:
    """
    功能：
        在指定箱位表中统计 TOPS 覆盖的空箱位数量。

    参数：
        tops_rows: 当前航次需要扣除的生效 TOPS 行。
        slots: 某一容量口径下的空箱位表。
        vessel: 当前计算扣减量的航次号。

    返回：
        键为 ``(voy_id, area_no)``、值为该航次在该箱区应扣除容量的字典。
        使用 set 对箱位去重，避免多条 TOPS 范围重叠时重复扣减。

    异常：
        KeyError: 输入表缺少 ``area_no``、``bay_no`` 或 ``slot_uid`` 列时抛出。
    """

    blocked_by_area: dict[str, set[Any]] = {}
    if tops_rows.empty or slots.empty:
        return {}
    slots_by_area = {area: sub for area, sub in slots.groupby("area_no")}
    for _, tops in tops_rows.iterrows():
        start_area, start_bay = parse_tops_area_bay(tops.get("SPR_STBAY"))
        end_area, end_bay = parse_tops_area_bay(tops.get("SPR_EDBAY"))
        area = start_area or end_area
        if start_area and end_area and start_area != end_area:
            area = end_area
        if not area or area not in slots_by_area:
            continue
        sub = slots_by_area[area]
        bay_mask = bay_range_mask(sub["bay_no"], start_bay, end_bay)
        matched = sub[bay_mask]
        if matched.empty:
            continue
        blocked_by_area.setdefault(area, set()).update(matched["slot_uid"].tolist())
    return {(vessel, area): float(len(uids)) for area, uids in blocked_by_area.items()}


def parse_tops_area_bay(value: Any) -> tuple[Optional[str], Optional[str]]:
    """
    功能：
        从 TOPS 的 ``SPR_STBAY`` 或 ``SPR_EDBAY`` 字段解析箱区和贝位。

    参数：
        value: TOPS 起止贝位原始编码。

    返回：
        二元组 ``(area_no, bay_no)``。无法解析时返回 ``(None, None)``。
    """

    code = normalize_code(value)
    if not code:
        return None, None
    code = code.replace(".0", "")
    if len(code) < 4:
        code = code.zfill(4)
    return code[:2], normalize_bay(code[-2:])


def bay_range_mask(values: pd.Series, start_bay: Optional[str], end_bay: Optional[str]) -> pd.Series:
    """
    功能：
        根据 TOPS 起止贝位构造箱位表的贝位筛选掩码。

    参数：
        values: 箱位表中的贝位 Series。
        start_bay: TOPS 起始贝位。
        end_bay: TOPS 结束贝位。

    返回：
        与 ``values`` 索引一致的布尔 Series。若起止贝位均为空，则全部为 True。
    """

    return slot_range_mask(values, start_bay, end_bay, bay_code_value)


def slot_range_mask(
    values: pd.Series,
    start_value: Any,
    end_value: Any,
    value_parser: Any,
) -> pd.Series:
    """
    Build an inclusive mask for numeric or alphanumeric yard coordinates.
    """

    start = normalize_code(start_value)
    end = normalize_code(end_value)
    if not start and not end:
        return pd.Series(True, index=values.index)
    if start and not end:
        return values.map(normalize_code).eq(start)
    if end and not start:
        return values.map(normalize_code).eq(end)

    start_key = value_parser(start)
    end_key = value_parser(end)
    if start_key is None or end_key is None:
        allowed = {value for value in [start, end] if value}
        return values.map(normalize_code).isin(allowed)

    lo = min(start_key, end_key)
    hi = max(start_key, end_key)
    parsed_values = values.map(value_parser)
    return parsed_values.map(lambda value: value is not None and lo <= value <= hi)


def bay_code_value(value: Any) -> Optional[int]:
    """
    Convert bay codes such as 09, 99, A1, C7 or D0 to a sortable base-36 value.
    """

    code = normalize_code(value)
    if not code:
        return None
    total = 0
    for char in code:
        if "0" <= char <= "9":
            digit = ord(char) - ord("0")
        elif "A" <= char <= "Z":
            digit = ord(char) - ord("A") + 10
        else:
            return None
        total = total * 36 + digit
    return total


def build_availability_flags(
    *,
    vessels: Sequence[str],
    flows: Sequence[str],
    areas: Sequence[str],
    area_functions: Mapping[str, set[str]],
    cbar20: Mapping[tuple[str, str], float],
    cbar20_direct: Mapping[tuple[str, str], float],
    cbar40: Mapping[tuple[str, str], float],
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
    """
    功能：
        构造求解器使用的 E20/E40 箱区可用性参数。

    参数：
        vessels: 当前规划航次集合。
        flows: 当前模型流向集合。
        areas: 当前模型箱区集合。
        area_functions: 箱区到允许流向集合的映射。
        cbar20: TOPS 扣减后的 20 尺等价容量。
        cbar20_direct: TOPS 扣减后的真实适放 20 尺容量。
        cbar40: TOPS 扣减后的真实适放 40 尺容量。

    返回：
        二元组 ``(e20, e40)``。字典键为 ``(voy_id, flow, area_no)``，值为 0/1。
    """

    e20: dict[tuple[str, str, str], int] = {}
    e40: dict[tuple[str, str, str], int] = {}
    for vessel in vessels:
        for flow in flows:
            for area in areas:
                func_ok = area_allows_flow(area, flow, area_functions)
                e20[(vessel, flow, area)] = int(
                    func_ok
                    and cbar20_direct.get((vessel, area), 0.0) > 0
                    and cbar20.get((vessel, area), 0.0) > 0
                )
                e40[(vessel, flow, area)] = int(
                    func_ok
                    and cbar40.get((vessel, area), 0.0) > 0
                    and cbar20.get((vessel, area), 0.0) >= 2
                )
    return e20, e40


def build_allocation_output_rows(
    solution: DailyRollingYardPlanningSolution,
    data: DailyRollingYardPlanningData,
    *,
    include_zero: bool = False,
    planning_time: pd.Timestamp | str | None = None,
) -> list[dict[str, Any]]:
    """
    Convert total X into output rows and expose the snapshot/new split.
    """

    metadata: dict[str, Any] = {}
    if planning_time is not None:
        metadata["planning_time"] = pd.Timestamp(planning_time).isoformat()
    if solution.status_name is not None:
        metadata["status_name"] = solution.status_name
    if solution.objective_value is not None:
        metadata["objective_value"] = solution.objective_value
    rows: list[dict[str, Any]] = []
    for size, values in (("20", solution.x20), ("40", solution.x40)):
        for key, qty in values.items():
            snapshot_qty = snapshot_quantity_for_output(data, size, key)
            new_qty = max(0.0, float(qty) - snapshot_qty)
            if not include_zero and not qty and not snapshot_qty and not new_qty:
                continue
            vessel, flow, area = key
            rows.append(
                {
                    "voy_id": vessel,
                    "flow": flow,
                    "area_no": area,
                    "size": size,
                    "planned_qty": int(qty),
                    "snapshot_qty": float(snapshot_qty),
                    "new_qty": float(new_qty),
                    **metadata,
                }
            )
    return rows


def snapshot_quantity_for_output(
    data: DailyRollingYardPlanningData,
    size: str,
    key: tuple[str, str, str],
) -> float:
    """
    Read S for output. If S is not explicitly supplied, use L+Q.
    """

    if size == "20":
        if key in data.S20:
            return float(data.S20[key])
        return float(data.L20.get(key, 0.0) + data.Q20.get(key, 0.0))
    if key in data.S40:
        return float(data.S40[key])
    return float(data.L40.get(key, 0.0) + data.Q40.get(key, 0.0))


def count_flow_function_mismatch_rows(
    allocation: pd.DataFrame,
    area_functions: Mapping[str, set[str]],
    *,
    qty_column: str,
) -> int:
    """
    Count output rows whose positive quantity is not allowed by the area's function.
    """

    if allocation.empty or qty_column not in allocation.columns:
        return 0
    rows = allocation[pd.to_numeric(allocation[qty_column], errors="coerce").fillna(0.0) > 1e-6].copy()
    if rows.empty:
        return 0
    return int(
        rows.apply(
            lambda row: not area_allows_flow(row["area_no"], row["flow"], area_functions),
            axis=1,
        ).sum()
    )


def write_run_outputs(
    output_dir: Path,
    artifacts: PlanningInputArtifacts,
    solution: DailyRollingYardPlanningSolution,
    state_rows: pd.DataFrame,
) -> None:
    """
    功能：
        将分配结果、追加状态记录和诊断信息写入本次运行输出目录。

    参数：
        output_dir: 输出目录路径。
        artifacts: 参数构造阶段产物。
        solution: 求解器返回的规划结果。
        state_rows: 本次追加到滚动状态表的记录。

    返回：
        无。

    异常：
        OSError: 创建目录或写入文件失败时抛出。
        pandas 写出 CSV 失败时会透传相应异常。
        TypeError: 诊断信息无法 JSON 序列化时抛出。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    allocation = pd.DataFrame(
        build_allocation_output_rows(
            solution,
            artifacts.data,
            planning_time=artifacts.planning_time,
        )
    )
    allocation.to_csv(output_dir / "allocation.csv", index=False, encoding="utf-8-sig")
    if allocation.empty:
        allocation_new = allocation.copy()
    else:
        allocation_new = allocation[
            pd.to_numeric(allocation["new_qty"], errors="coerce").fillna(0.0) > 1e-6
        ].copy()
    allocation_new.to_csv(output_dir / "allocation_new.csv", index=False, encoding="utf-8-sig")
    if not state_rows.empty:
        state_rows.to_csv(output_dir / "state_rows_appended.csv", index=False, encoding="utf-8-sig")
    diagnostics = {
        **artifacts.diagnostics,
        "status": solution.status_name,
        "objective_value": solution.objective_value,
        "best_bound": solution.best_bound,
        "mip_gap": solution.mip_gap,
        "runtime": solution.runtime,
        "objective_components": solution.objective_components,
        "unmet20": {str(k): v for k, v in solution.s20.items()},
        "unmet40": {str(k): v for k, v in solution.s40.items()},
        "operation_overage": solution.o,
        "area_share_overage": solution.h,
        "of_area_overage": solution.of_area_over,
        "berth_conflict_shared": {str(k): v for k, v in solution.berth_conflict_shared.items()},
        "required_area_unmet": {str(k): v for k, v in solution.required_area_unmet.items()},
        "of_area_used_count": {
            vessel: int(sum(used for (used_vessel, _area), used in solution.of_area_used.items() if used_vessel == vessel))
            for vessel in artifacts.export_vessels
        },
        "allocation_total_wrong_function_rows": count_flow_function_mismatch_rows(
            allocation,
            artifacts.area_functions,
            qty_column="planned_qty",
        ),
        "allocation_new_wrong_function_rows": count_flow_function_mismatch_rows(
            allocation,
            artifacts.area_functions,
            qty_column="new_qty",
        ),
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """
    功能：
        解析命令行参数。

    参数：
        无。

    返回：
        argparse 命名空间，包含数据目录、规划时间、航次列表、状态目录、
        输出目录、求解时间限制和日志开关等参数。

    异常：
        SystemExit: 命令行参数不合法或请求帮助信息时由 argparse 抛出。
    """
    parser = argparse.ArgumentParser(description="Build parameters and solve the daily rolling yard planning model.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--planning-time", default=DEFAULT_PLANNING_TIME)
    parser.add_argument("--export-vessels", nargs="+", default=DEFAULT_EXPORT_VESSELS)
    parser.add_argument("--import-vessels", nargs="+", default=DEFAULT_IMPORT_VESSELS)
    parser.add_argument("--state-dir", type=Path, default=Path("outputs_large/state"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_large/latest_run"))
    parser.add_argument("--visualization-dir", type=Path, default=Path("outputs_large/yard_visualization"))
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-write-state", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--disable-default-flow-aliases", action="store_true")
    parser.add_argument(
        "--user-design-large-plan-area",
        nargs="+",
        default=None,
        help="Restrict new large-plan allocation to these yard areas for every planned vessel.",
    )
    parser.add_argument(
        "--adjust-plan-info-json",
        type=Path,
        default=None,
        help="JSON file containing adjust_plan_info; large_plan[voy_id].add/remove controls large-plan areas.",
    )
    return parser.parse_args()


def main() -> None:
    """
    功能：
        命令行执行入口，串联参数构造、模型求解、状态写入和结果输出。

    参数：
        无。

    返回：
        无。

    异常：
        运行过程中数据读取、参数构造、Gurobi 求解或文件写入失败时会透传相应异常。
    """
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    planning_time = pd.Timestamp(args.planning_time)
    state = RollingPlanningState(resolve_output_path(args.state_dir, base_dir))
    flow_aliases = {} if args.disable_default_flow_aliases else DEFAULT_FLOW_ALIASES
    adjust_plan_info = read_adjust_plan_info(args.adjust_plan_info_json)

    artifacts = build_current_case_data(
        base_dir=base_dir,
        data_dir=args.data_dir,
        planning_time=planning_time,
        export_vessels=args.export_vessels,
        import_vessels=args.import_vessels,
        state=state,
        flow_aliases=flow_aliases,
        user_design=bool(args.user_design_large_plan_area),
        user_design_large_plan_area=args.user_design_large_plan_area,
        adjust_plan_info=adjust_plan_info,
    )
    print_case_summary(artifacts)
    solution = solve_daily_rolling_yard_plan(
        artifacts.data,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        verbose=not args.quiet,
    )
    print_solution_summary(solution)
    state_rows = pd.DataFrame()
    if not args.no_write_state and solution.objective_value is not None:
        state_rows = state.append_solution(planning_time, solution)
        print(f"State rows appended: {len(state_rows)} -> {state.plan_history_path}")
    output_dir = resolve_output_path(args.output_dir, base_dir)
    write_run_outputs(output_dir, artifacts, solution, state_rows)
    print(f"Run outputs written to: {output_dir}")
    if not args.skip_visualization:
        visualization_dir = resolve_output_path(args.visualization_dir, base_dir)
        run_visualization(
            base_dir=base_dir,
            allocation_path=output_dir / "allocation.csv",
            output_dir=visualization_dir,
            data_dir=Path(artifacts.diagnostics["data_dir"]),
        )
        print(f"Visualization written to: {visualization_dir}")


def run_visualization(*, base_dir: Path, allocation_path: Path, output_dir: Path, data_dir: Path) -> None:
    """
    Generate the yard visualization through the standalone visualization script.
    """

    cmd = [
        sys.executable,
        str(base_dir / "planning_large_visualize.py"),
        "--allocation",
        str(allocation_path),
        "--output-dir",
        str(output_dir),
        "--data-dir",
        str(data_dir),
    ]
    subprocess.run(cmd, cwd=base_dir, check=True)


def print_case_summary(artifacts: PlanningInputArtifacts) -> None:
    """
    功能：
        在控制台打印本次规划案例的关键输入摘要。

    参数：
        artifacts: 参数构造阶段产物。

    返回：
        无。

    异常：
        KeyError: 诊断信息中缺少摘要字段时抛出。
    """
    data = artifacts.data
    print("planning_time:", artifacts.planning_time)
    print("export_vessels:", artifacts.export_vessels)
    print("import_vessels:", artifacts.import_vessels)
    print("berth_by_vessel:", artifacts.berth_by_vessel)
    print("area_count:", len(data.A))
    print("flows:", list(data.F))
    print("demand20_total:", sum(data.D20.values()))
    print("demand40_total:", sum(data.D40.values()))
    print("capacity20_equiv_total:", artifacts.diagnostics["capacity20_total"])
    print("capacity20_direct_total:", artifacts.diagnostics["capacity20_direct_total"])
    print("capacity40_total:", artifacts.diagnostics["capacity40_total"])
    print("bad_bay_count:", artifacts.diagnostics["bad_bay_count"])
    print("active_tops_rows:", artifacts.diagnostics["active_tops_rows"])
    print("user_design_active:", artifacts.diagnostics["user_design_active"])
    print("user_design_large_plan_area:", artifacts.diagnostics["user_design_large_plan_area"])
    print("departure_operation_deduction_total:", artifacts.diagnostics["departure_operation_deduction_total"])
    print("close_berth_conflict_pairs:", artifacts.diagnostics["close_berth_conflict_pairs"])
    print("old_vessels:", artifacts.diagnostics["old_vessels"])
    print("of_work_lanes:", artifacts.diagnostics["of_work_lanes"])
    print("of_area_limits:", artifacts.diagnostics["of_area_limits"])


def print_solution_summary(solution: DailyRollingYardPlanningSolution) -> None:
    """
    功能：
        在控制台打印求解结果的关键摘要。

    参数：
        solution: 求解器返回的规划结果。

    返回：
        无。
    """
    print("status:", solution.status_name)
    print("objective_value:", solution.objective_value)
    print("mip_gap:", solution.mip_gap)
    print("runtime:", solution.runtime)
    print("objective_components:", solution.objective_components)
    print("unmet20_total:", sum(solution.s20.values()))
    print("unmet40_total:", sum(solution.s40.values()))
    print("operation_overage_total:", sum(solution.o.values()))
    print("of_area_overage_total:", sum(solution.of_area_over.values()))
    print("area_share_overage_total:", sum(solution.h.values()))
    print("required_area_unmet:", solution.required_area_unmet)


if __name__ == "__main__":
    main()
