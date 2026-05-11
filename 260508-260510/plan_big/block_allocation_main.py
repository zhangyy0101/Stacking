"""
实际航次箱区规划算例入口。

本文件负责把真实业务数据转换为 `YardAllocationData`：
1. 读取航次、箱量、堆场容量、泊位距离、TOPS 计划和历史计划；
2. 根据当前滚动时刻 theta 判断哪些航次需要本轮新增规划；
3. 构造 Gurobi 模型需要的 R/C/distance/B/H/P/G/K 等参数；
4. 调用 `solve_yard_area_allocation` 求解，并可将结果追加写入历史表。

Gurobi 数学模型本身仍放在 `gurobi_block_allocation_framework.py` 中。
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import dataclass
from datetime import timedelta
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from block_allocation_model import (
    Area,
    Vessel,
    YardAllocationData,
    solve_yard_area_allocation,
)


DEFAULT_THETA = "2026-05-08 09:30"
DEFAULT_VESSELS = ["453334", "453400"]
DEFAULT_PLAN_HISTORY = "plan_history.csv"

# 每个航次的滚动规划事件：
# - 开港前 24 小时累计规划 70%，属于 first plan；
# - 开港后 24 小时累计规划 90%，属于 followup plan；
# - 开港后 48 小时累计规划 100%，属于 followup plan。
PLANNING_EVENTS = [
    (timedelta(hours=-24), 0.7, "first_70"),
    (timedelta(hours=24), 0.9, "second_90"),
    (timedelta(hours=48), 1.0, "third_100"),
]


@dataclass(frozen=True)
class BuildArtifacts:
    """
    参数构造阶段的中间产物集合。

    `data` 是最终传给 Gurobi 模型的输入；其他字段用于打印摘要、
    追溯参数来源，以及求解后写回历史计划。
    """

    data: YardAllocationData                # 包含V_plus...等求解器需要的参数
    theta: pd.Timestamp                     # 当前滚动规划时刻
    stage_by_vessel: Dict[Vessel, str]      # 记录每个本轮规划航次对应的规划阶段
    ratio_by_vessel: Dict[Vessel, float]    # 记录每个本轮规划航次的 累计规划比例
    demand20_total: Dict[Vessel, int]       # 每个航次的20ft箱总需求量
    demand40_total: Dict[Vessel, int]       # 每个航次的40ft/45ft箱总需求量
    berth_by_vessel: Dict[Vessel, str]      # 每个活跃航次对应的泊位号
    capacity20_physical: Dict[Area, int]    # 每个箱区当前物理上的20ft等价空位数量
    capacity40_physical: Dict[Area, int]    # 每个箱区当前物理上的40ft空位数量


def normalize_code(value: object, *, upper: bool = True) -> Optional[str]:
    """
    统一清洗业务编码。

    用于航次号、箱区号、箱尺寸等字段；主要处理空值、首尾空格、
    pandas 读取数字编码时出现的 `.0` 后缀，以及是否转大写。
    """

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text.upper() if upper else text


def normalize_berth(value: object) -> Optional[str]:
    """
    将泊位字段统一为距离矩阵中的列名格式。

    例如 `6`、`6.0` 会被转换为 `B6`；如果原值已经是 `B6` 则保持不变。
    """

    code = normalize_code(value)
    if not code:
        return None
    if code.startswith("B"):
        return code
    return f"B{code}"


def discover_data_dir(base_dir: Path, data_dir: Optional[Path]) -> Path:
    """
    定位测试数据目录。

    如果命令行显式传入 `--data-dir`，则直接使用；否则在项目目录下
    自动寻找以 `20260508` 结尾的数据文件夹。
    """

    if data_dir is not None:
        return data_dir.resolve()

    search_roots = [base_dir]
    if base_dir.parent not in search_roots:
        search_roots.append(base_dir.parent)

    candidates = [
        p
        for root in search_roots
        for p in root.iterdir()
        if p.is_dir() and "20260508" in p.name
    ]
    if not candidates:
        searched = ", ".join(str(root) for root in search_roots)
        raise FileNotFoundError(
            f"No data directory containing 20260508 was found under: {searched}"
        )
    if len(candidates) > 1:
        raise ValueError(f"Multiple data directories found: {candidates}")
    return candidates[0]


def read_closed_areas(path: Path) -> set[Area]:
    """
    读取关闭箱区集合。

    支持 `['20', '25']` 这类 Python 字面量，也支持按逗号、空格或换行
    分隔的普通文本。
    """

    if not path.exists():
        return set()

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()

    try:
        raw_values = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        raw_values = re.split(r"[\s,，;；]+", text)

    if isinstance(raw_values, (str, int, float)):
        raw_values = [raw_values]

    return {
        area
        for area in (normalize_code(value) for value in raw_values)
        if area
    }


def read_of_areas(path: Path, closed_areas: set[Area]) -> List[Area]:
    """
    读取 OF 适放箱区，并剔除关闭箱区。

    返回值将作为模型候选箱区集合 `A`，因此这里会保留 Excel 中的原始顺序，
    方便后续结果和人工检查保持一致。
    """

    df = pd.read_excel(path)
    area_col = "area_no" if "area_no" in df.columns else df.columns[0]
    areas = (
        df[area_col]
        .map(normalize_code)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return [area for area in areas if area not in closed_areas]


def read_vessel_berth_info(path: Path) -> pd.DataFrame:
    """
    读取航次开港时间和泊位信息。

    开港时间使用 `SCD_RCVSTDT`；泊位优先使用实际泊位
    `VBT_BTH_ABTHNO`，为空时使用预计泊位 `VBT_BTH_PBTHNO`。
    离港时间优先使用实际离港 `VBT_ADPTDT`，为空时使用计划离港 `VBT_PDPTDT`。
    """

    df = pd.read_csv(path)
    df = df.copy()
    df["voy_id"] = df["VOY_ID"].map(lambda x: normalize_code(x, upper=False))
    df["open_time"] = pd.to_datetime(df["SCD_RCVSTDT"], errors="coerce")
    actual_departure = pd.to_datetime(df["VBT_ADPTDT"], errors="coerce")
    planned_departure = pd.to_datetime(df["VBT_PDPTDT"], errors="coerce")
    df["departure_time"] = actual_departure.fillna(planned_departure)

    berth_by_row = []
    for _, row in df.iterrows():
        actual = normalize_berth(row.get("VBT_BTH_ABTHNO"))
        planned = normalize_berth(row.get("VBT_BTH_PBTHNO"))
        berth_by_row.append(actual or planned)

    df["berth"] = berth_by_row
    return df[df["voy_id"].notna()].copy()


def read_vessel_demands(
    data_dir: Path,
    vessel_ids: Sequence[Vessel],
    *,
    of_only: bool = True,
) -> Tuple[Dict[Vessel, int], Dict[Vessel, int]]:
    """
    读取指定航次的总箱量需求 D20/D40。

    当前模型只区分 20 尺和 40 尺两类；45 尺在第一版中并入 40 尺容量处理。
    默认只统计 `IYC_STS_CSTATUSCD == "OF"` 的出口重箱/出口箱计划数据。
    """

    d20: Dict[Vessel, int] = {}
    d40: Dict[Vessel, int] = {}

    for vessel in vessel_ids:
        path = data_dir / f"container_info_{vessel}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing container file: {path}")

        df = pd.read_parquet(path)
        # 只保留出口航次号不为空的记录
        df = df[df["IYC_EVOY_ID"].notna()].copy()
        if of_only and "IYC_STS_CSTATUSCD" in df.columns:
            # 只保留IYC_STS_CSTATUSCD为OF的记录
            df = df[
                df["IYC_STS_CSTATUSCD"].astype(str).str.upper() == "OF"
            ].copy()

        # 新增标准化后的航次号字段voy_id，来自IYC_EVOY_ID
        df["voy_id"] = df["IYC_EVOY_ID"].map(lambda x: normalize_code(x, upper=False))
        # 新增标准化后的箱尺寸字段size，来自IYC_CSZ_CSIZECD
        df["size"] = df["IYC_CSZ_CSIZECD"].map(normalize_code)
        df = df[df["voy_id"] == vessel].copy()

        # 统计size以20开头的箱量作为D20；以40或45开头的箱量作为D40
        d20[vessel] = int(df["size"].fillna("").str.startswith("20").sum())
        d40[vessel] = int(
            df["size"].fillna("").str.startswith(("40", "45")).sum()
        )

    return d20, d40


def read_history(path: Path, theta: pd.Timestamp) -> pd.DataFrame:
    """
    输入历史表路径path和当前规划时刻theta，读取当前时刻之前的历史规划记录。

    历史表不存在时返回空表；存在时只保留 `event_time < theta` 且未取消的记录。
    这张表是滚动规划的状态载体，后续 H、P_prev、G_prev 和容量预留都来自它。
    """

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "event_time",
                "voy_id",
                "area_no",      # 箱区号
                "size",
                "planned_qty",  # 当前规划箱量
                "distance",     # 航次泊位到箱区的距离
                "stage",        # 规划阶段
                "status",       # 规划记录状态（planned, arrived, cancelled）
            ]
        )

    hist = pd.read_csv(path)
    hist = hist.copy()
    hist["event_time"] = pd.to_datetime(hist["event_time"], errors="coerce")

    # 只保留发生在当前规划时刻theta之前的历史记录
    hist = hist[hist["event_time"] < theta].copy()

    if "status" in hist.columns:
        # 去掉status == "cancelled"的记录
        hist = hist[
            hist["status"].astype(str).str.lower() != "cancelled"
        ].copy()

    # 标准化航次号、箱区号、箱尺寸、规划数量等字段
    hist["voy_id"] = hist["voy_id"].map(lambda x: normalize_code(x, upper=False))
    hist["area_no"] = hist["area_no"].map(normalize_code)
    hist["size"] = hist["size"].map(normalize_code)
    hist["planned_qty"] = pd.to_numeric(hist["planned_qty"], errors="coerce").fillna(0)
    return hist


def read_planned_quantities(
    hist: pd.DataFrame,
    vessels: Iterable[Vessel],
) -> Tuple[Dict[Vessel, int], Dict[Vessel, int]]:
    """
    从历史表统计每个航次当前时刻之前已经规划的 P20_prev/P40_prev。
    """

    # 初始化每个航次的历史规划量
    p20 = {v: 0 for v in vessels}
    p40 = {v: 0 for v in vessels}

    if hist.empty:
        return p20, p40

    # 按航次号分组，逐个航线统计历史规划量
    for vessel, sub in hist.groupby("voy_id"):
        if vessel not in p20:
            continue
        sizes = sub["size"].fillna("")
        p20[vessel] = int(sub.loc[sizes.str.startswith("20"), "planned_qty"].sum())
        p40[vessel] = int(sub.loc[sizes.str.startswith(("40", "45")), "planned_qty"].sum())

    return p20, p40


def select_due_vessels(
    vessel_ids: Sequence[Vessel],
    open_time_by_vessel: Dict[Vessel, pd.Timestamp],    # 每个航次的开港时间字典
    theta: pd.Timestamp,
    demand20_total: Dict[Vessel, int],
    demand40_total: Dict[Vessel, int],
    p20_prev: Dict[Vessel, int],
    p40_prev: Dict[Vessel, int],
) -> Tuple[List[Vessel], Dict[Vessel, str], Dict[Vessel, float], List[Vessel], List[Vessel]]:
    """
    判断当前滚动时刻需要新增规划的航次。会同时判断该航次是第一次规划还是后续规划。

    对每个航次，先找出截至 `theta` 已经到期的最高规划事件，再用累计目标箱量
    减去历史已规划箱量，得到本轮新增需求。若新增需求为 0，则该航次不进入本轮模型。
    """

    v_plus: List[Vessel] = []
    stage_by_vessel: Dict[Vessel, str] = {}     # 初始化航次到规划阶段的映射
    ratio_by_vessel: Dict[Vessel, float] = {}   # 初始化航次到累计规划比例的映射
    first_plan_vessels: List[Vessel] = []       # 初始化第一次规划航次列表
    followup_plan_vessels: List[Vessel] = []    # 初始化后续规划航次列表

    # 遍历所有待检查的船次
    for vessel in vessel_ids:
        open_time = open_time_by_vessel.get(vessel)
        if open_time is None or pd.isna(open_time):
            raise KeyError(f"Missing open time for vessel {vessel}")

        # 如果当前时间已经越过多个事件，直接补做到最新到期事件的累计比例。
        due_events = [
            (ratio, label)
            for offset, ratio, label in PLANNING_EVENTS
            if theta >= open_time + offset
        ]
        # 如果没有任何到期事件，说明该航次当前还不应该规划
        if not due_events:
            continue

        # 选取最后一个到期事件，也就是最新的到期阶段
        ratio, label = due_events[-1]
        # 计算截至当前阶段应该累计规划的20ft/40ft箱数量
        target20 = math.ceil(ratio * demand20_total.get(vessel, 0))
        target40 = math.ceil(ratio * demand40_total.get(vessel, 0))
        # 计算本轮新增的20ft/40ft箱需求
        inc20 = max(0, target20 - p20_prev.get(vessel, 0))
        inc40 = max(0, target40 - p40_prev.get(vessel, 0))

        if inc20 <= 0 and inc40 <= 0:
            continue

        v_plus.append(vessel)
        stage_by_vessel[vessel] = label
        ratio_by_vessel[vessel] = ratio

        # 没有历史规划量的到期航次按首次规划处理；已有历史则进入后续规划。
        prev_total = p20_prev.get(vessel, 0) + p40_prev.get(vessel, 0)
        if label == "first_70" or prev_total <= 0:
            first_plan_vessels.append(vessel)
        else:
            followup_plan_vessels.append(vessel)

    return (
        v_plus,
        stage_by_vessel,
        ratio_by_vessel,
        first_plan_vessels,
        followup_plan_vessels,
    )


def build_active_vessels(
    v_plus: Sequence[Vessel],
    hist: pd.DataFrame,
    vessel_info: pd.DataFrame,
    theta: pd.Timestamp,
) -> List[Vessel]:
    """
    构造活跃航次集合 V_act。

    当前规则：
    1. 本轮新增规划航次一定视为活跃；
    2. 历史中出现过的航次，只有在当前 theta 尚未超过离港时间时才继续视为活跃。

    离港后，该航次相关出口箱通常已经不再占用堆场资源，不应继续参与箱区共用、
    单箱区服务航次数、同日开港避让等约束统计。
    """

    active = set(v_plus)    # 先把本轮新增规划航次放入活跃集合
    vessel_info_by_id = vessel_info.drop_duplicates("voy_id").set_index("voy_id")

    if not hist.empty:
        for vessel in hist["voy_id"].dropna().astype(str).unique().tolist():
            if vessel not in vessel_info_by_id.index:
                continue

            # 历史中出现过的航次，只有当theta < departure_time的时候才继续视为active，这样离港之后的历史航次不会再进入V_act。
            departure_time = vessel_info_by_id.loc[vessel, "departure_time"]
            if pd.isna(departure_time) or theta < pd.Timestamp(departure_time):
                active.add(vessel)

    return sorted(active)


def infer_area_col(df: pd.DataFrame) -> str:
    """
    从堆场空位表中推断箱区字段名。
    """

    candidates = [
        "YAA_AREANO",
        "YardAreaNo",
        "IYC_YAREANO",
        "YARD_AREA_NO",
        "area_no",
        "AREA_NO",
        "YARD_AREA",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Unable to infer area column")


def infer_empty_mask(df: pd.DataFrame) -> pd.Series:
    """
    从堆场空位表中推断哪些行表示空位。

    优先使用 `HAS_CONTAINER == 0`；如果后续数据源没有这个字段，
    再退化为常见空位标记或箱号/箱 ID 为空的判断。
    """

    if "HAS_CONTAINER" in df.columns:
        values = pd.to_numeric(df["HAS_CONTAINER"], errors="coerce").fillna(-1)
        return values.eq(0)

    # 如果没有HAS_CONTAINER字段就尝试一些常见的“是否空位”字段名
    for col in ["is_empty", "IS_EMPTY", "empty", "EMPTY_FLAG", "isEmpty"]:
        if col in df.columns:
            return df[col].astype(bool)

    for col in ["CNTR_ID", "ContainerId", "IYC_CNTRID", "container_id"]:
        if col in df.columns:
            return df[col].isna() | (df[col].astype(str).str.strip() == "")

    for col in ["CNTR_NO", "ContainerNo", "IYC_CNTRNO", "container_no"]:
        if col in df.columns:
            return df[col].isna() | (df[col].astype(str).str.strip() == "")

    raise ValueError("Unable to infer empty-slot column")


def count_physical_capacity_by_area(
    path_20: Path,
    path_40: Path,
) -> Tuple[Dict[Area, int], Dict[Area, int]]:
    """
    统计每个候选箱区的物理空位容量。

    `bay_slots_detail_20.parquet` 用来计算 20 尺等价空位；
    `bay_slots_detail_40.parquet` 用来计算 40 尺空位。两者不能相加，
    因为它们代表不同容量约束。
    """

    def count_one(path: Path) -> Dict[Area, int]:
        """读取单张 bay slot 表，并按箱区统计空位行数。"""

        df = pd.read_parquet(path)
        area_col = infer_area_col(df)
        empty = infer_empty_mask(df)
        areas = df.loc[empty, area_col].map(normalize_code).dropna()
        return {str(k): int(v) for k, v in areas.value_counts().to_dict().items()}

    return count_one(path_20), count_one(path_40)


def read_reserved_capacity_by_area(
    hist: pd.DataFrame,
) -> Tuple[Dict[Area, int], Dict[Area, int]]:
    """
    从历史规划中统计尚未实际进场但已预留的容量。

    只扣除 `status == planned` 的记录，避免已进场箱量被实时堆场表和历史预留
    重复扣减。
    """

    reserve20_equiv: Dict[Area, int] = {}
    reserve40: Dict[Area, int] = {}

    if hist.empty:
        return reserve20_equiv, reserve40

    reserve = hist.copy()
    if "status" in reserve.columns:
        reserve = reserve[
            reserve["status"].astype(str).str.lower() == "planned"
        ].copy()

    if reserve.empty:
        return reserve20_equiv, reserve40

    for area, sub in reserve.groupby("area_no"):
        sizes = sub["size"].fillna("")
        qty20 = sub.loc[sizes.str.startswith("20"), "planned_qty"].sum()
        qty40 = sub.loc[sizes.str.startswith(("40", "45")), "planned_qty"].sum()
        reserve20_equiv[str(area)] = int(qty20 + 2 * qty40)
        reserve40[str(area)] = int(qty40)

    return reserve20_equiv, reserve40


def calc_available_capacity(
    physical20: Dict[Area, int],
    physical40: Dict[Area, int],
    reserve20_equiv: Dict[Area, int],
    reserve40: Dict[Area, int],
    candidate_areas: Sequence[Area],
) -> Tuple[Dict[Area, int], Dict[Area, int]]:
    """
    计算当前可用于新规划的 C20/C40。

    可用容量 = 实时物理空位 - 历史规划预留容量，结果下限为 0。
    """

    c20: Dict[Area, int] = {}
    c40: Dict[Area, int] = {}

    for area in candidate_areas:
        c20[area] = max(0, int(physical20.get(area, 0)) - int(reserve20_equiv.get(area, 0)))
        c40[area] = max(0, int(physical40.get(area, 0)) - int(reserve40.get(area, 0)))

    return c20, c40


def calc_incremental_demands(
    v_plus: Sequence[Vessel],
    ratio_by_vessel: Dict[Vessel, float],
    demand20_total: Dict[Vessel, int],
    demand40_total: Dict[Vessel, int],
    p20_prev: Dict[Vessel, int],
    p40_prev: Dict[Vessel, int],
) -> Tuple[Dict[Vessel, int], Dict[Vessel, int]]:
    """
    计算本轮模型真正需要新增分配的 R20/R40。

    使用当前到期事件的累计比例 rho 乘以总需求，再扣除历史已规划量。
    """

    r20: Dict[Vessel, int] = {}
    r40: Dict[Vessel, int] = {}

    for vessel in v_plus:
        ratio = ratio_by_vessel[vessel]
        r20[vessel] = max(
            0,
            math.ceil(ratio * demand20_total.get(vessel, 0)) - p20_prev.get(vessel, 0),
        )
        r40[vessel] = max(
            0,
            math.ceil(ratio * demand40_total.get(vessel, 0)) - p40_prev.get(vessel, 0),
        )

    return r20, r40


def find_distance_matrix_sheet(path: Path) -> pd.DataFrame:
    """
    在距离矩阵 Excel 中自动寻找包含 `area_no` 和泊位列的工作表。
    """

    xls = pd.ExcelFile(path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        columns = {str(c) for c in df.columns}
        if "area_no" in columns and any(c.startswith("B") for c in columns):
            return df
    raise ValueError(f"No distance-matrix sheet found in {path}")


def read_distance_param(
    distance_matrix_path: Path,
    berth_by_vessel: Dict[Vessel, str],
    v_act: Sequence[Vessel],
    candidate_areas: Sequence[Area],
) -> Dict[Tuple[Vessel, Area], float]:
    """
    构造航次到箱区的距离参数 `distance[(v, a)]`。

    先根据航次泊位找到距离矩阵中的泊位列，再为每个活跃航次和候选箱区
    生成一条距离记录。
    """

    dist_df = find_distance_matrix_sheet(distance_matrix_path)
    area_col = "area_no" if "area_no" in dist_df.columns else dist_df.columns[0]
    dist_df = dist_df.copy()
    dist_df[area_col] = dist_df[area_col].map(normalize_code)
    dist_df = dist_df.dropna(subset=[area_col]).set_index(area_col)

    distance: Dict[Tuple[Vessel, Area], float] = {}
    for vessel in v_act:
        berth = berth_by_vessel[vessel]
        # 构造可能的距离矩阵列名。例如泊位是B6，则可能列名是B6，也可能是6
        possible_cols = [berth, berth.removeprefix("B")]
        col = next((c for c in possible_cols if c in dist_df.columns), None)
        if col is None:
            raise KeyError(f"Distance matrix does not contain berth column {berth}")

        for area in candidate_areas:
            if area not in dist_df.index:
                raise KeyError(f"Distance matrix does not contain area {area}")
            distance[(vessel, area)] = float(dist_df.loc[area, col])

    return distance


def read_history_area_param(
    hist: pd.DataFrame,
    v_act: Sequence[Vessel],
    areas: Sequence[Area],
) -> Dict[Tuple[Vessel, Area], int]:
    """
    构造历史已选箱区参数 H。

    只要历史中某航次曾向某箱区分配过正箱量，则 `H[(v, a)] = 1`。
    没有历史记录时，所有组合默认为 0。
    """

    h = {(v, a): 0 for v in v_act for a in areas}
    if hist.empty:
        return h

    active_set = set(v_act)
    area_set = set(areas)
    for _, row in hist.iterrows():
        vessel = row["voy_id"]
        area = row["area_no"]
        qty = float(row["planned_qty"])
        if vessel in active_set and area in area_set and qty > 0:
            h[(vessel, area)] = 1

    return h


def read_previous_distance_cost(
    hist: pd.DataFrame,
    v_act: Sequence[Vessel],
    distance: Dict[Tuple[Vessel, Area], float],
) -> Dict[Vessel, float]:
    """
    计算历史累计距离成本 G_prev。

    如果历史表中保存了 `distance` 字段，就直接使用；否则使用当前距离矩阵
    按 `(航次, 箱区)` 重新计算。
    """

    g_prev = {v: 0.0 for v in v_act}
    if hist.empty:
        return g_prev

    if "distance" in hist.columns and hist["distance"].notna().any():
        work = hist.copy()
        work["distance"] = pd.to_numeric(work["distance"], errors="coerce")
        # 删除距离为空的记录后按航次号分组
        for vessel, sub in work.dropna(subset=["distance"]).groupby("voy_id"):
            # 只记录当前活跃航次
            if vessel in g_prev:
                g_prev[vessel] = float((sub["distance"] * sub["planned_qty"]).sum())
        return g_prev

    for _, row in hist.iterrows():
        vessel = row["voy_id"]
        area = row["area_no"]
        if vessel in g_prev and (vessel, area) in distance:
            g_prev[vessel] += float(row["planned_qty"]) * distance[(vessel, area)]

    return g_prev


def capacity_unavailable(
    vessel: Vessel,
    area: Area,
    r20: Dict[Vessel, int],
    r40: Dict[Vessel, int],
    c20: Dict[Area, int],
    c40: Dict[Area, int],
) -> bool:
    """
    判断箱区是否因容量不足而对某航次不可用。

    第一版采用保守规则：没有 20 尺等价容量则不可用；如果航次有 40 尺需求，
    但该箱区没有 40 尺容量，也不可用。
    """

    if c20.get(area, 0) <= 0:
        return True
    if r40.get(vessel, 0) > 0 and c40.get(area, 0) <= 0:
        return True
    return False


def bay_to_area(value: object) -> Optional[Area]:
    """
    将 TOPS 中的贝位/位置编码粗略映射到箱区编码。

    例如 `5602` 映射为 `56`，`E8D0` 映射为 `E8`。
    """

    code = normalize_code(value)
    if not code or len(code) < 2:
        return None
    return code[:2]


def parse_tops_time_column(series: pd.Series) -> pd.Series:
    """
    解析 TOPS 时间字段。

    TOPS 源表中的 `SPL_STDATE`/`SPL_EDDATE` 可能是 pandas 已识别的日期时间，
    也可能是 Unix 时间戳。这里兼容秒、毫秒、微秒、纳秒四种常见 Unix 单位；
    无法识别的值会转为 NaT，后续不会被判断为生效计划。
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    numeric_values = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric_values.notna()

    if numeric_mask.any():
        abs_values = numeric_values.abs()
        unit_masks = [
            (abs_values >= 1e17, "ns"),
            ((abs_values >= 1e14) & (abs_values < 1e17), "us"),
            ((abs_values >= 1e11) & (abs_values < 1e14), "ms"),
            ((abs_values < 1e11) & numeric_mask, "s"),
        ]

        for unit_mask, unit in unit_masks:
            mask = numeric_mask & unit_mask
            if mask.any():
                result.loc[mask] = pd.to_datetime(
                    numeric_values.loc[mask],
                    unit=unit,
                    errors="coerce",
                )

    text_mask = ~numeric_mask
    if text_mask.any():
        result.loc[text_mask] = pd.to_datetime(
            series.loc[text_mask],
            errors="coerce",
        )

    return result


def active_tops_areas_for_vessel(
    tops_df: Optional[pd.DataFrame],
    theta: pd.Timestamp,
    vessel: Vessel,
) -> set[Area]:
    """
    识别当前时刻，被其他 TOPS 生效计划占用的箱区。当前航次v不应该再使用这些箱区

    会排除当前航次自己的计划；目前根据 `SPR_STBAY`/`SPR_EDBAY` 前两位
    近似映射箱区，后续若有更精确的范围展开规则，可在这里替换。
    TOPS 是否生效只使用计划层面的 `SPL_STDATE <= theta < SPL_EDDATE` 判断。
    """

    if tops_df is None or tops_df.empty:
        return set()

    df = tops_df.copy()
    if not {"SPL_STDATE", "SPL_EDDATE"}.issubset(df.columns):
        return set()

    df["SPL_STDATE"] = parse_tops_time_column(df["SPL_STDATE"])
    df["SPL_EDDATE"] = parse_tops_time_column(df["SPL_EDDATE"])

    mask = pd.Series(True, index=df.index)
    if "SPL_ISVALID" in df.columns:
        mask &= df["SPL_ISVALID"].astype(str).str.upper().eq("Y")
    # 生效时间判断：当前时刻必须落在TOPS计划生效区间内
    mask &= (df["SPL_STDATE"] <= theta) & (theta < df["SPL_EDDATE"])

    # 得到当前时刻真正生效的TOPS计划记录
    active = df[mask].copy()
    # 排除当前航次自己的TOPS计划，只保留其他航次或其他计划的占用
    if "SPL_CONDITIONCODE" in active.columns:
        active["condition_vessel"] = active["SPL_CONDITIONCODE"].map(
            lambda x: normalize_code(x, upper=False)
        )
        active = active[active["condition_vessel"] != vessel].copy()

    areas: set[Area] = set()
    for _, row in active.iterrows():
        for col in ["SPR_STBAY", "SPR_EDBAY"]:
            if col in row:
                area = bay_to_area(row[col])
                if area:
                    areas.add(area)

    return areas


def build_unavailable_param(
    v_plus: Sequence[Vessel],
    v_act: Sequence[Vessel],
    areas: Sequence[Area],
    of_areas: set[Area],
    closed_areas: set[Area],
    r20: Dict[Vessel, int],
    r40: Dict[Vessel, int],
    c20: Dict[Area, int],
    c40: Dict[Area, int],
    theta: pd.Timestamp,
    tops_df: Optional[pd.DataFrame] = None,
    include_tops: bool = True,
) -> Dict[Tuple[Vessel, Area], int]:
    """
    构造箱区不可用参数 B。

    `B[(v, a)] = 1` 表示箱区 a 对航次 v 不可用。当前综合考虑：
    OF 适放范围、关闭箱区、当前容量不足，以及 TOPS 生效计划占用。
    """

    b: Dict[Tuple[Vessel, Area], int] = {}  # 初始化结果字典
    v_plus_set = set(v_plus)                

    # TOPS 检查相对耗时，按航次预先缓存，避免在箱区循环中重复过滤表。
    tops_cache = {
        vessel: active_tops_areas_for_vessel(tops_df, theta, vessel)
        for vessel in v_act
    } if include_tops else {vessel: set() for vessel in v_act}

    for vessel in v_act:
        for area in areas:
            unavailable = False
            if area not in of_areas:
                unavailable = True
            if area in closed_areas:
                unavailable = True
            if vessel in v_plus_set and capacity_unavailable(vessel, area, r20, r40, c20, c40):
                unavailable = True
            if area in tops_cache[vessel]:
                unavailable = True
            b[(vessel, area)] = 1 if unavailable else 0

    return b


def calc_mixed_capacity_for_vessel(
    vessel: Vessel,
    area: Area,
    demand20_total: Dict[Vessel, int],
    demand40_total: Dict[Vessel, int],
    c20: Dict[Area, int],
    c40: Dict[Area, int],
) -> float:
    """
    估算箱区对某航次的混合有效容量。

    该容量用于第一次规划时估算推荐箱区数量 K，不直接作为模型约束。
    它会根据航次 20/40 尺箱比例，同时考虑 20 尺等价容量和 40 尺容量。
    """

    total = demand20_total.get(vessel, 0) + demand40_total.get(vessel, 0)
    if total <= 0:
        return 0.0

    p40 = demand40_total.get(vessel, 0) / total
    p20 = 1.0 - p40

    if p40 == 0:
        return float(c20.get(area, 0))
    if p20 == 0:
        return min(c20.get(area, 0) / 2.0, float(c40.get(area, 0)))

    cap_by_20_equiv = c20.get(area, 0) / (p20 + 2 * p40)
    cap_by_40 = c40.get(area, 0) / p40
    return min(cap_by_20_equiv, cap_by_40)


def estimate_area_count_bounds(
    vessel: Vessel,
    demand20_total: Dict[Vessel, int],
    demand40_total: Dict[Vessel, int],
    c20: Dict[Area, int],
    c40: Dict[Area, int],
    b: Dict[Tuple[Vessel, Area], int],
    areas: Sequence[Area],
) -> Tuple[int, int]:
    """
    估算第一次规划航次的主箱区数量上下界 K_min/K_max。

    估算使用航次总箱量，而不是本轮 70% 新增箱量，因为首轮选出的主箱区
    后续还要承接剩余 30% 的补充规划。
    """

    total_demand = demand20_total.get(vessel, 0) + demand40_total.get(vessel, 0)
    if total_demand <= 0:
        return 0, 0

    available_areas = [area for area in areas if b.get((vessel, area), 1) == 0]
    if not available_areas:
        raise ValueError(f"Vessel {vessel} has no available area")

    mixed_caps = [
        calc_mixed_capacity_for_vessel(
            vessel, area, demand20_total, demand40_total, c20, c40
        )
        for area in available_areas
    ]
    positive_caps = [cap for cap in mixed_caps if cap > 0]
    if not positive_caps:
        raise ValueError(f"Vessel {vessel} has no positive mixed capacity")

    avg_cap = sum(positive_caps) / len(positive_caps)
    k0 = math.ceil(total_demand / avg_cap)
    k_min = max(1, k0 - 1)
    k_max = min(k0 + 1, len(available_areas))
    if k_min > k_max:
        k_min = k_max
    return k_min, k_max


def build_same_day_pairs(
    v_act: Sequence[Vessel],
    open_time_by_vessel: Dict[Vessel, pd.Timestamp],
) -> List[Tuple[Vessel, Vessel]]:
    """
    构造同日开港航次对 P_same。

    这些航次对共用同一箱区时，会在模型目标函数中产生惩罚。
    """

    pairs: List[Tuple[Vessel, Vessel]] = []
    for u, v in combinations(v_act, 2):
        if u not in open_time_by_vessel or v not in open_time_by_vessel:
            continue
        if pd.Timestamp(open_time_by_vessel[u]).date() == pd.Timestamp(open_time_by_vessel[v]).date():
            pairs.append((u, v))
    return pairs


def build_area_allocation_data(
    *,
    theta: pd.Timestamp,                        # 当前滚动规划时刻
    vessel_ids: Sequence[Vessel],               # 本次关注的航次列表
    base_dir: Path,                             # 项目根目录
    data_dir: Optional[Path] = None,            # 数据目录
    plan_history_path: Optional[Path] = None,   # 历史规划文件路径
    include_tops: bool = True,                  # 是否启用TOPS占用箱区判断
    allow_unmet_demand: bool = True,            # 是否允许模型出现未满足需求的解
) -> BuildArtifacts:
    """
    构造单个滚动时刻 theta 下的完整模型输入。

    这是本脚本的核心参数构造函数。它按以下顺序工作：
    1. 读取候选箱区、航次开港时间/泊位、总箱量和历史计划；
    2. 根据 theta 判断 V_plus、first_plan_vessels、followup_plan_vessels；
    3. 计算容量、距离、历史状态、不可用箱区和 K 上下界；
    4. 组装 `YardAllocationData` 并连同辅助信息一起返回。
    """

    data_dir = discover_data_dir(base_dir, data_dir)
    plan_history_path = plan_history_path or base_dir / DEFAULT_PLAN_HISTORY

    # 1. 候选箱区：OF 适放箱区减去关闭箱区。
    closed_areas = read_closed_areas(data_dir / "n_usefg_areas.txt")
    of_area_path = next(data_dir.glob("*.xlsx"))
    of_areas_list = read_of_areas(of_area_path, closed_areas)
    of_areas = set(of_areas_list)

    # 2. 航次基础信息：开港时间用于滚动事件判断，泊位用于距离参数。
    vessel_info = read_vessel_berth_info(data_dir / "vessel_berth_info.csv")
    vessel_info_by_id = vessel_info.drop_duplicates("voy_id").set_index("voy_id")
    # 构建航次到开港时间的字典
    open_time_by_vessel = {
        str(v): pd.Timestamp(row["open_time"])
        for v, row in vessel_info_by_id.iterrows()
        if pd.notna(row["open_time"])
    }
    # 构建航次到泊位的字典
    berth_by_vessel = {
        str(v): str(row["berth"])
        for v, row in vessel_info_by_id.iterrows()
        if pd.notna(row["berth"])
    }

    # 3. 总需求和历史状态。
    # 把输入航次号统一转成字符串
    vessel_ids = [str(v) for v in vessel_ids]
    # 读取每个航次的总需求
    demand20_total, demand40_total = read_vessel_demands(data_dir, vessel_ids)
    # 读取当前时刻之前的有效历史规划记录
    hist = read_history(plan_history_path, theta)
    # 统计关注航次当前时刻之前已规划的20/40ft箱量
    selected_p20_prev, selected_p40_prev = read_planned_quantities(hist, vessel_ids)

    # 4. 根据当前时刻判断哪些航次需要本轮新增规划。
    (
        v_plus,                 # 本轮需要新增规划的航次
        stage_by_vessel,        # 航次对应阶段
        ratio_by_vessel,        # 航次对应累计比例
        first_plan_vessels,     # 第一次规划航次
        followup_plan_vessels,  # 后续规划航次
    ) = select_due_vessels(
        vessel_ids,
        open_time_by_vessel,
        theta,
        demand20_total,
        demand40_total,
        selected_p20_prev,
        selected_p40_prev,
    )

    if not v_plus:
        raise ValueError(f"No selected vessel has incremental demand due at {theta}")

    # 5. 活跃航次和本轮新增需求。
    v_act = build_active_vessels(v_plus, hist, vessel_info, theta)
    p20_prev, p40_prev = read_planned_quantities(hist, v_act)
    r20, r40 = calc_incremental_demands(
        v_plus,
        ratio_by_vessel,
        demand20_total,
        demand40_total,
        p20_prev,
        p40_prev,
    )

    # 6. 当前容量：物理空位扣除历史预留。
    physical20, physical40 = count_physical_capacity_by_area(
        data_dir / "bay_slots_detail_20.parquet",
        data_dir / "bay_slots_detail_40.parquet",
    )
    reserve20_equiv, reserve40 = read_reserved_capacity_by_area(hist)
    c20, c40 = calc_available_capacity(
        physical20,
        physical40,
        reserve20_equiv,
        reserve40,
        of_areas_list,
    )

    # 7. 距离、历史已选箱区和历史累计距离成本，即构造distance[(v,a)], H[(v,a)], G_prev[v]。
    distance_matrix_path = next(base_dir.glob("of_*.xlsx"))
    missing_berth = [v for v in v_act if v not in berth_by_vessel]
    if missing_berth:
        raise KeyError(f"Missing berth for active vessels: {missing_berth}")
    distance = read_distance_param(
        distance_matrix_path,
        berth_by_vessel,
        v_act,
        of_areas_list,
    )

    h = read_history_area_param(hist, v_act, of_areas_list)
    g_prev = read_previous_distance_cost(hist, v_act, distance)

    # 8. 不可用箱区 B：把关闭、非适放、容量不足、TOPS 占用等规则汇总。
    tops_df = pd.read_parquet(data_dir / "tops_plan_info.parquet")
    b = build_unavailable_param(
        v_plus,
        v_act,
        of_areas_list,
        of_areas,
        closed_areas,
        r20,
        r40,
        c20,
        c40,
        theta,
        tops_df=tops_df,
        include_tops=include_tops,
    )

    # 9. 第一次规划航次需要主箱区数量上下界。
    k_min: Dict[Vessel, int] = {}
    k_max: Dict[Vessel, int] = {}
    for vessel in first_plan_vessels:
        k_min[vessel], k_max[vessel] = estimate_area_count_bounds(
            vessel,
            demand20_total,
            demand40_total,
            c20,
            c40,
            b,
            of_areas_list,
        )

    # 10. 同日开港航次对，用于减少共用箱区。
    p_same = build_same_day_pairs(v_act, open_time_by_vessel)

    # 组装YardAllocationData
    allocation_data = YardAllocationData(
        V_plus=v_plus,
        V_act=v_act,
        A=of_areas_list,
        P_same=p_same,
        R20=r20,
        R40=r40,
        C20=c20,
        C40=c40,
        distance=distance,
        B=b,
        H=h,
        P20_prev=p20_prev,
        P40_prev=p40_prev,
        G_prev=g_prev,
        first_plan_vessels=first_plan_vessels,
        followup_plan_vessels=followup_plan_vessels,
        K_min=k_min,
        K_max=k_max,
        allow_unmet_demand=allow_unmet_demand,
    )

    # 返回BuildArtifacts，包含构造的参数和一些辅助信息
    return BuildArtifacts(
        data=allocation_data,
        theta=theta,
        stage_by_vessel=stage_by_vessel,
        ratio_by_vessel=ratio_by_vessel,
        demand20_total=demand20_total,
        demand40_total=demand40_total,
        berth_by_vessel={v: berth_by_vessel[v] for v in v_act},
        capacity20_physical=physical20,
        capacity40_physical=physical40,
    )


def build_plan_rows(
    result: Dict[str, object],
    artifacts: BuildArtifacts,
    *,
    status: str = "planned",
) -> pd.DataFrame:
    """
    将Gurobi求解结果转换为可写入 `plan_history.csv` 的明细行。

    每行表示某航次、某箱区、某尺寸的一次规划箱量，同时保存距离和规划阶段，
    供下一轮滚动规划读取。
    """

    rows = []               # 初始化结果行列表，用来存放每一行的历史记录
    data = artifacts.data   # 取出模型输入数据，后面会用distance[(vessel, area)]

    # 分别处理20ft和40ft的求解结果
    for size, key in [("20", "x20"), ("40", "x40")]:
        allocations = result.get(key, {})
        if not isinstance(allocations, dict):
            continue
        # 遍历每一条分配结果
        for (vessel, area), qty in allocations.items():
            if qty <= 0:
                continue
            rows.append(
                {
                    "event_time": artifacts.theta,
                    "voy_id": vessel,
                    "area_no": area,
                    "size": size,
                    "planned_qty": int(qty),
                    "distance": data.distance[(vessel, area)],
                    "stage": artifacts.stage_by_vessel.get(vessel, ""),
                    "status": status,   # 写入状态，默认是planned
                }
            )

    return pd.DataFrame(rows)


def append_plan_history(plan_history_path: Path, rows: pd.DataFrame) -> None:
    """
    将本轮规划结果追加写入历史计划表。

    历史表是后续滚动时刻计算 H、P_prev、G_prev 和容量预留的依据。
    plan_history_path是历史文件路径，rows是要写入的规划结果表
    """

    if rows.empty:
        return
    plan_history_path.parent.mkdir(parents=True, exist_ok=True)
    if plan_history_path.exists():
        # 读取旧的历史文件，并将旧历史记录和本轮新纪录纵向拼接
        old = pd.read_csv(plan_history_path)
        combined = pd.concat([old, rows], ignore_index=True)
    else:
        combined = rows
    # 把合并后的历史表写回CSV文件
    combined.to_csv(plan_history_path, index=False, encoding="utf-8-sig")


def print_case_summary(artifacts: BuildArtifacts, result: Dict[str, object]) -> None:
    """
    打印本轮算例的关键输入参数和求解结果摘要。
    """

    data = artifacts.data
    print("theta:", artifacts.theta)
    print("V_plus:", data.V_plus)
    print("V_act:", data.V_act)
    print("first_plan_vessels:", data.first_plan_vessels)
    print("followup_plan_vessels:", data.followup_plan_vessels)
    print("P_same:", data.P_same)
    print("R20:", data.R20)
    print("R40:", data.R40)
    print("K_min:", data.K_min)
    print("K_max:", data.K_max)
    print("berth_by_vessel:", artifacts.berth_by_vessel)
    print("total C20:", sum(data.C20.values()))
    print("total C40:", sum(data.C40.values()))
    print("available area count by vessel:", {
        v: sum(1 for a in data.A if data.B[(v, a)] == 0)
        for v in data.V_plus
    })
    print("status:", result["status"])
    print("objective_value:", result["objective_value"])
    print("eta:", result["eta"])
    print("unmet s20:", result["s20"])
    print("unmet s40:", result["s40"])


build_yard_allocation_data = build_area_allocation_data


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    默认运行 2026-05-08 09:30 下 453334 和 453400 两个航次的首轮规划。
    """

    parser = argparse.ArgumentParser(
        description="Build real-case parameters and solve yard area allocation."
    )
    parser.add_argument("--theta", default=DEFAULT_THETA)
    parser.add_argument("--vessels", nargs="+", default=DEFAULT_VESSELS)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--plan-history", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-tops", action="store_true")
    parser.add_argument("--strict-demand", action="store_true")
    parser.add_argument("--write-history", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    命令行入口：构造参数、调用求解器、打印结果，并按需写回历史表。
    """

    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    theta = pd.Timestamp(args.theta)
    plan_history_path = args.plan_history or base_dir / DEFAULT_PLAN_HISTORY

    # 开始构造完整的模型输入
    artifacts = build_area_allocation_data(
        theta=theta,
        vessel_ids=args.vessels,
        base_dir=base_dir,
        data_dir=args.data_dir,
        plan_history_path=plan_history_path,
        include_tops=not args.no_tops,
        allow_unmet_demand=not args.strict_demand,
    )

    # 如果命令行带了--build-only，那么只构造参数不求解
    if args.build_only:
        print_case_summary(
            artifacts,
            {
                "status": "BUILD_ONLY",
                "objective_value": None,
                "eta": None,
                "s20": {},
                "s40": {},
            },
        )
        return

    # 调用gurobi模型求解
    result = solve_yard_area_allocation(
        artifacts.data,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        verbose=not args.quiet,
    )
    print_case_summary(artifacts, result)

    rows = build_plan_rows(result, artifacts)
    if rows.empty:
        print("No positive allocation rows were produced.")
    else:
        print("\nallocation rows:")
        print(rows.to_string(index=False))

    if args.write_history:
        append_plan_history(plan_history_path, rows)
        print(f"\nAppended plan history: {plan_history_path}")


if __name__ == "__main__":
    main()
