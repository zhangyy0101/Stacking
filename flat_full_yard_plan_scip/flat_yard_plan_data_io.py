from __future__ import annotations

import csv
import json
import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


DEFAULT_PLANNING_TIME = "2026-05-08 09:30:00"
DEFAULT_EXPORT_VESSELS = ["453334", "453400"]
DEFAULT_IMPORT_VESSELS = ["453886", "454063"]
DEFAULT_TARGET_VOYAGES = ("453334", "453400")
DEFAULT_FLOW_ALIASES = {"IE": "IF", "RF": "IF", "RE": "IF"}
DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO = 2.0 / 3.0
SIZE_MODES = ("20", "40", "45")


@dataclass(frozen=True)
class YardPlanningWeights:
    miss: float = 100.0
    operation: float = 50.0
    of_area: float = 40.0
    distance: float = 30.0
    share: float = 20.0
    adjustment: float = 10.0
    balance: float = 1.0


@dataclass
class LargePlanningData:
    V: Sequence[str]
    F: Sequence[str]
    A: Sequence[str]
    D20: Mapping[tuple[str, str], float]
    D40: Mapping[tuple[str, str], float]
    C20: Mapping[str, float]
    C40: Mapping[str, float]
    H: Mapping[str, float]
    distance: Mapping[tuple[str, str], float]
    U: Mapping[tuple[str, str], int]
    E20: Mapping[tuple[str, str, str], int]
    E40: Mapping[tuple[str, str, str], int]
    C20Direct: Mapping[str, float] | None = None
    S20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    S40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    L20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    L40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    Q20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    Q40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    R20S: Mapping[tuple[str, str], float] | None = None
    R40S: Mapping[tuple[str, str], float] | None = None
    TOPS20: Mapping[tuple[str, str], float] = field(default_factory=dict)
    TOPS40: Mapping[tuple[str, str], float] = field(default_factory=dict)
    Cbar20: Mapping[tuple[str, str], float] | None = None
    Cbar20Direct: Mapping[tuple[str, str], float] | None = None
    Cbar40: Mapping[tuple[str, str], float] | None = None
    P20: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    P40: Mapping[tuple[str, str, str], float] = field(default_factory=dict)
    O: Mapping[str, int] = field(default_factory=dict)
    M: Mapping[str, float] | None = None
    OFWorkLanes: Mapping[str, float] = field(default_factory=dict)
    weights: YardPlanningWeights = field(default_factory=YardPlanningWeights)
    allow_unmet_demand: bool = True
    strict_validation: bool = True
    default_U: int = 1
    default_E20: int = 1
    default_E40: int = 1
    default_distance: float = 0.0
    default_H: float | None = None
    name: str = "planning_large"


@dataclass(frozen=True)
class PlanningInputArtifacts:
    data: LargePlanningData
    planning_time: pd.Timestamp
    export_vessels: list[str]
    import_vessels: list[str]
    area_functions: dict[str, set[str]]
    berth_by_vessel: dict[str, str]
    previous_plan_rows: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DemandRow:
    voyage_id: str
    flow: str
    port: str
    size_mode: str
    predicted_boxes: int
    ratio_target_boxes: int
    doc_boxes: int
    planned_boxes: int
    planning_stage: str
    planning_ratio: float


@dataclass(frozen=True)
class BoxGroup:
    group_id: str
    voyage_id: str
    size: str
    height: str
    status: str
    port: str
    operator: str
    ctype: str
    weight_class: str
    reefer: bool
    dangerous: bool
    over_limit: bool
    special_codes: tuple[str, ...]
    demand: int

    @property
    def size_mode(self) -> str:
        return self.size if self.size in {"20", "40", "45"} else "40"

    @property
    def big_plan_size_mode(self) -> str:
        return "40" if self.size_mode == "45" else self.size_mode

    @property
    def special_signature(self) -> str:
        return self.business_special_signature

    @property
    def business_special_signature(self) -> str:
        return "+".join(self.special_codes) if self.special_codes else "NORMAL"


@dataclass
class Bay:
    area_no: str
    bay_no: str
    bay_key: str
    block_id: str
    block_bays: tuple[str, ...]
    block_bay_count: int
    block_boundary_adjusted: bool
    bay_order: int
    cap_by_size: dict[str, int] = field(default_factory=dict)
    physical_capacity: int = 0
    row_cap_by_size: dict[str, dict[str, int]] = field(default_factory=dict)
    row_physical_capacity: dict[str, int] = field(default_factory=dict)
    existing_size_modes: set[str] = field(default_factory=set)
    existing_heights: set[str] = field(default_factory=set)
    existing_special_signatures: set[str] = field(default_factory=set)
    existing_ports: set[str] = field(default_factory=set)
    fallback_reasons: set[str] = field(default_factory=set)
    locked: bool = False

    @property
    def is_fallback_bay(self) -> bool:
        return bool(self.fallback_reasons)

    def capacity_for(self, group: BoxGroup) -> int:
        return int(self.cap_by_size.get(group.size_mode, 0))


@dataclass(frozen=True)
class BigPlanRow:
    voyage_id: str
    flow: str
    area_no: str
    planned_boxes: int
    size_mode: str = "ALL"
    plan_date: str = ""


@dataclass(frozen=True)
class VoyageSchedule:
    voyage_id: str
    receive_start: datetime
    receive_end: datetime
    berth_no: str
    berth_time: datetime
    departure_time: datetime


@dataclass(frozen=True)
class AreaOperation:
    area_no: str
    voyage_id: str
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class SmallBoxGroup:
    group_id: str
    voyage_id: str
    status: str
    port: str
    size: str
    height: str
    weight_class: str
    demand: int
    pre_stow: bool = False
    special_stow: bool = False
    special_stow_code: str = ""


@dataclass
class ProblemData:
    groups: list[BoxGroup]
    small_groups: list[SmallBoxGroup]
    bays: dict[str, Bay]
    big_plan: list[BigPlanRow]
    assigned_areas: dict[tuple[str, str], set[str]]
    area_quota: dict[tuple[str, str, str], int]
    area_size_quota: dict[tuple[str, str, str, str], int]
    area_functions: dict[str, set[str]]
    business_special_codes: set[str]
    planning_time: datetime
    horizon_hours: float
    voyage_windows: dict[str, tuple[datetime, datetime]]
    area_operations: dict[str, list[AreaOperation]]
    target_voyages: list[str]
    berth_distances: dict[tuple[str, str], float] = field(default_factory=dict)
    berth_by_voyage: dict[str, str] = field(default_factory=dict)
    tops_reserved_slot_count: int = 0
    tops_closed_bay_count: int = 0
    misplaced_bay_exclusion_ratio: float = 0.0
    misplaced_excluded_bay_count: int = 0


@dataclass(frozen=True)
class MediumSmallInputs:
    big_plan: list[BigPlanRow]
    demand_rows: list[DemandRow]
    problem: ProblemData


class RollingPlanningState:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.plan_history_path = state_dir / "plan_history.csv"

    def read_history(self) -> pd.DataFrame:
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

    def latest_previous_plan(self, planning_time: pd.Timestamp, vessels: Sequence[str]) -> pd.DataFrame:
        history = self.read_history()
        if history.empty:
            return history
        vessel_set = {normalize_code(v) for v in vessels if normalize_code(v)}
        history = history[history["voy_id"].map(normalize_code).isin(vessel_set)].copy()
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
        previous["size"] = previous["size"].map(normalize_size_large)
        previous["planned_qty"] = pd.to_numeric(previous["planned_qty"], errors="coerce").fillna(0.0)
        for vessel in previous["voy_id"].dropna().unique():
            old_flags[str(vessel)] = 1
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

    def append_solution(self, planning_time: pd.Timestamp, solution: Any) -> pd.DataFrame:
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


def parse_datetime(value: object) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return pd.to_datetime(value, unit="D", origin="1899-12-30").to_pydatetime()
        except (ValueError, TypeError, OverflowError):
            return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return parsed.to_pydatetime()
    return None


def parse_planning_time(value: str) -> pd.Timestamp:
    planning_time = pd.Timestamp(value)
    if pd.isna(planning_time):
        raise ValueError(f"Invalid planning time: {value}")
    return planning_time


def normalize_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return ""
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def normalize_text(value: Any, default: str = "") -> str:
    text = normalize_code(value)
    return text or default


def normalize_voyage(value: Any, fallback: str = "") -> str:
    text = normalize_text(value, fallback)
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def normalize_size_large(value: Any) -> str:
    code = normalize_code(value)
    if code.startswith("20"):
        return "20"
    if code.startswith(("40", "45")):
        return "40"
    return ""


def normalize_size_small(value: Any) -> str:
    code = normalize_code(value)
    return code if code in {"20", "40", "45"} else "40"


def normalize_big_plan_size(value: Any) -> str:
    code = normalize_code(value)
    if code == "ALL":
        return "ALL"
    return normalize_size_small(code) if code else "ALL"


def normalize_flow(value: Any, aliases: Mapping[str, str] | None = None, default: str = "") -> str:
    flow = normalize_code(value)
    if not flow:
        return default
    return (aliases or {}).get(flow, flow)


def normalize_bay(value: Any) -> str:
    code = normalize_code(value)
    if code.isdigit():
        return f"{int(code):02d}"
    return code


def normalize_row(value: Any) -> str:
    return normalize_bay(value)


def parse_enable_size_flags(value: Any) -> tuple[bool, bool]:
    if value is None or pd.isna(value):
        return True, True
    tokens = re.findall(r"\d+", str(value))
    if not tokens:
        return True, True
    sizes = {normalize_size_large(token) for token in tokens}
    return "20" in sizes, "40" in sizes


def enabled_size_modes(value: Any) -> set[str]:
    text = normalize_code(value)
    if not text:
        return set(SIZE_MODES)
    modes = set()
    for part in re.findall(r"\d+", text):
        if part in {"20", "40", "45"}:
            modes.add(part)
    return modes


def size_enabled_mask(values: pd.Series, size_mode: str) -> pd.Series:
    text = values.astype("string")
    stripped = text.str.strip()
    missing = values.isna() | stripped.str.lower().isin(["", "nan", "none", "<na>"]).fillna(False)
    enabled = stripped.str.contains(rf"(?<!\d){re.escape(size_mode)}(?!\d)", regex=True, na=False)
    return missing | enabled


def find_flat_file(data_dir: Path, exact: str | None = None, pattern: str | None = None) -> Path:
    if exact:
        path = data_dir / exact
        if path.exists():
            return path
    if pattern:
        matches = sorted(path for path in data_dir.glob(pattern) if path.is_file() and not path.name.startswith("~$"))
        if matches:
            return matches[0]
    raise FileNotFoundError(exact or pattern or str(data_dir))


def find_area_function_file(data_dir: Path) -> Path:
    return find_flat_file(data_dir, pattern="*功能*.xlsx")


def find_distance_matrix_file(data_dir: Path) -> Path:
    preferred = sorted(data_dir.glob("*距离矩阵*.xlsx"))
    if preferred:
        return preferred[0]
    for path in sorted(data_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            continue
        if any(_sheet_has_area_and_berths(path, sheet) for sheet in xls.sheet_names):
            return path
    raise FileNotFoundError(f"No berth-area distance matrix workbook found under {data_dir}")


def _sheet_has_area_and_berths(path: Path, sheet: str) -> bool:
    try:
        frame = pd.read_excel(path, sheet_name=sheet, nrows=2)
    except Exception:
        return False
    columns = {str(column) for column in frame.columns}
    return "area_no" in columns and any(re.fullmatch(r"B\d+", column) for column in columns)


def read_vessel_info(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(find_flat_file(data_dir, exact="vessel_berth_info_new.csv"))
    frame = frame.copy()
    frame["voy_id"] = frame["VOY_ID"].map(normalize_voyage)
    frame["ie_flag"] = frame["VOY_IEFG"].map(normalize_code)
    frame["berth_no"] = frame.get("VBT_BTH_ABTHNO", pd.Series(index=frame.index)).map(normalize_code)
    fallback_berth = frame.get("VBT_BTH_PBTHNO", pd.Series(index=frame.index)).map(normalize_code)
    frame["berth_no"] = frame["berth_no"].where(frame["berth_no"].ne(""), fallback_berth)
    frame["berth_key"] = frame["berth_no"].map(lambda value: f"B{value}" if value and not str(value).startswith("B") else value)
    for column in ["SCD_RCVSTDT", "SCD_RCVEDDT", "VBT_ABTHDT", "VBT_PBTHDT", "VBT_ADPTDT", "VBT_PDPTDT"]:
        if column in frame.columns:
            frame[column] = frame[column].map(parse_datetime)
    return frame


def read_berths_for_vessels(vessel_info: pd.DataFrame, vessels: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for vessel in vessels:
        rows = vessel_info[vessel_info["voy_id"].eq(vessel)]
        if rows.empty:
            continue
        berth = normalize_code(rows.iloc[0].get("berth_key"))
        if berth:
            result[vessel] = berth
    return result


def read_vessel_schedules(data_dir: Path) -> dict[str, VoyageSchedule]:
    frame = read_vessel_info(data_dir)
    schedules: dict[str, VoyageSchedule] = {}
    for row in frame.to_dict("records"):
        if row.get("ie_flag") != "E":
            continue
        voyage_id = normalize_voyage(row.get("voy_id"))
        receive_start = row.get("SCD_RCVSTDT")
        receive_end = row.get("SCD_RCVEDDT")
        berth_time = row.get("VBT_ABTHDT") or row.get("VBT_PBTHDT")
        departure_time = row.get("VBT_ADPTDT") or row.get("VBT_PDPTDT")
        berth_no = normalize_code(row.get("berth_no"))
        if not (voyage_id and receive_start and receive_end and berth_time and departure_time):
            continue
        schedules[voyage_id] = VoyageSchedule(
            voyage_id=voyage_id,
            receive_start=receive_start,
            receive_end=receive_end,
            berth_no=berth_no,
            berth_time=berth_time,
            departure_time=departure_time,
        )
    return schedules


def read_target_vessel_schedules(
    data_dir: Path,
    target_voyages: list[str],
    planning_time: datetime,
    horizon_hours: float,
) -> dict[str, VoyageSchedule]:
    schedules = read_vessel_schedules(data_dir)
    frame = read_vessel_info(data_dir)
    rows_by_voyage = {
        normalize_voyage(row.get("voy_id")): row
        for row in frame.to_dict("records")
        if row.get("ie_flag") == "E"
    }
    for voyage_id in target_voyages:
        if voyage_id in schedules:
            continue
        row = rows_by_voyage.get(voyage_id, {})
        berth_no = normalize_code(row.get("berth_no"))
        schedules[voyage_id] = VoyageSchedule(
            voyage_id=voyage_id,
            receive_start=planning_time,
            receive_end=planning_time + timedelta(hours=horizon_hours),
            berth_no=berth_no,
            berth_time=planning_time + timedelta(hours=horizon_hours),
            departure_time=planning_time + timedelta(hours=horizon_hours + 12),
        )
    return schedules


def read_area_functions_large(data_dir: Path) -> tuple[list[str], dict[str, set[str]], dict[str, float]]:
    frame = pd.read_excel(find_area_function_file(data_dir))
    area_col = _first_existing(set(frame.columns), ["area_no", "AREA_NO", "YAA_AREANO"])
    type_col = _first_existing(set(frame.columns), ["cntr_type", "CNTR_TYPE", "function", "FUNCTION"])
    load_col = _first_existing(set(frame.columns), ["load_capacity", "H", "capacity", "作业能力"])
    if not area_col or not type_col:
        raise KeyError("Area function workbook must contain area_no and cntr_type columns.")
    area_functions: dict[str, set[str]] = {}
    load_capacity: dict[str, float] = {}
    for row in frame.to_dict("records"):
        area = normalize_code(row.get(area_col))
        if not area:
            continue
        funcs = {normalize_code(part) for part in str(row.get(type_col, "")).split(",") if normalize_code(part)}
        area_functions[area] = funcs
        value = pd.to_numeric(row.get(load_col), errors="coerce") if load_col else pd.NA
        load_capacity[area] = float(value) if pd.notna(value) else 999999.0
    return sorted(area_functions), area_functions, load_capacity


def read_area_functions(data_dir: Path) -> dict[str, set[str]]:
    _, area_functions, _ = read_area_functions_large(data_dir)
    return area_functions


def read_distance_matrix(
    data_dir: Path,
    areas: Sequence[str] | None = None,
    berth_by_vessel: Mapping[str, str] | None = None,
) -> dict[tuple[str, str], float]:
    path = find_distance_matrix_file(data_dir)
    xls = pd.ExcelFile(path)
    frame = None
    for sheet in xls.sheet_names:
        trial = pd.read_excel(path, sheet_name=sheet)
        if "area_no" in trial.columns and any(str(column).upper().startswith("B") for column in trial.columns):
            frame = trial
            break
    if frame is None:
        raise KeyError(f"No distance matrix sheet with area_no and berth columns in {path}")
    area_filter = set(areas or [])
    berth_columns = [column for column in frame.columns if str(column).upper().startswith("B")]
    berth_keys = {normalize_code(column): column for column in berth_columns}
    if berth_by_vessel:
        matrix = frame.copy()
        matrix["area_no"] = matrix["area_no"].map(normalize_code)
        matrix = matrix.dropna(subset=["area_no"]).set_index("area_no")
        distances: dict[tuple[str, str], float] = {}
        for vessel, berth in berth_by_vessel.items():
            berth_key = normalize_code(berth)
            column = berth_keys.get(berth_key)
            if column is None:
                raise KeyError(f"Distance matrix does not contain berth column {berth_key}.")
            for area in areas or list(matrix.index):
                if area not in matrix.index:
                    raise KeyError(f"Distance matrix does not contain area {area}.")
                distances[(vessel, area)] = float(matrix.loc[area, column])
        return distances

    distances: dict[tuple[str, str], float] = {}
    for row in frame.to_dict("records"):
        area = normalize_code(row.get("area_no"))
        if not area or (area_filter and area not in area_filter):
            continue
        for berth in berth_columns:
            berth_key = normalize_code(berth)
            value = row.get(berth)
            if pd.notna(value):
                distances[(area, berth_key)] = float(value)
    return distances


def read_snapshot(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(find_flat_file(data_dir, exact="bay_slots_detail.parquet"))


def extract_current_snapshot_rows(
    snapshot: pd.DataFrame,
    export_vessels: Sequence[str],
    import_vessels: Sequence[str],
    flow_aliases: Mapping[str, str],
) -> pd.DataFrame:
    occupied = snapshot[snapshot["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()
    if occupied.empty:
        return occupied
    occupied["e_voy"] = occupied["IYC_EVOY_ID"].map(normalize_voyage)
    occupied["i_voy"] = occupied["IYC_IVOY_ID"].map(normalize_voyage)
    export_set = set(export_vessels)
    import_set = set(import_vessels)
    rows = occupied[occupied["e_voy"].isin(export_set) | occupied["i_voy"].isin(import_set)].copy()
    rows["voy_id"] = rows["e_voy"].where(rows["e_voy"].isin(export_set), rows["i_voy"])
    rows["area_no"] = rows["YAA_AREANO"].map(normalize_code)
    rows["bay_no"] = rows["YBY_BAYNO"].map(normalize_bay)
    rows["cntr_id"] = rows["IYC_CNTRID"].map(normalize_code)
    rows["size"] = rows["IYC_CSZ_CSIZECD"].map(normalize_size_large)
    rows["flow"] = rows["IYC_STS_CSTATUSCD"].map(lambda value: normalize_flow(value, flow_aliases))
    return rows


def build_bay_total_slot_counts(snapshot: pd.DataFrame, areas: Sequence[str]) -> dict[tuple[str, str], int]:
    work = snapshot[snapshot["YAA_AREANO"].map(normalize_code).isin(set(areas))].copy()
    work["area_no"] = work["YAA_AREANO"].map(normalize_code)
    work["bay_no"] = work["YBY_BAYNO"].map(normalize_bay)
    counts = work.groupby(["area_no", "bay_no"], sort=False).size()
    return {(str(area), str(bay)): int(count) for (area, bay), count in counts.items()}


def identify_bad_bays(
    current_snapshot: pd.DataFrame,
    area_functions: Mapping[str, set[str]],
    bay_total_slots: Mapping[tuple[str, str], int],
    ratio: float = 2.0 / 3.0,
) -> set[tuple[str, str]]:
    if current_snapshot.empty:
        return set()
    bad_counts: Counter[tuple[str, str]] = Counter()
    for row in current_snapshot.to_dict("records"):
        area = normalize_code(row.get("area_no"))
        bay = normalize_bay(row.get("bay_no"))
        flow = normalize_code(row.get("flow"))
        if area and bay and flow and flow not in area_functions.get(area, set()):
            bad_counts[(area, bay)] += 1
    return {key for key, count in bad_counts.items() if count > bay_total_slots.get(key, 0) * ratio}


def build_snapshot_count_params(
    current_snapshot: pd.DataFrame,
    area_functions: Mapping[str, set[str]],
    areas: set[str],
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], float],
]:
    l20: dict[tuple[str, str, str], float] = {}
    l40: dict[tuple[str, str, str], float] = {}
    q20: dict[tuple[str, str, str], float] = {}
    q40: dict[tuple[str, str, str], float] = {}
    if current_snapshot.empty:
        return l20, l40, q20, q40
    unique_containers = current_snapshot.copy()
    unique_containers = unique_containers[unique_containers["cntr_id"].notna()].copy()
    unique_containers = unique_containers.sort_values(["cntr_id", "area_no", "bay_no"]).drop_duplicates(
        "cntr_id",
        keep="first",
    )
    unique_containers = unique_containers[unique_containers["cntr_id"].astype(str).ne("-1")].copy()
    grouped = unique_containers.groupby(["voy_id", "flow", "area_no", "size"], dropna=False).size()
    for (vessel, flow, area, size), qty in grouped.items():
        if not vessel or not flow or not area or area not in areas:
            continue
        target = (l20 if size == "20" else l40) if flow in area_functions.get(area, set()) else (q20 if size == "20" else q40)
        target[(str(vessel), str(flow), str(area))] = target.get((str(vessel), str(flow), str(area)), 0.0) + float(qty)
    return l20, l40, q20, q40


def prepare_slot_frame(snapshot: pd.DataFrame, areas: Sequence[str], bad_bays: set[tuple[str, str]]) -> pd.DataFrame:
    work = snapshot.copy()
    work["area_no"] = work["YAA_AREANO"].map(normalize_code)
    work["bay_no"] = work["YBY_BAYNO"].map(normalize_bay)
    work = work[work["area_no"].isin(set(areas))].copy()
    if bad_bays:
        mask = [(area, bay) not in bad_bays for area, bay in zip(work["area_no"], work["bay_no"])]
        work = work.loc[mask].copy()
    work = work[work["HAS_CONTAINER"].fillna(0).astype(int).eq(0)].copy()
    parsed = work["YBY_ENABLECSIZECD"].map(parse_enable_size_flags)
    work["enable_20"] = parsed.map(lambda item: item[0])
    work["enable_40"] = parsed.map(lambda item: item[1])
    work["slot_uid"] = (
        work["area_no"].astype(str)
        + "|"
        + work["bay_no"].astype(str)
        + "|"
        + work["YST_ROWNO"].map(normalize_row).astype(str)
        + "|"
        + work["YST_TIERNO"].map(normalize_row).astype(str)
        + "|"
        + work["YST_SLOTNO"].map(normalize_row).astype(str)
    )
    return work


def count_slots_by_area(slots: pd.DataFrame, areas: Sequence[str]) -> dict[str, float]:
    counts = slots.groupby("area_no").size() if not slots.empty else pd.Series(dtype=int)
    return {area: float(counts.get(area, 0)) for area in areas}


def normalize_container_frame(df: pd.DataFrame, flow_aliases: Mapping[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["cntr_id"] = df["IYC_CNTRID"].map(normalize_code)
    df["e_voy"] = df["IYC_EVOY_ID"].map(normalize_voyage)
    df["i_voy"] = df["IYC_IVOY_ID"].map(normalize_voyage)
    df["size"] = df["IYC_CSZ_CSIZECD"].map(normalize_size_large)
    df["raw_flow"] = df["IYC_STS_CSTATUSCD"].map(normalize_code)
    df["flow"] = df["raw_flow"].map(lambda value: normalize_flow(value, flow_aliases))
    return df


def merge_snapshot_and_doc(doc: pd.DataFrame, snapshot_rows: pd.DataFrame) -> pd.DataFrame:
    doc_part = doc[["cntr_id", "flow", "size"]].copy()
    doc_part["source_rank"] = 1
    if snapshot_rows.empty:
        snap_part = doc_part.iloc[0:0].copy()
    else:
        snap_part = snapshot_rows[["cntr_id", "flow", "size"]].copy()
        snap_part["source_rank"] = 0
    merged = pd.concat([snap_part, doc_part], ignore_index=True)
    merged = merged[merged["cntr_id"].notna() & merged["flow"].notna() & merged["size"].notna()].copy()
    return merged.sort_values("source_rank").drop_duplicates("cntr_id", keep="first")


def add_grouped_demand(
    rows: pd.DataFrame,
    vessel: str,
    d20: dict[tuple[str, str], float],
    d40: dict[tuple[str, str], float],
) -> None:
    if rows.empty:
        return
    grouped = rows.groupby(["flow", "size"]).size()
    for (flow, size), qty in grouped.items():
        target = d20 if size == "20" else d40
        key = (vessel, str(flow))
        target[key] = target.get(key, 0.0) + float(qty)


def read_prediction_counts(path: Path) -> tuple[float, float]:
    xls = pd.ExcelFile(path)
    frame = pd.read_excel(path, sheet_name=xls.sheet_names[0])
    total20 = 0.0
    total40 = 0.0
    for row in frame.to_dict("records"):
        size = normalize_size_large(row.get("IYC_CSZ_CSIZECD"))
        count = pd.to_numeric(row.get("count"), errors="coerce")
        count = float(count) if pd.notna(count) else 0.0
        if size == "20":
            total20 += count
        elif size == "40":
            total40 += count
    return total20, total40


def read_prediction_work_lanes(path: Path) -> float:
    xls = pd.ExcelFile(path)
    sheet_name = next((name for name in xls.sheet_names if "作业" in str(name) or "璺" in str(name)), xls.sheet_names[-1])
    frame = pd.read_excel(path, sheet_name=sheet_name)
    candidates: list[float] = []

    def collect(value: Any) -> None:
        if value is None or pd.isna(value):
            return
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric) and float(numeric) > 0:
            candidates.append(float(numeric))

    for column in frame.columns:
        collect(column)
    for value in frame.to_numpy().ravel():
        collect(value)
    if not candidates:
        raise ValueError(f"No positive work-lane count in {path}")
    return candidates[0]


def build_demand_params(
    data_dir: Path,
    export_vessels: Sequence[str],
    import_vessels: Sequence[str],
    current_snapshot: pd.DataFrame,
    flow_aliases: Mapping[str, str],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, dict[str, Any]]]:
    d20: dict[tuple[str, str], float] = {}
    d40: dict[tuple[str, str], float] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for vessel in export_vessels:
        doc_path = find_flat_file(data_dir, exact=f"container_info_{vessel}.parquet")
        doc = normalize_container_frame(pd.read_parquet(doc_path), flow_aliases)
        doc = doc[doc["e_voy"].eq(vessel)].copy()
        snap = current_snapshot[current_snapshot["voy_id"].eq(vessel)].copy()
        merged = merge_snapshot_and_doc(doc, snap)
        detail20 = float((merged["size"] == "20").sum())
        detail40 = float((merged["size"] == "40").sum())
        add_grouped_demand(merged, vessel, d20, d40)
        pred20, pred40 = read_prediction_counts(find_flat_file(data_dir, exact=f"predict_data_{vessel}.xlsx"))
        extra20 = max(0.0, pred20 - detail20)
        extra40 = max(0.0, pred40 - detail40)
        if extra20 > 0:
            d20[(vessel, "OF")] = d20.get((vessel, "OF"), 0.0) + extra20
        if extra40 > 0:
            d40[(vessel, "OF")] = d40.get((vessel, "OF"), 0.0) + extra40
        diagnostics[vessel] = {
            "type": "export",
            "doc_rows": int(len(doc)),
            "snapshot_rows": int(len(snap)),
            "dedup_rows": int(len(merged)),
            "prediction20": float(pred20),
            "prediction40": float(pred40),
            "extra_prediction20_to_OF": float(extra20),
            "extra_prediction40_to_OF": float(extra40),
        }
    for vessel in import_vessels:
        doc_path = find_flat_file(data_dir, exact=f"container_info_import_voy_{vessel}.parquet")
        doc = normalize_container_frame(pd.read_parquet(doc_path), flow_aliases)
        doc = doc[doc["i_voy"].eq(vessel)].copy()
        snap = current_snapshot[current_snapshot["voy_id"].eq(vessel)].copy()
        merged = merge_snapshot_and_doc(doc, snap)
        add_grouped_demand(merged, vessel, d20, d40)
        diagnostics[vessel] = {
            "type": "import",
            "doc_rows": int(len(doc)),
            "snapshot_rows": int(len(snap)),
            "dedup_rows": int(len(merged)),
        }
    return d20, d40, diagnostics


def parse_tops_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def parse_tops_area_bay(value: Any) -> tuple[str, str]:
    code = normalize_code(value).replace(".0", "")
    if not code:
        return "", ""
    if len(code) < 4:
        code = code.zfill(4)
    return code[:2], normalize_bay(code[-2:])


def bay_code_value(value: Any) -> int | None:
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


def slot_range_mask(values: pd.Series, start_value: Any, end_value: Any, value_parser: Any) -> pd.Series:
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


def bay_range_mask(values: pd.Series, start_bay: str, end_bay: str) -> pd.Series:
    return slot_range_mask(values, start_bay, end_bay, bay_code_value)


def row_range_mask(values: pd.Series, start_row: str, end_row: str) -> pd.Series:
    return slot_range_mask(values, start_row, end_row, bay_code_value)


def count_tops_blocked_slots(tops_rows: pd.DataFrame, slots: pd.DataFrame, vessel: str) -> dict[tuple[str, str], float]:
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
        matched = sub[bay_range_mask(sub["bay_no"], start_bay, end_bay)]
        if matched.empty:
            continue
        blocked_by_area.setdefault(area, set()).update(matched["slot_uid"].tolist())
    return {(vessel, area): float(len(uids)) for area, uids in blocked_by_area.items()}


def active_tops_rows(data_dir: Path, planning_time: datetime) -> pd.DataFrame:
    tops = pd.read_parquet(find_flat_file(data_dir, exact="tops_plan_info.parquet")).copy()
    tops["condition_vessel"] = tops["SPL_CONDITIONCODE"].map(normalize_voyage)
    tops["start_time"] = parse_tops_time(tops["SPL_STDATE"])
    tops["end_time"] = parse_tops_time(tops["SPL_EDDATE"])
    if "SPL_ISVALID" in tops.columns:
        tops = tops[tops["SPL_ISVALID"].astype(str).str.upper().eq("Y")].copy()
    if "SPR_ISVALID" in tops.columns:
        tops = tops[tops["SPR_ISVALID"].astype(str).str.upper().eq("Y")].copy()
    return tops[(tops["start_time"] <= planning_time) & (planning_time <= tops["end_time"])].copy()


def compute_tops_capacity_deductions(
    data_dir: Path,
    planning_time: datetime,
    vessels: Sequence[str],
    bay20_equiv: pd.DataFrame,
    bay20_direct: pd.DataFrame,
    bay40: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], dict[tuple[str, str], float], int]:
    active = active_tops_rows(data_dir, planning_time)
    tops20: dict[tuple[str, str], float] = {}
    tops20_direct: dict[tuple[str, str], float] = {}
    tops40: dict[tuple[str, str], float] = {}
    for vessel in vessels:
        relevant = active[active["condition_vessel"] != vessel].copy()
        tops20.update(count_tops_blocked_slots(relevant, bay20_equiv, vessel))
        tops20_direct.update(count_tops_blocked_slots(relevant, bay20_direct, vessel))
        tops40.update(count_tops_blocked_slots(relevant, bay40, vessel))
    return tops20, tops20_direct, tops40, int(len(active))


def build_availability_flags(
    vessels: Sequence[str],
    flows: Sequence[str],
    areas: Sequence[str],
    area_functions: Mapping[str, set[str]],
    cbar20: Mapping[tuple[str, str], float],
    cbar20_direct: Mapping[tuple[str, str], float],
    cbar40: Mapping[tuple[str, str], float],
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
    e20: dict[tuple[str, str, str], int] = {}
    e40: dict[tuple[str, str, str], int] = {}
    for vessel in vessels:
        for flow in flows:
            for area in areas:
                func_ok = flow in area_functions.get(area, set())
                e20[(vessel, flow, area)] = int(
                    func_ok and cbar20_direct.get((vessel, area), 0.0) > 0 and cbar20.get((vessel, area), 0.0) > 0
                )
                e40[(vessel, flow, area)] = int(
                    func_ok and cbar40.get((vessel, area), 0.0) > 0 and cbar20.get((vessel, area), 0.0) >= 2
                )
    return e20, e40


def build_large_inputs(
    data_dir: Path,
    state_dir: Path,
    planning_time: pd.Timestamp,
    export_vessels: Sequence[str] = DEFAULT_EXPORT_VESSELS,
    import_vessels: Sequence[str] = DEFAULT_IMPORT_VESSELS,
    disable_default_flow_aliases: bool = False,
) -> tuple[PlanningInputArtifacts, RollingPlanningState]:
    data_dir = data_dir.resolve()
    flow_aliases = {} if disable_default_flow_aliases else DEFAULT_FLOW_ALIASES
    export_vessels = [normalize_voyage(v) for v in export_vessels if normalize_voyage(v)]
    import_vessels = [normalize_voyage(v) for v in import_vessels if normalize_voyage(v)]
    all_vessels = export_vessels + import_vessels
    state = RollingPlanningState(state_dir)

    vessel_info = read_vessel_info(data_dir)
    areas, area_functions, load_capacity = read_area_functions_large(data_dir)
    berth_by_vessel = read_berths_for_vessels(vessel_info, all_vessels)
    distance = read_distance_matrix(data_dir, areas, berth_by_vessel)

    snapshot = read_snapshot(data_dir)
    current_snapshot = extract_current_snapshot_rows(snapshot, export_vessels, import_vessels, flow_aliases)
    bay_total_slots = build_bay_total_slot_counts(snapshot, areas)
    bad_bays = identify_bad_bays(current_snapshot, area_functions, bay_total_slots)
    l20, l40, q20, q40 = build_snapshot_count_params(current_snapshot, area_functions, set(areas))

    available_slots = prepare_slot_frame(snapshot, areas, bad_bays)
    bay20_equiv = available_slots
    bay20_direct = available_slots[available_slots["enable_20"]].copy()
    bay40 = available_slots[available_slots["enable_40"]].copy()
    c20 = count_slots_by_area(bay20_equiv, areas)
    c20_direct = count_slots_by_area(bay20_direct, areas)
    c40 = count_slots_by_area(bay40, areas)

    tops20, tops20_direct, tops40, active_tops_count = compute_tops_capacity_deductions(
        data_dir,
        planning_time.to_pydatetime(),
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

    d20, d40, demand_diagnostics = build_demand_params(
        data_dir,
        export_vessels,
        import_vessels,
        current_snapshot,
        flow_aliases,
    )
    of_work_lanes = {
        vessel: read_prediction_work_lanes(find_flat_file(data_dir, exact=f"predict_data_{vessel}.xlsx"))
        for vessel in export_vessels
    }
    of_work_lanes.update({vessel: 0.0 for vessel in import_vessels})
    flows = sorted(
        {flow for funcs in area_functions.values() for flow in funcs}
        | {flow for _, flow in d20}
        | {flow for _, flow in d40}
        | {flow for _, flow, _ in l20}
        | {flow for _, flow, _ in l40}
        | {flow for _, flow, _ in q20}
        | {flow for _, flow, _ in q40}
    )
    u = {(a, f): int(f in area_functions.get(a, set())) for a in areas for f in flows}
    e20, e40 = build_availability_flags(all_vessels, flows, areas, area_functions, cbar20, cbar20_direct, cbar40)
    p20, p40, old_flags, previous_rows = state.build_previous_plan_params(planning_time, all_vessels)
    model_data = LargePlanningData(
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
        H=load_capacity,
        distance=distance,
        U=u,
        E20=e20,
        E40=e40,
        P20=p20,
        P40=p40,
        O=old_flags,
        OFWorkLanes=of_work_lanes,
        weights=YardPlanningWeights(),
        allow_unmet_demand=True,
        strict_validation=True,
    )
    diagnostics = {
        "data_dir": str(data_dir),
        "area_count": len(areas),
        "flows": flows,
        "flow_aliases": flow_aliases,
        "bad_bay_count": len(bad_bays),
        "bad_bay_sample": sorted(list(bad_bays))[:20],
        "current_snapshot_rows": int(len(current_snapshot)),
        "active_tops_rows": int(active_tops_count),
        "capacity20_total": float(sum(c20.values())),
        "capacity20_direct_total": float(sum(c20_direct.values())),
        "capacity40_total": float(sum(c40.values())),
        "old_vessels": sorted([v for v, flag in old_flags.items() if flag]),
        "of_work_lanes": of_work_lanes,
        "of_area_limits": {vessel: 2.0 * lanes for vessel, lanes in of_work_lanes.items()},
        "demand": demand_diagnostics,
    }
    return (
        PlanningInputArtifacts(
            data=model_data,
            planning_time=planning_time,
            export_vessels=export_vessels,
            import_vessels=import_vessels,
            area_functions=area_functions,
            berth_by_vessel=berth_by_vessel,
            previous_plan_rows=previous_rows,
            diagnostics=diagnostics,
        ),
        state,
    )


def allocation_output_rows(solution: Any, data: LargePlanningData, include_zero: bool = False) -> list[dict[str, Any]]:
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
                }
            )
    return rows


def snapshot_quantity_for_output(data: LargePlanningData, size: str, key: tuple[str, str, str]) -> float:
    if size == "20":
        return float(data.S20.get(key, data.L20.get(key, 0.0) + data.Q20.get(key, 0.0)))
    return float(data.S40.get(key, data.L40.get(key, 0.0) + data.Q40.get(key, 0.0)))


def count_flow_function_mismatch_rows(allocation: pd.DataFrame, area_functions: Mapping[str, set[str]], qty_column: str) -> int:
    if allocation.empty or qty_column not in allocation.columns:
        return 0
    rows = allocation[pd.to_numeric(allocation[qty_column], errors="coerce").fillna(0.0) > 1e-6].copy()
    if rows.empty:
        return 0
    return int(rows.apply(lambda row: row["flow"] not in area_functions.get(row["area_no"], set()), axis=1).sum())


def write_large_outputs(output_dir: Path, artifacts: PlanningInputArtifacts, solution: Any, state_rows: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allocation = pd.DataFrame(allocation_output_rows(solution, artifacts.data))
    allocation.to_csv(output_dir / "allocation.csv", index=False, encoding="utf-8-sig")
    allocation_new = allocation[pd.to_numeric(allocation.get("new_qty", 0), errors="coerce").fillna(0.0) > 0].copy()
    allocation_new.to_csv(output_dir / "allocation_new.csv", index=False, encoding="utf-8-sig")
    state_rows.to_csv(output_dir / "state_rows_appended.csv", index=False, encoding="utf-8-sig")
    diagnostics = {
        **artifacts.diagnostics,
        "status": solution.status,
        "status_name": solution.status_name,
        "objective_value": solution.objective_value,
        "best_bound": solution.best_bound,
        "mip_gap": solution.mip_gap,
        "runtime": solution.runtime,
        "objective_components": solution.objective_components,
        "allocation_rows": int(len(allocation)),
        "allocation_new_rows": int(len(allocation_new)),
        "flow_function_mismatch_total": count_flow_function_mismatch_rows(allocation, artifacts.area_functions, "planned_qty"),
        "flow_function_mismatch_new": count_flow_function_mismatch_rows(allocation, artifacts.area_functions, "new_qty"),
    }
    write_json(output_dir / "diagnostics.json", diagnostics)


def read_closed_areas(data_dir: Path) -> set[str]:
    path = data_dir / "n_usefg_areas.txt"
    if not path.exists():
        return set()
    return set(re.findall(r"[A-Za-z0-9]+", path.read_text(encoding="utf-8")))


def calculate_medium_demands(
    data_dir: Path,
    voyage_ids: list[str] | tuple[str, ...] = DEFAULT_TARGET_VOYAGES,
    planning_time: datetime | None = None,
) -> list[DemandRow]:
    planning_time = planning_time or parse_datetime(DEFAULT_PLANNING_TIME) or datetime(2026, 5, 8, 9, 30)
    schedules = read_vessel_schedules(data_dir)
    rows: list[DemandRow] = []
    for voyage_id in [normalize_voyage(v) for v in voyage_ids]:
        receive_start = schedules.get(voyage_id).receive_start if voyage_id in schedules else planning_time
        stage, ratio = planning_stage(receive_start, planning_time)
        predicted = read_predicted_by_port_size(data_dir, voyage_id)
        ratio_targets = ratio_targets_by_port(predicted, ratio)
        docs = read_doc_by_port_size(data_dir, voyage_id)
        planned_source = choose_planned_source(ratio_targets, docs)
        for flow, size_mode, port in sorted(planned_source):
            planned = planned_source[(flow, size_mode, port)]
            if planned <= 0:
                continue
            rows.append(
                DemandRow(
                    voyage_id=voyage_id,
                    flow=flow,
                    port=port,
                    size_mode=size_mode,
                    predicted_boxes=predicted.get((flow, size_mode, port), 0),
                    ratio_target_boxes=ratio_targets.get((flow, size_mode, port), 0),
                    doc_boxes=docs.get((flow, size_mode, port), 0),
                    planned_boxes=planned,
                    planning_stage=stage,
                    planning_ratio=ratio,
                )
            )
    return rows


def planning_stage(receive_start: datetime, planning_time: datetime) -> tuple[str, float]:
    if planning_time < receive_start:
        if planning_time + timedelta(hours=24) >= receive_start:
            return "before_open_within_24h", 0.70
        return "before_open_beyond_24h", 0.0
    elapsed = planning_time - receive_start
    if elapsed < timedelta(hours=24):
        return "open_first_24h", 0.70
    if elapsed < timedelta(hours=48):
        return "open_second_24h", 0.20
    return "open_third_24h_or_later", 0.10


def read_predicted_by_port_size(data_dir: Path, voyage_id: str) -> Counter[tuple[str, str, str]]:
    path = find_flat_file(data_dir, exact=f"predict_data_{voyage_id}.xlsx")
    xls = pd.ExcelFile(path)
    sheet_name = next(
        (
            name
            for name in xls.sheet_names
            if "港口" in str(name)
        ),
        xls.sheet_names[0],
    )
    frame = pd.read_excel(path, sheet_name=sheet_name)
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in frame.to_dict("records"):
        size_mode = normalize_size_small(row.get("IYC_CSZ_CSIZECD"))
        port = normalize_text(row.get("IYC_POT_UNLDPORT"), "UNK")
        flow = normalize_flow(row.get("IYC_STS_CSTATUSCD") or row.get("flow") or row.get("cntr_type"), default="OF")
        counter[(flow, size_mode, port)] += int(round(float(row.get("count", 0) or 0)))
    return counter


def ratio_targets_by_port(predicted: Counter[tuple[str, str, str]], ratio: float) -> Counter[tuple[str, str, str]]:
    targets: Counter[tuple[str, str, str]] = Counter()
    by_flow_size: defaultdict[tuple[str, str], list[tuple[tuple[str, str, str], int]]] = defaultdict(list)
    for key, count in predicted.items():
        by_flow_size[(key[0], key[1])].append((key, count))
    for items in by_flow_size.values():
        total = sum(count for _, count in items)
        target_total = int(round(total * ratio))
        scaled = largest_remainder_scale([count for _, count in items], total, target_total)
        for (key, _), qty in zip(items, scaled):
            if qty > 0:
                targets[key] += qty
    return targets


def read_doc_by_port_size(data_dir: Path, voyage_id: str) -> Counter[tuple[str, str, str]]:
    path = data_dir / f"container_info_{voyage_id}.parquet"
    counter: Counter[tuple[str, str, str]] = Counter()
    if not path.exists():
        return counter
    frame = pd.read_parquet(path)
    if frame.empty:
        return counter
    work = pd.DataFrame(
        {
            "flow": frame.get("IYC_STS_CSTATUSCD", pd.Series(index=frame.index, dtype=object)).map(
                lambda value: normalize_flow(value, default="OF")
            ),
            "size": frame.get("IYC_CSZ_CSIZECD", pd.Series(index=frame.index, dtype=object)).map(normalize_size_small),
            "port": frame.get("IYC_POT_UNLDPORT", pd.Series(index=frame.index, dtype=object)).map(
                lambda value: normalize_text(value, "UNK")
            ),
        }
    )
    counts = work.groupby(["flow", "size", "port"], sort=False).size()
    counter.update({(str(flow), str(size), str(port)): int(count) for (flow, size, port), count in counts.items()})
    return counter


def choose_planned_source(
    ratio_targets: Counter[tuple[str, str, str]],
    docs: Counter[tuple[str, str, str]],
) -> Counter[tuple[str, str, str]]:
    planned: Counter[tuple[str, str, str]] = Counter()
    flows = sorted({flow for flow, _, _ in ratio_targets} | {flow for flow, _, _ in docs})
    for flow in flows:
        for size_mode in SIZE_MODES:
            ratio_total = sum(qty for (f, size, _), qty in ratio_targets.items() if f == flow and size == size_mode)
            doc_total = sum(qty for (f, size, _), qty in docs.items() if f == flow and size == size_mode)
            source = docs if doc_total > ratio_total else ratio_targets
            for (f, size, port), qty in source.items():
                if f == flow and size == size_mode and qty > 0:
                    planned[(f, size, port)] += qty
    return planned


def largest_remainder_scale(counts: list[int], source_total: int, target_total: int) -> list[int]:
    if target_total <= 0:
        return [0 for _ in counts]
    if source_total <= 0:
        out = [0 for _ in counts]
        if out:
            out[0] = target_total
        return out
    raw = [count * target_total / source_total for count in counts]
    base = [int(value) for value in raw]
    remain = target_total - sum(base)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
    for idx in order[:remain]:
        base[idx] += 1
    return base


def write_demand_rows(path: Path, rows: list[DemandRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        fieldnames = [
            "voyage_id",
            "flow",
            "port",
            "size_mode",
            "predicted_boxes",
            "ratio_target_boxes",
            "doc_boxes",
            "planned_boxes",
            "planning_stage",
            "planning_ratio",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def read_big_plan(path: Path) -> list[BigPlanRow]:
    counter: Counter[tuple[str, str, str, str, str]] = Counter()
    rows: list[BigPlanRow] = []
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        fieldnames = set(reader.fieldnames or [])
        if {"voyage_id", "area_no"}.issubset(fieldnames) and (
            {"qty_20", "qty_40"}.issubset(fieldnames)
            or {"planned_20", "planned_40"}.issubset(fieldnames)
            or {"20", "40"}.issubset(fieldnames)
        ):
            qty20_field = _first_existing(fieldnames, ["qty_20", "planned_20", "20", "c20", "C20"])
            qty40_field = _first_existing(fieldnames, ["qty_40", "planned_40", "40", "c40", "C40"])
            qty45_field = _first_existing(fieldnames, ["qty_45", "planned_45", "45", "c45", "C45"])
            date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
            flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
            for row in reader:
                flow = normalize_flow(row.get(flow_field), default="OF") if flow_field else "OF"
                voyage_id = normalize_voyage(row.get("voyage_id"))
                area_no = normalize_code(row.get("area_no"))
                plan_date = date_key(normalize_text(row.get(date_field))) if date_field else ""
                for size_mode, field_name in (("20", qty20_field), ("40", qty40_field), ("45", qty45_field)):
                    if not field_name:
                        continue
                    boxes = int(round(float(row.get(field_name, 0) or 0)))
                    if boxes > 0:
                        counter[(voyage_id, flow, area_no, size_mode, plan_date)] += boxes
        elif {"voy_id", "area_no", "planned_qty"}.issubset(fieldnames):
            date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
            flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
            for row in reader:
                flow = normalize_flow(row.get(flow_field), default="OF") if flow_field else "OF"
                boxes = int(round(float(row["planned_qty"])))
                if boxes > 0:
                    counter[
                        (
                            normalize_voyage(row["voy_id"]),
                            flow,
                            normalize_code(row["area_no"]),
                            normalize_big_plan_size(row.get("size")),
                            date_key(normalize_text(row.get(date_field))) if date_field else "",
                        )
                    ] += boxes
        elif {"voyage_id", "area_no", "planned_boxes"}.issubset(fieldnames):
            size_field = "size_mode" if "size_mode" in fieldnames else "size"
            date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
            flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
            for row in reader:
                flow = normalize_flow(row.get(flow_field), default="OF") if flow_field else "OF"
                boxes = int(round(float(row["planned_boxes"])))
                if boxes > 0:
                    counter[
                        (
                            normalize_voyage(row["voyage_id"]),
                            flow,
                            normalize_code(row["area_no"]),
                            normalize_big_plan_size(row.get(size_field)),
                            date_key(normalize_text(row.get(date_field))) if date_field else "",
                        )
                    ] += boxes
        else:
            raise ValueError(f"Unsupported big plan columns: {sorted(fieldnames)}")
    rows = [
        BigPlanRow(voyage_id, flow, area_no, boxes, size_mode, plan_date)
        for (voyage_id, flow, area_no, size_mode, plan_date), boxes in sorted(counter.items())
        if boxes > 0
    ]
    if not rows:
        raise ValueError("big plan file contains no positive planned boxes")
    return rows


def load_port_demand_groups(data_dir: Path, voyage_ids: list[str], planning_time: datetime) -> tuple[list[BoxGroup], list[DemandRow]]:
    demand_rows = calculate_medium_demands(data_dir, voyage_ids, planning_time)
    groups: list[BoxGroup] = []
    group_index: defaultdict[str, int] = defaultdict(int)
    for row in demand_rows:
        group_index[row.voyage_id] += 1
        groups.append(
            BoxGroup(
                group_id=f"{row.voyage_id}_P{group_index[row.voyage_id]:03d}",
                voyage_id=row.voyage_id,
                size=row.size_mode,
                height="UNK",
                status=row.flow,
                port=row.port,
                operator="UNK",
                ctype="UNK",
                weight_class="UNK",
                reefer=False,
                dangerous=False,
                over_limit=False,
                special_codes=(),
                demand=row.planned_boxes,
            )
        )
    return groups, demand_rows


def load_small_doc_groups(data_dir: Path, voyage_ids: list[str]) -> list[SmallBoxGroup]:
    groups: list[SmallBoxGroup] = []
    for voyage_id in voyage_ids:
        path = data_dir / f"container_info_{voyage_id}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        work = pd.DataFrame(
            {
                "status": frame.get("IYC_STS_CSTATUSCD", pd.Series(index=frame.index, dtype=object)).map(
                    lambda value: normalize_flow(value, default="OF")
                ),
                "size": frame.get("IYC_CSZ_CSIZECD", pd.Series(index=frame.index, dtype=object)).map(normalize_size_small),
                "port": frame.get("IYC_POT_UNLDPORT", pd.Series(index=frame.index, dtype=object)).map(
                    lambda value: normalize_text(value, "UNK")
                ),
                "height": frame.get("IYC_CHEIGHTCD", pd.Series(index=frame.index, dtype=object)).map(
                    lambda value: normalize_text(value, "UNK")
                ),
                "weight": frame.get("IYC_CWEIGHT", pd.Series(index=frame.index, dtype=object)).map(weight_class),
                "special_code": frame.apply(explicit_special_stow_code, axis=1),
                "pre_stow": False,
            }
        )
        counter: Counter[tuple[str, str, str, str, str, str, bool]] = Counter()
        for row in work.to_dict("records"):
            counter[
                (
                    row["status"],
                    row["size"],
                    row["port"],
                    row["height"],
                    row["weight"],
                    row["special_code"],
                    bool(row["pre_stow"]),
                )
            ] += 1
        for index, ((status, size, port, height, weight, special_code, pre_stow), demand) in enumerate(
            sorted(counter.items()), start=1
        ):
            groups.append(
                SmallBoxGroup(
                    group_id=f"{voyage_id}_S{index:03d}",
                    voyage_id=voyage_id,
                    status=status,
                    port=port,
                    size=size,
                    height=height,
                    weight_class=weight,
                    demand=demand,
                    pre_stow=pre_stow,
                    special_stow=bool(special_code),
                    special_stow_code=special_code,
                )
            )
    return groups


def build_bays(
    data_dir: Path,
    allowed_areas: set[str],
    closed_areas: set[str],
    planning_time: datetime,
    vessel_schedules: dict[str, VoyageSchedule],
    target_voyages: set[str],
    area_functions: dict[str, set[str]],
    misplaced_bay_exclusion_ratio: float,
) -> tuple[dict[str, Bay], int, int, int]:
    frame = pd.read_parquet(find_flat_file(data_dir, exact="bay_slots_detail.parquet")).copy()
    frame["YAA_AREANO"] = frame["YAA_AREANO"].map(normalize_code)
    frame["YBY_BAYNO"] = frame["YBY_BAYNO"].map(normalize_bay)
    frame["YST_ROWNO"] = frame["YST_ROWNO"].map(normalize_row)
    frame = frame[frame["YAA_AREANO"].isin(allowed_areas) & ~frame["YAA_AREANO"].isin(closed_areas)].copy()

    original_bay_capacity = total_slots_by_bay(frame)
    reserved_slots, closed_bays = tops_reserved_slots(data_dir, frame, planning_time, target_voyages)
    frame = drop_reserved_slots(frame, reserved_slots)
    excluded_bays = misplaced_bays_to_exclude(
        frame,
        area_functions,
        vessel_schedules,
        planning_time,
        misplaced_bay_exclusion_ratio,
        original_bay_capacity,
        target_voyages,
    )
    frame = drop_bays(frame, excluded_bays)

    available = available_or_released_slots(frame, vessel_schedules, planning_time)
    cap_by_size = capacity_by_bay_size(available)
    physical_cap = physical_capacity_by_bay(available)
    row_cap_by_size = capacity_by_bay_row_size(available)
    row_physical_cap = physical_capacity_by_bay_row(available)
    existing_attrs = existing_bay_attributes(frame, vessel_schedules, planning_time)
    bays: dict[str, Bay] = {}
    by_area: defaultdict[str, list[str]] = defaultdict(list)
    for area_no, bay_no in sorted(physical_cap, key=lambda item: (item[0], bay_sort_key(item[1]))):
        bay_key = f"{area_no}|{bay_no}"
        by_area[area_no].append(bay_key)
    block_lookup: dict[str, tuple[str, tuple[str, ...], bool]] = {}
    for area_no, bay_keys in by_area.items():
        bay_nos = [key.split("|", 1)[1] for key in bay_keys]
        big_starts = {
            bay
            for bay in bay_nos
            if cap_by_size["40"].get((area_no, bay), 0) > 0 or cap_by_size["45"].get((area_no, bay), 0) > 0
        }
        for index, members, adjusted in make_yard_blocks(bay_nos, big_starts):
            block_id = f"{area_no}-B{index:02d}"
            member_keys = tuple(f"{area_no}|{bay_no}" for bay_no in members)
            for key in member_keys:
                block_lookup[key] = (block_id, member_keys, adjusted)
    for (area_no, bay_no), physical in physical_cap.items():
        bay_key = f"{area_no}|{bay_no}"
        block_id, block_members, adjusted = block_lookup.get(bay_key, ("", tuple(), False))
        attrs = existing_attrs.get((area_no, bay_no), {})
        bays[bay_key] = Bay(
            area_no=area_no,
            bay_no=bay_no,
            bay_key=bay_key,
            block_id=block_id,
            block_bays=block_members,
            block_bay_count=len(block_members),
            block_boundary_adjusted=adjusted,
            bay_order=bay_code_value(bay_no) or 0,
            cap_by_size={size: cap_by_size[size].get((area_no, bay_no), 0) for size in SIZE_MODES},
            physical_capacity=physical,
            row_cap_by_size={
                size: {
                    row_no: qty
                    for (a, b, row_no), qty in row_cap_by_size[size].items()
                    if a == area_no and b == bay_no
                }
                for size in SIZE_MODES
            },
            row_physical_capacity={
                row_no: qty for (a, b, row_no), qty in row_physical_cap.items() if a == area_no and b == bay_no
            },
            existing_size_modes=set(attrs.get("sizes", set())),
            existing_heights=set(attrs.get("heights", set())),
            existing_special_signatures=set(attrs.get("specials", set())),
            existing_ports=set(attrs.get("ports", set())),
        )
    return bays, len(reserved_slots), len(closed_bays), len(excluded_bays)


def tops_reserved_slots(
    data_dir: Path,
    frame: pd.DataFrame,
    planning_time: datetime,
    target_voyages: set[str],
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str]]]:
    active = active_tops_rows(data_dir, planning_time)
    active = active[~active["condition_vessel"].isin({normalize_voyage(v) for v in target_voyages})].copy()
    reserved: set[tuple[str, str, str]] = set()
    closed_bays: set[tuple[str, str]] = set()
    if active.empty or frame.empty:
        return reserved, closed_bays
    empty = frame[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(0)].copy()
    by_area = {area: sub for area, sub in empty.groupby("YAA_AREANO")}
    for _, tops in active.iterrows():
        start_area, start_bay = parse_tops_area_bay(tops.get("SPR_STBAY"))
        end_area, end_bay = parse_tops_area_bay(tops.get("SPR_EDBAY"))
        area = start_area or end_area
        if start_area and end_area and start_area != end_area:
            area = end_area
        if not area or area not in by_area:
            continue
        sub = by_area[area]
        matched = sub[bay_range_mask(sub["YBY_BAYNO"], start_bay, end_bay)].copy()
        if matched.empty:
            continue
        start_row = normalize_row(tops.get("SPR_STROW"))
        end_row = normalize_row(tops.get("SPR_EDROW"))
        if start_row or end_row:
            matched = matched[row_range_mask(matched["YST_ROWNO"], start_row, end_row)]
        for row in matched.to_dict("records"):
            reserved.add((row["YAA_AREANO"], row["YBY_BAYNO"], row["YST_ROWNO"]))
            closed_bays.add((row["YAA_AREANO"], row["YBY_BAYNO"]))
    return reserved, closed_bays


def drop_reserved_slots(frame: pd.DataFrame, reserved_slots: set[tuple[str, str, str]]) -> pd.DataFrame:
    if not reserved_slots or frame.empty:
        return frame
    empty = frame["HAS_CONTAINER"].fillna(0).astype(int).eq(0)
    keys = list(zip(frame["YAA_AREANO"], frame["YBY_BAYNO"], frame["YST_ROWNO"]))
    mask = [not (is_empty and key in reserved_slots) for is_empty, key in zip(empty, keys)]
    return frame.loc[mask].copy()


def drop_bays(frame: pd.DataFrame, excluded_bays: set[tuple[str, str]]) -> pd.DataFrame:
    if not excluded_bays or frame.empty:
        return frame
    keys = list(zip(frame["YAA_AREANO"], frame["YBY_BAYNO"]))
    return frame.loc[[key not in excluded_bays for key in keys]].copy()


def active_occupied(frame: pd.DataFrame, vessel_schedules: dict[str, VoyageSchedule], planning_time: datetime) -> pd.DataFrame:
    occupied = frame[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()
    if occupied.empty:
        return occupied
    occupied["_voyage"] = occupied["IYC_EVOY_ID"].map(normalize_voyage)
    keep = []
    for row in occupied.to_dict("records"):
        voyage = row.get("_voyage")
        schedule = vessel_schedules.get(voyage)
        keep.append(schedule is None or schedule.departure_time > planning_time)
    return occupied.loc[keep].copy()


def available_or_released_slots(
    frame: pd.DataFrame,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> pd.DataFrame:
    empty = frame[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(0)].copy()
    occupied = frame[frame["HAS_CONTAINER"].fillna(0).astype(int).eq(1)].copy()
    if occupied.empty:
        return empty
    occupied["_voyage"] = occupied["IYC_EVOY_ID"].map(normalize_voyage)
    released = []
    for row in occupied.to_dict("records"):
        schedule = vessel_schedules.get(row.get("_voyage"))
        released.append(schedule is not None and schedule.departure_time <= planning_time)
    return pd.concat([empty, occupied.loc[released]], ignore_index=True)


def capacity_by_bay_size(base: pd.DataFrame) -> dict[str, dict[tuple[str, str], int]]:
    out: dict[str, dict[tuple[str, str], int]] = {}
    for size_mode in SIZE_MODES:
        sub = base[size_enabled_mask(base["YBY_ENABLECSIZECD"], size_mode)].copy()
        counts = sub.groupby(["YAA_AREANO", "YBY_BAYNO"]).size() if not sub.empty else pd.Series(dtype=int)
        out[size_mode] = {(str(a), str(b)): int(v) for (a, b), v in counts.items()}
    return out


def physical_capacity_by_bay(base: pd.DataFrame) -> dict[tuple[str, str], int]:
    counts = base.groupby(["YAA_AREANO", "YBY_BAYNO"]).size() if not base.empty else pd.Series(dtype=int)
    return {(str(a), str(b)): int(v) for (a, b), v in counts.items()}


def capacity_by_bay_row_size(base: pd.DataFrame) -> dict[str, dict[tuple[str, str, str], int]]:
    out: dict[str, dict[tuple[str, str, str], int]] = {}
    for size_mode in SIZE_MODES:
        sub = base[size_enabled_mask(base["YBY_ENABLECSIZECD"], size_mode)].copy()
        counts = sub.groupby(["YAA_AREANO", "YBY_BAYNO", "YST_ROWNO"]).size() if not sub.empty else pd.Series(dtype=int)
        out[size_mode] = {(str(a), str(b), str(r)): int(v) for (a, b, r), v in counts.items()}
    return out


def physical_capacity_by_bay_row(base: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    counts = base.groupby(["YAA_AREANO", "YBY_BAYNO", "YST_ROWNO"]).size() if not base.empty else pd.Series(dtype=int)
    return {(str(a), str(b), str(r)): int(v) for (a, b, r), v in counts.items()}


def total_slots_by_bay(base: pd.DataFrame) -> Counter[tuple[str, str]]:
    slots: Counter[tuple[str, str]] = Counter()
    if base.empty:
        return slots
    counts = base.groupby(["YAA_AREANO", "YBY_BAYNO"], sort=False).size()
    slots.update({(str(a), str(b)): int(v) for (a, b), v in counts.items()})
    return slots


def misplaced_bays_to_exclude(
    base: pd.DataFrame,
    area_functions: dict[str, set[str]],
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
    ratio: float,
    original_bay_capacity: Counter[tuple[str, str]],
    target_voyages: set[str],
) -> set[tuple[str, str]]:
    if ratio <= 0:
        return set()
    occupied = active_occupied(base, vessel_schedules, planning_time)
    if occupied.empty:
        return set()
    target_voyages = {normalize_voyage(v) for v in target_voyages}
    occupied = occupied[occupied["IYC_EVOY_ID"].map(normalize_voyage).isin(target_voyages)].copy()
    if occupied.empty:
        return set()
    occupied["_flow"] = occupied["IYC_STS_CSTATUSCD"].map(lambda value: normalize_flow(value, default="OF"))
    bad = occupied[
        [
            flow not in area_functions.get(area, set())
            for area, flow in zip(occupied["YAA_AREANO"], occupied["_flow"])
        ]
    ].copy()
    if bad.empty:
        return set()
    counts = bad.groupby(["YAA_AREANO", "YBY_BAYNO"]).size()
    return {
        (str(area), str(bay))
        for (area, bay), count in counts.items()
        if int(count) > original_bay_capacity.get((str(area), str(bay)), 0) * ratio
    }


def existing_bay_attributes(
    frame: pd.DataFrame,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> dict[tuple[str, str], dict[str, set[str]]]:
    occupied = active_occupied(frame, vessel_schedules, planning_time)
    out: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in occupied.to_dict("records"):
        key = (row["YAA_AREANO"], row["YBY_BAYNO"])
        attrs = out.setdefault(key, {"sizes": set(), "heights": set(), "specials": set(), "ports": set()})
        attrs["sizes"].add(normalize_size_small(row.get("IYC_CSZ_CSIZECD")))
        attrs["heights"].add(normalize_text(row.get("IYC_CHEIGHTCD"), "UNK"))
        special = special_stow_code(row) or "NORMAL"
        attrs["specials"].add(special)
        port = normalize_text(row.get("IYC_POT_UNLDPORT"))
        if port:
            attrs["ports"].add(port)
    return out


def make_yard_blocks(ordered_bays: list[str], big_bay_starts: set[str], target_bay_count: int = 6) -> list[tuple[int, tuple[str, ...], bool]]:
    blocks: list[tuple[int, tuple[str, ...], bool]] = []
    start = 0
    block_index = 1
    while start < len(ordered_bays):
        if len(ordered_bays) - start <= target_bay_count:
            end = len(ordered_bays)
            adjusted = False
        else:
            target_end = start + target_bay_count
            end = nearest_safe_block_end(ordered_bays, big_bay_starts, start, target_end)
            adjusted = end != target_end
        blocks.append((block_index, tuple(ordered_bays[start:end]), adjusted))
        block_index += 1
        start = end
    return blocks


def nearest_safe_block_end(ordered_bays: list[str], big_bay_starts: set[str], start: int, target_end: int) -> int:
    min_end = start + 1
    max_end = len(ordered_bays)
    for offset in range(0, len(ordered_bays) + 1):
        probes = [target_end] if offset == 0 else [target_end + offset, target_end - offset]
        for end in probes:
            if not (min_end <= end <= max_end):
                continue
            if end == len(ordered_bays) or ordered_bays[end - 1] not in big_bay_starts:
                return end
    return len(ordered_bays)


def build_area_operations(data_dir: Path, vessel_schedules: dict[str, VoyageSchedule]) -> dict[str, list[AreaOperation]]:
    operations: defaultdict[str, list[AreaOperation]] = defaultdict(list)
    tops = active_tops_rows(data_dir, datetime.max.replace(year=2099))
    if tops.empty:
        return dict(operations)
    for row in tops.to_dict("records"):
        start_area, _ = parse_tops_area_bay(row.get("SPR_STBAY"))
        end_area, _ = parse_tops_area_bay(row.get("SPR_EDBAY"))
        area = start_area or end_area
        voyage = normalize_voyage(row.get("condition_vessel"))
        start_time = row.get("start_time")
        end_time = row.get("end_time")
        if area and voyage and start_time and end_time:
            operations[area].append(AreaOperation(area, voyage, start_time, end_time))
    return dict(operations)


def build_problem(
    data_dir: Path,
    big_plan: list[BigPlanRow],
    planning_time: datetime,
    horizon_hours: float,
    target_voyages: list[str],
    misplaced_bay_exclusion_ratio: float,
) -> ProblemData:
    data_dir = data_dir.resolve()
    closed = read_closed_areas(data_dir)
    area_functions = read_area_functions(data_dir)
    function_areas = set(area_functions)
    area_quota: dict[tuple[str, str, str], int] = {}
    area_size_quota: dict[tuple[str, str, str, str], int] = {}
    assigned_areas: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    cleaned_plan: list[BigPlanRow] = []
    target_voyages = [normalize_voyage(v) for v in target_voyages]
    vessel_schedules = read_target_vessel_schedules(data_dir, target_voyages, planning_time, horizon_hours)
    plan_date = planning_time.date().isoformat()
    input_plan = [
        row for row in big_plan if row.voyage_id in target_voyages and (not row.plan_date or row.plan_date == plan_date)
    ]
    allowed_areas = set(function_areas)
    for row in input_plan:
        if row.area_no in closed:
            raise ValueError(f"big plan uses closed area {row.area_no} for voyage {row.voyage_id}")
        if row.flow not in area_functions.get(row.area_no, set()):
            continue
        cleaned_plan.append(row)
        allowed_areas.add(row.area_no)
        assigned_areas[(row.voyage_id, row.flow)].add(row.area_no)
    if not cleaned_plan:
        raise ValueError("no big-plan rows remain after target-voyage, flow, date, and closed-area filtering")
    groups, _demand_rows = load_port_demand_groups(data_dir, target_voyages, planning_time)
    small_groups = load_small_doc_groups(data_dir, target_voyages)
    demand_by_voyage_size: Counter[tuple[str, str, str]] = Counter()
    for group in groups:
        demand_by_voyage_size[(group.voyage_id, group.status, group.big_plan_size_mode)] += group.demand
    raw_area_quota: Counter[tuple[str, str, str]] = Counter()
    raw_area_size_quota: Counter[tuple[str, str, str, str]] = Counter()
    raw_all_size_area_quota: Counter[tuple[str, str, str]] = Counter()
    for row in cleaned_plan:
        raw_area_quota[(row.voyage_id, row.flow, row.area_no)] += row.planned_boxes
        if row.size_mode == "ALL":
            raw_all_size_area_quota[(row.voyage_id, row.flow, row.area_no)] += row.planned_boxes
        else:
            raw_area_size_quota[(row.voyage_id, row.flow, row.area_no, row.size_mode)] += row.planned_boxes
    for voyage_id in target_voyages:
        flows = sorted({flow for (v, flow, _size), qty in demand_by_voyage_size.items() if v == voyage_id and qty > 0})
        for flow in flows:
            for size_mode in SIZE_MODES:
                target_qty = demand_by_voyage_size[(voyage_id, flow, size_mode)]
                if target_qty <= 0:
                    continue
                exact_upper = Counter(
                    {
                        area_no: qty
                        for (v, f, area_no, size), qty in raw_area_size_quota.items()
                        if v == voyage_id and f == flow and size == size_mode and qty > 0
                    }
                )
                if exact_upper:
                    if sum(exact_upper.values()) < target_qty:
                        raise ValueError(
                            f"medium demand exceeds big-plan strict upper bound for "
                            f"voyage={voyage_id}, flow={flow}, size={size_mode}: "
                            f"demand={target_qty}, big_plan_upper={sum(exact_upper.values())}"
                        )
                    for area_no, qty in exact_upper.items():
                        area_quota[(voyage_id, flow, area_no)] = raw_area_quota[(voyage_id, flow, area_no)]
                        area_size_quota[(voyage_id, flow, area_no, size_mode)] = qty
                        assigned_areas[(voyage_id, flow)].add(area_no)
                    continue
                all_size_weights = Counter(
                    {
                        area_no: qty
                        for (v, f, area_no), qty in raw_all_size_area_quota.items()
                        if v == voyage_id and f == flow and qty > 0
                    }
                )
                if not all_size_weights:
                    raise ValueError(f"big plan has no area pattern for voyage={voyage_id}, flow={flow}, size={size_mode}")
                if sum(all_size_weights.values()) < target_qty:
                    raise ValueError(
                        f"medium demand exceeds big-plan ALL-size strict upper bound for "
                        f"voyage={voyage_id}, flow={flow}, size={size_mode}: "
                        f"demand={target_qty}, big_plan_upper={sum(all_size_weights.values())}"
                    )
                allocations = allocate_by_weights(dict(all_size_weights), target_qty)
                for area_no, qty in allocations.items():
                    if qty <= 0:
                        continue
                    area_quota[(voyage_id, flow, area_no)] = raw_area_quota[(voyage_id, flow, area_no)]
                    area_size_quota[(voyage_id, flow, area_no, size_mode)] = qty
                    assigned_areas[(voyage_id, flow)].add(area_no)
    voyage_windows = {
        voyage_id: (
            vessel_schedules[voyage_id].receive_start if voyage_id in vessel_schedules else planning_time,
            (vessel_schedules[voyage_id].receive_start if voyage_id in vessel_schedules else planning_time)
            + timedelta(hours=horizon_hours),
        )
        for voyage_id in target_voyages
    }
    bays, reserved_count, closed_bay_count, misplaced_count = build_bays(
        data_dir,
        allowed_areas,
        closed,
        planning_time,
        vessel_schedules,
        set(target_voyages),
        area_functions,
        misplaced_bay_exclusion_ratio,
    )
    area_operations = build_area_operations(data_dir, vessel_schedules)
    berth_distances = read_distance_matrix(data_dir)
    berth_by_voyage = {
        voyage_id: f"B{vessel_schedules[voyage_id].berth_no}"
        for voyage_id in target_voyages
        if voyage_id in vessel_schedules and vessel_schedules[voyage_id].berth_no
    }
    return ProblemData(
        groups=groups,
        small_groups=small_groups,
        bays=bays,
        big_plan=cleaned_plan,
        assigned_areas=dict(assigned_areas),
        area_quota=area_quota,
        area_size_quota=area_size_quota,
        area_functions=area_functions,
        business_special_codes=collect_business_special_codes(groups),
        planning_time=planning_time,
        horizon_hours=horizon_hours,
        voyage_windows=voyage_windows,
        area_operations=area_operations,
        target_voyages=target_voyages,
        berth_distances=berth_distances,
        berth_by_voyage=berth_by_voyage,
        tops_reserved_slot_count=reserved_count,
        tops_closed_bay_count=closed_bay_count,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
        misplaced_excluded_bay_count=misplaced_count,
    )


def load_medium_small_inputs(
    data_dir: Path,
    big_plan_path: Path,
    planning_time: datetime,
    voyages: Sequence[str],
    horizon_hours: float,
    misplaced_bay_exclusion_ratio: float,
) -> MediumSmallInputs:
    big_plan = read_big_plan(big_plan_path)
    demand_rows = calculate_medium_demands(data_dir, list(voyages), planning_time)
    problem = build_problem(
        data_dir,
        big_plan,
        planning_time=planning_time,
        horizon_hours=horizon_hours,
        target_voyages=list(voyages),
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
    )
    return MediumSmallInputs(big_plan=big_plan, demand_rows=demand_rows, problem=problem)


def allocate_by_weights(weights: dict[str, int], target_total: int) -> dict[str, int]:
    items = [(key, value) for key, value in sorted(weights.items()) if value > 0]
    if not items or target_total <= 0:
        return {}
    source_total = sum(value for _, value in items)
    raw = [value * target_total / source_total for _, value in items]
    base = [int(value) for value in raw]
    remain = target_total - sum(base)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
    for idx in order[:remain]:
        base[idx] += 1
    return {key: qty for (key, _), qty in zip(items, base) if qty > 0}


def weight_class(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "UNK"
    numeric = float(numeric)
    bins = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 999)]
    for lo, hi in bins:
        if lo <= numeric < hi:
            return f"{lo}_{hi}"
    return "UNK"


def business_special_codes(row: Mapping[str, Any]) -> set[str]:
    codes = set()
    ctype = normalize_code(row.get("IYC_CTYPECD"))
    if ctype in {"RF", "DG", "OT", "TK"}:
        codes.add(ctype)
    if pd.notna(row.get("IYC_SETTMPT")):
        codes.add("REEFER")
    if pd.notna(row.get("IYC_DTP_DNGGCD")):
        codes.add("DANGER")
    if pd.notna(row.get("IYC_OVLMTCD")):
        codes.add("OVER")
    return codes


def special_stow_code(row: Mapping[str, Any]) -> str:
    return "+".join(sorted(business_special_codes(row)))


def explicit_special_stow_code(row: Mapping[str, Any]) -> str:
    return ""


def collect_business_special_codes(groups: list[BoxGroup]) -> set[str]:
    out: set[str] = set()
    for group in groups:
        out.update(group.special_codes)
    return out


def bay_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        nums = re.findall(r"\d+", value)
        return (int(nums[0]) if nums else 9999), value


def date_key(value: str) -> str:
    if not value:
        return ""
    parsed = parse_datetime(value)
    return parsed.date().isoformat() if parsed else value


def _first_existing(fieldnames: set[Any], candidates: list[str]) -> str | None:
    string_names = {str(field): field for field in fieldnames}
    for candidate in candidates:
        if candidate in string_names:
            return string_names[candidate]
    return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_EXPORT_VESSELS",
    "DEFAULT_IMPORT_VESSELS",
    "DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO",
    "DEFAULT_PLANNING_TIME",
    "DEFAULT_TARGET_VOYAGES",
    "MediumSmallInputs",
    "PlanningInputArtifacts",
    "RollingPlanningState",
    "build_large_inputs",
    "load_medium_small_inputs",
    "parse_datetime",
    "parse_planning_time",
    "write_demand_rows",
    "write_json",
    "write_large_outputs",
]
