from __future__ import annotations

import csv
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from .demand_calculator import DemandRow, calculate_medium_demands, planning_stage_window, read_excel_compat
from .input_json import (
    has_input_json,
    input_dataframe,
    input_value,
    vessel_container_ids,
    vessel_doc_frame,
)
from .models import AreaOperation, AttributeRules, Bay, BigPlanRow, BoxGroup, ProblemData, SmallBoxGroup, VoyageSchedule


DEFAULT_PLANNING_TIME = datetime(2026, 5, 19, 9, 30)
SIZE_MODES = ("20", "40", "45")
DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO = 2.0 / 3.0
DEFAULT_TARGET_BIG_PLAN_FLOWS = frozenset({"OF"})
ATTRIBUTE_ALIASES = {
    "voyage": "voyage_id",
    "voyage_id": "voyage_id",
    "flow": "status",
    "status": "status",
    "iyc_sts_cstatuscd": "status",
    "port": "port",
    "discharge_port": "port",
    "pod": "port",
    "iyc_pot_unldport": "port",
    "size": "size",
    "size_mode": "size",
    "iyc_csz_csizecd": "size",
    "height": "height",
    "iyc_cheightcd": "height",
    "iyc_csz_cheightd": "height",
    "weight": "weight_class",
    "weight_class": "weight_class",
    "iyc_cweight": "weight_class",
    "special": "special_stow_code",
    "special_code": "special_stow_code",
    "special_stow_code": "special_stow_code",
    "pre_stow": "pre_stow",
    "operator": "operator",
    "ctype": "ctype",
    "container_type": "ctype",
}
SMALL_GROUP_COLUMN_BY_ATTR = {
    "status": "status",
    "port": "port",
    "size": "size",
    "height": "height",
    "weight_class": "weight",
    "special_stow_code": "special_code",
    "pre_stow": "pre_stow",
}
DEFAULT_FINE_GROUP_COLUMNS = ("status", "size", "port", "height", "weight", "special_code", "pre_stow")
DEFAULT_MEDIUM_GROUP_COLUMNS = ("status", "size")


def _read_csv_compat(path: str | Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def _norm(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text or default


def _voyage(value: object, fallback: str = "") -> str:
    text = _norm(value, fallback)
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _voyage_sort_key(value: object) -> tuple[int, str]:
    text = _voyage(value)
    try:
        return int(text), text
    except ValueError:
        return 10**12, text


def _map_unique(series: pd.Series, func) -> pd.Series:
    mapping = {value: func(value) for value in series.drop_duplicates()}
    return series.map(mapping).fillna("").astype(str)


def _norm_series(series: pd.Series, default: str = "") -> pd.Series:
    return _map_unique(series, lambda value: _norm(value, default))


def _voyage_series(series: pd.Series, fallback: str = "") -> pd.Series:
    return _map_unique(series, lambda value: _voyage(value, fallback))


def _bay_no_series(series: pd.Series) -> pd.Series:
    return _map_unique(series, _bay_no)


def _row_no_series(series: pd.Series) -> pd.Series:
    return _bay_no_series(series)


def parse_datetime(value: object) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
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
    if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        parsed = pd.to_datetime(text, errors="coerce")
        if not pd.isna(parsed):
            return parsed.to_pydatetime()
    return None


def _read_one_column_xlsx(path: Path, header: str = "area_no") -> set[str]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
        sheet = ElementTree.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    values: list[str] = []
    for cell in sheet.findall(".//a:sheetData/a:row/a:c", ns):
        raw = cell.find("a:v", ns)
        if raw is None:
            continue
        value = raw.text or ""
        if cell.attrib.get("t") == "s":
            value = shared[int(value)]
        values.append(value.strip())
    return {v for v in values if v and v != header}


def read_closed_areas(data_dir: str | Path) -> set[str]:
    if has_input_json(data_dir):
        return {_norm(value) for value in input_value(data_dir, "closed_area", []) if _norm(value)}
    path = Path(data_dir) / "n_usefg_areas.txt"
    text = path.read_text(encoding="utf-8").strip()
    return set(re.findall(r"[A-Za-z0-9]+", text))


def read_export_areas(data_dir: str | Path) -> set[str]:
    return {area for area, funcs in read_area_functions(data_dir).items() if "OF" in funcs}


def read_area_functions(data_dir: str | Path) -> dict[str, set[str]]:
    data_path = Path(data_dir)
    if has_input_json(data_path):
        frame = input_dataframe(data_path, "area_function_info")
        if {"area_no", "cntr_type"}.issubset(frame.columns):
            area_functions = {
                _norm(row["area_no"]): {_norm(part).upper() for part in str(row.get("cntr_type")).split(",") if _norm(part)}
                for row in frame.to_dict("records")
                if _norm(row.get("area_no"))
            }
            if area_functions:
                return area_functions
    function_files = [path for path in data_path.glob("*功能*.xlsx") if not path.name.startswith("~$")]
    if function_files:
        frame = read_excel_compat(function_files[0])
        if {"area_no", "cntr_type"}.issubset(frame.columns):
            area_functions = {
                _norm(row["area_no"]): {_norm(part).upper() for part in str(row.get("cntr_type")).split(",") if _norm(part)}
                for row in frame.to_dict("records")
                if _norm(row.get("area_no"))
            }
            if area_functions:
                return area_functions

    matrix_files = [
        path for path in data_path.glob("*.xlsx")
        if _is_area_distance_workbook(path)
    ]
    if matrix_files:
        frame = read_excel_compat(matrix_files[0], sheet_name="箱区坐标")
        if "area_no" in frame.columns:
            return {_norm(value): {"OF"} for value in frame["area_no"] if _norm(value)}
    raise FileNotFoundError(f"No area function definition found under {data_path}")


def read_vessel_schedules(data_dir: str | Path) -> dict[str, VoyageSchedule]:
    # vessel_berth_info.csv can support several planning decisions. In the current
    # medium/small solver, area-level choices are fixed by the supplied big plan,
    # so the schedule fields are currently consumed for capacity timing: whether
    # containers already in the yard still occupy capacity at the planning timestamp.
    data_path = Path(data_dir)
    if has_input_json(data_path):
        frame = input_dataframe(data_path, "vessel_berth_info")
    else:
        path = data_path / "vessel_berth_info_new.csv"
        if not path.exists():
            path = data_path / "vessel_berth_info.csv"
        frame = _read_csv_compat(path)
    schedules: dict[str, VoyageSchedule] = {}
    for row in frame.to_dict("records"):
        if _norm(row.get("VOY_IEFG")) != "E":
            continue
        voyage_id = _voyage(row.get("VOY_ID"))
        receive_start = parse_datetime(row.get("SCD_RCVSTDT"))
        receive_end = parse_datetime(row.get("SCD_RCVEDDT"))
        berth_time = parse_datetime(row.get("VBT_ABTHDT")) or parse_datetime(row.get("VBT_PBTHDT"))
        departure_time = parse_datetime(row.get("VBT_ADPTDT")) or parse_datetime(row.get("VBT_PDPTDT"))
        berth_no = _norm(row.get("VBT_BTH_ABTHNO")) or _norm(row.get("VBT_BTH_PBTHNO"))
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
    data_dir: str | Path,
    target_voyages: list[str],
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    horizon_hours: float = 24.0,
) -> dict[str, VoyageSchedule]:
    schedules = read_vessel_schedules(data_dir)
    missing = [_voyage(v) for v in target_voyages if _voyage(v) not in schedules]
    if not missing:
        return schedules

    data_path = Path(data_dir)
    path = data_path / "vessel_berth_info_new.csv"
    if not path.exists():
        path = data_path / "vessel_berth_info.csv"
    if not path.exists():
        return schedules

    frame = _read_csv_compat(path)
    rows_by_voyage = {
        _voyage(row.get("VOY_ID")): row
        for row in frame.to_dict("records")
        if _norm(row.get("VOY_IEFG")) == "E"
    }
    for voyage_id in missing:
        row = rows_by_voyage.get(voyage_id)
        if row is None:
            continue
        berth_no = _norm(row.get("VBT_BTH_ABTHNO")) or _norm(row.get("VBT_BTH_PBTHNO"))
        receive_start = planning_time
        receive_end = planning_time + timedelta(hours=horizon_hours)
        berth_time = planning_time + timedelta(hours=horizon_hours)
        departure_time = berth_time + timedelta(hours=12)
        schedules[voyage_id] = VoyageSchedule(
            voyage_id=voyage_id,
            receive_start=receive_start,
            receive_end=receive_end,
            berth_no=berth_no,
            berth_time=berth_time,
            departure_time=departure_time,
        )
    return schedules


def read_berth_distances(data_dir: str | Path) -> dict[tuple[str, str], float]:
    data_path = Path(data_dir)
    if has_input_json(data_path):
        frame = input_dataframe(data_path, "berth_area_dist_matrix")
        if frame.empty:
            return {}
        return _berth_distances_from_matrix_frame(frame)
    candidates = _distance_workbook_candidates(data_path)
    if not candidates:
        return {}
    frame = read_excel_compat(candidates[0], sheet_name="距离矩阵")
    return _berth_distances_from_matrix_frame(frame)


def _berth_distances_from_matrix_frame(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    distances: dict[tuple[str, str], float] = {}
    berth_columns = [col for col in frame.columns if re.fullmatch(r"B\d+", str(col))]
    for row in frame.to_dict("records"):
        area_no = _norm(row.get("area_no"))
        for berth in berth_columns:
            value = row.get(berth)
            if area_no and not pd.isna(value):
                distances[(area_no, str(berth))] = float(value)
    return distances


def _distance_workbook_candidates(data_path: Path) -> list[Path]:
    candidates = [
        path for path in data_path.glob("*.xlsx")
        if _is_area_distance_workbook(path)
    ]
    repo_large = Path(__file__).resolve().parents[2] / "large" / "适放箱区_泊位距离矩阵.xlsx"
    if repo_large.exists() and repo_large not in candidates:
        candidates.append(repo_large)
    return candidates


def _is_area_distance_workbook(path: Path) -> bool:
    if path.name.startswith("~$") or path.suffix.lower() != ".xlsx":
        return False
    name = path.stem.lower()
    return name.startswith("of") or ("适放箱区" in path.stem and "距离矩阵" in path.stem)


def select_upcoming_opening_voyages(
    data_dir: str | Path,
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    horizon_hours: float = 24.0,
    count: int = 2,
) -> list[str]:
    horizon_end = planning_time + timedelta(hours=horizon_hours)
    candidates = [
        schedule
        for schedule in read_vessel_schedules(data_dir).values()
        if planning_time <= schedule.receive_start < horizon_end
    ]
    candidates.sort(key=lambda item: (item.receive_start, item.berth_time, item.voyage_id))
    return [item.voyage_id for item in candidates[:count]]


def read_big_plan(path: str | Path) -> list[BigPlanRow]:
    """读取大计划结果，并保留 20/40 尺寸配额。

    支持中小计划标准格式，以及大计划 ``allocation.csv`` 格式
    ``voy_id,flow,area_no,size,new_qty``。没有 ``flow`` 列时默认 ``OF``。
    """

    counter: Counter[tuple[str, str, str, str, str]] = Counter()
    rows: list[BigPlanRow] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as fp:
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
                flow = _flow(row.get(flow_field)) if flow_field else "OF"
                voyage_id = _voyage(row.get("voyage_id"))
                area_no = _norm(row.get("area_no"))
                plan_date = _norm(row.get(date_field)) if date_field else ""
                for size_mode, field in (("20", qty20_field), ("40", qty40_field), ("45", qty45_field)):
                    if not field:
                        continue
                    boxes = int(round(float(row.get(field, 0) or 0)))
                    if boxes > 0:
                        counter[(voyage_id, flow, area_no, size_mode, _date_key(plan_date))] += boxes
            rows = [
                BigPlanRow(voyage_id, flow, area_no, boxes, size_mode, plan_date)
                for (voyage_id, flow, area_no, size_mode, plan_date), boxes in sorted(counter.items())
                if boxes > 0
            ]
            if not rows:
                raise ValueError("big plan file contains no positive planned boxes")
            return rows

        required = {"voyage_id", "area_no", "planned_boxes"}
        missing = required.difference(reader.fieldnames or [])
        if missing and {"voy_id", "area_no"}.issubset(reader.fieldnames or []) and (
            "new_qty" in fieldnames or "planned_qty" in fieldnames
        ):
            date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
            flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
            qty_field = _first_existing(fieldnames, ["new_qty", "planned_qty"])
            for row in reader:
                flow = _flow(row.get(flow_field)) if flow_field else "OF"
                boxes = int(round(float(row.get(qty_field, 0) or 0)))
                if boxes > 0:
                    size_mode = _big_plan_size_mode(row.get("size"))
                    plan_date = _date_key(_norm(row.get(date_field)) if date_field else "")
                    counter[(_voyage(row["voy_id"]), flow, _norm(row["area_no"]), size_mode, plan_date)] += boxes
            rows = [
                BigPlanRow(voyage_id, flow, area_no, boxes, size_mode, plan_date)
                for (voyage_id, flow, area_no, size_mode, plan_date), boxes in sorted(counter.items())
                if boxes > 0
            ]
        elif missing and not (
            {"voyage_id", "area_no"}.issubset(reader.fieldnames or [])
            and ("new_qty" in fieldnames or "planned_qty" in fieldnames)
        ):
            raise ValueError(f"big plan file missing columns: {sorted(missing)}")
        else:
            size_field = "size_mode" if "size_mode" in (reader.fieldnames or []) else "size"
            date_field = _first_existing(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
            flow_field = _first_existing(fieldnames, ["flow", "cntr_type", "status"])
            qty_field = _first_existing(fieldnames, ["new_qty", "planned_boxes", "planned_qty"])
            for row in reader:
                flow = _flow(row.get(flow_field)) if flow_field else "OF"
                boxes = int(round(float(row.get(qty_field, 0) or 0)))
                if boxes > 0:
                    size_mode = _big_plan_size_mode(row.get(size_field)) if size_field in row else "ALL"
                    plan_date = _date_key(_norm(row.get(date_field)) if date_field else "")
                    rows.append(BigPlanRow(_voyage(row["voyage_id"]), flow, _norm(row["area_no"]), boxes, size_mode, plan_date))
    if not rows:
        raise ValueError("big plan file contains no positive planned_boxes")
    return rows


def infer_target_voyages_from_big_plan(
    big_plan: list[BigPlanRow],
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    target_flows: set[str] | frozenset[str] = DEFAULT_TARGET_BIG_PLAN_FLOWS,
) -> list[str]:
    """Infer target voyages from active OF big-plan rows by default."""

    plan_date = planning_time.date().isoformat()
    normalized_flows = {_flow(flow) for flow in target_flows}
    voyages = {
        row.voyage_id
        for row in big_plan
        if row.voyage_id
        and row.flow in normalized_flows
        and (not row.plan_date or row.plan_date == plan_date)
    }
    return sorted(voyages, key=_voyage_sort_key)


def medium_demand_caps_from_big_plan(
    big_plan: list[BigPlanRow],
    target_voyages: list[str] | tuple[str, ...] | set[str],
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    target_flows: set[str] | frozenset[str] = DEFAULT_TARGET_BIG_PLAN_FLOWS,
) -> dict[tuple[str, str, str], int]:
    """Build medium-demand caps from active OF big-plan new_qty rows.

    The demand calculator works at (voyage, flow, size, port), while the big
    plan gives area/size quotas. Real 45ft demand inherits the 40ft big-plan
    size mode.
    """

    plan_date = planning_time.date().isoformat()
    voyage_set = {_voyage(voyage_id) for voyage_id in target_voyages}
    normalized_flows = {_flow(flow) for flow in target_flows}
    size_pool: Counter[tuple[str, str]] = Counter()
    all_size_pool: Counter[str] = Counter()
    for row in big_plan:
        if row.voyage_id not in voyage_set:
            continue
        if row.flow not in normalized_flows:
            continue
        if row.plan_date and row.plan_date != plan_date:
            continue
        if row.size_mode == "ALL":
            all_size_pool[row.voyage_id] += row.planned_boxes
        else:
            size_pool[(row.voyage_id, "40" if row.size_mode == "45" else row.size_mode)] += row.planned_boxes

    caps: dict[tuple[str, str, str], int] = {}
    for (voyage_id, size_mode), qty in size_pool.items():
        if qty <= 0:
            continue
        for flow in normalized_flows:
            caps[(voyage_id, flow, size_mode)] = qty
    for voyage_id, qty in all_size_pool.items():
        if qty <= 0 or any(v == voyage_id for v, _size in size_pool):
            continue
        for flow in normalized_flows:
            caps[(voyage_id, flow, "ALL")] = qty
    return caps


def load_box_groups(
    data_dir: str | Path,
    voyage_ids: set[str],
    planned_by_voyage: dict[str, int],
    planned_by_voyage_size: dict[tuple[str, str], int] | None = None,
) -> list[BoxGroup]:
    """读取箱明细并缩放成中小计划需求组。

    箱量目标完全来自大计划，不再按 70% 或其他比例缩放。
    如果大计划提供 20/40 尺寸拆分，则按航次和尺寸分别对齐；45ft
    明细归入 40ft 目标。
    """
    data_path = Path(data_dir)
    counters: Counter[tuple] = Counter()
    if has_input_json(data_path):
        sources = [(_voyage(voyage_id), vessel_doc_frame(data_path, voyage_id)) for voyage_id in vessel_container_ids(data_path)]
    else:
        sources = [
            (path.stem.replace("container_info_", ""), pd.read_parquet(path))
            for path in sorted(data_path.glob("container_info_*.parquet"))
        ]
    for file_voyage, frame in sources:
        if frame.empty:
            continue
        for row in frame.to_dict("records"):
            voyage_id = _voyage(row.get("IYC_EVOY_ID"), file_voyage)
            if voyage_id not in voyage_ids:
                continue
            ctype = _norm(row.get("IYC_CTYPECD"), "UNK")
            status = _norm(row.get("IYC_STS_CSTATUSCD"), "UNK")
            key = (
                voyage_id,
                _norm(row.get("IYC_CSZ_CSIZECD"), "40"),
                _norm(row.get("IYC_CHEIGHTCD"), "UNK"),
                status,
                _norm(row.get("IYC_POT_UNLDPORT"), "UNK"),
                _norm(row.get("IYC_CST_COPERCD"), "UNK"),
                ctype,
                _weight_class(row.get("IYC_CWEIGHT")),
                bool(ctype == "RF" or not pd.isna(row.get("IYC_SETTMPT"))),
                bool(not pd.isna(row.get("IYC_DTP_DNGGCD"))),
                bool(not pd.isna(row.get("IYC_OVLMTCD"))),
                tuple(sorted(_business_special_codes(row))),
            )
            counters[key] += 1

    raw_by_voyage: defaultdict[str, list[tuple[tuple, int]]] = defaultdict(list)
    for key, count in counters.items():
        raw_by_voyage[key[0]].append((key, count))

    groups: list[BoxGroup] = []
    for voyage_id in sorted(voyage_ids):
        raw = raw_by_voyage.get(voyage_id, [])
        if planned_by_voyage_size:
            size_targets = {
                size_mode: qty
                for (v, size_mode), qty in planned_by_voyage_size.items()
                if v == voyage_id and qty > 0
            }
            group_index = 1
            for size_mode in SIZE_MODES:
                target = size_targets.get(size_mode, 0)
                if target <= 0:
                    continue
                raw_for_size = [
                    item for item in raw
                    if _big_plan_size_mode(item[0][1]) == size_mode
                ]
                if not raw_for_size:
                    raw_for_size = [(_generic_box_key(voyage_id, size_mode), 1)]
                group_index = _append_scaled_groups(groups, raw_for_size, target, group_index)
            continue

        if not raw:
            raw = [(_generic_box_key(voyage_id, "40"), 1)]
        _append_scaled_groups(groups, raw, planned_by_voyage[voyage_id], 1)
    return groups


def load_port_demand_groups(
    data_dir: str | Path,
    voyage_ids: list[str],
    planning_time: datetime,
    horizon_hours: float = 24.0,
    big_plan_caps: dict[tuple[str, str, str], int] | None = None,
    attribute_rules: AttributeRules | None = None,
) -> tuple[list[BoxGroup], list[DemandRow]]:
    demand_rows = calculate_medium_demands(
        data_dir,
        voyage_ids,
        planning_time,
        horizon_hours,
        big_plan_caps=big_plan_caps,
    )
    groups: list[BoxGroup] = []
    group_index: defaultdict[str, int] = defaultdict(int)
    counters: defaultdict[str, Counter[tuple[tuple[str, ...], tuple[object, ...]]]] = defaultdict(Counter)
    for row in demand_rows:
        attrs = {
            "status": row.flow,
            "size": row.size_mode,
            "port": row.port,
            "height": "UNK",
            "weight_class": "UNK",
            "special_code": "",
            "special_stow_code": "",
            "pre_stow": False,
        }
        group_columns = _medium_groupby_columns(attribute_rules, row.voyage_id)
        key = tuple(attrs.get(column, "MIXED") for column in group_columns)
        counters[row.voyage_id][(group_columns, key)] += int(row.planned_boxes)
    for voyage_id in sorted(counters):
        for (group_columns, key), demand in sorted(counters[voyage_id].items(), key=lambda item: item[0][1]):
            values = dict(zip(group_columns, key))
            group_index[voyage_id] += 1
            groups.append(
                BoxGroup(
                    group_id=f"{voyage_id}_P{group_index[voyage_id]:03d}",
                    voyage_id=voyage_id,
                    size=str(values.get("size", "40")),
                    height=str(values.get("height", "UNK")),
                    status=str(values.get("status", "OF")),
                    port=str(values.get("port", "MIXED")),
                    operator="UNK",
                    ctype="UNK",
                    weight_class=str(values.get("weight_class", "UNK")),
                    reefer=False,
                    dangerous=False,
                    over_limit=False,
                    special_codes=(),
                    demand=int(demand),
                )
            )
    return groups, demand_rows


def load_small_doc_groups(
    data_dir: str | Path,
    voyage_ids: list[str],
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    attribute_rules: AttributeRules | None = None,
) -> list[SmallBoxGroup]:
    data_path = Path(data_dir)
    yard_ids, yard_numbers = _current_yard_container_keys(data_path, planning_time)
    groups: list[SmallBoxGroup] = []
    for voyage_id in voyage_ids:
        if has_input_json(data_path):
            frame = vessel_doc_frame(data_path, voyage_id)
        else:
            path = data_path / f"container_info_{voyage_id}.parquet"
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
        if frame.empty:
            continue
        if yard_ids or yard_numbers:
            cntr_ids = frame.get("IYC_CNTRID", pd.Series(index=frame.index, dtype=object)).map(_norm)
            cntr_numbers = frame.get("IYC_CNTRNO", pd.Series(index=frame.index, dtype=object)).map(_norm)
            in_yard = cntr_ids.isin(yard_ids) | cntr_numbers.isin(yard_numbers)
            frame = frame.loc[~in_yard].copy()
            if frame.empty:
                continue
        work = pd.DataFrame(
            {
                "status": frame.get("IYC_STS_CSTATUSCD", pd.Series(index=frame.index, dtype=object)).map(_flow),
                "size": frame.get("IYC_CSZ_CSIZECD", pd.Series(index=frame.index, dtype=object)).map(_size_mode),
                "port": frame.get("IYC_POT_UNLDPORT", pd.Series(index=frame.index, dtype=object)).map(lambda value: _norm(value, "UNK")),
                "height": frame.get("IYC_CHEIGHTCD", pd.Series(index=frame.index, dtype=object)).map(lambda value: _norm(value, "UNK")),
                "weight": frame.get("IYC_CWEIGHT", pd.Series(index=frame.index, dtype=object)).map(
                    lambda value: _weight_class(
                        value,
                        attribute_rules.weight_levels_for(voyage_id)
                        if attribute_rules is not None and hasattr(attribute_rules, "weight_levels_for")
                        else (0, 10, 15, 20, 25, 30),
                    )
                ),
                "special_code": "",
                "pre_stow": False,
            }
        )
        group_columns = _small_groupby_columns(attribute_rules, voyage_id)
        counts = work.groupby(list(group_columns), sort=True).size()
        counter: Counter[tuple] = Counter()
        for key, count in counts.items():
            if len(group_columns) == 1:
                key = (key,)
            values = dict(zip(group_columns, key))
            counter[tuple(values.get(column, "") for column in group_columns)] = int(count)
        for index, (key, demand) in enumerate(sorted(counter.items()), start=1):
            values = dict(zip(group_columns, key))
            status = str(values.get("status", "MIXED"))
            size = str(values.get("size", "40"))
            port = str(values.get("port", "MIXED"))
            height = str(values.get("height", "MIXED"))
            weight = str(values.get("weight", "MIXED"))
            special_code = str(values.get("special_code", ""))
            pre_stow = bool(values.get("pre_stow", False))
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


def enforce_medium_groups_cover_small_groups(
    groups: list[BoxGroup],
    small_groups: list[SmallBoxGroup],
) -> tuple[list[BoxGroup], dict[str, int], dict[str, int]]:
    """Adjust medium demand so it covers document floors without inflating totals first.

    The document floor is exact at (voyage, flow, discharge-port, true-size).
    When a floor is short, first shift forecast demand from other coarse groups
    in the same (voyage, flow, big-plan-size) bucket. Only the remaining
    uncovered quantity is added as true extra medium demand.
    """
    updated = list(groups)
    medium_by_key: Counter[tuple[str, str, str, str]] = Counter()
    group_indices_by_key: defaultdict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, group in enumerate(updated):
        key = _medium_small_coarse_key(group.voyage_id, group.status, group.port, group.size_mode)
        medium_by_key[key] += group.demand
        group_indices_by_key[key].append(index)

    small_by_key: Counter[tuple[str, str, str, str]] = Counter()
    for group in small_groups:
        key = _medium_small_coarse_key(group.voyage_id, group.status, group.port, group.size)
        small_by_key[key] += group.demand

    used_group_ids = {group.group_id for group in updated}
    added_by_key: dict[str, int] = {}
    shifted_by_key: dict[str, int] = {}

    def bucket_key(key: tuple[str, str, str, str]) -> tuple[str, str, str]:
        voyage_id, flow, _port, size = key
        return voyage_id, flow, _medium_doc_floor_big_size(size)

    def increase_key(key: tuple[str, str, str, str], qty: int) -> None:
        if qty <= 0:
            return
        if group_indices_by_key.get(key):
            index = group_indices_by_key[key][0]
            updated[index] = replace(updated[index], demand=updated[index].demand + qty)
        else:
            voyage_id, flow, port, size = key
            group = BoxGroup(
                group_id=_next_medium_group_id(voyage_id, used_group_ids),
                voyage_id=voyage_id,
                size=size,
                height="UNK",
                status=flow,
                port=port,
                operator="UNK",
                ctype="UNK",
                weight_class="UNK",
                reefer=False,
                dangerous=False,
                over_limit=False,
                special_codes=(),
                demand=qty,
            )
            group_indices_by_key[key].append(len(updated))
            updated.append(group)
        medium_by_key[key] += qty

    def reduce_key(key: tuple[str, str, str, str], qty: int) -> None:
        remaining = max(0, int(qty))
        if remaining <= 0:
            return
        for index in sorted(group_indices_by_key.get(key, []), key=lambda idx: updated[idx].demand):
            if remaining <= 0:
                break
            current = max(0, int(updated[index].demand))
            if current <= 0:
                continue
            take = min(current, remaining)
            updated[index] = replace(updated[index], demand=current - take)
            remaining -= take
        if remaining > 0:
            raise ValueError(f"cannot shift {qty} boxes from medium group {'|'.join(key)}")
        medium_by_key[key] -= qty

    def donor_keys_for(key: tuple[str, str, str, str]) -> list[tuple[str, str, str, str]]:
        target_bucket = bucket_key(key)
        donors = [
            donor_key
            for donor_key, qty in medium_by_key.items()
            if donor_key != key
            and bucket_key(donor_key) == target_bucket
            and qty > small_by_key.get(donor_key, 0)
        ]
        return sorted(
            donors,
            key=lambda donor_key: (
                medium_by_key[donor_key] - small_by_key.get(donor_key, 0),
                medium_by_key[donor_key],
                donor_key,
            ),
        )

    for key, small_qty in sorted(small_by_key.items()):
        missing = int(small_qty - medium_by_key.get(key, 0))
        if missing <= 0:
            continue
        shifted = 0
        for donor_key in donor_keys_for(key):
            if missing <= 0:
                break
            surplus = int(medium_by_key[donor_key] - small_by_key.get(donor_key, 0))
            take = min(missing, max(0, surplus))
            if take <= 0:
                continue
            reduce_key(donor_key, take)
            increase_key(key, take)
            shifted += take
            missing -= take
        if shifted > 0:
            shifted_by_key["|".join(key)] = shifted_by_key.get("|".join(key), 0) + shifted
        if missing > 0:
            increase_key(key, missing)
            added_by_key["|".join(key)] = added_by_key.get("|".join(key), 0) + missing

    return [group for group in updated if group.demand > 0], added_by_key, shifted_by_key


def _medium_doc_floor_big_size(size: str) -> str:
    return "40" if _size_mode(size) == "45" else _size_mode(size)


def _medium_small_coarse_key(voyage_id: str, flow: str, port: str, size: str) -> tuple[str, str, str, str]:
    return (_voyage(voyage_id), _flow(flow), _norm(port, "UNK"), _size_mode(size))


def _next_medium_group_id(voyage_id: str, used_group_ids: set[str]) -> str:
    index = 1
    while True:
        group_id = f"{voyage_id}_P{index:03d}"
        if group_id not in used_group_ids:
            used_group_ids.add(group_id)
            return group_id
        index += 1


def read_user_area_constraints(
    data_dir: str | Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, list[str]]]]:
    """Read the same voyage-area controls used by the large plan."""
    if not has_input_json(data_dir):
        return {}, {}, {}, {}

    allowlist: defaultdict[str, set[str]] = defaultdict(set)
    blocklist: defaultdict[str, set[str]] = defaultdict(set)
    requirements: defaultdict[str, set[str]] = defaultdict(set)

    def as_area_set(value: object) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            parts = re.split(r"[,，;；\s]+", value)
        elif isinstance(value, dict):
            parts = value.values()
        else:
            try:
                parts = list(value)  # type: ignore[arg-type]
            except TypeError:
                parts = [value]
        return {_norm(part) for part in parts if _norm(part)}

    def apply_large_plan_record(voyage_id: object, record: object) -> None:
        voyage = _voyage(voyage_id)
        if not voyage:
            return
        if not isinstance(record, dict):
            return
        requirements[voyage].update(as_area_set(record.get("add")))
        blocklist[voyage].update(as_area_set(record.get("remove")))

    if bool(input_value(data_dir, "user_design", False)):
        design_areas = as_area_set(input_value(data_dir, "user_design_large_plan_area", None))
        if design_areas:
            for voyage_id in vessel_container_ids(data_dir):
                voyage = _voyage(voyage_id)
                if voyage:
                    allowlist[voyage].update(design_areas)

    def apply_large_adjust_payload(payload: object) -> None:
        if not isinstance(payload, dict):
            return
        source = payload.get("adjust_plan_info", payload)
        if not isinstance(source, dict):
            return
        large_plan = source.get("large_plan", source)
        if not isinstance(large_plan, dict):
            return
        if "add" in large_plan or "remove" in large_plan:
            for voyage_id in vessel_container_ids(data_dir):
                apply_large_plan_record(voyage_id, large_plan)
            return
        for voyage_id, record in large_plan.items():
            apply_large_plan_record(voyage_id, record)

    apply_large_adjust_payload(input_value(data_dir, "adjust_plan_info", None))

    summary: dict[str, dict[str, list[str]]] = {}
    voyages = set(allowlist) | set(blocklist) | set(requirements)
    for voyage in sorted(voyages):
        if allowlist[voyage]:
            allowlist[voyage].difference_update(blocklist[voyage])
        requirements[voyage].difference_update(blocklist[voyage])
        summary[voyage] = {
            "only_areas": sorted(allowlist[voyage]),
            "forbidden_areas": sorted(blocklist[voyage]),
            "required_areas": sorted(requirements[voyage]),
        }
    return dict(allowlist), dict(blocklist), dict(requirements), summary


def _canonical_attribute_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return ATTRIBUTE_ALIASES.get(text, text)


def _canonical_attribute_tuple(raw: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        raw_items = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        raw_items = list(raw)
    else:
        return default
    out: list[str] = []
    for item in raw_items:
        attr = _canonical_attribute_name(item)
        if attr and attr not in out:
            out.append(attr)
    return tuple(out) if out else default


def _raw_attribute_value(raw: object, *keys: str) -> object:
    if not isinstance(raw, dict):
        return None
    for key in keys:
        if key in raw:
            return raw[key]
    return None


def _canonical_weight_levels(raw: object, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        raw_items = re.split(r"[,|;/\s]+", raw.strip())
    elif isinstance(raw, (list, tuple, set)):
        raw_items = list(raw)
    else:
        return default
    levels: list[int] = []
    for item in raw_items:
        try:
            levels.append(int(round(float(item))))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(levels))) if levels else default


def _voyage_rule_map(raw: object, voyages: list[str], default: tuple, canonicalizer) -> dict[str, tuple]:
    if isinstance(raw, dict):
        out = {
            _voyage(voyage): canonicalizer(value, default)
            for voyage, value in raw.items()
            if _voyage(voyage)
        }
    elif raw is None:
        out = {}
    else:
        shared = canonicalizer(raw, default)
        out = {_voyage(voyage): shared for voyage in voyages if _voyage(voyage)}
    for voyage in voyages:
        normalized = _voyage(voyage)
        if normalized:
            out.setdefault(normalized, canonicalizer(None, default))
    return out


def read_medium_small_bay_controls(
    data_dir: str | Path,
    groups: list[BoxGroup],
    small_groups: list[SmallBoxGroup],
    bays: dict[str, Bay],
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[dict[str, object]], dict[str, object]]:
    required: defaultdict[str, set[str]] = defaultdict(set)
    blocked: defaultdict[str, set[str]] = defaultdict(set)
    rule_records: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "matched_rules": 0,
        "matched_groups": 0,
        "unknown_bays": [],
        "ignored_rules": 0,
    }
    by_voyage: defaultdict[str, list[object]] = defaultdict(list)
    for group in list(groups) + list(small_groups):
        by_voyage[_voyage(getattr(group, "voyage_id", ""))].append(group)
    adjust_plan_info = input_value(data_dir, "adjust_plan_info", None)
    for plan_level in ("medium_plan", "small_plan"):
        for voyage_id, rules in _plan_adjust_rules(adjust_plan_info, plan_level).items():
            normalized_voyage = _voyage(voyage_id)
            if isinstance(rules, dict):
                rules = [rules]
            if not isinstance(rules, (list, tuple)):
                summary["ignored_rules"] = int(summary["ignored_rules"]) + 1
                continue
            for rule in rules:
                if not isinstance(rule, dict):
                    summary["ignored_rules"] = int(summary["ignored_rules"]) + 1
                    continue
                attr_filter = _canonical_adjust_attributes(rule.get("attribute", {}))
                add_bays, add_unknown = _canonical_adjust_bays(rule.get("add"), bays)
                remove_bays, remove_unknown = _canonical_adjust_bays(rule.get("remove"), bays)
                rule_records.append(
                    {
                        "plan_level": plan_level,
                        "voyage_id": normalized_voyage,
                        "attributes": dict(attr_filter),
                        "required_bays": set(add_bays),
                        "blocked_bays": set(remove_bays),
                    }
                )
                matched = [
                    group
                    for group in by_voyage.get(normalized_voyage, [])
                    if _group_matches_adjust_attributes(group, attr_filter)
                ]
                if not matched:
                    continue
                summary["matched_rules"] = int(summary["matched_rules"]) + 1
                summary["matched_groups"] = int(summary["matched_groups"]) + len(matched)
                for group in matched:
                    required[group.group_id].update(add_bays)
                    blocked[group.group_id].update(remove_bays)
                unknown_bays = summary.setdefault("unknown_bays", [])
                if isinstance(unknown_bays, list):
                    for item in add_unknown + remove_unknown:
                        unknown_bays.append({"plan_level": plan_level, "voyage_id": normalized_voyage, "bay": item})
    cleaned_required = {group_id: values - blocked.get(group_id, set()) for group_id, values in required.items()}
    cleaned_required = {group_id: values for group_id, values in cleaned_required.items() if values}
    cleaned_blocked = {group_id: values for group_id, values in blocked.items() if values}
    summary["required_group_count"] = len(cleaned_required)
    summary["blocked_group_count"] = len(cleaned_blocked)
    summary["required_bay_count"] = sum(len(values) for values in cleaned_required.values())
    summary["blocked_bay_count"] = sum(len(values) for values in cleaned_blocked.values())
    return cleaned_required, cleaned_blocked, rule_records, summary


def _plan_adjust_rules(adjust_plan_info: object, plan_level: str) -> dict:
    if not isinstance(adjust_plan_info, dict):
        return {}
    source = adjust_plan_info.get("adjust_plan_info", adjust_plan_info)
    if not isinstance(source, dict):
        return {}
    rules = source.get(plan_level, {})
    return rules if isinstance(rules, dict) else {}


def _canonical_adjust_attributes(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        attr = _canonical_attribute_name(key)
        out[attr] = _normalize_adjust_attr_value(attr, value)
    return out


def _group_matches_adjust_attributes(group: object, attr_filter: dict[str, str]) -> bool:
    for attr, expected in attr_filter.items():
        actual = _normalize_adjust_attr_value(attr, getattr(group, attr, ""))
        if actual != expected:
            return False
    return True


def _normalize_adjust_attr_value(attr: str, value: object) -> str:
    if attr in {"voyage", "voyage_id"}:
        return _voyage(value)
    if attr in {"size", "size_mode"}:
        return _size_mode(value)
    if attr in {"status", "flow"}:
        return _flow(value)
    if attr == "pre_stow":
        return "1" if bool(value) and str(value).strip().lower() not in {"0", "false", "no", ""} else "0"
    return _norm(value)


def _canonical_adjust_bays(raw: object, bays: dict[str, Bay]) -> tuple[set[str], list[str]]:
    out: set[str] = set()
    unknown: list[str] = []
    if raw is None:
        return out, unknown
    if isinstance(raw, dict):
        items = [(_norm(area), values) for area, values in raw.items()]
    else:
        items = [("", raw)]
    for area_no, values in items:
        bay_values = values if isinstance(values, (list, tuple, set)) and not isinstance(values, (str, bytes)) else [values]
        for value in bay_values:
            key = _canonical_adjust_bay_key(area_no, value, bays)
            if key:
                out.add(key)
            else:
                unknown.append(f"{area_no}-{value}" if area_no else str(value))
    return out, unknown


def _canonical_adjust_bay_key(area_no: str, bay_value: object, bays: dict[str, Bay]) -> str:
    text = _norm(bay_value)
    if not text:
        return ""
    candidates = []
    if "-" in text:
        left, right = text.split("-", 1)
        candidates.append(f"{_norm(left)}-{_norm(right)}")
    if "|" in text:
        left, right = text.split("|", 1)
        candidates.append(f"{_norm(left)}-{_norm(right)}")
    if area_no:
        candidates.append(f"{area_no}-{text}")
    candidates.append(text)
    for candidate in candidates:
        if candidate in bays:
            return candidate
    return ""


def read_attribute_rules(data_dir: str | Path) -> AttributeRules:
    if not has_input_json(data_dir):
        return AttributeRules()
    raw = input_value(data_dir, "attribute_rules", None)
    if raw is None:
        raw = input_value(data_dir, "planning_attribute_rules", None)
    if raw is None:
        raw = input_value(data_dir, "grouping_rules", None)
    defaults = AttributeRules()
    voyages = [_voyage(voyage) for voyage in vessel_container_ids(data_dir) if _voyage(voyage)]
    rough_attr = input_value(data_dir, "rough_attr", None)
    detail_attr = input_value(data_dir, "detail_attr", None)
    bay_rules = input_value(data_dir, "bay_rules", None)
    row_rules = input_value(data_dir, "row_rules", None)
    weight_level = input_value(data_dir, "weight_level", None)
    return AttributeRules(
        coarse_group_attributes=_canonical_attribute_tuple(
            _raw_attribute_value(raw, "coarse_group_attributes", "medium_group_attributes", "coarse_attributes") or rough_attr,
            defaults.coarse_group_attributes,
        ),
        fine_group_attributes=_canonical_attribute_tuple(
            _raw_attribute_value(raw, "fine_group_attributes", "small_group_attributes", "fine_attributes") or detail_attr,
            defaults.fine_group_attributes,
        ),
        bay_no_mix_attributes=_canonical_attribute_tuple(
            _raw_attribute_value(raw, "bay_no_mix_attributes", "no_mix_bay_attributes", "bay_attributes") or bay_rules,
            defaults.bay_no_mix_attributes,
        ),
        row_no_mix_attributes=_canonical_attribute_tuple(
            _raw_attribute_value(raw, "row_no_mix_attributes", "stack_no_mix_attributes", "no_mix_row_attributes", "no_mix_stack_attributes", "row_attributes") or row_rules,
            defaults.row_no_mix_attributes,
        ),
        weight_levels=_canonical_weight_levels(weight_level, defaults.weight_levels),
        coarse_group_attributes_by_voyage=_voyage_rule_map(rough_attr, voyages, defaults.coarse_group_attributes, _canonical_attribute_tuple),
        fine_group_attributes_by_voyage=_voyage_rule_map(detail_attr, voyages, defaults.fine_group_attributes, _canonical_attribute_tuple),
        bay_no_mix_attributes_by_voyage=_voyage_rule_map(bay_rules, voyages, defaults.bay_no_mix_attributes, _canonical_attribute_tuple),
        row_no_mix_attributes_by_voyage=_voyage_rule_map(row_rules, voyages, defaults.row_no_mix_attributes, _canonical_attribute_tuple),
        weight_levels_by_voyage=_voyage_rule_map(weight_level, voyages, defaults.weight_levels, _canonical_weight_levels),
    )


def _small_groupby_columns(attribute_rules: AttributeRules | None, voyage_id: object = None) -> tuple[str, ...]:
    if attribute_rules is None:
        return DEFAULT_FINE_GROUP_COLUMNS
    attrs = list(attribute_rules.fine_for(voyage_id) if voyage_id is not None and hasattr(attribute_rules, "fine_for") else attribute_rules.fine_group_attributes)
    attrs.extend(attribute_rules.bay_no_mix_for(voyage_id) if voyage_id is not None and hasattr(attribute_rules, "bay_no_mix_for") else attribute_rules.bay_no_mix_attributes)
    attrs.extend(attribute_rules.row_no_mix_for(voyage_id) if voyage_id is not None and hasattr(attribute_rules, "row_no_mix_for") else attribute_rules.row_no_mix_attributes)
    columns = ["status", "size", "port"]
    for attr in attrs:
        column = SMALL_GROUP_COLUMN_BY_ATTR.get(attr)
        if column and column not in columns:
            columns.append(column)
    return tuple(columns)


def _medium_groupby_columns(attribute_rules: AttributeRules | None, voyage_id: object = None) -> tuple[str, ...]:
    if attribute_rules is None:
        return DEFAULT_MEDIUM_GROUP_COLUMNS
    attrs = list(
        attribute_rules.coarse_for(voyage_id)
        if voyage_id is not None and hasattr(attribute_rules, "coarse_for")
        else attribute_rules.coarse_group_attributes
    )
    columns = ["status", "size"]
    for attr in attrs:
        column = SMALL_GROUP_COLUMN_BY_ATTR.get(attr)
        if column == "weight":
            column = "weight_class"
        if column and column not in columns:
            columns.append(column)
    return tuple(columns)


def _current_yard_container_keys(data_path: Path, planning_time: datetime) -> tuple[set[str], set[str]]:
    columns = ["HAS_CONTAINER", "IYC_CNTRID", "IYC_CNTRNO", "IYC_INYTM"]
    if has_input_json(data_path):
        frame = input_dataframe(data_path, "bay_slots_detail", columns=columns)
    else:
        path = data_path / "bay_slots_detail.parquet"
        if not path.exists():
            return set(), set()
        try:
            frame = pd.read_parquet(path, columns=columns)
        except (KeyError, ValueError):
            frame = pd.read_parquet(path)
    if frame.empty or "HAS_CONTAINER" not in frame.columns:
        return set(), set()
    occupied = frame.loc[frame["HAS_CONTAINER"].fillna(0).astype(int) == 1].copy()
    if occupied.empty:
        return set(), set()
    if "IYC_INYTM" in occupied.columns:
        in_time = pd.to_datetime(occupied["IYC_INYTM"], errors="coerce")
        occupied = occupied.loc[in_time.isna() | (in_time <= planning_time)]
    ids = set()
    numbers = set()
    if "IYC_CNTRID" in occupied.columns:
        ids = {_norm(value) for value in occupied["IYC_CNTRID"] if _norm(value)}
    if "IYC_CNTRNO" in occupied.columns:
        numbers = {_norm(value) for value in occupied["IYC_CNTRNO"] if _norm(value)}
    return ids, numbers


def _is_pre_stow_group(row: dict, size: str, port: str, height: str, weight: str) -> bool:
    # Reserved extension point: the current 20260508 data has no explicit
    # pre-stow marker. Keep this hook so a future field can map into the
    # small-plan soft isolation preference without changing solver code.
    return False


def _special_stow_code(row: dict) -> str:
    # Reserved extension point: future data should provide an explicit fixed
    # special-stow marker. The current dataset has no such field, so no
    # inferred special-stow category is produced here.
    return _explicit_special_stow_code(row)


def _business_special_codes(row: dict) -> set[str]:
    code = _explicit_special_stow_code(row)
    return {code} if code else set()


def _explicit_special_stow_code(row: dict) -> str:
    return ""


def _weight_class(value: object, levels: tuple[int, ...] = (0, 10, 15, 20, 25, 30)) -> str:
    if value is None or pd.isna(value):
        return "UNK"
    try:
        tons = float(value) / 1000.0
    except (TypeError, ValueError):
        return "UNK"
    ordered = sorted(set(int(level) for level in levels)) or [0, 10, 15, 20, 25, 30]
    bands = list(zip(ordered, ordered[1:]))
    for lower, upper in bands:
        if lower <= tons < upper:
            return f"{lower}_{upper}"
    if tons >= ordered[-1]:
        return f"GT{ordered[-1]}"
    return "UNK"


def _largest_remainder_scale(counts: list[int], source_total: int, target_total: int) -> list[int]:
    if target_total <= 0:
        return [0 for _ in counts]
    if source_total <= 0:
        out = [0 for _ in counts]
        out[0] = target_total
        return out
    raw = [c * target_total / source_total for c in counts]
    base = [int(math.floor(v)) for v in raw]
    remain = target_total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:remain]:
        base[i] += 1
    return base


def _allocate_by_weights(weights: dict[str, int], target_total: int) -> dict[str, int]:
    items = [(area, qty) for area, qty in sorted(weights.items()) if qty > 0]
    if not items:
        return {}
    scaled = _largest_remainder_scale([qty for _, qty in items], sum(qty for _, qty in items), target_total)
    return {area: qty for (area, _), qty in zip(items, scaled) if qty > 0}


def _first_existing(fieldnames: set[str], candidates: list[str]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _date_key(value: str) -> str:
    if not value:
        return ""
    parsed = parse_datetime(value)
    if parsed is not None:
        return parsed.date().isoformat()
    text = str(value).strip().replace("/", "-")
    if len(text) >= 10:
        return text[:10]
    return text


def _big_plan_size_mode(value: object) -> str:
    """杞崲涓哄ぇ璁″垝灏哄鍙ｅ緞銆?
    澶ц鍒掑彧鍖哄垎 20 鍜?40锛屽叾涓?45 灏哄綊鍏?40 灏恒€傚鏋滄棫鏂囦欢娌℃湁灏哄鍒楋紝
    杩斿洖 `ALL`锛屽悗缁細閫€鍥炲埌鍙寜鑸-绠卞尯鎬婚噺绾︽潫銆?    """
    size = _norm(value).upper()
    if not size:
        return "ALL"
    if size in {"20", "40", "45"}:
        return size
    return "40"


def _flow(value: object) -> str:
    return _norm(value, "OF").upper() or "OF"


def _generic_box_key(voyage_id: str, size_mode: str) -> tuple:
    """鏋勯€犲厹搴曞睘鎬х粍銆?
    褰撴煇鑸鏈夊ぇ璁″垝閰嶉锛屼絾绠辨槑缁嗕腑鎵句笉鍒板搴斿昂瀵哥殑绠卞瓙鏃讹紝鐢ㄤ竴涓櫘閫?    灞炴€х粍鎵挎帴杩欓儴鍒嗚鍒掗噺锛屼繚璇佹ā鍨嬩粛鑳界粰鍑哄彲妫€鏌ョ殑缁撴灉銆?    """
    return (voyage_id, size_mode, "UNK", "OF", "UNK", "UNK", "GP", "UNK", False, False, False, ())


def _append_scaled_groups(
    groups: list[BoxGroup],
    raw: list[tuple[tuple, int]],
    target: int,
    start_index: int,
) -> int:
    """鎶婂師濮嬪睘鎬х粍鎸夌洰鏍囩閲忕缉鏀惧悗杩藉姞鍒?`groups`銆?
    杩斿洖涓嬩竴涓彲鐢ㄧ殑缁勭紪鍙凤紝渚夸簬鍚屼竴鑸鎸?20/40 鍒嗘《缂╂斁鍚庝粛淇濇寔缁勫彿鍞竴銆?    """
    total_raw = sum(c for _, c in raw)
    scaled_counts = _largest_remainder_scale([c for _, c in raw], total_raw, target)
    group_index = start_index
    for key, demand in zip((item[0] for item in raw), scaled_counts):
        if demand <= 0:
            continue
        v, size, height, status, port, operator, ctype, weight_class, reefer, dangerous, over_limit, special_codes = key
        groups.append(
            BoxGroup(
                group_id=f"{v}_G{group_index:03d}",
                voyage_id=v,
                size=size,
                height=height,
                status=status,
                port=port,
                operator=operator,
                ctype=ctype,
                weight_class=weight_class,
                reefer=reefer,
                dangerous=dangerous,
                over_limit=over_limit,
                special_codes=special_codes,
                demand=demand,
            )
        )
        group_index += 1
    return group_index


def build_bays(
    data_dir: str | Path,
    allowed_areas: set[str],
    closed_areas: set[str],
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    vessel_schedules: dict[str, VoyageSchedule] | None = None,
    target_voyages: set[str] | None = None,
    area_functions: dict[str, set[str]] | None = None,
    misplaced_bay_exclusion_ratio: float = DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
) -> dict[str, Bay]:
    data_path = Path(data_dir)
    base = input_dataframe(data_path, "bay_slots_detail") if has_input_json(data_path) else pd.read_parquet(data_path / "bay_slots_detail.parquet")
    original_bay_capacity = _total_slots_by_bay(base)
    base, reserved_slots = apply_tops_reservations(base, data_path, planning_time, target_voyages or set())
    vessel_schedules = vessel_schedules or read_vessel_schedules(data_dir)
    area_functions = area_functions or read_area_functions(data_dir)
    excluded_bays = _misplaced_bays_to_exclude(
        base,
        area_functions,
        vessel_schedules,
        planning_time,
        misplaced_bay_exclusion_ratio,
        original_bay_capacity,
        target_voyages or set(),
    )
    if excluded_bays:
        base = _drop_bays(base, excluded_bays)
    cap_by_size = _capacity_by_bay_size(base)
    row_cap_by_size = _capacity_by_bay_row_size(base)
    physical_cap = _physical_capacity_by_bay(base)
    row_physical_cap = _physical_capacity_by_bay_row(base)
    row_cap_by_bay_size = {
        size_mode: _row_capacity_index_by_bay(row_cap_by_size[size_mode])
        for size_mode in SIZE_MODES
    }
    row_physical_by_bay = _row_capacity_index_by_bay(row_physical_cap)
    released_cap = _released_capacity_by_bay(base, vessel_schedules, planning_time)
    for size_mode in SIZE_MODES:
        for key, count in released_cap[size_mode].items():
            cap_by_size[size_mode][key] = cap_by_size[size_mode].get(key, 0) + count
    for key, count in released_cap["physical"].items():
        physical_cap[key] = physical_cap.get(key, 0) + count

    bay_numbers = (
        base[["YAA_AREANO", "YBY_BAYNO"]]
        .drop_duplicates()
        .assign(
            YAA_AREANO=lambda x: _norm_series(x["YAA_AREANO"]),
            YBY_BAYNO=lambda x: _bay_no_series(x["YBY_BAYNO"]),
        )
        .drop_duplicates()
    )
    bay_order: dict[tuple[str, str], int] = {}
    block_by_bay: dict[tuple[str, str], tuple[int, tuple[str, ...], bool]] = {}
    large_bay_partner: dict[tuple[str, str], str] = {}
    for area_no, area_df in bay_numbers.groupby("YAA_AREANO"):
        ordered = sorted(area_df["YBY_BAYNO"].tolist(), key=_bay_sort_key)
        for idx, bay_no in enumerate(ordered):
            bay_order[(area_no, bay_no)] = idx
        large_bay_partner.update(_apply_large_bay_pair_capacities(area_no, ordered, cap_by_size, physical_cap))
        big_bay_starts = {
            bay_no
            for bay_no in ordered
            if cap_by_size["40"].get((area_no, bay_no), 0) > 0 or cap_by_size["45"].get((area_no, bay_no), 0) > 0
        }
        for block_index, members, adjusted in _make_yard_blocks(ordered, big_bay_starts):
            for bay_no in members:
                block_by_bay[(area_no, bay_no)] = (block_index, members, adjusted)

    bays: dict[str, Bay] = {}
    for (area_no, bay_no), order in bay_order.items():
        if area_no in closed_areas or area_no not in allowed_areas:
            continue
        block_index, block_members, block_adjusted = block_by_bay.get((area_no, bay_no), (order + 1, (bay_no,), False))
        bay_key = f"{area_no}-{bay_no}"
        bays[bay_key] = Bay(
            area_no=area_no,
            bay_no=bay_no,
            bay_key=bay_key,
            block_id=f"{area_no}-B{block_index:02d}",
            block_bays=block_members,
            block_bay_count=len(block_members),
            block_boundary_adjusted=block_adjusted,
            bay_order=order,
            cap_by_size={
                size_mode: cap_by_size[size_mode].get((area_no, bay_no), 0)
                for size_mode in SIZE_MODES
            },
            physical_capacity=physical_cap.get((area_no, bay_no), 0),
            row_cap_by_size={
                size_mode: row_cap_by_bay_size[size_mode].get((area_no, bay_no), {})
                for size_mode in SIZE_MODES
            },
            row_physical_capacity=row_physical_by_bay.get((area_no, bay_no), {}),
            large_bay_partner_no=large_bay_partner.get((area_no, bay_no), ""),
            large_bay_partner_key=(
                f"{area_no}-{large_bay_partner[(area_no, bay_no)]}"
                if (area_no, bay_no) in large_bay_partner
                else ""
            ),
        )
    build_bays.last_reserved_count = len(reserved_slots)
    build_bays.last_tops_closed_bay_count = getattr(apply_tops_reservations, "last_reserved_bay_count", 0)
    build_bays.last_misplaced_excluded_bay_count = len(excluded_bays)

    _populate_existing_bay_attributes(base, bays, vessel_schedules, planning_time)

    return bays


def build_area_operations(
    data_dir: str | Path,
    vessel_schedules: dict[str, VoyageSchedule],
) -> dict[str, list[AreaOperation]]:
    columns = ["HAS_CONTAINER", "YAA_AREANO", "IYC_EVOY_ID"]
    data_path = Path(data_dir)
    base = input_dataframe(data_path, "bay_slots_detail", columns=columns) if has_input_json(data_path) else pd.read_parquet(
        data_path / "bay_slots_detail.parquet",
        columns=columns,
    )
    occupied = base.loc[base["HAS_CONTAINER"] == 1, ["YAA_AREANO", "IYC_EVOY_ID"]].copy()
    if occupied.empty:
        return {}
    occupied["IYC_EVOY_ID"] = _voyage_series(occupied["IYC_EVOY_ID"])
    occupied["YAA_AREANO"] = _norm_series(occupied["YAA_AREANO"])
    occupied = occupied[(occupied["IYC_EVOY_ID"] != "") & (occupied["YAA_AREANO"] != "")]
    seen: set[tuple[str, str]] = set()
    operations: defaultdict[str, list[AreaOperation]] = defaultdict(list)
    for area_no, voyage_id in occupied.drop_duplicates().itertuples(index=False, name=None):
        schedule = vessel_schedules.get(voyage_id)
        if schedule is None:
            continue
        key = (area_no, voyage_id)
        if key in seen:
            continue
        seen.add(key)
        operations[area_no].append(
            AreaOperation(
                area_no=area_no,
                voyage_id=voyage_id,
                start_time=schedule.berth_time,
                end_time=schedule.departure_time,
            )
        )
    for items in operations.values():
        items.sort(key=lambda item: (item.start_time, item.end_time, item.voyage_id))
    return dict(operations)


def _populate_existing_bay_attributes(
    base: pd.DataFrame,
    bays: dict[str, Bay],
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> None:
    columns = [
        "YAA_AREANO",
        "YBY_BAYNO",
        "YST_ROWNO",
        "IYC_EVOY_ID",
        "IYC_CSZ_CSIZECD",
        "IYC_CHEIGHTCD",
        "IYC_POT_UNLDPORT",
        "IYC_STS_CSTATUSCD",
        "IYC_CTYPECD",
        "IYC_SETTMPT",
    ]
    occupied = base.loc[base["HAS_CONTAINER"] == 1, columns].copy()
    if occupied.empty:
        return

    active_voyages = {
        voyage_id
        for voyage_id, schedule in vessel_schedules.items()
        if schedule.departure_time > planning_time
    }
    known_voyages = set(vessel_schedules)
    occupied["_voyage"] = _voyage_series(occupied["IYC_EVOY_ID"])
    occupied = occupied.loc[
        occupied["_voyage"].eq("")
        | ~occupied["_voyage"].isin(known_voyages)
        | occupied["_voyage"].isin(active_voyages)
    ]
    if occupied.empty:
        return

    occupied["_area"] = _norm_series(occupied["YAA_AREANO"])
    occupied["_bay"] = _bay_no_series(occupied["YBY_BAYNO"])
    occupied["_row"] = _row_no_series(occupied["YST_ROWNO"])
    occupied["_bay_key"] = occupied["_area"] + "-" + occupied["_bay"]
    occupied = occupied.loc[occupied["_bay_key"].isin(bays)]
    if occupied.empty:
        return

    occupied["_size"] = occupied["IYC_CSZ_CSIZECD"].map(_size_mode)
    occupied["_height"] = _norm_series(occupied["IYC_CHEIGHTCD"])
    occupied["_port"] = _norm_series(occupied["IYC_POT_UNLDPORT"])
    occupied["_status"] = _norm_series(occupied["IYC_STS_CSTATUSCD"])
    occupied["_ctype"] = _norm_series(occupied["IYC_CTYPECD"])

    for bay_key, values in occupied.groupby("_bay_key", sort=False):
        bay = bays.get(str(bay_key))
        if bay is None:
            continue
        bay.existing_size_modes.update(str(item) for item in values["_size"].dropna().unique() if str(item))
        bay.existing_heights.update(str(item) for item in values["_height"].dropna().unique() if str(item))
        bay.existing_ports.update(str(item) for item in values["_port"].dropna().unique() if str(item))
        for attr, column in {
            "voyage_id": "_voyage",
            "status": "_status",
            "size": "_size",
            "height": "_height",
            "port": "_port",
            "ctype": "_ctype",
        }.items():
            attr_values = {str(item) for item in values[column].dropna().unique() if str(item)}
            if attr_values:
                bay.existing_attrs.setdefault(attr, set()).update(attr_values)
        for row_no, row_values in values.groupby("_row", sort=False):
            row_key = str(row_no)
            if not row_key:
                continue
            ports = {str(item) for item in row_values["_port"].dropna().unique() if str(item)}
            if ports:
                bay.existing_ports_by_row.setdefault(row_key, set()).update(ports)
            for attr, column in {
                "voyage_id": "_voyage",
                "status": "_status",
                "size": "_size",
                "height": "_height",
                "port": "_port",
                "ctype": "_ctype",
            }.items():
                attr_values = {str(item) for item in row_values[column].dropna().unique() if str(item)}
                if attr_values:
                    bay.existing_attrs_by_row.setdefault(row_key, {}).setdefault(attr, set()).update(attr_values)

        statuses = values["_status"].dropna().astype(str)
        if statuses.str.endswith("E").any() or statuses.isin(["IE", "OE", "TE", "RE"]).any():
            bay.fallback_reasons.add("existing_empty_container")
        ctypes = values["_ctype"].dropna().astype(str)
        if ctypes.eq("RF").any() or values["IYC_SETTMPT"].notna().any():
            bay.fallback_reasons.add("existing_reefer_container")


def apply_tops_reservations(
    base: pd.DataFrame,
    data_path: Path,
    planning_time: datetime,
    target_voyages: set[str],
) -> tuple[pd.DataFrame, set[tuple[str, str, str]]]:
    if has_input_json(data_path):
        tops = input_dataframe(data_path, "tops_plan")
    else:
        path = data_path / "tops_plan_info.parquet"
        if not path.exists():
            apply_tops_reservations.last_reserved_bay_count = 0
            return base, set()
        tops = pd.read_parquet(path)
    reserved_bays: set[tuple[str, str]] = set()
    reserved_slots: set[tuple[str, str, str]] = set()
    bay_rows = (
        base[["YAA_AREANO", "YBY_BAYNO", "YST_ROWNO"]]
        .drop_duplicates()
        .assign(
            YAA_AREANO=lambda x: _norm_series(x["YAA_AREANO"]),
            YBY_BAYNO=lambda x: _bay_no_series(x["YBY_BAYNO"]),
            YST_ROWNO=lambda x: _row_no_series(x["YST_ROWNO"]),
        )
    )
    bay_rows = bay_rows[(bay_rows["YAA_AREANO"] != "") & (bay_rows["YBY_BAYNO"] != "")]
    by_area = {
        str(area_no): set(values["YBY_BAYNO"].astype(str))
        for area_no, values in bay_rows.groupby("YAA_AREANO", sort=False)
    }
    slots_by_bay = {
        (str(area_no), str(bay_no)): set(values["YST_ROWNO"].astype(str))
        for (area_no, bay_no), values in bay_rows.groupby(["YAA_AREANO", "YBY_BAYNO"], sort=False)
    }

    tops_cols = ["SPL_ISVALID", "SPR_ISVALID", "SPL_CONDITIONCODE", "SPL_STDATE", "SPL_EDDATE", "SPR_STBAY", "SPR_EDBAY"]
    for row in tops.loc[:, [col for col in tops_cols if col in tops.columns]].itertuples(index=False):
        row_data = row._asdict()
        if _norm(row_data.get("SPL_ISVALID")).upper() != "Y" or _norm(row_data.get("SPR_ISVALID")).upper() != "Y":
            continue
        voyage_id = _voyage(row_data.get("SPL_CONDITIONCODE"))
        if voyage_id in target_voyages:
            continue
        start = parse_datetime(row_data.get("SPL_STDATE"))
        end = parse_datetime(row_data.get("SPL_EDDATE"))
        if start is None or end is None or not (start <= planning_time <= end):
            continue
        parsed_start = _parse_tops_bay_code(row_data.get("SPR_STBAY"))
        parsed_end = _parse_tops_bay_code(row_data.get("SPR_EDBAY"))
        if parsed_start is None or parsed_end is None or parsed_start[0] != parsed_end[0]:
            continue
        area_no, start_bay = parsed_start
        _, end_bay = parsed_end
        for bay_no in by_area.get(area_no, set()):
            if not _bay_between(bay_no, start_bay, end_bay):
                continue
            reserved_bays.add((area_no, bay_no))
            for row_no in slots_by_bay.get((area_no, bay_no), set()):
                reserved_slots.add((area_no, bay_no, row_no))
    if not reserved_bays:
        apply_tops_reservations.last_reserved_bay_count = 0
        return base, reserved_slots
    apply_tops_reservations.last_reserved_bay_count = len(reserved_bays)
    return _drop_bays(base, reserved_bays), reserved_slots


def _parse_tops_bay_code(value: object) -> tuple[str, str] | None:
    text = _norm(value).upper()
    if len(text) < 3:
        return None
    return text[:-2], _bay_no(text[-2:])


def _bay_between(bay_no: str, start_bay: str, end_bay: str) -> bool:
    try:
        bay = int(bay_no)
        start = int(start_bay)
        end = int(end_bay)
    except ValueError:
        return start_bay <= bay_no <= end_bay
    return min(start, end) <= bay <= max(start, end)


def _row_between(row_no: str, start_row: str, end_row: str) -> bool:
    try:
        row = int(row_no)
        start = int(start_row)
        end = int(end_row)
    except ValueError:
        return True
    return min(start, end) <= row <= max(start, end)


def _released_capacity_by_bay(
    base: pd.DataFrame,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> dict[str, Counter[tuple[str, str]]]:
    released = {"20": Counter(), "40": Counter(), "45": Counter(), "physical": Counter()}
    released_voyages = {
        voyage_id
        for voyage_id, schedule in vessel_schedules.items()
        if schedule.departure_time <= planning_time
    }
    if not released_voyages:
        return released

    cols = ["YAA_AREANO", "YBY_BAYNO", "YBY_ENABLECSIZECD", "IYC_EVOY_ID"]
    occupied = base.loc[base["HAS_CONTAINER"] == 1, cols].copy()
    if occupied.empty:
        return released

    occupied["IYC_EVOY_ID"] = _voyage_series(occupied["IYC_EVOY_ID"])
    occupied = occupied[occupied["IYC_EVOY_ID"].isin(released_voyages)]
    if occupied.empty:
        return released

    occupied["YAA_AREANO"] = _norm_series(occupied["YAA_AREANO"])
    occupied["YBY_BAYNO"] = _bay_no_series(occupied["YBY_BAYNO"])
    occupied = occupied[(occupied["YAA_AREANO"] != "") & (occupied["YBY_BAYNO"] != "")]
    if occupied.empty:
        return released

    physical_counts = occupied.groupby(["YAA_AREANO", "YBY_BAYNO"]).size()
    released["physical"].update({(str(a), str(b)): int(v) for (a, b), v in physical_counts.items()})
    for size_mode in SIZE_MODES:
        counts = occupied.loc[_size_enabled_mask(occupied["YBY_ENABLECSIZECD"], size_mode)].groupby(
            ["YAA_AREANO", "YBY_BAYNO"]
        ).size()
        released[size_mode].update({(str(a), str(b)): int(v) for (a, b), v in counts.items()})
    return released


def _is_occupied_at_planning_time(
    row: dict,
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
) -> bool:
    if row.get("HAS_CONTAINER") != 1:
        return False
    voyage_id = _voyage(row.get("IYC_EVOY_ID"))
    if not voyage_id:
        return True
    schedule = vessel_schedules.get(voyage_id)
    if schedule is None:
        return True
    return schedule.departure_time > planning_time


def _capacity_by_bay_size(base: pd.DataFrame) -> dict[str, dict[tuple[str, str], int]]:
    empty = _normalized_empty_slots(base, include_row=False)
    out: dict[str, dict[tuple[str, str], int]] = {}
    for size_mode in SIZE_MODES:
        counts = empty.loc[_size_enabled_mask(empty["YBY_ENABLECSIZECD"], size_mode)].groupby(
            ["YAA_AREANO", "YBY_BAYNO"]
        ).size()
        out[size_mode] = {(str(a), str(b)): int(v) for (a, b), v in counts.items()}
    return out


def _physical_capacity_by_bay(base: pd.DataFrame) -> dict[tuple[str, str], int]:
    empty = _normalized_empty_slots(base, include_row=False)
    counts = empty.groupby(["YAA_AREANO", "YBY_BAYNO"]).size()
    return {(str(a), str(b)): int(v) for (a, b), v in counts.items()}


def _apply_large_bay_pair_capacities(
    area_no: str,
    ordered_bays: list[str],
    cap_by_size: dict[str, dict[tuple[str, str], int]],
    physical_cap: dict[tuple[str, str], int],
) -> dict[tuple[str, str], str]:
    partner_by_start: dict[tuple[str, str], str] = {}
    original = {
        size_mode: {
            bay_no: int(cap_by_size[size_mode].get((area_no, bay_no), 0))
            for bay_no in ordered_bays
        }
        for size_mode in ("40", "45")
    }
    for size_mode in ("40", "45"):
        for bay_no in ordered_bays:
            cap_by_size[size_mode][(area_no, bay_no)] = 0
    idx = 0
    while idx < len(ordered_bays) - 1:
        left = ordered_bays[idx]
        right = ordered_bays[idx + 1]
        if not _are_consecutive_small_bays(left, right):
            idx += 1
            continue
        left_key = (area_no, left)
        right_key = (area_no, right)
        pair_physical = min(
            int(physical_cap.get(left_key, 0)),
            int(physical_cap.get(right_key, 0)),
        )
        if pair_physical <= 0:
            idx += 2
            continue
        has_large_capacity = False
        for size_mode in ("40", "45"):
            pair_cap = min(original[size_mode].get(left, 0), original[size_mode].get(right, 0), pair_physical)
            cap_by_size[size_mode][left_key] = pair_cap
            has_large_capacity = has_large_capacity or pair_cap > 0
        if has_large_capacity:
            partner_by_start[left_key] = right
        idx += 2
    return partner_by_start


def _are_consecutive_small_bays(left: str, right: str) -> bool:
    try:
        return int(left) + 2 == int(right)
    except ValueError:
        return False


def _capacity_by_bay_row_size(base: pd.DataFrame) -> dict[str, dict[tuple[str, str, str], int]]:
    empty = _normalized_empty_slots(base, include_row=True)
    out: dict[str, dict[tuple[str, str, str], int]] = {}
    for size_mode in SIZE_MODES:
        counts = empty.loc[_size_enabled_mask(empty["YBY_ENABLECSIZECD"], size_mode)].groupby(
            ["YAA_AREANO", "YBY_BAYNO", "YST_ROWNO"]
        ).size()
        out[size_mode] = {(str(a), str(b), str(r)): int(v) for (a, b, r), v in counts.items()}
    return out


def _physical_capacity_by_bay_row(base: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    empty = _normalized_empty_slots(base, include_row=True)
    counts = empty.groupby(["YAA_AREANO", "YBY_BAYNO", "YST_ROWNO"]).size()
    return {(str(a), str(b), str(r)): int(v) for (a, b, r), v in counts.items()}


def _normalized_empty_slots(base: pd.DataFrame, *, include_row: bool) -> pd.DataFrame:
    columns = ["YAA_AREANO", "YBY_BAYNO", "YBY_ENABLECSIZECD"]
    if include_row:
        columns.append("YST_ROWNO")
    empty = base.loc[base["HAS_CONTAINER"] == 0, columns].copy()
    empty["YAA_AREANO"] = _norm_series(empty["YAA_AREANO"])
    empty["YBY_BAYNO"] = _bay_no_series(empty["YBY_BAYNO"])
    if include_row:
        empty["YST_ROWNO"] = _row_no_series(empty["YST_ROWNO"])
    return empty


def _size_enabled_mask(values: pd.Series, size_mode: str) -> pd.Series:
    text = values.astype("string")
    stripped = text.str.strip()
    missing = values.isna() | stripped.str.lower().isin(["", "nan", "none", "<na>"]).fillna(False)
    enabled = stripped.str.contains(rf"(?<!\d){re.escape(size_mode)}(?!\d)", regex=True, na=False)
    return missing | enabled


def _drop_reserved_slots(frame: pd.DataFrame, reserved_slots: set[tuple[str, str, str]]) -> pd.DataFrame:
    if not reserved_slots or frame.empty:
        return frame
    work = frame.copy()
    area = _norm_series(work["YAA_AREANO"])
    bay = _bay_no_series(work["YBY_BAYNO"])
    row = _row_no_series(work["YST_ROWNO"])
    empty = work["HAS_CONTAINER"].fillna(0).astype(int) == 0 if "HAS_CONTAINER" in work.columns else True
    mask = [is_empty and (a, b, r) in reserved_slots for is_empty, a, b, r in zip(empty, area, bay, row)]
    return work.loc[[not item for item in mask]].copy()


def _drop_bays(frame: pd.DataFrame, excluded_bays: set[tuple[str, str]]) -> pd.DataFrame:
    if not excluded_bays or frame.empty:
        return frame
    area = _norm_series(frame["YAA_AREANO"])
    bay = _bay_no_series(frame["YBY_BAYNO"])
    mask = [(a, b) in excluded_bays for a, b in zip(area, bay)]
    return frame.loc[[not item for item in mask]].copy()


def _misplaced_bays_to_exclude(
    base: pd.DataFrame,
    area_functions: dict[str, set[str]],
    vessel_schedules: dict[str, VoyageSchedule],
    planning_time: datetime,
    ratio: float,
    original_bay_capacity: Counter[tuple[str, str]] | None = None,
    target_voyages: set[str] | None = None,
) -> set[tuple[str, str]]:
    if ratio <= 0:
        return set()
    # `base` has already had TOPS reservations removed, so TOPS is handled
    # before misplaced-bay exclusion. The threshold denominator remains the
    # bay's original total slot capacity before TOPS, not the current occupied
    # count and not the TOPS-reduced capacity.
    bay_capacity = original_bay_capacity or _total_slots_by_bay(base)
    target_voyages = {_voyage(v) for v in (target_voyages or set())}
    if not target_voyages:
        return set()

    occupied = base.loc[base["HAS_CONTAINER"] == 1].copy()
    if occupied.empty:
        return set()
    occupied["_voyage"] = _voyage_series(occupied["IYC_EVOY_ID"])
    occupied = occupied.loc[occupied["_voyage"].isin(target_voyages)]
    if occupied.empty:
        return set()
    active_voyages = {
        voyage_id
        for voyage_id in target_voyages
        if voyage_id not in vessel_schedules or vessel_schedules[voyage_id].departure_time > planning_time
    }
    occupied = occupied.loc[occupied["_voyage"].isin(active_voyages)]
    if occupied.empty:
        return set()

    occupied["_area"] = _norm_series(occupied["YAA_AREANO"])
    occupied = occupied.loc[occupied["_area"].isin(area_functions)]
    if occupied.empty:
        return set()

    occupied["_bay"] = _bay_no_series(occupied["YBY_BAYNO"])
    occupied["_flow"] = occupied["IYC_STS_CSTATUSCD"].map(_flow)
    occupied = occupied.loc[occupied["_area"].ne("") & occupied["_bay"].ne("")]
    allowed_pairs = {
        (area_no, flow)
        for area_no, flows in area_functions.items()
        for flow in flows
    }
    area_flow_pairs = pd.MultiIndex.from_frame(occupied[["_area", "_flow"]])
    occupied = occupied.loc[~area_flow_pairs.isin(allowed_pairs)]
    if occupied.empty:
        return set()

    misplaced = Counter(
        {
            (str(area_no), str(bay_no)): int(count)
            for (area_no, bay_no), count in occupied.groupby(["_area", "_bay"], sort=False).size().items()
        }
    )
    return {
        key
        for key, count in misplaced.items()
        if count > bay_capacity.get(key, 0) * ratio
    }


def _total_slots_by_bay(base: pd.DataFrame) -> Counter[tuple[str, str]]:
    slots: Counter[tuple[str, str]] = Counter()
    if base.empty:
        return slots
    work = base[["YAA_AREANO", "YBY_BAYNO"]].copy()
    work["YAA_AREANO"] = _norm_series(work["YAA_AREANO"])
    work["YBY_BAYNO"] = _bay_no_series(work["YBY_BAYNO"])
    work = work[(work["YAA_AREANO"] != "") & (work["YBY_BAYNO"] != "")]
    counts = work.groupby(["YAA_AREANO", "YBY_BAYNO"], sort=False).size()
    slots.update({(str(a), str(b)): int(v) for (a, b), v in counts.items()})
    return slots


def _row_capacity_index_by_bay(capacity: dict[tuple[str, str, str], int]) -> dict[tuple[str, str], dict[str, int]]:
    indexed: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for (area_no, bay_no, row_no), cap in capacity.items():
        if cap > 0:
            indexed[(area_no, bay_no)][row_no] = cap
    return dict(indexed)


def _enabled_size_modes(value: object) -> set[str]:
    text = _norm(value)
    if not text:
        return set(SIZE_MODES)
    modes = set()
    for part in re.findall(r"\d+", text):
        if part in {"20", "40", "45"}:
            modes.add(part)
    return modes


def _slot_allows_size(value: object, required_size: str) -> bool:
    enabled = _enabled_size_modes(value)
    if enabled:
        return required_size in enabled
    return False


def _make_yard_blocks(
    ordered_bays: list[str],
    big_bay_starts: set[str],
    target_bay_count: int = 6,
) -> list[tuple[int, tuple[str, ...], bool]]:
    """鏋勯€犱腑璁″垝浣跨敤鐨勫尯鍧椼€?
    鐩爣鏄瘡涓尯鍧楀寘鍚?6 涓繛缁皬璐濅綅銆傚鏋滆竟鐣屼細鍒囧紑涓€涓?40ft 澶ц礉浣嶅锛?    灏卞悜闄勮繎绉诲姩杈圭晫銆?5ft 鍦ㄦ湰妯″瀷涓寜 40ft 澶勭悊锛屽洜姝や笉鍐嶅崟鐙垽鏂?45ft銆?    """

    blocks: list[tuple[int, tuple[str, ...], bool]] = []
    start = 0
    block_index = 1
    while start < len(ordered_bays):
        if len(ordered_bays) - start <= target_bay_count:
            end = len(ordered_bays)
            adjusted = False
        else:
            target_end = start + target_bay_count
            end = _nearest_safe_block_end(
                ordered_bays,
                big_bay_starts,
                start=start,
                target_end=target_end,
            )
            adjusted = end != target_end
        members = tuple(ordered_bays[start:end])
        blocks.append((block_index, members, adjusted))
        block_index += 1
        start = end
    return blocks


def _nearest_safe_block_end(
    ordered_bays: list[str],
    big_bay_starts: set[str],
    start: int,
    target_end: int,
) -> int:
    # 如果 end-1 位置的小贝位可以作为 40ft 大贝位起点，则边界会切开
    # 推断的大贝位对，属于不安全边界。
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


def _size_mode(value: object) -> str:
    size = _norm(value, "40")
    return size if size in {"20", "40", "45"} else "40"


def _existing_special_signature(row: dict) -> str:
    return _special_stow_code(row) or "NORMAL"


def _bay_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        nums = re.findall(r"\d+", value)
        return (int(nums[0]) if nums else 9999), value


def _bay_no(value: object) -> str:
    """缁熶竴璐濅綅鍙锋牸寮忋€?
    涓嶅悓婧愭枃浠堕噷鍚屼竴涓礉浣嶅彲鑳藉啓鎴?`5`銆乣5.0` 鎴?`05`銆傚閲忕粺璁″拰璐濅綅瀵硅薄
    蹇呴』浣跨敤鍚屼竴涓敭锛屽惁鍒欎細鍑虹幇鈥滄槑鏄庢湁绌轰綅锛屼絾鍖哄潡瀹归噺涓?0鈥濈殑鍋囪薄銆?    """
    text = _norm(value)
    if text.isdigit() and len(text) == 1:
        return f"0{text}"
    return text


def _row_no(value: object) -> str:
    text = _norm(value)
    if text.isdigit() and len(text) == 1:
        return f"0{text}"
    return text


def build_problem(
    data_dir: str | Path,
    big_plan: list[BigPlanRow],
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    horizon_hours: float = 24.0,
    target_voyages: list[str] | None = None,
    misplaced_bay_exclusion_ratio: float = DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
) -> ProblemData:
    """Build medium/small input.

    The big plan determines the active voyage scope and OF area/size pattern.
    Medium demand is calculated independently by the rolling planning-time rule,
    then distributed across the ``new_qty`` pattern inherited from the
    selected voyages' OF big-plan rows.
    """
    closed = read_closed_areas(data_dir)
    area_functions = read_area_functions(data_dir)
    function_areas = set(area_functions)
    area_quota: dict[tuple[str, str, str], int] = {}
    area_size_quota: dict[tuple[str, str, str, str], int] = {}
    assigned_areas: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    cleaned_plan: list[BigPlanRow] = []
    if target_voyages is None:
        target_voyages = infer_target_voyages_from_big_plan(big_plan, planning_time)
    else:
        target_voyages = [_voyage(v) for v in target_voyages]
    if not target_voyages:
        raise ValueError("no target voyages were provided or inferred from the big plan")
    vessel_schedules = read_target_vessel_schedules(data_dir, target_voyages, planning_time, horizon_hours)
    plan_date = planning_time.date().isoformat()
    target_big_plan_flows = {_flow(flow) for flow in DEFAULT_TARGET_BIG_PLAN_FLOWS}
    input_plan = [
        row for row in big_plan
        if row.voyage_id in target_voyages
        and row.flow in target_big_plan_flows
        and (not row.plan_date or row.plan_date == plan_date)
    ]
    allowed_areas = set(function_areas)
    user_area_allowlist, user_area_blocklist, user_area_requirements, user_area_constraint_summary = (
        read_user_area_constraints(data_dir)
    )
    for area_set in [
        *user_area_allowlist.values(),
        *user_area_blocklist.values(),
        *user_area_requirements.values(),
    ]:
        allowed_areas.update(area_set)
    attribute_rules = read_attribute_rules(data_dir)
    for row in input_plan:
        if row.area_no in closed:
            raise ValueError(f"big plan uses closed area {row.area_no} for voyage {row.voyage_id}")
        cleaned_plan.append(row)
        allowed_areas.add(row.area_no)
        assigned_areas[(row.voyage_id, row.flow)].add(row.area_no)
    if not cleaned_plan:
        raise ValueError("no OF big-plan rows remain after target-voyage, date, and closed-area filtering")

    big_plan_caps = medium_demand_caps_from_big_plan(
        big_plan,
        target_voyages,
        planning_time,
        target_big_plan_flows,
    )
    groups, _demand_rows = load_port_demand_groups(
        data_dir,
        target_voyages,
        planning_time,
        horizon_hours,
        big_plan_caps=big_plan_caps,
        attribute_rules=attribute_rules,
    )
    small_groups = load_small_doc_groups(data_dir, target_voyages, planning_time, attribute_rules=attribute_rules)
    groups, medium_doc_floor_added_by_coarse_group, medium_doc_floor_shifted_by_coarse_group = (
        enforce_medium_groups_cover_small_groups(groups, small_groups)
    )
    medium_doc_floor_by_coarse_group = dict(
        Counter(medium_doc_floor_added_by_coarse_group) + Counter(medium_doc_floor_shifted_by_coarse_group)
    )
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
            compatible_plan_flows = target_big_plan_flows if flow in target_big_plan_flows else {flow}
            for size_mode in SIZE_MODES:
                target_qty = demand_by_voyage_size[(voyage_id, flow, size_mode)]
                if target_qty <= 0:
                    continue
                exact_upper: Counter[str] = Counter()
                for (v, f, area_no, size), qty in raw_area_size_quota.items():
                    if v == voyage_id and f in compatible_plan_flows and size == size_mode and qty > 0:
                        exact_upper[area_no] += qty
                if exact_upper:
                    for area_no, qty in exact_upper.items():
                        area_size_quota[(voyage_id, flow, area_no, size_mode)] = qty
                        assigned_areas[(voyage_id, flow)].add(area_no)
                    continue

                all_size_weights: Counter[str] = Counter()
                for (v, f, area_no), qty in raw_all_size_area_quota.items():
                    if v == voyage_id and f in compatible_plan_flows and qty > 0:
                        all_size_weights[area_no] += qty
                if not all_size_weights:
                    raise ValueError(f"big plan has no area pattern for voyage={voyage_id}, flow={flow}, size={size_mode}")
                big_plan_total = sum(all_size_weights.values())
                if big_plan_total <= target_qty:
                    allocations = Counter(all_size_weights)
                else:
                    allocations = _allocate_by_weights(dict(all_size_weights), target_qty)
                for area_no, qty in allocations.items():
                    if qty <= 0:
                        continue
                    area_size_quota[(voyage_id, flow, area_no, size_mode)] = qty
                    assigned_areas[(voyage_id, flow)].add(area_no)

    area_quota_counter: Counter[tuple[str, str, str]] = Counter()
    for (voyage_id, flow, area_no, _size_mode), qty in area_size_quota.items():
        area_quota_counter[(voyage_id, flow, area_no)] += qty
    area_quota = dict(area_quota_counter)

    voyage_windows = _build_voyage_windows(target_voyages, vessel_schedules, horizon_hours, planning_time)
    # `planning_time` 是当前运行分配算法的时刻，也是判断堆场已有箱是否仍占用
    # 容量的快照时刻。不能用开港窗口开始时间覆盖它，否则诊断输出和容量判断
    # 都会偏到第一个航次的 receive_start。
    capacity_snapshot_time = planning_time
    bays = build_bays(
        data_dir,
        allowed_areas,
        closed,
        planning_time=capacity_snapshot_time,
        vessel_schedules=vessel_schedules,
        target_voyages=set(target_voyages),
        area_functions=area_functions,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
    )
    bay_requirements, bay_blocklist, bay_adjust_rules, bay_constraint_summary = read_medium_small_bay_controls(
        data_dir,
        groups,
        small_groups,
        bays,
    )
    area_operations = build_area_operations(data_dir, vessel_schedules)
    berth_distances = read_berth_distances(data_dir)
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
        business_special_codes=_collect_business_special_codes(groups),
        planning_time=capacity_snapshot_time,
        horizon_hours=horizon_hours,
        voyage_windows=voyage_windows,
        area_operations=area_operations,
        target_voyages=target_voyages,
        berth_distances=berth_distances,
        berth_by_voyage=berth_by_voyage,
        tops_reserved_slot_count=getattr(build_bays, "last_reserved_count", 0),
        tops_closed_bay_count=getattr(build_bays, "last_tops_closed_bay_count", 0),
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
        misplaced_excluded_bay_count=getattr(build_bays, "last_misplaced_excluded_bay_count", 0),
        user_voyage_area_allowlist=user_area_allowlist,
        user_voyage_area_blocklist=user_area_blocklist,
        user_voyage_area_requirements=user_area_requirements,
        user_area_constraint_summary=user_area_constraint_summary,
        user_group_bay_requirements=bay_requirements,
        user_group_bay_blocklist=bay_blocklist,
        user_bay_adjust_rules=bay_adjust_rules,
        user_bay_constraint_summary=bay_constraint_summary,
        attribute_rules=attribute_rules,
        medium_doc_floor_added_boxes=sum(medium_doc_floor_added_by_coarse_group.values()),
        medium_doc_floor_added_groups=len(medium_doc_floor_added_by_coarse_group),
        medium_doc_floor_shifted_boxes=sum(medium_doc_floor_shifted_by_coarse_group.values()),
        medium_doc_floor_shifted_groups=len(medium_doc_floor_shifted_by_coarse_group),
        medium_doc_floor_by_coarse_group=medium_doc_floor_by_coarse_group,
        medium_doc_floor_added_by_coarse_group=medium_doc_floor_added_by_coarse_group,
        medium_doc_floor_shifted_by_coarse_group=medium_doc_floor_shifted_by_coarse_group,
    )


def _build_voyage_windows(
    voyage_ids: list[str],
    schedules: dict[str, VoyageSchedule],
    horizon_hours: float,
    fallback_start: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    windows: dict[str, tuple[datetime, datetime]] = {}
    for voyage_id in voyage_ids:
        if voyage_id in schedules:
            schedule = schedules[voyage_id]
            _stage, _ratio, start, end = planning_stage_window(
                schedule.receive_start,
                schedule.receive_end,
                fallback_start,
                horizon_hours,
            )
            windows[voyage_id] = (start, end)
        else:
            start = fallback_start
            windows[voyage_id] = (start, start + timedelta(hours=horizon_hours))
    return windows


def _collect_business_special_codes(groups: list[BoxGroup]) -> set[str]:
    out: set[str] = set()
    for group in groups:
        out.update(group.special_codes)
    return out
