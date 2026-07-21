from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from .input_json import has_input_json, input_dataframe, input_value, input_voyage_value, vessel_doc_frame, vessel_predict_cntrs


DEFAULT_PLANNING_TIME = datetime(2026, 5, 19, 9, 30)
DEFAULT_TARGET_VOYAGES: tuple[str, ...] = ()
DEFAULT_TARGET_BIG_PLAN_FLOWS = frozenset({"OF"})


def _read_csv_compat(path: str | Path, **kwargs) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def read_excel_compat(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ImportError:
        return _read_xlsx_worksheet(Path(path), sheet_name)


def _read_xlsx_worksheet(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf, ns)
        sheet_path = _xlsx_sheet_path(zf, ns, sheet_name)
        sheet = ElementTree.fromstring(zf.read(sheet_path))
    rows: list[list[object]] = []
    for row in sheet.findall(".//a:sheetData/a:row", ns):
        values: dict[int, object] = {}
        for cell in row.findall("a:c", ns):
            col = _xlsx_column_index(cell.attrib.get("r", ""))
            if col is None:
                col = len(values)
            values[col] = _xlsx_cell_value(cell, shared, ns)
        if values:
            rows.append([values.get(idx) for idx in range(max(values) + 1)])
    while rows and all(value in (None, "") for value in rows[0]):
        rows.pop(0)
    if not rows:
        return pd.DataFrame()
    headers = _xlsx_headers(rows[0])
    records = []
    for row in rows[1:]:
        if all(value in (None, "") for value in row):
            continue
        records.append({headers[idx]: row[idx] if idx < len(row) else None for idx in range(len(headers))})
    return pd.DataFrame(records)


def _xlsx_shared_strings(zf: zipfile.ZipFile, ns: dict[str, str]) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.findall(".//a:t", ns)) for si in root.findall("a:si", ns)]


def _xlsx_sheet_path(zf: zipfile.ZipFile, ns: dict[str, str], sheet_name: str | int) -> str:
    workbook = ElementTree.fromstring(zf.read("xl/workbook.xml"))
    rels = ElementTree.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("rel:Relationship", ns)}
    sheets = workbook.findall(".//a:sheets/a:sheet", ns)
    index = int(sheet_name) if isinstance(sheet_name, int) else None
    for idx, sheet in enumerate(sheets):
        if (index is not None and idx == index) or sheet.attrib.get("name") == sheet_name:
            rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = targets.get(rel_id or "", f"worksheets/sheet{idx + 1}.xml")
            return target.lstrip("/") if target.startswith("xl/") else "xl/" + target.lstrip("/")
    raise ValueError(f"Worksheet {sheet_name!r} not found in {zf.filename}")


def _xlsx_column_index(ref: str) -> int | None:
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    if not letters:
        return None
    value = 0
    for ch in letters:
        value = value * 26 + ord(ch) - ord("A") + 1
    return value - 1


def _xlsx_cell_value(cell: ElementTree.Element, shared: list[str], ns: dict[str, str]) -> object:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        inline = cell.find("a:is", ns)
        return "" if inline is None else "".join(t.text or "" for t in inline.findall(".//a:t", ns))
    raw = cell.find("a:v", ns)
    if raw is None:
        return ""
    text = raw.text or ""
    if cell_type == "s":
        try:
            return shared[int(text)]
        except (IndexError, ValueError):
            return ""
    return text


def _xlsx_headers(row: list[object]) -> list[str]:
    headers: list[str] = []
    seen: Counter[str] = Counter()
    for idx, value in enumerate(row):
        header = str(value or f"column_{idx + 1}").strip() or f"column_{idx + 1}"
        seen[header] += 1
        if seen[header] > 1:
            header = f"{header}_{seen[header]}"
        headers.append(header)
    return headers


@dataclass(frozen=True)
class DemandRow:
    voyage_id: str
    flow: str
    port: str
    size_mode: str
    predicted_boxes: int
    ratio_target_boxes: int
    doc_boxes: int
    yard_boxes: int
    planned_boxes: int
    planning_stage: str
    planning_ratio: float


def calculate_medium_demands(
    data_dir: str | Path,
    voyage_ids: list[str] | tuple[str, ...] = DEFAULT_TARGET_VOYAGES,
    planning_time: datetime = DEFAULT_PLANNING_TIME,
    horizon_hours: float = 24.0,
    big_plan_caps: dict[tuple[str, str, str], int] | None = None,
) -> list[DemandRow]:
    data_path = Path(data_dir)
    schedules = _read_receive_windows(data_path)
    import_voyages = _read_import_voyages(data_path)
    yard_by_voyage = _read_yard_by_voyage_port_size(
        data_path,
        {_voyage(voyage_id) for voyage_id in voyage_ids},
        planning_time,
    )
    rows: list[DemandRow] = []
    for voyage_id in voyage_ids:
        receive_window = schedules.get(voyage_id)
        if receive_window is None:
            receive_start = planning_time
            receive_end = planning_time + timedelta(hours=horizon_hours)
        else:
            receive_start, receive_end = receive_window
        stage, ratio, _window_start, _window_end = planning_stage_window(
            receive_start,
            receive_end,
            planning_time,
            horizon_hours,
        )
        docs = _read_doc_by_port_size(data_path, voyage_id)
        yard_counts = yard_by_voyage.get(voyage_id, Counter())
        if _voyage(voyage_id) in import_voyages:
            predicted = Counter()
            ratio_targets = Counter()
            planned_source = Counter(docs)
        else:
            predicted = _read_predicted_by_port_size(data_path, voyage_id)
            ratio_targets = _ratio_targets(predicted, ratio)
            planned_source = _choose_planned_source(_net_prediction_targets(ratio_targets, yard_counts), docs)
        keys = sorted(planned_source)
        for flow, size_mode, port in keys:
            yard_boxes = yard_counts.get((flow, size_mode, port), 0)
            planned = max(0, planned_source[(flow, size_mode, port)])
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
                    yard_boxes=yard_boxes,
                    planned_boxes=planned,
                    planning_stage=stage,
                    planning_ratio=ratio,
                )
            )
    if big_plan_caps:
        rows = _cap_rows_by_big_plan(rows, big_plan_caps)
    return rows


def _net_prediction_targets(
    ratio_targets: Counter[tuple[str, str, str]],
    yard_counts: Counter[tuple[str, str, str]],
) -> Counter[tuple[str, str, str]]:
    return Counter(
        {
            key: max(0, int(qty) - int(yard_counts.get(key, 0)))
            for key, qty in ratio_targets.items()
            if max(0, int(qty) - int(yard_counts.get(key, 0))) > 0
        }
    )


def _read_import_voyages(data_path: Path) -> set[str]:
    if has_input_json(data_path):
        voyages = set()
        containers = input_value(data_path, "vessel_containers", {})
        if not isinstance(containers, dict):
            return voyages
        for voyage_id in containers.keys():
            content = input_voyage_value(data_path, "vessel_containers", voyage_id, {})
            if isinstance(content, dict) and _norm(content.get("type")) == "I":
                voyages.add(_voyage(voyage_id))
        return voyages

    path = data_path / "vessel_berth_info_new.csv"
    if not path.exists():
        path = data_path / "vessel_berth_info.csv"
    if not path.exists():
        return set()
    frame = _read_csv_compat(path)
    if frame.empty:
        return set()
    voyages = set()
    for row in frame.to_dict("records"):
        if _norm(row.get("VOY_IEFG")) == "I":
            voyage_id = _voyage(row.get("VOY_ID"))
            if voyage_id:
                voyages.add(voyage_id)
    return voyages


def planning_stage_window(
    receive_start: datetime,
    receive_end: datetime | None,
    planning_time: datetime,
    horizon_hours: float = 24.0,
) -> tuple[str, float, datetime, datetime]:
    """Return the current planning stage, ratio, and voyage-specific time window.

    The stage window is the x-th horizon after the vessel's receiving-open time.
    For a vessel not opened yet but opening within the next horizon, the first
    receiving window is planned. When a receiving end time is available, the
    selected window is clipped to the total receiving interval.
    """

    horizon = timedelta(hours=horizon_hours if horizon_hours > 0 else 24.0)

    def clipped_end(start: datetime) -> datetime:
        end = start + horizon
        if receive_end is not None and receive_end > start:
            return min(end, receive_end)
        return end

    if planning_time < receive_start:
        stage = "before_open_within_24h" if planning_time + horizon >= receive_start else "before_open_beyond_24h"
        ratio = 0.70 if stage == "before_open_within_24h" else 0.0
        return stage, ratio, receive_start, clipped_end(receive_start)

    elapsed = planning_time - receive_start
    period_index = max(0, int(elapsed.total_seconds() // horizon.total_seconds()))
    if receive_end is not None and receive_end > receive_start:
        total_periods = max(1, math.ceil((receive_end - receive_start).total_seconds() / horizon.total_seconds()))
        period_index = min(period_index, total_periods - 1)

    window_start = receive_start + period_index * horizon
    window_end = clipped_end(window_start)
    if period_index == 0:
        return "open_first_24h", 0.70, window_start, window_end
    if period_index == 1:
        return "open_second_24h", 0.90, window_start, window_end
    return "open_third_24h_or_later", 1.00, window_start, window_end


def _choose_planned_source(
    ratio_targets: Counter[tuple[str, str, str]],
    docs: Counter[tuple[str, str, str]],
) -> Counter[tuple[str, str, str]]:
    planned: Counter[tuple[str, str, str]] = Counter()
    for key in sorted(set(ratio_targets) | set(docs)):
        qty = max(int(ratio_targets.get(key, 0)), int(docs.get(key, 0)))
        if qty > 0:
            planned[key] += qty
    return planned


def _cap_rows_by_big_plan(
    rows: list[DemandRow],
    big_plan_caps: dict[tuple[str, str, str], int],
) -> list[DemandRow]:
    grouped: defaultdict[tuple[str, str, str], list[DemandRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.voyage_id, row.flow, _cap_size_mode(row.size_mode))].append(row)

    capped_rows: list[DemandRow] = []
    for key, items in grouped.items():
        total = sum(item.planned_boxes for item in items)
        cap = big_plan_caps.get(key)
        if cap is None:
            cap = big_plan_caps.get((key[0], key[1], "ALL"))
        if cap is None or total <= cap:
            capped_rows.extend(items)
            continue
        scaled = _largest_remainder_scale([item.planned_boxes for item in items], total, cap)
        for item, planned_boxes in zip(items, scaled):
            if planned_boxes > 0:
                capped_rows.append(replace(item, planned_boxes=planned_boxes))
    return sorted(capped_rows, key=lambda row: (row.voyage_id, row.flow, row.size_mode, row.port))


def write_demand_rows(path: str | Path, rows: list[DemandRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        fieldnames = [
            "voyage_id",
            "flow",
            "port",
            "size_mode",
            "predicted_boxes",
            "ratio_target_boxes",
            "doc_boxes",
            "yard_boxes",
            "planned_boxes",
            "planning_stage",
            "planning_ratio",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _read_receive_windows(data_path: Path) -> dict[str, tuple[datetime, datetime | None]]:
    if has_input_json(data_path):
        frame = input_dataframe(data_path, "vessel_berth_info")
    else:
        path = data_path / "vessel_berth_info_new.csv"
        if not path.exists():
            path = data_path / "vessel_berth_info.csv"
        frame = _read_csv_compat(path)
    windows: dict[str, tuple[datetime, datetime | None]] = {}
    for row in frame.to_dict("records"):
        if _norm(row.get("VOY_IEFG")) != "E":
            continue
        voyage_id = _voyage(row.get("VOY_ID"))
        receive_start = _parse_datetime(row.get("SCD_RCVSTDT"))
        receive_end = _parse_datetime(row.get("SCD_RCVEDDT"))
        if voyage_id and receive_start:
            windows[voyage_id] = (receive_start, receive_end)
    return windows


def _planning_stage(receive_start: datetime, planning_time: datetime, horizon_hours: float = 24.0) -> tuple[str, float]:
    horizon = timedelta(hours=horizon_hours if horizon_hours > 0 else 24.0)
    if planning_time < receive_start:
        if planning_time + horizon >= receive_start:
            return "before_open_within_24h", 0.70
        return "before_open_beyond_24h", 0.0
    elapsed = planning_time - receive_start
    if elapsed < horizon:
        return "open_first_24h", 0.70
    if elapsed < horizon * 2:
        return "open_second_24h", 0.90
    return "open_third_24h_or_later", 1.00


def _read_predicted_by_port_size(data_path: Path, voyage_id: str) -> Counter[tuple[str, str, str]]:
    if has_input_json(data_path):
        return _predict_counter_from_cntrs(vessel_predict_cntrs(data_path, voyage_id))
    path = _find_prediction_file(data_path, voyage_id)
    if path is None:
        return Counter()
    frame = read_excel_compat(path, sheet_name="尺寸港口统计")
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in frame.to_dict("records"):
        size_mode = _size_mode(row.get("IYC_CSZ_CSIZECD"))
        port = _norm(row.get("IYC_POT_UNLDPORT"), "UNK")
        flow = _flow(row.get("IYC_STS_CSTATUSCD") or row.get("flow") or row.get("cntr_type"))
        counter[(flow, size_mode, port)] += int(round(float(row.get("count", 0) or 0)))
    return counter


def _find_prediction_file(data_path: Path, voyage_id: str) -> Path | None:
    exact = data_path / f"predict_data_{voyage_id}.xlsx"
    if exact.exists():
        return exact
    matches = sorted(
        path for path in data_path.glob(f"predict_data_{voyage_id}*.xlsx")
        if not path.name.startswith("~$")
    )
    return matches[0] if matches else None


def _predict_counter_from_cntrs(predict_cntrs: dict) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for raw_size, info in predict_cntrs.items():
        if not isinstance(info, dict):
            continue
        size_mode = _size_mode(raw_size)
        detail = info.get("detail_info", {})
        if isinstance(detail, dict) and detail:
            for port, count in detail.items():
                counter[("OF", size_mode, _norm(port, "UNK"))] += int(round(float(count or 0)))
        else:
            counter[("OF", size_mode, "UNK")] += int(round(float(info.get("total_volume", 0) or 0)))
    return counter


def _ratio_targets(predicted: Counter[tuple[str, str, str]], ratio: float) -> Counter[tuple[str, str, str]]:
    targets: Counter[tuple[str, str, str]] = Counter()
    by_flow_size: defaultdict[tuple[str, str], list[tuple[tuple[str, str, str], int]]] = defaultdict(list)
    for key, count in predicted.items():
        by_flow_size[(key[0], key[1])].append((key, count))
    for items in by_flow_size.values():
        total = sum(count for _, count in items)
        target_total = int(round(total * ratio))
        scaled = _largest_remainder_scale([count for _, count in items], total, target_total)
        for (key, _), qty in zip(items, scaled):
            if qty > 0:
                targets[key] += qty
    return targets


def _read_doc_by_port_size(data_path: Path, voyage_id: str) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    if has_input_json(data_path):
        frame = vessel_doc_frame(data_path, voyage_id)
    else:
        path = data_path / f"container_info_{voyage_id}.parquet"
        if not path.exists():
            return counter
        frame = pd.read_parquet(path)
    if frame.empty:
        return counter
    work = pd.DataFrame(
        {
            "flow": frame.get("IYC_STS_CSTATUSCD", pd.Series(index=frame.index, dtype=object)).map(_flow),
            "size": frame.get("IYC_CSZ_CSIZECD", pd.Series(index=frame.index, dtype=object)).map(_size_mode),
            "port": frame.get("IYC_POT_UNLDPORT", pd.Series(index=frame.index, dtype=object)).map(lambda value: _norm(value, "UNK")),
        }
    )
    counts = work.groupby(["flow", "size", "port"], sort=False).size()
    counter.update({(str(flow), str(size), str(port)): int(count) for (flow, size, port), count in counts.items()})
    return counter


def _read_yard_by_voyage_port_size(
    data_path: Path,
    voyage_ids: set[str],
    planning_time: datetime,
) -> dict[str, Counter[tuple[str, str, str]]]:
    if not voyage_ids:
        return {}
    columns = [
        "HAS_CONTAINER",
        "IYC_CNTRID",
        "IYC_CNTRNO",
        "IYC_EVOY_ID",
        "IYC_IVOY_ID",
        "IYC_INYTM",
        "IYC_STS_CSTATUSCD",
        "IYC_CSZ_CSIZECD",
        "IYC_POT_UNLDPORT",
    ]
    if has_input_json(data_path):
        frame = input_dataframe(data_path, "bay_slots_detail", columns=columns)
    else:
        path = data_path / "bay_slots_detail.parquet"
        if not path.exists():
            return {}
        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception:
            frame = pd.read_parquet(path)
    if frame.empty or "HAS_CONTAINER" not in frame.columns:
        return {}

    occupied = frame.loc[frame["HAS_CONTAINER"].fillna(0).astype(int) == 1].copy()
    if occupied.empty or "IYC_EVOY_ID" not in occupied.columns:
        return {}
    if "IYC_INYTM" in occupied.columns:
        in_time = pd.to_datetime(occupied["IYC_INYTM"], errors="coerce")
        occupied = occupied.loc[in_time.isna() | (in_time <= planning_time)]
    occupied = _medium_small_yard_included_rows(occupied)
    if occupied.empty:
        return {}

    occupied["_voyage"] = occupied["IYC_EVOY_ID"].map(_voyage)
    occupied = occupied.loc[occupied["_voyage"].isin(voyage_ids)].copy()
    if occupied.empty:
        return {}
    occupied["_container_key"] = [
        _container_identity(row, index)
        for index, row in occupied.iterrows()
    ]
    occupied = occupied.drop_duplicates("_container_key")
    occupied["_flow"] = occupied.get("IYC_STS_CSTATUSCD", pd.Series(index=occupied.index, dtype=object)).map(_flow)
    occupied["_size"] = occupied.get("IYC_CSZ_CSIZECD", pd.Series(index=occupied.index, dtype=object)).map(_size_mode)
    occupied["_port"] = occupied.get("IYC_POT_UNLDPORT", pd.Series(index=occupied.index, dtype=object)).map(
        lambda value: _norm(value, "UNK")
    )

    out: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)
    counts = occupied.groupby(["_voyage", "_flow", "_size", "_port"], sort=False).size()
    for (voyage_id, flow, size, port), count in counts.items():
        out[str(voyage_id)][(str(flow), str(size), str(port))] += int(count)
    return dict(out)


def _yard_transshipment_mask(rows: pd.DataFrame) -> pd.Series:
    if rows.empty or "IYC_EVOY_ID" not in rows.columns or "IYC_IVOY_ID" not in rows.columns:
        return pd.Series(False, index=rows.index)
    export_voyage = rows["IYC_EVOY_ID"].map(lambda value: bool(_voyage(value)))
    import_voyage = rows["IYC_IVOY_ID"].map(lambda value: bool(_voyage(value)))
    return export_voyage & import_voyage


def _medium_small_yard_included_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    return rows.loc[~_yard_transshipment_mask(rows)].copy()


def _container_identity(row: pd.Series, index: object) -> str:
    number = _norm(row.get("IYC_CNTRNO"))
    if number:
        return f"NO:{number}"
    cntr_id = _norm(row.get("IYC_CNTRID"))
    if cntr_id and cntr_id not in {"-1", "0"}:
        return f"ID:{cntr_id}"
    return f"ROW:{index}"


def _largest_remainder_scale(counts: list[int], source_total: int, target_total: int) -> list[int]:
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


def _size_mode(value: object) -> str:
    text = _norm(value)
    return text if text in {"20", "40", "45"} else "40"


def _cap_size_mode(size_mode: str) -> str:
    return "40" if size_mode == "45" else size_mode


def _flow(value: object) -> str:
    return _norm(value, "OF").upper() or "OF"


def _parse_datetime(value: object) -> datetime | None:
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


def _norm(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text or default


def _voyage(value: object) -> str:
    text = _norm(value)
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


def _date_key(value: object) -> str:
    text = _norm(value)
    if not text:
        return ""
    parsed = _parse_datetime(text)
    if parsed is not None:
        return parsed.date().isoformat()
    text = text.replace("/", "-")
    return text[:10] if len(text) >= 10 else text


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "\u5806\u5b58\u8ba1\u5212\u6d4b\u8bd5\u6570\u636e20260519"


def _default_big_plan_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "large" / "outputs_large" / "latest_run" / "allocation.csv"


def _infer_target_voyages_from_big_plan_csv(path: str | Path, planning_time: datetime) -> list[str]:
    target_flows = {_flow(flow) for flow in DEFAULT_TARGET_BIG_PLAN_FLOWS}
    plan_date = planning_time.date().isoformat()
    voyages: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        fieldnames = set(reader.fieldnames or [])
        voyage_field = "voy_id" if "voy_id" in fieldnames else "voyage_id" if "voyage_id" in fieldnames else ""
        if not voyage_field:
            return []
        flow_field = next((field for field in ("flow", "cntr_type", "status") if field in fieldnames), "")
        date_field = next(
            (field for field in ("plan_date", "date", "work_date", "planning_date", "day") if field in fieldnames),
            "",
        )
        for row in reader:
            flow = _flow(row.get(flow_field)) if flow_field else "OF"
            row_date = _date_key(row.get(date_field)) if date_field else ""
            voyage_id = _voyage(row.get(voyage_field))
            if voyage_id and flow in target_flows and (not row_date or row_date == plan_date):
                voyages.add(voyage_id)
    return sorted(voyages, key=_voyage_sort_key)


def _medium_demand_caps_from_big_plan_csv(
    path: str | Path,
    voyages: list[str] | tuple[str, ...],
    planning_time: datetime,
) -> dict[tuple[str, str, str], int]:
    target_flows = {_flow(flow) for flow in DEFAULT_TARGET_BIG_PLAN_FLOWS}
    voyage_set = {_voyage(voyage_id) for voyage_id in voyages}
    plan_date = planning_time.date().isoformat()
    size_pool: Counter[tuple[str, str]] = Counter()
    all_size_pool: Counter[str] = Counter()

    def add(voyage_id: str, flow: str, size_mode: str, boxes: int, row_date: str) -> None:
        if (
            boxes <= 0
            or voyage_id not in voyage_set
            or flow not in target_flows
            or (row_date and row_date != plan_date)
        ):
            return
        if size_mode == "ALL":
            all_size_pool[voyage_id] += boxes
        else:
            size_pool[(voyage_id, "40" if size_mode == "45" else size_mode)] += boxes

    with Path(path).open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        fieldnames = set(reader.fieldnames or [])
        flow_field = _csv_field(fieldnames, ["flow", "cntr_type", "status"])
        date_field = _csv_field(fieldnames, ["plan_date", "date", "work_date", "planning_date", "day"])
        if {"voy_id", "area_no"}.issubset(fieldnames) and ("new_qty" in fieldnames or "planned_qty" in fieldnames):
            size_field = _csv_field(fieldnames, ["size", "size_mode"])
            qty_field = _csv_field(fieldnames, ["new_qty", "planned_qty"])
            for row in reader:
                add(
                    _voyage(row.get("voy_id")),
                    _flow(row.get(flow_field)) if flow_field else "OF",
                    _big_plan_size_mode(row.get(size_field)) if size_field else "ALL",
                    int(round(float(row.get(qty_field, 0) or 0))),
                    _date_key(row.get(date_field)) if date_field else "",
                )
        elif {"voyage_id", "area_no"}.issubset(fieldnames) and (
            {"qty_20", "qty_40"}.issubset(fieldnames)
            or {"planned_20", "planned_40"}.issubset(fieldnames)
            or {"20", "40"}.issubset(fieldnames)
        ):
            fields_by_size = {
                "20": _csv_field(fieldnames, ["qty_20", "planned_20", "20", "c20", "C20"]),
                "40": _csv_field(fieldnames, ["qty_40", "planned_40", "40", "c40", "C40"]),
                "45": _csv_field(fieldnames, ["qty_45", "planned_45", "45", "c45", "C45"]),
            }
            for row in reader:
                for size_mode, qty_field in fields_by_size.items():
                    if qty_field:
                        add(
                            _voyage(row.get("voyage_id")),
                            _flow(row.get(flow_field)) if flow_field else "OF",
                            size_mode,
                            int(round(float(row.get(qty_field, 0) or 0))),
                            _date_key(row.get(date_field)) if date_field else "",
                        )
        else:
            voyage_field = _csv_field(fieldnames, ["voyage_id", "voy_id"])
            qty_field = _csv_field(fieldnames, ["new_qty", "planned_boxes", "planned_qty"])
            size_field = _csv_field(fieldnames, ["size_mode", "size"])
            if not (voyage_field and qty_field):
                return {}
            for row in reader:
                add(
                    _voyage(row.get(voyage_field)),
                    _flow(row.get(flow_field)) if flow_field else "OF",
                    _big_plan_size_mode(row.get(size_field)) if size_field else "ALL",
                    int(round(float(row.get(qty_field, 0) or 0))),
                    _date_key(row.get(date_field)) if date_field else "",
                )

    caps: dict[tuple[str, str, str], int] = {}
    for (voyage_id, size_mode), qty in size_pool.items():
        for flow in target_flows:
            caps[(voyage_id, flow, size_mode)] = qty
    for voyage_id, qty in all_size_pool.items():
        if any(v == voyage_id for v, _size in size_pool):
            continue
        for flow in target_flows:
            caps[(voyage_id, flow, "ALL")] = qty
    return caps


def _csv_field(fieldnames: set[str], candidates: list[str]) -> str | None:
    lowered = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _big_plan_size_mode(value: object) -> str:
    text = _norm(value).upper()
    if not text:
        return "ALL"
    if text in {"20", "40", "45"}:
        return text
    return "40"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate medium-plan demand by voyage, discharge port, and size.")
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--big-plan", type=Path, default=_default_big_plan_path())
    parser.add_argument("--voyages", nargs="+", default=None)
    parser.add_argument(
        "--planning-time",
        default=None,
        help="Planning time override. Raw 0519 defaults to 2026-05-19 09:30:00; JSON data directories use input_data.json.",
    )
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--output", type=Path, default=Path("medium_demand_by_port.csv"))
    args = parser.parse_args()
    if args.planning_time:
        planning_time = _parse_datetime(args.planning_time)
        if planning_time is None:
            raise SystemExit(f"Invalid --planning-time: {args.planning_time}")
    elif has_input_json(args.data_dir):
        raw_planning_time = input_value(args.data_dir, "planning_time")
        planning_time = _parse_datetime(raw_planning_time)
        if raw_planning_time is not None and planning_time is None:
            raise SystemExit(f"Invalid planning_time in input_data.json: {raw_planning_time}")
        planning_time = planning_time or DEFAULT_PLANNING_TIME
    else:
        planning_time = DEFAULT_PLANNING_TIME
    voyages = args.voyages or _infer_target_voyages_from_big_plan_csv(args.big_plan, planning_time)
    if not voyages:
        raise SystemExit("No target voyages provided or inferred from OF big-plan rows")
    big_plan_caps = _medium_demand_caps_from_big_plan_csv(args.big_plan, voyages, planning_time)
    rows = calculate_medium_demands(
        args.data_dir,
        voyages,
        planning_time,
        args.horizon_hours,
        big_plan_caps=big_plan_caps,
    )
    write_demand_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
