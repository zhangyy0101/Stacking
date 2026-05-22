from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_PLANNING_TIME = datetime(2026, 5, 8, 9, 30)
DEFAULT_TARGET_VOYAGES = ("453334", "453400")


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


def calculate_medium_demands(
    data_dir: str | Path,
    voyage_ids: list[str] | tuple[str, ...] = DEFAULT_TARGET_VOYAGES,
    planning_time: datetime = DEFAULT_PLANNING_TIME,
) -> list[DemandRow]:
    data_path = Path(data_dir)
    schedules = _read_receive_starts(data_path)
    rows: list[DemandRow] = []
    for voyage_id in voyage_ids:
        receive_start = schedules.get(voyage_id)
        if receive_start is None:
            receive_start = planning_time
        stage, ratio = _planning_stage(receive_start, planning_time)
        predicted = _read_predicted_by_port_size(data_path, voyage_id)
        ratio_targets = _ratio_targets(predicted, ratio)
        docs = _read_doc_by_port_size(data_path, voyage_id)
        planned_source = _choose_planned_source(ratio_targets, docs)
        keys = sorted(planned_source)
        for flow, size_mode, port in keys:
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


def _choose_planned_source(
    ratio_targets: Counter[tuple[str, str, str]],
    docs: Counter[tuple[str, str, str]],
) -> Counter[tuple[str, str, str]]:
    planned: Counter[tuple[str, str, str]] = Counter()
    flows = sorted({flow for flow, _, _ in ratio_targets} | {flow for flow, _, _ in docs})
    for flow in flows:
        for size_mode in ("20", "40", "45"):
            ratio_total = sum(qty for (f, size, _), qty in ratio_targets.items() if f == flow and size == size_mode)
            doc_total = sum(qty for (f, size, _), qty in docs.items() if f == flow and size == size_mode)
            source = docs if doc_total > ratio_total else ratio_targets
            for (f, size, port), qty in source.items():
                if f == flow and size == size_mode and qty > 0:
                    planned[(f, size, port)] += qty
    return planned


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
            "planned_boxes",
            "planning_stage",
            "planning_ratio",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _read_receive_starts(data_path: Path) -> dict[str, datetime]:
    path = data_path / "vessel_berth_info_new.csv"
    if not path.exists():
        path = data_path / "vessel_berth_info.csv"
    frame = pd.read_csv(path)
    starts: dict[str, datetime] = {}
    for row in frame.to_dict("records"):
        if _norm(row.get("VOY_IEFG")) != "E":
            continue
        voyage_id = _voyage(row.get("VOY_ID"))
        receive_start = _parse_datetime(row.get("SCD_RCVSTDT"))
        if voyage_id and receive_start:
            starts[voyage_id] = receive_start
    return starts


def _planning_stage(receive_start: datetime, planning_time: datetime) -> tuple[str, float]:
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


def _read_predicted_by_port_size(data_path: Path, voyage_id: str) -> Counter[tuple[str, str, str]]:
    path = data_path / f"predict_data_{voyage_id}.xlsx"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path, sheet_name="尺寸港口统计")
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in frame.to_dict("records"):
        size_mode = _size_mode(row.get("IYC_CSZ_CSIZECD"))
        port = _norm(row.get("IYC_POT_UNLDPORT"), "UNK")
        flow = _flow(row.get("IYC_STS_CSTATUSCD") or row.get("flow") or row.get("cntr_type"))
        counter[(flow, size_mode, port)] += int(round(float(row.get("count", 0) or 0)))
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
    path = data_path / f"container_info_{voyage_id}.parquet"
    counter: Counter[tuple[str, str, str]] = Counter()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate medium-plan demand by voyage, discharge port, and size.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "堆存计划测试数据20260508补充")
    parser.add_argument("--voyages", nargs="+", default=list(DEFAULT_TARGET_VOYAGES))
    parser.add_argument("--planning-time", default=DEFAULT_PLANNING_TIME.strftime("%Y-%m-%d %H:%M:%S"))
    parser.add_argument("--output", type=Path, default=Path("medium_demand_by_port.csv"))
    args = parser.parse_args()
    planning_time = _parse_datetime(args.planning_time)
    if planning_time is None:
        raise SystemExit(f"Invalid --planning-time: {args.planning_time}")
    rows = calculate_medium_demands(args.data_dir, args.voyages, planning_time)
    write_demand_rows(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
