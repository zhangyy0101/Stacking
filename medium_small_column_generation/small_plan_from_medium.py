from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from medium_small_column_generation.block_bay_planning.models import BigPlanRow, ProblemData
from medium_small_column_generation.column_generation_planner import ColumnGenerationConfig


MediumPlanQuotaKey = tuple[str, str, str, str, str]
MediumPlanBayQuotaKey = tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class ExternalMediumPlan:
    rows: list[dict]
    coarse_area_quota: Counter[MediumPlanQuotaKey]
    coarse_bay_quota: Counter[MediumPlanBayQuotaKey]
    big_plan_rows: list[BigPlanRow]
    target_voyages: list[str]


def read_external_medium_plan(path: str | Path) -> ExternalMediumPlan:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"medium plan has no header: {path}")
        field_map = _build_field_map(reader.fieldnames)
        rows: list[dict] = []
        coarse_quota: Counter[MediumPlanQuotaKey] = Counter()
        coarse_bay_quota: Counter[MediumPlanBayQuotaKey] = Counter()
        area_size_quota: Counter[tuple[str, str, str, str]] = Counter()
        voyages: set[str] = set()
        for raw in reader:
            voyage_id = _normalize_voyage(_get_required(raw, field_map, "voyage_id"))
            flow = _normalize_flow(_get_required(raw, field_map, "flow"))
            port = str(_get_required(raw, field_map, "port")).strip()
            size = _normalize_size(_get_required(raw, field_map, "size"))
            area_no = str(_get_required(raw, field_map, "area_no")).strip()
            bay_key = _optional_bay_key(raw, field_map, area_no)
            qty = _parse_qty(_get_required(raw, field_map, "planned_boxes"))
            if qty <= 0:
                continue
            key = (voyage_id, flow, port, size, area_no)
            coarse_quota[key] += qty
            if bay_key:
                coarse_bay_quota[key + (bay_key,)] += qty
            area_size_quota[(voyage_id, flow, area_no, _big_plan_size(size))] += qty
            voyages.add(voyage_id)
            row = {
                "plan_level": "external_medium",
                "voyage_id": voyage_id,
                "flow": flow,
                "port": port,
                "size": size,
                "area_no": area_no,
                "planned_boxes": qty,
            }
            if bay_key:
                row["bay_key"] = bay_key
                row["bay_no"] = bay_key.split("-", 1)[1] if "-" in bay_key else bay_key
            rows.append(row)
    if not coarse_quota:
        raise ValueError(f"medium plan has no positive planned boxes: {path}")
    big_plan_rows = [
        BigPlanRow(
            voyage_id=voyage_id,
            flow=flow,
            area_no=area_no,
            planned_boxes=int(qty),
            size_mode=size_mode,
        )
        for (voyage_id, flow, area_no, size_mode), qty in sorted(area_size_quota.items())
        if qty > 0
    ]
    return ExternalMediumPlan(
        rows=sorted(rows, key=lambda row: (row["voyage_id"], row["flow"], row["area_no"], row.get("bay_key", ""), row["port"], row["size"])),
        coarse_area_quota=coarse_quota,
        coarse_bay_quota=coarse_bay_quota,
        big_plan_rows=big_plan_rows,
        target_voyages=sorted(voyages),
    )


def filter_external_medium_plan(plan: ExternalMediumPlan, voyages: list[str] | set[str] | tuple[str, ...]) -> ExternalMediumPlan:
    voyage_set = {_normalize_voyage(voyage) for voyage in voyages}
    rows = [row for row in plan.rows if row["voyage_id"] in voyage_set]
    coarse_quota: Counter[MediumPlanQuotaKey] = Counter(
        {key: qty for key, qty in plan.coarse_area_quota.items() if key[0] in voyage_set and qty > 0}
    )
    coarse_bay_quota: Counter[MediumPlanBayQuotaKey] = Counter(
        {key: qty for key, qty in plan.coarse_bay_quota.items() if key[0] in voyage_set and qty > 0}
    )
    big_plan_rows = [row for row in plan.big_plan_rows if row.voyage_id in voyage_set and row.planned_boxes > 0]
    if not coarse_quota:
        raise ValueError("external medium plan has no rows for the selected voyages")
    return ExternalMediumPlan(
        rows=rows,
        coarse_area_quota=coarse_quota,
        coarse_bay_quota=coarse_bay_quota,
        big_plan_rows=big_plan_rows,
        target_voyages=sorted(voyage_set),
    )


def apply_external_medium_plan(problem: ProblemData, plan: ExternalMediumPlan) -> ProblemData:
    area_quota: Counter[tuple[str, str, str]] = Counter()
    area_size_quota: Counter[tuple[str, str, str, str]] = Counter()
    assigned_areas: dict[tuple[str, str], set[str]] = {}
    for row in plan.big_plan_rows:
        area_quota[(row.voyage_id, row.flow, row.area_no)] += row.planned_boxes
        area_size_quota[(row.voyage_id, row.flow, row.area_no, row.size_mode)] += row.planned_boxes
        assigned_areas.setdefault((row.voyage_id, row.flow), set()).add(row.area_no)
    problem.big_plan = list(plan.big_plan_rows)
    problem.groups = []
    problem.area_quota = dict(area_quota)
    problem.area_size_quota = dict(area_size_quota)
    problem.assigned_areas = assigned_areas
    problem.target_voyages = list(plan.target_voyages)
    return problem


def configure_small_plan_from_medium(config: ColumnGenerationConfig, plan: ExternalMediumPlan) -> ColumnGenerationConfig:
    config.demand_mode = "doc-only"
    config.medium_plan_quota = dict(plan.coarse_area_quota)
    config.medium_plan_bay_quota = dict(plan.coarse_bay_quota) if plan.coarse_bay_quota else None
    return config


def _build_field_map(fieldnames: list[str]) -> dict[str, str]:
    normalized = {_normalize_header(name): name for name in fieldnames}
    aliases = {
        "voyage_id": ("voyage_id", "voy_id", "voyage", "voy", "ship_voyage"),
        "flow": ("flow", "status", "io_type", "inout", "in_out"),
        "port": ("port", "pod", "discharge_port", "disc_port", "unload_port", "port_cd"),
        "size": ("size", "size_mode", "cntr_size", "container_size", "cntr_siz_cod"),
        "area_no": ("area_no", "area", "yard_area", "block", "block_no"),
        "planned_boxes": ("planned_boxes", "planned_qty", "qty", "quantity", "box_qty", "boxes"),
    }
    optional_aliases = {
        "bay_key": ("bay_key", "block_bay_key", "yard_bay_key"),
        "bay_no": ("bay_no", "bay", "bay_code", "yard_bay", "block_bay"),
    }
    field_map: dict[str, str] = {}
    for canonical, names in {**aliases, **optional_aliases}.items():
        for name in names:
            actual = normalized.get(_normalize_header(name))
            if actual is not None:
                field_map[canonical] = actual
                break
    missing = sorted(set(aliases) - set(field_map))
    if missing:
        raise ValueError(f"medium plan missing required columns: {', '.join(missing)}")
    return field_map


def _get_required(row: dict, field_map: dict[str, str], key: str) -> object:
    value = row.get(field_map[key])
    if value is None or str(value).strip() == "":
        raise ValueError(f"medium plan row has empty {key}: {row}")
    return value


def _get_optional(row: dict, field_map: dict[str, str], key: str) -> str:
    field = field_map.get(key)
    if not field:
        return ""
    value = row.get(field)
    return "" if value is None else str(value).strip()


def _optional_bay_key(row: dict, field_map: dict[str, str], area_no: str) -> str:
    bay_key = _get_optional(row, field_map, "bay_key")
    if bay_key:
        return bay_key if "-" in bay_key else f"{area_no}-{bay_key}"
    bay_no = _get_optional(row, field_map, "bay_no")
    if bay_no:
        return f"{area_no}-{bay_no}"
    return ""


def _normalize_header(value: object) -> str:
    text = str(value).strip().lower().replace("\ufeff", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _normalize_voyage(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _normalize_flow(value: object) -> str:
    return str(value).strip().upper()


def _normalize_size(value: object) -> str:
    text = str(value).strip().upper()
    compact = re.sub(r"[^0-9A-Z]+", "", text)
    if compact.startswith("45"):
        return "45"
    if compact.startswith("20") or compact == "2":
        return "20"
    if compact.startswith("40") or compact == "4":
        return "40"
    raise ValueError(f"unsupported size in medium plan: {value}")


def _big_plan_size(size: str) -> str:
    return "40" if size == "45" else size


def _parse_qty(value: object) -> int:
    return max(0, int(round(float(str(value).strip()))))
