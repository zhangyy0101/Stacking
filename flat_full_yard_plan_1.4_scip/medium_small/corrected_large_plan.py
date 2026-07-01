from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


QUANTITY_FIELDS = {
    "voy_id",
    "voyage_id",
    "flow",
    "area_no",
    "size",
    "size_mode",
    "planned_qty",
    "snapshot_qty",
    "new_qty",
    "planned_boxes",
}


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def write_corrected_large_plan_outputs(
    corrected_large_plan_path: str | Path,
    diagnostics_path: str | Path,
    original_large_rows: list[dict[str, Any]],
    medium_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, diagnostics = build_corrected_large_plan_rows(original_large_rows, medium_rows)
    write_rows(corrected_large_plan_path, rows)
    write_json(diagnostics_path, diagnostics)
    return rows, diagnostics


def build_corrected_large_plan_rows(
    original_large_rows: list[dict[str, Any]],
    medium_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_area: dict[tuple[str, str, str, str], dict[str, float]] = {}
    original_fieldnames: list[str] = []
    metadata_by_group: dict[tuple[str, str, str], dict[str, Any]] = {}
    metadata_fallback: dict[str, Any] = {}

    for row in original_large_rows:
        if not original_fieldnames:
            original_fieldnames = list(row.keys())
            metadata_fallback = {key: value for key, value in row.items() if key not in QUANTITY_FIELDS}
        voyage_id = str(_first_value(row, "voy_id", "voyage_id", "voyage", "voy") or "").strip()
        flow = str(_first_value(row, "flow", "cntr_type", "status") or "OF").strip() or "OF"
        area_no = str(_first_value(row, "area_no", "area", "yard_area", "block") or "").strip()
        size = _large_plan_size(_first_value(row, "size", "size_mode", "cntr_size"))
        if not voyage_id or not area_no or not size:
            continue

        snapshot = _number(_first_value(row, "snapshot_qty", "snapshot_boxes"))
        explicit_new = _first_value(row, "new_qty")
        if explicit_new is None:
            explicit_new = _first_value(row, "planned_boxes", "qty", "boxes")
        explicit_planned = _first_value(row, "planned_qty", "planned_boxes")
        new_qty = _number(explicit_new)
        if explicit_new is None:
            new_qty = max(0.0, _number(explicit_planned) - snapshot)
        planned_qty = _number(explicit_planned)
        if explicit_planned is None:
            planned_qty = snapshot + new_qty

        key = (voyage_id, flow, area_no, size)
        group_key = (voyage_id, flow, size)
        entry = original_area.setdefault(key, {"planned": 0.0, "snapshot": 0.0, "new": 0.0})
        entry["planned"] += planned_qty
        entry["snapshot"] += snapshot
        entry["new"] += new_qty
        metadata_by_group.setdefault(group_key, {key: value for key, value in row.items() if key not in QUANTITY_FIELDS})

    medium_area: dict[tuple[str, str, str, str], int] = {}
    for row in medium_rows:
        voyage_id = str(_first_value(row, "voyage_id", "voy_id", "voyage", "voy") or "").strip()
        flow = str(_first_value(row, "flow", "cntr_type", "status") or "OF").strip() or "OF"
        area_no = str(_first_value(row, "area_no", "area", "yard_area", "block") or "").strip()
        size = _large_plan_size(_first_value(row, "size", "size_mode", "cntr_size", "big_size"))
        qty = int(round(_number(_first_value(row, "planned_boxes", "planned_qty", "qty", "boxes"))))
        if not voyage_id or not area_no or not size or qty <= 0:
            continue
        key = (voyage_id, flow, area_no, size)
        medium_area[key] = medium_area.get(key, 0) + qty

    all_group_keys = sorted(
        {(voyage_id, flow, size) for voyage_id, flow, _area_no, size in original_area}
        | {(voyage_id, flow, size) for voyage_id, flow, _area_no, size in medium_area}
    )
    corrected_new_by_area: dict[tuple[str, str, str, str], int] = {}
    diagnostics: dict[str, Any] = {
        "algorithm": "corrected_large_plan_after_medium_small_new_qty",
        "feasible": True,
        "groups": [],
        "coverage_violation_count": 0,
        "medium_area_count": len(medium_area),
        "expanded_groups": [],
        "new_total_expansion_group_count": 0,
        "new_total_expansion_boxes": 0,
        "planned_total_expansion_boxes": 0,
    }

    for group_key in all_group_keys:
        voyage_id, flow, size = group_key
        area_keys = sorted(
            {key for key in original_area if (key[0], key[1], key[3]) == group_key}
            | {key for key in medium_area if (key[0], key[1], key[3]) == group_key}
        )
        snapshot_by_area = {
            key: int(round(original_area.get(key, {}).get("snapshot", 0.0)))
            for key in area_keys
        }
        original_new_by_area = {
            key: int(round(original_area.get(key, {}).get("new", 0.0)))
            for key in area_keys
        }
        medium_by_area = {key: int(medium_area.get(key, 0)) for key in area_keys}

        original_planned_total = int(round(sum(original_area.get(key, {}).get("planned", 0.0) for key in area_keys)))
        original_snapshot_total = int(sum(snapshot_by_area.values()))
        original_new_total = int(sum(original_new_by_area.values()))
        medium_used_total = int(sum(medium_by_area.values()))
        target_new_total = max(original_new_total, medium_used_total)

        current_new = {
            key: max(original_new_by_area.get(key, 0), medium_by_area.get(key, 0))
            for key in area_keys
        }
        current_new_total = int(sum(current_new.values()))
        if current_new_total > target_new_total:
            over = current_new_total - target_new_total
            reducible = {
                key: max(0, current_new[key] - medium_by_area.get(key, 0))
                for key in area_keys
            }
            for key, qty in _largest_remainder_int_allocation(reducible, over).items():
                current_new[key] -= qty
        elif current_new_total < target_new_total:
            extra = target_new_total - current_new_total
            weights = {key: original_new_by_area.get(key, 0) for key in area_keys}
            if not any(weights.values()):
                weights = {key: medium_by_area.get(key, 0) for key in area_keys}
            if not any(weights.values()):
                weights = {key: max(1, snapshot_by_area.get(key, 0)) for key in area_keys}
            for key, qty in _largest_remainder_int_allocation(weights, extra).items():
                current_new[key] += qty

        coverage_violations = []
        for key in area_keys:
            corrected_new = int(current_new.get(key, 0))
            medium_used = int(medium_by_area.get(key, 0))
            if corrected_new < medium_used:
                coverage_violations.append(
                    {
                        "voy_id": key[0],
                        "flow": key[1],
                        "area_no": key[2],
                        "size": key[3],
                        "medium_used": medium_used,
                        "corrected_new_qty": corrected_new,
                    }
                )
        if coverage_violations:
            diagnostics["feasible"] = False
            diagnostics["coverage_violation_count"] += len(coverage_violations)

        corrected_new_total = int(sum(current_new.values()))
        corrected_planned_total = int(original_snapshot_total + corrected_new_total)
        new_expansion = max(0, corrected_new_total - original_new_total)
        planned_expansion = corrected_planned_total - original_planned_total
        group_diagnostics = {
            "voy_id": voyage_id,
            "flow": flow,
            "size": size,
            "original_planned_total": original_planned_total,
            "corrected_planned_total": corrected_planned_total,
            "original_snapshot_total": original_snapshot_total,
            "corrected_snapshot_total": original_snapshot_total,
            "original_new_total": original_new_total,
            "corrected_new_total": corrected_new_total,
            "medium_used_total": medium_used_total,
            "target_new_total": target_new_total,
            "new_total_expansion": new_expansion,
            "planned_total_expansion": planned_expansion,
            "coverage_violations": coverage_violations,
            "expanded_for_medium_plan": corrected_new_total > original_new_total,
        }
        diagnostics["groups"].append(group_diagnostics)
        if group_diagnostics["expanded_for_medium_plan"]:
            diagnostics["expanded_groups"].append(group_diagnostics)
            diagnostics["new_total_expansion_group_count"] += 1
            diagnostics["new_total_expansion_boxes"] += new_expansion
            diagnostics["planned_total_expansion_boxes"] += planned_expansion

        for key, qty in current_new.items():
            if qty > 0 or snapshot_by_area.get(key, 0) > 0 or medium_by_area.get(key, 0) > 0:
                corrected_new_by_area[key] = int(qty)

    output_fields = _output_fields(original_fieldnames)
    rows: list[dict[str, Any]] = []
    for key in sorted(corrected_new_by_area, key=lambda item: (item[0], item[1], item[3], item[2])):
        voyage_id, flow, area_no, size = key
        group_key = (voyage_id, flow, size)
        snapshot = int(round(original_area.get(key, {}).get("snapshot", 0.0)))
        new_qty = int(corrected_new_by_area[key])
        planned = snapshot + new_qty
        metadata = dict(metadata_fallback)
        metadata.update(metadata_by_group.get(group_key, {}))
        row = {field: "" for field in output_fields}
        row.update(metadata)
        row.update(
            {
                "voy_id": voyage_id,
                "flow": flow,
                "area_no": area_no,
                "size": size,
                "planned_qty": planned,
                "snapshot_qty": float(snapshot),
                "new_qty": float(new_qty),
            }
        )
        if "voyage_id" in output_fields:
            row["voyage_id"] = voyage_id
        if "size_mode" in output_fields:
            row["size_mode"] = size
        if "planned_boxes" in output_fields:
            row["planned_boxes"] = float(new_qty)
        rows.append(row)

    diagnostics["row_count"] = len(rows)
    diagnostics["original_row_count"] = len(original_large_rows)
    diagnostics["original_planned_total"] = int(sum(group["original_planned_total"] for group in diagnostics["groups"]))
    diagnostics["corrected_planned_total"] = int(sum(group["corrected_planned_total"] for group in diagnostics["groups"]))
    diagnostics["original_snapshot_total"] = int(sum(group["original_snapshot_total"] for group in diagnostics["groups"]))
    diagnostics["corrected_snapshot_total"] = int(sum(group["corrected_snapshot_total"] for group in diagnostics["groups"]))
    diagnostics["original_new_total"] = int(sum(group["original_new_total"] for group in diagnostics["groups"]))
    diagnostics["corrected_new_total"] = int(sum(group["corrected_new_total"] for group in diagnostics["groups"]))
    diagnostics["medium_used_total"] = int(sum(group["medium_used_total"] for group in diagnostics["groups"]))
    return rows, diagnostics


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _output_fields(original_fieldnames: list[str]) -> list[str]:
    fields = list(original_fieldnames)
    for required in ("voy_id", "flow", "area_no", "size", "planned_qty", "snapshot_qty", "new_qty"):
        if required not in fields:
            fields.append(required)
    return fields


def _large_plan_size(size: object) -> str:
    text = "" if size is None else str(size).strip()
    return "40" if text in {"40", "45"} else "20"


def _first_value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _number(value: object) -> float:
    try:
        if value is None or value == "":
            return 0.0
        number = float(value)
        if math.isnan(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def _largest_remainder_int_allocation(weights: dict[tuple[str, str, str, str], float], total: int) -> dict[tuple[str, str, str, str], int]:
    total = int(total)
    if total <= 0 or not weights:
        return {}
    positive = {key: max(0.0, float(value)) for key, value in weights.items() if float(value) > 0}
    if not positive:
        keys = sorted(weights)
        return {key: total if index == 0 else 0 for index, key in enumerate(keys)}
    weight_total = sum(positive.values())
    raw = {key: total * value / weight_total for key, value in positive.items()}
    out = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(out.values())
    if remainder > 0:
        order = sorted(positive, key=lambda key: (raw[key] - int(raw[key]), positive[key], key), reverse=True)
        for key in order[:remainder]:
            out[key] = out.get(key, 0) + 1
    return {key: qty for key, qty in out.items() if qty > 0}
