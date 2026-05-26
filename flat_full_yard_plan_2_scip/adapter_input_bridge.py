from __future__ import annotations

import json
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = REPO_ROOT / "test_data"
LOCAL_INPUT_ADAPTER = SCRIPT_DIR / "input_adapter_gd.py"


@dataclass(frozen=True)
class AdapterFlatDataExport:
    data_dir: Path
    adapter_json: Path
    planning_time: str | None
    export_vessels: list[str]
    import_vessels: list[str]
    metadata_path: Path


def export_adapter_json_to_flat_data(adapter_json_or_dir: Path, output_dir: Path) -> AdapterFlatDataExport:
    """Convert an InputAdapterGd JSON snapshot into the flat files used by the solver."""

    adapter_json = resolve_adapter_json(adapter_json_or_dir)
    adapter = _load_adapter_json(adapter_json)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_parquet(adapter.bay_slots_detail, output_dir / "bay_slots_detail.parquet", "bay_slots_detail")
    _write_slot_size_views(adapter.bay_slots_detail, output_dir)
    _write_parquet(adapter.tops_plan, output_dir / "tops_plan_info.parquet", "tops_plan")
    _write_area_functions(adapter.area_function_info, output_dir / "箱区功能.xlsx")
    _write_vessel_info(adapter.vessel_berth_info, output_dir)
    _write_distance_matrix(adapter.berth_area_dist_matrix, output_dir / "适放箱区_泊位距离矩阵.xlsx")
    _write_closed_areas(adapter.closed_area, output_dir / "n_usefg_areas.txt")

    export_vessels, import_vessels = _write_vessel_container_files(adapter.vessel_containers, output_dir)

    planning_time = None
    if adapter.planning_time is not None and not pd.isna(adapter.planning_time):
        planning_time = pd.Timestamp(adapter.planning_time).strftime("%Y-%m-%d %H:%M:%S")

    metadata = {
        "source_adapter_json": str(adapter_json),
        "generated_data_dir": str(output_dir),
        "planning_time": planning_time,
        "export_vessels": export_vessels,
        "import_vessels": import_vessels,
        "bay_slots_rows": _frame_len(adapter.bay_slots_detail),
        "tops_rows": _frame_len(adapter.tops_plan),
        "area_function_rows": _frame_len(adapter.area_function_info),
        "vessel_berth_rows": _frame_len(adapter.vessel_berth_info),
        "closed_areas": sorted(_normalize_code(area) for area in (adapter.closed_area or set())),
    }
    metadata_path = output_dir / "adapter_flat_data_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return AdapterFlatDataExport(
        data_dir=output_dir,
        adapter_json=adapter_json,
        planning_time=planning_time,
        export_vessels=export_vessels,
        import_vessels=import_vessels,
        metadata_path=metadata_path,
    )


def resolve_adapter_json(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "input_data.json"
    if not path.exists():
        raise FileNotFoundError(f"InputAdapter JSON not found: {path}")
    return path


def _load_adapter_json(adapter_json: Path) -> Any:
    if LOCAL_INPUT_ADAPTER.exists():
        spec = importlib.util.spec_from_file_location("local_input_adapter_gd", LOCAL_INPUT_ADAPTER)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load local input adapter: {LOCAL_INPUT_ADAPTER}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.InputAdapterGd.load_from_json(str(adapter_json))

    try:
        if TEST_DATA_DIR.exists() and str(TEST_DATA_DIR) not in sys.path:
            sys.path.insert(0, str(TEST_DATA_DIR))
        from input_adapter_gd import InputAdapterGd
    except ImportError as exc:  # pragma: no cover - this is an environment/setup error.
        raise ImportError(
            f"Cannot import input_adapter_gd.py. Expected it under {SCRIPT_DIR} or {TEST_DATA_DIR}."
        ) from exc
    return InputAdapterGd.load_from_json(str(adapter_json))


def _write_parquet(frame: pd.DataFrame | None, path: Path, name: str) -> None:
    if frame is None:
        raise ValueError(f"InputAdapter field {name} is empty; cannot write {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_slot_size_views(frame: pd.DataFrame | None, output_dir: Path) -> None:
    if frame is None or "YBY_ENABLECSIZECD" not in frame.columns:
        return
    size_text = frame["YBY_ENABLECSIZECD"].astype("string")
    enabled_20 = size_text.isna() | size_text.str.contains(r"(?<!\d)20(?!\d)", regex=True, na=False)
    enabled_40 = size_text.isna() | size_text.str.contains(r"(?<!\d)(?:40|45)(?!\d)", regex=True, na=False)
    frame[enabled_20].to_parquet(output_dir / "bay_slots_detail_20.parquet", index=False)
    frame[enabled_40].to_parquet(output_dir / "bay_slots_detail_40.parquet", index=False)


def _write_area_functions(frame: pd.DataFrame | None, path: Path) -> None:
    if frame is None:
        raise ValueError("InputAdapter field area_function_info is empty.")
    frame.to_excel(path, index=False)


def _write_vessel_info(frame: pd.DataFrame | None, output_dir: Path) -> None:
    if frame is None:
        raise ValueError("InputAdapter field vessel_berth_info is empty.")
    frame.to_csv(output_dir / "vessel_berth_info_new.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(output_dir / "vessel_berth_info.csv", index=False, encoding="utf-8-sig")


def _write_distance_matrix(frame: pd.DataFrame | None, path: Path) -> None:
    if frame is None:
        raise ValueError("InputAdapter field berth_area_dist_matrix is empty.")
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="距离矩阵", index=False)


def _write_closed_areas(closed_areas: set[str] | list[str] | tuple[str, ...] | None, path: Path) -> None:
    values = sorted(_normalize_code(area) for area in (closed_areas or []))
    path.write_text("\n".join(area for area in values if area), encoding="utf-8")


def _write_vessel_container_files(
    vessel_containers: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    export_vessels: list[str] = []
    import_vessels: list[str] = []
    import_dir = output_dir / "进口卸船箱"

    for raw_voyage, content in sorted((vessel_containers or {}).items()):
        voyage_id = _normalize_voyage(raw_voyage)
        if not voyage_id:
            continue
        doc = content.get("doc_cntrs")
        if doc is None:
            continue
        vessel_type = _normalize_code(content.get("type"))
        if vessel_type == "I":
            import_dir.mkdir(parents=True, exist_ok=True)
            doc.to_parquet(import_dir / f"container_info_import_{voyage_id}.parquet", index=False)
            import_vessels.append(voyage_id)
        else:
            doc.to_parquet(output_dir / f"container_info_{voyage_id}.parquet", index=False)
            export_vessels.append(voyage_id)
            predict_cntrs = content.get("predict_cntrs") or {}
            if predict_cntrs:
                _write_prediction_workbook(
                    output_dir / f"predict_data_{voyage_id}.xlsx",
                    predict_cntrs,
                    content.get("work_lanes"),
                )

    return sorted(export_vessels), sorted(import_vessels)


def _write_prediction_workbook(path: Path, predict_cntrs: dict[str, Any], work_lanes: Any) -> None:
    size_rows: list[dict[str, Any]] = []
    port_rows: list[dict[str, Any]] = []
    for raw_size, payload in sorted(predict_cntrs.items(), key=lambda item: _size_sort_key(item[0])):
        size = _normalize_size(raw_size)
        if not size:
            continue
        payload = payload or {}
        total_volume = _numeric_value(payload.get("total_volume"), 0)
        size_rows.append({"IYC_CSZ_CSIZECD": size, "count": total_volume})
        detail = payload.get("detail_info") or {}
        for port, count in sorted(detail.items()):
            port_rows.append(
                {
                    "IYC_CSZ_CSIZECD": size,
                    "IYC_POT_UNLDPORT": _normalize_text(port, "UNK"),
                    "count": _numeric_value(count, 0),
                }
            )

    work_lanes_value = _numeric_value(work_lanes, 0)
    work_lanes_rows = [{"作业路数": "预估", "数量": work_lanes_value}]

    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(size_rows, columns=["IYC_CSZ_CSIZECD", "count"]).to_excel(
            writer, sheet_name="尺寸统计", index=False
        )
        pd.DataFrame(port_rows, columns=["IYC_CSZ_CSIZECD", "IYC_POT_UNLDPORT", "count"]).to_excel(
            writer, sheet_name="尺寸港口统计", index=False
        )
        pd.DataFrame(work_lanes_rows, columns=["作业路数", "数量"]).to_excel(
            writer, sheet_name="作业路", index=False
        )


def _normalize_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def _normalize_voyage(value: Any) -> str:
    return _normalize_code(value)


def _normalize_size(value: Any) -> str:
    code = _normalize_code(value)
    if code.startswith("20"):
        return "20"
    if code.startswith("40"):
        return "40"
    if code.startswith("45"):
        return "45"
    return code


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().upper()
    return text or default


def _numeric_value(value: Any, default: float = 0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else float(default)


def _size_sort_key(value: Any) -> tuple[int, str]:
    size = _normalize_size(value)
    return (int(size) if size.isdigit() else 999, size)


def _frame_len(frame: pd.DataFrame | None) -> int:
    return int(len(frame)) if frame is not None else 0
