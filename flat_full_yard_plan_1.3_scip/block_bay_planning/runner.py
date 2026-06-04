from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .data_loader import (
    DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
    DEFAULT_PLANNING_TIME,
    build_problem,
    infer_target_voyages_from_big_plan,
    parse_datetime,
    read_big_plan,
)
from .input_json import has_input_json, input_value

DEFAULT_DATASET = "0519"
LEGACY_0519_DATA_DIR_NAME = "\u5806\u5b58\u8ba1\u5212\u6d4b\u8bd5\u6570\u636e20260519"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir(dataset: str = DEFAULT_DATASET) -> Path:
    root = repo_root()
    if dataset == "0508":
        return root / "test_data" / "pro_test_data_0508"
    return root / LEGACY_0519_DATA_DIR_NAME


def default_big_plan_path(dataset: str = DEFAULT_DATASET) -> Path:
    root = repo_root()
    if dataset == "0508":
        return (
            root
            / "flat_full_yard_plan_scip"
            / "full_plan_outputs"
            / "run_scip_visual_20260521_001"
            / "outputs_large"
            / "latest_run"
            / "allocation.csv"
        )
    return root / "large" / "outputs_large" / "latest_run" / "allocation.csv"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=["0519", "0508"],
        default=DEFAULT_DATASET,
        help="Dataset preset to use when --data-dir/--big-plan are not provided. Default: 0519.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--voyages",
        nargs="+",
        default=None,
        help="Voyages to plan. By default, voyages with OF rows in the active big-plan CSV are planned.",
    )
    parser.add_argument(
        "--planning-time",
        default=None,
        help=(
            "Planning time override. Raw 0519 defaults to 2026-05-19 09:30:00; "
            "JSON data directories use input_data.json when this is omitted."
        ),
    )
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument(
        "--misplaced-bay-exclusion-ratio",
        type=float,
        default=DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
        help=(
            "Exclude a bay from medium/small planning when currently occupied containers whose "
            "flow does not match the area's function exceed this share of the bay's slot capacity. "
            "Default: 0.6666667."
        ),
    )


def add_big_plan_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--big-plan",
        type=Path,
        default=None,
        help=(
            "Existing big-plan CSV. By default, 0519 uses large/outputs_large/latest_run/allocation.csv, "
            "and 0508 uses flat_full_yard_plan_scip/full_plan_outputs/run_scip_visual_20260521_001/"
            "outputs_large/latest_run/allocation.csv. "
            "For allocation.csv input, flow is preserved; missing flow defaults to OF. Supported columns include "
            "voy_id,flow,area_no,size,new_qty or "
            "voyage_id,area_no,qty_20,qty_40[,plan_date] or "
            "voyage_id,area_no,new_qty[,size_mode,plan_date]."
        ),
    )


def parse_required_planning_time(value: object) -> datetime:
    planning_time = parse_datetime(value)
    if planning_time is None:
        raise SystemExit(f"Invalid --planning-time: {value}")
    return planning_time


def resolve_planning_time(args: argparse.Namespace, data_dir: Path) -> datetime:
    if getattr(args, "planning_time", None):
        return parse_required_planning_time(args.planning_time)
    return read_planning_time_from_data_dir(data_dir) or DEFAULT_PLANNING_TIME


def read_planning_time_from_data_dir(data_dir: str | Path) -> datetime | None:
    if not has_input_json(data_dir):
        return None
    raw_value = input_value(data_dir, "planning_time")
    planning_time = parse_datetime(raw_value)
    if raw_value is not None and planning_time is None:
        raise SystemExit(f"Invalid planning_time in input_data.json: {raw_value}")
    return planning_time


def _normalize_voyage_arg(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def discover_data_dir(data_dir: Path | None, dataset: str = DEFAULT_DATASET) -> Path:
    if data_dir is not None:
        return data_dir.resolve()
    root = repo_root()
    preferred = default_data_dir(dataset)
    json_fallback = root / "test_data" / "pro_test_data_0519"
    roots = [preferred, json_fallback, root, Path(__file__).resolve().parents[1]]
    required_files = {"bay_slots_detail.parquet", "tops_plan_info.parquet", "vessel_berth_info.csv"}
    seen: set[Path] = set()
    for root in roots:
        if root.is_dir() and _looks_like_data_dir(root, required_files):
            return root.resolve()
        try:
            candidates = [root, *root.iterdir()]
        except PermissionError:
            continue
        for candidate in candidates:
            if candidate in seen or not candidate.is_dir():
                continue
            seen.add(candidate)
            if _looks_like_data_dir(candidate, required_files):
                return candidate.resolve()
    message = (
        "No data directory found. Expected a directory containing bay_slots_detail.parquet, "
        "tops_plan_info.parquet, container_info_*.parquet, predict_data_*.xlsx, and an area-function workbook."
    )
    raise FileNotFoundError(message)


def build_problem_from_big_plan(args: argparse.Namespace):
    dataset = getattr(args, "dataset", DEFAULT_DATASET)
    data_dir = discover_data_dir(args.data_dir, dataset)
    planning_time = resolve_planning_time(args, data_dir)
    big_plan_path = args.big_plan or default_big_plan_path(dataset)
    args.big_plan = big_plan_path
    big_plan = read_big_plan(big_plan_path)
    target_voyages = (
        [_normalize_voyage_arg(voyage) for voyage in args.voyages]
        if args.voyages
        else infer_target_voyages_from_big_plan(big_plan, planning_time)
    )
    args.voyages = target_voyages
    problem = build_problem(
        data_dir,
        big_plan,
        planning_time=planning_time,
        horizon_hours=args.horizon_hours,
        target_voyages=target_voyages,
        misplaced_bay_exclusion_ratio=args.misplaced_bay_exclusion_ratio,
    )
    return data_dir, planning_time, problem, big_plan


def _looks_like_data_dir(candidate: Path, required_files: set[str]) -> bool:
    try:
        filenames = {path.name for path in candidate.iterdir() if path.is_file()}
    except PermissionError:
        return False
    if "input_data.json" in filenames:
        return True
    if not required_files.issubset(filenames):
        return False
    has_container_info = any(candidate.glob("container_info_*.parquet"))
    has_prediction_or_docs = any(candidate.glob("predict_data_*.xlsx")) or has_container_info
    has_area_function = any("\u529f\u80fd" in path.name for path in candidate.glob("*.xlsx"))
    return has_container_info and has_prediction_or_docs and has_area_function


def create_run_output_dir(output_root: Path) -> Path:
    root = output_root.resolve()
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{timestamp}_{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


__all__ = [
    "add_big_plan_argument",
    "add_common_arguments",
    "build_problem_from_big_plan",
    "create_run_output_dir",
    "discover_data_dir",
    "log",
    "parse_required_planning_time",
    "write_json",
]
