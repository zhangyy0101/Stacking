from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from .data_loader import (
    DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
    DEFAULT_PLANNING_TIME,
    build_problem,
    infer_target_voyages_from_big_plan,
    medium_demand_caps_from_big_plan,
    parse_datetime,
    read_big_plan,
)
from .demand_calculator import calculate_medium_demands, write_demand_rows
from .input_json import has_input_json, input_value
from .models import SAConfig
from .sa_solver import SimulatedAnnealingSolver, write_rows


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
        help="Voyages to plan. By default, voyages with active OF rows in the big-plan CSV are inferred.",
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


def add_sa_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--energy-record-every", type=int, default=3000)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print simulated-annealing progress every N iterations. Use 0 to disable iteration logs.",
    )
    parser.add_argument(
        "--max-small-plan-retries",
        type=int,
        default=0,
        help="Rerun the medium-plan simulated annealing search when the area allocation cannot produce a feasible small plan.",
    )
    parser.add_argument(
        "--small-plan-check-every",
        type=int,
        default=0,
        help=(
            "During medium-plan simulated annealing, test the current best assignment with the "
            "small-plan heuristic every N iterations and feed infeasible area combinations back "
            "immediately. Use 0 to disable in-loop checks."
        ),
    )
    parser.add_argument(
        "--medium-initial-assignment-attempts",
        type=int,
        default=30,
        help=(
            "Number of greedy fallback attempts for the medium-plan initial assignment "
            "after the SCIP feasibility model is unavailable or fails."
        ),
    )
    parser.add_argument(
        "--small-plan-proxy-height-capacity-penalty",
        type=float,
        default=240.0,
        help=(
            "Penalty for projected small-plan size+height demand exceeding bay-feasible "
            "area capacity in the medium-plan proxy score."
        ),
    )
    parser.add_argument(
        "--small-plan-proxy-every",
        type=int,
        default=0,
        help=(
            "Evaluate the small-plan proxy every N simulated-annealing iterations. "
            "Use 0 to disable. With medium/small feedback enabled this is usually unnecessary."
        ),
    )
    parser.add_argument(
        "--medium-small-feedback-iterations",
        type=int,
        default=10,
        help=(
            "Run structured medium/small feedback rounds before SA: solve medium by SCIP, "
            "construct/repair the small plan, learn area-size caps from the repair, and re-solve. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--medium-small-feedback-cap-penalty",
        type=float,
        default=20000.0,
        help=(
            "SCIP/SA penalty per box for exceeding a small-plan-learned voyage/flow/area/size cap."
        ),
    )
    parser.add_argument(
        "--medium-concentrated-group-threshold",
        type=int,
        default=26,
        help=(
            "Medium-plan coarse groups with demand at or below this threshold prefer concentrated "
            "yard-area assignment instead of proportional area balancing. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--small-plan-strict-feedback-penalty",
        type=float,
        default=45.0,
        help=(
            "Soft penalty per box for area combinations that fail strict small-plan inheritance "
            "but can still be repaired by direct bay repair."
        ),
    )
    parser.add_argument(
        "--small-plan-repair-failure-feedback-multiplier",
        type=float,
        default=4.0,
        help=(
            "Multiplier applied to small-plan infeasibility feedback when both "
            "strict inheritance and direct bay repair fail."
        ),
    )
    parser.add_argument("--fine-group-area-penalty", type=float, default=80.0)
    parser.add_argument("--medium-small-group-area-split-penalty", type=float, default=1500.0)
    parser.add_argument("--medium-small-group-fragment-penalty", type=float, default=80.0)
    parser.add_argument("--medium-large-group-min-area-boxes", type=int, default=10)
    parser.add_argument("--medium-large-group-small-area-penalty", type=float, default=1500.0)
    parser.add_argument(
        "--big-plan-area-deviation-penalty",
        type=float,
        default=30.0,
        help="Per-box penalty for deviating from the inherited big-plan voyage/area/size pattern.",
    )
    parser.add_argument(
        "--big-plan-fallback-tier-penalty",
        type=float,
        default=500.0,
        help="Base per-box penalty for each fallback tier away from exact big-plan new_qty inheritance.",
    )
    parser.add_argument(
        "--berth-distance-penalty",
        type=float,
        default=3.0,
        help="Weight for berth-to-yard distance, applied as distance / 100 per voyage-area.",
    )
    parser.add_argument(
        "--active-loading-area-penalty",
        type=float,
        default=1000.0,
        help="Penalty for using a yard area that has loading during the voyage window.",
    )
    parser.add_argument(
        "--post-window-loading-area-reward",
        type=float,
        default=1000.0,
        help="Reward for using a yard area with loading in the 24h after the voyage window.",
    )


def make_sa_config(args: argparse.Namespace) -> SAConfig:
    return SAConfig(
        iterations=args.iterations,
        seed=args.seed,
        progress_every=args.energy_record_every,
        log_every=args.log_every,
        max_small_plan_retries=args.max_small_plan_retries,
        small_plan_check_every=args.small_plan_check_every,
        medium_initial_assignment_attempts=args.medium_initial_assignment_attempts,
        small_plan_proxy_height_capacity_penalty=args.small_plan_proxy_height_capacity_penalty,
        small_plan_strict_feedback_penalty=args.small_plan_strict_feedback_penalty,
        small_plan_repair_failure_feedback_multiplier=args.small_plan_repair_failure_feedback_multiplier,
        small_plan_proxy_every=args.small_plan_proxy_every,
        medium_concentrated_group_threshold=args.medium_concentrated_group_threshold,
        small_plan_group_area_split_penalty=args.fine_group_area_penalty,
        medium_small_group_area_split_penalty=args.medium_small_group_area_split_penalty,
        medium_small_group_fragment_penalty=args.medium_small_group_fragment_penalty,
        medium_large_group_min_area_boxes=args.medium_large_group_min_area_boxes,
        medium_large_group_small_area_penalty=args.medium_large_group_small_area_penalty,
        big_plan_area_deviation_penalty=args.big_plan_area_deviation_penalty,
        big_plan_fallback_tier_penalty=args.big_plan_fallback_tier_penalty,
        berth_distance_penalty=args.berth_distance_penalty,
        active_loading_area_penalty=args.active_loading_area_penalty,
        post_window_loading_area_reward=args.post_window_loading_area_reward,
        medium_small_feedback_iterations=args.medium_small_feedback_iterations,
        medium_small_feedback_cap_penalty=args.medium_small_feedback_cap_penalty,
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


def build_problem_from_big_plan(args: argparse.Namespace) -> tuple[Path, datetime, object, list]:
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


def build_problem_from_medium_plan(args: argparse.Namespace) -> tuple[Path, datetime, object, list[dict]]:
    data_dir, planning_time, problem, _big_plan = build_problem_from_big_plan(args)
    medium_rows = read_csv_rows(args.medium_plan)
    return data_dir, planning_time, problem, medium_rows


def read_csv_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


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


def write_medium_artifacts(output_dir: Path, result, problem, demand_rows: list | None = None) -> None:
    if demand_rows is not None:
        write_demand_rows(output_dir / "medium_demand_by_port.csv", demand_rows)
    write_rows(output_dir / "medium_plan.csv", result.medium_rows)
    write_rows(output_dir / "energy_convergence.csv", result.convergence_rows)
    write_rows(output_dir / "big_plan_used.csv", [row.__dict__ for row in problem.big_plan])
    write_json(output_dir / "diagnostics.json", result.diagnostics)


def write_small_artifacts(output_dir: Path, small_rows: list[dict], diagnostics: dict | None = None) -> None:
    write_rows(output_dir / "small_plan.csv", small_rows)
    write_rows(output_dir / "small_plan_six_bay_blocks.csv", make_six_bay_block_rows(small_rows))
    if diagnostics is not None:
        write_json(output_dir / "diagnostics.json", diagnostics)


def make_six_bay_block_rows(small_rows: list[dict]) -> list[dict]:
    blocks: dict[str, dict] = {}
    details: dict[str, dict[str, int]] = {}
    for row in small_rows:
        block_id = str(row.get("six_bay_block_id") or "")
        if not block_id:
            continue
        blocks.setdefault(
            block_id,
            {
                "plan_level": "small_six_bay_block",
                "six_bay_block_id": block_id,
                "area_no": row.get("area_no", ""),
                "six_bay_block_bays": row.get("six_bay_block_bays", ""),
                "six_bay_block_total_boxes": row.get("six_bay_block_total_boxes", 0),
            },
        )
        detail_key = str(row.get("group_id", ""))
        details.setdefault(block_id, {})
        details[block_id][detail_key] = details[block_id].get(detail_key, 0) + int(row.get("planned_boxes", 0) or 0)

    out = []
    for block_id, row in sorted(blocks.items()):
        row = dict(row)
        row["allocation_summary"] = ";".join(
            f"{detail}|{qty}" for detail, qty in sorted(details.get(block_id, {}).items())
        )
        out.append(row)
    return out


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def calculate_and_write_demand(data_dir: Path, output_dir: Path, args: argparse.Namespace, planning_time: datetime) -> list:
    big_plan = read_big_plan(args.big_plan)
    big_plan_caps = medium_demand_caps_from_big_plan(big_plan, args.voyages, planning_time)
    demand_rows = calculate_medium_demands(
        data_dir,
        args.voyages,
        planning_time,
        args.horizon_hours,
        big_plan_caps=big_plan_caps,
    )
    write_demand_rows(output_dir / "medium_demand_by_port.csv", demand_rows)
    return demand_rows


__all__ = [
    "SimulatedAnnealingSolver",
    "add_big_plan_argument",
    "add_common_arguments",
    "add_sa_arguments",
    "build_problem_from_big_plan",
    "build_problem_from_medium_plan",
    "calculate_and_write_demand",
    "create_run_output_dir",
    "log",
    "make_sa_config",
    "make_six_bay_block_rows",
    "write_json",
    "write_medium_artifacts",
    "write_rows",
    "write_small_artifacts",
]
