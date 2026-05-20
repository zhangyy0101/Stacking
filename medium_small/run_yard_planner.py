from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from block_bay_planning.data_loader import (
    DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
    DEFAULT_PLANNING_TIME,
    build_problem,
    parse_datetime,
    read_big_plan,
)
from block_bay_planning.demand_calculator import (
    DEFAULT_TARGET_VOYAGES,
    calculate_medium_demands,
    write_demand_rows,
)
from block_bay_planning.models import SAConfig
from block_bay_planning.sa_solver import SimulatedAnnealingSolver, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the medium yard plan from an existing big-plan CSV using simulated annealing, "
            "then construct the small bay plan from document-container groups."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--big-plan",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "large" / "outputs_large" / "latest_run" / "allocation.csv",
        help=(
            "Existing big-plan CSV. The default is large/outputs_large/latest_run/allocation.csv. "
            "For allocation.csv input, flow is preserved; missing flow defaults to OF. Supported columns include "
            "voy_id,flow,area_no,size,planned_qty or "
            "voyage_id,area_no,qty_20,qty_40[,plan_date] or "
            "voyage_id,area_no,planned_boxes[,size_mode,plan_date]."
        ),
    )
    parser.add_argument("--voyages", nargs="+", default=list(DEFAULT_TARGET_VOYAGES))
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--planning-time",
        default=DEFAULT_PLANNING_TIME.strftime("%Y-%m-%d %H:%M:%S"),
        help="Default: 2026-05-08 09:30:00.",
    )
    parser.add_argument("--horizon-hours", type=float, default=24.0)
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
        default=5,
        help="Rerun the medium-plan simulated annealing search when the area allocation cannot produce a feasible small plan.",
    )
    parser.add_argument(
        "--small-plan-check-every",
        type=int,
        default=3000,
        help=(
            "During medium-plan simulated annealing, test the current best assignment with the "
            "small-plan heuristic every N iterations and feed infeasible area combinations back "
            "immediately. Use 0 to disable in-loop checks."
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
        default=1,
        help=(
            "Evaluate the small-plan proxy every N simulated-annealing iterations. "
            "Use 1 for the exact current behavior; larger values run faster but make "
            "the medium-plan search less tightly guided by the small-plan proxy."
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
        "--misplaced-bay-exclusion-ratio",
        type=float,
        default=DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
        help=(
            "Exclude a bay from medium/small planning when currently occupied containers whose "
            "flow does not match the area's function exceed this share of the bay's slot capacity. "
            "Default: 0.6666667."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "sa_plan",
    )
    args = parser.parse_args()

    planning_time = parse_datetime(args.planning_time)
    if planning_time is None:
        raise SystemExit(f"Invalid --planning-time: {args.planning_time}")

    log("discovering data directory")
    data_dir = discover_data_dir(args.data_dir)
    output_dir = create_run_output_dir(args.output_dir)
    log(f"output directory: {output_dir}")

    log(f"reading big plan: {args.big_plan}")
    big_plan = read_big_plan(args.big_plan)
    log("calculating medium demand by discharge port")
    demand_rows = calculate_medium_demands(data_dir, args.voyages, planning_time)
    write_demand_rows(output_dir / "medium_demand_by_port.csv", demand_rows)

    log("building problem data: yard capacity, TOPS reservations, schedules, distances")
    problem = build_problem(
        data_dir,
        big_plan,
        planning_time=planning_time,
        horizon_hours=args.horizon_hours,
        target_voyages=args.voyages,
        misplaced_bay_exclusion_ratio=args.misplaced_bay_exclusion_ratio,
    )

    config = SAConfig(
        iterations=args.iterations,
        seed=args.seed,
        progress_every=args.energy_record_every,
        log_every=args.log_every,
        max_small_plan_retries=args.max_small_plan_retries,
        small_plan_check_every=args.small_plan_check_every,
        small_plan_proxy_height_capacity_penalty=args.small_plan_proxy_height_capacity_penalty,
        small_plan_proxy_every=args.small_plan_proxy_every,
        medium_concentrated_group_threshold=args.medium_concentrated_group_threshold,
    )
    solver = SimulatedAnnealingSolver(problem, config)
    log("starting simulated annealing and small-plan construction")
    result = solver.solve()

    log("writing output CSV and diagnostics")
    write_rows(output_dir / "medium_plan.csv", result.medium_rows)
    write_rows(output_dir / "small_plan.csv", result.small_rows)
    write_rows(output_dir / "small_plan_six_bay_blocks.csv", make_six_bay_block_rows(result.small_rows))
    write_rows(output_dir / "energy_convergence.csv", result.convergence_rows)
    write_rows(output_dir / "big_plan_used.csv", [row.__dict__ for row in problem.big_plan])
    (output_dir / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result.diagnostics, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")
    print(f"medium_demand_by_port: {output_dir / 'medium_demand_by_port.csv'}")
    print(f"medium_plan: {output_dir / 'medium_plan.csv'}")
    print(f"small_plan: {output_dir / 'small_plan.csv'}")
    print(f"small_plan_six_bay_blocks: {output_dir / 'small_plan_six_bay_blocks.csv'}")
    print(f"diagnostics: {output_dir / 'diagnostics.json'}")
    log("done")


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


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


def discover_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        return data_dir.resolve()
    roots = [Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent]
    required_files = {
        "bay_slots_detail.parquet",
        "tops_plan_info.parquet",
    }
    required_patterns = [
        "container_info_*.parquet",
        "predict_data_*.xlsx",
        "*功能*.xlsx",
    ]
    seen: set[Path] = set()
    for root in roots:
        try:
            candidates = [root, *root.iterdir()]
        except PermissionError:
            continue
        for candidate in candidates:
            if candidate in seen or not candidate.is_dir():
                continue
            seen.add(candidate)
            try:
                filenames = {path.name for path in candidate.iterdir() if path.is_file()}
            except PermissionError:
                continue
            if not required_files.issubset(filenames):
                continue
            has_container_info = any(candidate.glob("container_info_*.parquet"))
            has_prediction = any(candidate.glob("predict_data_*.xlsx"))
            has_area_function = all(
                any(candidate.glob(pattern))
                for pattern in required_patterns
                if pattern not in {"container_info_*.parquet", "predict_data_*.xlsx"}
            )
            if has_container_info and has_prediction and has_area_function:
                return candidate.resolve()
    raise FileNotFoundError(
        "No data directory found. Expected a directory containing bay_slots_detail.parquet, "
        "tops_plan_info.parquet, container_info_*.parquet, predict_data_*.xlsx, and *功能*.xlsx."
    )


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


if __name__ == "__main__":
    main()
