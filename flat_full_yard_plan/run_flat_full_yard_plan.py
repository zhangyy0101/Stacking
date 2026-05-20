from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (REPO_ROOT / "large", REPO_ROOT / "medium_small"):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from flat_yard_plan_data_io import (
    DEFAULT_EXPORT_VESSELS,
    DEFAULT_IMPORT_VESSELS,
    DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
    DEFAULT_PLANNING_TIME,
    build_large_inputs,
    load_medium_small_inputs,
    parse_datetime,
    parse_planning_time,
    write_demand_rows,
    write_json,
    write_large_outputs,
)
from planning_large_solver import solve_daily_rolling_yard_plan
from block_bay_planning.models import SAConfig
from block_bay_planning.sa_solver import SimulatedAnnealingSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete large, medium, and small yard planning pipeline from one flat data folder."
    )
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "full_plan_outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--planning-time", default=DEFAULT_PLANNING_TIME)
    parser.add_argument("--export-vessels", nargs="+", default=DEFAULT_EXPORT_VESSELS)
    parser.add_argument("--import-vessels", nargs="+", default=DEFAULT_IMPORT_VESSELS)
    parser.add_argument("--medium-voyages", nargs="+", default=None)
    parser.add_argument("--large-time-limit", type=float, default=120.0)
    parser.add_argument("--large-mip-gap", type=float, default=0.0)
    parser.add_argument("--large-quiet", action="store_true")
    parser.add_argument("--no-write-large-state", action="store_true")
    parser.add_argument("--disable-default-flow-aliases", action="store_true")
    parser.add_argument("--medium-iterations", type=int, default=30000)
    parser.add_argument("--medium-seed", type=int, default=7)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--energy-record-every", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--max-small-plan-retries", type=int, default=5)
    parser.add_argument("--small-plan-check-every", type=int, default=3000)
    parser.add_argument("--small-plan-proxy-height-capacity-penalty", type=float, default=240.0)
    parser.add_argument("--small-plan-proxy-every", type=int, default=1)
    parser.add_argument("--misplaced-bay-exclusion-ratio", type=float, default=DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = parse_args()
    planning_time = parse_planning_time(args.planning_time)
    medium_planning_time = parse_datetime(args.planning_time)
    if medium_planning_time is None:
        raise SystemExit(f"Invalid --planning-time: {args.planning_time}")

    data_dir = args.data_dir.resolve()
    run_dir = create_run_dir(args.output_root.resolve(), args.run_name)
    large_output_dir = run_dir / "outputs_large" / "latest_run"
    large_state_dir = run_dir / "outputs_large" / "state"
    medium_small_output_dir = run_dir / "medium_small_plan"
    for path in (large_output_dir, large_state_dir, medium_small_output_dir):
        path.mkdir(parents=True, exist_ok=False)

    medium_voyages = list(args.medium_voyages or args.export_vessels)

    print(f"Flat data directory: {data_dir}")
    print(f"Run output directory: {run_dir}")

    print("\n[1/2] Building flat-data large-plan inputs")
    artifacts, state = build_large_inputs(
        data_dir,
        large_state_dir,
        planning_time,
        export_vessels=args.export_vessels,
        import_vessels=args.import_vessels,
        disable_default_flow_aliases=args.disable_default_flow_aliases,
    )
    print_case_summary(artifacts)

    print("\n[1/2] Solving large plan with weighted objectives")
    large_solution = solve_daily_rolling_yard_plan(
        artifacts.data,
        time_limit=args.large_time_limit,
        mip_gap=args.large_mip_gap,
        verbose=not args.large_quiet,
    )
    print_solution_summary(large_solution)

    large_state_rows = pd.DataFrame()
    if not args.no_write_large_state and large_solution.objective_value is not None:
        large_state_rows = state.append_solution(planning_time, large_solution)
    write_large_outputs(large_output_dir, artifacts, large_solution, large_state_rows)

    large_allocation_path = large_output_dir / "allocation.csv"
    if not large_allocation_path.exists():
        raise FileNotFoundError(f"Large allocation was not written: {large_allocation_path}")

    print("\n[2/2] Building flat-data medium/small inputs")
    medium_inputs = load_medium_small_inputs(
        data_dir,
        large_allocation_path,
        planning_time=medium_planning_time,
        voyages=medium_voyages,
        horizon_hours=args.horizon_hours,
        misplaced_bay_exclusion_ratio=args.misplaced_bay_exclusion_ratio,
    )
    write_demand_rows(medium_small_output_dir / "medium_demand_by_port.csv", medium_inputs.demand_rows)

    print("[2/2] Solving medium and small plans")
    config = SAConfig(
        iterations=args.medium_iterations,
        seed=args.medium_seed,
        progress_every=args.energy_record_every,
        log_every=args.log_every,
        max_small_plan_retries=args.max_small_plan_retries,
        small_plan_check_every=args.small_plan_check_every,
        small_plan_proxy_height_capacity_penalty=args.small_plan_proxy_height_capacity_penalty,
        small_plan_proxy_every=args.small_plan_proxy_every,
    )
    solver = SimulatedAnnealingSolver(medium_inputs.problem, config)
    result = solver.solve()

    write_rows(medium_small_output_dir / "medium_plan.csv", result.medium_rows)
    write_rows(medium_small_output_dir / "small_plan.csv", result.small_rows)
    write_rows(medium_small_output_dir / "small_plan_six_bay_blocks.csv", make_six_bay_block_rows(result.small_rows))
    write_rows(medium_small_output_dir / "energy_convergence.csv", result.convergence_rows)
    write_rows(medium_small_output_dir / "big_plan_used.csv", [row.__dict__ for row in medium_inputs.problem.big_plan])
    write_json(medium_small_output_dir / "diagnostics.json", result.diagnostics)

    summary = {
        "planning_time": str(planning_time),
        "data_dir": str(data_dir),
        "large_objective_mode": "weighted_sum",
        "large_output_dir": str(large_output_dir),
        "medium_small_output_dir": str(medium_small_output_dir),
        "large_allocation_used_by_medium_small": str(large_allocation_path),
        "export_vessels": list(args.export_vessels),
        "import_vessels": list(args.import_vessels),
        "medium_voyages": medium_voyages,
        "large_status": large_solution.status_name,
        "large_objective_value": large_solution.objective_value,
        "medium_energy": result.energy,
        "medium_row_count": len(result.medium_rows),
        "small_row_count": len(result.small_rows),
    }
    write_json(run_dir / "pipeline_summary.json", summary)

    print("\nPipeline complete.")
    print(f"large_plan: {large_output_dir}")
    print(f"medium_small_plan: {medium_small_output_dir}")
    print(f"summary: {run_dir / 'pipeline_summary.json'}")


def create_run_dir(output_root: Path, run_name: str | None) -> Path:
    name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def print_case_summary(artifacts) -> None:
    data = artifacts.data
    print(f"planning_time: {artifacts.planning_time}")
    print(f"export_vessels: {artifacts.export_vessels}")
    print(f"import_vessels: {artifacts.import_vessels}")
    print(f"berth_by_vessel: {artifacts.berth_by_vessel}")
    print(f"area_count: {len(data.A)}")
    print(f"flows: {list(data.F)}")
    print(f"demand20_total: {sum(data.D20.values())}")
    print(f"demand40_total: {sum(data.D40.values())}")
    print(f"capacity20_equiv_total: {sum(data.C20.values())}")
    print(f"capacity20_direct_total: {sum((data.C20Direct or {}).values())}")
    print(f"capacity40_total: {sum(data.C40.values())}")
    print(f"bad_bay_count: {artifacts.diagnostics.get('bad_bay_count')}")
    print(f"active_tops_rows: {artifacts.diagnostics.get('active_tops_rows')}")
    print(f"old_vessels: {artifacts.diagnostics.get('old_vessels')}")
    print(f"of_work_lanes: {artifacts.diagnostics.get('of_work_lanes')}")
    print(f"of_area_limits: {artifacts.diagnostics.get('of_area_limits')}")


def print_solution_summary(solution) -> None:
    print(f"status: {solution.status_name}")
    print(f"objective_value: {solution.objective_value}")
    print(f"mip_gap: {solution.mip_gap}")
    print(f"runtime: {solution.runtime}")
    print(f"objective_components: {solution.objective_components}")
    print(f"unmet20_total: {sum(solution.s20.values())}")
    print(f"unmet40_total: {sum(solution.s40.values())}")
    print(f"operation_overage_total: {sum(solution.o.values())}")
    print(f"of_area_overage_total: {sum(solution.of_area_over.values())}")
    print(f"area_share_overage_total: {sum(solution.h.values())}")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        group_id = str(row.get("group_id", ""))
        details.setdefault(block_id, {})
        details[block_id][group_id] = details[block_id].get(group_id, 0) + int(row.get("planned_boxes", 0) or 0)
    out = []
    for block_id, row in sorted(blocks.items()):
        row = dict(row)
        row["allocation_summary"] = ";".join(
            f"{group_id}|{qty}" for group_id, qty in sorted(details.get(block_id, {}).items())
        )
        out.append(row)
    return out


if __name__ == "__main__":
    main()
