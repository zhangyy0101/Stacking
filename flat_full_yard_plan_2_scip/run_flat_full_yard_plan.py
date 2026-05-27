from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_JSON = SCRIPT_DIR / "input_data.json"

import input_adapter_standard as adapter_data_io
from input_adapter_gd import InputAdapterGd
from input_adapter_standard import (
    DEFAULT_EXPORT_VESSELS,
    DEFAULT_IMPORT_VESSELS,
    DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
    DEFAULT_PLANNING_TIME,
    parse_datetime,
    parse_planning_time,
    write_json,
)
from planning_large_solver import solve_daily_rolling_yard_plan
from planning_large_visualize import generate_yard_visualization
from block_bay_planning.models import SAConfig
from block_bay_planning.sa_solver import SimulatedAnnealingSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete large, medium, and small yard planning pipeline from an InputAdapterGd JSON file."
    )
    parser.add_argument(
        "--adapter-json",
        type=Path,
        default=DEFAULT_ADAPTER_JSON,
        help="InputAdapterGd JSON file, or a pro_test_data_* directory containing input_data.json.",
    )
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "full_plan_outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--export-vessels", nargs="+", default=DEFAULT_EXPORT_VESSELS)
    parser.add_argument("--import-vessels", nargs="+", default=DEFAULT_IMPORT_VESSELS)
    parser.add_argument("--medium-voyages", nargs="+", default=None)
    parser.add_argument("--large-time-limit", type=float, default=120.0)
    parser.add_argument("--large-mip-gap", type=float, default=0.001)
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
    run_dir = create_run_dir(args.output_root.resolve(), args.run_name)
    large_output_dir = run_dir / "outputs_large" / "latest_run"
    large_state_dir = run_dir / "outputs_large" / "state"
    medium_small_output_dir = run_dir / "medium_small_plan"
    for path in (large_output_dir, large_state_dir, medium_small_output_dir):
        path.mkdir(parents=True, exist_ok=False)

    adapter_json_path = resolve_adapter_json(args.adapter_json)
    adapter_input = InputAdapterGd.load_from_json(str(adapter_json_path))

    planning_time_value = args.planning_time
    if planning_time_value is None:
        adapter_planning_time = getattr(adapter_input, "planning_time", None)
        if adapter_planning_time is not None and not pd.isna(adapter_planning_time):
            planning_time_value = pd.Timestamp(adapter_planning_time).strftime("%Y-%m-%d %H:%M:%S")
    planning_time_value = planning_time_value or DEFAULT_PLANNING_TIME
    planning_time = parse_planning_time(planning_time_value)
    medium_planning_time = parse_datetime(planning_time_value)
    if medium_planning_time is None:
        raise SystemExit(f"Invalid --planning-time: {planning_time_value}")

    print(f"Adapter JSON: {adapter_json_path}")
    print("Input mode: InputAdapterGd object (no adapter_flat_data files are generated)")
    print(f"Run output directory: {run_dir}")

    print("\n[1/2] Building large-plan inputs")
    artifacts, state = adapter_data_io.build_large_inputs(
        adapter_input,
        planning_time,
        export_vessels=args.export_vessels,
        import_vessels=args.import_vessels,
        disable_default_flow_aliases=args.disable_default_flow_aliases,
    )
    print_case_summary(artifacts)
    medium_voyages = list(args.medium_voyages or artifacts.export_vessels)

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
    adapter_data_io.write_large_outputs(large_output_dir, artifacts, large_solution, large_state_rows)

    large_allocation_path = large_output_dir / "allocation.csv"
    if not large_allocation_path.exists():
        raise FileNotFoundError(f"Large allocation was not written: {large_allocation_path}")
    large_visualization_dir = run_dir / "outputs_large" / "yard_visualization"
    visualization_kwargs = {
        "allocation_path": large_allocation_path,
        "output_dir": large_visualization_dir,
        "area_function_info": adapter_input.area_function_info,
    }
    visualization_info = generate_yard_visualization(**visualization_kwargs)
    print(f"large_visualization: {large_visualization_dir}")

    print("\n[2/2] Building medium/small inputs")
    large_allocation_frame = pd.DataFrame(adapter_data_io.allocation_output_rows(large_solution, artifacts.data))
    medium_inputs = adapter_data_io.load_medium_small_inputs(
        adapter_input,
        planning_time=medium_planning_time,
        voyages=medium_voyages,
        horizon_hours=args.horizon_hours,
        misplaced_bay_exclusion_ratio=args.misplaced_bay_exclusion_ratio,
        big_plan=large_allocation_frame,
    )
    adapter_data_io.write_demand_rows(medium_small_output_dir / "medium_demand_by_port.csv", medium_inputs.demand_rows)

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
        "input_mode": "adapter_object",
        "data_dir": None,
        "adapter_json": str(adapter_json_path),
        "adapter_flat_data_metadata": None,
        "large_objective_mode": "weighted_sum",
        "large_output_dir": str(large_output_dir),
        "large_visualization_dir": str(large_visualization_dir),
        "large_visualization": visualization_info,
        "medium_small_output_dir": str(medium_small_output_dir),
        "large_allocation_output": str(large_allocation_path),
        "large_allocation_used_by_medium_small": "in_memory",
        "export_vessels": list(artifacts.export_vessels),
        "import_vessels": list(artifacts.import_vessels),
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


def resolve_adapter_json(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.is_dir():
        resolved = resolved / "input_data.json"
    if not resolved.exists():
        raise FileNotFoundError(f"InputAdapterGd JSON not found: {resolved}")
    return resolved


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
