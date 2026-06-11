from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
DEFAULT_ADAPTER_JSON = SCRIPT_DIR / "data_examples" / "input_data.json"

from adapters import input_adapter_standard as adapter_data_io
from adapters.input_adapter_gd import InputAdapterGd
from adapters.input_adapter_standard import write_json
from api.planning_api import YardPlanAlgorithmConfig, solve_large_plan_df
from large_plan.solver import solve_daily_rolling_yard_plan

try:
    from large_plan.visualize import generate_yard_visualization
except ModuleNotFoundError as exc:
    generate_yard_visualization = None
    VISUALIZATION_IMPORT_ERROR = str(exc)
else:
    VISUALIZATION_IMPORT_ERROR = ""


def run_large_plan(
    input_adapter: InputAdapterGd,
    previous_large_plan_df: pd.DataFrame | None = None,
    config: YardPlanAlgorithmConfig | None = None,
) -> pd.DataFrame:
    return solve_large_plan_df(input_adapter, previous_large_plan_df, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the large yard plan from an InputAdapterGd JSON file.")
    parser.add_argument("--adapter-json", type=Path, default=DEFAULT_ADAPTER_JSON)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "large_plan_outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--large-time-limit", type=float, default=120.0)
    parser.add_argument("--large-mip-gap", type=float, default=0.001)
    parser.add_argument("--large-quiet", action="store_true")
    parser.add_argument("--no-write-large-state", action="store_true")
    parser.add_argument("--disable-default-flow-aliases", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = parse_args()
    run_dir = create_run_dir(args.output_root.resolve(), args.run_name)
    large_output_dir = run_dir / "outputs_large" / "latest_run"
    large_state_dir = run_dir / "outputs_large" / "state"
    for path in (large_output_dir, large_state_dir):
        path.mkdir(parents=True, exist_ok=False)

    adapter_json_path = resolve_adapter_json(args.adapter_json)
    adapter_input = InputAdapterGd.load_from_json(str(adapter_json_path))
    planning_time = resolve_planning_time(adapter_input)

    print(f"Adapter JSON: {adapter_json_path}")
    print("Input mode: InputAdapterGd object (no adapter_flat_data files are generated)")
    print(f"Run output directory: {run_dir}")

    print("\n[1/1] Building large-plan inputs")
    artifacts, state = adapter_data_io.build_large_inputs(
        adapter_input,
        planning_time,
        disable_default_flow_aliases=args.disable_default_flow_aliases,
    )
    print_case_summary(artifacts)

    print("\n[1/1] Solving large plan with weighted objectives")
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
    visualization_info = None
    large_visualization_dir = run_dir / "outputs_large" / "yard_visualization"
    if args.skip_visualization:
        visualization_info = {"skipped": True, "reason": "--skip-visualization"}
    elif generate_yard_visualization is None:
        visualization_info = {"skipped": True, "reason": VISUALIZATION_IMPORT_ERROR}
        print(f"large_visualization skipped: {VISUALIZATION_IMPORT_ERROR}")
    else:
        visualization_info = generate_yard_visualization(
            allocation_path=large_allocation_path,
            output_dir=large_visualization_dir,
            area_function_info=adapter_input.area_function_info,
        )
        print(f"large_visualization: {large_visualization_dir}")

    summary = {
        "planning_time": str(planning_time),
        "input_mode": "adapter_object",
        "adapter_json": str(adapter_json_path),
        "large_objective_mode": "weighted_sum",
        "large_output_dir": str(large_output_dir),
        "large_visualization_dir": str(large_visualization_dir),
        "large_visualization": visualization_info,
        "export_vessels": list(artifacts.export_vessels),
        "import_vessels": list(artifacts.import_vessels),
        "large_status": large_solution.status_name,
        "large_objective_value": large_solution.objective_value,
        "large_mip_gap": large_solution.mip_gap,
        "large_runtime": large_solution.runtime,
    }
    write_json(run_dir / "pipeline_summary.json", summary)

    print("\nLarge plan complete.")
    print(f"large_plan: {large_output_dir}")
    print(f"summary: {run_dir / 'pipeline_summary.json'}")


def resolve_adapter_json(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.is_dir():
        resolved = resolved / "input_data.json"
    if not resolved.exists():
        raise FileNotFoundError(f"InputAdapterGd JSON not found: {resolved}")
    return resolved


def resolve_planning_time(adapter_input: InputAdapterGd) -> pd.Timestamp:
    adapter_planning_time = getattr(adapter_input, "planning_time", None)
    if adapter_planning_time is None or pd.isna(adapter_planning_time):
        raise SystemExit("InputAdapterGd JSON field planning_time is empty.")
    planning_time = pd.Timestamp(adapter_planning_time)
    if pd.isna(planning_time):
        raise SystemExit(f"Invalid InputAdapterGd planning_time: {adapter_planning_time}")
    return planning_time


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
    print(f"user_design_active: {artifacts.diagnostics.get('user_design_active')}")
    print(f"user_design_large_plan_area: {artifacts.diagnostics.get('user_design_large_plan_area')}")
    print(f"old_vessels: {artifacts.diagnostics.get('old_vessels')}")
    print(f"departure_operation_deduction_total: {artifacts.diagnostics.get('departure_operation_deduction_total')}")
    print(f"close_berth_conflict_pairs: {artifacts.diagnostics.get('close_berth_conflict_pairs')}")
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
    print(f"required_area_unmet: {getattr(solution, 'required_area_unmet', {})}")


if __name__ == "__main__":
    main()
