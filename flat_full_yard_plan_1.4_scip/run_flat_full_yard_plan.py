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
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
DEFAULT_ADAPTER_JSON = SCRIPT_DIR / "data_examples" / "input_data.json"

from adapters import input_adapter_standard as adapter_data_io
from adapters.input_adapter_gd import InputAdapterGd
from adapters.input_adapter_standard import (
    DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
    write_json,
)
from api.planning_api import YardPlanAlgorithmConfig, solve_full_yard_plan_df
from large_plan.solver import solve_daily_rolling_yard_plan
try:
    from large_plan.visualize import generate_yard_visualization
except ModuleNotFoundError as exc:
    generate_yard_visualization = None
    VISUALIZATION_IMPORT_ERROR = str(exc)
else:
    VISUALIZATION_IMPORT_ERROR = ""


def run_full_yard_plan(
    input_adapter: InputAdapterGd,
    previous_large_plan_df: pd.DataFrame | None = None,
    config: YardPlanAlgorithmConfig | None = None,
) -> dict[str, pd.DataFrame]:
    return solve_full_yard_plan_df(input_adapter, previous_large_plan_df, config)
from medium_small.bridge import (
    add_column_generation_arguments,
    build_problem_from_large_plan_records_adapter,
    print_diagnostics_summary,
    solve_medium_small_problem,
    write_medium_demand_rows_from_rows,
)


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
    parser.add_argument("--medium-voyages", nargs="+", default=None)
    parser.add_argument("--large-time-limit", type=float, default=120.0)
    parser.add_argument("--large-mip-gap", type=float, default=0.001)
    parser.add_argument("--large-quiet", action="store_true")
    parser.add_argument("--no-write-large-state", action="store_true")
    parser.add_argument("--disable-default-flow-aliases", action="store_true")
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--misplaced-bay-exclusion-ratio", type=float, default=DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO)
    add_column_generation_arguments(parser)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = parse_args()
    algorithm_config = YardPlanAlgorithmConfig.from_cli_args(args)
    run_dir = create_run_dir(args.output_root.resolve(), args.run_name)
    large_output_dir = run_dir / "outputs_large" / "latest_run"
    large_state_dir = run_dir / "outputs_large" / "state"
    medium_small_output_dir = run_dir / "medium_small_plan"
    for path in (large_output_dir, large_state_dir, medium_small_output_dir):
        path.mkdir(parents=True, exist_ok=False)

    adapter_json_path = resolve_adapter_json(args.adapter_json)
    adapter_input = InputAdapterGd.load_from_json(str(adapter_json_path))

    adapter_planning_time = getattr(adapter_input, "planning_time", None)
    if adapter_planning_time is None or pd.isna(adapter_planning_time):
        raise SystemExit("InputAdapterGd JSON field planning_time is empty.")
    planning_time = pd.Timestamp(adapter_planning_time)
    if pd.isna(planning_time):
        raise SystemExit(f"Invalid InputAdapterGd planning_time: {adapter_planning_time}")
    medium_planning_time = planning_time.to_pydatetime()

    print(f"Adapter JSON: {adapter_json_path}")
    print("Input mode: InputAdapterGd object (no adapter_flat_data files are generated)")
    print(f"Run output directory: {run_dir}")

    print("\n[1/2] Building large-plan inputs")
    artifacts, state = adapter_data_io.build_large_inputs(
        adapter_input,
        planning_time,
        disable_default_flow_aliases=algorithm_config.disable_default_flow_aliases,
        config=algorithm_config.large,
    )
    print_case_summary(artifacts)
    medium_voyages = list(
        algorithm_config.medium_voyages
        or (list(artifacts.export_vessels) + list(artifacts.import_vessels))
    )

    print("\n[1/2] Solving large plan with weighted objectives")
    large_solution = solve_daily_rolling_yard_plan(
        artifacts.data,
        time_limit=algorithm_config.large_time_limit,
        mip_gap=algorithm_config.large_mip_gap,
        verbose=algorithm_config.large_verbose,
    )
    print_solution_summary(large_solution)

    large_state_rows = pd.DataFrame()
    if not args.no_write_large_state and large_solution.objective_value is not None:
        large_state_rows = state.append_solution(planning_time, large_solution)
    large_allocation_rows = adapter_data_io.allocation_output_rows(
        large_solution,
        artifacts.data,
        planning_time=planning_time,
    )
    adapter_data_io.write_large_outputs(large_output_dir, artifacts, large_solution, large_state_rows)

    large_allocation_path = large_output_dir / "allocation.csv"
    large_visualization_dir = run_dir / "outputs_large" / "yard_visualization"
    visualization_kwargs = {
        "allocation_path": large_allocation_path,
        "output_dir": large_visualization_dir,
        "area_function_info": adapter_input.area_function_info,
    }
    if generate_yard_visualization is None:
        visualization_info = {"skipped": True, "reason": VISUALIZATION_IMPORT_ERROR}
        print(f"large_visualization skipped: {VISUALIZATION_IMPORT_ERROR}")
    elif not large_allocation_path.exists():
        visualization_info = {"skipped": True, "reason": f"allocation output not found: {large_allocation_path}"}
        print(f"large_visualization skipped: allocation output not found: {large_allocation_path}")
    else:
        visualization_info = generate_yard_visualization(**visualization_kwargs)
        print(f"large_visualization: {large_visualization_dir}")

    print("\n[2/2] Building medium/small inputs for SCIP column generation")
    data_dir, medium_problem, medium_big_plan, medium_demand_rows = build_problem_from_large_plan_records_adapter(
        adapter_input,
        large_allocation_rows,
        planning_time=medium_planning_time,
        voyages=medium_voyages,
        horizon_hours=algorithm_config.horizon_hours,
        misplaced_bay_exclusion_ratio=algorithm_config.misplaced_bay_exclusion_ratio,
    )
    write_medium_demand_rows_from_rows(
        medium_small_output_dir / "medium_demand_by_port.csv",
        medium_demand_rows,
    )

    print("[2/2] Solving medium and small plans with SCIP column generation")
    result, medium_diagnostics = solve_medium_small_problem(
        medium_problem,
        algorithm_config.medium_small,
        medium_small_output_dir,
        diagnostics_context={
            "adapter_json": str(adapter_json_path),
            "data_dir": str(data_dir),
            "planning_time": medium_planning_time.isoformat(sep=" "),
            "large_allocation_input": "in_memory_large_solution",
            "large_allocation_output": str(large_allocation_path),
            "flat_pipeline_mode": "large_then_embedded_column_generation_scip",
        },
    )
    print_diagnostics_summary(medium_diagnostics)

    medium_plan_path = medium_small_output_dir / "medium_plan.csv"
    small_plan_path = medium_small_output_dir / "small_plan.csv"
    small_plan_six_bay_blocks_path = medium_small_output_dir / "small_plan_six_bay_blocks.csv"
    unplaced_boxes_path = medium_small_output_dir / "unplaced_boxes.csv"
    big_plan_used_path = medium_small_output_dir / "big_plan_used.csv"
    generated_columns_path = medium_small_output_dir / "generated_columns.csv"
    medium_small_diagnostics_path = medium_small_output_dir / "diagnostics.json"

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
        "large_plan": str(large_allocation_path),
        "medium_plan": str(medium_plan_path),
        "small_plan": str(small_plan_path),
        "small_plan_six_bay_blocks": str(small_plan_six_bay_blocks_path),
        "unplaced_boxes": str(unplaced_boxes_path),
        "big_plan_used": str(big_plan_used_path),
        "generated_columns": str(generated_columns_path),
        "medium_small_diagnostics": str(medium_small_diagnostics_path),
        "large_allocation_output": str(large_allocation_path),
        "large_allocation_used_by_medium_small": "in_memory_large_solution",
        "export_vessels": list(artifacts.export_vessels),
        "import_vessels": list(artifacts.import_vessels),
        "medium_voyages": medium_voyages,
        "large_status": large_solution.status_name,
        "large_objective_value": large_solution.objective_value,
        "medium_small_solver": "flat_full_yard_plan_1.3_scip_column_generation",
        "column_generation_used_greedy_fallback": medium_diagnostics.get("used_greedy_fallback"),
        "column_generation_scip_available": medium_diagnostics.get("scip_available"),
        "column_generation_elapsed_seconds": medium_diagnostics.get("elapsed_seconds"),
        "generated_column_count": medium_diagnostics.get("final_column_count"),
        "medium_objective": medium_diagnostics.get("final_objective"),
        "medium_algorithm": medium_diagnostics.get("algorithm"),
        "medium_master_status": medium_diagnostics.get("master_status"),
        "medium_unplaced_boxes": medium_diagnostics.get("unplaced_boxes"),
        "medium_unplaced_row_count": len(result.unplaced_rows),
        "medium_row_count": len(result.medium_rows),
        "small_row_count": len(result.small_rows),
    }
    write_json(run_dir / "pipeline_summary.json", summary)

    print("\nPipeline complete.")
    print(f"large_plan: {large_output_dir}")
    print(f"medium_small_plan: {medium_small_output_dir}")
    print(f"unplaced_boxes: {unplaced_boxes_path}")
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
