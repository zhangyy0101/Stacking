from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

sys.dont_write_bytecode = True

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ADAPTER_JSON = SCRIPT_DIR / "input_data.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import input_adapter_standard as adapter_data_io
from input_adapter_gd import InputAdapterGd
from input_adapter_standard import (
    DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
    write_json,
)
from planning_large_solver import solve_daily_rolling_yard_plan
from planning_large_visualize import generate_yard_visualization
from medium_small_column_generation_scip.column_generation_planner import (
    ColumnGenerationConfig,
    ColumnGenerationPlanner,
    write_columns,
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
    parser.add_argument("--cg-max-iterations", type=int, default=30)
    parser.add_argument("--cg-columns-per-iteration", type=int, default=2500)
    parser.add_argument("--cg-initial-columns-per-group", type=int, default=16)
    parser.add_argument("--cg-max-candidate-bays-per-group", type=int, default=500)
    parser.add_argument("--cg-mip-time-limit", type=float, default=120.0)
    parser.add_argument("--cg-mip-gap", type=float, default=0.01)
    parser.add_argument(
        "--cg-demand-mode",
        choices=["original", "medium", "medium-with-doc-floor", "doc-only"],
        default="original",
    )
    parser.add_argument("--cg-fine-group-area-penalty", type=float, default=80.0)
    parser.add_argument("--cg-fine-group-block-penalty", type=float, default=35.0)
    parser.add_argument("--cg-fine-group-bay-penalty", type=float, default=8.0)
    parser.add_argument("--cg-coarse-area-block-penalty", type=float, default=24.0)
    parser.add_argument("--cg-coarse-area-bay-penalty", type=float, default=2.5)
    parser.add_argument("--cg-medium-concentrated-group-threshold", type=int, default=26)
    parser.add_argument("--cg-medium-small-group-area-split-penalty", type=float, default=500.0)
    parser.add_argument("--cg-medium-small-group-fragment-penalty", type=float, default=20.0)
    parser.add_argument("--cg-medium-large-group-min-area-boxes", type=int, default=5)
    parser.add_argument("--cg-medium-large-group-small-area-penalty", type=float, default=120.0)
    parser.add_argument("--cg-quiet", action="store_true")
    parser.add_argument(
        "--cg-no-scip",
        action="store_true",
        help="Disable SCIP column generation and use the deterministic greedy fallback.",
    )
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

    large_config = adapter_data_io.LargePlanningConfig()

    print("\n[1/2] Building large-plan inputs")
    artifacts, state = adapter_data_io.build_large_inputs(
        adapter_input,
        planning_time,
        disable_default_flow_aliases=args.disable_default_flow_aliases,
        config=large_config,
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

    print("[2/2] Solving medium and small plans with column generation")
    column_generation_config = make_column_generation_config(args)
    solver_started_at = perf_counter()
    solver = ColumnGenerationPlanner(medium_inputs.problem, column_generation_config)
    result = solver.solve()
    column_generation_elapsed_seconds = perf_counter() - solver_started_at

    medium_plan_path = medium_small_output_dir / "medium_plan.csv"
    small_plan_path = medium_small_output_dir / "small_plan.csv"
    small_plan_six_bay_blocks_path = medium_small_output_dir / "small_plan_six_bay_blocks.csv"
    big_plan_used_path = medium_small_output_dir / "big_plan_used.csv"
    generated_columns_path = medium_small_output_dir / "generated_columns.csv"
    medium_small_diagnostics_path = medium_small_output_dir / "diagnostics.json"

    write_rows(medium_plan_path, result.medium_rows)
    write_rows(small_plan_path, result.small_rows)
    write_rows(small_plan_six_bay_blocks_path, make_six_bay_block_rows(result.small_rows))
    write_rows(big_plan_used_path, [row.__dict__ for row in medium_inputs.problem.big_plan])
    write_columns(generated_columns_path, result.columns)
    medium_small_diagnostics = dict(result.diagnostics)
    medium_small_diagnostics.update(
        {
            "solver": "medium_small_column_generation_scip",
            "input_mode": "adapter_object",
            "big_plan_source": "in_memory_large_solution",
            "column_generation_elapsed_seconds": round(column_generation_elapsed_seconds, 3),
            "column_generation_config": column_generation_config_dict(column_generation_config),
        }
    )
    write_json(medium_small_diagnostics_path, medium_small_diagnostics)

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
        "big_plan_used": str(big_plan_used_path),
        "generated_columns": str(generated_columns_path),
        "medium_small_diagnostics": str(medium_small_diagnostics_path),
        "large_allocation_output": str(large_allocation_path),
        "large_allocation_used_by_medium_small": "in_memory",
        "export_vessels": list(artifacts.export_vessels),
        "import_vessels": list(artifacts.import_vessels),
        "medium_voyages": medium_voyages,
        "large_status": large_solution.status_name,
        "large_objective_value": large_solution.objective_value,
        "medium_small_solver": "medium_small_column_generation_scip",
        "column_generation_used_greedy_fallback": medium_small_diagnostics.get("used_greedy_fallback"),
        "column_generation_scip_available": medium_small_diagnostics.get("scip_available"),
        "column_generation_elapsed_seconds": round(column_generation_elapsed_seconds, 3),
        "generated_column_count": len(result.columns),
        "medium_row_count": len(result.medium_rows),
        "small_row_count": len(result.small_rows),
    }
    write_json(run_dir / "pipeline_summary.json", summary)

    print("\nPipeline complete.")
    print(f"large_plan: {large_allocation_path}")
    print(f"medium_plan: {medium_plan_path}")
    print(f"small_plan: {small_plan_path}")
    print(f"medium_small_plan_dir: {medium_small_output_dir}")
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


def make_column_generation_config(args: argparse.Namespace) -> ColumnGenerationConfig:
    return ColumnGenerationConfig(
        max_iterations=args.cg_max_iterations,
        columns_per_iteration=args.cg_columns_per_iteration,
        initial_columns_per_group=args.cg_initial_columns_per_group,
        max_candidate_bays_per_group=args.cg_max_candidate_bays_per_group,
        mip_time_limit=args.cg_mip_time_limit,
        mip_gap=args.cg_mip_gap,
        demand_mode=args.cg_demand_mode,
        small_plan_group_area_split_penalty=args.cg_fine_group_area_penalty,
        small_plan_group_block_split_penalty=args.cg_fine_group_block_penalty,
        small_plan_group_bay_split_penalty=args.cg_fine_group_bay_penalty,
        small_plan_coarse_area_block_split_penalty=args.cg_coarse_area_block_penalty,
        small_plan_coarse_area_bay_split_penalty=args.cg_coarse_area_bay_penalty,
        medium_concentrated_group_threshold=args.cg_medium_concentrated_group_threshold,
        medium_small_group_area_split_penalty=args.cg_medium_small_group_area_split_penalty,
        medium_small_group_fragment_penalty=args.cg_medium_small_group_fragment_penalty,
        medium_large_group_min_area_boxes=args.cg_medium_large_group_min_area_boxes,
        medium_large_group_small_area_penalty=args.cg_medium_large_group_small_area_penalty,
        verbose=not args.cg_quiet,
        use_scip=not args.cg_no_scip,
    )


def column_generation_config_dict(config: ColumnGenerationConfig) -> dict:
    return {
        "max_iterations": config.max_iterations,
        "columns_per_iteration": config.columns_per_iteration,
        "initial_columns_per_group": config.initial_columns_per_group,
        "max_candidate_bays_per_group": config.max_candidate_bays_per_group,
        "mip_time_limit": config.mip_time_limit,
        "mip_gap": config.mip_gap,
        "verbose": config.verbose,
        "use_scip": config.use_scip,
        "demand_mode": config.demand_mode,
        "unplaced_penalty": config.unplaced_penalty,
        "group_area_balance_penalty": config.group_area_balance_penalty,
        "medium_concentrated_group_threshold": config.medium_concentrated_group_threshold,
        "medium_small_group_area_split_penalty": config.medium_small_group_area_split_penalty,
        "medium_small_group_fragment_penalty": config.medium_small_group_fragment_penalty,
        "medium_large_group_min_area_boxes": config.medium_large_group_min_area_boxes,
        "medium_large_group_small_area_penalty": config.medium_large_group_small_area_penalty,
        "big_plan_area_deviation_penalty": config.big_plan_area_deviation_penalty,
        "small_plan_group_area_split_penalty": config.small_plan_group_area_split_penalty,
        "small_plan_group_block_split_penalty": config.small_plan_group_block_split_penalty,
        "small_plan_group_bay_split_penalty": config.small_plan_group_bay_split_penalty,
        "small_plan_coarse_area_block_split_penalty": config.small_plan_coarse_area_block_split_penalty,
        "small_plan_coarse_area_bay_split_penalty": config.small_plan_coarse_area_bay_split_penalty,
        "berth_distance_penalty": config.berth_distance_penalty,
        "active_loading_area_penalty": config.active_loading_area_penalty,
        "post_window_loading_area_reward": config.post_window_loading_area_reward,
        "fallback_bay_penalty": config.fallback_bay_penalty,
        "non_preferred_block_penalty": config.non_preferred_block_penalty,
        "port_mismatch_penalty": config.port_mismatch_penalty,
    }


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
