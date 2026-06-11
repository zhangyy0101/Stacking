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

from adapters.input_adapter_gd import InputAdapterGd
from adapters.input_adapter_standard import DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO, write_json
from medium_small.bridge import (
    add_column_generation_arguments,
    build_problem_from_large_plan_adapter,
    make_config,
    print_diagnostics_summary,
    resolve_adapter_json,
    solve_medium_small_problem,
    write_medium_demand_rows_from_rows,
)
from api.planning_api import YardPlanAlgorithmConfig, solve_medium_small_plan_df


DEFAULT_ADAPTER_JSON = SCRIPT_DIR / "data_examples" / "input_data.json"


def run_medium_small_plan(
    input_adapter: InputAdapterGd,
    latest_large_plan_df: pd.DataFrame,
    config: YardPlanAlgorithmConfig | None = None,
) -> dict[str, pd.DataFrame]:
    return solve_medium_small_plan_df(input_adapter, latest_large_plan_df, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the medium/small plan from an InputAdapterGd JSON file and an external large-plan CSV, "
            "using the embedded SCIP column-generation medium/small planner."
        )
    )
    parser.add_argument("--adapter-json", type=Path, default=DEFAULT_ADAPTER_JSON)
    parser.add_argument("--big-plan", type=Path, required=True, help="External large-plan allocation CSV.")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "medium_small_column_generation_outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--medium-voyages", nargs="+", default=None)
    parser.add_argument("--planning-time", default=None)
    parser.add_argument("--horizon-hours", type=float, default=24.0)
    parser.add_argument("--misplaced-bay-exclusion-ratio", type=float, default=DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO)
    add_column_generation_arguments(parser)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    args = parse_args()
    adapter_json_path = resolve_adapter_json(args.adapter_json)
    adapter_input = InputAdapterGd.load_from_json(str(adapter_json_path))
    planning_time = resolve_planning_time(adapter_input, args.planning_time)
    output_dir = create_run_dir(args.output_root.resolve(), args.run_name)

    print(f"Adapter JSON: {adapter_json_path}")
    print(f"External large plan: {args.big_plan.resolve()}")
    print(f"Output directory: {output_dir}")

    data_dir, problem, big_plan, demand_rows = build_problem_from_large_plan_adapter(
        adapter_input,
        args.big_plan,
        planning_time=planning_time,
        voyages=args.medium_voyages,
        horizon_hours=args.horizon_hours,
        misplaced_bay_exclusion_ratio=args.misplaced_bay_exclusion_ratio,
    )
    write_medium_demand_rows_from_rows(
        output_dir / "medium_demand_by_port.csv",
        demand_rows,
    )
    result, diagnostics = solve_medium_small_problem(
        problem,
        make_config(args),
        output_dir,
        diagnostics_context={
            "adapter_json": str(adapter_json_path),
            "data_dir": str(data_dir),
            "planning_time": planning_time.isoformat(sep=" "),
            "large_allocation_input": str(args.big_plan.resolve()),
            "flat_pipeline_mode": "external_large_plan_to_embedded_column_generation_scip",
        },
    )
    summary = {
        "planning_time": planning_time.isoformat(sep=" "),
        "adapter_json": str(adapter_json_path),
        "large_allocation_input": str(args.big_plan.resolve()),
        "medium_small_output_dir": str(output_dir),
        "medium_voyages": problem.target_voyages,
        "medium_row_count": len(result.medium_rows),
        "small_row_count": len(result.small_rows),
        "medium_algorithm": diagnostics.get("algorithm"),
        "medium_master_status": diagnostics.get("master_status"),
        "medium_unplaced_boxes": diagnostics.get("unplaced_boxes"),
    }
    write_json(output_dir / "pipeline_summary.json", summary)

    print_diagnostics_summary(diagnostics)
    print(f"medium_plan: {output_dir / 'medium_plan.csv'}")
    print(f"small_plan: {output_dir / 'small_plan.csv'}")
    print(f"diagnostics: {output_dir / 'diagnostics.json'}")


def resolve_planning_time(adapter_input: InputAdapterGd, override: str | None) -> datetime:
    if override:
        value = override
    else:
        value = getattr(adapter_input, "planning_time", None)
    planning_time = pd.Timestamp(value)
    if pd.isna(planning_time):
        raise SystemExit(f"Invalid planning_time: {value}")
    return planning_time.to_pydatetime()


def create_run_dir(output_root: Path, run_name: str | None) -> Path:
    name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


if __name__ == "__main__":
    main()
