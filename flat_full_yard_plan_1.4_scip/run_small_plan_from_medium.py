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
    print_diagnostics_summary,
    resolve_adapter_json,
    solve_small_plan_from_medium_adapter,
)
from api.planning_api import YardPlanAlgorithmConfig, solve_small_plan_df


DEFAULT_ADAPTER_JSON = SCRIPT_DIR / "data_examples" / "input_data.json"


def run_small_plan(
    input_adapter: InputAdapterGd,
    latest_large_plan_df: pd.DataFrame,
    latest_medium_plan_df: pd.DataFrame,
    config: YardPlanAlgorithmConfig | None = None,
) -> pd.DataFrame:
    return solve_small_plan_df(input_adapter, latest_large_plan_df, latest_medium_plan_df, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the small plan from an InputAdapterGd JSON file and an external bay-level medium-plan CSV, "
            "using the embedded SCIP column-generation small-plan planner."
        )
    )
    parser.add_argument("--adapter-json", type=Path, default=DEFAULT_ADAPTER_JSON)
    parser.add_argument("--medium-plan", type=Path, required=True, help="External medium_plan.csv.")
    parser.add_argument(
        "--big-plan",
        type=Path,
        default=None,
        help="Optional external large-plan allocation CSV. If omitted, large-plan quotas are derived from medium_plan.csv.",
    )
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "small_from_medium_outputs")
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
    algorithm_config = YardPlanAlgorithmConfig.from_cli_args(args)
    adapter_json_path = resolve_adapter_json(args.adapter_json)
    adapter_input = InputAdapterGd.load_from_json(str(adapter_json_path))
    planning_time = resolve_planning_time(adapter_input, args.planning_time)
    output_dir = create_run_dir(args.output_root.resolve(), args.run_name)

    print(f"Adapter JSON: {adapter_json_path}")
    print(f"External medium plan: {args.medium_plan.resolve()}")
    if args.big_plan:
        print(f"External large plan: {args.big_plan.resolve()}")
    print(f"Output directory: {output_dir}")

    result, diagnostics = solve_small_plan_from_medium_adapter(
        adapter_input=adapter_input,
        medium_plan_path=args.medium_plan,
        config=algorithm_config.medium_small,
        output_dir=output_dir,
        planning_time=planning_time,
        horizon_hours=algorithm_config.horizon_hours,
        voyages=algorithm_config.medium_voyages,
        big_plan_path=args.big_plan,
        misplaced_bay_exclusion_ratio=algorithm_config.misplaced_bay_exclusion_ratio,
        diagnostics_context={
            "adapter_json": str(adapter_json_path),
            "large_allocation_input": str(args.big_plan.resolve()) if args.big_plan else "derived_from_medium_plan",
            "flat_pipeline_mode": "external_medium_plan_to_small_plan_column_generation_scip",
        },
    )
    summary = {
        "planning_time": planning_time.isoformat(sep=" "),
        "adapter_json": str(adapter_json_path),
        "medium_plan_input": str(args.medium_plan.resolve()),
        "large_allocation_input": str(args.big_plan.resolve()) if args.big_plan else "derived_from_medium_plan",
        "small_plan_output_dir": str(output_dir),
        "small_row_count": len(result.small_rows),
        "medium_algorithm": diagnostics.get("algorithm"),
        "medium_master_status": diagnostics.get("master_status"),
        "medium_unplaced_boxes": diagnostics.get("unplaced_boxes"),
    }
    write_json(output_dir / "pipeline_summary.json", summary)

    print_diagnostics_summary(diagnostics)
    print(f"small_plan: {output_dir / 'small_plan.csv'}")
    print(f"small_plan_medium_summary: {output_dir / 'small_plan_medium_summary.csv'}")
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
