from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medium_small_column_generation.block_bay_planning.runner import (  # noqa: E402
    add_big_plan_argument,
    add_common_arguments,
    build_problem_from_big_plan,
    create_run_output_dir,
    log,
)
from medium_small_column_generation.column_generation_planner import (  # noqa: E402
    ColumnGenerationPlanner,
    write_columns,
    write_json,
    write_rows,
)
from medium_small_column_generation.run_column_generation_planner import (  # noqa: E402
    add_column_generation_arguments,
    format_duration,
    make_config,
    make_six_bay_block_rows,
)
from medium_small_column_generation.small_plan_from_medium import (  # noqa: E402
    apply_external_medium_plan,
    configure_small_plan_from_medium,
    filter_external_medium_plan,
    read_external_medium_plan,
)


def main() -> None:
    run_started_at = perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Solve only the small plan from an externally supplied medium plan. "
            "The external medium plan is enforced as coarse-group area quotas."
        )
    )
    add_common_arguments(parser)
    add_big_plan_argument(parser)
    add_column_generation_arguments(parser)
    parser.add_argument("--medium-plan", type=Path, required=True, help="External medium_plan.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "small_from_medium",
    )
    args = parser.parse_args()

    external_plan = read_external_medium_plan(args.medium_plan)
    if not args.voyages:
        args.voyages = list(external_plan.target_voyages)

    log(f"building {args.dataset} small-plan problem data from external medium plan")
    data_dir, planning_time, problem, _big_plan = build_problem_from_big_plan(args)
    external_plan = filter_external_medium_plan(external_plan, problem.target_voyages)
    problem = apply_external_medium_plan(problem, external_plan)
    output_dir = create_run_output_dir(args.output_dir)
    log(f"output directory: {output_dir}")

    config = configure_small_plan_from_medium(make_config(args), external_plan)
    planner = ColumnGenerationPlanner(problem, config)
    log("starting small-plan-only column generation")
    result = planner.solve()

    small_medium_summary = result.medium_rows
    log("writing small-plan-only outputs")
    write_rows(output_dir / "small_plan.csv", result.small_rows)
    write_rows(output_dir / "small_plan_six_bay_blocks.csv", make_six_bay_block_rows(result.small_rows))
    write_rows(output_dir / "small_plan_medium_summary.csv", small_medium_summary)
    write_rows(output_dir / "external_medium_plan_used.csv", external_plan.rows)
    write_rows(output_dir / "medium_plan_big_quota.csv", [row.__dict__ for row in external_plan.big_plan_rows])
    write_columns(output_dir / "generated_columns.csv", result.columns)

    elapsed_seconds = perf_counter() - run_started_at
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "algorithm": "small_plan_from_external_medium_column_generation",
            "medium_plan_input": str(args.medium_plan),
            "external_medium_row_count": len(external_plan.rows),
            "external_medium_box_count": sum(int(row["planned_boxes"]) for row in external_plan.rows),
            "external_medium_quota_count": len(external_plan.coarse_area_quota),
            "data_dir": str(data_dir),
            "dataset": args.dataset,
            "big_plan": str(args.big_plan),
            "planning_time": planning_time.isoformat(sep=" "),
            "output_dir": str(output_dir),
            "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_time": format_duration(elapsed_seconds),
        }
    )
    write_json(output_dir / "diagnostics.json", diagnostics)

    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")
    print(f"small_plan: {output_dir / 'small_plan.csv'}")
    print(f"small_plan_six_bay_blocks: {output_dir / 'small_plan_six_bay_blocks.csv'}")
    print(f"small_plan_medium_summary: {output_dir / 'small_plan_medium_summary.csv'}")
    print(f"external_medium_plan_used: {output_dir / 'external_medium_plan_used.csv'}")
    print(f"generated_columns: {output_dir / 'generated_columns.csv'}")
    print(f"diagnostics: {output_dir / 'diagnostics.json'}")
    print(f"elapsed_time: {format_duration(elapsed_seconds)} ({elapsed_seconds:.3f}s)")
    log(f"done in {format_duration(elapsed_seconds)} ({elapsed_seconds:.3f}s)")


if __name__ == "__main__":
    main()
