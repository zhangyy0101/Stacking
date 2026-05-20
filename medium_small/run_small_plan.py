from __future__ import annotations

import argparse
import json
from pathlib import Path

from block_bay_planning.runner import (
    SimulatedAnnealingSolver,
    add_big_plan_argument,
    add_common_arguments,
    build_problem_from_medium_plan,
    create_run_output_dir,
    log,
    write_json,
    write_rows,
    write_small_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct only the small bay plan from an external medium_plan.csv. "
            "The medium file should use the current medium output schema."
        )
    )
    add_common_arguments(parser)
    add_big_plan_argument(parser)
    parser.add_argument(
        "--medium-plan",
        type=Path,
        required=True,
        help="Existing medium_plan.csv with columns voyage_id,flow,port,size,area_no,planned_boxes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "small_plan",
    )
    args = parser.parse_args()

    log("discovering data directory and building small-plan problem data")
    _data_dir, _planning_time, problem, medium_rows = build_problem_from_medium_plan(args)
    output_dir = create_run_output_dir(args.output_dir)
    log(f"output directory: {output_dir}")

    solver = SimulatedAnnealingSolver(problem)
    log(f"reading medium plan: {args.medium_plan}")
    small_rows = solver.make_small_rows_from_medium_rows(medium_rows)

    diagnostics = {
        "mode": "small_only",
        "big_plan": str(args.big_plan.resolve()),
        "medium_plan": str(args.medium_plan.resolve()),
        "medium_row_count": len(medium_rows),
        "small_row_count": len(small_rows),
        "small_plan_used_six_bay_block_count": len(
            {row.get("six_bay_block_id") for row in small_rows if row.get("six_bay_block_id")}
        ),
        "planning_time": problem.planning_time.isoformat(sep=" "),
        "horizon_hours": problem.horizon_hours,
        "target_voyages": problem.target_voyages,
    }

    log("writing small plan outputs")
    write_rows(output_dir / "medium_plan_used.csv", medium_rows)
    write_small_artifacts(output_dir, small_rows)
    write_json(output_dir / "diagnostics.json", diagnostics)

    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")
    print(f"medium_plan_used: {output_dir / 'medium_plan_used.csv'}")
    print(f"small_plan: {output_dir / 'small_plan.csv'}")
    print(f"small_plan_six_bay_blocks: {output_dir / 'small_plan_six_bay_blocks.csv'}")
    print(f"diagnostics: {output_dir / 'diagnostics.json'}")
    log("done")


if __name__ == "__main__":
    main()
