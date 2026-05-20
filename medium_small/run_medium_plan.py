from __future__ import annotations

import argparse
import json
from pathlib import Path

from block_bay_planning.runner import (
    SimulatedAnnealingSolver,
    add_big_plan_argument,
    add_common_arguments,
    add_sa_arguments,
    build_problem_from_big_plan,
    calculate_and_write_demand,
    create_run_output_dir,
    log,
    make_sa_config,
    write_medium_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve only the medium yard plan. The medium search still scores and probes "
            "small-plan feasibility, but only medium-plan outputs are written."
        )
    )
    add_common_arguments(parser)
    add_big_plan_argument(parser)
    add_sa_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "medium_plan",
    )
    args = parser.parse_args()

    log("discovering data directory and building problem data")
    data_dir, planning_time, problem, _big_plan = build_problem_from_big_plan(args)
    output_dir = create_run_output_dir(args.output_dir)
    log(f"output directory: {output_dir}")

    log("calculating medium demand by discharge port")
    calculate_and_write_demand(data_dir, output_dir, args, planning_time)

    config = make_sa_config(args)
    solver = SimulatedAnnealingSolver(problem, config)
    log("starting medium simulated annealing with small-plan feasibility scoring")
    result = solver.solve()

    log("writing medium plan outputs")
    write_medium_artifacts(output_dir, result, problem, demand_rows=None)

    print(json.dumps(result.diagnostics, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")
    print(f"medium_demand_by_port: {output_dir / 'medium_demand_by_port.csv'}")
    print(f"medium_plan: {output_dir / 'medium_plan.csv'}")
    print(f"diagnostics: {output_dir / 'diagnostics.json'}")
    log("done")


if __name__ == "__main__":
    main()
