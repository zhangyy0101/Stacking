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
    ColumnGenerationConfig,
    ColumnGenerationPlanner,
    write_columns,
    write_json,
    write_rows,
)


def add_column_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--columns-per-iteration", type=int, default=2500)
    parser.add_argument("--initial-columns-per-group", type=int, default=16)
    parser.add_argument("--max-candidate-bays-per-group", type=int, default=500)
    parser.add_argument("--mip-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument(
        "--demand-mode",
        choices=["original", "medium", "medium-with-doc-floor", "doc-only"],
        default="original",
        help=(
            "original: match the SA+heuristic output scopes, with medium_plan from original medium demand and small_plan from document boxes; "
            "medium: match the original medium-plan demand and use forecast fallback groups when document boxes are insufficient; "
            "medium-with-doc-floor: also keep document boxes that exceed the medium target; "
            "doc-only: previous column-generation behavior, planning only not-yet-in-yard document boxes."
        ),
    )
    parser.add_argument("--fine-group-area-penalty", type=float, default=80.0)
    parser.add_argument("--fine-group-block-penalty", type=float, default=35.0)
    parser.add_argument("--fine-group-bay-penalty", type=float, default=8.0)
    parser.add_argument("--coarse-area-block-penalty", type=float, default=24.0)
    parser.add_argument("--coarse-area-bay-penalty", type=float, default=2.5)
    parser.add_argument("--medium-concentrated-group-threshold", type=int, default=26)
    parser.add_argument("--medium-small-group-area-split-penalty", type=float, default=500.0)
    parser.add_argument("--medium-small-group-fragment-penalty", type=float, default=20.0)
    parser.add_argument("--medium-large-group-min-area-boxes", type=int, default=5)
    parser.add_argument("--medium-large-group-small-area-penalty", type=float, default=120.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-gurobi",
        action="store_true",
        help="Skip the Gurobi master problem and run the deterministic greedy fallback.",
    )


def make_config(args: argparse.Namespace) -> ColumnGenerationConfig:
    return ColumnGenerationConfig(
        max_iterations=args.max_iterations,
        columns_per_iteration=args.columns_per_iteration,
        initial_columns_per_group=args.initial_columns_per_group,
        max_candidate_bays_per_group=args.max_candidate_bays_per_group,
        mip_time_limit=args.mip_time_limit,
        mip_gap=args.mip_gap,
        demand_mode=args.demand_mode,
        small_plan_group_area_split_penalty=args.fine_group_area_penalty,
        small_plan_group_block_split_penalty=args.fine_group_block_penalty,
        small_plan_group_bay_split_penalty=args.fine_group_bay_penalty,
        small_plan_coarse_area_block_split_penalty=args.coarse_area_block_penalty,
        small_plan_coarse_area_bay_split_penalty=args.coarse_area_bay_penalty,
        medium_concentrated_group_threshold=args.medium_concentrated_group_threshold,
        medium_small_group_area_split_penalty=args.medium_small_group_area_split_penalty,
        medium_small_group_fragment_penalty=args.medium_small_group_fragment_penalty,
        medium_large_group_min_area_boxes=args.medium_large_group_min_area_boxes,
        medium_large_group_small_area_penalty=args.medium_large_group_small_area_penalty,
        verbose=not args.quiet,
        use_gurobi=not args.no_gurobi,
    )


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
        detail_key = str(row.get("group_id", ""))
        details.setdefault(block_id, {})
        details[block_id][detail_key] = details[block_id].get(detail_key, 0) + int(row.get("planned_boxes", 0) or 0)

    out = []
    for block_id, row in sorted(blocks.items()):
        row = dict(row)
        row["allocation_summary"] = ";".join(
            f"{detail}|{qty}" for detail, qty in sorted(details.get(block_id, {}).items())
        )
        out.append(row)
    return out


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main() -> None:
    run_started_at = perf_counter()
    parser = argparse.ArgumentParser(
        description=(
            "Solve the yard plan by small-plan-first column generation. "
            "By default, medium_plan and small_plan use the same demand scopes as the SA+heuristic pipeline."
        )
    )
    add_common_arguments(parser)
    add_big_plan_argument(parser)
    add_column_generation_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "column_generation",
    )
    args = parser.parse_args()

    log(f"building {args.dataset} medium/small problem data for column generation")
    data_dir, planning_time, problem, _big_plan = build_problem_from_big_plan(args)
    output_dir = create_run_output_dir(args.output_dir)
    log(f"output directory: {output_dir}")

    planner = ColumnGenerationPlanner(problem, make_config(args))
    log("starting small-plan-first column generation")
    result = planner.solve()

    log("writing column-generation outputs")
    write_rows(output_dir / "small_plan.csv", result.small_rows)
    write_rows(output_dir / "medium_plan.csv", result.medium_rows)
    write_rows(output_dir / "small_plan_six_bay_blocks.csv", make_six_bay_block_rows(result.small_rows))
    write_rows(output_dir / "big_plan_used.csv", [row.__dict__ for row in problem.big_plan])
    write_columns(output_dir / "generated_columns.csv", result.columns)
    elapsed_seconds = perf_counter() - run_started_at
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
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
    print(f"medium_plan: {output_dir / 'medium_plan.csv'}")
    print(f"small_plan: {output_dir / 'small_plan.csv'}")
    print(f"small_plan_six_bay_blocks: {output_dir / 'small_plan_six_bay_blocks.csv'}")
    print(f"generated_columns: {output_dir / 'generated_columns.csv'}")
    print(f"diagnostics: {output_dir / 'diagnostics.json'}")
    print(f"elapsed_time: {format_duration(elapsed_seconds)} ({elapsed_seconds:.3f}s)")
    log(f"done in {format_duration(elapsed_seconds)} ({elapsed_seconds:.3f}s)")


if __name__ == "__main__":
    main()
