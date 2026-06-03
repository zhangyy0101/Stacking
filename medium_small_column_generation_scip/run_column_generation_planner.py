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

from medium_small_column_generation_scip.block_bay_planning.runner import (  # noqa: E402
    add_big_plan_argument,
    add_common_arguments,
    build_problem_from_big_plan,
    create_run_output_dir,
    log,
)

from medium_small_column_generation_scip.column_generation_planner import (  # noqa: E402
    ColumnGenerationConfig,
    ColumnGenerationPlanner,
    write_columns,
    write_json,
    write_rows,
)


def add_column_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--columns-per-iteration", type=int, default=2500)
    parser.add_argument("--stalled-pricing-columns", type=int, default=500)
    parser.add_argument("--primal-expansion-columns", type=int, default=800)
    parser.add_argument("--max-primal-expansion-rounds", type=int, default=3)
    parser.add_argument("--primal-expansion-reduced-cost-limit", type=float, default=1_000_000.0)
    parser.add_argument("--initial-columns-per-group", type=int, default=16)
    parser.add_argument("--max-candidate-bays-per-group", type=int, default=500)
    parser.add_argument("--mip-time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument(
        "--diving-max-steps",
        type=int,
        default=200,
        help="Maximum LP fixing steps for the SCIP column-generation diving master.",
    )
    parser.add_argument(
        "--diving-fractional-tolerance",
        type=float,
        default=1e-5,
        help="Tolerance used to decide whether a diving LP column value is integral.",
    )
    parser.add_argument(
        "--diving-fix-batch-size",
        type=int,
        default=8,
        help="Maximum number of fractional columns to fix after each diving LP solve.",
    )
    parser.add_argument(
        "--diving-max-no-improve-steps",
        type=int,
        default=5,
        help="Stop SCIP diving after this many fixing steps without improving the incumbent; use 0 to disable.",
    )
    parser.add_argument(
        "--diving-improvement-rounds",
        type=int,
        default=6,
        help="Number of local reoptimization rounds to run after the fixing dive.",
    )
    parser.add_argument(
        "--diving-improvement-time-limit",
        type=float,
        default=6.0,
        help="SCIP time limit in seconds for each local diving-improvement round.",
    )
    parser.add_argument(
        "--diving-improvement-max-groups",
        type=int,
        default=14,
        help="Maximum fine groups released in each local diving-improvement neighborhood.",
    )
    parser.add_argument(
        "--diving-improvement-max-no-improve-rounds",
        type=int,
        default=2,
        help="Stop local diving improvement after this many consecutive rounds without incumbent improvement; use 0 to disable.",
    )
    parser.add_argument(
        "--repair-lns-rounds",
        type=int,
        default=2,
        help="Number of small MIP LNS rounds after staged repair.",
    )
    parser.add_argument(
        "--repair-lns-time-limit",
        type=float,
        default=3.0,
        help="SCIP time limit in seconds for each staged-repair LNS round.",
    )
    parser.add_argument(
        "--repair-lns-max-groups",
        type=int,
        default=16,
        help="Maximum fine groups released in each staged-repair LNS neighborhood.",
    )
    parser.add_argument(
        "--repair-lns-max-no-improve-rounds",
        type=int,
        default=2,
        help="Stop staged-repair LNS after this many rounds without incumbent improvement; use 0 to disable.",
    )
    parser.add_argument(
        "--coarse-compaction-lns-rounds",
        type=int,
        default=0,
        help="Number of coarse-group compaction LNS rounds after staged repair. Disabled by default because repair LNS already carries the coarse compaction objective.",
    )
    parser.add_argument(
        "--coarse-compaction-lns-time-limit",
        type=float,
        default=4.0,
        help="SCIP time limit in seconds for each coarse-group compaction LNS round.",
    )
    parser.add_argument(
        "--coarse-compaction-lns-max-groups",
        type=int,
        default=16,
        help="Maximum fine groups released in each coarse-group compaction LNS neighborhood.",
    )
    parser.add_argument(
        "--coarse-compaction-lns-max-no-improve-rounds",
        type=int,
        default=1,
        help="Stop coarse-group compaction LNS after this many rounds without improvement; use 0 to disable.",
    )
    parser.add_argument(
        "--diving-price-columns",
        action="store_true",
        help="Run pricing inside each diving LP step. By default, diving uses the converged column pool.",
    )
    parser.add_argument(
        "--diving-stop-on-lp-unplaced",
        action="store_true",
        default=True,
        help="Stop and reduce the diving batch when an intermediate diving LP has unplaced boxes.",
    )
    parser.add_argument(
        "--diving-continue-on-lp-unplaced",
        action="store_false",
        dest="diving_stop_on_lp_unplaced",
        help="Keep diving after an intermediate diving LP has unplaced boxes; final repair will compare rollback candidates.",
    )
    parser.add_argument(
        "--full-column-pool",
        action="store_true",
        help="Generate every currently enumerable placement column before solving the Master, skipping pricing iterations.",
    )
    parser.add_argument(
        "--demand-mode",
        choices=["original", "medium", "medium-with-doc-floor", "doc-only"],
        default="original",
        help=(
            "original: match the SA+heuristic output scopes, lifting medium demand to cover document boxes exactly by voyage/flow/port/size; "
            "medium: match the lifted medium-plan demand and use forecast fallback groups when document boxes are insufficient; "
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
    parser.add_argument("--medium-small-group-area-split-penalty", type=float, default=1200.0)
    parser.add_argument("--medium-small-group-fragment-penalty", type=float, default=60.0)
    parser.add_argument("--medium-large-group-min-area-boxes", type=int, default=10)
    parser.add_argument("--medium-large-group-small-area-penalty", type=float, default=900.0)
    parser.add_argument("--medium-large-group-area-open-penalty", type=float, default=0.0)
    parser.add_argument("--medium-large-group-target-area-boxes", type=int, default=60)
    parser.add_argument("--unplaced-penalty", type=float, default=100_000.0)
    parser.add_argument("--required-area-reward", type=float, default=1_000.0)
    parser.add_argument("--big-plan-area-deviation-penalty", type=float, default=8.0)
    parser.add_argument("--big-plan-fallback-tier-penalty", type=float, default=120.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-scip",
        action="store_true",
        help="Skip the SCIP master problem and run the deterministic greedy fallback.",
    )


def make_config(args: argparse.Namespace) -> ColumnGenerationConfig:
    return ColumnGenerationConfig(
        max_iterations=args.max_iterations,
        columns_per_iteration=args.columns_per_iteration,
        stalled_pricing_columns=args.stalled_pricing_columns,
        primal_expansion_columns=args.primal_expansion_columns,
        max_primal_expansion_rounds=args.max_primal_expansion_rounds,
        primal_expansion_reduced_cost_limit=args.primal_expansion_reduced_cost_limit,
        initial_columns_per_group=args.initial_columns_per_group,
        max_candidate_bays_per_group=args.max_candidate_bays_per_group,
        mip_time_limit=args.mip_time_limit,
        mip_gap=args.mip_gap,
        diving_max_steps=args.diving_max_steps,
        diving_fractional_tolerance=args.diving_fractional_tolerance,
        diving_fix_batch_size=args.diving_fix_batch_size,
        diving_max_no_improve_steps=args.diving_max_no_improve_steps,
        diving_improvement_rounds=args.diving_improvement_rounds,
        diving_improvement_time_limit=args.diving_improvement_time_limit,
        diving_improvement_max_groups=args.diving_improvement_max_groups,
        diving_improvement_max_no_improve_rounds=args.diving_improvement_max_no_improve_rounds,
        repair_lns_rounds=args.repair_lns_rounds,
        repair_lns_time_limit=args.repair_lns_time_limit,
        repair_lns_max_groups=args.repair_lns_max_groups,
        repair_lns_max_no_improve_rounds=args.repair_lns_max_no_improve_rounds,
        coarse_compaction_lns_rounds=args.coarse_compaction_lns_rounds,
        coarse_compaction_lns_time_limit=args.coarse_compaction_lns_time_limit,
        coarse_compaction_lns_max_groups=args.coarse_compaction_lns_max_groups,
        coarse_compaction_lns_max_no_improve_rounds=args.coarse_compaction_lns_max_no_improve_rounds,
        diving_price_columns=args.diving_price_columns,
        diving_stop_on_lp_unplaced=args.diving_stop_on_lp_unplaced,
        full_column_pool=args.full_column_pool,
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
        medium_large_group_area_open_penalty=args.medium_large_group_area_open_penalty,
        medium_large_group_target_area_boxes=args.medium_large_group_target_area_boxes,
        unplaced_penalty=args.unplaced_penalty,
        required_area_reward=args.required_area_reward,
        big_plan_area_deviation_penalty=args.big_plan_area_deviation_penalty,
        big_plan_fallback_tier_penalty=args.big_plan_fallback_tier_penalty,
        verbose=not args.quiet,
        use_scip=not args.no_scip,
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


def make_console_diagnostics_summary(diagnostics: dict) -> dict:
    inheritance = diagnostics.get("medium_big_plan_inheritance", {}) or {}
    fragmentation = diagnostics.get("medium_fragmentation", {}) or {}
    return {
        "algorithm": diagnostics.get("algorithm"),
        "master_algorithm": diagnostics.get("master_algorithm"),
        "master_status": diagnostics.get("master_status"),
        "medium_plan_source": diagnostics.get("medium_plan_source"),
        "medium_plan_granularity": diagnostics.get("medium_plan_granularity"),
        "planned_medium_boxes": diagnostics.get("planned_medium_boxes"),
        "planned_small_boxes": diagnostics.get("planned_small_boxes"),
        "planned_medium_by_source": diagnostics.get("planned_medium_by_source", {}),
        "unplaced_boxes": diagnostics.get("unplaced_boxes"),
        "small_medium_consistency_violations": diagnostics.get("small_medium_consistency_violations"),
        "small_medium_bay_consistency_violations": diagnostics.get("small_medium_bay_consistency_violations"),
        "inheritance_ratio": inheritance.get("inheritance_ratio"),
        "medium_area_rows_below_min_boxes": diagnostics.get("medium_area_rows_below_min_boxes"),
        "tiny_area_rows": fragmentation.get("tiny_area_rows"),
        "small_coarse_multi_area_groups": fragmentation.get("small_coarse_multi_area_groups"),
        "large_coarse_tiny_area_rows": fragmentation.get("large_coarse_tiny_area_rows"),
        "diving_improvement_rounds_run": diagnostics.get("diving_improvement_rounds_run"),
        "diving_improvement_improvements": diagnostics.get("diving_improvement_improvements"),
        "diving_improvement_incumbent_source": diagnostics.get("diving_improvement_incumbent_source"),
        "diving_improvement_stop_reason": diagnostics.get("diving_improvement_stop_reason"),
        "stage0_unplaced_cap": diagnostics.get("stage0_unplaced_cap"),
        "stage0_unplaced_cap_source": diagnostics.get("stage0_unplaced_cap_source"),
        "repair_lns_rounds_run": diagnostics.get("repair_lns_rounds_run"),
        "repair_lns_improvements": diagnostics.get("repair_lns_improvements"),
        "repair_lns_stop_reason": diagnostics.get("repair_lns_stop_reason"),
        "coarse_compaction_lns_rounds_run": diagnostics.get("coarse_compaction_lns_rounds_run"),
        "coarse_compaction_lns_improvements": diagnostics.get("coarse_compaction_lns_improvements"),
        "coarse_compaction_lns_stop_reason": diagnostics.get("coarse_compaction_lns_stop_reason"),
        "selected_column_count": diagnostics.get("selected_column_count"),
        "final_column_count": diagnostics.get("final_column_count"),
        "elapsed_time": diagnostics.get("elapsed_time"),
    }


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

    print(json.dumps(make_console_diagnostics_summary(diagnostics), ensure_ascii=False, indent=2))
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
