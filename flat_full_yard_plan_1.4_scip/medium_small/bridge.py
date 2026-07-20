from __future__ import annotations

import json
import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapters import input_adapter_standard as adapter_data_io  # noqa: E402
from adapters.input_adapter_gd import InputAdapterGd  # noqa: E402
from block_bay_planning.data_loader import (  # noqa: E402
    medium_demand_caps_from_big_plan,
    read_big_plan as read_external_big_plan,
)
from block_bay_planning.models import BigPlanRow  # noqa: E402
from block_bay_planning.demand_calculator import (  # noqa: E402
    calculate_medium_demands,
)
from medium_small.column_generation_planner import (  # noqa: E402
    ColumnGenerationConfig,
    ColumnGenerationPlanner,
    write_columns,
    write_json,
    write_rows,
)
from medium_small.small_plan_from_medium import (  # noqa: E402
    apply_external_medium_plan,
    configure_small_plan_from_medium,
    filter_external_medium_plan,
    read_external_medium_plan,
)


DEFAULT_TARGET_BIG_PLAN_FLOWS = {"OF", "IF", "IZ", "T", "OZ"}


def add_column_generation_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = ColumnGenerationConfig()
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument("--columns-per-iteration", type=int, default=defaults.columns_per_iteration)
    parser.add_argument("--stalled-pricing-columns", type=int, default=defaults.stalled_pricing_columns)
    parser.add_argument("--min-pricing-iterations", type=int, default=defaults.min_pricing_iterations)
    parser.add_argument("--pricing-early-stop-new-columns", type=int, default=defaults.pricing_early_stop_new_columns)
    parser.add_argument("--pricing-max-no-improve-iterations", type=int, default=defaults.pricing_max_no_improve_iterations)
    parser.add_argument("--pricing-min-lp-improvement", type=float, default=defaults.pricing_min_lp_improvement)
    parser.add_argument("--pricing-min-lp-improvement-relative", type=float, default=defaults.pricing_min_lp_improvement_relative)
    parser.add_argument("--pricing-min-lp-improvement-per-group", type=float, default=defaults.pricing_min_lp_improvement_per_group)
    parser.add_argument(
        "--pricing-min-lp-improvement-per-1000-columns",
        type=float,
        default=defaults.pricing_min_lp_improvement_per_1000_columns,
    )
    parser.add_argument(
        "--enable-feasibility-early-stop",
        action="store_true",
        default=defaults.feasibility_early_stop_enabled,
    )
    parser.add_argument("--disable-feasibility-early-stop", action="store_true")
    parser.add_argument("--feasibility-early-stop-min-iteration", type=int, default=defaults.feasibility_early_stop_min_iteration)
    parser.add_argument("--stage0-closure-max-extra-columns", type=int, default=defaults.stage0_closure_max_extra_columns)
    parser.add_argument("--primal-expansion-columns", type=int, default=defaults.primal_expansion_columns)
    parser.add_argument("--max-primal-expansion-rounds", type=int, default=defaults.max_primal_expansion_rounds)
    parser.add_argument("--primal-expansion-reduced-cost-limit", type=float, default=defaults.primal_expansion_reduced_cost_limit)
    parser.add_argument("--initial-columns-per-group", type=int, default=defaults.initial_columns_per_group)
    parser.add_argument("--max-candidate-bays-per-group", type=int, default=defaults.max_candidate_bays_per_group)
    parser.add_argument(
        "--disable-adaptive-pricing",
        action="store_true",
        default=not defaults.adaptive_pricing_enabled,
    )
    parser.add_argument("--pricing-candidate-bays-initial", type=int, default=defaults.pricing_candidate_bays_initial)
    parser.add_argument("--pricing-candidate-bays-growth-per-iteration", type=int, default=defaults.pricing_candidate_bays_growth_per_iteration)
    parser.add_argument("--pricing-candidate-bays-high-dual", type=int, default=defaults.pricing_candidate_bays_high_dual)
    parser.add_argument("--pricing-candidate-bays-unplaced", type=int, default=defaults.pricing_candidate_bays_unplaced)
    parser.add_argument(
        "--total-time-limit",
        type=float,
        default=defaults.total_time_limit,
        help=(
            "Medium/small column-generation solve budget in seconds; excludes input preprocessing, "
            "planner initialization, and output writing. Use 0 to disable."
        ),
    )
    parser.add_argument("--pricing-min-lp-time-limit", type=float, default=defaults.pricing_min_lp_time_limit)
    parser.add_argument("--pricing-iteration-setup-reserve", type=float, default=defaults.pricing_iteration_setup_reserve)
    parser.add_argument("--staged-repair-reserve-fraction", type=float, default=defaults.staged_repair_reserve_fraction)
    parser.add_argument("--staged-repair-reserve-min", type=float, default=defaults.staged_repair_reserve_min)
    parser.add_argument("--staged-repair-reserve-max", type=float, default=defaults.staged_repair_reserve_max)
    parser.add_argument("--staged-repair-secondary-order-min-remaining", type=float, default=defaults.staged_repair_secondary_order_min_remaining)
    parser.add_argument("--staged-repair-rebuild-first-unplaced-threshold", type=int, default=defaults.staged_repair_rebuild_first_unplaced_threshold)
    parser.add_argument("--mip-time-limit", type=float, default=defaults.mip_time_limit)
    parser.add_argument("--mip-gap", type=float, default=defaults.mip_gap)
    parser.add_argument("--full-column-pool", action="store_true", default=defaults.full_column_pool)
    parser.add_argument(
        "--demand-mode",
        choices=["original", "medium", "medium-with-doc-floor", "doc-only"],
        default=defaults.demand_mode,
    )
    parser.add_argument("--fine-group-area-penalty", type=float, default=defaults.small_plan_group_area_split_penalty)
    parser.add_argument("--fine-group-block-penalty", type=float, default=defaults.small_plan_group_block_split_penalty)
    parser.add_argument("--fine-group-bay-penalty", type=float, default=defaults.small_plan_group_bay_split_penalty)
    parser.add_argument("--coarse-area-block-penalty", type=float, default=defaults.small_plan_coarse_area_block_split_penalty)
    parser.add_argument("--coarse-area-bay-penalty", type=float, default=defaults.small_plan_coarse_area_bay_split_penalty)
    parser.add_argument("--group-area-balance-penalty", type=float, default=defaults.group_area_balance_penalty)
    parser.add_argument("--medium-concentrated-group-threshold", type=int, default=defaults.medium_concentrated_group_threshold)
    parser.add_argument("--medium-small-group-area-split-penalty", type=float, default=defaults.medium_small_group_area_split_penalty)
    parser.add_argument("--medium-small-group-fragment-penalty", type=float, default=defaults.medium_small_group_fragment_penalty)
    parser.add_argument("--medium-large-group-min-area-boxes", type=int, default=defaults.medium_large_group_min_area_boxes)
    parser.add_argument("--medium-large-group-small-area-penalty", type=float, default=defaults.medium_large_group_small_area_penalty)
    parser.add_argument("--medium-large-group-area-open-penalty", type=float, default=defaults.medium_large_group_area_open_penalty)
    parser.add_argument("--medium-large-group-target-area-boxes", type=int, default=defaults.medium_large_group_target_area_boxes)
    parser.add_argument(
        "--medium-large-group-area-excess-penalty",
        type=float,
        default=defaults.medium_large_group_area_excess_penalty,
    )
    parser.add_argument("--unplaced-penalty", type=float, default=defaults.unplaced_penalty)
    parser.add_argument("--required-area-reward", type=float, default=defaults.required_area_reward)
    parser.add_argument("--big-plan-area-deviation-penalty", type=float, default=defaults.big_plan_area_deviation_penalty)
    parser.add_argument("--big-plan-fallback-tier-penalty", type=float, default=defaults.big_plan_fallback_tier_penalty)
    parser.add_argument(
        "--export-e-area-max-bays-per-voyage-area",
        type=int,
        default=defaults.export_e_area_max_bays_per_voyage_area,
        help="Max bay starts an export voyage may use inside one E-prefixed area; use -1 to disable.",
    )
    parser.add_argument(
        "--export-e-area-non-40-penalty",
        type=float,
        default=defaults.export_e_area_non_40_penalty,
        help="Soft penalty for placing non-40/45ft export boxes in E-prefixed areas.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-scip", action="store_true")
    parser.add_argument(
        "--enable-scip-symmetry",
        action="store_true",
        help="Enable SCIP symmetry handling; disabled by default because large medium/small LPs may crash native SCIP builds.",
    )


def make_config(args: argparse.Namespace) -> ColumnGenerationConfig:
    defaults = ColumnGenerationConfig()
    return ColumnGenerationConfig(
        max_iterations=args.max_iterations,
        columns_per_iteration=args.columns_per_iteration,
        stalled_pricing_columns=args.stalled_pricing_columns,
        min_pricing_iterations=args.min_pricing_iterations,
        pricing_early_stop_new_columns=args.pricing_early_stop_new_columns,
        pricing_max_no_improve_iterations=getattr(args, "pricing_max_no_improve_iterations", defaults.pricing_max_no_improve_iterations),
        pricing_min_lp_improvement=getattr(args, "pricing_min_lp_improvement", defaults.pricing_min_lp_improvement),
        pricing_min_lp_improvement_relative=getattr(
            args,
            "pricing_min_lp_improvement_relative",
            defaults.pricing_min_lp_improvement_relative,
        ),
        pricing_min_lp_improvement_per_group=getattr(
            args,
            "pricing_min_lp_improvement_per_group",
            defaults.pricing_min_lp_improvement_per_group,
        ),
        pricing_min_lp_improvement_per_1000_columns=getattr(
            args,
            "pricing_min_lp_improvement_per_1000_columns",
            defaults.pricing_min_lp_improvement_per_1000_columns,
        ),
        feasibility_early_stop_enabled=bool(
            getattr(args, "enable_feasibility_early_stop", defaults.feasibility_early_stop_enabled)
            and not getattr(args, "disable_feasibility_early_stop", False)
        ),
        feasibility_early_stop_min_iteration=args.feasibility_early_stop_min_iteration,
        stage0_closure_max_extra_columns=args.stage0_closure_max_extra_columns,
        primal_expansion_columns=args.primal_expansion_columns,
        max_primal_expansion_rounds=args.max_primal_expansion_rounds,
        primal_expansion_reduced_cost_limit=args.primal_expansion_reduced_cost_limit,
        initial_columns_per_group=args.initial_columns_per_group,
        max_candidate_bays_per_group=args.max_candidate_bays_per_group,
        adaptive_pricing_enabled=not getattr(args, "disable_adaptive_pricing", not defaults.adaptive_pricing_enabled),
        pricing_candidate_bays_initial=getattr(args, "pricing_candidate_bays_initial", defaults.pricing_candidate_bays_initial),
        pricing_candidate_bays_growth_per_iteration=getattr(args, "pricing_candidate_bays_growth_per_iteration", defaults.pricing_candidate_bays_growth_per_iteration),
        pricing_candidate_bays_high_dual=getattr(args, "pricing_candidate_bays_high_dual", defaults.pricing_candidate_bays_high_dual),
        pricing_candidate_bays_unplaced=getattr(args, "pricing_candidate_bays_unplaced", defaults.pricing_candidate_bays_unplaced),
        total_time_limit=getattr(args, "total_time_limit", defaults.total_time_limit),
        pricing_min_lp_time_limit=getattr(args, "pricing_min_lp_time_limit", defaults.pricing_min_lp_time_limit),
        pricing_iteration_setup_reserve=getattr(args, "pricing_iteration_setup_reserve", defaults.pricing_iteration_setup_reserve),
        staged_repair_reserve_fraction=getattr(args, "staged_repair_reserve_fraction", defaults.staged_repair_reserve_fraction),
        staged_repair_reserve_min=getattr(args, "staged_repair_reserve_min", defaults.staged_repair_reserve_min),
        staged_repair_reserve_max=getattr(args, "staged_repair_reserve_max", defaults.staged_repair_reserve_max),
        staged_repair_secondary_order_min_remaining=getattr(
            args,
            "staged_repair_secondary_order_min_remaining",
            defaults.staged_repair_secondary_order_min_remaining,
        ),
        staged_repair_rebuild_first_unplaced_threshold=getattr(
            args,
            "staged_repair_rebuild_first_unplaced_threshold",
            defaults.staged_repair_rebuild_first_unplaced_threshold,
        ),
        mip_time_limit=args.mip_time_limit,
        mip_gap=args.mip_gap,
        full_column_pool=args.full_column_pool,
        demand_mode=args.demand_mode,
        small_plan_group_area_split_penalty=args.fine_group_area_penalty,
        small_plan_group_block_split_penalty=args.fine_group_block_penalty,
        small_plan_group_bay_split_penalty=args.fine_group_bay_penalty,
        small_plan_coarse_area_block_split_penalty=args.coarse_area_block_penalty,
        small_plan_coarse_area_bay_split_penalty=args.coarse_area_bay_penalty,
        group_area_balance_penalty=args.group_area_balance_penalty,
        medium_concentrated_group_threshold=args.medium_concentrated_group_threshold,
        medium_small_group_area_split_penalty=args.medium_small_group_area_split_penalty,
        medium_small_group_fragment_penalty=args.medium_small_group_fragment_penalty,
        medium_large_group_min_area_boxes=args.medium_large_group_min_area_boxes,
        medium_large_group_small_area_penalty=args.medium_large_group_small_area_penalty,
        medium_large_group_area_open_penalty=args.medium_large_group_area_open_penalty,
        medium_large_group_target_area_boxes=args.medium_large_group_target_area_boxes,
        medium_large_group_area_excess_penalty=args.medium_large_group_area_excess_penalty,
        unplaced_penalty=args.unplaced_penalty,
        required_area_reward=args.required_area_reward,
        big_plan_area_deviation_penalty=args.big_plan_area_deviation_penalty,
        big_plan_fallback_tier_penalty=args.big_plan_fallback_tier_penalty,
        export_e_area_max_bays_per_voyage_area=getattr(
            args,
            "export_e_area_max_bays_per_voyage_area",
            defaults.export_e_area_max_bays_per_voyage_area,
        ),
        export_e_area_non_40_penalty=getattr(
            args,
            "export_e_area_non_40_penalty",
            defaults.export_e_area_non_40_penalty,
        ),
        verbose=not args.quiet,
        use_scip=not args.no_scip,
        scip_disable_symmetry=not getattr(args, "enable_scip_symmetry", False),
    )


def resolve_adapter_json(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved.is_dir():
        resolved = resolved / "input_data.json"
    if not resolved.exists():
        raise FileNotFoundError(f"InputAdapterGd JSON not found: {resolved}")
    return resolved


def adapter_data_dir(adapter_json: str | Path) -> Path:
    return resolve_adapter_json(adapter_json).parent


def adapter_object_label(adapter_input: InputAdapterGd) -> str:
    local_path = getattr(adapter_input, "local_path", None)
    return str(local_path) if local_path else "InputAdapterGd object"


def build_problem_from_large_plan_adapter(
    adapter_input: InputAdapterGd,
    big_plan_path: str | Path,
    planning_time: datetime,
    voyages: list[str] | None,
    horizon_hours: float,
    misplaced_bay_exclusion_ratio: float,
):
    big_plan_frame = pd.read_csv(big_plan_path, encoding="utf-8-sig")
    target_voyages = [_normalize_voyage(voyage) for voyage in voyages] if voyages else None
    if target_voyages is None:
        big_plan_rows = adapter_data_io.read_big_plan(big_plan_frame)
        target_voyages = infer_target_voyages_from_big_plan(big_plan_rows, planning_time)
    inputs = adapter_data_io.load_medium_small_inputs(
        adapter_input,
        planning_time=planning_time,
        voyages=target_voyages,
        horizon_hours=horizon_hours,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
        big_plan=big_plan_frame,
    )
    return adapter_object_label(adapter_input), inputs.problem, inputs.big_plan, inputs.demand_rows


def build_problem_from_large_plan_json(
    adapter_json: str | Path,
    big_plan_path: str | Path,
    planning_time: datetime,
    voyages: list[str] | None,
    horizon_hours: float,
    misplaced_bay_exclusion_ratio: float,
):
    adapter_input = InputAdapterGd.load_from_json(str(resolve_adapter_json(adapter_json)))
    return build_problem_from_large_plan_adapter(
        adapter_input,
        big_plan_path,
        planning_time=planning_time,
        voyages=voyages,
        horizon_hours=horizon_hours,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
    )


def build_problem_from_large_plan_records_adapter(
    adapter_input: InputAdapterGd,
    allocation_rows: list[dict[str, Any]],
    planning_time: datetime,
    voyages: list[str] | None,
    horizon_hours: float,
    misplaced_bay_exclusion_ratio: float,
):
    big_plan = big_plan_rows_from_allocation_records(allocation_rows)
    target_voyages = [_normalize_voyage(voyage) for voyage in voyages] if voyages else None
    if target_voyages is None:
        target_voyages = infer_target_voyages_from_big_plan(big_plan, planning_time)
    inputs = adapter_data_io.load_medium_small_inputs(
        adapter_input,
        planning_time=planning_time,
        voyages=target_voyages,
        horizon_hours=horizon_hours,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
        big_plan=big_plan,
    )
    return adapter_object_label(adapter_input), inputs.problem, inputs.big_plan, inputs.demand_rows


def build_problem_from_large_plan_records_json(
    adapter_json: str | Path,
    allocation_rows: list[dict[str, Any]],
    planning_time: datetime,
    voyages: list[str] | None,
    horizon_hours: float,
    misplaced_bay_exclusion_ratio: float,
):
    adapter_input = InputAdapterGd.load_from_json(str(resolve_adapter_json(adapter_json)))
    return build_problem_from_large_plan_records_adapter(
        adapter_input,
        allocation_rows,
        planning_time=planning_time,
        voyages=voyages,
        horizon_hours=horizon_hours,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
    )


def big_plan_rows_from_allocation_records(records: list[dict[str, Any]]) -> list[BigPlanRow]:
    counter: Counter[tuple[str, str, str, str, str]] = Counter()
    for record in records:
        qty_value = _first_value(record, "new_qty", "planned_boxes", "planned_qty")
        try:
            boxes = int(round(float(qty_value or 0)))
        except (TypeError, ValueError):
            boxes = 0
        if boxes <= 0:
            continue
        voyage_id = _normalize_voyage(_first_value(record, "voy_id", "voyage_id", "voyage", "voy"))
        flow = _flow(_first_value(record, "flow", "cntr_type", "status") or "OF")
        area_no = _norm(_first_value(record, "area_no", "area", "yard_area", "block"))
        size_mode = _big_plan_size_mode(_first_value(record, "size", "size_mode", "cntr_size"))
        plan_date = _date_key(_norm(_first_value(record, "plan_date", "date", "work_date", "planning_date", "day")))
        if voyage_id and area_no:
            counter[(voyage_id, flow, area_no, size_mode, plan_date)] += boxes
    rows = [
        BigPlanRow(voyage_id, flow, area_no, boxes, size_mode, plan_date)
        for (voyage_id, flow, area_no, size_mode, plan_date), boxes in sorted(counter.items())
        if boxes > 0
    ]
    if not rows:
        raise ValueError("large plan allocation records contain no positive planned quantity")
    return rows


def infer_target_voyages_from_big_plan(big_plan: list[Any], planning_time: datetime) -> list[str]:
    plan_date = planning_time.date().isoformat()
    target_flows = {_flow(flow) for flow in DEFAULT_TARGET_BIG_PLAN_FLOWS}
    voyages = {
        str(row.voyage_id)
        for row in big_plan
        if str(getattr(row, "voyage_id", ""))
        and _flow(getattr(row, "flow", "")) in target_flows
        and (not getattr(row, "plan_date", "") or str(getattr(row, "plan_date", "")) == plan_date)
    }
    return sorted(voyages, key=_voyage_sort_key)


def write_medium_demand_rows(
    output_path: str | Path,
    data_dir: str | Path,
    big_plan: list,
    target_voyages: list[str],
    planning_time: datetime,
    horizon_hours: float,
) -> None:
    caps = medium_demand_caps_from_big_plan(
        big_plan,
        target_voyages,
        planning_time,
        DEFAULT_TARGET_BIG_PLAN_FLOWS,
    )
    rows = calculate_medium_demands(
        data_dir,
        target_voyages,
        planning_time=planning_time,
        horizon_hours=horizon_hours,
        big_plan_caps=caps,
    )
    write_rows(output_path, [row.__dict__ for row in rows])


def write_medium_demand_rows_from_rows(output_path: str | Path, rows: list[Any]) -> None:
    write_rows(output_path, [row.__dict__ for row in rows])


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


def solve_medium_small_problem(
    problem,
    config: ColumnGenerationConfig,
    output_dir: str | Path,
    diagnostics_context: dict[str, Any] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()

    planner_init_start = perf_counter()
    planner = ColumnGenerationPlanner(problem, config)
    planner_init_seconds = perf_counter() - planner_init_start
    planner_solve_start = perf_counter()
    result = planner.solve()
    planner_solve_seconds = perf_counter() - planner_solve_start

    output_write_timings: dict[str, float] = {}
    output_write_start = perf_counter()
    step_start = perf_counter()
    write_rows(output_path / "small_plan.csv", result.small_rows)
    output_write_timings["write_small_plan_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "medium_plan.csv", result.medium_rows)
    output_write_timings["write_medium_plan_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "unplaced_boxes.csv", result.unplaced_rows)
    output_write_timings["write_unplaced_boxes_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "small_plan_six_bay_blocks.csv", make_six_bay_block_rows(result.small_rows))
    output_write_timings["write_small_plan_six_bay_blocks_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "big_plan_used.csv", [row.__dict__ for row in problem.big_plan])
    output_write_timings["write_big_plan_used_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_columns(output_path / "generated_columns.csv", result.columns)
    output_write_timings["write_generated_columns_seconds"] = round(perf_counter() - step_start, 3)
    output_write_seconds = perf_counter() - output_write_start

    elapsed_seconds = perf_counter() - started_at
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "output_dir": str(output_path),
            "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "planner_init_seconds": round(planner_init_seconds, 3),
            "planner_solve_seconds": round(planner_solve_seconds, 3),
            "output_write_seconds": round(output_write_seconds, 3),
            **output_write_timings,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_time": format_duration(elapsed_seconds),
        }
    )
    if diagnostics_context:
        diagnostics.update(diagnostics_context)
    write_json(output_path / "diagnostics.json", diagnostics)
    return result, diagnostics


def solve_small_plan_from_medium_adapter(
    adapter_input: InputAdapterGd,
    medium_plan_path: str | Path,
    config: ColumnGenerationConfig,
    output_dir: str | Path,
    planning_time: datetime,
    horizon_hours: float,
    voyages: list[str] | None = None,
    big_plan_path: str | Path | None = None,
    misplaced_bay_exclusion_ratio: float = 0.0,
    diagnostics_context: dict[str, Any] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    external_plan = read_external_medium_plan(medium_plan_path)
    target_voyages = [_normalize_voyage(v) for v in voyages] if voyages else list(external_plan.target_voyages)
    big_plan = read_external_big_plan(big_plan_path) if big_plan_path else list(external_plan.big_plan_rows)
    inputs = adapter_data_io.load_medium_small_inputs(
        adapter_input,
        planning_time=planning_time,
        voyages=target_voyages,
        horizon_hours=horizon_hours,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
        big_plan=big_plan,
    )
    problem = inputs.problem
    external_plan = filter_external_medium_plan(external_plan, problem.target_voyages)
    problem = apply_external_medium_plan(problem, external_plan)

    started_at = perf_counter()
    planner_init_start = perf_counter()
    planner = ColumnGenerationPlanner(problem, configure_small_plan_from_medium(config, external_plan))
    planner_init_seconds = perf_counter() - planner_init_start
    planner_solve_start = perf_counter()
    result = planner.solve()
    planner_solve_seconds = perf_counter() - planner_solve_start

    output_write_timings: dict[str, float] = {}
    output_write_start = perf_counter()
    step_start = perf_counter()
    write_rows(output_path / "small_plan.csv", result.small_rows)
    output_write_timings["write_small_plan_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "small_plan_six_bay_blocks.csv", make_six_bay_block_rows(result.small_rows))
    output_write_timings["write_small_plan_six_bay_blocks_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "small_plan_medium_summary.csv", result.medium_rows)
    output_write_timings["write_small_plan_medium_summary_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "unplaced_boxes.csv", result.unplaced_rows)
    output_write_timings["write_unplaced_boxes_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "external_medium_plan_used.csv", external_plan.rows)
    output_write_timings["write_external_medium_plan_used_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_rows(output_path / "medium_plan_big_quota.csv", [row.__dict__ for row in external_plan.big_plan_rows])
    output_write_timings["write_medium_plan_big_quota_seconds"] = round(perf_counter() - step_start, 3)
    step_start = perf_counter()
    write_columns(output_path / "generated_columns.csv", result.columns)
    output_write_timings["write_generated_columns_seconds"] = round(perf_counter() - step_start, 3)
    output_write_seconds = perf_counter() - output_write_start

    elapsed_seconds = perf_counter() - started_at
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "algorithm": "flat_json_small_plan_from_external_medium_column_generation",
            "medium_plan_input": str(Path(medium_plan_path).resolve()),
            "external_medium_row_count": len(external_plan.rows),
            "external_medium_box_count": sum(int(row["planned_boxes"]) for row in external_plan.rows),
            "external_medium_quota_count": len(external_plan.coarse_area_quota),
            "external_medium_bay_quota_count": len(external_plan.coarse_bay_quota),
            "data_dir": adapter_object_label(adapter_input),
            "planning_time": planning_time.isoformat(sep=" "),
            "output_dir": str(output_path),
            "created_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "planner_init_seconds": round(planner_init_seconds, 3),
            "planner_solve_seconds": round(planner_solve_seconds, 3),
            "output_write_seconds": round(output_write_seconds, 3),
            **output_write_timings,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "elapsed_time": format_duration(elapsed_seconds),
        }
    )
    if diagnostics_context:
        diagnostics.update(diagnostics_context)
    write_json(output_path / "diagnostics.json", diagnostics)
    return result, diagnostics


def solve_small_plan_from_medium_json(
    adapter_json: str | Path,
    medium_plan_path: str | Path,
    config: ColumnGenerationConfig,
    output_dir: str | Path,
    planning_time: datetime,
    horizon_hours: float,
    voyages: list[str] | None = None,
    big_plan_path: str | Path | None = None,
    misplaced_bay_exclusion_ratio: float = 0.0,
    diagnostics_context: dict[str, Any] | None = None,
):
    adapter_input = InputAdapterGd.load_from_json(str(resolve_adapter_json(adapter_json)))
    return solve_small_plan_from_medium_adapter(
        adapter_input,
        medium_plan_path=medium_plan_path,
        config=config,
        output_dir=output_dir,
        planning_time=planning_time,
        horizon_hours=horizon_hours,
        voyages=voyages,
        big_plan_path=big_plan_path,
        misplaced_bay_exclusion_ratio=misplaced_bay_exclusion_ratio,
        diagnostics_context=diagnostics_context,
    )


def print_diagnostics_summary(diagnostics: dict[str, Any]) -> None:
    inheritance = diagnostics.get("medium_big_plan_inheritance", {}) or {}
    fragmentation = diagnostics.get("medium_fragmentation", {}) or {}
    summary = {
        "algorithm": diagnostics.get("algorithm"),
        "master_algorithm": diagnostics.get("master_algorithm"),
        "master_status": diagnostics.get("master_status"),
        "medium_plan_source": diagnostics.get("medium_plan_source"),
        "medium_plan_granularity": diagnostics.get("medium_plan_granularity"),
        "planned_medium_boxes": diagnostics.get("planned_medium_boxes"),
        "planned_small_boxes": diagnostics.get("planned_small_boxes"),
        "unplaced_boxes": diagnostics.get("unplaced_boxes"),
        "inheritance_ratio": inheritance.get("inheritance_ratio"),
        "tiny_area_rows": fragmentation.get("tiny_area_rows"),
        "large_coarse_tiny_area_rows": fragmentation.get("large_coarse_tiny_area_rows"),
        "final_column_count": diagnostics.get("final_column_count"),
        "elapsed_time": diagnostics.get("elapsed_time"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _normalize_voyage(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _voyage_sort_key(value: object) -> tuple[int, str]:
    text = _normalize_voyage(value)
    try:
        return int(text), text
    except ValueError:
        return 10**12, text


def _first_value(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record.get(name)
    return None


def _norm(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _flow(value: object) -> str:
    return adapter_data_io.medium_small_area_flow(_norm(value).upper())


def _date_key(value: str) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return value[:10]


def _big_plan_size_mode(value: object) -> str:
    text = _norm(value).upper()
    if "20" in text:
        return "20"
    if "45" in text:
        return "40"
    if "40" in text:
        return "40"
    return "ALL"
