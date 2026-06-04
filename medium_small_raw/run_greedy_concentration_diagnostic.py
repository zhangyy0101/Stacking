from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medium_small_raw.block_bay_planning.runner import (  # noqa: E402
    add_big_plan_argument,
    add_common_arguments,
    build_problem_from_big_plan,
    create_run_output_dir,
    log,
)
from medium_small_raw.column_generation_planner import (  # noqa: E402
    ColumnGenerationConfig,
    ColumnGenerationPlanner,
    SmallBoxGroup,
    write_columns,
    write_json,
    write_rows,
)


SCENARIOS = {
    "strict_quota": ("stage0", True),
    "big_plan_area": ("stage1b", False),
    "physical": ("stage3", False),
}
STAGED_SCENARIO = "staged_concentration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a concentration-first greedy diagnostic for medium/small raw planning. "
            "This is an analysis tool, not the production optimizer."
        )
    )
    add_common_arguments(parser)
    add_big_plan_argument(parser)
    parser.add_argument(
        "--scenario",
        choices=sorted([*SCENARIOS, STAGED_SCENARIO]),
        default=STAGED_SCENARIO,
        help=(
            "staged_concentration runs stage0 under hard big-plan quota upper bounds, then stage1-3 concentration-only repair; "
            "strict_quota/big_plan_area/physical run one-shot legacy diagnostics."
        ),
    )
    parser.add_argument("--demand-mode", default="original")
    parser.add_argument("--small-group-threshold", type=int, default=26)
    parser.add_argument("--large-target-area-boxes", type=int, default=60)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "greedy_concentration_diagnostic",
    )
    return parser.parse_args()


def group_key(group: SmallBoxGroup) -> tuple[str, str, str, str]:
    return group.voyage_id, group.status, group.port, group.size


def group_sort_key(group: SmallBoxGroup) -> tuple:
    return (
        group.voyage_id,
        group.status,
        group.port,
        {"20": 0, "40": 1, "45": 2}.get(group.size, 9),
        -int(group.demand),
        group.height,
        group.group_id,
    )


def bucket_order(planner: ColumnGenerationPlanner, groups_by_coarse: dict[tuple[str, str, str, str], list[SmallBoxGroup]]) -> list[tuple[str, str, str, str]]:
    threshold = int(planner.config.medium_concentrated_group_threshold or 0)

    def key(coarse_key: tuple[str, str, str, str]) -> tuple:
        demand = int(planner.coarse_demand.get(coarse_key, 0))
        is_small = threshold > 0 and demand <= threshold
        return (
            0 if is_small else 1,
            demand if is_small else -demand,
            coarse_key,
        )

    return sorted(groups_by_coarse, key=key)


def area_candidates_for_bucket(
    planner: ColumnGenerationPlanner,
    groups: list[SmallBoxGroup],
    scope: str,
) -> list[str]:
    areas: set[str] = set()
    for group in groups:
        areas.update(planner._candidate_areas_for_group(group, scope=scope))
    sample = groups[0]
    return sorted(
        areas,
        key=lambda area_no: (
            planner._area_fallback_tier_for_group(sample, area_no),
            area_no,
        ),
    )


def add_column_for_choice(
    planner: ColumnGenerationPlanner,
    group: SmallBoxGroup,
    bay_key: str,
    quantity: int,
    base_cost: float,
    state: dict,
) -> int | None:
    patterns = planner._row_allocation_patterns_for_column(group, bay_key, quantity, state=state, max_patterns=1)
    if not patterns:
        return None
    before = len(planner.columns)
    added = planner._add_column(group, bay_key, quantity, base_cost, row_allocation=patterns[0], state=state)
    if added and len(planner.columns) > before:
        return len(planner.columns) - 1
    return planner._column_index_for(group.group_id, bay_key, quantity, state=state)


def concentration_choice_score(
    planner: ColumnGenerationPlanner,
    group: SmallBoxGroup,
    bay_key: str,
    quantity: int,
    state: dict,
    coarse_area_load: Counter[tuple[str, str, str, str, str]],
) -> tuple:
    bay = planner.bays[bay_key]
    area_no = bay.area_no
    block_id = planner.block_by_bay.get((area_no, bay_key), "")
    coarse_key = planner._coarse_key(group)
    coarse_area_key = coarse_key + (area_no,)
    coarse_delta = planner._coarse_area_incremental_repair_cost(
        coarse_area_key,
        quantity,
        coarse_area_load,
    )
    return (
        coarse_delta,
        0 if (group.group_id, area_no) in state["used_group_area"] else 1,
        0 if coarse_area_load.get(coarse_area_key, 0) > 0 else 1,
        0 if block_id and (group.group_id, block_id) in state["used_group_block"] else 1,
        0 if coarse_key + (area_no, bay_key) in state["used_coarse_area_bay"] else 1,
        -int(coarse_area_load.get(coarse_area_key, 0)),
        -int(quantity),
        area_no,
        bay.bay_order,
        bay_key,
    )


def place_group(
    planner: ColumnGenerationPlanner,
    group: SmallBoxGroup,
    allowed_areas: set[str],
    scope: str,
    enforce_quota: bool,
    state: dict,
    selected: Counter[int],
    placed: Counter[str],
    coarse_area_load: Counter[tuple[str, str, str, str, str]],
    balance_large: bool,
    concentration_only: bool = False,
) -> int:
    placed_now = 0
    remaining = int(group.demand) - int(placed.get(group.group_id, 0))
    while remaining > 0:
        choices = []
        for bay_key, _max_qty, base_cost in planner._candidate_bays_for_group(group, scope=scope):
            bay = planner.bays[bay_key]
            if bay.area_no not in allowed_areas:
                continue
            capacity = planner._remaining_capacity_for_group_bay(
                group,
                bay_key,
                state,
                remaining,
                enforce_quota=enforce_quota,
            )
            if capacity <= 0:
                continue
            quantity = min(remaining, int(capacity))
            coarse_key = planner._coarse_key(group)
            area_load = coarse_area_load[coarse_key + (bay.area_no,)]
            if concentration_only:
                score = concentration_choice_score(
                    planner,
                    group,
                    bay_key,
                    quantity,
                    state,
                    coarse_area_load,
                )
            else:
                score = (
                    area_load if balance_large else 0,
                    planner._area_fallback_tier_for_group(group, bay.area_no),
                    base_cost,
                    -quantity,
                    bay.area_no,
                    bay.bay_order,
                    bay_key,
                )
            choices.append((score, bay_key, quantity, base_cost))
        if not choices:
            break
        _score, bay_key, quantity, base_cost = min(choices)
        idx = add_column_for_choice(planner, group, bay_key, quantity, base_cost, state)
        if idx is None:
            break
        col = planner.columns[idx]
        planner._apply_column_to_state(col, state)
        selected[idx] += 1
        placed[group.group_id] += col.quantity
        coarse_area_load[col.coarse_key + (col.area_no,)] += col.quantity
        placed_now += col.quantity
        remaining -= col.quantity
    return placed_now


def place_bucket_in_areas(
    planner: ColumnGenerationPlanner,
    groups: list[SmallBoxGroup],
    allowed_areas: set[str],
    scope: str,
    enforce_quota: bool,
    state: dict,
    selected: Counter[int],
    placed: Counter[str],
    coarse_area_load: Counter[tuple[str, str, str, str, str]],
    balance_large: bool,
    concentration_only: bool = False,
) -> int:
    before = sum(placed.get(group.group_id, 0) for group in groups)
    for group in sorted(groups, key=lambda g: (-int(g.demand), g.height, g.group_id)):
        place_group(
            planner,
            group,
            allowed_areas,
            scope,
            enforce_quota,
            state,
            selected,
            placed,
            coarse_area_load,
            balance_large=balance_large,
            concentration_only=concentration_only,
        )
    after = sum(placed.get(group.group_id, 0) for group in groups)
    return int(after - before)


def choose_large_areas(
    planner: ColumnGenerationPlanner,
    groups: list[SmallBoxGroup],
    candidate_areas: list[str],
    target_count: int,
) -> set[str]:
    scored = []
    for area_no in candidate_areas:
        rough_capacity = 0
        fallback_tier = 0
        for group in groups:
            fallback_tier += planner._area_fallback_tier_for_group(group, area_no)
            for bay_key, max_qty, _base_cost in planner._candidate_bays_for_group(group, scope="stage3"):
                if planner.bays[bay_key].area_no == area_no:
                    rough_capacity += int(max_qty)
        scored.append((-rough_capacity, fallback_tier, area_no))
    selected = [area_no for _cap, _tier, area_no in sorted(scored)[: max(1, target_count)]]
    return set(selected)


def used_areas_for_coarse(
    coarse_area_load: Counter[tuple[str, str, str, str, str]],
    coarse_key: tuple[str, str, str, str],
) -> set[str]:
    return {
        area_no
        for (*key, area_no), qty in coarse_area_load.items()
        if tuple(key) == coarse_key and qty > 0
    }


def place_bucket_concentration_first(
    planner: ColumnGenerationPlanner,
    groups: list[SmallBoxGroup],
    coarse_key: tuple[str, str, str, str],
    scope: str,
    enforce_quota: bool,
    state: dict,
    selected: Counter[int],
    placed: Counter[str],
    coarse_area_load: Counter[tuple[str, str, str, str, str]],
    try_single_area_for_small: bool,
) -> dict:
    demand = sum(int(group.demand) for group in groups)
    before = sum(placed.get(group.group_id, 0) for group in groups)
    threshold = int(planner.config.medium_concentrated_group_threshold or 0)
    is_small = threshold > 0 and demand <= threshold
    candidate_areas = area_candidates_for_bucket(planner, groups, scope=scope)
    chosen_areas: set[str] = set(candidate_areas)
    single_area_fit = False
    reason = ""

    if not candidate_areas:
        reason = "no_candidate_area"
    elif is_small and try_single_area_for_small:
        best = None
        for area_no in candidate_areas:
            trial_state = copy.deepcopy(state)
            trial_selected = Counter(selected)
            trial_placed = Counter(placed)
            trial_load = Counter(coarse_area_load)
            trial_added = place_bucket_in_areas(
                planner,
                groups,
                {area_no},
                scope,
                enforce_quota,
                trial_state,
                trial_selected,
                trial_placed,
                trial_load,
                balance_large=False,
                concentration_only=True,
            )
            candidate = (-trial_added, area_no, trial_state, trial_selected, trial_placed, trial_load)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            _neg_added, best_area, best_state, best_selected, best_placed, best_load = best
            state.clear()
            state.update(best_state)
            selected.clear()
            selected.update(best_selected)
            placed.clear()
            placed.update(best_placed)
            coarse_area_load.clear()
            coarse_area_load.update(best_load)
            chosen_areas = {best_area}
            single_area_fit = (sum(placed.get(group.group_id, 0) for group in groups) - before) >= demand
            if not single_area_fit:
                reason = "single_area_row_or_capacity_shortage"
                place_bucket_in_areas(
                    planner,
                    groups,
                    set(candidate_areas),
                    scope,
                    enforce_quota,
                    state,
                    selected,
                    placed,
                    coarse_area_load,
                    balance_large=False,
                    concentration_only=True,
                )
    else:
        existing_areas = used_areas_for_coarse(coarse_area_load, coarse_key) & set(candidate_areas)
        if existing_areas:
            placed_existing = place_bucket_in_areas(
                planner,
                groups,
                existing_areas,
                scope,
                enforce_quota,
                state,
                selected,
                placed,
                coarse_area_load,
                balance_large=not is_small,
                concentration_only=True,
            )
            chosen_areas = set(existing_areas)
            if placed_existing < demand - before:
                reason = "existing_area_shortage"
        remaining_after_existing = demand - sum(placed.get(group.group_id, 0) for group in groups)
        if remaining_after_existing > 0:
            if is_small:
                target_areas = set(candidate_areas)
            else:
                target_count = planner._target_large_group_area_count(coarse_key, demand)
                target_areas = choose_large_areas(planner, groups, candidate_areas, target_count)
            chosen_areas |= set(target_areas)
            placed_in_target = place_bucket_in_areas(
                planner,
                groups,
                target_areas,
                scope,
                enforce_quota,
                state,
                selected,
                placed,
                coarse_area_load,
                balance_large=not is_small,
                concentration_only=True,
            )
            if placed_in_target < remaining_after_existing:
                reason = reason or "target_area_set_shortage"
                place_bucket_in_areas(
                    planner,
                    groups,
                    set(candidate_areas),
                    scope,
                    enforce_quota,
                    state,
                    selected,
                    placed,
                    coarse_area_load,
                    balance_large=not is_small,
                    concentration_only=True,
                )

    after = sum(placed.get(group.group_id, 0) for group in groups)
    used_areas = used_areas_for_coarse(coarse_area_load, coarse_key)
    return {
        "coarse_key": "|".join(coarse_key),
        "demand": demand,
        "placed": int(after),
        "placed_delta": int(after - before),
        "unplaced": max(0, demand - int(after)),
        "is_small_coarse_group": bool(is_small),
        "single_area_fit": bool(single_area_fit) if is_small else None,
        "candidate_area_count": len(candidate_areas),
        "chosen_area_count": len(chosen_areas),
        "used_area_count": len(used_areas),
        "reason": reason,
    }


def run_staged_concentration_greedy(planner: ColumnGenerationPlanner) -> tuple[Counter[int], Counter[str], dict]:
    state = planner._empty_selection_state()
    selected: Counter[int] = Counter()
    placed: Counter[str] = Counter()
    coarse_area_load: Counter[tuple[str, str, str, str, str]] = Counter()
    groups_by_coarse: defaultdict[tuple[str, str, str, str], list[SmallBoxGroup]] = defaultdict(list)
    for group in planner.groups:
        groups_by_coarse[group_key(group)].append(group)

    ordered_coarse = bucket_order(planner, groups_by_coarse)
    stage_stats = []
    stages = [
        ("stage0", True, True),
        ("stage1a", True, False),
        ("stage1b", False, False),
        ("stage2", False, False),
        ("stage3", False, False),
    ]
    for stage, enforce_quota, try_single_area in stages:
        before_unplaced = sum(max(0, int(group.demand) - int(placed.get(group.group_id, 0))) for group in planner.groups)
        before_columns = len(planner.columns)
        before_selected = sum(1 for qty in selected.values() if qty > 0)
        if before_unplaced <= 0:
            stage_stats.append(
                {
                    "stage": stage,
                    "enforce_big_plan_quota": enforce_quota,
                    "before_unplaced_boxes": 0,
                    "after_unplaced_boxes": 0,
                    "placed_boxes": 0,
                    "added_columns": 0,
                    "added_selected_columns": 0,
                    "skipped": True,
                }
            )
            continue
        for coarse_key in ordered_coarse:
            groups = sorted(groups_by_coarse[coarse_key], key=group_sort_key)
            if sum(max(0, int(group.demand) - int(placed.get(group.group_id, 0))) for group in groups) <= 0:
                continue
            place_bucket_concentration_first(
                planner,
                groups,
                coarse_key,
                stage,
                enforce_quota,
                state,
                selected,
                placed,
                coarse_area_load,
                try_single_area_for_small=try_single_area,
            )
        after_unplaced = sum(max(0, int(group.demand) - int(placed.get(group.group_id, 0))) for group in planner.groups)
        stage_stats.append(
            {
                "stage": stage,
                "enforce_big_plan_quota": enforce_quota,
                "before_unplaced_boxes": before_unplaced,
                "after_unplaced_boxes": after_unplaced,
                "placed_boxes": before_unplaced - after_unplaced,
                "added_columns": len(planner.columns) - before_columns,
                "added_selected_columns": sum(1 for qty in selected.values() if qty > 0) - before_selected,
                "skipped": False,
            }
        )

    bucket_stats = []
    threshold = int(planner.config.medium_concentrated_group_threshold or 0)
    for coarse_key in ordered_coarse:
        groups = sorted(groups_by_coarse[coarse_key], key=group_sort_key)
        demand = sum(int(group.demand) for group in groups)
        placed_qty = sum(placed.get(group.group_id, 0) for group in groups)
        used_areas = used_areas_for_coarse(coarse_area_load, coarse_key)
        is_small = threshold > 0 and demand <= threshold
        bucket_stats.append(
            {
                "coarse_key": "|".join(coarse_key),
                "demand": demand,
                "placed": int(placed_qty),
                "unplaced": max(0, demand - int(placed_qty)),
                "is_small_coarse_group": bool(is_small),
                "single_area_fit": bool(is_small and placed_qty >= demand and len(used_areas) <= 1) if is_small else None,
                "candidate_area_count": None,
                "chosen_area_count": None,
                "used_area_count": len(used_areas),
                "reason": "unplaced_after_stage3" if placed_qty < demand else "",
            }
        )

    unplaced = Counter(
        {
            group.group_id: int(group.demand) - int(placed.get(group.group_id, 0))
            for group in planner.groups
            if int(group.demand) - int(placed.get(group.group_id, 0)) > 0
        }
    )
    stats = {
        "scenario": STAGED_SCENARIO,
        "candidate_scope": "stage0->stage1a->stage1b->stage2->stage3",
        "enforce_big_plan_quota": "stage0_and_stage1a_only",
        "bucket_stats": bucket_stats,
        "stage_stats": stage_stats,
    }
    return selected, unplaced, stats


def run_greedy(planner: ColumnGenerationPlanner, scenario: str) -> tuple[Counter[int], Counter[str], dict]:
    if scenario == STAGED_SCENARIO:
        return run_staged_concentration_greedy(planner)

    scope, enforce_quota = SCENARIOS[scenario]
    state = planner._empty_selection_state()
    selected: Counter[int] = Counter()
    placed: Counter[str] = Counter()
    coarse_area_load: Counter[tuple[str, str, str, str, str]] = Counter()
    groups_by_coarse: defaultdict[tuple[str, str, str, str], list[SmallBoxGroup]] = defaultdict(list)
    for group in planner.groups:
        groups_by_coarse[group_key(group)].append(group)

    bucket_stats = []
    threshold = int(planner.config.medium_concentrated_group_threshold or 0)
    for coarse_key in bucket_order(planner, groups_by_coarse):
        groups = sorted(groups_by_coarse[coarse_key], key=group_sort_key)
        demand = sum(int(group.demand) for group in groups)
        before = sum(placed.get(group.group_id, 0) for group in groups)
        candidate_areas = area_candidates_for_bucket(planner, groups, scope=scope)
        is_small = threshold > 0 and demand <= threshold
        chosen_areas: set[str] = set(candidate_areas)
        single_area_fit = False
        reason = ""

        if is_small and candidate_areas:
            best = None
            for area_no in candidate_areas:
                trial_state = copy.deepcopy(state)
                trial_selected = Counter(selected)
                trial_placed = Counter(placed)
                trial_load = Counter(coarse_area_load)
                trial_added = place_bucket_in_areas(
                    planner,
                    groups,
                    {area_no},
                    scope,
                    enforce_quota,
                    trial_state,
                    trial_selected,
                    trial_placed,
                    trial_load,
                    balance_large=False,
                )
                tier = planner._area_fallback_tier_for_group(groups[0], area_no)
                candidate = (-trial_added, tier, area_no, trial_state, trial_selected, trial_placed, trial_load)
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                _neg_added, _tier, best_area, state, selected, placed, coarse_area_load = best
                chosen_areas = {best_area}
                single_area_fit = (sum(placed.get(group.group_id, 0) for group in groups) - before) >= demand
                if not single_area_fit:
                    reason = "single_area_row_or_capacity_shortage"
                    place_bucket_in_areas(
                        planner,
                        groups,
                        set(candidate_areas),
                        scope,
                        enforce_quota,
                        state,
                        selected,
                        placed,
                        coarse_area_load,
                        balance_large=False,
                    )
        elif candidate_areas:
            target_count = planner._target_large_group_area_count(coarse_key, demand)
            chosen_areas = choose_large_areas(planner, groups, candidate_areas, target_count)
            placed_in_target = place_bucket_in_areas(
                planner,
                groups,
                chosen_areas,
                scope,
                enforce_quota,
                state,
                selected,
                placed,
                coarse_area_load,
                balance_large=True,
            )
            if placed_in_target < demand:
                reason = "target_area_set_shortage"
                place_bucket_in_areas(
                    planner,
                    groups,
                    set(candidate_areas),
                    scope,
                    enforce_quota,
                    state,
                    selected,
                    placed,
                    coarse_area_load,
                    balance_large=True,
                )
        else:
            reason = "no_candidate_area"

        after = sum(placed.get(group.group_id, 0) for group in groups)
        used_areas = {
            area_no
            for (*key, area_no), qty in coarse_area_load.items()
            if tuple(key) == coarse_key and qty > 0
        }
        bucket_stats.append(
            {
                "coarse_key": "|".join(coarse_key),
                "demand": demand,
                "placed": int(after - before),
                "unplaced": max(0, demand - int(after - before)),
                "is_small_coarse_group": bool(is_small),
                "single_area_fit": bool(single_area_fit) if is_small else None,
                "candidate_area_count": len(candidate_areas),
                "chosen_area_count": len(chosen_areas),
                "used_area_count": len(used_areas),
                "reason": reason,
            }
        )

    unplaced = Counter(
        {
            group.group_id: int(group.demand) - int(placed.get(group.group_id, 0))
            for group in planner.groups
            if int(group.demand) - int(placed.get(group.group_id, 0)) > 0
        }
    )
    stats = {
        "scenario": scenario,
        "candidate_scope": scope,
        "enforce_big_plan_quota": enforce_quota,
        "bucket_stats": bucket_stats,
    }
    return selected, unplaced, stats


def make_summary(planner: ColumnGenerationPlanner, selected: Counter[int], unplaced: Counter[str], stats: dict) -> dict:
    small_rows = planner._make_small_rows(selected)
    medium_rows = planner._make_medium_rows_from_selected_columns(selected, plan_level="medium_greedy")
    fragmentation = planner._medium_fragmentation_stats(medium_rows)
    inheritance = planner._medium_big_plan_inheritance_stats(medium_rows)
    bucket_stats = stats["bucket_stats"]
    small_buckets = [row for row in bucket_stats if row["is_small_coarse_group"]]
    large_buckets = [row for row in bucket_stats if not row["is_small_coarse_group"]]
    planned_boxes = sum(int(row.get("planned_boxes", 0) or 0) for row in small_rows)
    return {
        "algorithm": "greedy_concentration_diagnostic",
        "scenario": stats["scenario"],
        "candidate_scope": stats["candidate_scope"],
        "enforce_big_plan_quota": stats["enforce_big_plan_quota"],
        "planned_boxes": planned_boxes,
        "planned_small_boxes": planned_boxes,
        "unplaced_boxes": sum(unplaced.values()),
        "selected_column_count": sum(1 for qty in selected.values() if qty > 0),
        "final_column_count": len(planner.columns),
        "medium_fragmentation": fragmentation,
        "medium_big_plan_inheritance": inheritance,
        "medium_area_rows_below_min_boxes": planner._count_medium_area_rows_below_min(medium_rows),
        "small_single_area_fit_groups": sum(1 for row in small_buckets if row["single_area_fit"]),
        "small_single_area_shortage_groups": sum(1 for row in small_buckets if row["reason"] == "single_area_row_or_capacity_shortage"),
        "small_unplaced_groups": sum(1 for row in small_buckets if row["unplaced"] > 0),
        "large_target_area_shortage_groups": sum(1 for row in large_buckets if row["reason"] == "target_area_set_shortage"),
        "large_unplaced_groups": sum(1 for row in large_buckets if row["unplaced"] > 0),
        "stage_stats": stats.get("stage_stats", []),
    }


def main() -> None:
    args = parse_args()
    log("building problem data for greedy concentration diagnostic")
    data_dir, planning_time, problem, _big_plan = build_problem_from_big_plan(args)
    output_dir = create_run_output_dir(args.output_dir)
    config = ColumnGenerationConfig(
        demand_mode=args.demand_mode,
        verbose=False,
        max_candidate_bays_per_group=0,
        medium_concentrated_group_threshold=args.small_group_threshold,
        medium_large_group_target_area_boxes=args.large_target_area_boxes,
    )
    planner = ColumnGenerationPlanner(problem, config)
    log(f"running greedy diagnostic scenario={args.scenario}")
    selected, unplaced, stats = run_greedy(planner, args.scenario)
    small_rows = planner._make_small_rows(selected)
    medium_rows = planner._make_medium_rows_from_selected_columns(selected, plan_level="medium_greedy")
    summary = make_summary(planner, selected, unplaced, stats)
    summary.update(
        {
            "data_dir": str(data_dir),
            "big_plan": str(args.big_plan),
            "planning_time": planning_time.isoformat(sep=" "),
            "output_dir": str(output_dir),
        }
    )
    write_rows(output_dir / "greedy_small_plan.csv", small_rows)
    write_rows(output_dir / "greedy_medium_plan.csv", medium_rows)
    write_columns(output_dir / "greedy_generated_columns.csv", planner.columns)
    write_rows(output_dir / "greedy_bucket_diagnostics.csv", stats["bucket_stats"])
    if stats.get("stage_stats"):
        write_rows(output_dir / "greedy_stage_diagnostics.csv", stats["stage_stats"])
    write_json(output_dir / "diagnostics.json", {**summary, "bucket_stats": stats["bucket_stats"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
