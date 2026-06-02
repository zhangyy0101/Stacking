from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from planning_large_main import (
    DEFAULT_EXPORT_VESSELS,
    DEFAULT_FLOW_ALIASES,
    DEFAULT_IMPORT_VESSELS,
    DEFAULT_PLANNING_TIME,
    RollingPlanningState,
    build_current_case_data,
    print_case_summary,
    print_solution_summary,
    read_adjust_plan_info,
    resolve_output_path,
    write_run_outputs,
)
from planning_large_solver import (
    DailyRollingYardPlanningData,
    DailyRollingYardPlanningSolution,
    _extract_float_tupledict,
    _extract_int_tupledict,
    _safe_model_attr,
    _status_name,
    build_daily_rolling_yard_model_gurobi,
)


LEX_OBJECTIVES: tuple[tuple[str, int], ...] = (
    ("miss", 70),
    ("required_area", 65),
    ("operation", 60),
    ("of_area", 50),
    ("distance", 40),
    ("share", 30),
    ("berth_conflict", 25),
    ("adjustment", 20),
    ("balance", 10),
)


def solve_daily_rolling_yard_plan_lex(
    data: DailyRollingYardPlanningData,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
    gurobi_params: Optional[Mapping[str, Any]] = None,
    keep_model: bool = False,
) -> DailyRollingYardPlanningSolution:
    """
    Solve the large yard planning model with Gurobi lexicographic multi-objectives.
    """

    model, variables, params = build_daily_rolling_yard_model_gurobi(
        data,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
        gurobi_params=gurobi_params,
    )
    components = variables["objective_components"]

    # Replace the weighted-sum objective with strict priority levels.
    for index, (name, priority) in enumerate(LEX_OBJECTIVES):
        model.setObjectiveN(
            components[name],
            index=index,
            priority=priority,
            weight=1.0,
            abstol=0.0,
            reltol=0.0,
            name=name,
        )

    model.optimize()

    if model.SolCount <= 0:
        return DailyRollingYardPlanningSolution(
            status=model.Status,
            status_name=_status_name(model.Status),
            objective_value=None,
            best_bound=_safe_model_attr(model, "ObjBound"),
            mip_gap=_safe_model_attr(model, "MIPGap"),
            runtime=_safe_model_attr(model, "Runtime"),
            x20={},
            x40={},
            y={},
            h={},
            o={},
            of_area_used={},
            of_area_over={},
            s20={},
            s40={},
            m20={},
            m40={},
            r20_pos={},
            r20_neg={},
            r40_pos={},
            r40_neg={},
            berth_conflict_shared={},
            required_area_unmet={},
            objective_components={},
            model=model if keep_model else None,
        )

    V = params["V"]
    F = params["F"]
    A = params["A"]
    V_old = params["V_old"]
    OF_area_vessels = params["OF_area_vessels"]
    berth_conflict_keys = params["berth_conflict_keys"]
    required_area_keys = params["required_area_keys"]

    return DailyRollingYardPlanningSolution(
        status=model.Status,
        status_name=_status_name(model.Status),
        objective_value=model.ObjVal,
        best_bound=_safe_model_attr(model, "ObjBound"),
        mip_gap=_safe_model_attr(model, "MIPGap"),
        runtime=_safe_model_attr(model, "Runtime"),
        x20=_extract_int_tupledict(variables["X20"], (V, F, A)),
        x40=_extract_int_tupledict(variables["X40"], (V, F, A)),
        y=_extract_int_tupledict(variables["y"], (V, A)),
        h=_extract_int_tupledict(variables["h"], (A,)),
        o=_extract_float_tupledict(variables["o"], (A,)),
        of_area_used=_extract_int_tupledict(variables["of_area_used"], (OF_area_vessels, A)),
        of_area_over=_extract_float_tupledict(variables["of_area_over"], (OF_area_vessels,)),
        s20=_extract_float_tupledict(variables["s20"], (V, F)),
        s40=_extract_float_tupledict(variables["s40"], (V, F)),
        m20=_extract_float_tupledict(variables["m20"], (V, F)),
        m40=_extract_float_tupledict(variables["m40"], (V, F)),
        r20_pos=_extract_float_tupledict(variables["r20_pos"], (V_old, F, A)),
        r20_neg=_extract_float_tupledict(variables["r20_neg"], (V_old, F, A)),
        r40_pos=_extract_float_tupledict(variables["r40_pos"], (V_old, F, A)),
        r40_neg=_extract_float_tupledict(variables["r40_neg"], (V_old, F, A)),
        berth_conflict_shared=_extract_int_tupledict(
            variables["berth_conflict_shared"],
            (berth_conflict_keys,),
        ),
        required_area_unmet=_extract_float_tupledict(
            variables["required_area_unmet"],
            (required_area_keys,),
        ),
        objective_components={name: expr.getValue() for name, expr in components.items()},
        model=model if keep_model else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the large yard planning model with lexicographic objectives.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--planning-time", default=DEFAULT_PLANNING_TIME)
    parser.add_argument("--export-vessels", nargs="+", default=DEFAULT_EXPORT_VESSELS)
    parser.add_argument("--import-vessels", nargs="+", default=DEFAULT_IMPORT_VESSELS)
    parser.add_argument("--state-dir", type=Path, default=Path("outputs_large_lex/state"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_large_lex/latest_run"))
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-write-state", action="store_true")
    parser.add_argument("--disable-default-flow-aliases", action="store_true")
    parser.add_argument(
        "--user-design-large-plan-area",
        nargs="+",
        default=None,
        help="Restrict new large-plan allocation to these yard areas for every planned vessel.",
    )
    parser.add_argument(
        "--adjust-plan-info-json",
        type=Path,
        default=None,
        help="JSON file containing adjust_plan_info; large_plan[voy_id].add/remove controls large-plan areas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    planning_time = pd.Timestamp(args.planning_time)
    state = RollingPlanningState(resolve_output_path(args.state_dir, base_dir))
    flow_aliases = {} if args.disable_default_flow_aliases else DEFAULT_FLOW_ALIASES
    adjust_plan_info = read_adjust_plan_info(args.adjust_plan_info_json)

    artifacts = build_current_case_data(
        base_dir=base_dir,
        data_dir=args.data_dir,
        planning_time=planning_time,
        export_vessels=args.export_vessels,
        import_vessels=args.import_vessels,
        state=state,
        flow_aliases=flow_aliases,
        user_design=bool(args.user_design_large_plan_area),
        user_design_large_plan_area=args.user_design_large_plan_area,
        adjust_plan_info=adjust_plan_info,
    )
    print_case_summary(artifacts)
    print("lex_objectives:", list(LEX_OBJECTIVES))

    solution = solve_daily_rolling_yard_plan_lex(
        artifacts.data,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        verbose=not args.quiet,
    )
    print_solution_summary(solution)

    state_rows = pd.DataFrame()
    if not args.no_write_state and solution.objective_value is not None:
        state_rows = state.append_solution(planning_time, solution)
        print(f"State rows appended: {len(state_rows)} -> {state.plan_history_path}")

    output_dir = resolve_output_path(args.output_dir, base_dir)
    write_run_outputs(output_dir, artifacts, solution, state_rows)
    diagnostics_path = output_dir / "diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["objective_mode"] = "lexicographic"
        diagnostics["lex_objectives"] = [
            {"name": name, "priority": priority, "weight": 1.0}
            for name, priority in LEX_OBJECTIVES
        ]
        diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Run outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
