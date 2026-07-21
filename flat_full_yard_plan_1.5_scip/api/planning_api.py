from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapters import input_adapter_standard as adapter_data_io
from adapters.input_adapter_gd import InputAdapterGd
from adapters.input_adapter_standard import DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO, LargePlanningConfig
from large_plan.solver import solve_daily_rolling_yard_plan
from medium_small.bridge import (
    big_plan_rows_from_allocation_records,
    infer_target_voyages_from_big_plan,
    make_config as make_medium_small_config,
)
from medium_small.column_generation_planner import ColumnGenerationConfig, ColumnGenerationPlanner
from medium_small.corrected_large_plan import build_corrected_large_plan_rows
from medium_small.small_plan_from_medium import (
    apply_external_medium_plan,
    configure_small_plan_from_medium,
    external_medium_plan_from_records,
    filter_external_medium_plan,
)


@dataclass
class YardPlanAlgorithmConfig:
    large: LargePlanningConfig = field(default_factory=LargePlanningConfig)
    medium_small: ColumnGenerationConfig = field(default_factory=ColumnGenerationConfig)
    planning_time: datetime | pd.Timestamp | str | None = None
    medium_voyages: Sequence[str] | None = None
    horizon_hours: float = 24.0
    misplaced_bay_exclusion_ratio: float = DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO
    large_time_limit: float = 120.0
    large_mip_gap: float = 0.001
    large_verbose: bool = True
    disable_default_flow_aliases: bool = False

    @classmethod
    def from_cli_args(cls, args: Any) -> "YardPlanAlgorithmConfig":
        horizon_hours = getattr(args, "horizon_hours", 24.0)
        misplaced_ratio = getattr(
            args,
            "misplaced_bay_exclusion_ratio",
            DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO,
        )
        large_time_limit = getattr(args, "large_time_limit", 120.0)
        large_mip_gap = getattr(args, "large_mip_gap", 0.001)
        return cls(
            large=LargePlanningConfig(),
            medium_small=make_medium_small_config(args) if hasattr(args, "max_iterations") else ColumnGenerationConfig(),
            planning_time=getattr(args, "planning_time", None),
            medium_voyages=getattr(args, "medium_voyages", None),
            horizon_hours=float(24.0 if horizon_hours is None else horizon_hours),
            misplaced_bay_exclusion_ratio=float(
                DEFAULT_MISPLACED_BAY_EXCLUSION_RATIO if misplaced_ratio is None else misplaced_ratio
            ),
            large_time_limit=float(120.0 if large_time_limit is None else large_time_limit),
            large_mip_gap=float(0.001 if large_mip_gap is None else large_mip_gap),
            large_verbose=not bool(getattr(args, "large_quiet", False)),
            disable_default_flow_aliases=bool(getattr(args, "disable_default_flow_aliases", False)),
        )


def solve_large_plan_df(
    input_adapter: InputAdapterGd,
    config: YardPlanAlgorithmConfig | None = None,
) -> pd.DataFrame:
    config = config or YardPlanAlgorithmConfig()
    planning_time = _planning_timestamp(input_adapter, config)
    artifacts, _state = adapter_data_io.build_large_inputs(
        input_adapter,
        planning_time,
        disable_default_flow_aliases=config.disable_default_flow_aliases,
        config=config.large,
    )
    solution = solve_daily_rolling_yard_plan(
        artifacts.data,
        time_limit=config.large_time_limit,
        mip_gap=config.large_mip_gap,
        verbose=config.large_verbose,
    )
    return pd.DataFrame(
        adapter_data_io.allocation_output_rows(
            solution,
            artifacts.data,
            planning_time=planning_time,
        )
    )


def solve_medium_small_plan_df(
    input_adapter: InputAdapterGd,
    latest_large_plan_df: pd.DataFrame,
    config: YardPlanAlgorithmConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or YardPlanAlgorithmConfig()
    planning_time = _planning_datetime(input_adapter, config)
    target_voyages = _target_voyages_from_large_plan(latest_large_plan_df, planning_time, config)
    inputs = adapter_data_io.load_medium_small_inputs(
        input_adapter,
        planning_time=planning_time,
        voyages=target_voyages,
        horizon_hours=config.horizon_hours,
        misplaced_bay_exclusion_ratio=config.misplaced_bay_exclusion_ratio,
        big_plan=latest_large_plan_df,
    )
    result = ColumnGenerationPlanner(inputs.problem, replace(config.medium_small)).solve()
    corrected_large_rows, corrected_large_diagnostics = build_corrected_large_plan_rows(
        latest_large_plan_df.to_dict("records"),
        result.medium_rows,
    )
    return {
        "medium_plan": pd.DataFrame(result.medium_rows),
        "small_plan": pd.DataFrame(result.small_rows),
        "unplaced_boxes": pd.DataFrame(result.unplaced_rows),
        "corrected_large_plan": pd.DataFrame(corrected_large_rows),
        "corrected_large_plan_diagnostics": pd.DataFrame([corrected_large_diagnostics]),
    }


def solve_small_plan_df(
    input_adapter: InputAdapterGd,
    latest_large_plan_df: pd.DataFrame,
    latest_medium_plan_df: pd.DataFrame,
    config: YardPlanAlgorithmConfig | None = None,
) -> pd.DataFrame:
    config = config or YardPlanAlgorithmConfig()
    planning_time = _planning_datetime(input_adapter, config)
    external_plan = external_medium_plan_from_records(
        latest_medium_plan_df.to_dict("records"),
        list(latest_medium_plan_df.columns),
    )
    target_voyages = [_normalize_voyage(voyage) for voyage in config.medium_voyages] if config.medium_voyages else list(external_plan.target_voyages)
    inputs = adapter_data_io.load_medium_small_inputs(
        input_adapter,
        planning_time=planning_time,
        voyages=target_voyages,
        horizon_hours=config.horizon_hours,
        misplaced_bay_exclusion_ratio=config.misplaced_bay_exclusion_ratio,
        big_plan=latest_large_plan_df,
    )
    external_plan = filter_external_medium_plan(external_plan, inputs.problem.target_voyages)
    problem = apply_external_medium_plan(inputs.problem, external_plan)
    medium_small_config = configure_small_plan_from_medium(replace(config.medium_small), external_plan)
    result = ColumnGenerationPlanner(problem, medium_small_config).solve()
    return pd.DataFrame(result.small_rows)


def solve_full_yard_plan_df(
    input_adapter: InputAdapterGd,
    config: YardPlanAlgorithmConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or YardPlanAlgorithmConfig()
    large_result_df = solve_large_plan_df(input_adapter, config)
    medium_small = solve_medium_small_plan_df(input_adapter, large_result_df, config)
    return {
        "large_plan": large_result_df,
        "medium_plan": medium_small["medium_plan"],
        "small_plan": medium_small["small_plan"],
        "unplaced_boxes": medium_small["unplaced_boxes"],
        "corrected_large_plan": medium_small["corrected_large_plan"],
        "corrected_large_plan_diagnostics": medium_small["corrected_large_plan_diagnostics"],
    }


def _planning_timestamp(input_adapter: InputAdapterGd, config: YardPlanAlgorithmConfig) -> pd.Timestamp:
    value = config.planning_time if config.planning_time is not None else getattr(input_adapter, "planning_time", None)
    planning_time = pd.Timestamp(value)
    if pd.isna(planning_time):
        raise ValueError(f"Invalid planning_time: {value}")
    return planning_time


def _planning_datetime(input_adapter: InputAdapterGd, config: YardPlanAlgorithmConfig) -> datetime:
    return _planning_timestamp(input_adapter, config).to_pydatetime()


def _target_voyages_from_large_plan(
    latest_large_plan_df: pd.DataFrame,
    planning_time: datetime,
    config: YardPlanAlgorithmConfig,
) -> list[str]:
    if config.medium_voyages:
        return [_normalize_voyage(voyage) for voyage in config.medium_voyages]
    big_plan = big_plan_rows_from_allocation_records(latest_large_plan_df.to_dict("records"))
    return infer_target_voyages_from_big_plan(big_plan, planning_time)


def _normalize_voyage(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


__all__ = [
    "YardPlanAlgorithmConfig",
    "solve_large_plan_df",
    "solve_medium_small_plan_df",
    "solve_small_plan_df",
    "solve_full_yard_plan_df",
]
