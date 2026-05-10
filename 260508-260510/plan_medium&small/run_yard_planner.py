from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIG_PLAN_DIR = PROJECT_ROOT / "plan_big"
BIG_PLAN_MAIN = BIG_PLAN_DIR / "block_allocation_main.py"


def load_big_area_planner() -> ModuleType:
    """按文件路径加载大计划入口模块。

    `plan_medium&small` 不是标准 Python 包名，直接 `import block_allocation_main`
    时 IDE 静态分析常常找不到 `plan_big` 目录。这里用文件路径显式加载，运行时
    和 IDE 都不需要依赖额外的搜索路径配置。
    """
    spec = importlib.util.spec_from_file_location("big_area_planner", BIG_PLAN_MAIN)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load big area planner from {BIG_PLAN_MAIN}")
    module = importlib.util.module_from_spec(spec)
    if str(BIG_PLAN_DIR) not in sys.path:
        sys.path.insert(0, str(BIG_PLAN_DIR))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


big_area_planner = load_big_area_planner()

from block_bay_planning.data_loader import DEFAULT_PLANNING_TIME, build_problem, parse_datetime, read_big_plan
from block_bay_planning.models import BigPlanRow, SAConfig
from block_bay_planning.sa_solver import SimulatedAnnealingSolver, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve Guandong export medium/small yard planning with simulated annealing."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--big-plan",
        type=Path,
        help=(
            "Optional CSV from an existing big plan. Accepts either "
            "voyage_id,area_no,planned_boxes or voy_id,area_no,planned_qty."
        ),
    )
    parser.add_argument("--big-theta", default=DEFAULT_PLANNING_TIME.strftime("%Y-%m-%d %H:%M:%S"))
    parser.add_argument("--big-vessels", nargs="+", default=big_area_planner.DEFAULT_VESSELS)
    parser.add_argument("--big-time-limit", type=float, default=60.0)
    parser.add_argument("--big-mip-gap", type=float, default=0.01)
    parser.add_argument("--big-plan-history", type=Path, default=None)
    parser.add_argument("--big-quiet", action="store_true")
    parser.add_argument("--big-no-tops", action="store_true")
    parser.add_argument("--big-strict-demand", action="store_true")
    parser.add_argument("--write-big-plan-detail", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--planning-time",
        default=DEFAULT_PLANNING_TIME.strftime("%Y-%m-%d %H:%M:%S"),
        help="Planning timestamp. Default follows the test data snapshot: 2026-05-08 09:30:00.",
    )
    parser.add_argument("--horizon-hours", type=float, default=24.0, help="Planning horizon in hours.")
    parser.add_argument(
        "--small-plan-threshold",
        type=int,
        default=10,
        help="Attribute groups with planned boxes at or above this threshold are output to bay-level small plan.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sa_plan",
        help=(
            "Output root directory. Each run creates a new timestamped subfolder "
            "under this root, so previous results are never overwritten."
        ),
    )
    args = parser.parse_args()

    planning_time = parse_datetime(args.planning_time)
    if planning_time is None:
        raise SystemExit(f"Invalid --planning-time: {args.planning_time}")

    data_dir = discover_data_dir(args.data_dir)
    output_root = Path(args.output_dir)
    output_dir = create_run_output_dir(output_root)

    print_section("大计划")
    if args.big_plan:
        big_plan = read_big_plan(args.big_plan)
        print(f"读取已有大计划文件: {args.big_plan}")
    else:
        big_plan = run_big_area_plan(args, data_dir, output_dir)
    big_plan_path = output_dir / "big_plan.csv"
    write_big_plan_summary(big_plan_path, big_plan)
    print(f"大计划汇总输出: {big_plan_path}")

    print_section("中计划与小计划")
    problem = build_problem(
        data_dir,
        big_plan,
        planning_time=planning_time,
        horizon_hours=args.horizon_hours,
        small_plan_threshold=args.small_plan_threshold,
    )
    config = SAConfig(iterations=args.iterations, seed=args.seed)
    solver = SimulatedAnnealingSolver(problem, config)
    result = solver.solve()

    medium_plan_path = output_dir / "medium_plan.csv"
    small_plan_path = output_dir / "small_plan.csv"
    diagnostics_path = output_dir / "diagnostics.json"
    write_rows(medium_plan_path, result.medium_rows)
    write_rows(small_plan_path, result.small_rows)
    diagnostics_path.write_text(
        json.dumps(result.diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_section("中计划与小计划诊断")
    print(json.dumps(result.diagnostics, ensure_ascii=False, indent=2))
    print_section("输出文件")
    print(f"output_dir: {output_dir}")
    print(f"big_plan: {big_plan_path}")
    if not args.big_plan:
        print(f"big_plan_detail: {resolve_big_plan_detail_path(args, output_dir)}")
    print(f"medium_plan: {medium_plan_path}")
    print(f"small_plan: {small_plan_path}")
    print(f"diagnostics: {diagnostics_path}")


def discover_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        candidate = data_dir
        if not candidate.is_absolute():
            for root in (Path.cwd(), Path(__file__).resolve().parent, PROJECT_ROOT):
                resolved = root / candidate
                if resolved.exists():
                    return resolved.resolve()
        return candidate.resolve()

    for root in (Path.cwd(), Path(__file__).resolve().parent, PROJECT_ROOT):
        for candidate in root.iterdir():
            if candidate.is_dir() and "20260508" in candidate.name:
                return candidate.resolve()
    raise FileNotFoundError("No data directory containing 20260508 was found")


def create_run_output_dir(output_root: str | Path) -> Path:
    """为本次运行创建独立输出目录，避免覆盖或占用旧结果文件。"""
    root = Path(output_root)
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{timestamp}_{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_big_area_plan(args: argparse.Namespace, data_dir: Path, output_dir: Path) -> list[BigPlanRow]:
    theta = big_area_planner.pd.Timestamp(args.big_theta)
    plan_history = args.big_plan_history or BIG_PLAN_DIR / big_area_planner.DEFAULT_PLAN_HISTORY
    print(f"大计划基准时刻 theta: {theta}")
    print(f"大计划航次: {args.big_vessels}")
    artifacts = big_area_planner.build_area_allocation_data(
        theta=theta,
        vessel_ids=args.big_vessels,
        base_dir=BIG_PLAN_DIR,
        data_dir=data_dir,
        plan_history_path=plan_history,
        include_tops=not args.big_no_tops,
        allow_unmet_demand=not args.big_strict_demand,
    )
    result = big_area_planner.solve_yard_area_allocation(
        artifacts.data,
        time_limit=args.big_time_limit,
        mip_gap=args.big_mip_gap,
        verbose=not args.big_quiet,
    )
    big_area_planner.print_case_summary(artifacts, result)
    detail_rows = big_area_planner.build_plan_rows(result, artifacts)
    detail_path = resolve_big_plan_detail_path(args, output_dir)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_rows.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"大计划明细输出: {detail_path}")
    return summarize_big_plan_rows(detail_rows)


def resolve_big_plan_detail_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.write_big_plan_detail:
        return args.write_big_plan_detail
    return output_dir / "big_plan_detail.csv"


def write_big_plan_summary(path: str | Path, rows: list[BigPlanRow]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["voyage_id", "area_no", "size_mode", "planned_boxes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "voyage_id": row.voyage_id,
                    "area_no": row.area_no,
                    "size_mode": row.size_mode,
                    "planned_boxes": row.planned_boxes,
                }
            )


def print_section(title: str) -> None:
    print("")
    print("=" * 72)
    print(title)
    print("=" * 72)


def summarize_big_plan_rows(detail_rows) -> list[BigPlanRow]:
    """把大计划明细汇总成中小计划输入。

    大计划求解结果按 `size=20/40` 输出，其中 45 尺已经计入 40 尺。这里保留
    这个尺寸字段，后续中计划会按航次-箱区-尺寸做硬配额校验。
    """
    if detail_rows.empty:
        raise ValueError("big area plan produced no positive allocation rows")
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in detail_rows.to_dict("records"):
        size_mode = "20" if str(row.get("size", "")).strip() == "20" else "40"
        counter[(str(row["voy_id"]), str(row["area_no"]), size_mode)] += int(row["planned_qty"])
    return [
        BigPlanRow(voyage_id, area_no, planned_boxes, size_mode)
        for (voyage_id, area_no, size_mode), planned_boxes in sorted(counter.items())
        if planned_boxes > 0
    ]


if __name__ == "__main__":
    main()
