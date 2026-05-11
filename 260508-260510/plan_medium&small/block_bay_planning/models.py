from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class BoxGroup:
    """同一类出口箱需求。

    中计划和小计划先把同航次、同真实尺寸、同高度、同卸港、同船公司、
    同状态、同特殊属性的箱子聚合成属性组。求解时再展开成单位箱 `Unit`。

    注意：属性组仍保留真实尺寸，例如真实 45ft 不会和真实 40ft 合成同一组；
    只是求解容量和大计划配额时，45ft 会通过 `size_mode` 消耗 40ft 容量。
    """

    group_id: str
    voyage_id: str
    size: str
    height: str
    status: str
    port: str
    operator: str
    ctype: str
    reefer: bool
    dangerous: bool
    over_limit: bool
    special_codes: tuple[str, ...]
    demand: int

    @property
    def size_mode(self) -> str:
        """中计划和小计划使用的尺寸口径。

        与大计划一致，求解时只区分 20ft 和 40ft。真实 45ft 箱在这里统一
        归入 40ft，因此容量、区块、贝位等约束都按 40ft 处理。
        """
        return "20" if self.size == "20" else "40"

    @property
    def big_plan_size_mode(self) -> str:
        """大计划尺寸口径，与中小计划求解尺寸口径保持一致。"""
        return self.size_mode

    @property
    def special_signature(self) -> str:
        marks = []
        if self.reefer:
            marks.append("RF")
        if self.dangerous:
            marks.append("DG")
        if self.over_limit:
            marks.append("OV")
        return "+".join(marks) if marks else "NORMAL"

    @property
    def business_special_signature(self) -> str:
        return "+".join(self.special_codes) if self.special_codes else "NORMAL"

    def requires_small_plan(self, threshold: int) -> bool:
        return self.demand >= threshold


@dataclass
class Bay:
    """贝位状态和容量快照。

    `cap_by_size` 只保存 20/40 两类容量。45ft 箱在中计划和小计划中消耗
    40ft 容量；同时 `physical_capacity` 仍限制同一贝位的总占用不超量。
    """

    area_no: str
    bay_no: str
    bay_key: str
    block_id: str
    block_bays: tuple[str, ...]
    block_bay_count: int
    block_boundary_adjusted: bool
    bay_order: int
    cap_by_size: dict[str, int] = field(default_factory=dict)
    physical_capacity: int = 0
    existing_size_modes: set[str] = field(default_factory=set)
    existing_heights: set[str] = field(default_factory=set)
    existing_special_signatures: set[str] = field(default_factory=set)
    existing_ports: set[str] = field(default_factory=set)
    fallback_reasons: set[str] = field(default_factory=set)
    locked: bool = False

    @property
    def is_fallback_bay(self) -> bool:
        return bool(self.fallback_reasons)

    def capacity_for(self, group: BoxGroup) -> int:
        return int(self.cap_by_size.get(group.size_mode, 0))


@dataclass(frozen=True)
class BigPlanRow:
    """大计划输出的一行。

    `size_mode` 使用大计划口径，只允许 `20`、`40` 或 `ALL`。其中 `40`
    同时包含真实 40ft 和真实 45ft。
    """

    voyage_id: str
    area_no: str
    planned_boxes: int
    size_mode: str = "ALL"


@dataclass(frozen=True)
class VoyageSchedule:
    voyage_id: str
    receive_start: datetime
    receive_end: datetime
    berth_no: str
    berth_time: datetime
    departure_time: datetime


@dataclass(frozen=True)
class AreaOperation:
    area_no: str
    voyage_id: str
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class Unit:
    unit_id: int
    group: BoxGroup


@dataclass
class ProblemData:
    """中计划和小计划共用的求解输入。

    `area_quota` 是大计划给出的航次-箱区总量硬约束；
    `area_size_quota` 是航次-箱区-尺寸硬约束，尺寸只包含 20/40。
    """

    groups: list[BoxGroup]
    units: list[Unit]
    bays: dict[str, Bay]
    big_plan: list[BigPlanRow]
    assigned_areas: dict[str, set[str]]
    area_quota: dict[tuple[str, str], int]
    area_size_quota: dict[tuple[str, str, str], int]
    small_plan_threshold: int
    business_special_codes: set[str]
    planning_time: datetime
    horizon_hours: float
    voyage_windows: dict[str, tuple[datetime, datetime]]
    area_operations: dict[str, list[AreaOperation]]
    target_voyages: list[str]


@dataclass
class SAConfig:
    seed: int = 7
    iterations: int = 30000
    initial_temperature: float = 80.0
    final_temperature: float = 0.2
    progress_every: int = 3000

    # 软约束权重。硬约束由 move feasibility 检查保证；这里的权重只影响偏好。
    group_area_split_penalty: float = 25.0
    group_block_split_penalty: float = 10.0
    group_bay_split_penalty: float = 2.0
    bay_port_mix_penalty: float = 3.0
    bay_operator_mix_penalty: float = 1.0
    bay_status_mix_penalty: float = 3.0
    bay_size_mix_penalty: float = 4.0
    bay_height_mix_penalty: float = 8.0
    fallback_bay_penalty: float = 20.0
    voyage_block_count_penalty: float = 6.0
    voyage_block_gap_penalty: float = 4.0
    special_block_split_penalty: float = 12.0
    bay_special_mix_penalty: float = 12.0
    block_utilization_penalty: float = 2.0
    active_loading_area_penalty: float = 16.0
    post_window_loading_area_reward: float = 5.0


@dataclass
class SolveResult:
    assignment: list[str]
    energy: float
    medium_rows: list[dict]
    small_rows: list[dict]
    diagnostics: dict
