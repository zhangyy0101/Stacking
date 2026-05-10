"""
事件驱动滚动箱区分配模型：Gurobi 求解框架

说明：
1. 本代码只实现单个规划时刻 theta 下的 MILP 求解模型。
2. 时间滚动、读取实时堆场、更新容量、更新历史箱区等逻辑暂时不放入本文件。
3. 模型区分第一次规划与后续规划：
   - 第一次规划：需要控制航次主箱区数量 K_min <= sum_a w[v,a] <= K_max。
   - 后续规划：尽量沿用历史箱区 H[v,a]，若启用新箱区则通过 n[v,a] 计入惩罚。
4. 模型不考虑箱属性组，但区分 20 尺和 40 尺箱，因为二者对应不同物理容量约束。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB


# =========================
# 1. 类型别名
# =========================

Vessel = str          # 航次编号，例如 "453334"
Area = str            # 箱区编号，例如 "12"、"4F"
VesselPair = Tuple[Vessel, Vessel]


# =========================
# 2. 参数容器
# =========================

@dataclass
class YardAllocationData:
    """
    单个规划时刻 theta 下的模型输入参数。
    注意：这些参数应由外部数据预处理和滚动更新模块提前计算好。
    """

    # 当前规划时刻需要新增规划的航次集合 V+(theta)
    V_plus: List[Vessel]

    # 当前规划时刻仍然占用堆场规划资源的活跃航次集合 V_act(theta)
    # 该集合用于统计每个箱区当前服务了几个航次。
    V_act: List[Vessel]

    # 候选箱区集合 A
    A: List[Area]

    # 同日开港航次对集合 P_same(theta)
    # 例如 [("453334", "453335"), ("453334", "453336")]
    P_same: List[VesselPair]

    # 当前新增规划需求：20 尺箱 R_v^20(theta)
    R20: Dict[Vessel, int]

    # 当前新增规划需求：40 尺箱 R_v^40(theta)
    R40: Dict[Vessel, int]

    # 当前箱区 20 尺等价剩余容量 C_a^20(theta)
    C20: Dict[Area, int]

    # 当前箱区 40 尺剩余容量 C_a^40(theta)
    C40: Dict[Area, int]

    # 航次 v 到箱区 a 的距离 d_{v,a}
    distance: Dict[Tuple[Vessel, Area], float]

    # 箱区不可用参数 B_{v,a}(theta)
    # B[v,a] = 1 表示箱区 a 对航次 v 当前不可用；B[v,a] = 0 表示可用。
    B: Dict[Tuple[Vessel, Area], int]

    # 历史已选箱区参数 H_{v,a}(theta^-)
    # H[v,a] = 1 表示航次 v 在当前时刻之前已经使用过箱区 a。
    H: Dict[Tuple[Vessel, Area], int]

    # 当前时刻之前已规划 20 尺箱量 P_v^20(theta^-)
    P20_prev: Dict[Vessel, int]

    # 当前时刻之前已规划 40 尺箱量 P_v^40(theta^-)
    P40_prev: Dict[Vessel, int]

    # 当前时刻之前累计距离成本 G_v(theta^-)
    G_prev: Dict[Vessel, float]

    # 第一次规划航次集合：这些航次需要施加箱区数量上下界约束。
    first_plan_vessels: List[Vessel] = field(default_factory=list)

    # 后续规划航次集合：这些航次需要施加“尽量不新开箱区”的逻辑与惩罚。
    followup_plan_vessels: List[Vessel] = field(default_factory=list)

    # 第一次规划时，每个航次推荐箱区数量下界 K_v^min
    K_min: Dict[Vessel, int] = field(default_factory=dict)

    # 第一次规划时，每个航次推荐箱区数量上界 K_v^max
    K_max: Dict[Vessel, int] = field(default_factory=dict)

    # 目标函数权重：总距离成本
    lambda_dist: float = 1.0

    # 目标函数权重：最大累计平均距离公平项 eta
    lambda_fair: float = 1.0

    # 目标函数权重：当前新增分配碎片化项 sum z[v,a]
    lambda_frag: float = 1.0

    # 目标函数权重：同日开港航次共用箱区惩罚项 sum q[u,v,a]
    lambda_same: float = 10.0

    # 目标函数权重：箱区服务超过 2 个航次的惩罚项 sum h[a]
    lambda_over: float = 10.0

    # 目标函数权重：后续规划中新开箱区惩罚项 sum n[v,a]
    M_new: float = 1000.0

    # 目标函数权重：未满足需求惩罚项 sum s20[v] + s40[v]
    M_miss: float = 1_000_000.0

    # 是否允许未满足需求。实际测试时建议先设为 True，避免模型直接 infeasible。
    allow_unmet_demand: bool = True


# =========================
# 3. 数据完整性检查
# =========================

def validate_input(data: YardAllocationData) -> None:
    """
    对模型输入进行基础检查。
    这里不做复杂业务校验，只检查关键索引是否存在。
    """

    V_plus = set(data.V_plus)   # 当前规划航次
    V_act = set(data.V_act)     # 活跃航次列表（正在新增规划、需要补充规划、箱子仍未全部离开堆场）
    A = set(data.A)             # 箱区列表

    # 当前新增规划航次应属于活跃航次集合。
    missing_active = V_plus - V_act
    if missing_active:
        raise ValueError(f"V_plus 中存在不属于 V_act 的航次: {missing_active}")

    # 检查需求参数。
    for v in data.V_plus:
        if v not in data.R20:
            raise KeyError(f"缺少航次 {v} 的 R20 参数")
        if v not in data.R40:
            raise KeyError(f"缺少航次 {v} 的 R40 参数")

    # 检查容量参数。
    for a in data.A:
        if a not in data.C20:
            raise KeyError(f"缺少箱区 {a} 的 C20 参数")
        if a not in data.C40:
            raise KeyError(f"缺少箱区 {a} 的 C40 参数")

    # 检查距离、不可用参数、历史箱区参数。
    for v in data.V_act:
        for a in data.A:
            if (v, a) not in data.distance:
                raise KeyError(f"缺少距离参数 distance[{v}, {a}]")
            if (v, a) not in data.B:
                raise KeyError(f"缺少不可用参数 B[{v}, {a}]")
            if (v, a) not in data.H:
                raise KeyError(f"缺少历史箱区参数 H[{v}, {a}]")

    # 检查历史累计参数。
    for v in data.V_act:
        if v not in data.P20_prev:
            raise KeyError(f"缺少航次 {v} 的 P20_prev 参数")
        if v not in data.P40_prev:
            raise KeyError(f"缺少航次 {v} 的 P40_prev 参数")
        if v not in data.G_prev:
            raise KeyError(f"缺少航次 {v} 的 G_prev 参数")

    # 第一次规划航次需要有 K_min 和 K_max。
    for v in data.first_plan_vessels:
        if v not in data.K_min:
            raise KeyError(f"第一次规划航次 {v} 缺少 K_min 参数")
        if v not in data.K_max:
            raise KeyError(f"第一次规划航次 {v} 缺少 K_max 参数")
        if data.K_min[v] > data.K_max[v]:
            raise ValueError(f"航次 {v} 的 K_min 大于 K_max")

    # 同日开港航次对需要属于活跃航次集合。
    for u, v in data.P_same:
        if u not in V_act or v not in V_act:
            raise ValueError(f"同日开港航次对 {(u, v)} 中存在非活跃航次")


# =========================
# 4. Gurobi 模型构建与求解
# =========================

def solve_yard_area_allocation(
    data: YardAllocationData,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    构建并求解单个规划时刻 theta 下的箱区分配模型。

    返回值包含：
    - status: Gurobi 求解状态
    - objective_value: 目标函数值
    - x20, x40: 20/40 尺箱分配结果
    - z: 当前规划是否使用箱区
    - w: 当前规划完成后航次总体是否占用箱区
    - n: 后续规划中是否新开箱区
    - q: 同日开港航次是否共用箱区
    - h: 箱区服务航次数超额量
    - eta: 最大累计平均距离
    - s20, s40: 未满足需求量
    """

    validate_input(data)

    V_plus = data.V_plus    # 新增规划航次
    V_act = data.V_act      # 活跃航次
    A = data.A              # 箱区
    P_same = data.P_same    # 同日开港的航次对

    # 判断哪些航次当前分别属于哪个阶段
    first_set = set(data.first_plan_vessels)        # 首次规划的航次集合
    followup_set = set(data.followup_plan_vessels)  # 后续规划的航次集合

    # 创建 Gurobi 模型。
    model = gp.Model("event_driven_yard_area_allocation")

    # Gurobi参数设置
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # =========================
    # 4.1 决策变量定义
    # =========================

    # x20[v,a]：当前规划时刻，航次 v 新增分配到箱区 a 的 20 尺箱数量。
    x20 = model.addVars(
        V_plus,
        A,
        vtype=GRB.INTEGER,
        lb=0,
        name="x20",
    )

    # x40[v,a]：当前规划时刻，航次 v 新增分配到箱区 a 的 40 尺箱数量。
    x40 = model.addVars(
        V_plus,
        A,
        vtype=GRB.INTEGER,
        lb=0,
        name="x40",
    )

    # z[v,a]：当前这一次规划中，航次 v 是否往箱区 a 分配了新增箱量。
    z = model.addVars(
        V_plus,
        A,
        vtype=GRB.BINARY,
        name="z_current_use",
    )

    # w[v,a]：当前规划完成后，航次 v 的整体计划中是否使用箱区 a。
    # 对活跃航次都定义 w，因为它用于统计一个箱区当前服务了几个航次。
    w = model.addVars(
        V_act,
        A,
        vtype=GRB.BINARY,
        name="w_total_use",
    )

    # n[v,a]：后续规划中，航次 v 是否新开箱区 a。
    # 对当前新增规划航次定义。第一次规划时可以固定为 0，后续规划时才真正起作用。
    n = model.addVars(
        V_plus,
        A,
        vtype=GRB.BINARY,
        name="n_new_area",
    )

    # q[u,v,a]：同日开港航次 u 和 v 是否共用箱区 a。
    q = model.addVars(
        P_same,
        A,
        vtype=GRB.BINARY,
        name="q_same_day_share",
    )

    # h[a]：箱区 a 服务超过 2 个航次的超额数量。
    h = model.addVars(
        A,
        vtype=GRB.INTEGER,
        lb=0.0,
        name="h_over_two_vessels",
    )

    # eta：所有活跃航次中的最大累计平均距离。
    eta = model.addVar(
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="eta_max_average_distance",
    )

    # s20[v]、s40[v]：未满足的 20 尺、40 尺新增规划需求。
    # 若不允许未满足需求，则后续通过约束固定为 0。
    s20 = model.addVars(
        V_plus,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="s20_unmet",
    )
    s40 = model.addVars(
        V_plus,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="s40_unmet",
    )

    # =========================
    # 4.2 需求满足约束
    # =========================

    # 约束 1：20 尺箱新增规划需求满足。
    # 对每个当前需要规划的航次 v，分配到所有箱区的 20 尺箱量加上未满足量，等于 R20[v]。
    for v in V_plus:
        model.addConstr(
            gp.quicksum(x20[v, a] for a in A) + s20[v] == data.R20[v],
            name=f"demand_20[{v}]",
        )

    # 约束 2：40 尺箱新增规划需求满足。
    # 对每个当前需要规划的航次 v，分配到所有箱区的 40 尺箱量加上未满足量，等于 R40[v]。
    for v in V_plus:
        model.addConstr(
            gp.quicksum(x40[v, a] for a in A) + s40[v] == data.R40[v],
            name=f"demand_40[{v}]",
        )

    # 若不允许未满足需求，则将松弛变量固定为 0。
    if not data.allow_unmet_demand:
        for v in V_plus:
            model.addConstr(s20[v] == 0, name=f"no_unmet_20[{v}]")
            model.addConstr(s40[v] == 0, name=f"no_unmet_40[{v}]")

    # =========================
    # 4.3 箱区容量约束
    # =========================

    # 约束 3：40 尺箱容量约束。
    # 当前新增规划分配到箱区 a 的 40 尺箱总数，不能超过 C40[a]。
    for a in A:
        model.addConstr(
            gp.quicksum(x40[v, a] for v in V_plus) <= data.C40[a],
            name=f"capacity_40[{a}]",
        )

    # 约束 4：20 尺等价容量约束。
    # 20 尺箱占用 1 个 20 尺等价位置；40 尺箱占用 2 个 20 尺等价位置。
    # 因此 x20 + 2*x40 不能超过 C20[a]。
    for a in A:
        model.addConstr(
            gp.quicksum(x20[v, a] for v in V_plus)
            + 2 * gp.quicksum(x40[v, a] for v in V_plus)
            <= data.C20[a],
            name=f"capacity_20_equivalent[{a}]",
        )

    # =========================
    # 4.4 当前分配变量与当前使用变量 z 的逻辑联动
    # =========================

    # 约束 5：如果 z[v,a] = 0，则航次 v 当前不能往箱区 a 分配任何箱子。
    # M_v 取当前航次 v 的新增规划总量 R20[v] + R40[v]。
    for v in V_plus:
        M_v = data.R20[v] + data.R40[v]
        for a in A:
            model.addConstr(
                x20[v, a] + x40[v, a] <= M_v * z[v, a],
                name=f"link_upper_x_z[{v},{a}]",
            )

    # 约束 6：如果 z[v,a] = 1，则航次 v 当前至少要往箱区 a 分配 1 个箱子。
    # 这只是基本逻辑约束，不是单箱区最小分配量规则。
    for v in V_plus:
        for a in A:
            model.addConstr(
                x20[v, a] + x40[v, a] >= z[v, a],
                name=f"link_lower_x_z[{v},{a}]",
            )

    # =========================
    # 4.5 不可用箱区约束
    # =========================

    # 约束 7：如果 B[v,a] = 1，说明箱区 a 对航次 v 当前不可用，则 z[v,a] 必须为 0。
    # 不可用可能来自关闭箱区、非 OF 适放、开港当天装船冲突、TOPS 计划占用等。
    for v in V_plus:
        for a in A:
            model.addConstr(
                z[v, a] <= 1 - data.B[v, a],
                name=f"unavailable_area[{v},{a}]",
            )

    # =========================
    # 4.6 总体使用变量 w 与历史 H、当前 z 的关系
    # =========================

    # 对当前新增规划航次，w[v,a] = max(H[v,a], z[v,a])。
    # 约束 8.1：如果历史已经选过箱区 a，则当前规划完成后 w[v,a] 必须为 1。
    for v in V_plus:
        for a in A:
            model.addConstr(
                w[v, a] >= data.H[v, a],
                name=f"w_ge_history[{v},{a}]",
            )

    # 约束 8.2：如果当前新增规划使用箱区 a，则当前规划完成后 w[v,a] 必须为 1。
    for v in V_plus:
        for a in A:
            model.addConstr(
                w[v, a] >= z[v, a],
                name=f"w_ge_current[{v},{a}]",
            )

    # 约束 8.3：如果历史没用、当前也没用，则 w[v,a] 必须为 0。
    for v in V_plus:
        for a in A:
            model.addConstr(
                w[v, a] <= data.H[v, a] + z[v, a],
                name=f"w_le_history_plus_current[{v},{a}]",
            )

    # 对非当前新增规划但仍然活跃的航次，不在本次模型中新增分配箱量。
    # 因此其总体使用情况 w[v,a] 直接固定为历史值 H[v,a]。
    for v in V_act:
        if v not in V_plus:
            for a in A:
                model.addConstr(
                    w[v, a] == data.H[v, a],
                    name=f"inactive_w_fixed_to_history[{v},{a}]",
                )

    # =========================
    # 4.7 第一次规划专用约束：主箱区数量控制
    # =========================

    # 约束 9：第一次规划时，航次 v 使用的主箱区数量应在 K_min[v] 和 K_max[v] 之间。
    # 这里控制的是当前规划完成后的总体箱区集合 w[v,a]，不是仅当前新增分配的 z[v,a]。
    for v in first_set:
        if v not in V_plus:
            continue
        model.addConstr(
            gp.quicksum(w[v, a] for a in A) >= data.K_min[v],
            name=f"first_plan_area_count_min[{v}]",
        )
        model.addConstr(
            gp.quicksum(w[v, a] for a in A) <= data.K_max[v],
            name=f"first_plan_area_count_max[{v}]",
        )

    # =========================
    # 4.8 后续规划专用约束：新开箱区识别
    # =========================

    # 对第一次规划航次，将 n[v,a] 固定为 0。
    # 第一次规划本身就是在建立主箱区集合，不应计为“新开箱区惩罚”。
    for v in V_plus:
        if v in first_set:
            for a in A:
                model.addConstr(
                    n[v, a] == 0,
                    name=f"first_plan_no_new_penalty[{v},{a}]",
                )

    # 对后续规划航次，n[v,a] = 1 表示箱区 a 此前未被航次 v 使用，但当前规划完成后被启用。
    # 约束 10.1：当 w=1 且 H=0 时，强制 n=1。
    for v in followup_set:
        if v not in V_plus:
            continue
        for a in A:
            model.addConstr(
                n[v, a] >= w[v, a] - data.H[v, a],
                name=f"new_area_lb[{v},{a}]",
            )

    # 约束 10.2：n[v,a] 不能大于 w[v,a]。
    for v in followup_set:
        if v not in V_plus:
            continue
        for a in A:
            model.addConstr(
                n[v, a] <= w[v, a],
                name=f"new_area_le_w[{v},{a}]",
            )

    # 约束 10.3：如果 H[v,a] = 1，说明箱区 a 历史上已经被航次 v 使用过，则 n[v,a] 必须为 0。
    for v in followup_set:
        if v not in V_plus:
            continue
        for a in A:
            model.addConstr(
                n[v, a] <= 1 - data.H[v, a],
                name=f"new_area_le_not_history[{v},{a}]",
            )

    # 对既不是第一次规划、也不是后续规划的航次，保险起见固定 n=0。
    # 正常情况下 V_plus 应该被 first_set 和 followup_set 覆盖。
    for v in V_plus:
        if v not in first_set and v not in followup_set:
            for a in A:
                model.addConstr(
                    n[v, a] == 0,
                    name=f"undefined_stage_n_fixed[{v},{a}]",
                )

    # =========================
    # 4.9 同日开港航次避让约束
    # =========================

    # 约束 11：同日开港航次 u 和 v 尽量不要共用箱区 a。
    # 若二者都使用箱区 a，则 q[u,v,a] 必须为 1，并在目标函数中被惩罚。
    for (u, v) in P_same:
        for a in A:
            model.addConstr(
                w[u, a] + w[v, a] <= 1 + q[u, v, a],
                name=f"same_day_share[{u},{v},{a}]",
            )

    # =========================
    # 4.10 单箱区服务航次数限制
    # =========================

    # 约束 12：一个箱区最好不要服务超过 2 个活跃航次。
    # 若超过 2 个，则通过 h[a] 表示超额数量，并在目标函数中惩罚。
    for a in A:
        model.addConstr(
            gp.quicksum(w[v, a] for v in V_act) <= 2 + h[a],
            name=f"over_two_vessels[{a}]",
        )

    # =========================
    # 4.11 累计平均距离公平约束
    # =========================

    # 约束 13：每个活跃航次的累计平均距离不超过 eta。
    # 对当前新增规划航次，累计距离 = 历史累计距离 + 当前新增分配距离。
    for v in V_plus:
        total_planned_after = (
            data.P20_prev[v]
            + data.P40_prev[v]
            + data.R20[v]
            + data.R40[v]
        )

        # 若该航次当前规划后仍无规划箱量，则不施加平均距离约束，避免右端为 0。
        if total_planned_after > 0:
            model.addConstr(
                data.G_prev[v]
                + gp.quicksum(
                    data.distance[v, a] * (x20[v, a] + x40[v, a])
                    for a in A
                )
                <= eta * total_planned_after,
                name=f"average_distance_active_new[{v}]",
            )

    # 对非当前新增规划但仍然活跃的航次，累计距离和累计箱量均为历史值。
    # 若其历史平均距离已经较大，也应被 eta 覆盖，保证 eta 表示所有活跃航次中的最大累计平均距离。
    for v in V_act:
        if v not in V_plus:
            total_planned_history = data.P20_prev[v] + data.P40_prev[v]
            if total_planned_history > 0:
                model.addConstr(
                    data.G_prev[v] <= eta * total_planned_history,
                    name=f"average_distance_active_history[{v}]",
                )

    # =========================
    # 4.12 目标函数
    # =========================

    # 目标项 1：当前新增规划总距离成本。
    # 距离成本越小，表示箱子越倾向于分配到靠近航次泊位的箱区。
    Z_dist = gp.quicksum(
        data.distance[v, a] * (x20[v, a] + x40[v, a])
        for v in V_plus
        for a in A
    )

    # 目标项 2：最大累计平均距离公平项。
    # 最小化 eta 可以避免某些航次被系统性分配到较远箱区。
    Z_fair = eta

    # 目标项 3：当前分配碎片化惩罚。
    # z[v,a] 越多，表示当前新增箱量被分散到越多箱区。
    Z_frag = gp.quicksum(z[v, a] for v in V_plus for a in A)

    # 目标项 4：同日开港航次共用箱区惩罚。
    # q[u,v,a] 越多，表示同日开港航次之间共用箱区越严重。
    Z_same = gp.quicksum(q[u, v, a] for (u, v) in P_same for a in A)

    # 目标项 5：箱区服务超过 2 个航次的超额惩罚。
    # h[a] 越大，表示箱区 a 被过多航次共用。
    Z_over = gp.quicksum(h[a] for a in A)

    # 目标项 6：后续规划中新开箱区惩罚。
    # n[v,a] 越多，表示后续 20%/10% 规划越没有沿用第一次规划的主箱区。
    Z_new = gp.quicksum(n[v, a] for v in V_plus for a in A)

    # 目标项 7：未满足需求惩罚。
    # s20/s40 越大，表示当前应规划箱量没有被完全分配。
    Z_miss = gp.quicksum(s20[v] + s40[v] for v in V_plus)

    # 综合目标函数。
    # 建议权重优先级：M_miss > M_new > lambda_same/lambda_over > lambda_dist/lambda_fair/lambda_frag。
    model.setObjective(
        data.lambda_dist * Z_dist
        + data.lambda_fair * Z_fair
        + data.lambda_frag * Z_frag
        + data.lambda_same * Z_same
        + data.lambda_over * Z_over
        + data.M_new * Z_new
        + data.M_miss * Z_miss,
        GRB.MINIMIZE,
    )

    # 求解模型。
    model.optimize()

    # =========================
    # 4.13 结果提取
    # =========================

    result: Dict[str, object] = {
        "status": model.Status,
        "objective_value": None,
        "x20": {},
        "x40": {},
        "z": {},
        "w": {},
        "n": {},
        "q": {},
        "h": {},
        "eta": None,
        "s20": {},
        "s40": {},
    }

    if model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} and model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["eta"] = eta.X

        # 只导出非零分配结果，方便查看。
        result["x20"] = {
            (v, a): round(x20[v, a].X)
            for v in V_plus
            for a in A
            if x20[v, a].X > 1e-6
        }
        result["x40"] = {
            (v, a): round(x40[v, a].X)
            for v in V_plus
            for a in A
            if x40[v, a].X > 1e-6
        }
        result["z"] = {
            (v, a): round(z[v, a].X)
            for v in V_plus
            for a in A
            if z[v, a].X > 1e-6
        }
        result["w"] = {
            (v, a): round(w[v, a].X)
            for v in V_act
            for a in A
            if w[v, a].X > 1e-6
        }
        result["n"] = {
            (v, a): round(n[v, a].X)
            for v in V_plus
            for a in A
            if n[v, a].X > 1e-6
        }
        result["q"] = {
            (u, v, a): round(q[u, v, a].X)
            for (u, v) in P_same
            for a in A
            if q[u, v, a].X > 1e-6
        }
        result["h"] = {
            a: round(h[a].X)
            for a in A
            if h[a].X > 1e-6
        }
        result["s20"] = {
            v: s20[v].X
            for v in V_plus
            if s20[v].X > 1e-6
        }
        result["s40"] = {
            v: s40[v].X
            for v in V_plus
            if s40[v].X > 1e-6
        }

    return result


# =========================
# 5. 示例：第一次规划
# =========================

def example_first_plan() -> None:
    """
    第一次规划示例：
    - 当前两个航次都处于开港前 24 小时规划事件。
    - 需要规划累计 70% 对应的新增箱量。
    - 需要施加 K_min/K_max 主箱区数量约束。
    - 不施加新开箱区惩罚，因为第一次规划本身就是选择主箱区。
    """

    V_plus = ["453334", "453335"]
    V_act = ["453334", "453335"]
    A = ["A01", "A02", "A03", "A04", "A05"]
    P_same = [("453334", "453335")]

    data = YardAllocationData(
        V_plus=V_plus,
        V_act=V_act,
        A=A,
        P_same=P_same,
        R20={"453334": 70, "453335": 50},
        R40={"453334": 42, "453335": 35},
        C20={"A01": 120, "A02": 100, "A03": 80, "A04": 90, "A05": 70},
        C40={"A01": 50, "A02": 40, "A03": 30, "A04": 35, "A05": 25},
        distance={
            (v, a): dist
            for v, dist_list in {
                "453334": [10, 12, 18, 25, 30],
                "453335": [22, 16, 12, 14, 26],
            }.items()
            for a, dist in zip(A, dist_list)
        },
        B={(v, a): 0 for v in V_act for a in A},
        H={(v, a): 0 for v in V_act for a in A},
        P20_prev={"453334": 0, "453335": 0},
        P40_prev={"453334": 0, "453335": 0},
        G_prev={"453334": 0.0, "453335": 0.0},
        first_plan_vessels=V_plus,
        followup_plan_vessels=[],
        K_min={"453334": 2, "453335": 2},
        K_max={"453334": 4, "453335": 4},
    )

    result = solve_yard_area_allocation(data, verbose=True)
    print(result)


# =========================
# 6. 示例：后续规划
# =========================

def example_followup_plan() -> None:
    """
    后续规划示例：
    - 当前航次 453334 处于开港后 24 小时，需要补充规划到 90%。
    - 历史 H 中记录了第一次规划选出的主箱区。
    - 当前模型会优先把新增箱量放进历史箱区；如果启用新箱区，n[v,a] 会等于 1 并被惩罚。
    """

    V_plus = ["453334"]
    V_act = ["453334", "453335"]
    A = ["A01", "A02", "A03", "A04", "A05"]
    P_same = [("453334", "453335")]

    # 假设第一次规划后，453334 使用了 A01、A02，453335 使用了 A03、A04。
    H = {(v, a): 0 for v in V_act for a in A}
    H["453334", "A01"] = 1
    H["453334", "A02"] = 1
    H["453335", "A03"] = 1
    H["453335", "A04"] = 1

    data = YardAllocationData(
        V_plus=V_plus,
        V_act=V_act,
        A=A,
        P_same=P_same,
        R20={"453334": 20},
        R40={"453334": 12},
        C20={"A01": 30, "A02": 25, "A03": 80, "A04": 90, "A05": 70},
        C40={"A01": 12, "A02": 10, "A03": 30, "A04": 35, "A05": 25},
        distance={
            (v, a): dist
            for v, dist_list in {
                "453334": [10, 12, 18, 25, 30],
                "453335": [22, 16, 12, 14, 26],
            }.items()
            for a, dist in zip(A, dist_list)
        },
        B={(v, a): 0 for v in V_act for a in A},
        H=H,
        P20_prev={"453334": 70, "453335": 50},
        P40_prev={"453334": 42, "453335": 35},
        G_prev={"453334": 70 * 10 + 42 * 12, "453335": 50 * 12 + 35 * 14},
        first_plan_vessels=[],
        followup_plan_vessels=V_plus,
    )

    result = solve_yard_area_allocation(data, verbose=True)
    print(result)


if __name__ == "__main__":
    # 按需打开示例。
    # example_first_plan()
    # example_followup_plan()
    pass
