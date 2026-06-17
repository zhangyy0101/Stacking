from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

gp = None
GRB = None


Vessel = str
Flow = str
Area = str

VF = Tuple[Vessel, Flow]
VA = Tuple[Vessel, Area]
AF = Tuple[Area, Flow]
VFA = Tuple[Vessel, Flow, Area]


@dataclass(frozen=True)
class YardPlanningWeights:
    """
    目标函数权重。

    当前优先级口径：
    未满足需求惩罚 > 作业能力超额惩罚 > 距离成本 > 箱区服务航次数超额
    > 旧计划调整幅度 > 箱型分布均衡。

    距离项会在模型内部除以最大距离做归一化，因此这里的 distance
    可以使用和其它项接近的数量级。
    """

    miss: float = 100.0
    operation: float = 50.0
    of_area: float = 40.0
    distance: float = 30.0
    share: float = 20.0
    berth_conflict: float = 25.0
    adjustment: float = 10.0
    balance: float = 1.0
    priority_area: float = 0.01


@dataclass
class DailyRollingYardPlanningData:
    """
    单个规划节点的 Gurobi 模型输入。

    这里的字段尽量一一对应数学模型中的集合和参数。数据预处理层需要先完成：
    需求 D、快照覆盖 S/L/Q、容量 C/Cbar、箱区用途 U、可用性 E、
    距离 d、昨日计划 P、新旧航次标记 O 等参数构造。
    """

    # 集合 V：当前规划节点参与重规划的所有活跃航次。
    V: Sequence[Vessel]

    # 集合 F：作业流向集合，例如 OF/OZ/IF/IZ/T。
    F: Sequence[Flow]

    # 集合 A：当前参与规划的箱区集合。
    A: Sequence[Area]

    # 参数 D：最终需求箱量。出口需求已完成资料箱、快照箱、预估箱合并；进口直接使用资料箱。
    D20: Mapping[VF, float]
    D40: Mapping[VF, float]

    # 参数 C20：未拆分场箱位下的 20 尺等价物理剩余容量。
    C20: Mapping[Area, float]

    # 参数 C40：拆分后适放 40 尺箱的剩余容量。
    C40: Mapping[Area, float]

    # 参数 H：箱区当前规划期建议承接的最大新增作业箱量。
    H: Mapping[Area, float]

    # 参数 d：航次泊位到箱区的距离。
    distance: Mapping[VA, float]

    # 参数 U：箱区用途，U[a,f]=1 表示箱区 a 允许新增流向 f 的箱。
    U: Mapping[AF, int]

    # 参数 E：箱区可用性，综合箱区开放、功能、容量、TOPS 等因素。
    E20: Mapping[VFA, int]
    E40: Mapping[VFA, int]

    # 参数 C20Direct：拆分后适放 20 尺箱的剩余容量。
    # 它与 C20 的区别是：C20 控制 20 尺等价物理空间，C20Direct 控制真实可放 20 尺的位置。
    C20Direct: Optional[Mapping[Area, float]] = None

    # 参数 S/L/Q：当前快照已出现并关联到活跃航次的箱量。
    # 若 S 未传，求解器会使用 L+Q 自动构造 S。
    S20: Mapping[VFA, float] = field(default_factory=dict)
    S40: Mapping[VFA, float] = field(default_factory=dict)
    L20: Mapping[VFA, float] = field(default_factory=dict)
    L40: Mapping[VFA, float] = field(default_factory=dict)
    Q20: Mapping[VFA, float] = field(default_factory=dict)
    Q40: Mapping[VFA, float] = field(default_factory=dict)

    # 参数 R^S：扣除当前快照已出现箱 S 后的剩余需求。通常由 D - sum_a S 自动计算。
    R20S: Optional[Mapping[VF, float]] = None
    R40S: Optional[Mapping[VF, float]] = None

    # 参数 TOPS/Cbar：TOPS 对航次-箱区容量的差异化扣减。
    TOPS20: Mapping[VA, float] = field(default_factory=dict)
    TOPS40: Mapping[VA, float] = field(default_factory=dict)
    Cbar20: Optional[Mapping[VA, float]] = None
    Cbar20Direct: Optional[Mapping[VA, float]] = None
    Cbar40: Optional[Mapping[VA, float]] = None

    # 参数 P/O：昨日计划和新旧航次标记。O[v]=1 时，航次 v 进入旧计划调整惩罚。
    P20: Mapping[VFA, float] = field(default_factory=dict)
    P40: Mapping[VFA, float] = field(default_factory=dict)
    O: Mapping[Vessel, int] = field(default_factory=dict)

    # 参数 M：航次计划总量大 M，用于联动 X 和 y。若未传，默认等于该航次总需求。
    M: Optional[Mapping[Vessel, float]] = None

    # 出口 OF 作业路数。软约束上限为 2 * OFWorkLanes[v] 个箱区。
    OFWorkLanes: Mapping[Vessel, float] = field(default_factory=dict)

    # 预计靠泊时间相近的出口航次对；模型会软惩罚这些航次共用同一箱区。
    berth_conflict_pairs: Sequence[Tuple[Vessel, Vessel]] = field(default_factory=tuple)

    # 人工指定的航次-箱区控制。allowed 限制新增箱只能进入这些箱区；
    # required 通过高权重软惩罚尽量让航次使用指定箱区。
    allowed_areas_by_vessel: Mapping[Vessel, Sequence[Area]] = field(default_factory=dict)
    required_areas_by_vessel: Mapping[Vessel, Sequence[Area]] = field(default_factory=dict)
    priority_areas_by_vessel: Mapping[Vessel, Sequence[Area]] = field(default_factory=dict)

    # 目标函数权重。
    weights: YardPlanningWeights = field(default_factory=YardPlanningWeights)
    required_area_penalty: float = 10000.0

    # 是否允许未满足需求。True 时用 s 变量承接不可满足需求；False 时强制 s=0。
    allow_unmet_demand: bool = True

    # 是否严格检查所有必需参数索引。
    strict_validation: bool = True

    # 非严格模式下的默认值。
    default_U: int = 1
    default_E20: int = 1
    default_E40: int = 1
    default_distance: float = 0.0
    default_H: Optional[float] = None

    name: str = "planning_large"


@dataclass(frozen=True)
class DailyRollingYardPlanningSolution:
    """求解结果容器，保存非零变量值和目标函数分项。"""

    # Gurobi 求解状态及基础求解信息。
    status: int
    status_name: str
    objective_value: Optional[float]
    best_bound: Optional[float]
    mip_gap: Optional[float]
    runtime: Optional[float]

    # 决策变量 X：当前规划完成后的总计划箱量。
    x20: Dict[VFA, int]
    x40: Dict[VFA, int]

    # 决策变量 y：航次是否使用某箱区。
    y: Dict[VA, int]

    # 软约束变量：箱区服务航次数超额量 h、作业能力超额量 o。
    h: Dict[Area, int]
    o: Dict[Area, float]
    of_area_used: Dict[VA, int]
    of_area_over: Dict[Vessel, float]

    # 未满足需求变量 s。
    s20: Dict[VF, float]
    s40: Dict[VF, float]

    # 箱型分布均衡变量 m。
    m20: Dict[VF, float]
    m40: Dict[VF, float]

    # 旧航次相对上一日计划的正负调整变量 r。
    r20_pos: Dict[VFA, float]
    r20_neg: Dict[VFA, float]
    r40_pos: Dict[VFA, float]
    r40_neg: Dict[VFA, float]
    berth_conflict_shared: Dict[Tuple[Vessel, Vessel, Area], int]
    required_area_unmet: Dict[VA, float]
    objective_components: Dict[str, float]
    model: Optional[Any] = None

    def allocation_rows(self, *, include_zero: bool = False) -> list[dict[str, Any]]:
        """
        功能：
            将 20 尺和 40 尺箱的分配结果转换为行式明细数据。

        参数：
            include_zero: 是否保留现有结果字典中数量为 0 的分配记录。

        返回：
            分配明细列表。每行包含航次、流向、箱区、箱型和计划箱量字段。
        """
        rows: list[dict[str, Any]] = []
        for size, values in (("20", self.x20), ("40", self.x40)):
            for (vessel, flow, area), qty in values.items():
                if include_zero or qty:
                    rows.append(
                        {
                            "voy_id": vessel,
                            "flow": flow,
                            "area_no": area,
                            "size": size,
                            "planned_qty": int(qty),
                        }
                    )
        return rows


def build_daily_rolling_yard_model(
    data: DailyRollingYardPlanningData,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
    gurobi_params: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """
    功能：
        构建但不求解日滚动堆场规划的 Gurobi MILP 模型。

    参数：
        data: 单个规划节点的模型输入数据。
        time_limit: 可选的 Gurobi 求解时间上限，单位为秒。
        mip_gap: 可选的 Gurobi MIPGap 停止阈值。
        verbose: 是否输出 Gurobi 求解日志。
        gurobi_params: 额外传入 Gurobi 的参数字典，键为参数名，值为参数值。

    返回：
        ``(model, variables, derived_params)`` 三元组。其中 ``model`` 是 Gurobi
        模型，``variables`` 保存模型变量和目标函数分项表达式，``derived_params``
        保存补齐和派生后的模型参数。

    异常：
        ImportError: 未安装或无法导入 ``gurobipy`` 时抛出。
        KeyError: 严格校验模式下缺少必需参数索引时抛出。
        ValueError: 输入集合、容量、需求或二元参数不满足模型要求时抛出。
    """

    _ensure_gurobi()

    # -------------------------
    # 集合定义
    # -------------------------
    # V：活跃航次集合；F：作业流向集合；A：箱区集合。
    V = _unique_list(data.V, "V")
    F = _unique_list(data.F, "F")
    A = _unique_list(data.A, "A")
    _validate_basic_sets(V, F, A)

    # 派生参数：补齐默认值、自动计算 S=L+Q、R=D-S、Cbar、M、V_old。
    params = _prepare_params(data, V, F, A)

    # V_old：旧航次集合，仅这些航次会施加计划调整幅度惩罚。
    V_old = params["V_old"]
    weights = data.weights

    model = gp.Model(data.name)
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if gurobi_params:
        for name, value in gurobi_params.items():
            setattr(model.Params, name, value)

    # -------------------------
    # 决策变量
    # -------------------------
    # X20/X40[v,f,a]：当前规划完成后，航次 v、流向 f、箱区 a 的总计划箱量。
    # 注意 X 是“总计划量”，不是新增量；已在场箱 S 也包含在 X 中。
    X20 = model.addVars(V, F, A, vtype=GRB.INTEGER, lb=0, name="X20")
    X40 = model.addVars(V, F, A, vtype=GRB.INTEGER, lb=0, name="X40")

    # y[v,a]：若航次 v 在箱区 a 有任意计划箱量，则为 1。
    y = model.addVars(V, A, vtype=GRB.BINARY, name="y")

    # h[a]：箱区 a 服务超过 2 个航次的超额数量。
    h = model.addVars(A, vtype=GRB.INTEGER, lb=0, name="h")

    # o[a]：箱区 a 超过建议作业能力 H[a] 的新增箱量。
    o = model.addVars(A, vtype=GRB.CONTINUOUS, lb=0, name="o")

    # r_pos/r_neg：旧航次当前计划 X 与上一日计划 P 的正负偏差，用于线性化 |X-P|。
    r20_pos = model.addVars(V_old, F, A, vtype=GRB.CONTINUOUS, lb=0, name="r20_pos")
    r20_neg = model.addVars(V_old, F, A, vtype=GRB.CONTINUOUS, lb=0, name="r20_neg")
    r40_pos = model.addVars(V_old, F, A, vtype=GRB.CONTINUOUS, lb=0, name="r40_pos")
    r40_neg = model.addVars(V_old, F, A, vtype=GRB.CONTINUOUS, lb=0, name="r40_neg")

    # m20/m40[v,f]：同一航次、同一流向、同一箱型在单个箱区的最大计划量。
    # 最小化 m 可以鼓励箱量不要过度集中在单个箱区。
    m20 = model.addVars(V, F, vtype=GRB.CONTINUOUS, lb=0, name="m20")
    m40 = model.addVars(V, F, vtype=GRB.CONTINUOUS, lb=0, name="m40")

    # s20/s40[v,f]：未满足需求量。若 allow_unmet_demand=False，则后面强制 s=0。
    s20 = model.addVars(V, F, vtype=GRB.CONTINUOUS, lb=0, name="s20")
    s40 = model.addVars(V, F, vtype=GRB.CONTINUOUS, lb=0, name="s40")

    # of_area_used[v,a]：出口航次 v 的 OF 箱是否使用箱区 a。
    # of_area_over[v]：OF 使用箱区数超过 2 倍作业路数的超额量。
    OF_area_vessels = params["OF_area_vessels"]
    berth_conflict_pairs = params["berth_conflict_pairs"]
    berth_conflict_keys = params["berth_conflict_keys"]
    required_area_keys = params["required_area_keys"]
    priority_area_keys = params["priority_area_keys"]
    of_area_used = model.addVars(OF_area_vessels, A, vtype=GRB.BINARY, name="of_area_used")
    of_area_over = model.addVars(OF_area_vessels, vtype=GRB.CONTINUOUS, lb=0, name="of_area_over")
    berth_conflict_shared = model.addVars(
        berth_conflict_keys,
        vtype=GRB.BINARY,
        name="berth_conflict_shared",
    )
    required_area_unmet = model.addVars(
        required_area_keys,
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name="required_area_unmet",
    )

    # OF balance range variables. These only apply to export vessels with OF work lanes.
    of_balance_u20 = model.addVars(OF_area_vessels, vtype=GRB.CONTINUOUS, lb=0, name="of_balance_u20")
    of_balance_l20 = model.addVars(OF_area_vessels, vtype=GRB.CONTINUOUS, lb=0, name="of_balance_l20")
    of_balance_u40 = model.addVars(OF_area_vessels, vtype=GRB.CONTINUOUS, lb=0, name="of_balance_u40")
    of_balance_l40 = model.addVars(OF_area_vessels, vtype=GRB.CONTINUOUS, lb=0, name="of_balance_l40")
    for v in OF_area_vessels:
        of_balance_u20[v].UB = max(0.0, params["D20"][v, "OF"])
        of_balance_l20[v].UB = max(0.0, params["D20"][v, "OF"])
        of_balance_u40[v].UB = max(0.0, params["D40"][v, "OF"])
        of_balance_l40[v].UB = max(0.0, params["D40"][v, "OF"])

    # -------------------------
    # 约束 1：需求满足约束
    # sum_a X[v,f,a] + s[v,f] = D[v,f]
    # 若 s>0，表示该航次/流向/箱型有未满足需求。
    # -------------------------
    for v in V:
        for f in F:
            model.addConstr(
                gp.quicksum(X20[v, f, a] for a in A) + s20[v, f] == params["D20"][v, f],
                name=f"demand20[{v},{f}]",
            )
            model.addConstr(
                gp.quicksum(X40[v, f, a] for a in A) + s40[v, f] == params["D40"][v, f],
                name=f"demand40[{v},{f}]",
            )
            if not data.allow_unmet_demand:
                # 严格需求模式：不允许任何未满足需求。
                model.addConstr(s20[v, f] == 0, name=f"no_unmet20[{v},{f}]")
                model.addConstr(s40[v, f] == 0, name=f"no_unmet40[{v},{f}]")

    # -------------------------
    # 约束 2：快照已出现箱覆盖约束
    # X[v,f,a] >= S[v,f,a]
    # 已经在堆场并关联到当前活跃航次的箱，无论正常贝位 L 还是异常贝位 Q，
    # 都必须被当前总计划覆盖。
    # -------------------------
    for v in V:
        for f in F:
            for a in A:
                model.addConstr(
                    X20[v, f, a] >= params["S20"][v, f, a],
                    name=f"snapshot_cover20[{v},{f},{a}]",
                )
                model.addConstr(
                    X40[v, f, a] >= params["S40"][v, f, a],
                    name=f"snapshot_cover40[{v},{f},{a}]",
                )

                # -------------------------
                # 约束 3：箱区用途硬约束
                # X-S <= R^S * U[a,f]
                # 只限制快照以外的新增计划量，不限制已经在场的 S。
                # -------------------------
                model.addConstr(
                    X20[v, f, a] - params["S20"][v, f, a]
                    <= params["R20S"][v, f] * params["U"][a, f],
                    name=f"use_rule20[{v},{f},{a}]",
                )
                model.addConstr(
                    X40[v, f, a] - params["S40"][v, f, a]
                    <= params["R40S"][v, f] * params["U"][a, f],
                    name=f"use_rule40[{v},{f},{a}]",
                )

                # -------------------------
                # 约束 4：箱区可用性约束
                # X-S <= R^S * E[v,f,a]
                # E 综合考虑箱区开放、功能、容量/TOPS 后是否仍可用等因素。
                # 同样只限制新增计划量，不限制快照中的 S。
                # -------------------------
                model.addConstr(
                    X20[v, f, a] - params["S20"][v, f, a]
                    <= params["R20S"][v, f] * params["E20"][v, f, a],
                    name=f"available20[{v},{f},{a}]",
                )
                model.addConstr(
                    X40[v, f, a] - params["S40"][v, f, a]
                    <= params["R40S"][v, f] * params["E40"][v, f, a],
                    name=f"available40[{v},{f},{a}]",
                )

    # -------------------------
    # 约束 5：箱区容量约束
    # 5.1 20 尺适放位容量：sum(X20-S20) <= C20Direct
    # 5.2 20 尺等价物理容量：sum(X20-S20)+2*sum(X40-S40) <= C20
    # 5.3 40 尺适放位容量：sum(X40-S40) <= C40
    # C20Direct/C40 来自拆分后的适放位表，C20 来自未拆分场箱位的物理等价空间。
    # -------------------------
    for a in A:
        model.addConstr(
            gp.quicksum(X20[v, f, a] - params["S20"][v, f, a] for v in V for f in F)
            <= params["C20Direct"][a],
            name=f"capacity20_direct[{a}]",
        )
        model.addConstr(
            gp.quicksum(X20[v, f, a] - params["S20"][v, f, a] for v in V for f in F)
            + 2 * gp.quicksum(X40[v, f, a] - params["S40"][v, f, a] for v in V for f in F)
            <= params["C20"][a],
            name=f"capacity20_equiv[{a}]",
        )
        model.addConstr(
            gp.quicksum(X40[v, f, a] - params["S40"][v, f, a] for v in V for f in F)
            <= params["C40"][a],
            name=f"capacity40[{a}]",
        )

        # 约束 6：箱区作业能力软约束
        # 新增作业箱量不能超过 H[a]+o[a]。o[a] 会在目标函数中被惩罚。
        model.addConstr(
            gp.quicksum(
                (X20[v, f, a] - params["S20"][v, f, a])
                + (X40[v, f, a] - params["S40"][v, f, a])
                for v in V
                for f in F
            )
            <= params["H"][a] + o[a],
            name=f"operation_capacity[{a}]",
        )

        # 约束 7：单箱区服务航次数软约束
        # 一个箱区最好服务不超过 2 个航次；超过部分由 h[a] 记录并惩罚。
        model.addConstr(
            gp.quicksum(y[v, a] for v in V) <= 2 + h[a],
            name=f"area_share[{a}]",
        )

    # -------------------------
    # 约束 8：航次级 TOPS 容量约束
    # 对每个航次 v、箱区 a，使用扣除其它航次 TOPS 后的 Cbar。
    # 当前实现同时约束 Cbar20Direct、Cbar20、Cbar40。
    # -------------------------
    for v in V:
        for a in A:
            model.addConstr(
                gp.quicksum(X20[v, f, a] - params["S20"][v, f, a] for f in F)
                <= params["Cbar20Direct"][v, a],
                name=f"tops_capacity20_direct[{v},{a}]",
            )
            model.addConstr(
                gp.quicksum(
                    (X20[v, f, a] - params["S20"][v, f, a])
                    + 2 * (X40[v, f, a] - params["S40"][v, f, a])
                    for f in F
                )
                <= params["Cbar20"][v, a],
                name=f"tops_capacity20[{v},{a}]",
            )
            model.addConstr(
                gp.quicksum(X40[v, f, a] - params["S40"][v, f, a] for f in F)
                <= params["Cbar40"][v, a],
                name=f"tops_capacity40[{v},{a}]",
            )

            # 约束 9：X 与 y 的联动约束
            # 若 y[v,a]=0，则该航次在该箱区不能有任何计划箱量；
            # 若 y[v,a]=1，则该航次在该箱区至少有 1 个计划箱。
            total_plan = gp.quicksum(X20[v, f, a] + X40[v, f, a] for f in F)
            model.addConstr(
                total_plan <= params["M"][v] * y[v, a],
                name=f"link_y_upper[{v},{a}]",
            )
            model.addConstr(
                total_plan >= y[v, a],
                name=f"link_y_lower[{v},{a}]",
            )

    # 约束 10：预计靠泊时间接近的出口航次尽量不共用箱区。
    # 该规则为软约束；若两个航次都使用同一箱区，则 berth_conflict_shared=1 并进入目标函数惩罚。
    for left, right in berth_conflict_pairs:
        for a in A:
            shared = berth_conflict_shared[left, right, a]
            model.addConstr(shared >= y[left, a] + y[right, a] - 1, name=f"berth_conflict_lb[{left},{right},{a}]")
            model.addConstr(shared <= y[left, a], name=f"berth_conflict_left[{left},{right},{a}]")
            model.addConstr(shared <= y[right, a], name=f"berth_conflict_right[{left},{right},{a}]")

    # 约束 11：人工 add 箱区尽量被使用；若容量或功能原因放不进去，required_area_unmet 会记录。
    for v, a in required_area_keys:
        model.addConstr(required_area_unmet[v, a] >= 1 - y[v, a], name=f"required_area_unmet[{v},{a}]")

    # -------------------------
    # 约束 12：出口 OF 使用箱区数软约束
    # 每个出口航次的 OF 箱使用箱区数建议不超过 2 * 作业路数；超过部分进入 of_area_over。
    # -------------------------
    if "OF" in F:
        for v in OF_area_vessels:
            for a in A:
                of_total_plan = X20[v, "OF", a] + X40[v, "OF", a]
                model.addConstr(
                    of_total_plan <= params["M"][v] * of_area_used[v, a],
                    name=f"link_of_area_upper[{v},{a}]",
                )
                model.addConstr(
                    of_total_plan >= of_area_used[v, a],
                    name=f"link_of_area_lower[{v},{a}]",
                )
            model.addConstr(
                gp.quicksum(of_area_used[v, a] for a in A)
                <= params["OF_area_limit"][v] + of_area_over[v],
                name=f"of_area_limit[{v}]",
            )

    # -------------------------
    # 约束 11：旧航次计划调整幅度约束
    # X - P = r_pos - r_neg
    # 目标函数惩罚 r_pos+r_neg，等价于惩罚 |X-P|。
    # 只对 V_old 建立，新航次不惩罚首次规划。
    # -------------------------
    for v in V_old:
        for f in F:
            for a in A:
                model.addConstr(
                    X20[v, f, a] - params["P20"][v, f, a]
                    == r20_pos[v, f, a] - r20_neg[v, f, a],
                    name=f"adjust20[{v},{f},{a}]",
                )
                model.addConstr(
                    X40[v, f, a] - params["P40"][v, f, a]
                    == r40_pos[v, f, a] - r40_neg[v, f, a],
                    name=f"adjust40[{v},{f},{a}]",
                )

    # -------------------------
    # 约束 12：箱型分布均衡约束
    # X20[v,f,a] <= m20[v,f]，X40[v,f,a] <= m40[v,f]
    # 最小化 m 后，同一航次/流向/箱型不会过度集中到单个箱区。
    # -------------------------
    if "OF" in F:
        for v in OF_area_vessels:
            big_m20 = max(0.0, params["D20"][v, "OF"])
            big_m40 = max(0.0, params["D40"][v, "OF"])
            for a in A:
                model.addConstr(
                    X20[v, "OF", a] <= of_balance_u20[v],
                    name=f"of_balance20_upper[{v},{a}]",
                )
                model.addConstr(
                    X40[v, "OF", a] <= of_balance_u40[v],
                    name=f"of_balance40_upper[{v},{a}]",
                )
                model.addConstr(
                    X20[v, "OF", a] >= of_balance_l20[v] - big_m20 * (1 - of_area_used[v, a]),
                    name=f"of_balance20_lower[{v},{a}]",
                )
                model.addConstr(
                    X40[v, "OF", a] >= of_balance_l40[v] - big_m40 * (1 - of_area_used[v, a]),
                    name=f"of_balance40_lower[{v},{a}]",
                )

    # -------------------------
    # 目标函数分项
    # -------------------------
    # Z_miss：未满足需求总量，最高优先级惩罚。
    Z_miss = gp.quicksum(s20[v, f] + s40[v, f] for v in V for f in F)

    # Z_adj：旧航次相对上一日计划的调整幅度。
    Z_adj = gp.quicksum(
        r20_pos[v, f, a]
        + r20_neg[v, f, a]
        + r40_pos[v, f, a]
        + r40_neg[v, f, a]
        for v in V_old
        for f in F
        for a in A
    )

    # Z_dist：泊位到箱区距离成本，按总计划箱量加权。
    # 为避免距离原始数值量级过大导致权重难以调节，这里除以最大距离做归一化。
    distance_scale = max(1.0, max(params["distance"].values()) if params["distance"] else 1.0)
    Z_dist_raw = gp.quicksum(
        params["distance"][v, a] * (X20[v, f, a] + X40[v, f, a])
        for v in V
        for f in F
        for a in A
    )
    Z_dist = gp.quicksum(
        (params["distance"][v, a] / distance_scale) * (X20[v, f, a] + X40[v, f, a])
        for v in V
        for f in F
        for a in A
    )

    # Z_share：单箱区服务航次数超过 2 的超额惩罚。
    Z_share = gp.quicksum(h[a] for a in A)

    # Z_berth_conflict：预计靠泊时间接近的出口航次共用箱区惩罚。
    Z_berth_conflict = gp.quicksum(berth_conflict_shared[key] for key in berth_conflict_keys)

    # Z_required_area：人工 add 箱区未被使用的软惩罚。
    Z_required_area = gp.quicksum(required_area_unmet[key] for key in required_area_keys)
    Z_priority_area = gp.quicksum(y[v, a] for v, a in priority_area_keys)

    # Z_op：箱区作业能力超额惩罚。
    Z_op = gp.quicksum(o[a] for a in A)

    # Z_of_area：出口 OF 箱使用箱区数超过 2 倍作业路数的超额惩罚。
    Z_of_area = gp.quicksum(of_area_over[v] for v in OF_area_vessels)

    # Z_bal：20/40 箱型分布均衡惩罚。
    Z_bal = gp.quicksum(
        (of_balance_u20[v] - of_balance_l20[v]) + (of_balance_u40[v] - of_balance_l40[v])
        for v in OF_area_vessels
    )

    # 综合目标函数：按业务优先级加权求和并最小化。
    model.setObjective(
        weights.miss * Z_miss
        + weights.operation * Z_op
        + weights.of_area * Z_of_area
        + weights.distance * Z_dist
        + weights.share * Z_share
        + weights.berth_conflict * Z_berth_conflict
        + data.required_area_penalty * Z_required_area
        + getattr(weights, "priority_area", 0.01) * Z_priority_area
        + weights.adjustment * Z_adj
        + weights.balance * Z_bal,
        GRB.MINIMIZE,
    )

    variables = {
        "X20": X20,
        "X40": X40,
        "y": y,
        "h": h,
        "o": o,
        "r20_pos": r20_pos,
        "r20_neg": r20_neg,
        "r40_pos": r40_pos,
        "r40_neg": r40_neg,
        "m20": m20,
        "m40": m40,
        "s20": s20,
        "s40": s40,
        "of_area_used": of_area_used,
        "of_area_over": of_area_over,
        "berth_conflict_shared": berth_conflict_shared,
        "required_area_unmet": required_area_unmet,
        "of_balance_u20": of_balance_u20,
        "of_balance_l20": of_balance_l20,
        "of_balance_u40": of_balance_u40,
        "of_balance_l40": of_balance_l40,
        "objective_components": {
            "miss": Z_miss,
            "operation": Z_op,
            "of_area": Z_of_area,
            "distance": Z_dist,
            "distance_raw": Z_dist_raw,
            "share": Z_share,
            "berth_conflict": Z_berth_conflict,
            "required_area": Z_required_area,
            "priority_area": Z_priority_area,
            "adjustment": Z_adj,
            "balance": Z_bal,
        },
    }
    return model, variables, params


def solve_daily_rolling_yard_plan(
    data: DailyRollingYardPlanningData,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
    gurobi_params: Optional[Mapping[str, Any]] = None,
    keep_model: bool = False,
) -> DailyRollingYardPlanningSolution:
    """
    功能：
        构建并求解日滚动堆场规划 MILP 模型。

    参数：
        data: 单个规划节点的模型输入数据。
        time_limit: 可选的 Gurobi 求解时间上限，单位为秒。
        mip_gap: 可选的 Gurobi MIPGap 停止阈值。
        verbose: 是否输出 Gurobi 求解日志。
        gurobi_params: 额外传入 Gurobi 的参数字典，键为参数名，值为参数值。
        keep_model: 是否在返回结果中保留 Gurobi 模型对象，便于调试或后续分析。

    返回：
        ``DailyRollingYardPlanningSolution`` 实例。若模型没有可用解，则变量结果
        字典为空，求解状态和边界信息仍会保留。

    异常：
        ImportError: 未安装或无法导入 ``gurobipy`` 时抛出。
        KeyError: 严格校验模式下缺少必需参数索引时抛出。
        ValueError: 输入集合、容量、需求或二元参数不满足模型要求时抛出。
    """

    model, variables, params = build_daily_rolling_yard_model(
        data,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
        gurobi_params=gurobi_params,
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
    components = variables["objective_components"]

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
            (params["berth_conflict_keys"],),
        ),
        required_area_unmet=_extract_float_tupledict(
            variables["required_area_unmet"],
            (params["required_area_keys"],),
        ),
        objective_components={name: expr.getValue() for name, expr in components.items()},
        model=model if keep_model else None,
    )


def _prepare_params(
    data: DailyRollingYardPlanningData,
    V: list[Vessel],
    F: list[Flow],
    A: list[Area],
) -> dict[str, Any]:
    """
    功能：
        将外部输入参数整理成完整的模型索引字典。

    参数：
        data: 单个规划节点的原始输入数据。
        V: 去重后的航次集合。
        F: 去重后的作业流向集合。
        A: 去重后的箱区集合。

    返回：
        补齐和派生后的参数字典，包括 V/F/A/V_old、需求、快照、容量、用途、
        可用性、距离、昨日计划和大 M 参数。

    异常：
        KeyError: 严格校验模式下缺少必需参数索引时抛出。
        ValueError: 快照超过需求、R^S 与自动计算值不一致、容量为负、二元参数
            非 0/1 或大 M 小于航次总需求时抛出。
    """

    # 严格模式：所有硬约束需要用到的参数必须完整覆盖索引，避免静默使用默认值造成误判。
    if data.strict_validation:
        _require_keys(data.C20, ((a,) for a in A), "C20")
        if data.C20Direct is not None:
            _require_keys(data.C20Direct, ((a,) for a in A), "C20Direct")
        _require_keys(data.C40, ((a,) for a in A), "C40")
        _require_keys(data.H, ((a,) for a in A), "H")
        _require_keys(data.distance, ((v, a) for v in V for a in A), "distance")
        _require_keys(data.U, ((a, f) for a in A for f in F), "U")
        _require_keys(data.E20, ((v, f, a) for v in V for f in F for a in A), "E20")
        _require_keys(data.E40, ((v, f, a) for v in V for f in F for a in A), "E40")
        if data.R20S is not None:
            _require_keys(data.R20S, ((v, f) for v in V for f in F), "R20S")
        if data.R40S is not None:
            _require_keys(data.R40S, ((v, f) for v in V for f in F), "R40S")
        if data.Cbar20 is not None:
            _require_keys(data.Cbar20, ((v, a) for v in V for a in A), "Cbar20")
        if data.Cbar20Direct is not None:
            _require_keys(data.Cbar20Direct, ((v, a) for v in V for a in A), "Cbar20Direct")
        if data.Cbar40 is not None:
            _require_keys(data.Cbar40, ((v, a) for v in V for a in A), "Cbar40")
        if data.M is not None:
            _require_keys(data.M, ((v,) for v in V), "M")

    # 若未给定 H 的默认值，则用当前总需求作为足够大的默认作业能力。
    total_demand = sum(_num(data.D20, (v, f)) + _num(data.D40, (v, f)) for v in V for f in F)
    default_H = data.default_H if data.default_H is not None else total_demand

    # 需求参数 D：缺失索引默认为 0。
    D20 = {(v, f): _num(data.D20, (v, f)) for v in V for f in F}
    D40 = {(v, f): _num(data.D40, (v, f)) for v in V for f in F}

    # 快照参数 S：优先使用外部显式 S；若没有，则按 S=L+Q 自动构造。
    S20 = {
        (v, f, a): _snapshot_value(data.S20, data.L20, data.Q20, (v, f, a))
        for v in V
        for f in F
        for a in A
    }
    S40 = {
        (v, f, a): _snapshot_value(data.S40, data.L40, data.Q40, (v, f, a))
        for v in V
        for f in F
        for a in A
    }

    # 昨日计划 P：仅在旧航次调整幅度约束中使用；缺失默认为 0。
    P20 = {(v, f, a): _num(data.P20, (v, f, a)) for v in V for f in F for a in A}
    P40 = {(v, f, a): _num(data.P40, (v, f, a)) for v in V for f in F for a in A}

    # 剩余需求 R^S：只能非负，否则说明 D 没有覆盖已经在场的箱。
    R20S = {}
    R40S = {}
    for v in V:
        for f in F:
            computed20 = D20[v, f] - sum(S20[v, f, a] for a in A)
            computed40 = D40[v, f] - sum(S40[v, f, a] for a in A)
            R20S[v, f] = _num(data.R20S, (v, f), computed20) if data.R20S is not None else computed20
            R40S[v, f] = _num(data.R40S, (v, f), computed40) if data.R40S is not None else computed40
            if computed20 < -1e-6 or computed40 < -1e-6:
                raise ValueError(
                    "Demand must cover snapshot quantities. "
                    f"Got D-S < 0 for vessel={v}, flow={f}: "
                    f"R20S={computed20}, R40S={computed40}."
                )
            if abs(R20S[v, f] - computed20) > 1e-6 or abs(R40S[v, f] - computed40) > 1e-6:
                raise ValueError(
                    "Provided R^S must equal D minus snapshot quantities. "
                    f"Got vessel={v}, flow={f}, provided=({R20S[v, f]}, {R40S[v, f]}), "
                    f"computed=({computed20}, {computed40})."
                )
            if R20S[v, f] < -1e-6 or R40S[v, f] < -1e-6:
                raise ValueError(
                    f"R^S must be nonnegative for vessel={v}, flow={f}: "
                    f"R20S={R20S[v, f]}, R40S={R40S[v, f]}."
                )

    # 容量、距离、用途、可用性参数。C20Direct 若未传入，则退化为 C20。
    C20 = {a: _num(data.C20, (a,)) for a in A}
    C20Direct = {
        a: _num(data.C20Direct, (a,), C20[a]) if data.C20Direct is not None else C20[a]
        for a in A
    }
    C40 = {a: _num(data.C40, (a,)) for a in A}
    H = {a: _num(data.H, (a,), default_H) for a in A}
    distance = {(v, a): _num(data.distance, (v, a), data.default_distance) for v in V for a in A}
    U = {(a, f): _binary(data.U, (a, f), data.default_U) for a in A for f in F}
    E20 = {(v, f, a): _binary(data.E20, (v, f, a), data.default_E20) for v in V for f in F for a in A}
    E40 = {(v, f, a): _binary(data.E40, (v, f, a), data.default_E40) for v in V for f in F for a in A}
    allowed_areas_by_vessel, required_areas_by_vessel = _normalize_area_controls(
        data.allowed_areas_by_vessel,
        data.required_areas_by_vessel,
        V,
        A,
    )
    priority_areas_by_vessel = _normalize_priority_area_controls(
        getattr(data, "priority_areas_by_vessel", {}),
        allowed_areas_by_vessel,
        V,
        A,
    )
    for v, allowed_areas in allowed_areas_by_vessel.items():
        for f in F:
            for a in A:
                if a not in allowed_areas:
                    E20[v, f, a] = 0
                    E40[v, f, a] = 0

    # 航次级 TOPS 扣减后容量。若外部直接传 Cbar，则使用外部值；否则用 C - TOPS 自动计算。
    Cbar20 = {}
    Cbar20Direct = {}
    Cbar40 = {}
    for v in V:
        for a in A:
            if data.Cbar20 is not None:
                Cbar20[v, a] = _num(data.Cbar20, (v, a))
            else:
                Cbar20[v, a] = max(0.0, C20[a] - _num(data.TOPS20, (v, a)))
            if data.Cbar20Direct is not None:
                Cbar20Direct[v, a] = _num(data.Cbar20Direct, (v, a))
            else:
                Cbar20Direct[v, a] = Cbar20[v, a]
            if data.Cbar40 is not None:
                Cbar40[v, a] = _num(data.Cbar40, (v, a))
            else:
                Cbar40[v, a] = max(0.0, C40[a] - _num(data.TOPS40, (v, a)))

    # 航次大 M：用于 X 与 y 的联动。默认等于该航次 20+40 总需求。
    M = {}
    for v in V:
        computed_m = sum(D20[v, f] + D40[v, f] for f in F)
        M[v] = _num(data.M, (v,), computed_m) if data.M is not None else computed_m
        if M[v] < computed_m - 1e-6:
            raise ValueError(f"M[{v}]={M[v]} is smaller than total demand {computed_m}.")

    OFWorkLanes = {v: _num(data.OFWorkLanes, (v,), 0.0) for v in V}
    OF_area_limit = {v: 2.0 * OFWorkLanes[v] for v in V}
    OF_area_vessels = [v for v in V if "OF" in F and OFWorkLanes[v] > 0]
    berth_conflict_pairs = _normalize_vessel_pairs(data.berth_conflict_pairs, V)
    berth_conflict_keys = [(left, right, a) for left, right in berth_conflict_pairs for a in A]
    required_area_keys = [
        (v, a)
        for v, required_areas in required_areas_by_vessel.items()
        for a in sorted(required_areas)
    ]
    priority_area_keys = [
        (v, a)
        for v, priority_areas in priority_areas_by_vessel.items()
        for a in A
        if a not in priority_areas
    ]

    # 旧航次集合：只有 O[v]=1 的航次进入调整幅度变量和调整惩罚。
    V_old = [v for v in V if int(data.O.get(v, 0)) == 1]

    # 基础非负性检查，防止负容量/负需求进入模型。
    _validate_nonnegative("D20", D20)
    _validate_nonnegative("D40", D40)
    _validate_nonnegative("S20", S20)
    _validate_nonnegative("S40", S40)
    _validate_nonnegative("C20", C20)
    _validate_nonnegative("C20Direct", C20Direct)
    _validate_nonnegative("C40", C40)
    _validate_nonnegative("H", H)
    _validate_nonnegative("Cbar20", Cbar20)
    _validate_nonnegative("Cbar20Direct", Cbar20Direct)
    _validate_nonnegative("Cbar40", Cbar40)
    _validate_nonnegative("OFWorkLanes", OFWorkLanes)

    return {
        "V": V,
        "F": F,
        "A": A,
        "V_old": V_old,
        "D20": D20,
        "D40": D40,
        "S20": S20,
        "S40": S40,
        "R20S": R20S,
        "R40S": R40S,
        "U": U,
        "E20": E20,
        "E40": E40,
        "C20": C20,
        "C20Direct": C20Direct,
        "C40": C40,
        "Cbar20": Cbar20,
        "Cbar20Direct": Cbar20Direct,
        "Cbar40": Cbar40,
        "H": H,
        "distance": distance,
        "P20": P20,
        "P40": P40,
        "M": M,
        "OFWorkLanes": OFWorkLanes,
        "OF_area_limit": OF_area_limit,
        "OF_area_vessels": OF_area_vessels,
        "berth_conflict_pairs": berth_conflict_pairs,
        "berth_conflict_keys": berth_conflict_keys,
        "allowed_areas_by_vessel": allowed_areas_by_vessel,
        "required_areas_by_vessel": required_areas_by_vessel,
        "priority_areas_by_vessel": priority_areas_by_vessel,
        "required_area_keys": required_area_keys,
        "priority_area_keys": priority_area_keys,
    }


def _ensure_gurobi() -> None:
    """
    功能：
        检查当前环境是否已成功导入 Gurobi Python API。

    参数：
        无。

    返回：
        无。

    异常：
        ImportError: ``gurobipy`` 或 ``GRB`` 不可用时抛出。
    """
    if gp is None or GRB is None:
        raise ImportError("gurobipy is required to solve the daily rolling yard planning model.")


def _validate_basic_sets(V: Sequence[str], F: Sequence[str], A: Sequence[str]) -> None:
    """
    功能：
        校验模型三类基础集合是否非空。

    参数：
        V: 航次集合。
        F: 作业流向集合。
        A: 箱区集合。

    返回：
        无。

    异常：
        ValueError: 任一基础集合为空时抛出。
    """
    if not V:
        raise ValueError("V must not be empty.")
    if not F:
        raise ValueError("F must not be empty.")
    if not A:
        raise ValueError("A must not be empty.")


def _unique_list(values: Sequence[str], name: str) -> list[str]:
    """
    功能：
        将输入序列转换为列表，并校验其中没有重复值。

    参数：
        values: 待转换和校验的序列。
        name: 序列名称，用于错误信息。

    返回：
        保持原始顺序的列表。

    异常：
        ValueError: 序列中存在重复值时抛出。
    """
    result = list(values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicated values.")
    return result


def _normalize_vessel_pairs(pairs: Sequence[Tuple[Vessel, Vessel]], vessels: Sequence[Vessel]) -> list[tuple[Vessel, Vessel]]:
    vessel_set = set(vessels)
    seen: set[frozenset[Vessel]] = set()
    result: list[tuple[Vessel, Vessel]] = []
    for pair in pairs or ():
        if len(pair) != 2:
            continue
        left, right = str(pair[0]), str(pair[1])
        if left == right or left not in vessel_set or right not in vessel_set:
            continue
        pair_key = frozenset((left, right))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        result.append((left, right))
    return result


def _normalize_area_controls(
    allowed: Mapping[Vessel, Sequence[Area]],
    required: Mapping[Vessel, Sequence[Area]],
    vessels: Sequence[Vessel],
    areas: Sequence[Area],
) -> tuple[dict[Vessel, set[Area]], dict[Vessel, set[Area]]]:
    vessel_set = set(vessels)
    area_set = set(areas)
    allowed_result: dict[Vessel, set[Area]] = {}
    required_result: dict[Vessel, set[Area]] = {}

    for raw_vessel, raw_areas in (allowed or {}).items():
        vessel = str(raw_vessel)
        if vessel not in vessel_set:
            continue
        normalized = {str(area) for area in (raw_areas or ()) if str(area) in area_set}
        allowed_result[vessel] = normalized

    for raw_vessel, raw_areas in (required or {}).items():
        vessel = str(raw_vessel)
        if vessel not in vessel_set:
            continue
        normalized = {str(area) for area in (raw_areas or ()) if str(area) in area_set}
        if vessel in allowed_result:
            normalized &= allowed_result[vessel]
        if normalized:
            required_result[vessel] = normalized

    return allowed_result, required_result


def _normalize_priority_area_controls(
    priority: Mapping[Vessel, Sequence[Area]],
    allowed: Mapping[Vessel, set[Area]],
    vessels: Sequence[Vessel],
    areas: Sequence[Area],
) -> dict[Vessel, set[Area]]:
    vessel_set = set(vessels)
    area_set = set(areas)
    result: dict[Vessel, set[Area]] = {}
    for raw_vessel, raw_areas in (priority or {}).items():
        vessel = str(raw_vessel)
        if vessel not in vessel_set:
            continue
        normalized = {str(area) for area in (raw_areas or ()) if str(area) in area_set}
        if vessel in allowed:
            normalized &= allowed[vessel]
        if normalized:
            result[vessel] = normalized
    return result


def _num(mapping: Optional[Mapping[Any, Any]], key: tuple[Any, ...], default: float = 0.0) -> float:
    """
    功能：
        从映射中按模型索引读取数值，并统一转换为浮点数。

    参数：
        mapping: 参数映射。为 ``None`` 时直接返回默认值。
        key: 模型索引元组；一维索引会自动退化为单个键。
        default: 映射为空或键缺失时使用的默认值。

    返回：
        读取到的浮点数值或默认值。

    异常：
        TypeError: 读取值无法转换为浮点数时可能由 ``float`` 抛出。
        ValueError: 读取值无法转换为浮点数时可能由 ``float`` 抛出。
    """
    if mapping is None:
        return float(default)
    lookup_key: Any = key[0] if len(key) == 1 else key
    value = mapping.get(lookup_key, default)
    return float(value)


def _snapshot_value(
    S: Mapping[Any, Any],
    L: Mapping[Any, Any],
    Q: Mapping[Any, Any],
    key: tuple[Any, ...],
) -> float:
    """
    功能：
        获取快照箱量 S；若 S 未显式给出，则使用 L+Q 自动构造。

    参数：
        S: 显式快照箱量映射。
        L: 正常贝位快照箱量映射。
        Q: 异常贝位快照箱量映射。
        key: 模型索引元组；一维索引会自动退化为单个键。

    返回：
        指定索引下的快照箱量。

    异常：
        TypeError: 读取值无法转换为浮点数时可能由 ``float`` 抛出。
        ValueError: 读取值无法转换为浮点数时可能由 ``float`` 抛出。
    """
    lookup_key: Any = key[0] if len(key) == 1 else key
    if lookup_key in S:
        return float(S[lookup_key])
    return _num(L, key) + _num(Q, key)


def _binary(mapping: Mapping[Any, Any], key: tuple[Any, ...], default: int) -> int:
    """
    功能：
        从映射中读取二元参数，并校验结果只能为 0 或 1。

    参数：
        mapping: 参数映射。
        key: 模型索引元组；一维索引会自动退化为单个键。
        default: 映射中键缺失时使用的默认值。

    返回：
        读取并四舍五入后的二元整数值。

    异常：
        ValueError: 参数值不是 0 或 1 时抛出。
    """
    value = int(round(_num(mapping, key, default)))
    if value not in (0, 1):
        lookup_key = key[0] if len(key) == 1 else key
        raise ValueError(f"Binary parameter at {lookup_key} must be 0 or 1, got {value}.")
    return value


def _require_keys(mapping: Mapping[Any, Any], keys: Any, name: str) -> None:
    """
    功能：
        校验参数映射是否覆盖所有必需索引。

    参数：
        mapping: 待校验的参数映射。
        keys: 必需索引的可迭代对象，每个索引用元组表示。
        name: 参数名称，用于错误信息。

    返回：
        无。

    异常：
        KeyError: 发现缺失索引时抛出，并在错误信息中展示前若干个缺失键。
    """
    missing = []
    for key_tuple in keys:
        key = key_tuple[0] if len(key_tuple) == 1 else key_tuple
        if key not in mapping:
            missing.append(key)
            if len(missing) >= 10:
                break
    if missing:
        raise KeyError(f"Missing required keys in {name}; first missing keys: {missing}")


def _validate_nonnegative(name: str, values: Mapping[Any, float]) -> None:
    """
    功能：
        校验一组数值参数是否均不小于 0。

    参数：
        name: 参数名称，用于错误信息。
        values: 待校验的参数映射。

    返回：
        无。

    异常：
        ValueError: 任一数值小于容差阈值 ``-1e-6`` 时抛出。
    """
    bad = [(key, value) for key, value in values.items() if value < -1e-6]
    if bad:
        preview = bad[:10]
        raise ValueError(f"{name} contains negative values: {preview}")


def _status_name(status: int) -> str:
    """
    功能：
        将 Gurobi 状态码转换为可读的状态名称。

    参数：
        status: Gurobi 模型状态码。

    返回：
        状态名称字符串；无法识别时返回状态码字符串。
    """
    if GRB is None:
        return str(status)
    names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(status, str(status))


def _safe_model_attr(model: Any, attr: str) -> Optional[float]:
    """
    功能：
        安全读取 Gurobi 模型属性，并尝试转换为浮点数。

    参数：
        model: Gurobi 模型对象或兼容对象。
        attr: 需要读取的属性名。

    返回：
        属性的浮点数值；属性不存在、读取失败或无法转换时返回 ``None``。
    """
    try:
        value = getattr(model, attr)
    except Exception:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_int_tupledict(tupledict: Any, dimensions: tuple[Sequence[Any], ...]) -> Dict[Any, int]:
    """
    功能：
        从 Gurobi tupledict 中提取非零整数解。

    参数：
        tupledict: Gurobi 变量 tupledict。
        dimensions: tupledict 各维度的索引集合，当前支持一维、二维和三维。

    返回：
        非零变量解字典。键为对应索引，值为四舍五入后的整数解。

    异常：
        ValueError: ``dimensions`` 不是一维、二维或三维时由 ``_iter_keys`` 抛出。
    """
    result: Dict[Any, int] = {}
    for key in _iter_keys(dimensions):
        var = tupledict[key]
        value = int(round(var.X))
        if abs(value) > 0:
            result[key] = value
    return result


def _extract_float_tupledict(tupledict: Any, dimensions: tuple[Sequence[Any], ...]) -> Dict[Any, float]:
    """
    功能：
        从 Gurobi tupledict 中提取非零浮点解。

    参数：
        tupledict: Gurobi 变量 tupledict。
        dimensions: tupledict 各维度的索引集合，当前支持一维、二维和三维。

    返回：
        非零变量解字典。键为对应索引，值为浮点解。

    异常：
        ValueError: ``dimensions`` 不是一维、二维或三维时由 ``_iter_keys`` 抛出。
    """
    result: Dict[Any, float] = {}
    for key in _iter_keys(dimensions):
        var = tupledict[key]
        value = float(var.X)
        if abs(value) > 1e-6:
            result[key] = value
    return result


def _iter_keys(dimensions: tuple[Sequence[Any], ...]) -> Any:
    """
    功能：
        根据维度索引集合生成 tupledict 访问键。

    参数：
        dimensions: 一维、二维或三维索引集合组成的元组。

    返回：
        键迭代器。一维时逐个返回单个索引，二维和三维时返回索引元组。

    异常：
        ValueError: 输入维度数量不是 1、2 或 3 时抛出。
    """
    if len(dimensions) == 1:
        for a in dimensions[0]:
            yield a
    elif len(dimensions) == 2:
        for a in dimensions[0]:
            for b in dimensions[1]:
                yield (a, b)
    elif len(dimensions) == 3:
        for a in dimensions[0]:
            for b in dimensions[1]:
                for c in dimensions[2]:
                    yield (a, b, c)
    else:
        raise ValueError("Only 1D, 2D, and 3D tupledict extraction is supported.")


build_daily_rolling_yard_model_gurobi = build_daily_rolling_yard_model
solve_daily_rolling_yard_plan_gurobi = solve_daily_rolling_yard_plan


__all__ = [
    "Area",
    "DailyRollingYardPlanningData",
    "DailyRollingYardPlanningSolution",
    "Flow",
    "Vessel",
    "YardPlanningWeights",
    "build_daily_rolling_yard_model",
    "build_daily_rolling_yard_model_gurobi",
    "solve_daily_rolling_yard_plan",
    "solve_daily_rolling_yard_plan_gurobi",
]


# ---------------------------------------------------------------------------
# SCIP implementation
# ---------------------------------------------------------------------------
#
# The flat SCIP package keeps the public API identical to the Gurobi version:
# run_flat_full_yard_plan.py still imports solve_daily_rolling_yard_plan from
# this module, while the MILP is built and solved with PySCIPOpt.


def _ensure_scip() -> tuple[Any, Any]:
    try:
        from pyscipopt import Model, quicksum
    except ImportError as exc:  # pragma: no cover - handled at runtime.
        raise ImportError(
            "pyscipopt is required to solve the flat_full_yard_plan_scip model. "
            "Install it with: python -m pip install pyscipopt"
        ) from exc
    return Model, quicksum


def build_daily_rolling_yard_model_scip(
    data: DailyRollingYardPlanningData,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
    gurobi_params: Optional[Mapping[str, Any]] = None,
    scip_params: Optional[Mapping[str, Any]] = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    Model, quicksum = _ensure_scip()

    V = _unique_list(data.V, "V")
    F = _unique_list(data.F, "F")
    A = _unique_list(data.A, "A")
    _validate_basic_sets(V, F, A)
    params = _prepare_params(data, V, F, A)

    V_old = params["V_old"]
    OF_area_vessels = params["OF_area_vessels"]
    berth_conflict_pairs = params["berth_conflict_pairs"]
    berth_conflict_keys = params["berth_conflict_keys"]
    required_area_keys = params["required_area_keys"]
    priority_area_keys = params["priority_area_keys"]
    weights = data.weights

    model = Model(data.name)
    if not verbose:
        model.setParam("display/verblevel", 0)
    if time_limit is not None:
        model.setParam("limits/time", float(time_limit))
    if mip_gap is not None:
        model.setParam("limits/gap", float(mip_gap))
    for name, value in (scip_params or {}).items():
        model.setParam(name, value)
    # Keep the old keyword accepted for API compatibility. SCIP parameters can
    # be passed here by advanced callers if they reuse existing call sites.
    for name, value in (gurobi_params or {}).items():
        if "/" in name:
            model.setParam(name, value)

    X20 = {
        (v, f, a): model.addVar(vtype="I", lb=0.0, name=f"X20[{v},{f},{a}]")
        for v in V
        for f in F
        for a in A
    }
    X40 = {
        (v, f, a): model.addVar(vtype="I", lb=0.0, name=f"X40[{v},{f},{a}]")
        for v in V
        for f in F
        for a in A
    }
    y = {(v, a): model.addVar(vtype="B", name=f"y[{v},{a}]") for v in V for a in A}
    h = {a: model.addVar(vtype="I", lb=0.0, name=f"h[{a}]") for a in A}
    o = {a: model.addVar(vtype="C", lb=0.0, name=f"o[{a}]") for a in A}
    r20_pos = {
        (v, f, a): model.addVar(vtype="C", lb=0.0, name=f"r20_pos[{v},{f},{a}]")
        for v in V_old
        for f in F
        for a in A
    }
    r20_neg = {
        (v, f, a): model.addVar(vtype="C", lb=0.0, name=f"r20_neg[{v},{f},{a}]")
        for v in V_old
        for f in F
        for a in A
    }
    r40_pos = {
        (v, f, a): model.addVar(vtype="C", lb=0.0, name=f"r40_pos[{v},{f},{a}]")
        for v in V_old
        for f in F
        for a in A
    }
    r40_neg = {
        (v, f, a): model.addVar(vtype="C", lb=0.0, name=f"r40_neg[{v},{f},{a}]")
        for v in V_old
        for f in F
        for a in A
    }
    m20 = {(v, f): model.addVar(vtype="C", lb=0.0, name=f"m20[{v},{f}]") for v in V for f in F}
    m40 = {(v, f): model.addVar(vtype="C", lb=0.0, name=f"m40[{v},{f}]") for v in V for f in F}
    s20 = {(v, f): model.addVar(vtype="C", lb=0.0, name=f"s20[{v},{f}]") for v in V for f in F}
    s40 = {(v, f): model.addVar(vtype="C", lb=0.0, name=f"s40[{v},{f}]") for v in V for f in F}
    of_area_used = {
        (v, a): model.addVar(vtype="B", name=f"of_area_used[{v},{a}]")
        for v in OF_area_vessels
        for a in A
    }
    of_area_over = {
        v: model.addVar(vtype="C", lb=0.0, name=f"of_area_over[{v}]")
        for v in OF_area_vessels
    }
    berth_conflict_shared = {
        key: model.addVar(vtype="B", name=f"berth_conflict_shared[{key[0]},{key[1]},{key[2]}]")
        for key in berth_conflict_keys
    }
    required_area_unmet = {
        key: model.addVar(vtype="C", lb=0.0, ub=1.0, name=f"required_area_unmet[{key[0]},{key[1]}]")
        for key in required_area_keys
    }
    of_balance_u20 = {
        v: model.addVar(vtype="C", lb=0.0, ub=max(0.0, params["D20"][v, "OF"]), name=f"of_balance_u20[{v}]")
        for v in OF_area_vessels
    }
    of_balance_l20 = {
        v: model.addVar(vtype="C", lb=0.0, ub=max(0.0, params["D20"][v, "OF"]), name=f"of_balance_l20[{v}]")
        for v in OF_area_vessels
    }
    of_balance_u40 = {
        v: model.addVar(vtype="C", lb=0.0, ub=max(0.0, params["D40"][v, "OF"]), name=f"of_balance_u40[{v}]")
        for v in OF_area_vessels
    }
    of_balance_l40 = {
        v: model.addVar(vtype="C", lb=0.0, ub=max(0.0, params["D40"][v, "OF"]), name=f"of_balance_l40[{v}]")
        for v in OF_area_vessels
    }

    for v in V:
        for f in F:
            model.addCons(quicksum(X20[v, f, a] for a in A) + s20[v, f] == params["D20"][v, f])
            model.addCons(quicksum(X40[v, f, a] for a in A) + s40[v, f] == params["D40"][v, f])
            if not data.allow_unmet_demand:
                model.addCons(s20[v, f] == 0.0)
                model.addCons(s40[v, f] == 0.0)

    for v in V:
        for f in F:
            for a in A:
                model.addCons(X20[v, f, a] >= params["S20"][v, f, a])
                model.addCons(X40[v, f, a] >= params["S40"][v, f, a])
                model.addCons(
                    X20[v, f, a] - params["S20"][v, f, a]
                    <= params["R20S"][v, f] * params["U"][a, f]
                )
                model.addCons(
                    X40[v, f, a] - params["S40"][v, f, a]
                    <= params["R40S"][v, f] * params["U"][a, f]
                )
                model.addCons(
                    X20[v, f, a] - params["S20"][v, f, a]
                    <= params["R20S"][v, f] * params["E20"][v, f, a]
                )
                model.addCons(
                    X40[v, f, a] - params["S40"][v, f, a]
                    <= params["R40S"][v, f] * params["E40"][v, f, a]
                )

    for a in A:
        model.addCons(
            quicksum(X20[v, f, a] - params["S20"][v, f, a] for v in V for f in F)
            <= params["C20Direct"][a]
        )
        model.addCons(
            quicksum(X20[v, f, a] - params["S20"][v, f, a] for v in V for f in F)
            + 2.0 * quicksum(X40[v, f, a] - params["S40"][v, f, a] for v in V for f in F)
            <= params["C20"][a]
        )
        model.addCons(
            quicksum(X40[v, f, a] - params["S40"][v, f, a] for v in V for f in F)
            <= params["C40"][a]
        )
        model.addCons(
            quicksum(
                (X20[v, f, a] - params["S20"][v, f, a])
                + (X40[v, f, a] - params["S40"][v, f, a])
                for v in V
                for f in F
            )
            <= params["H"][a] + o[a]
        )
        model.addCons(quicksum(y[v, a] for v in V) <= 2.0 + h[a])

    for v in V:
        for a in A:
            model.addCons(
                quicksum(X20[v, f, a] - params["S20"][v, f, a] for f in F)
                <= params["Cbar20Direct"][v, a]
            )
            model.addCons(
                quicksum(
                    (X20[v, f, a] - params["S20"][v, f, a])
                    + 2.0 * (X40[v, f, a] - params["S40"][v, f, a])
                    for f in F
                )
                <= params["Cbar20"][v, a]
            )
            model.addCons(
                quicksum(X40[v, f, a] - params["S40"][v, f, a] for f in F)
                <= params["Cbar40"][v, a]
            )
            total_plan = quicksum(X20[v, f, a] + X40[v, f, a] for f in F)
            model.addCons(total_plan <= params["M"][v] * y[v, a])
            model.addCons(total_plan >= y[v, a])

    for left, right in berth_conflict_pairs:
        for a in A:
            shared = berth_conflict_shared[left, right, a]
            model.addCons(shared >= y[left, a] + y[right, a] - 1.0)
            model.addCons(shared <= y[left, a])
            model.addCons(shared <= y[right, a])

    for v, a in required_area_keys:
        model.addCons(required_area_unmet[v, a] >= 1.0 - y[v, a])

    if "OF" in F:
        for v in OF_area_vessels:
            for a in A:
                of_total_plan = X20[v, "OF", a] + X40[v, "OF", a]
                model.addCons(of_total_plan <= params["M"][v] * of_area_used[v, a])
                model.addCons(of_total_plan >= of_area_used[v, a])
            model.addCons(
                quicksum(of_area_used[v, a] for a in A)
                <= params["OF_area_limit"][v] + of_area_over[v]
            )

    for v in V_old:
        for f in F:
            for a in A:
                model.addCons(
                    X20[v, f, a] - params["P20"][v, f, a]
                    == r20_pos[v, f, a] - r20_neg[v, f, a]
                )
                model.addCons(
                    X40[v, f, a] - params["P40"][v, f, a]
                    == r40_pos[v, f, a] - r40_neg[v, f, a]
                )

    if "OF" in F:
        for v in OF_area_vessels:
            big_m20 = max(0.0, params["D20"][v, "OF"])
            big_m40 = max(0.0, params["D40"][v, "OF"])
            for a in A:
                model.addCons(X20[v, "OF", a] <= of_balance_u20[v])
                model.addCons(X40[v, "OF", a] <= of_balance_u40[v])
                model.addCons(X20[v, "OF", a] >= of_balance_l20[v] - big_m20 * (1 - of_area_used[v, a]))
                model.addCons(X40[v, "OF", a] >= of_balance_l40[v] - big_m40 * (1 - of_area_used[v, a]))

    Z_miss = quicksum(s20[v, f] + s40[v, f] for v in V for f in F)
    Z_adj = quicksum(
        r20_pos[v, f, a]
        + r20_neg[v, f, a]
        + r40_pos[v, f, a]
        + r40_neg[v, f, a]
        for v in V_old
        for f in F
        for a in A
    )
    distance_scale = max(1.0, max(params["distance"].values()) if params["distance"] else 1.0)
    Z_dist_raw = quicksum(
        params["distance"][v, a] * (X20[v, f, a] + X40[v, f, a])
        for v in V
        for f in F
        for a in A
    )
    Z_dist = quicksum(
        (params["distance"][v, a] / distance_scale) * (X20[v, f, a] + X40[v, f, a])
        for v in V
        for f in F
        for a in A
    )
    Z_share = quicksum(h[a] for a in A)
    Z_berth_conflict = quicksum(berth_conflict_shared[key] for key in berth_conflict_keys)
    Z_required_area = quicksum(required_area_unmet[key] for key in required_area_keys)
    Z_priority_area = quicksum(y[v, a] for v, a in priority_area_keys)
    Z_op = quicksum(o[a] for a in A)
    Z_of_area = quicksum(of_area_over[v] for v in OF_area_vessels)
    Z_bal = quicksum(
        (of_balance_u20[v] - of_balance_l20[v])
        + (of_balance_u40[v] - of_balance_l40[v])
        for v in OF_area_vessels
    )

    model.setObjective(
        weights.miss * Z_miss
        + weights.operation * Z_op
        + weights.of_area * Z_of_area
        + weights.distance * Z_dist
        + weights.share * Z_share
        + weights.berth_conflict * Z_berth_conflict
        + data.required_area_penalty * Z_required_area
        + getattr(weights, "priority_area", 0.01) * Z_priority_area
        + weights.adjustment * Z_adj
        + weights.balance * Z_bal,
        "minimize",
    )

    variables = {
        "X20": X20,
        "X40": X40,
        "y": y,
        "h": h,
        "o": o,
        "r20_pos": r20_pos,
        "r20_neg": r20_neg,
        "r40_pos": r40_pos,
        "r40_neg": r40_neg,
        "m20": m20,
        "m40": m40,
        "s20": s20,
        "s40": s40,
        "of_area_used": of_area_used,
        "of_area_over": of_area_over,
        "berth_conflict_shared": berth_conflict_shared,
        "required_area_unmet": required_area_unmet,
        "objective_components": {
            "miss": Z_miss,
            "operation": Z_op,
            "of_area": Z_of_area,
            "distance": Z_dist,
            "distance_raw": Z_dist_raw,
            "share": Z_share,
            "berth_conflict": Z_berth_conflict,
            "required_area": Z_required_area,
            "priority_area": Z_priority_area,
            "adjustment": Z_adj,
            "balance": Z_bal,
        },
    }
    return model, variables, params


def solve_daily_rolling_yard_plan(
    data: DailyRollingYardPlanningData,
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
    gurobi_params: Optional[Mapping[str, Any]] = None,
    keep_model: bool = False,
) -> DailyRollingYardPlanningSolution:
    model, variables, params = build_daily_rolling_yard_model_scip(
        data,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
        gurobi_params=gurobi_params,
    )
    model.optimize()

    status = str(model.getStatus())
    has_solution = model.getNSols() > 0
    if not has_solution:
        return DailyRollingYardPlanningSolution(
            status=status,
            status_name=status.upper(),
            objective_value=None,
            best_bound=_safe_scip_call(model.getDualbound),
            mip_gap=_safe_scip_call(model.getGap),
            runtime=_safe_scip_call(model.getSolvingTime),
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
    components = variables["objective_components"]

    return DailyRollingYardPlanningSolution(
        status=status,
        status_name=status.upper(),
        objective_value=_safe_scip_call(model.getObjVal),
        best_bound=_safe_scip_call(model.getDualbound),
        mip_gap=_safe_scip_call(model.getGap),
        runtime=_safe_scip_call(model.getSolvingTime),
        x20=_extract_scip_int_vars(model, variables["X20"], (V, F, A)),
        x40=_extract_scip_int_vars(model, variables["X40"], (V, F, A)),
        y=_extract_scip_int_vars(model, variables["y"], (V, A)),
        h=_extract_scip_int_vars(model, variables["h"], (A,)),
        o=_extract_scip_float_vars(model, variables["o"], (A,)),
        of_area_used=_extract_scip_int_vars(model, variables["of_area_used"], (OF_area_vessels, A)),
        of_area_over=_extract_scip_float_vars(model, variables["of_area_over"], (OF_area_vessels,)),
        s20=_extract_scip_float_vars(model, variables["s20"], (V, F)),
        s40=_extract_scip_float_vars(model, variables["s40"], (V, F)),
        m20=_extract_scip_float_vars(model, variables["m20"], (V, F)),
        m40=_extract_scip_float_vars(model, variables["m40"], (V, F)),
        r20_pos=_extract_scip_float_vars(model, variables["r20_pos"], (V_old, F, A)),
        r20_neg=_extract_scip_float_vars(model, variables["r20_neg"], (V_old, F, A)),
        r40_pos=_extract_scip_float_vars(model, variables["r40_pos"], (V_old, F, A)),
        r40_neg=_extract_scip_float_vars(model, variables["r40_neg"], (V_old, F, A)),
        berth_conflict_shared=_extract_scip_int_vars(
            model,
            variables["berth_conflict_shared"],
            (params["berth_conflict_keys"],),
        ),
        required_area_unmet=_extract_scip_float_vars(
            model,
            variables["required_area_unmet"],
            (params["required_area_keys"],),
        ),
        objective_components={name: _safe_scip_expr_value(model, expr) for name, expr in components.items()},
        model=model if keep_model else None,
    )


def _safe_scip_call(func: Any) -> Optional[float]:
    try:
        value = func()
    except Exception:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_scip_expr_value(model: Any, expr: Any) -> float:
    try:
        return float(model.getVal(expr))
    except Exception:
        return 0.0


def _extract_scip_int_vars(model: Any, variables: Mapping[Any, Any], dimensions: tuple[Sequence[Any], ...]) -> Dict[Any, int]:
    result: Dict[Any, int] = {}
    for key in _iter_keys(dimensions):
        try:
            value = int(round(float(model.getVal(variables[key]))))
        except Exception:
            value = 0
        if value:
            result[key] = value
    return result


def _extract_scip_float_vars(model: Any, variables: Mapping[Any, Any], dimensions: tuple[Sequence[Any], ...]) -> Dict[Any, float]:
    result: Dict[Any, float] = {}
    for key in _iter_keys(dimensions):
        try:
            value = float(model.getVal(variables[key]))
        except Exception:
            value = 0.0
        if abs(value) > 1e-6:
            result[key] = value
    return result


build_daily_rolling_yard_model = build_daily_rolling_yard_model_scip
