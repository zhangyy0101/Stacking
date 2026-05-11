"""
浜嬩欢椹卞姩婊氬姩绠卞尯鍒嗛厤妯″瀷锛欸urobi 姹傝В妗嗘灦

璇存槑锛?1. 鏈唬鐮佸彧瀹炵幇鍗曚釜瑙勫垝鏃跺埢 theta 涓嬬殑 MILP 姹傝В妯″瀷銆?2. 鏃堕棿婊氬姩銆佽鍙栧疄鏃跺爢鍦恒€佹洿鏂板閲忋€佹洿鏂板巻鍙茬鍖虹瓑閫昏緫鏆傛椂涓嶆斁鍏ユ湰鏂囦欢銆?3. 妯″瀷鍖哄垎绗竴娆¤鍒掍笌鍚庣画瑙勫垝锛?   - 绗竴娆¤鍒掞細闇€瑕佹帶鍒惰埅娆′富绠卞尯鏁伴噺 K_min <= sum_a w[v,a] <= K_max銆?   - 鍚庣画瑙勫垝锛氬敖閲忔部鐢ㄥ巻鍙茬鍖?H[v,a]锛岃嫢鍚敤鏂扮鍖哄垯閫氳繃 n[v,a] 璁″叆鎯╃綒銆?4. 妯″瀷涓嶈€冭檻绠卞睘鎬х粍锛屼絾鍖哄垎 20 灏哄拰 40 灏虹锛屽洜涓轰簩鑰呭搴斾笉鍚岀墿鐞嗗閲忕害鏉熴€?"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB


# =========================
# 1. 绫诲瀷鍒悕
# =========================

Vessel = str          # 鑸缂栧彿锛屼緥濡?"453334"
Area = str            # 绠卞尯缂栧彿锛屼緥濡?"12"銆?4F"
VesselPair = Tuple[Vessel, Vessel]


# =========================
# 2. 鍙傛暟瀹瑰櫒
# =========================

@dataclass
class YardAllocationData:
    """
    鍗曚釜瑙勫垝鏃跺埢 theta 涓嬬殑妯″瀷杈撳叆鍙傛暟銆?    娉ㄦ剰锛氳繖浜涘弬鏁板簲鐢卞閮ㄦ暟鎹澶勭悊鍜屾粴鍔ㄦ洿鏂版ā鍧楁彁鍓嶈绠楀ソ銆?    """

    # 褰撳墠瑙勫垝鏃跺埢闇€瑕佹柊澧炶鍒掔殑鑸闆嗗悎 V+(theta)
    V_plus: List[Vessel]

    # 褰撳墠瑙勫垝鏃跺埢浠嶇劧鍗犵敤鍫嗗満瑙勫垝璧勬簮鐨勬椿璺冭埅娆￠泦鍚?V_act(theta)
    # 璇ラ泦鍚堢敤浜庣粺璁℃瘡涓鍖哄綋鍓嶆湇鍔′簡鍑犱釜鑸銆?
    V_act: List[Vessel]

    # 鍊欓€夌鍖洪泦鍚?A
    A: List[Area]

    # 鍚屾棩寮€娓埅娆″闆嗗悎 P_same(theta)
    # 渚嬪 [("453334", "453335"), ("453334", "453336")]
    P_same: List[VesselPair]

    # 褰撳墠鏂板瑙勫垝闇€姹傦細20 灏虹 R_v^20(theta)
    R20: Dict[Vessel, int]

    # 褰撳墠鏂板瑙勫垝闇€姹傦細40 灏虹 R_v^40(theta)
    R40: Dict[Vessel, int]

    # 褰撳墠绠卞尯 20 灏虹瓑浠峰墿浣欏閲?C_a^20(theta)
    C20: Dict[Area, int]

    # 褰撳墠绠卞尯 40 灏哄墿浣欏閲?C_a^40(theta)
    C40: Dict[Area, int]

    # 鑸 v 鍒扮鍖?a 鐨勮窛绂?d_{v,a}
    distance: Dict[Tuple[Vessel, Area], float]

    # 绠卞尯涓嶅彲鐢ㄥ弬鏁?B_{v,a}(theta)
    # B[v,a] = 1 琛ㄧず绠卞尯 a 瀵硅埅娆?v 褰撳墠涓嶅彲鐢紱B[v,a] = 0 琛ㄧず鍙敤銆?
    B: Dict[Tuple[Vessel, Area], int]

    # 鍘嗗彶宸查€夌鍖哄弬鏁?H_{v,a}(theta^-)
    # H[v,a] = 1 琛ㄧず鑸 v 鍦ㄥ綋鍓嶆椂鍒讳箣鍓嶅凡缁忎娇鐢ㄨ繃绠卞尯 a銆?
    H: Dict[Tuple[Vessel, Area], int]

    # 褰撳墠鏃跺埢涔嬪墠宸茶鍒?20 灏虹閲?P_v^20(theta^-)
    P20_prev: Dict[Vessel, int]

    # 褰撳墠鏃跺埢涔嬪墠宸茶鍒?40 灏虹閲?P_v^40(theta^-)
    P40_prev: Dict[Vessel, int]

    # 褰撳墠鏃跺埢涔嬪墠绱璺濈鎴愭湰 G_v(theta^-)
    G_prev: Dict[Vessel, float]

    # 绗竴娆¤鍒掕埅娆￠泦鍚堬細杩欎簺鑸闇€瑕佹柦鍔犵鍖烘暟閲忎笂涓嬬晫绾︽潫銆?
    first_plan_vessels: List[Vessel] = field(default_factory=list)

    # 鍚庣画瑙勫垝鑸闆嗗悎锛氳繖浜涜埅娆￠渶瑕佹柦鍔犫€滃敖閲忎笉鏂板紑绠卞尯鈥濈殑閫昏緫涓庢儵缃氥€?
    followup_plan_vessels: List[Vessel] = field(default_factory=list)

    # 绗竴娆¤鍒掓椂锛屾瘡涓埅娆℃帹鑽愮鍖烘暟閲忎笅鐣?K_v^min
    K_min: Dict[Vessel, int] = field(default_factory=dict)

    # 绗竴娆¤鍒掓椂锛屾瘡涓埅娆℃帹鑽愮鍖烘暟閲忎笂鐣?K_v^max
    K_max: Dict[Vessel, int] = field(default_factory=dict)

    # 鐩爣鍑芥暟鏉冮噸锛氭€昏窛绂绘垚鏈?
    lambda_dist: float = 1.0

    # 鐩爣鍑芥暟鏉冮噸锛氭渶澶х疮璁″钩鍧囪窛绂诲叕骞抽」 eta
    lambda_fair: float = 1.0

    # 鐩爣鍑芥暟鏉冮噸锛氬綋鍓嶆柊澧炲垎閰嶇鐗囧寲椤?sum z[v,a]
    lambda_frag: float = 1.0

    # 鐩爣鍑芥暟鏉冮噸锛氬悓鏃ュ紑娓埅娆″叡鐢ㄧ鍖烘儵缃氶」 sum q[u,v,a]
    lambda_same: float = 10.0

    # 鐩爣鍑芥暟鏉冮噸锛氱鍖烘湇鍔¤秴杩?2 涓埅娆＄殑鎯╃綒椤?sum h[a]
    lambda_over: float = 10.0

    # 鐩爣鍑芥暟鏉冮噸锛氬悗缁鍒掍腑鏂板紑绠卞尯鎯╃綒椤?sum n[v,a]
    M_new: float = 1000.0

    # 鐩爣鍑芥暟鏉冮噸锛氭湭婊¤冻闇€姹傛儵缃氶」 sum s20[v] + s40[v]
    M_miss: float = 1_000_000.0

    # 鏄惁鍏佽鏈弧瓒抽渶姹傘€傚疄闄呮祴璇曟椂寤鸿鍏堣涓?True锛岄伩鍏嶆ā鍨嬬洿鎺?infeasible銆?
    allow_unmet_demand: bool = True


# =========================
# 3. 鏁版嵁瀹屾暣鎬ф鏌?# =========================

def validate_input(data: YardAllocationData) -> None:
    """
    瀵规ā鍨嬭緭鍏ヨ繘琛屽熀纭€妫€鏌ャ€?    杩欓噷涓嶅仛澶嶆潅涓氬姟鏍￠獙锛屽彧妫€鏌ュ叧閿储寮曟槸鍚﹀瓨鍦ㄣ€?    """

    V_plus = set(data.V_plus)   # 褰撳墠瑙勫垝鑸
    V_act = set(data.V_act)
    A = set(data.A)

    missing_active = V_plus - V_act
    if missing_active:
        raise ValueError(f"V_plus 涓瓨鍦ㄤ笉灞炰簬 V_act 鐨勮埅娆? {missing_active}")

    # 妫€鏌ラ渶姹傚弬鏁般€?
    for v in data.V_plus:
        if v not in data.R20:
            raise KeyError(f"缂哄皯鑸 {v} 鐨?R20 鍙傛暟")
        if v not in data.R40:
            raise KeyError(f"缂哄皯鑸 {v} 鐨?R40 鍙傛暟")

    # 妫€鏌ュ閲忓弬鏁般€?
    for a in data.A:
        if a not in data.C20:
            raise KeyError(f"缂哄皯绠卞尯 {a} 鐨?C20 鍙傛暟")
        if a not in data.C40:
            raise KeyError(f"缂哄皯绠卞尯 {a} 鐨?C40 鍙傛暟")

    # 妫€鏌ヨ窛绂汇€佷笉鍙敤鍙傛暟銆佸巻鍙茬鍖哄弬鏁般€?
    for v in data.V_act:
        for a in data.A:
            if (v, a) not in data.distance:
                raise KeyError(f"缂哄皯璺濈鍙傛暟 distance[{v}, {a}]")
            if (v, a) not in data.B:
                raise KeyError(f"缂哄皯涓嶅彲鐢ㄥ弬鏁?B[{v}, {a}]")
            if (v, a) not in data.H:
                raise KeyError(f"缂哄皯鍘嗗彶绠卞尯鍙傛暟 H[{v}, {a}]")

    # 妫€鏌ュ巻鍙茬疮璁″弬鏁般€?
    for v in data.V_act:
        if v not in data.P20_prev:
            raise KeyError(f"缂哄皯鑸 {v} 鐨?P20_prev 鍙傛暟")
        if v not in data.P40_prev:
            raise KeyError(f"缂哄皯鑸 {v} 鐨?P40_prev 鍙傛暟")
        if v not in data.G_prev:
            raise KeyError(f"缂哄皯鑸 {v} 鐨?G_prev 鍙傛暟")

    # 绗竴娆¤鍒掕埅娆￠渶瑕佹湁 K_min 鍜?K_max銆?
    for v in data.first_plan_vessels:
        if v not in data.K_min:
            raise KeyError(f"绗竴娆¤鍒掕埅娆?{v} 缂哄皯 K_min 鍙傛暟")
        if v not in data.K_max:
            raise KeyError(f"绗竴娆¤鍒掕埅娆?{v} 缂哄皯 K_max 鍙傛暟")
        if data.K_min[v] > data.K_max[v]:
            raise ValueError(f"鑸 {v} 鐨?K_min 澶т簬 K_max")

    # 鍚屾棩寮€娓埅娆″闇€瑕佸睘浜庢椿璺冭埅娆￠泦鍚堛€?
    for u, v in data.P_same:
        if u not in V_act or v not in V_act:
            raise ValueError(f"鍚屾棩寮€娓埅娆″ {(u, v)} 涓瓨鍦ㄩ潪娲昏穬鑸")


# =========================
# 4. Gurobi 妯″瀷鏋勫缓涓庢眰瑙?# =========================

def solve_yard_area_allocation(
    data: YardAllocationData,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    鏋勫缓骞舵眰瑙ｅ崟涓鍒掓椂鍒?theta 涓嬬殑绠卞尯鍒嗛厤妯″瀷銆?
    杩斿洖鍊煎寘鍚細
    - status: Gurobi 姹傝В鐘舵€?    - objective_value: 鐩爣鍑芥暟鍊?    - x20, x40: 20/40 灏虹鍒嗛厤缁撴灉
    - z: 褰撳墠瑙勫垝鏄惁浣跨敤绠卞尯
    - w: 褰撳墠瑙勫垝瀹屾垚鍚庤埅娆℃€讳綋鏄惁鍗犵敤绠卞尯
    - n: 鍚庣画瑙勫垝涓槸鍚︽柊寮€绠卞尯
    - q: 鍚屾棩寮€娓埅娆℃槸鍚﹀叡鐢ㄧ鍖?    - h: 绠卞尯鏈嶅姟鑸鏁拌秴棰濋噺
    - eta: 鏈€澶х疮璁″钩鍧囪窛绂?    - s20, s40: 鏈弧瓒抽渶姹傞噺
    """

    validate_input(data)

    V_plus = data.V_plus    # 鏂板瑙勫垝鑸
    V_act = data.V_act      # 娲昏穬鑸
    A = data.A              # 绠卞尯
    P_same = data.P_same    # 鍚屾棩寮€娓殑鑸瀵?
    # 鍒ゆ柇鍝簺鑸褰撳墠鍒嗗埆灞炰簬鍝釜闃舵
    first_set = set(data.first_plan_vessels)
    followup_set = set(data.followup_plan_vessels)
    # 创建 Gurobi 模型。
    model = gp.Model("event_driven_yard_area_allocation")

    # Gurobi鍙傛暟璁剧疆
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # =========================
    # 4.1 鍐崇瓥鍙橀噺瀹氫箟
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

    # z[v,a]：当前规划中航次 v 是否向箱区 a 分配了新增箱量。
    z = model.addVars(
        V_plus,
        A,
        vtype=GRB.BINARY,
        name="z_current_use",
    )

    # w[v,a]：当前规划完成后，航次 v 整体计划中是否使用箱区 a。
    w = model.addVars(
        V_act,
        A,
        vtype=GRB.BINARY,
        name="w_total_use",
    )

    # n[v,a]：后续规划中航次 v 是否新开箱区 a。
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

    # s20[v]、s40[v]：未满足的 20/40 尺新增规划需求。
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
    # 4.2 闇€姹傛弧瓒崇害鏉?    # =========================

    # 绾︽潫 1锛?0 灏虹鏂板瑙勫垝闇€姹傛弧瓒炽€?    # 瀵规瘡涓綋鍓嶉渶瑕佽鍒掔殑鑸 v锛屽垎閰嶅埌鎵€鏈夌鍖虹殑 20 灏虹閲忓姞涓婃湭婊¤冻閲忥紝绛変簬 R20[v]銆?
    for v in V_plus:
        model.addConstr(
            gp.quicksum(x20[v, a] for a in A) + s20[v] == data.R20[v],
            name=f"demand_20[{v}]",
        )

    # 绾︽潫 2锛?0 灏虹鏂板瑙勫垝闇€姹傛弧瓒炽€?    # 瀵规瘡涓綋鍓嶉渶瑕佽鍒掔殑鑸 v锛屽垎閰嶅埌鎵€鏈夌鍖虹殑 40 灏虹閲忓姞涓婃湭婊¤冻閲忥紝绛変簬 R40[v]銆?
    for v in V_plus:
        model.addConstr(
            gp.quicksum(x40[v, a] for a in A) + s40[v] == data.R40[v],
            name=f"demand_40[{v}]",
        )

    # 鑻ヤ笉鍏佽鏈弧瓒抽渶姹傦紝鍒欏皢鏉惧紱鍙橀噺鍥哄畾涓?0銆?
    if not data.allow_unmet_demand:
        for v in V_plus:
            model.addConstr(s20[v] == 0, name=f"no_unmet_20[{v}]")
            model.addConstr(s40[v] == 0, name=f"no_unmet_40[{v}]")

    # =========================
    # 4.3 绠卞尯瀹归噺绾︽潫
    # =========================

    # 绾︽潫 3锛?0 灏虹瀹归噺绾︽潫銆?    # 褰撳墠鏂板瑙勫垝鍒嗛厤鍒扮鍖?a 鐨?40 灏虹鎬绘暟锛屼笉鑳借秴杩?C40[a]銆?
    for a in A:
        model.addConstr(
            gp.quicksum(x40[v, a] for v in V_plus) <= data.C40[a],
            name=f"capacity_40[{a}]",
        )

    # 绾︽潫 4锛?0 灏虹瓑浠峰閲忕害鏉熴€?    # 20 灏虹鍗犵敤 1 涓?20 灏虹瓑浠蜂綅缃紱40 灏虹鍗犵敤 2 涓?20 灏虹瓑浠蜂綅缃€?    # 鍥犳 x20 + 2*x40 涓嶈兘瓒呰繃 C20[a]銆?
    for a in A:
        model.addConstr(
            gp.quicksum(x20[v, a] for v in V_plus)
            + 2 * gp.quicksum(x40[v, a] for v in V_plus)
            <= data.C20[a],
            name=f"capacity_20_equivalent[{a}]",
        )

    # =========================
    # 4.4 褰撳墠鍒嗛厤鍙橀噺涓庡綋鍓嶄娇鐢ㄥ彉閲?z 鐨勯€昏緫鑱斿姩
    # =========================

    # 绾︽潫 5锛氬鏋?z[v,a] = 0锛屽垯鑸 v 褰撳墠涓嶈兘寰€绠卞尯 a 鍒嗛厤浠讳綍绠卞瓙銆?    # M_v 鍙栧綋鍓嶈埅娆?v 鐨勬柊澧炶鍒掓€婚噺 R20[v] + R40[v]銆?
    for v in V_plus:
        M_v = data.R20[v] + data.R40[v]
        for a in A:
            model.addConstr(
                x20[v, a] + x40[v, a] <= M_v * z[v, a],
                name=f"link_upper_x_z[{v},{a}]",
            )

    # 绾︽潫 6锛氬鏋?z[v,a] = 1锛屽垯鑸 v 褰撳墠鑷冲皯瑕佸線绠卞尯 a 鍒嗛厤 1 涓瀛愩€?    # 杩欏彧鏄熀鏈€昏緫绾︽潫锛屼笉鏄崟绠卞尯鏈€灏忓垎閰嶉噺瑙勫垯銆?
    for v in V_plus:
        for a in A:
            model.addConstr(
                x20[v, a] + x40[v, a] >= z[v, a],
                name=f"link_lower_x_z[{v},{a}]",
            )

    # =========================
    # 4.5 涓嶅彲鐢ㄧ鍖虹害鏉?    # =========================

    # 绾︽潫 7锛氬鏋?B[v,a] = 1锛岃鏄庣鍖?a 瀵硅埅娆?v 褰撳墠涓嶅彲鐢紝鍒?z[v,a] 蹇呴』涓?0銆?    # 涓嶅彲鐢ㄥ彲鑳芥潵鑷叧闂鍖恒€侀潪 OF 閫傛斁銆佸紑娓綋澶╄鑸瑰啿绐併€乀OPS 璁″垝鍗犵敤绛夈€?
    for v in V_plus:
        for a in A:
            model.addConstr(
                z[v, a] <= 1 - data.B[v, a],
                name=f"unavailable_area[{v},{a}]",
            )

    # =========================
    # 4.6 鎬讳綋浣跨敤鍙橀噺 w 涓庡巻鍙?H銆佸綋鍓?z 鐨勫叧绯?    # =========================

    # 瀵瑰綋鍓嶆柊澧炶鍒掕埅娆★紝w[v,a] = max(H[v,a], z[v,a])銆?    # 绾︽潫 8.1锛氬鏋滃巻鍙插凡缁忛€夎繃绠卞尯 a锛屽垯褰撳墠瑙勫垝瀹屾垚鍚?w[v,a] 蹇呴』涓?1銆?
    for v in V_plus:
        for a in A:
            model.addConstr(
                w[v, a] >= data.H[v, a],
                name=f"w_ge_history[{v},{a}]",
            )

    # 绾︽潫 8.2锛氬鏋滃綋鍓嶆柊澧炶鍒掍娇鐢ㄧ鍖?a锛屽垯褰撳墠瑙勫垝瀹屾垚鍚?w[v,a] 蹇呴』涓?1銆?
    for v in V_plus:
        for a in A:
            model.addConstr(
                w[v, a] >= z[v, a],
                name=f"w_ge_current[{v},{a}]",
            )

    # 绾︽潫 8.3锛氬鏋滃巻鍙叉病鐢ㄣ€佸綋鍓嶄篃娌＄敤锛屽垯 w[v,a] 蹇呴』涓?0銆?
    for v in V_plus:
        for a in A:
            model.addConstr(
                w[v, a] <= data.H[v, a] + z[v, a],
                name=f"w_le_history_plus_current[{v},{a}]",
            )

    # 瀵归潪褰撳墠鏂板瑙勫垝浣嗕粛鐒舵椿璺冪殑鑸锛屼笉鍦ㄦ湰娆℃ā鍨嬩腑鏂板鍒嗛厤绠遍噺銆?    # 鍥犳鍏舵€讳綋浣跨敤鎯呭喌 w[v,a] 鐩存帴鍥哄畾涓哄巻鍙插€?H[v,a]銆?
    for v in V_act:
        if v not in V_plus:
            for a in A:
                model.addConstr(
                    w[v, a] == data.H[v, a],
                    name=f"inactive_w_fixed_to_history[{v},{a}]",
                )

    # =========================
    # 4.7 绗竴娆¤鍒掍笓鐢ㄧ害鏉燂細涓荤鍖烘暟閲忔帶鍒?    # =========================

    # 绾︽潫 9锛氱涓€娆¤鍒掓椂锛岃埅娆?v 浣跨敤鐨勪富绠卞尯鏁伴噺搴斿湪 K_min[v] 鍜?K_max[v] 涔嬮棿銆?    # 杩欓噷鎺у埗鐨勬槸褰撳墠瑙勫垝瀹屾垚鍚庣殑鎬讳綋绠卞尯闆嗗悎 w[v,a]锛屼笉鏄粎褰撳墠鏂板鍒嗛厤鐨?z[v,a]銆?
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
    # 4.8 鍚庣画瑙勫垝涓撶敤绾︽潫锛氭柊寮€绠卞尯璇嗗埆
    # =========================

    # 瀵圭涓€娆¤鍒掕埅娆★紝灏?n[v,a] 鍥哄畾涓?0銆?    # 绗竴娆¤鍒掓湰韬氨鏄湪寤虹珛涓荤鍖洪泦鍚堬紝涓嶅簲璁′负鈥滄柊寮€绠卞尯鎯╃綒鈥濄€?
    for v in V_plus:
        if v in first_set:
            for a in A:
                model.addConstr(
                    n[v, a] == 0,
                    name=f"first_plan_no_new_penalty[{v},{a}]",
                )

    # 瀵瑰悗缁鍒掕埅娆★紝n[v,a] = 1 琛ㄧず绠卞尯 a 姝ゅ墠鏈鑸 v 浣跨敤锛屼絾褰撳墠瑙勫垝瀹屾垚鍚庤鍚敤銆?    # 绾︽潫 10.1锛氬綋 w=1 涓?H=0 鏃讹紝寮哄埗 n=1銆?
    for v in followup_set:
        if v not in V_plus:
            continue
        for a in A:
            model.addConstr(
                n[v, a] >= w[v, a] - data.H[v, a],
                name=f"new_area_lb[{v},{a}]",
            )

    # 绾︽潫 10.2锛歯[v,a] 涓嶈兘澶т簬 w[v,a]銆?
    for v in followup_set:
        if v not in V_plus:
            continue
        for a in A:
            model.addConstr(
                n[v, a] <= w[v, a],
                name=f"new_area_le_w[{v},{a}]",
            )

    # 绾︽潫 10.3锛氬鏋?H[v,a] = 1锛岃鏄庣鍖?a 鍘嗗彶涓婂凡缁忚鑸 v 浣跨敤杩囷紝鍒?n[v,a] 蹇呴』涓?0銆?
    for v in followup_set:
        if v not in V_plus:
            continue
        for a in A:
            model.addConstr(
                n[v, a] <= 1 - data.H[v, a],
                name=f"new_area_le_not_history[{v},{a}]",
            )

    # 瀵规棦涓嶆槸绗竴娆¤鍒掋€佷篃涓嶆槸鍚庣画瑙勫垝鐨勮埅娆★紝淇濋櫓璧疯鍥哄畾 n=0銆?    # 姝ｅ父鎯呭喌涓?V_plus 搴旇琚?first_set 鍜?followup_set 瑕嗙洊銆?
    for v in V_plus:
        if v not in first_set and v not in followup_set:
            for a in A:
                model.addConstr(
                    n[v, a] == 0,
                    name=f"undefined_stage_n_fixed[{v},{a}]",
                )

    # =========================
    # 4.9 鍚屾棩寮€娓埅娆￠伩璁╃害鏉?    # =========================

    # 绾︽潫 11锛氬悓鏃ュ紑娓埅娆?u 鍜?v 灏介噺涓嶈鍏辩敤绠卞尯 a銆?    # 鑻ヤ簩鑰呴兘浣跨敤绠卞尯 a锛屽垯 q[u,v,a] 蹇呴』涓?1锛屽苟鍦ㄧ洰鏍囧嚱鏁颁腑琚儵缃氥€?
    for (u, v) in P_same:
        for a in A:
            model.addConstr(
                w[u, a] + w[v, a] <= 1 + q[u, v, a],
                name=f"same_day_share[{u},{v},{a}]",
            )

    # =========================
    # 4.10 鍗曠鍖烘湇鍔¤埅娆℃暟闄愬埗
    # =========================

    # 绾︽潫 12锛氫竴涓鍖烘渶濂戒笉瑕佹湇鍔¤秴杩?2 涓椿璺冭埅娆°€?    # 鑻ヨ秴杩?2 涓紝鍒欓€氳繃 h[a] 琛ㄧず瓒呴鏁伴噺锛屽苟鍦ㄧ洰鏍囧嚱鏁颁腑鎯╃綒銆?
    for a in A:
        model.addConstr(
            gp.quicksum(w[v, a] for v in V_act) <= 2 + h[a],
            name=f"over_two_vessels[{a}]",
        )

    # =========================
    # 4.11 绱骞冲潎璺濈鍏钩绾︽潫
    # =========================

    # 绾︽潫 13锛氭瘡涓椿璺冭埅娆＄殑绱骞冲潎璺濈涓嶈秴杩?eta銆?    # 瀵瑰綋鍓嶆柊澧炶鍒掕埅娆★紝绱璺濈 = 鍘嗗彶绱璺濈 + 褰撳墠鏂板鍒嗛厤璺濈銆?
    for v in V_plus:
        total_planned_after = (
            data.P20_prev[v]
            + data.P40_prev[v]
            + data.R20[v]
            + data.R40[v]
        )

        # 鑻ヨ鑸褰撳墠瑙勫垝鍚庝粛鏃犺鍒掔閲忥紝鍒欎笉鏂藉姞骞冲潎璺濈绾︽潫锛岄伩鍏嶅彸绔负 0銆?
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

    # 瀵归潪褰撳墠鏂板瑙勫垝浣嗕粛鐒舵椿璺冪殑鑸锛岀疮璁¤窛绂诲拰绱绠遍噺鍧囦负鍘嗗彶鍊笺€?    # 鑻ュ叾鍘嗗彶骞冲潎璺濈宸茬粡杈冨ぇ锛屼篃搴旇 eta 瑕嗙洊锛屼繚璇?eta 琛ㄧず鎵€鏈夋椿璺冭埅娆′腑鐨勬渶澶х疮璁″钩鍧囪窛绂汇€?
    for v in V_act:
        if v not in V_plus:
            total_planned_history = data.P20_prev[v] + data.P40_prev[v]
            if total_planned_history > 0:
                model.addConstr(
                    data.G_prev[v] <= eta * total_planned_history,
                    name=f"average_distance_active_history[{v}]",
                )

    # =========================
    # 4.12 鐩爣鍑芥暟
    # =========================

    Z_dist = gp.quicksum(
        data.distance[v, a] * (x20[v, a] + x40[v, a])
        for v in V_plus
        for a in A
    )

    Z_fair = eta

    Z_frag = gp.quicksum(z[v, a] for v in V_plus for a in A)

    Z_same = gp.quicksum(q[u, v, a] for (u, v) in P_same for a in A)

    Z_over = gp.quicksum(h[a] for a in A)

    Z_new = gp.quicksum(n[v, a] for v in V_plus for a in A)

    Z_miss = gp.quicksum(s20[v] + s40[v] for v in V_plus)

    # 缁煎悎鐩爣鍑芥暟銆?    # 寤鸿鏉冮噸浼樺厛绾э細M_miss > M_new > lambda_same/lambda_over > lambda_dist/lambda_fair/lambda_frag銆?
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

    # 姹傝В妯″瀷銆?
    model.optimize()

    # =========================
    # 4.13 缁撴灉鎻愬彇
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

        # 鍙鍑洪潪闆跺垎閰嶇粨鏋滐紝鏂逛究鏌ョ湅銆?
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
# 5. 绀轰緥锛氱涓€娆¤鍒?# =========================
