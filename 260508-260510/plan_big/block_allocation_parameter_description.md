# 箱区分配模型参数读取与计算说明（完整版）

本文档用于指导 Codex 将实际数据文件转换为 Gurobi 箱区分配模型所需的参数。

当前模型是一个**事件驱动滚动箱区分配模型**。模型本身只求解某一个规划时刻 `theta` 下的箱区分配问题；时间滚动、参数刷新、历史规划结果更新由外部程序完成。

在每一个规划时刻 `theta`，外部程序需要构造 `YardAllocationData`，然后调用：

```python
result = solve_yard_area_allocation(data)
```

---

## 1. 当前模型需要的参数总览

Gurobi 模型需要以下参数：

| 参数 | 含义 | 来源 |
|---|---|---|
| `V_plus` | 当前时刻需要新增规划的航次集合 | 由航次开港时间和当前规划事件判断 |
| `V_act` | 当前仍占用堆场规划资源的活跃航次集合 | 当前规划航次 + 历史仍有效航次 |
| `A` | 候选箱区集合 | OF 适放箱区列表 - 关闭箱区 |
| `P_same` | 同日开港航次对集合 | 航次开港时间 |
| `R20[v]` | 航次 `v` 当前新增规划的 20 尺箱量 | 航次总需求、规划比例、历史已规划量 |
| `R40[v]` | 航次 `v` 当前新增规划的 40 尺箱量 | 航次总需求、规划比例、历史已规划量 |
| `C20[a]` | 箱区 `a` 当前可用于新规划的 20 尺等价容量 | 实时堆场空位 + 历史预留扣减 |
| `C40[a]` | 箱区 `a` 当前可用于新规划的 40 尺容量 | 实时堆场空位 + 历史预留扣减 |
| `distance[(v,a)]` | 航次 `v` 泊位到箱区 `a` 的距离 | 距离矩阵 + 航次泊位 |
| `B[(v,a)]` | 箱区 `a` 对航次 `v` 是否不可用 | 关闭箱区、OF 适放、容量、装船冲突、TOPS 计划 |
| `H[(v,a)]` | 航次 `v` 在当前时刻之前是否已经选中过箱区 `a` | 历史规划结果 |
| `P20_prev[v]` | 航次 `v` 当前时刻之前已规划的 20 尺箱量 | 历史规划结果 |
| `P40_prev[v]` | 航次 `v` 当前时刻之前已规划的 40 尺箱量 | 历史规划结果 |
| `G_prev[v]` | 航次 `v` 当前时刻之前的累计距离成本 | 历史规划结果 |
| `K_min[v]` | 第一次规划时航次 `v` 推荐箱区数量下界 | 混合有效容量公式估算 |
| `K_max[v]` | 第一次规划时航次 `v` 推荐箱区数量上界 | 混合有效容量公式估算 |
| `first_plan_vessels` | 当前时刻处于第一次规划的航次集合 | 规划事件判断 |
| `followup_plan_vessels` | 当前时刻处于后续规划的航次集合 | 规划事件判断 |

---

## 2. 建议的数据文件

建议项目目录中包含以下数据文件。

```text
data/
├── container_info_453334.parquet
├── bay_slots_detail_20.parquet
├── bay_slots_detail_40.parquet
├── of适放箱区列表.xlsx
├── of_适放箱区_泊位距离矩阵.xlsx
├── vessel_berth_info.csv
├── tops_plan_info.parquet
├── n_usefg_areas.txt
└── plan_history.csv
```

其中：

| 文件 | 用途 |
|---|---|
| `container_info_*.parquet` | 读取各航次的箱量需求 |
| `bay_slots_detail_20.parquet` | 统计箱区 20 尺等价空位 |
| `bay_slots_detail_40.parquet` | 统计箱区 40 尺空位 |
| `of适放箱区列表.xlsx` | 获取 OF 可选箱区集合 |
| `of_适放箱区_泊位距离矩阵.xlsx` | 读取箱区到泊位的距离 |
| `vessel_berth_info.csv` | 读取航次开港时间、泊位信息 |
| `tops_plan_info.parquet` | 判断 TOPS 生效计划占用 |
| `n_usefg_areas.txt` | 读取关闭箱区 |
| `plan_history.csv` | 保存模型滚动规划历史结果 |

`plan_history.csv` 是滚动规划模块自己维护的文件。第一次运行时可以不存在。

---

## 3. 航次总需求 `D20`、`D40` 的读取

### 3.1 数学定义

\[
D_v^{20}
\]

表示航次 \(v\) 的 20 尺箱总需求。

\[
D_v^{40}
\]

表示航次 \(v\) 的 40 尺箱总需求。

航次总自然箱量为：

\[
D_v = D_v^{20}+D_v^{40}
\]

这里的自然箱量不是 TEU。20 尺箱算 1 个自然箱，40 尺箱也算 1 个自然箱。

---

### 3.2 数据来源

来自 `container_info_*.parquet`。

常用字段：

| 字段 | 含义 |
|---|---|
| `IYC_EVOY_ID` | 出口航次号 |
| `IYC_STS_CSTATUSCD` | 箱状态，例如 `OF` |
| `IYC_CSZ_CSIZECD` | 箱尺寸，例如 `20`、`40` |
| `IYC_CNTRID` | 箱 ID |
| `IYC_CNTRNO` | 箱号 |

---

### 3.3 读取逻辑

推荐逻辑：

1. 读取一个或多个 `container_info_*.parquet`；
2. 保留 `IYC_EVOY_ID` 非空的记录；
3. 如果只研究 OF 出口箱，则筛选 `IYC_STS_CSTATUSCD == "OF"`；
4. 统一航次号和箱尺寸字段为字符串；
5. 按航次号和箱尺寸计数；
6. 得到 `D20[v]` 和 `D40[v]`。

---

### 3.4 示例代码

```python
import pandas as pd

def read_vessel_demand(container_files):
    frames = []
    for path in container_files:
        df = pd.read_parquet(path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # 只保留出口航次号非空的记录
    df = df[df["IYC_EVOY_ID"].notna()].copy()

    # 如果只考虑 OF 箱，则进行筛选
    if "IYC_STS_CSTATUSCD" in df.columns:
        df = df[df["IYC_STS_CSTATUSCD"].astype(str).str.upper() == "OF"].copy()

    df["voy_id"] = df["IYC_EVOY_ID"].astype(str)
    df["size"] = df["IYC_CSZ_CSIZECD"].astype(str)

    demand = (
        df.groupby(["voy_id", "size"])
          .size()
          .reset_index(name="qty")
    )

    D20 = {}
    D40 = {}

    for voy_id in demand["voy_id"].unique():
        sub = demand[demand["voy_id"] == voy_id]

        D20[voy_id] = int(
            sub.loc[sub["size"].str.startswith("20"), "qty"].sum()
        )

        D40[voy_id] = int(
            sub.loc[sub["size"].str.startswith("40"), "qty"].sum()
        )

    return D20, D40
```

---

## 4. 航次规划事件与当前新增规划需求 `R20`、`R40`

### 4.1 航次开港时间

设航次 \(v\) 的开港时间为：

\[
T_v^{open}
\]

该时间通常来自 `vessel_berth_info.csv` 中的：

```text
SCD_RCVSTDT
```

即开始进箱时间。

---

### 4.2 规划事件

每个航次生成三个规划事件：

\[
\theta_{v,0}=T_v^{open}-24h
\]

\[
\theta_{v,1}=T_v^{open}+24h
\]

\[
\theta_{v,2}=T_v^{open}+48h
\]

对应累计规划比例：

| 规划事件 | 累计规划比例 |
|---|---:|
| 开港前 24 小时 | 0.7 |
| 开港后 24 小时 | 0.9 |
| 开港后 48 小时 | 1.0 |

---

### 4.3 当前新增规划需求公式

当前时刻 `theta` 下，航次 \(v\) 的 20 尺新增规划需求为：

\[
R_v^{20}(\theta)
=
\max
\left\{
0,\ 
\left\lceil \rho_v(\theta)D_v^{20}\right\rceil
-
P_v^{20}(\theta^-)
\right\}
\]

40 尺新增规划需求为：

\[
R_v^{40}(\theta)
=
\max
\left\{
0,\ 
\left\lceil \rho_v(\theta)D_v^{40}\right\rceil
-
P_v^{40}(\theta^-)
\right\}
\]

其中：

- \(\rho_v(\theta)\)：当前规划事件对应的累计规划比例；
- \(D_v^{20},D_v^{40}\)：航次总需求；
- \(P_v^{20}(\theta^-),P_v^{40}(\theta^-)\)：当前时刻之前已规划箱量。

---

### 4.4 示例

如果：

\[
D_v^{20}=100,\quad D_v^{40}=80
\]

第一次规划：

\[
\rho_v=0.7
\]

若之前没有规划过，则：

\[
R_v^{20}=70
\]

\[
R_v^{40}=56
\]

第二次规划：

\[
\rho_v=0.9
\]

若第一次已经规划 70 个 20 尺、56 个 40 尺，则：

\[
R_v^{20}=90-70=20
\]

\[
R_v^{40}=72-56=16
\]

第三次规划：

\[
\rho_v=1.0
\]

则规划剩余部分。

---

### 4.5 示例代码

```python
import math

def calc_incremental_demand(v, rho, D20, D40, P20_prev, P40_prev):
    target20 = math.ceil(rho * D20.get(v, 0))
    target40 = math.ceil(rho * D40.get(v, 0))

    R20 = max(0, target20 - P20_prev.get(v, 0))
    R40 = max(0, target40 - P40_prev.get(v, 0))

    return R20, R40
```

---

## 5. 已规划箱量 `P20_prev`、`P40_prev`

### 5.1 数学定义

\[
P_v^{20}(\theta^-)
\]

表示当前时刻 `theta` 之前，航次 \(v\) 已规划过的 20 尺箱量。

\[
P_v^{40}(\theta^-)
\]

表示当前时刻 `theta` 之前，航次 \(v\) 已规划过的 40 尺箱量。

---

### 5.2 数据来源

这些参数通常不来自原始业务表，而来自模型求解后保存的历史规划结果 `plan_history.csv`。

建议 `plan_history.csv` 至少包含：

| 字段 | 含义 |
|---|---|
| `event_time` | 规划事件时间 |
| `voy_id` | 航次号 |
| `area_no` | 箱区 |
| `size` | 箱尺寸，`20` 或 `40` |
| `planned_qty` | 本次规划箱量 |
| `distance` | 航次泊位到箱区距离 |
| `stage` | 规划阶段，例如 `first_70`、`second_90`、`third_100` |
| `status` | 规划状态，例如 `planned`、`arrived`、`cancelled` |

---

### 5.3 读取逻辑

统计 `event_time < theta` 的历史规划记录，并排除 `cancelled`。

示例代码：

```python
def read_planned_quantities(plan_history_path, theta):
    if not plan_history_path.exists():
        return {}, {}

    hist = pd.read_csv(plan_history_path)
    hist["event_time"] = pd.to_datetime(hist["event_time"])

    hist = hist[hist["event_time"] < theta].copy()

    if "status" in hist.columns:
        hist = hist[hist["status"].astype(str).str.lower() != "cancelled"].copy()

    P20_prev = {}
    P40_prev = {}

    for voy_id, sub in hist.groupby("voy_id"):
        voy_id = str(voy_id)

        qty20 = sub.loc[
            sub["size"].astype(str).str.startswith("20"),
            "planned_qty"
        ].sum()

        qty40 = sub.loc[
            sub["size"].astype(str).str.startswith("40"),
            "planned_qty"
        ].sum()

        P20_prev[voy_id] = int(qty20)
        P40_prev[voy_id] = int(qty40)

    return P20_prev, P40_prev
```

---

## 6. 箱区当前可用于新规划的 20ft / 40ft 容量 `C20`、`C40`

### 6.1 数学定义

\[
C_a^{20}(\theta)
\]

表示箱区 \(a\) 在时刻 `theta` 可用于新规划的 20 尺等价剩余容量。

\[
C_a^{40}(\theta)
\]

表示箱区 \(a\) 在时刻 `theta` 可用于新规划的 40 尺剩余容量。

---

### 6.2 数据来源

物理空位来自：

```text
bay_slots_detail_20.parquet
bay_slots_detail_40.parquet
```

其中：

- `bay_slots_detail_20.parquet` 用于统计 20 尺等价空位；
- `bay_slots_detail_40.parquet` 用于统计 40 尺空位。

注意：20 尺空位和 40 尺空位可能存在重叠，不能直接相加。

---

### 6.3 字段识别

由于真实字段可能不同，建议写字段推断函数。

箱区字段推断：

```python
def infer_area_col(df):
    candidates = [
        "YardAreaNo",
        "IYC_YAREANO",
        "YARD_AREA_NO",
        "area_no",
        "AREA_NO",
        "YARD_AREA",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError("无法识别箱区字段")
```

空位判断：

```python
def infer_empty_mask(df):
    # 如果存在显式空位标记，优先使用
    for col in ["is_empty", "IS_EMPTY", "empty", "EMPTY_FLAG", "isEmpty"]:
        if col in df.columns:
            return df[col].astype(bool)

    # 如果存在箱 ID 字段，箱 ID 为空通常表示空位
    for col in ["CNTR_ID", "ContainerId", "IYC_CNTRID", "container_id"]:
        if col in df.columns:
            return df[col].isna() | (df[col].astype(str).str.strip() == "")

    # 如果存在箱号字段，箱号为空通常表示空位
    for col in ["CNTR_NO", "ContainerNo", "container_no"]:
        if col in df.columns:
            return df[col].isna() | (df[col].astype(str).str.strip() == "")

    raise ValueError("无法识别空位字段")
```

---

### 6.4 统计物理容量

```python
def count_physical_capacity_by_area(path_20, path_40):
    df20 = pd.read_parquet(path_20)
    df40 = pd.read_parquet(path_40)

    area_col_20 = infer_area_col(df20)
    area_col_40 = infer_area_col(df40)

    empty20 = infer_empty_mask(df20)
    empty40 = infer_empty_mask(df40)

    physical20 = (
        df20.loc[empty20]
            .groupby(area_col_20)
            .size()
            .to_dict()
    )

    physical40 = (
        df40.loc[empty40]
            .groupby(area_col_40)
            .size()
            .to_dict()
    )

    physical20 = {str(k): int(v) for k, v in physical20.items()}
    physical40 = {str(k): int(v) for k, v in physical40.items()}

    return physical20, physical40
```

---

### 6.5 扣除历史规划预留容量

当前可用于新规划的容量不只是物理空位，还要扣除历史规划中尚未实际进场、但已经预留的容量。

20 尺等价容量扣减公式：

\[
C_a^{20}(\theta)
=
\widehat C_a^{20}(\theta)
-
\sum_{\text{历史预留}}x_{v,a}^{20}
-
2\sum_{\text{历史预留}}x_{v,a}^{40}
\]

40 尺容量扣减公式：

\[
C_a^{40}(\theta)
=
\widehat C_a^{40}(\theta)
-
\sum_{\text{历史预留}}x_{v,a}^{40}
\]

其中：

- \(\widehat C_a^{20}(\theta)\)：物理 20 尺等价空位；
- \(\widehat C_a^{40}(\theta)\)：物理 40 尺空位。

示例代码：

```python
def read_reserved_capacity_by_area(plan_history_path, theta):
    reserve20_equiv = {}
    reserve40 = {}

    if not plan_history_path.exists():
        return reserve20_equiv, reserve40

    hist = pd.read_csv(plan_history_path)
    hist["event_time"] = pd.to_datetime(hist["event_time"])
    hist = hist[hist["event_time"] < theta].copy()

    if "status" in hist.columns:
        # 只扣除 planned，避免 arrived 被实时堆场和历史预留重复扣除
        hist = hist[hist["status"].astype(str).str.lower() == "planned"].copy()

    for area, sub in hist.groupby("area_no"):
        area = str(area)

        qty20 = sub.loc[
            sub["size"].astype(str).str.startswith("20"),
            "planned_qty"
        ].sum()

        qty40 = sub.loc[
            sub["size"].astype(str).str.startswith("40"),
            "planned_qty"
        ].sum()

        reserve20_equiv[area] = int(qty20 + 2 * qty40)
        reserve40[area] = int(qty40)

    return reserve20_equiv, reserve40
```

最终容量计算：

```python
def calc_available_capacity(physical20, physical40, reserve20_equiv, reserve40, candidate_areas):
    C20 = {}
    C40 = {}

    for a in candidate_areas:
        a = str(a)

        C20[a] = max(
            0,
            int(physical20.get(a, 0)) - int(reserve20_equiv.get(a, 0))
        )

        C40[a] = max(
            0,
            int(physical40.get(a, 0)) - int(reserve40.get(a, 0))
        )

    return C20, C40
```

---

## 7. 航次泊位到所有箱区的距离 `distance[(v,a)]`

### 7.1 数学定义

\[
d_{v,a}
\]

表示航次 \(v\) 的靠泊泊位到箱区 \(a\) 的距离。

---

### 7.2 数据来源

距离来自：

```text
of_适放箱区_泊位距离矩阵.xlsx
```

航次泊位来自：

```text
vessel_berth_info.csv
```

相关字段：

| 字段 | 含义 |
|---|---|
| `VOY_ID` | 航次号 |
| `VBT_BTH_PBTHNO` | 预计靠泊泊位号 |
| `VBT_BTH_ABTHNO` | 实际靠泊泊位号 |

---

### 7.3 泊位选择逻辑

优先使用实际泊位：

\[
b_v =
\begin{cases}
\text{实际泊位}, & \text{如果实际泊位存在} \\
\text{预计泊位}, & \text{否则}
\end{cases}
\]

示例代码：

```python
def read_vessel_berth(vessel_berth_info_path):
    df = pd.read_csv(vessel_berth_info_path)

    berth = {}

    for _, row in df.iterrows():
        voy_id = str(row["VOY_ID"])

        actual = row.get("VBT_BTH_ABTHNO", None)
        planned = row.get("VBT_BTH_PBTHNO", None)

        if pd.notna(actual) and str(actual).strip() != "":
            berth[voy_id] = str(actual).strip()
        else:
            berth[voy_id] = str(planned).strip()

    return berth
```

---

### 7.4 距离矩阵读取

距离矩阵可能是：

| area_no | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|

也可能是：

| 箱区 | 泊位1 | 泊位2 | 泊位3 |
|---|---:|---:|---:|

建议先打印列名，根据实际格式做兼容。

示例代码：

```python
def infer_distance_area_col(df):
    candidates = ["area_no", "AreaNo", "YardAreaNo", "箱区", "AREA_NO"]
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def read_distance_param(distance_matrix_path, vessel_berth, V_act, candidate_areas):
    dist_df = pd.read_excel(distance_matrix_path)
    area_col = infer_distance_area_col(dist_df)

    dist_df[area_col] = dist_df[area_col].astype(str)
    dist_df = dist_df.set_index(area_col)

    distance = {}

    for v in V_act:
        berth_no = str(vessel_berth[v])

        possible_cols = [
            berth_no,
            int(berth_no) if berth_no.isdigit() else berth_no,
            f"泊位{berth_no}",
            f"berth_{berth_no}",
            f"B{berth_no}",
        ]

        col = None
        for c in possible_cols:
            if c in dist_df.columns:
                col = c
                break

        if col is None:
            raise KeyError(f"距离矩阵中找不到泊位 {berth_no} 对应的列")

        for a in candidate_areas:
            a = str(a)
            distance[(v, a)] = float(dist_df.loc[a, col])

    return distance
```

---

## 8. 箱区是否可用 `B[(v,a)]`

### 8.1 数学定义

\[
B_{v,a}(\theta)
=
\begin{cases}
1, & \text{箱区 }a\text{ 对航次 }v\text{ 当前不可用} \\
0, & \text{箱区 }a\text{ 对航次 }v\text{ 当前可用}
\end{cases}
\]

---

### 8.2 判断规则

如果满足以下任意一个条件，就令：

\[
B_{v,a}(\theta)=1
\]

1. 箱区 \(a\) 不在 OF 适放箱区列表中；
2. 箱区 \(a\) 是关闭箱区；
3. 箱区 \(a\) 当前容量不足；
4. 箱区 \(a\) 在航次 \(v\) 开港当天处于装船任务时间段；
5. 箱区 \(a\) 被其他 TOPS 生效计划占用；
6. 其他业务规则认为箱区 \(a\) 不适合航次 \(v\)。

---

### 8.3 读取 OF 适放箱区

```python
def read_of_area_list(path):
    df = pd.read_excel(path)

    if "area_no" in df.columns:
        col = "area_no"
    else:
        col = df.columns[0]

    return set(df[col].astype(str).str.strip())
```

---

### 8.4 读取关闭箱区

```python
import ast
from pathlib import Path

def read_closed_areas(path):
    text = Path(path).read_text(encoding="utf-8").strip()

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple, set)):
            return set(str(x).strip() for x in obj)
    except Exception:
        pass

    parts = []
    for line in text.splitlines():
        parts.extend(line.split(","))

    return set(p.strip() for p in parts if p.strip())
```

如果没有关闭箱区文件，可以先写死：

```python
closed_areas = {"20", "25", "4F"}
```

---

### 8.5 容量不足判断

第一版建议用保守规则：

```python
def capacity_unavailable(v, a, R20, R40, C20, C40):
    # 没有 20 尺等价空间，则不可用
    if C20.get(a, 0) <= 0:
        return True

    # 如果当前航次有 40 尺新增需求，但该箱区没有 40 尺容量，则不可用
    if R40.get(v, 0) > 0 and C40.get(a, 0) <= 0:
        return True

    return False
```

---

### 8.6 装船任务冲突判断

规则：

> 开港前一天规划的区域，需要避开该航次开港当天处于装船任务时间段的箱区。

对航次 \(v\)，取得开港日期：

\[
date(T_v^{open})
\]

若箱区 \(a\) 的装船任务时间区间与该日期有重叠，则认为冲突。

时间区间重叠函数：

```python
def intervals_overlap(start1, end1, start2, end2):
    return start1 < end2 and start2 < end1
```

装船冲突函数：

```python
def has_loading_conflict_on_open_day(v, a, open_time, loading_tasks):
    open_day_start = pd.Timestamp(open_time).normalize()
    open_day_end = open_day_start + pd.Timedelta(days=1)

    area_tasks = loading_tasks[
        loading_tasks["area_no"].astype(str) == str(a)
    ].copy()

    for _, row in area_tasks.iterrows():
        task_start = pd.Timestamp(row["start_time"])
        task_end = pd.Timestamp(row["end_time"])

        if intervals_overlap(task_start, task_end, open_day_start, open_day_end):
            return True

    return False
```

---

### 8.7 TOPS 生效计划占用判断

TOPS 计划来自：

```text
tops_plan_info.parquet
```

相关字段可能包括：

| 字段 | 含义 |
|---|---|
| `SPL_YPLANID` | 计划号 |
| `SPL_TYPE` | 进出口标志 |
| `SPL_CONDITIONCODE` | 航次号 |
| `SPR_STBAY` | 起始箱区贝位或区域 |
| `SPR_EDBAY` | 结束箱区贝位或区域 |
| `SPL_STDATE` | 起始生效时间 |
| `SPL_EDDATE` | 终止生效时间 |

第一版可以先实现简单逻辑：

1. 找出在当前时刻 `theta` 生效的 TOPS 计划；
2. 排除当前接管航次自己的计划；
3. 若该计划占用箱区 \(a\)，则令该箱区不可用。

注意：`SPR_STBAY`、`SPR_EDBAY` 是否能直接映射到箱区，需要根据真实字段进一步确认。

---

### 8.8 构造 `B`

```python
def build_unavailable_param(
    V_plus,
    V_act,
    A,
    of_areas,
    closed_areas,
    R20,
    R40,
    C20,
    C40,
    vessel_open_time,
    loading_tasks=None,
    tops_df=None,
):
    B = {}

    for v in V_act:
        for a in A:
            unavailable = False

            if str(a) not in of_areas:
                unavailable = True

            if str(a) in closed_areas:
                unavailable = True

            if v in V_plus:
                if capacity_unavailable(v, a, R20, R40, C20, C40):
                    unavailable = True

            if loading_tasks is not None and v in vessel_open_time:
                if has_loading_conflict_on_open_day(
                    v=v,
                    a=a,
                    open_time=vessel_open_time[v],
                    loading_tasks=loading_tasks,
                ):
                    unavailable = True

            # TOPS 计划冲突可以在确认字段后打开
            # if tops_df is not None:
            #     if has_active_tops_conflict(a, theta, tops_df, current_voy_id=v):
            #         unavailable = True

            B[(v, a)] = 1 if unavailable else 0

    return B
```

---

## 9. 历史已选箱区 `H[(v,a)]`

### 9.1 数学定义

\[
H_{v,a}(\theta^-)
=
\begin{cases}
1, & \text{航次 }v\text{ 在当前时刻之前已经选中过箱区 }a \\
0, & \text{否则}
\end{cases}
\]

---

### 9.2 来源

来自 `plan_history.csv`。

判断逻辑：

只要航次 \(v\) 在当前时刻 `theta` 之前曾经向箱区 \(a\) 分配过正箱量，并且记录没有取消，则：

\[
H_{v,a}(\theta^-)=1
\]

---

### 9.3 读取代码

```python
def read_history_area_param(plan_history_path, theta, V_act, A):
    H = {(v, a): 0 for v in V_act for a in A}

    if not plan_history_path.exists():
        return H

    hist = pd.read_csv(plan_history_path)
    hist["event_time"] = pd.to_datetime(hist["event_time"])
    hist = hist[hist["event_time"] < theta].copy()

    if "status" in hist.columns:
        hist = hist[hist["status"].astype(str).str.lower() != "cancelled"].copy()

    for _, row in hist.iterrows():
        v = str(row["voy_id"])
        a = str(row["area_no"])
        qty = float(row["planned_qty"])

        if v in V_act and a in A and qty > 0:
            H[(v, a)] = 1

    return H
```

---

## 10. 推荐箱区数量 `K_min`、`K_max` 的估算

### 10.1 目标

第一次规划时，模型需要控制航次主箱区数量：

\[
K_v^{min}
\leq
\sum_{a\in\mathcal A}w_{v,a}
\leq
K_v^{max}
\]

这里：

- \(K_v^{min}\)：推荐箱区数量下界；
- \(K_v^{max}\)：推荐箱区数量上界。

---

### 10.2 为什么使用总箱量估算

第一次规划只分配 70% 箱量，但第一次选出的主箱区后续还要承接剩余 30%。

因此估算箱区数量时，应使用航次总箱量：

\[
D_v = D_v^{20}+D_v^{40}
\]

而不是当前新增箱量：

\[
R_v^{20}(\theta)+R_v^{40}(\theta)
\]

---

### 10.3 箱型比例

40 尺比例：

\[
p_v^{40}
=
\frac{D_v^{40}}{D_v^{20}+D_v^{40}}
\]

20 尺比例：

\[
p_v^{20}
=
1-p_v^{40}
\]

---

### 10.4 箱区混合有效容量

对于箱区 \(a\)，定义对航次 \(v\) 的混合有效容量：

\[
C_{v,a}^{mix}
=
\min
\left\{
\frac{C_a^{20}(\theta)}{p_v^{20}+2p_v^{40}},
\frac{C_a^{40}(\theta)}{p_v^{40}}
\right\}
\]

解释：

如果在箱区 \(a\) 放 \(N\) 个该航次的自然箱，则：

- 大约有 \(p_v^{20}N\) 个 20 尺箱；
- 大约有 \(p_v^{40}N\) 个 40 尺箱。

20 尺等价容量消耗为：

\[
p_v^{20}N+2p_v^{40}N
=
(p_v^{20}+2p_v^{40})N
\]

所以：

\[
N
\leq
\frac{C_a^{20}(\theta)}{p_v^{20}+2p_v^{40}}
\]

同时，40 尺容量要求：

\[
p_v^{40}N\leq C_a^{40}(\theta)
\]

所以：

\[
N
\leq
\frac{C_a^{40}(\theta)}{p_v^{40}}
\]

两个约束必须同时满足，因此取较小值。

---

### 10.5 特殊情况

如果航次全是 20 尺箱：

\[
p_v^{40}=0
\]

则：

\[
C_{v,a}^{mix}=C_a^{20}(\theta)
\]

如果航次全是 40 尺箱：

\[
p_v^{20}=0,\quad p_v^{40}=1
\]

则：

\[
C_{v,a}^{mix}
=
\min
\left\{
\frac{C_a^{20}(\theta)}{2},
C_a^{40}(\theta)
\right\}
\]

---

### 10.6 可用箱区集合

\[
\mathcal A_v
=
\{a\in\mathcal A:B_{v,a}(\theta)=0\}
\]

只在可用箱区集合上计算平均有效容量。

---

### 10.7 平均有效容量

\[
\bar C_v
=
\frac{1}{|\mathcal A_v|}
\sum_{a\in\mathcal A_v}
C_{v,a}^{mix}
\]

---

### 10.8 推荐箱区数

\[
K_v^0
=
\left\lceil
\frac{D_v}{\bar C_v}
\right\rceil
\]

上下界：

\[
K_v^{min}
=
\max\{1,K_v^0-1\}
\]

\[
K_v^{max}
=
K_v^0+1
\]

同时：

\[
K_v^{max}
=
\min\{K_v^{max},|\mathcal A_v|\}
\]

若出现：

\[
K_v^{min}>K_v^{max}
\]

可以令：

\[
K_v^{min}=K_v^{max}
\]

---

### 10.9 示例代码

```python
import math

def calc_mixed_capacity_for_vessel(v, a, D20, D40, C20, C40):
    total = D20.get(v, 0) + D40.get(v, 0)
    if total <= 0:
        return 0.0

    p40 = D40.get(v, 0) / total
    p20 = 1.0 - p40

    if p40 == 0:
        return float(C20.get(a, 0))

    if p20 == 0:
        return min(
            C20.get(a, 0) / 2.0,
            C40.get(a, 0)
        )

    cap_by_20_equiv = C20.get(a, 0) / (p20 + 2 * p40)
    cap_by_40 = C40.get(a, 0) / p40

    return min(cap_by_20_equiv, cap_by_40)


def estimate_area_count_bounds(v, D20, D40, C20, C40, B, A):
    total_demand = D20.get(v, 0) + D40.get(v, 0)

    if total_demand <= 0:
        return 0, 0, 0

    available_areas = [
        a for a in A
        if B.get((v, a), 1) == 0
    ]

    if not available_areas:
        raise ValueError(f"航次 {v} 没有可用箱区，无法估算 K")

    mixed_caps = [
        calc_mixed_capacity_for_vessel(v, a, D20, D40, C20, C40)
        for a in available_areas
    ]

    positive_caps = [c for c in mixed_caps if c > 0]

    if not positive_caps:
        raise ValueError(f"航次 {v} 的可用箱区容量均为 0，无法估算 K")

    avg_cap = sum(positive_caps) / len(positive_caps)

    K0 = math.ceil(total_demand / avg_cap)

    K_min = max(1, K0 - 1)
    K_max = K0 + 1

    K_max = min(K_max, len(available_areas))

    if K_min > K_max:
        K_min = K_max

    return K0, K_min, K_max
```

---

## 11. 累计距离成本 `G_prev`

### 11.1 数学定义

\[
G_v(\theta^-)
\]

表示当前时刻 `theta` 之前，航次 \(v\) 已规划箱量的累计距离成本。

---

### 11.2 计算公式

\[
G_v(\theta^-)
=
\sum_{\text{历史规划记录}}
d_{v,a}
\cdot
\text{planned\_qty}_{v,a}
\]

其中：

\[
\text{planned\_qty}_{v,a}
=
x_{v,a}^{20}+x_{v,a}^{40}
\]

即使用自然箱数量。

---

### 11.3 示例

如果历史规划中：

| 航次 | 箱区 | 箱量 | 距离 |
|---|---|---:|---:|
| 453334 | A01 | 50 | 10 |
| 453334 | A02 | 40 | 12 |
| 453334 | A03 | 20 | 18 |

则：

\[
G_{453334}
=
50\times 10
+
40\times 12
+
20\times 18
=
1340
\]

---

### 11.4 读取代码：历史表中有 distance 字段

```python
def read_previous_distance_cost(plan_history_path, theta):
    G_prev = {}

    if not plan_history_path.exists():
        return G_prev

    hist = pd.read_csv(plan_history_path)
    hist["event_time"] = pd.to_datetime(hist["event_time"])
    hist = hist[hist["event_time"] < theta].copy()

    if "status" in hist.columns:
        hist = hist[hist["status"].astype(str).str.lower() != "cancelled"].copy()

    hist["distance_cost"] = hist["distance"] * hist["planned_qty"]

    for voy_id, sub in hist.groupby("voy_id"):
        G_prev[str(voy_id)] = float(sub["distance_cost"].sum())

    return G_prev
```

---

### 11.5 读取代码：历史表中没有 distance 字段

如果历史表没有保存距离，则用 `distance[(v,a)]` 重新计算：

```python
def read_previous_distance_cost_by_distance_param(plan_history_path, theta, distance):
    if not plan_history_path.exists():
        return {}

    hist = pd.read_csv(plan_history_path)
    hist["event_time"] = pd.to_datetime(hist["event_time"])
    hist = hist[hist["event_time"] < theta].copy()

    if "status" in hist.columns:
        hist = hist[hist["status"].astype(str).str.lower() != "cancelled"].copy()

    G_prev = {}

    for _, row in hist.iterrows():
        v = str(row["voy_id"])
        a = str(row["area_no"])
        qty = float(row["planned_qty"])

        d = distance[(v, a)]
        G_prev[v] = G_prev.get(v, 0.0) + d * qty

    return G_prev
```

---

### 11.6 求解后更新

求解完成后：

\[
G_v(\theta)
=
G_v(\theta^-)
+
\sum_{a\in\mathcal A}
d_{v,a}
\left(
x_{v,a}^{20}+x_{v,a}^{40}
\right)
\]

示例代码：

```python
def update_distance_cost_after_solve(G_prev, result, distance):
    G_new = dict(G_prev)

    x20 = result["x20"]
    x40 = result["x40"]

    all_keys = set(x20.keys()) | set(x40.keys())

    for v, a in all_keys:
        qty = x20.get((v, a), 0) + x40.get((v, a), 0)
        G_new[v] = G_new.get(v, 0.0) + distance[(v, a)] * qty

    return G_new
```

---

## 12. 同日开港航次对 `P_same`

### 12.1 数学定义

\[
\mathcal P^{same}(\theta)
\]

表示当前活跃航次中，同一天开港的航次对集合。

如果航次 \(u\) 和航次 \(v\) 的开港日期相同，则：

\[
(u,v)\in \mathcal P^{same}(\theta)
\]

---

### 12.2 示例代码

```python
from itertools import combinations

def build_same_day_pairs(V_act, vessel_open_time):
    P_same = []

    for u, v in combinations(V_act, 2):
        day_u = pd.Timestamp(vessel_open_time[u]).date()
        day_v = pd.Timestamp(vessel_open_time[v]).date()

        if day_u == day_v:
            P_same.append((u, v))

    return P_same
```

---

## 13. 活跃航次集合 `V_act`

### 13.1 定义

\[
\mathcal V^{act}(\theta)
\]

表示当前时刻仍占用堆场规划资源的航次集合。

它至少包括：

1. 当前新增规划航次 `V_plus`；
2. 历史上已经规划过箱区、但箱子尚未全部离场的航次；
3. 后续仍可能补充规划的航次。

---

### 13.2 简化实现

第一版可以将以下航次加入 `V_act`：

```text
V_act = 当前新增规划航次 + plan_history 中 status != cancelled 的航次
```

如果能够判断航次已经离港且箱子已经不再占用堆场，可以从 `V_act` 中移除。

---

### 13.3 示例代码

```python
def build_active_vessels(V_plus, plan_history_path, theta):
    V_act = set(V_plus)

    if plan_history_path.exists():
        hist = pd.read_csv(plan_history_path)
        hist["event_time"] = pd.to_datetime(hist["event_time"])
        hist = hist[hist["event_time"] < theta].copy()

        if "status" in hist.columns:
            hist = hist[hist["status"].astype(str).str.lower() != "cancelled"].copy()

        V_act.update(hist["voy_id"].astype(str).unique().tolist())

    return sorted(V_act)
```

---

## 14. 推荐的参数构造总流程

在某个规划时刻 `theta`，建议按以下顺序构造参数：

```text
1. 读取 vessel_berth_info.csv
   - 获得每个航次的开港时间 T_open
   - 获得每个航次的泊位 berth

2. 根据 theta 判断当前规划事件
   - 生成 V_plus
   - 判断 first_plan_vessels
   - 判断 followup_plan_vessels
   - 得到每个 V_plus 航次对应的 rho

3. 读取航次总需求
   - D20[v]
   - D40[v]

4. 读取历史规划结果
   - P20_prev[v]
   - P40_prev[v]
   - H[(v,a)]
   - G_prev[v]

5. 构造活跃航次集合
   - V_act

6. 读取候选箱区
   - OF 适放箱区
   - 关闭箱区
   - 得到 A

7. 读取并计算当前箱区容量
   - physical20[a]
   - physical40[a]
   - reserve20_equiv[a]
   - reserve40[a]
   - C20[a]
   - C40[a]

8. 计算当前新增规划需求
   - R20[v]
   - R40[v]

9. 读取泊位距离参数
   - distance[(v,a)]

10. 构造不可用参数
    - B[(v,a)]

11. 对第一次规划航次估算箱区数量
    - K0[v]
    - K_min[v]
    - K_max[v]

12. 构造同日开港航次对
    - P_same

13. 组装 YardAllocationData
    - 调用 Gurobi 求解模型
```

---

## 15. 组装 `YardAllocationData` 示例

```python
data = YardAllocationData(
    V_plus=V_plus,
    V_act=V_act,
    A=A,
    P_same=P_same,

    R20=R20,
    R40=R40,

    C20=C20,
    C40=C40,

    distance=distance,
    B=B,
    H=H,

    P20_prev=P20_prev,
    P40_prev=P40_prev,
    G_prev=G_prev,

    first_plan_vessels=first_plan_vessels,
    followup_plan_vessels=followup_plan_vessels,

    K_min=K_min,
    K_max=K_max,

    lambda_dist=1.0,
    lambda_fair=1.0,
    lambda_frag=1.0,
    lambda_same=10.0,
    lambda_over=10.0,
    M_new=1000.0,
    M_miss=1_000_000.0,

    allow_unmet_demand=True,
)
```

---

## 16. 求解后需要更新的历史状态

模型求解后，外部程序需要保存并更新历史状态。

---

### 16.1 更新已规划箱量

\[
P_v^{20}(\theta)
=
P_v^{20}(\theta^-)
+
\sum_{a\in\mathcal A}x_{v,a}^{20}
\]

\[
P_v^{40}(\theta)
=
P_v^{40}(\theta^-)
+
\sum_{a\in\mathcal A}x_{v,a}^{40}
\]

---

### 16.2 更新历史箱区集合

\[
H_{v,a}(\theta)=w_{v,a}
\]

---

### 16.3 更新累计距离成本

\[
G_v(\theta)
=
G_v(\theta^-)
+
\sum_{a\in\mathcal A}
d_{v,a}(x_{v,a}^{20}+x_{v,a}^{40})
\]

---

### 16.4 更新容量

如果当前规划结果作为预留写入历史表，则下一轮计算容量时会自动通过 `plan_history.csv` 扣除。

建议将本次规划结果追加写入 `plan_history.csv`，每一行表示一个航次、箱区、尺寸的规划结果。

---

### 16.5 推荐历史记录格式

```text
event_time,voy_id,area_no,size,planned_qty,distance,stage,status
2026-05-07 10:00:00,453334,A01,20,50,10,first_70,planned
2026-05-07 10:00:00,453334,A01,40,30,10,first_70,planned
```

---

## 17. Codex 实现注意事项

### 17.1 字段名不要完全写死

真实数据字段名可能有差异。建议写推断函数：

- `infer_area_col(df)`
- `infer_empty_mask(df)`
- `infer_distance_area_col(df)`

---

### 17.2 不要重复扣减容量

如果历史规划的箱子已经实际进场，而实时堆场已经体现占用，就不要再从历史预留中扣除。

建议通过 `status` 区分：

| status | 是否扣除历史预留 |
|---|---|
| `planned` | 是 |
| `arrived` | 否 |
| `cancelled` | 否 |

---

### 17.3 第一次规划和后续规划必须区分

第一次规划：

```python
first_plan_vessels = [...]
followup_plan_vessels = []
```

后续规划：

```python
first_plan_vessels = []
followup_plan_vessels = [...]
```

如果同一时刻同时有第一次规划航次和后续规划航次，则两个列表都可以非空，但不能重叠。

---

### 17.4 估算箱区数必须用总箱量

估算 \(K_v\) 时使用：

\[
D_v=D_v^{20}+D_v^{40}
\]

不要使用当前新增规划量：

\[
R_v^{20}+R_v^{40}
\]

---

### 17.5 `B` 参数可以先保守构造

如果某个箱区是否可用不确定，第一版可以先设为不可用。若模型频繁出现未满足需求，再逐步放宽。

---

### 17.6 20/40 容量约束必须同时保留

Gurobi 模型中必须保留：

\[
\sum_v x_{v,a}^{40}\leq C_a^{40}
\]

和：

\[
\sum_v x_{v,a}^{20}+2\sum_v x_{v,a}^{40}\leq C_a^{20}
\]

不能只保留其中一条。

---

## 18. 最终摘要

完整参数读取流程可以概括为：

```text
原始数据
  -> 航次总需求 D20/D40
  -> 历史规划状态 P20_prev/P40_prev/H/G_prev
  -> 实时容量 C20/C40
  -> 泊位距离 distance
  -> 不可用箱区 B
  -> 推荐箱区数量 K_min/K_max
  -> YardAllocationData
  -> Gurobi 模型求解
  -> 更新 plan_history
  -> 下一规划事件继续求解
```

该文档的目标是让 Codex 明确：**Gurobi 模型只负责单个规划时刻的优化，所有时间滚动、参数更新、历史状态维护都应该在模型外部完成。**
