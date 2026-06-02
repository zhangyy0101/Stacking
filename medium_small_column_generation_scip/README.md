# 中小计划列生成 SCIP 版本

这个目录放一套独立于 `medium_small/block_bay_planning` 的 SCIP 列生成算法。目录内包含一份本地 `block_bay_planning` 副本，用于数据读取、需求计算和 `ProblemData` 构造；列生成入口不会再导入原 `medium_small` 目录里的中小计划代码。

## 口径

- 默认数据目录为仓库根目录的 `堆存计划测试数据20260519` 原始数据目录。
- 默认大计划仍为仓库根目录的 `allocation.csv`。
- 目标航次、OF/OZ 继承池、当前堆场快照过滤、规划窗口等输入口径沿用现有 0519 版本。
- 默认 `--demand-mode original` 对齐原模拟退火+启发式输出口径：
  - `medium_plan.csv` 按原中计划需求 `ProblemData.groups` 输出。
  - `small_plan.csv` 按原小计划资料箱 `ProblemData.small_groups` 输出。
  - 预测兜底组参与列生成优化和中计划预留，但默认不写入 `small_plan.csv`。
- 求解时直接决策资料箱细分组和预测兜底组到箱区、贝位和连续 6 小贝区块的分配。
- 每次运行结束后，`diagnostics.json` 和终端输出都会记录运行时长。

## 需求模式

- `--demand-mode original`：默认模式，中计划箱量对齐原中计划，小计划箱量对齐原资料箱小计划。
- `--demand-mode medium`：总箱量对齐原中计划，资料箱不足的粗属性组用预测兜底组补足，并把兜底组也作为小计划行输出。
- `--demand-mode medium-with-doc-floor`：在 `medium` 基础上保留所有未进场资料箱；若资料箱超过原中计划粗属性组目标，总箱量会高于原中计划。
- `--demand-mode doc-only`：只规划当前未进场资料箱，这是列生成版本早期的口径。

## 列定义

一列表示：

```text
某个资料箱细分组或预测兜底组 -> 某个箱区的某个贝位，固定箱量 q
```

列生成主问题选择这些列，同时满足：

- 同一细分组在同一贝位最多选择一种列模式；
- 同一贝位按物理容量和尺寸容量限制总箱量；
- 不超过 OF/OZ 大计划继承的航次、流向、箱区、尺寸配额；
- 20ft 不放箱区边贝；
- 45ft 必须放箱区边贝；
- 同贝位不混尺寸、不混箱高；
- 当前堆场已有箱的尺寸、箱高约束会被保留；
- 粗属性组低于阈值时强偏好集中到同一箱区，高于阈值时在已使用箱区间尽量均衡并惩罚小碎片箱区；
- 保留总的大计划箱区模式偏离、泊位距离、作业冲突、后续作业偏好；
- 保留小计划的细分组集中、连续 6 小贝区块偏好、已有港口偏好等目标。

## 运行

```powershell
.\.venv\Scripts\python.exe medium_small_column_generation_scip\run_column_generation_planner.py `
  --big-plan allocation.csv
```

如果当前环境没有 PySCIPOpt/SCIP，或希望先检查数据链路，可以先跑贪心回退：

```powershell
.\.venv\Scripts\python.exe medium_small_column_generation_scip\run_column_generation_planner.py `
  --big-plan allocation.csv `
  --no-scip
```

## 外部中计划单独生成小计划

如果中计划已经由外部给定，可以只运行小计划入口。外部中计划需要包含 `voyage_id`、`flow`、`port`、`size`、`area_no`、`planned_boxes` 这些字段；字段名也兼容 `voy_id/status/planned_qty` 等常见别名。程序会把外部中计划解释为粗属性组-箱区硬配额，输出的小计划按粗属性组汇总后不会超过该配额。

```powershell
.\.venv\Scripts\python.exe medium_small_column_generation_scip\run_small_plan_from_medium.py `
  --medium-plan path\to\medium_plan.csv `
  --big-plan allocation.csv
```

该入口固定使用资料箱小计划口径，不生成预测兜底组；`--demand-mode` 参数即使传入也会被覆盖为 `doc-only`。输出目录包含 `small_plan.csv`、`small_plan_six_bay_blocks.csv`、`small_plan_medium_summary.csv`、`external_medium_plan_used.csv`、`medium_plan_big_quota.csv`、`generated_columns.csv` 和 `diagnostics.json`。

## 集中堆存参数

如果细属性组或同箱区内粗属性组仍然太分散，可以调大这些权重：

- `--fine-group-area-penalty`：同一细属性组使用多个箱区的惩罚，默认 `80`。
- `--fine-group-block-penalty`：同一细属性组使用多个连续 6 小贝区块的惩罚，默认 `35`。
- `--fine-group-bay-penalty`：同一细属性组使用更多贝位的固定惩罚，默认 `8`。
- `--coarse-area-block-penalty`：同一粗属性组在同一箱区内使用多个连续 6 小贝区块的惩罚，默认 `24`。
- `--coarse-area-bay-penalty`：同一粗属性组在同一箱区内使用更多贝位的惩罚，默认 `2.5`。
- `--medium-concentrated-group-threshold`：粗属性组箱量小于等于该阈值时走集中堆存目标，大于该阈值时走箱区均衡目标，默认 `26`。
- `--medium-small-group-area-split-penalty`：小量粗属性组使用额外箱区的惩罚，默认 `500`。
- `--medium-small-group-fragment-penalty`：小量粗属性组没有放在最大箱区的碎片箱量惩罚，默认 `20`。
- `--medium-large-group-min-area-boxes`：大量粗属性组选中某个箱区后，低于该箱量会被视为小碎片并惩罚，默认 `10`。
- `--medium-large-group-small-area-penalty`：大量粗属性组小碎片箱区的惩罚，默认 `300`。
