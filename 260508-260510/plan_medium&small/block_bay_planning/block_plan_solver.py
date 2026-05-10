from __future__ import annotations

import math
from collections import Counter, defaultdict

from .models import Unit


class BlockPlanSolverMixin:
    """中计划求解逻辑：把属性箱组分配到区块。

    本模块只负责“箱区内部选哪些区块”。大计划已经决定了航次可以使用哪些
    箱区以及各箱区箱量，因此这里的硬约束是：
    1. 不能跳出大计划选定箱区；
    2. 航次-箱区总箱量必须等于大计划；
    3. 如果大计划提供了尺寸拆分，则航次-箱区-20/40 箱量也必须等于大计划；
    4. 区块容量不能超出由贝位容量汇总得到的能力。
    """

    def _solve_medium(self) -> list[str]:
        """用模拟退火搜索中计划。

        `assignment[unit_id] = block_id`，表示某个单位箱被分配到哪个区块。
        初始解先严格满足大计划配额，随后邻域动作只在可行解之间移动。
        """
        assignment = self._initial_medium_assignment()
        energy = self.medium_energy(assignment)
        best_assignment = assignment[:]
        best_energy = energy

        for it in range(1, self.config.iterations + 1):
            candidate = assignment[:]
            if not self._mutate_medium(candidate):
                continue
            candidate_energy = self.medium_energy(candidate)
            delta = candidate_energy - energy
            temperature = self._temperature(it)
            if delta <= 0 or self.random.random() < math.exp(-delta / max(temperature, 1e-9)):
                assignment = candidate
                energy = candidate_energy
                if energy < best_energy:
                    best_energy = energy
                    best_assignment = assignment[:]
        return best_assignment

    def _temperature(self, iteration: int) -> float:
        ratio = iteration / max(1, self.config.iterations)
        return self.config.initial_temperature * (
            self.config.final_temperature / self.config.initial_temperature
        ) ** ratio

    def _initial_medium_assignment(self) -> list[str]:
        """构造满足大计划配额的中计划初始解。

        如果大计划有 20/40 尺寸输出，则先按航次和尺寸分桶，再分别把每个
        尺寸桶分配到大计划指定的箱区；45 尺箱使用 `size_mode` 归入 40 尺桶。
        """
        assignment = ["" for _ in self.units]
        block_load: Counter[str] = Counter()
        block_size_load: Counter[tuple[str, str]] = Counter()
        units_by_voyage: defaultdict[str, list[Unit]] = defaultdict(list)
        for unit in self.units:
            units_by_voyage[unit.group.voyage_id].append(unit)

        for voyage_id, units in units_by_voyage.items():
            units.sort(key=lambda u: (u.group.port, u.group.operator, u.group.size_mode, u.group.height))
            if self.problem.area_size_quota:
                self._assign_voyage_units_by_size_quota(
                    voyage_id,
                    units,
                    assignment,
                    block_load,
                    block_size_load,
                )
                continue

            quotas = {
                area: qty
                for (v, area), qty in self.problem.area_quota.items()
                if v == voyage_id and qty > 0
            }
            if sum(quotas.values()) != len(units):
                raise ValueError(f"big-plan quota for voyage {voyage_id} does not match demand units")
            cursor = 0
            for area_no, qty in quotas.items():
                for unit in units[cursor : cursor + qty]:
                    block_id = self._best_initial_block(unit, area_no, block_load, block_size_load)
                    if block_id is None:
                        raise ValueError(
                            f"no feasible block for voyage={voyage_id}, group={unit.group.group_id}, area={area_no}"
                        )
                    assignment[unit.unit_id] = block_id
                    block_load[block_id] += 1
                    block_size_load[(block_id, unit.group.size_mode)] += 1
                cursor += qty
        return assignment

    def _assign_voyage_units_by_size_quota(
        self,
        voyage_id: str,
        units: list[Unit],
        assignment: list[str],
        block_load: Counter[str],
        block_size_load: Counter[tuple[str, str]],
    ) -> None:
        """按大计划尺寸配额分配单个航次。

        例如大计划给出 `453334-12-20=30`、`453334-12-40=40`，
        那么中计划必须保证航次 453334 在箱区 12 中正好放 30 个大计划
        20 尺口径箱和 40 个大计划 40 尺口径箱。45 尺箱属于后者。
        """
        units_by_big_size: defaultdict[str, list[Unit]] = defaultdict(list)
        for unit in units:
            units_by_big_size[unit.group.big_plan_size_mode].append(unit)

        for size_mode in ("20", "40"):
            size_units = units_by_big_size.get(size_mode, [])
            quotas = {
                area: qty
                for (v, area, size), qty in self.problem.area_size_quota.items()
                if v == voyage_id and size == size_mode and qty > 0
            }
            if not quotas and not size_units:
                continue
            if sum(quotas.values()) != len(size_units):
                raise ValueError(
                    f"big-plan {size_mode} quota for voyage {voyage_id} "
                    "does not match demand units"
                )

            cursor = 0
            for area_no, qty in quotas.items():
                for unit in size_units[cursor : cursor + qty]:
                    block_id = self._best_initial_block(unit, area_no, block_load, block_size_load)
                    if block_id is None:
                        area_cap = self._area_size_capacity(area_no, unit.group.size_mode)
                        area_load = self._area_size_load(area_no, unit.group.size_mode, block_size_load)
                        raise ValueError(
                            f"no feasible block for voyage={voyage_id}, "
                            f"big_plan_size={size_mode}, real_size={unit.group.size_mode}, "
                            f"group={unit.group.group_id}, area={area_no}; "
                            f"recognized_area_capacity={area_cap}, current_area_load={area_load}, "
                            f"big_plan_area_size_quota={qty}"
                        )
                    assignment[unit.unit_id] = block_id
                    block_load[block_id] += 1
                    block_size_load[(block_id, unit.group.size_mode)] += 1
                cursor += qty

    def _best_initial_block(
        self,
        unit: Unit,
        area_no: str,
        block_load: Counter[str],
        block_size_load: Counter[tuple[str, str]],
    ) -> str | None:
        """在指定箱区内选择一个当前最合适的区块。

        这里同时检查区块总容量和 20/40 尺寸容量。45 尺箱按 40 尺容量检查。
        """
        best_id = None
        best_score = float("inf")
        for block_id in self.blocks_by_area.get(area_no, []):
            if block_load[block_id] + 1 > self.block_total_cap[block_id]:
                continue
            if block_size_load[(block_id, unit.group.size_mode)] + 1 > self.block_size_cap[(block_id, unit.group.size_mode)]:
                continue
            compatible_capacity = self._block_compatible_capacity(
                block_id,
                unit.group.size_mode,
                unit.group.height,
            )
            score = block_load[block_id] * 0.1 + self._block_index(block_id) * 0.01
            if compatible_capacity <= block_size_load[(block_id, unit.group.size_mode)]:
                score += 100.0
            if score < best_score:
                best_score = score
                best_id = block_id
        return best_id

    def _area_size_capacity(self, area_no: str, real_size: str) -> int:
        """统计某箱区中指定真实尺寸的区块容量。"""
        return sum(
            self.block_size_cap[(block_id, real_size)]
            for block_id in self.blocks_by_area.get(area_no, [])
        )

    def _area_size_load(
        self,
        area_no: str,
        real_size: str,
        block_size_load: Counter[tuple[str, str]],
    ) -> int:
        """统计当前初始解构造过程中某箱区已占用的指定真实尺寸容量。"""
        return sum(
            block_size_load[(block_id, real_size)]
            for block_id in self.blocks_by_area.get(area_no, [])
        )

    def _block_compatible_capacity(self, block_id: str, size_mode: str, height: str) -> int:
        """估算区块内某尺寸/高度属性较理想的贝位容量。

        这个值只用于初始解打分：如果某区块虽然 20/40 总容量足够，但同高度
        贝位明显不足，就给它加惩罚，优先把箱子放去更容易落贝的区块。高度
        本身不是硬约束，最终仍允许通过小计划软惩罚处理。
        """
        total = 0
        for bay_key in self.bays_by_block.get(block_id, []):
            bay = self.bays[bay_key]
            if bay.locked:
                continue
            if size_mode == "20" and self._is_area_edge_bay(bay_key):
                continue
            if height != "UNK" and bay.existing_heights and bay.existing_heights != {height}:
                continue
            total += min(bay.physical_capacity, bay.cap_by_size.get(size_mode, 0))
        return total

    def _mutate_medium(self, assignment: list[str]) -> bool:
        """生成中计划邻域动作。

        第一类动作是在同一箱区内换区块，不改变大计划配额；第二类动作是在
        同航次不同箱区之间交换两个单位箱，交换后仍必须通过硬约束检查。
        """
        if not assignment:
            return False
        if self.random.random() < 0.7:
            idx = self.random.randrange(len(self.units))
            unit = self.units[idx]
            old_block = assignment[idx]
            area_no = self.block_area[old_block]
            candidates = self.blocks_by_area.get(area_no, [])[:]
            self.random.shuffle(candidates)
            for new_block in candidates[:30]:
                if new_block != old_block and self._medium_feasible_after(assignment, [(idx, new_block)]):
                    assignment[idx] = new_block
                    return True
            return False

        i = self.random.randrange(len(self.units))
        u1 = self.units[i]
        old1 = assignment[i]
        area1 = self.block_area[old1]
        possible = [
            j
            for j, u2 in enumerate(self.units)
            if u2.group.voyage_id == u1.group.voyage_id and self.block_area[assignment[j]] != area1
        ]
        self.random.shuffle(possible)
        for j in possible[:50]:
            old2 = assignment[j]
            if self._medium_feasible_after(assignment, [(i, old2), (j, old1)]):
                assignment[i], assignment[j] = old2, old1
                return True
        return False

    def _medium_feasible_after(self, assignment: list[str], changes: list[tuple[int, str]]) -> bool:
        trial = assignment[:]
        for idx, block_id in changes:
            trial[idx] = block_id
        return self._check_medium_hard_constraints(trial)

    def _check_medium_hard_constraints(self, assignment: list[str]) -> bool:
        """检查中计划硬约束。

        这里是大计划与中计划对齐的关键位置：除了航次-箱区总量，还会检查
        航次-箱区-大计划尺寸口径的数量，确保中计划不会把大计划的 20 尺量
        挪成 40 尺量，或反过来。
        """
        area_count: Counter[tuple[str, str]] = Counter()
        area_size_count: Counter[tuple[str, str, str]] = Counter()
        block_units: defaultdict[str, list[Unit]] = defaultdict(list)
        for unit, block_id in zip(self.units, assignment):
            if not block_id:
                return False
            area_no = self.block_area[block_id]
            if area_no not in self.problem.assigned_areas.get(unit.group.voyage_id, set()):
                return False
            area_count[(unit.group.voyage_id, area_no)] += 1
            area_size_count[(unit.group.voyage_id, area_no, unit.group.big_plan_size_mode)] += 1
            block_units[block_id].append(unit)
        for key, quota in self.problem.area_quota.items():
            if area_count[key] != quota:
                return False
        for key, quota in self.problem.area_size_quota.items():
            if area_size_count[key] != quota:
                return False
        for block_id, units in block_units.items():
            if len(units) > self.block_total_cap[block_id]:
                return False
            by_size = Counter(unit.group.size_mode for unit in units)
            for size_mode, count in by_size.items():
                if count > self.block_size_cap[(block_id, size_mode)]:
                    return False
        return True

    def medium_energy(self, assignment: list[str]) -> float:
        group_areas: defaultdict[str, set[str]] = defaultdict(set)
        group_blocks: defaultdict[str, set[str]] = defaultdict(set)
        voyage_area_blocks: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
        voyage_blocks: set[tuple[str, str]] = set()
        block_load: Counter[str] = Counter()
        energy = 0.0
        for unit, block_id in zip(self.units, assignment):
            area_no = self.block_area[block_id]
            group = unit.group
            group_areas[group.group_id].add(area_no)
            group_blocks[group.group_id].add(block_id)
            voyage_area_blocks[(group.voyage_id, area_no)].add(self._block_index(block_id))
            voyage_blocks.add((group.voyage_id, block_id))
            block_load[block_id] += 1

        for group in self.problem.groups:
            energy += self.config.group_area_split_penalty * max(0, len(group_areas[group.group_id]) - 1)
            blocks = len(group_blocks[group.group_id])
            energy += self.config.group_block_split_penalty * max(0, blocks - 1)
            if group.special_signature != "NORMAL":
                energy += self.config.special_block_split_penalty * max(0, blocks - 1)

        for blocks in voyage_area_blocks.values():
            if not blocks:
                continue
            energy += self.config.voyage_block_count_penalty * max(0, len(blocks) - 1)
            span = max(blocks) - min(blocks) + 1
            energy += self.config.voyage_block_gap_penalty * (span - len(blocks))

        for block_id, load in block_load.items():
            cap = max(1, self.block_total_cap[block_id])
            util = load / cap
            if 0.0 < util < 0.35:
                energy += self.config.block_utilization_penalty * (0.35 - util)

        for voyage_id, block_id in voyage_blocks:
            area_no = self.block_area[block_id]
            if self._area_has_loading_during_window(voyage_id, area_no):
                energy += self.config.active_loading_area_penalty
            if self._area_has_loading_after_window(voyage_id, area_no):
                energy -= self.config.post_window_loading_area_reward
        return energy

