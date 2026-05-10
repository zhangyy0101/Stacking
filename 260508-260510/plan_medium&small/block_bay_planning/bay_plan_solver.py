from __future__ import annotations

import math
from collections import Counter, defaultdict

from .models import Bay, Unit


class BayPlanSolverMixin:
    """小计划求解逻辑：把中计划区块内的大属性组分配到贝位。

    小计划不再改变大计划的箱区配额，也不主动改变中计划的区块结果；只有在
    某个区块内部找不到可用贝位时，才会在同一箱区内修复到另一个区块。这样
    可以保证大计划的箱区边界和中计划的区块边界基本稳定。
    """

    def _solve_small(self, medium_assignment: list[str]) -> tuple[list[str], list[str]]:
        """用模拟退火搜索贝位分配方案。

        `medium_assignment` 给出每个单位箱所在区块，`assignment` 给出每个单位箱
        所在贝位。邻域动作只允许在当前区块内部换贝位，因此不会破坏大计划
        的航次-箱区-尺寸配额。
        """
        medium_assignment = medium_assignment[:]
        assignment = self._initial_small_assignment(medium_assignment)
        energy = self.small_energy(medium_assignment, assignment)
        best_assignment = assignment[:]
        best_energy = energy

        for it in range(1, self.config.iterations + 1):
            candidate = assignment[:]
            if not self._mutate_small(medium_assignment, candidate):
                continue
            candidate_energy = self.small_energy(medium_assignment, candidate)
            delta = candidate_energy - energy
            temperature = self._temperature(it)
            if delta <= 0 or self.random.random() < math.exp(-delta / max(temperature, 1e-9)):
                assignment = candidate
                energy = candidate_energy
                if energy < best_energy:
                    best_energy = energy
                    best_assignment = assignment[:]
        return medium_assignment, best_assignment

    def _initial_small_assignment(self, medium_assignment: list[str]) -> list[str]:
        """构造小计划初始解。

        初始分配以区块为单位，把已经分到同一区块的单位箱按属性排序后依次
        放入贝位。45 尺箱在这里和大计划保持一致，直接按 40 尺容量消耗。
        """
        assignment = ["" for _ in self.units]
        bay_load: Counter[str] = Counter()
        bay_size_load: Counter[tuple[str, str]] = Counter()
        planned_heights: defaultdict[str, set[str]] = defaultdict(set)

        units_by_block: defaultdict[str, list[Unit]] = defaultdict(list)
        for unit, block_id in zip(self.units, medium_assignment):
            units_by_block[block_id].append(unit)

        for block_id, units in units_by_block.items():
            units.sort(key=lambda u: (not u.group.special_signature != "NORMAL", u.group.port, u.group.size_mode, u.group.height))
            for unit in units:
                bay_key = self._best_initial_bay(unit, block_id, bay_load, bay_size_load, planned_heights)
                if bay_key is None:
                    block_id, bay_key = self._repair_medium_block_for_small(
                        unit, medium_assignment, block_id, bay_load, bay_size_load, planned_heights
                    )
                if bay_key is None:
                    detail = self._bay_failure_detail(unit, block_id, bay_load, bay_size_load, planned_heights)
                    raise ValueError(
                        f"no feasible bay inside block={block_id} for voyage={unit.group.voyage_id}, "
                        f"group={unit.group.group_id}; {detail}"
                    )
                medium_assignment[unit.unit_id] = block_id
                assignment[unit.unit_id] = bay_key
                bay_load[bay_key] += 1
                bay_size_load[(bay_key, unit.group.size_mode)] += 1
                if unit.group.height != "UNK":
                    planned_heights[bay_key].add(unit.group.height)
        return assignment

    def _bay_failure_detail(
        self,
        unit: Unit,
        block_id: str,
        bay_load: Counter[str],
        bay_size_load: Counter[tuple[str, str]],
        planned_heights: dict[str, set[str]],
    ) -> str:
        """生成小计划落贝失败的诊断信息。"""
        parts = [
            f"size={unit.group.size}",
            f"size_mode={unit.group.size_mode}",
            f"height={unit.group.height}",
        ]
        candidates = []
        for bay_key in self.bays_by_block.get(block_id, []):
            bay = self.bays[bay_key]
            reasons = []
            if bay.locked:
                reasons.append("locked")
            if unit.group.size_mode == "20" and self._is_area_edge_bay(bay_key):
                reasons.append("area_edge_20ft")
            if bay_load[bay_key] + 1 > bay.physical_capacity:
                reasons.append(f"physical_full:{bay_load[bay_key]}/{bay.physical_capacity}")
            size_cap = bay.cap_by_size.get(unit.group.size_mode, 0)
            if bay_size_load[(bay_key, unit.group.size_mode)] + 1 > size_cap:
                reasons.append(f"size_full:{bay_size_load[(bay_key, unit.group.size_mode)]}/{size_cap}")
            heights = set(bay.existing_heights) | set(planned_heights[bay_key])
            if unit.group.height != "UNK" and heights and heights != {unit.group.height}:
                reasons.append(f"height_mix_penalty:{','.join(sorted(heights))}")
            candidates.append(f"{bay_key}({';'.join(reasons) if reasons else 'ok'})")
        parts.append("bays=" + "|".join(candidates))
        return ", ".join(parts)

    def _repair_medium_block_for_small(
        self,
        unit: Unit,
        medium_assignment: list[str],
        old_block: str,
        bay_load: Counter[str],
        bay_size_load: Counter[tuple[str, str]],
        planned_heights: dict[str, set[str]],
    ) -> tuple[str, str | None]:
        """小计划兜底修复。

        如果某个单位箱在中计划给定区块内找不到合法贝位，会尝试在同一箱区
        的其他区块中找贝位。这个修复不会跨箱区，因此仍满足大计划箱区配额。
        """
        area_no = self.block_area[old_block]
        candidates = self.blocks_by_area.get(area_no, [])[:]
        candidates.sort(key=lambda b: (b == old_block, self._block_index(b)))
        for block_id in candidates:
            if block_id == old_block:
                continue
            bay_key = self._best_initial_bay(unit, block_id, bay_load, bay_size_load, planned_heights)
            if bay_key is not None:
                return block_id, bay_key
        return old_block, None

    def _best_initial_bay(
        self,
        unit: Unit,
        block_id: str,
        bay_load: Counter[str],
        bay_size_load: Counter[tuple[str, str]],
        planned_heights: dict[str, set[str]],
    ) -> str | None:
        """在指定区块内选择当前最合适的贝位。

        选择时先尝试普通贝位，再尝试带兜底原因的贝位；评分会偏好已有同卸港
        箱子的贝位，并惩罚冷箱、危险品、空箱等不太理想的兜底贝位。
        """
        keys = self.bays_by_block.get(block_id, [])
        primary_keys = [key for key in keys if not self.bays[key].is_fallback_bay]
        fallback_keys = [key for key in keys if self.bays[key].is_fallback_bay]
        best_key = None
        best_score = float("inf")
        for candidate_keys in (primary_keys, fallback_keys):
            for key in candidate_keys:
                bay = self.bays[key]
                if unit.group.size_mode == "20" and self._is_area_edge_bay(bay.bay_key):
                    continue
                if not self._can_place_in_bay(
                    unit,
                    bay,
                    bay_load[key],
                    bay_size_load[(key, unit.group.size_mode)],
                    planned_heights[key],
                ):
                    continue
                score = bay_load[key] * 0.05 + bay.bay_order * 0.01
                if unit.group.port in bay.existing_ports:
                    score -= 3
                heights = set(bay.existing_heights) | set(planned_heights[key])
                if unit.group.height != "UNK" and heights and heights != {unit.group.height}:
                    score += self.config.bay_height_mix_penalty
                if bay.is_fallback_bay:
                    score += self.config.fallback_bay_penalty
                if score < best_score:
                    best_score = score
                    best_key = key
            if best_key is not None:
                return best_key
        return best_key

    def _can_place_in_bay(
        self,
        unit: Unit,
        bay: Bay,
        current_total_load: int,
        current_size_load: int,
        planned_heights: set[str],
    ) -> bool:
        """检查一个单位箱是否能放入某个贝位。

        小计划硬约束包括：贝位未锁定、物理容量不超、20/40 尺寸容量不超、
        以及 20 尺不能放在箱区边界贝位。高度是否混放不再作为硬约束，
        而是在小计划评分中作为软惩罚处理。这里使用 `group.size_mode`，
        其中 45 尺已经归入 40 尺。
        """
        group = unit.group
        if bay.locked:
            return False
        if current_total_load + 1 > bay.physical_capacity:
            return False
        if current_size_load + 1 > bay.capacity_for(group):
            return False
        return True

    def _mutate_small(self, medium_assignment: list[str], assignment: list[str]) -> bool:
        """生成小计划邻域动作：把一个单位箱换到同区块的另一个贝位。"""
        if not assignment:
            return False
        idx = self.random.randrange(len(self.units))
        unit = self.units[idx]
        old_key = assignment[idx]
        block_id = medium_assignment[idx]
        candidates = self.bays_by_block.get(block_id, [])[:]
        self.random.shuffle(candidates)
        for new_key in candidates[:40]:
            if new_key != old_key and self._small_feasible_after(medium_assignment, assignment, [(idx, new_key)]):
                assignment[idx] = new_key
                return True
        return False

    def _small_feasible_after(
        self,
        medium_assignment: list[str],
        assignment: list[str],
        changes: list[tuple[int, str]],
    ) -> bool:
        trial = assignment[:]
        for idx, bay_key in changes:
            trial[idx] = bay_key
        return self._check_small_hard_constraints(medium_assignment, trial)

    def _check_small_hard_constraints(self, medium_assignment: list[str], assignment: list[str]) -> bool:
        """检查小计划硬约束。

        这里重点保证小计划没有破坏中计划区块边界，同时检查贝位物理容量
        和 20/40 尺寸容量。高度混放只进入软约束评分，不会直接判不可行。
        由于区块已经满足大计划尺寸配额，小计划只要不跨区块/箱区移动，
        就天然保持和大计划 20/40 尺寸输出一致。
        """
        bay_units: defaultdict[str, list[Unit]] = defaultdict(list)
        for unit, block_id, bay_key in zip(self.units, medium_assignment, assignment):
            if not bay_key:
                return False
            bay = self.bays[bay_key]
            if bay.block_id != block_id:
                return False
            if unit.group.size_mode == "20" and self._is_area_edge_bay(bay_key):
                return False
            bay_units[bay_key].append(unit)
        for bay_key, units in bay_units.items():
            bay = self.bays[bay_key]
            by_size = Counter(unit.group.size_mode for unit in units)
            if len(units) > bay.physical_capacity:
                return False
            for size_mode, count in by_size.items():
                if count > bay.cap_by_size.get(size_mode, 0):
                    return False
        return True

    def small_energy(self, medium_assignment: list[str], assignment: list[str]) -> float:
        """小计划软约束评分。

        分数越低越好。主要惩罚属性组被拆到太多贝位、同贝位混卸港/船公司/
        状态/尺寸/特殊属性，以及使用兜底贝位。
        """
        group_bays: defaultdict[str, set[str]] = defaultdict(set)
        bay_ports: defaultdict[str, set[str]] = defaultdict(set)
        bay_ops: defaultdict[str, set[str]] = defaultdict(set)
        bay_status: defaultdict[str, set[str]] = defaultdict(set)
        bay_sizes: defaultdict[str, set[str]] = defaultdict(set)
        bay_heights: defaultdict[str, set[str]] = defaultdict(set)
        bay_specials: defaultdict[str, set[str]] = defaultdict(set)
        energy = 0.0
        for unit, bay_key in zip(self.units, assignment):
            bay = self.bays[bay_key]
            group = unit.group
            group_bays[group.group_id].add(bay_key)
            bay_ports[bay_key].add(group.port)
            bay_ops[bay_key].add(group.operator)
            bay_status[bay_key].add(group.status)
            bay_sizes[bay_key].add(group.size_mode)
            if group.height != "UNK":
                bay_heights[bay_key].add(group.height)
            bay_specials[bay_key].add(group.special_signature if group.special_signature != "NORMAL" else "NORMAL")
            if bay.is_fallback_bay:
                energy += self.config.fallback_bay_penalty

        for group in self.problem.groups:
            energy += self.config.group_bay_split_penalty * max(0, len(group_bays[group.group_id]) - 1)

        for bay_key in bay_ports:
            energy += self.config.bay_port_mix_penalty * max(0, len(bay_ports[bay_key]) - 1)
            energy += self.config.bay_operator_mix_penalty * max(0, len(bay_ops[bay_key]) - 1)
            energy += self.config.bay_status_mix_penalty * max(0, len(bay_status[bay_key]) - 1)

        for bay_key, sizes in bay_sizes.items():
            all_sizes = set(self.bays[bay_key].existing_size_modes) | sizes
            energy += self.config.bay_size_mix_penalty * max(0, len(all_sizes) - 1)

        for bay_key, heights in bay_heights.items():
            all_heights = set(self.bays[bay_key].existing_heights) | heights
            energy += self.config.bay_height_mix_penalty * max(0, len(all_heights) - 1)

        for bay_key, specials in bay_specials.items():
            all_specials = set(self.bays[bay_key].existing_special_signatures) | specials
            energy += self.config.bay_special_mix_penalty * max(0, len(all_specials) - 1)
        return energy

