from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime

from .models import BoxGroup


MediumAssignment = dict[str, Counter[str]]


class AreaPlanSolverMixin:
    """中计划求解逻辑：按属性组-箱区-箱量进行聚合分配。"""

    def _solve_medium(self) -> MediumAssignment:
        self._log_sa_progress("start", 0, 0.0, 0.0, self.config.initial_temperature, 0)
        assignment = self._initial_medium_assignment()
        energy = self.medium_energy(assignment, include_small_proxy=True)
        best_assignment = self._copy_assignment(assignment)
        best_energy = energy
        accepted_count = 0
        self._record_energy_history(
            "medium",
            0,
            energy,
            best_energy,
            self.config.initial_temperature,
            accepted_count,
        )
        self._log_sa_progress("medium", 0, energy, best_energy, self.config.initial_temperature, accepted_count)

        for it in range(1, self.config.iterations + 1):
            temperature = self._temperature(it)
            include_small_proxy = self._should_score_small_plan_proxy(it)
            if include_small_proxy != self._should_score_small_plan_proxy(it - 1):
                energy = self.medium_energy(assignment, include_small_proxy=include_small_proxy)
                best_energy = self.medium_energy(best_assignment, include_small_proxy=include_small_proxy)
            candidate = self._copy_assignment(assignment)
            if not self._mutate_medium(candidate):
                if self._should_record_energy(it):
                    self._record_energy_history("medium", it, energy, best_energy, temperature, accepted_count)
                if self._should_log_progress(it):
                    self._log_sa_progress("medium", it, energy, best_energy, temperature, accepted_count)
                continue
            candidate_energy = self.medium_energy(candidate, include_small_proxy=include_small_proxy)
            delta = candidate_energy - energy
            if delta <= 0 or self.random.random() < math.exp(-delta / max(temperature, 1e-9)):
                assignment = candidate
                energy = candidate_energy
                accepted_count += 1
                if energy < best_energy:
                    best_energy = energy
                    best_assignment = self._copy_assignment(assignment)
            if self._should_record_energy(it):
                self._record_energy_history("medium", it, energy, best_energy, temperature, accepted_count)
            if self._should_log_progress(it):
                self._log_sa_progress("medium", it, energy, best_energy, temperature, accepted_count)
            if self._should_check_small_plan(it):
                if self._check_small_plan_feedback(best_assignment, it):
                    energy = self.medium_energy(assignment, include_small_proxy=include_small_proxy)
                    best_energy = self.medium_energy(best_assignment, include_small_proxy=include_small_proxy)
                    if energy < best_energy:
                        best_assignment = self._copy_assignment(assignment)
                        best_energy = energy
        self._log_sa_progress("finish", self.config.iterations, energy, best_energy, self.config.final_temperature, accepted_count)
        return best_assignment

    def _should_log_progress(self, iteration: int) -> bool:
        return self.config.log_every > 0 and iteration % self.config.log_every == 0

    def _should_check_small_plan(self, iteration: int) -> bool:
        return (
            0 < iteration < self.config.iterations
            and self.config.small_plan_check_every > 0
            and iteration % self.config.small_plan_check_every == 0
        )

    def _should_score_small_plan_proxy(self, iteration: int) -> bool:
        every = max(1, int(getattr(self.config, "small_plan_proxy_every", 1) or 1))
        return every == 1 or iteration == 0 or iteration == self.config.iterations or iteration % every == 0

    def _log_sa_progress(
        self,
        phase: str,
        iteration: int,
        current_energy: float,
        best_energy: float,
        temperature: float,
        accepted_count: int,
    ) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if phase == "start":
            print(
                f"[{stamp}] medium SA start: iterations={self.config.iterations}, "
                f"groups={len(self.problem.groups)}, boxes={sum(group.demand for group in self.problem.groups)}",
                flush=True,
            )
            return
        print(
            f"[{stamp}] {phase} iter={iteration}/{self.config.iterations} "
            f"current={current_energy:.4f} best={best_energy:.4f} "
            f"temp={temperature:.4f} accepted={accepted_count}",
            flush=True,
        )

    def _initial_medium_assignment(self) -> MediumAssignment:
        """按大计划箱区-尺寸软目标构造聚合初始解。"""
        assignment: MediumAssignment = {group.group_id: Counter() for group in self.problem.groups}
        area_load: Counter[str] = Counter()
        area_size_load: Counter[tuple[str, str]] = Counter()
        quota_load: Counter[tuple[str, str, str, str]] = Counter()

        groups_by_voyage_size: defaultdict[tuple[str, str, str], list[BoxGroup]] = defaultdict(list)
        for group in self.problem.groups:
            groups_by_voyage_size[(group.voyage_id, group.status, group.big_plan_size_mode)].append(group)

        for (voyage_id, flow, size_mode), groups in sorted(groups_by_voyage_size.items()):
            groups.sort(key=lambda group: (group.port, group.group_id))
            quota_weights = {
                area: qty
                for (v, f, area, size), qty in self.problem.area_size_quota.items()
                if v == voyage_id and f == flow and size == size_mode and qty > 0
            }
            if not quota_weights:
                quota_weights = {
                    area: qty
                    for (v, f, area), qty in self.problem.area_quota.items()
                    if v == voyage_id and f == flow and qty > 0
                }
            if not quota_weights:
                raise ValueError(f"big plan has no area pattern for voyage={voyage_id}, flow={flow}, size={size_mode}")

            for group in groups:
                allocations = self._allocate_group_to_areas(
                    group,
                    quota_weights,
                    area_load,
                    area_size_load,
                    quota_load,
                )
                if sum(allocations.values()) != group.demand:
                    raise ValueError(
                        f"cannot build feasible medium initial allocation for group={group.group_id}, "
                        f"demand={group.demand}, allocated={sum(allocations.values())}"
                    )
                assignment[group.group_id].update(allocations)
                for area_no, qty in allocations.items():
                    area_load[area_no] += qty
                    area_size_load[(area_no, group.size_mode)] += qty
                    quota_load[(group.voyage_id, group.status, area_no, group.big_plan_size_mode)] += qty
        return assignment

    def _allocate_group_to_areas(
        self,
        group: BoxGroup,
        weights: dict[str, int],
        area_load: Counter[str],
        area_size_load: Counter[tuple[str, str]],
        quota_load: Counter[tuple[str, str, str, str]],
    ) -> Counter[str]:
        allocations: Counter[str] = Counter()
        candidates = [
            area_no for area_no in self._candidate_areas_for_group(group)
            if self._area_free_capacity(group, area_no, area_load, area_size_load, quota_load) > 0
        ]
        if not candidates:
            return allocations

        preferred = {area: qty for area, qty in weights.items() if area in candidates and qty > 0}
        if preferred and self._prefers_concentrated_medium_group(group):
            for area_no in sorted(
                preferred,
                key=lambda area: (
                    0 if self._area_free_capacity(group, area, area_load, area_size_load, quota_load) >= group.demand else 1,
                    -self._area_free_capacity(group, area, area_load, area_size_load, quota_load),
                    -weights.get(area, 0),
                    area,
                ),
            ):
                free = self._area_free_capacity(group, area_no, area_load, area_size_load, quota_load)
                take = min(group.demand, free)
                if take > 0:
                    allocations[area_no] += take
                    break
        elif preferred:
            proportional = self._allocate_by_weights(preferred, group.demand)
            for area_no, qty in proportional.items():
                free = self._area_free_capacity(group, area_no, area_load, area_size_load, quota_load)
                take = min(qty, free)
                if take > 0:
                    allocations[area_no] += take

        remaining = group.demand - sum(allocations.values())
        if remaining <= 0:
            return allocations

        def fill_key(area_no: str) -> tuple[int, int, int, str]:
            in_big_plan = area_no in self.problem.assigned_areas.get((group.voyage_id, group.status), set())
            free = self._area_free_capacity(group, area_no, area_load, area_size_load, quota_load)
            return (0 if in_big_plan else 1, 0 if free >= remaining else 1, -weights.get(area_no, 0), area_no)

        for area_no in sorted(candidates, key=fill_key):
            trial_area_load = area_load + allocations
            trial_area_size_load = Counter(area_size_load)
            trial_quota_load = Counter(quota_load)
            for allocated_area, allocated_qty in allocations.items():
                trial_area_size_load[(allocated_area, group.size_mode)] += allocated_qty
                trial_quota_load[(group.voyage_id, group.status, allocated_area, group.big_plan_size_mode)] += allocated_qty
            free = self._area_free_capacity(group, area_no, trial_area_load, trial_area_size_load, trial_quota_load)
            take = min(remaining, free)
            if take > 0:
                allocations[area_no] += take
                remaining -= take
            if remaining == 0:
                break
        return allocations

    def _area_free_capacity(
        self,
        group: BoxGroup,
        area_no: str,
        area_load: Counter[str],
        area_size_load: Counter[tuple[str, str]],
        quota_load: Counter[tuple[str, str, str, str]],
    ) -> int:
        total_free = self.area_total_cap[area_no] - area_load[area_no]
        size_free = self.area_size_cap[(area_no, group.size_mode)] - area_size_load[(area_no, group.size_mode)]
        quota_key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
        if quota_key in self.problem.area_size_quota:
            quota_free = self.problem.area_size_quota[quota_key] - quota_load[quota_key]
        else:
            quota_free = min(total_free, size_free)
        return max(0, min(total_free, size_free, quota_free))

    def _mutate_medium(self, assignment: MediumAssignment) -> bool:
        """Move a batch of boxes for one coarse group to another feasible area."""
        movable = [
            group for group in self.problem.groups
            if sum(assignment.get(group.group_id, Counter()).values()) > 0
        ]
        if not movable:
            return False
        group = self.random.choice(movable)
        group_assignment = assignment[group.group_id]
        source_areas = [area_no for area_no, qty in group_assignment.items() if qty > 0]
        self.random.shuffle(source_areas)
        candidates = list(self._candidate_areas_for_group(group))
        self.random.shuffle(candidates)
        area_load, area_size_load, quota_load = self._medium_loads(assignment)

        for source_area in source_areas:
            source_qty = group_assignment[source_area]
            max_move = max(1, min(source_qty, max(5, math.ceil(group.demand * 0.20))))
            move_qty = self.random.randint(1, max_move)
            for target_area in candidates[:80]:
                if target_area == source_area:
                    continue
                if self._medium_feasible_after(
                    assignment,
                    group,
                    source_area,
                    target_area,
                    move_qty,
                    area_load,
                    area_size_load,
                    quota_load,
                ):
                    group_assignment[source_area] -= move_qty
                    if group_assignment[source_area] <= 0:
                        del group_assignment[source_area]
                    group_assignment[target_area] += move_qty
                    area_load[source_area] -= move_qty
                    area_load[target_area] += move_qty
                    area_size_load[(source_area, group.size_mode)] -= move_qty
                    area_size_load[(target_area, group.size_mode)] += move_qty
                    quota_load[(group.voyage_id, group.status, source_area, group.big_plan_size_mode)] -= move_qty
                    quota_load[(group.voyage_id, group.status, target_area, group.big_plan_size_mode)] += move_qty
                    return True
        return False

    def _medium_feasible_after(
        self,
        assignment: MediumAssignment,
        group: BoxGroup,
        source_area: str,
        target_area: str,
        qty: int,
        area_load: Counter[str],
        area_size_load: Counter[tuple[str, str]],
        quota_load: Counter[tuple[str, str, str, str]],
    ) -> bool:
        if qty <= 0 or assignment[group.group_id].get(source_area, 0) < qty:
            return False
        if target_area not in self._candidate_areas_for_group(group):
            return False
        target_total = area_load[target_area] + qty
        if target_total > self.area_total_cap[target_area]:
            return False
        target_size = area_size_load[(target_area, group.size_mode)] + qty
        if target_size > self.area_size_cap[(target_area, group.size_mode)]:
            return False
        quota_key = (group.voyage_id, group.status, target_area, group.big_plan_size_mode)
        if quota_key not in self.problem.area_size_quota:
            return True
        return quota_load[quota_key] + qty <= self.problem.area_size_quota[quota_key]

    def _medium_loads(self, assignment: MediumAssignment) -> tuple[Counter[str], Counter[tuple[str, str]], Counter[tuple[str, str, str, str]]]:
        groups_by_id = {group.group_id: group for group in self.problem.groups}
        area_load: Counter[str] = Counter()
        area_size_load: Counter[tuple[str, str]] = Counter()
        quota_load: Counter[tuple[str, str, str, str]] = Counter()
        for group_id, area_counts in assignment.items():
            group = groups_by_id[group_id]
            for area_no, qty in area_counts.items():
                if qty > 0:
                    area_load[area_no] += qty
                    area_size_load[(area_no, group.size_mode)] += qty
                    quota_load[(group.voyage_id, group.status, area_no, group.big_plan_size_mode)] += qty
        return area_load, area_size_load, quota_load

    def _check_medium_hard_constraints(self, assignment: MediumAssignment) -> bool:
        groups_by_id = {group.group_id: group for group in self.problem.groups}
        area_load: Counter[str] = Counter()
        area_size_load: Counter[tuple[str, str]] = Counter()
        quota_load: Counter[tuple[str, str, str, str]] = Counter()
        for group_id, area_counts in assignment.items():
            group = groups_by_id[group_id]
            if sum(area_counts.values()) != group.demand:
                return False
            candidates = self._candidate_areas_for_group(group)
            for area_no, qty in area_counts.items():
                if qty <= 0:
                    return False
                if area_no not in candidates:
                    return False
                area_load[area_no] += qty
                area_size_load[(area_no, group.size_mode)] += qty
                quota_load[(group.voyage_id, group.status, area_no, group.big_plan_size_mode)] += qty
        for area_no, load in area_load.items():
            if load > self.area_total_cap[area_no]:
                return False
        for key, load in area_size_load.items():
            if load > self.area_size_cap[key]:
                return False
        for key, load in quota_load.items():
            if key in self.problem.area_size_quota and load > self.problem.area_size_quota[key]:
                return False
        return True

    def medium_energy(self, assignment: MediumAssignment, include_small_proxy: bool = True) -> float:
        group_areas: defaultdict[str, set[str]] = defaultdict(set)
        group_area_count: defaultdict[tuple[str, str], int] = defaultdict(int)
        area_size_count: Counter[tuple[str, str, str, str]] = Counter()
        voyage_areas: set[tuple[str, str]] = set()
        coarse_area_count: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        energy = 0.0
        groups_by_id = {group.group_id: group for group in self.problem.groups}

        for group_id, area_counts in assignment.items():
            group = groups_by_id[group_id]
            for area_no, qty in area_counts.items():
                if qty <= 0:
                    continue
                group_areas[group.group_id].add(area_no)
                group_area_count[(group.group_id, area_no)] += qty
                area_size_count[(group.voyage_id, group.status, area_no, group.big_plan_size_mode)] += qty
                coarse_area_count[(group.voyage_id, group.status, group.port, group.size)][area_no] += qty
                voyage_areas.add((group.voyage_id, area_no))
                if area_no not in self.problem.assigned_areas.get((group.voyage_id, group.status), set()):
                    energy += self.config.non_big_plan_area_penalty * qty
                exact_feedback = getattr(self, "small_plan_area_feedback", {}).get(
                    (group.voyage_id, group.status, group.port, group.size, area_no),
                    0,
                )
                broad_feedback = getattr(self, "small_plan_area_feedback", {}).get(
                    (group.voyage_id, group.status, "*", group.size, area_no),
                    0,
                )
                feedback_count = max(exact_feedback, broad_feedback)
                if feedback_count:
                    energy += self.config.small_plan_feedback_penalty * feedback_count * qty

        for group in self.problem.groups:
            energy += self.config.group_area_split_penalty * max(0, len(group_areas[group.group_id]) - 1)
            if self._prefers_concentrated_medium_group(group):
                energy += self._group_area_concentration_energy(group, group_area_count)
            else:
                energy += self._group_area_balance_energy(group, group_area_count)
        energy += self._big_plan_deviation_energy(area_size_count)

        for voyage_id, area_no in voyage_areas:
            berth = self._berth_key(voyage_id)
            if berth:
                distance = self.problem.berth_distances.get((area_no, berth))
                if distance is not None:
                    energy += self.config.berth_distance_penalty * distance / 100.0
            if self._area_has_loading_during_window(voyage_id, area_no):
                energy += self.config.active_loading_area_penalty
            if self._area_has_loading_after_window(voyage_id, area_no):
                energy -= self.config.post_window_loading_area_reward

        small_proxy = getattr(self, "_small_plan_proxy_energy", None) if include_small_proxy else None
        if small_proxy is not None:
            energy += small_proxy(coarse_area_count)
        return energy

    def _candidate_areas_for_group(self, group: BoxGroup) -> set[str]:
        cached = getattr(self, "_candidate_area_cache", {}).get(group.group_id)
        if cached is not None:
            return cached
        return self._compute_candidate_areas_for_group(group)

    def _compute_candidate_areas_for_group(self, group: BoxGroup) -> set[str]:
        big_plan_candidates = {
            area_no
            for (voyage_id, flow, area_no, size_mode), quota in self.problem.area_size_quota.items()
            if voyage_id == group.voyage_id
            and flow == group.status
            and size_mode == group.big_plan_size_mode
            and quota > 0
            and self.area_size_cap[(area_no, group.size_mode)] > 0
            and self.area_total_cap[area_no] > 0
            and group.status in self.problem.area_functions.get(area_no, set())
        }
        fallback_candidates = {
            area_no
            for area_no in self.bays_by_area
            if self.area_size_cap[(area_no, group.size_mode)] > 0
            and self.area_total_cap[area_no] > 0
            and group.status in self.problem.area_functions.get(area_no, set())
        }
        return big_plan_candidates | fallback_candidates

    def _big_plan_deviation_energy(self, area_size_count: Counter[tuple[str, str, str, str]]) -> float:
        energy = 0.0
        keys = set(area_size_count) | set(self.problem.area_size_quota)
        for key in keys:
            target = self.problem.area_size_quota.get(key, 0)
            actual = area_size_count.get(key, 0)
            if target == 0 and actual == 0:
                continue
            energy += self.config.big_plan_area_deviation_penalty * abs(actual - target) / max(1, target)
        return energy

    def _group_area_balance_energy(self, group: BoxGroup, group_area_count: dict[tuple[str, str], int]) -> float:
        quotas = {
            area: qty
            for (voyage_id, flow, area, size_mode), qty in self.problem.area_size_quota.items()
            if voyage_id == group.voyage_id and flow == group.status and size_mode == group.big_plan_size_mode and qty > 0
        }
        total_quota = sum(quotas.values())
        if total_quota <= 0 or group.demand <= 0:
            return 0.0
        penalty = 0.0
        for area, quota in quotas.items():
            expected = group.demand * quota / total_quota
            actual = group_area_count.get((group.group_id, area), 0)
            penalty += abs(actual - expected) / max(1.0, group.demand)
        return self.config.group_area_balance_penalty * penalty

    def _prefers_concentrated_medium_group(self, group: BoxGroup) -> bool:
        threshold = int(getattr(self.config, "medium_concentrated_group_threshold", 26) or 0)
        return threshold > 0 and group.demand <= threshold

    def _group_area_concentration_energy(self, group: BoxGroup, group_area_count: dict[tuple[str, str], int]) -> float:
        counts = [
            qty
            for (group_id, _area_no), qty in group_area_count.items()
            if group_id == group.group_id and qty > 0
        ]
        if len(counts) <= 1:
            return 0.0
        split_penalty = getattr(self.config, "medium_small_group_area_split_penalty", 28.0)
        fragment_penalty = getattr(self.config, "medium_small_group_fragment_penalty", 0.8)
        largest = max(counts)
        fragments = group.demand - largest
        return split_penalty * (len(counts) - 1) + fragment_penalty * fragments

    @staticmethod
    def _copy_assignment(assignment: MediumAssignment) -> MediumAssignment:
        return {group_id: Counter(area_counts) for group_id, area_counts in assignment.items()}

    @staticmethod
    def _allocate_by_weights(weights: dict[str, int], total: int) -> dict[str, int]:
        items = [(key, value) for key, value in sorted(weights.items()) if value > 0]
        if not items or total <= 0:
            return {}
        source_total = sum(value for _, value in items)
        raw = [value * total / source_total for _, value in items]
        base = [int(value) for value in raw]
        remain = total - sum(base)
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
        for idx in order[:remain]:
            base[idx] += 1
        return {key: qty for (key, _), qty in zip(items, base) if qty > 0}

    def _berth_key(self, voyage_id: str) -> str:
        return self.problem.berth_by_voyage.get(voyage_id, "")
