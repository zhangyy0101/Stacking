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
        include_small_proxy = self._should_score_small_plan_proxy(0)
        energy = self.medium_energy(assignment, include_small_proxy=include_small_proxy)
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
        self._log_sa_progress(
            "medium",
            0,
            energy,
            best_energy,
            self.config.initial_temperature,
            accepted_count,
        )

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
                    self._log_sa_progress(
                        "medium",
                        it,
                        energy,
                        best_energy,
                        temperature,
                        accepted_count,
                    )
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
                self._log_sa_progress(
                    "medium",
                    it,
                    energy,
                    best_energy,
                    temperature,
                    accepted_count,
                )
            if self._should_check_small_plan(it):
                check_changed, repaired_assignment = self._check_small_plan_feedback(best_assignment, it)
                if repaired_assignment is not None:
                    repaired_energy = self.medium_energy(repaired_assignment, include_small_proxy=include_small_proxy)
                    assignment = self._copy_assignment(repaired_assignment)
                    energy = repaired_energy
                    best_assignment = self._copy_assignment(repaired_assignment)
                    best_energy = repaired_energy
                if check_changed:
                    energy = self.medium_energy(assignment, include_small_proxy=include_small_proxy)
                    best_energy = self.medium_energy(best_assignment, include_small_proxy=include_small_proxy)
                    if energy < best_energy:
                        best_assignment = self._copy_assignment(assignment)
                        best_energy = energy
        self._log_sa_progress(
            "finish",
            self.config.iterations,
            energy,
            best_energy,
            self.config.final_temperature,
            accepted_count,
        )
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
        configured = int(getattr(self.config, "small_plan_proxy_every", 0) or 0)
        if configured <= 0:
            return False
        every = max(1, configured)
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
        message = (
            f"[{stamp}] {phase} iter={iteration}/{self.config.iterations} "
            f"current={current_energy:.4f} best={best_energy:.4f} "
            f"temp={temperature:.4f} accepted={accepted_count}"
        )
        print(message, flush=True)

    def _initial_medium_assignment(self) -> MediumAssignment:
        return self._initial_medium_assignment_with_retries()

    def _initial_medium_assignment_with_retries(self) -> MediumAssignment:
        diagnostics: dict = {
            "method": "",
            "configured_greedy_attempts": max(
                0,
                int(getattr(self.config, "medium_initial_assignment_attempts", 30) or 0),
            ),
            "scip": {},
            "greedy_attempts": [],
            "best_greedy": None,
        }

        scip_assignment, scip_diagnostics = self._build_scip_initial_assignment()
        diagnostics["scip"] = scip_diagnostics
        if scip_assignment is not None:
            diagnostics["method"] = "scip"
            diagnostics["attempt_count"] = 0
            self.medium_initial_assignment_diagnostics = diagnostics
            self._log_medium_initial_assignment_success("scip", scip_diagnostics)
            return scip_assignment

        best_attempt: dict | None = None
        best_assignment: MediumAssignment | None = None
        max_attempts = diagnostics["configured_greedy_attempts"]
        for attempt_no, (order_strategy, allocation_strategy, random_offset) in enumerate(
            self._medium_initial_attempt_specs(max_attempts),
            start=1,
        ):
            assignment, attempt = self._build_greedy_medium_assignment(
                order_strategy,
                allocation_strategy,
                attempt_no,
                random_offset,
            )
            diagnostics["greedy_attempts"].append(attempt)
            if best_attempt is None or attempt["shortage"] < best_attempt["shortage"]:
                best_attempt = dict(attempt)
                best_assignment = self._copy_assignment(assignment)
                diagnostics["best_greedy"] = best_attempt
            if attempt["shortage"] == 0 and self._check_medium_hard_constraints(assignment):
                diagnostics["method"] = "greedy"
                diagnostics["attempt_count"] = attempt_no
                self.medium_initial_assignment_diagnostics = diagnostics
                self._log_medium_initial_assignment_success("greedy", attempt)
                return assignment

        diagnostics["method"] = "failed"
        diagnostics["attempt_count"] = max_attempts
        self.medium_initial_assignment_diagnostics = diagnostics
        best = best_attempt or {"strategy": "none", "shortage": None, "shortage_by_group": {}}
        message = (
            "cannot build feasible medium initial allocation after SCIP and "
            f"{max_attempts} greedy attempt(s): "
            f"best_strategy={best.get('strategy')}, "
            f"shortage={best.get('shortage')}, "
            f"shortage_by_group={best.get('shortage_by_group')}"
        )
        self._log_medium_initial_assignment_failure(message)
        if best_assignment is not None:
            diagnostics["method"] = "partial_greedy"
            diagnostics["used_partial_assignment"] = True
            diagnostics["partial_assignment_reason"] = message
            self.medium_initial_assignment_diagnostics = diagnostics
            return best_assignment
        raise ValueError(message)

    def _build_scip_initial_assignment(self) -> tuple[MediumAssignment | None, dict]:
        diagnostics: dict = {
            "available": False,
            "status": "",
            "solution_count": 0,
            "variable_count": 0,
            "constraint_count": 0,
            "shortage_by_group": {},
        }
        try:
            from pyscipopt import Model, quicksum
        except Exception as exc:  # pragma: no cover - depends on local SCIP install.
            diagnostics["failure"] = f"{type(exc).__name__}: {exc}"
            return None, diagnostics

        diagnostics["available"] = True
        model = None
        groups_by_id = {group.group_id: group for group in self.problem.groups}
        x_vars: dict[tuple[str, str], object] = {}
        group_terms: defaultdict[str, list] = defaultdict(list)
        area_terms: defaultdict[str, list] = defaultdict(list)
        area_size_terms: defaultdict[tuple[str, str], list] = defaultdict(list)
        quota_terms: defaultdict[tuple[str, str, str, str], list] = defaultdict(list)
        voyage_area_terms: defaultdict[tuple[str, str], list] = defaultdict(list)
        objective_terms: list = []

        try:
            model = Model("medium_initial_assignment")
            self._configure_initial_scip_output(model)
            for group_index, group in enumerate(self.problem.groups):
                candidate_areas = sorted(self._candidate_areas_for_group(group))
                if not candidate_areas:
                    diagnostics["shortage_by_group"][group.group_id] = group.demand
                    continue
                for area_no in candidate_areas:
                    quota_key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
                    upper = min(
                        group.demand,
                        self.area_total_cap[area_no],
                        self.area_size_cap[(area_no, group.size_mode)],
                    )
                    if upper <= 0:
                        continue
                    var = model.addVar(vtype="I", lb=0.0, ub=float(upper), name=f"x_{group_index}_{area_no}")
                    use = model.addVar(vtype="B", name=f"use_{group_index}_{area_no}")
                    model.addCons(var <= float(group.demand) * use)
                    x_vars[(group.group_id, area_no)] = var
                    group_terms[group.group_id].append(var)
                    area_terms[area_no].append(var)
                    area_size_terms[(area_no, group.size_mode)].append(var)
                    quota_terms[quota_key].append(var)
                    voyage_area_terms[(group.voyage_id, area_no)].append(var)
                    unit_cost = self._medium_initial_unit_cost(group, area_no)
                    split_cost = self._medium_initial_split_cost(group)
                    if unit_cost:
                        objective_terms.append(unit_cost * var)
                    if split_cost:
                        objective_terms.append(split_cost * use)

            missing_groups = [
                group
                for group in self.problem.groups
                if not group_terms.get(group.group_id)
            ]
            if missing_groups:
                diagnostics["status"] = "missing_candidates"
                diagnostics["shortage_by_group"] = {
                    group.group_id: group.demand for group in missing_groups
                }
                return None, diagnostics

            for group in self.problem.groups:
                model.addCons(quicksum(group_terms[group.group_id]) == float(group.demand))
            for area_no, terms in area_terms.items():
                model.addCons(quicksum(terms) <= float(self.area_total_cap[area_no]))
            for key, terms in area_size_terms.items():
                model.addCons(quicksum(terms) <= float(self.area_size_cap[key]))
            for key, terms in quota_terms.items():
                cap = self.problem.area_size_quota.get(key, 0)
                if cap > 0:
                    model.addCons(quicksum(terms) <= float(cap))
            for voyage_id, areas in sorted(getattr(self.problem, "user_voyage_area_requirements", {}).items()):
                for area_no in sorted(areas):
                    terms = voyage_area_terms.get((voyage_id, area_no), [])
                    if terms:
                        model.addCons(quicksum(terms) >= 1.0)
            learned_caps = getattr(self, "medium_small_learned_area_size_caps", {})
            learned_penalty = float(getattr(self.config, "medium_small_feedback_cap_penalty", 300000.0) or 300000.0)
            learned_slack_count = 0
            for key, cap in sorted(learned_caps.items()):
                terms = quota_terms.get(key, [])
                if not terms:
                    continue
                slack = model.addVar(lb=0.0, name=f"small_cap_slack_{learned_slack_count}")
                learned_slack_count += 1
                model.addCons(quicksum(terms) <= float(max(0, int(cap))) + slack)
                objective_terms.append(learned_penalty * slack)
            deviation_penalty = self._initial_big_plan_deviation_unit_penalty()
            for key, target in self._effective_big_plan_area_size_targets().items():
                if target <= 0:
                    continue
                terms = quota_terms.get(key, [])
                pos = model.addVar(lb=0.0, name=f"quota_pos_{len(objective_terms)}")
                neg = model.addVar(lb=0.0, name=f"quota_neg_{len(objective_terms)}")
                model.addCons(quicksum(terms) - float(target) == pos - neg)
                objective_terms.append(deviation_penalty * pos)
                objective_terms.append(deviation_penalty * neg)
            if objective_terms:
                model.setObjective(quicksum(objective_terms), "minimize")
            diagnostics["variable_count"] = len(x_vars) * 2
            diagnostics["constraint_count"] = len(model.getConss())
            diagnostics["learned_small_cap_count"] = learned_slack_count
            model.optimize()
            diagnostics["status"] = str(model.getStatus()).lower()
            diagnostics["solution_count"] = self._initial_scip_solution_count(model)
            if diagnostics["solution_count"] <= 0:
                return None, diagnostics

            assignment: MediumAssignment = {group.group_id: Counter() for group in self.problem.groups}
            for (group_id, area_no), var in x_vars.items():
                value = int(round(float(model.getVal(var))))
                if value > 0:
                    assignment[group_id][area_no] += value
            shortage_by_group = {
                group_id: max(0, groups_by_id[group_id].demand - sum(area_counts.values()))
                for group_id, area_counts in assignment.items()
                if sum(area_counts.values()) != groups_by_id[group_id].demand
            }
            if shortage_by_group:
                diagnostics["status"] = f"{diagnostics['status']}:incomplete_solution"
                diagnostics["shortage_by_group"] = shortage_by_group
                return None, diagnostics
            if not self._check_medium_hard_constraints(assignment):
                diagnostics["status"] = f"{diagnostics['status']}:hard_constraint_violation"
                return None, diagnostics
            return assignment, diagnostics
        except Exception as exc:
            diagnostics["failure"] = f"{type(exc).__name__}: {exc}"
            return None, diagnostics
        finally:
            if model is not None:
                self._free_initial_scip_model(model)

    def _build_greedy_medium_assignment(
        self,
        order_strategy: str,
        allocation_strategy: str,
        attempt_no: int,
        random_offset: int,
    ) -> tuple[MediumAssignment, dict]:
        randomizer = self._initial_attempt_random(random_offset)
        groups = self._order_medium_groups_for_initial_assignment(order_strategy, randomizer)
        assignment: MediumAssignment = {group.group_id: Counter() for group in self.problem.groups}
        area_load: Counter[str] = Counter()
        area_size_load: Counter[tuple[str, str]] = Counter()
        quota_load: Counter[tuple[str, str, str, str]] = Counter()
        shortage_by_group: dict[str, int] = {}

        for group in groups:
            allocations = self._allocate_group_to_areas(
                group,
                self._quota_weights_for_group(group),
                area_load,
                area_size_load,
                quota_load,
                strategy=allocation_strategy,
            )
            allocated = sum(allocations.values())
            if allocated != group.demand:
                shortage_by_group[group.group_id] = group.demand - allocated
            assignment[group.group_id].update(allocations)
            for area_no, qty in allocations.items():
                area_load[area_no] += qty
                area_size_load[(area_no, group.size_mode)] += qty
                quota_load[(group.voyage_id, group.status, area_no, group.big_plan_size_mode)] += qty

        hard_feasible = not shortage_by_group and self._check_medium_hard_constraints(assignment)
        if not hard_feasible and not shortage_by_group:
            shortage_by_group["__hard_constraints__"] = 1
        return assignment, {
            "attempt": attempt_no,
            "order_strategy": order_strategy,
            "allocation_strategy": allocation_strategy,
            "strategy": f"{order_strategy}/{allocation_strategy}",
            "shortage": sum(shortage_by_group.values()),
            "shortage_by_group": shortage_by_group,
            "hard_feasible": hard_feasible,
        }

    def _medium_initial_attempt_specs(self, attempts: int):
        base_specs = [
            ("original", "weighted", 0),
            ("hard_group_first", "weighted", 1),
            ("large_demand_first", "weighted", 2),
            ("tight_capacity_first", "capacity", 3),
            ("small_group_first", "concentrated", 4),
            ("original", "capacity", 5),
            ("hard_group_first", "capacity", 6),
            ("large_demand_first", "concentrated", 7),
            ("tight_capacity_first", "weighted", 8),
        ]
        emitted = 0
        for spec in base_specs:
            if emitted >= attempts:
                return
            emitted += 1
            yield spec
        orders = (
            "random",
            "hard_group_first",
            "large_demand_first",
            "tight_capacity_first",
            "small_group_first",
        )
        strategies = ("weighted", "capacity", "concentrated")
        while emitted < attempts:
            order = orders[emitted % len(orders)]
            strategy = strategies[(emitted // len(orders)) % len(strategies)]
            yield order, strategy, 1000 + emitted
            emitted += 1

    def _order_medium_groups_for_initial_assignment(self, order_strategy: str, randomizer) -> list[BoxGroup]:
        groups = list(self.problem.groups)
        if order_strategy == "original":
            return self._original_medium_group_order()
        if order_strategy == "random":
            randomizer.shuffle(groups)
            return groups
        if order_strategy == "large_demand_first":
            return sorted(groups, key=lambda group: (-group.demand, self._medium_group_difficulty_key(group)))
        if order_strategy == "tight_capacity_first":
            return sorted(groups, key=lambda group: self._medium_group_capacity_key(group))
        if order_strategy == "small_group_first":
            return sorted(groups, key=lambda group: (group.demand, self._medium_group_difficulty_key(group)))
        return sorted(groups, key=self._medium_group_difficulty_key)

    def _original_medium_group_order(self) -> list[BoxGroup]:
        groups_by_voyage_size: defaultdict[tuple[str, str, str], list[BoxGroup]] = defaultdict(list)
        for group in self.problem.groups:
            groups_by_voyage_size[(group.voyage_id, group.status, group.big_plan_size_mode)].append(group)
        ordered: list[BoxGroup] = []
        for _key, groups in sorted(groups_by_voyage_size.items()):
            ordered.extend(sorted(groups, key=lambda group: (group.port, group.group_id)))
        return ordered

    def _medium_group_difficulty_key(self, group: BoxGroup) -> tuple[int, int, int, int, str]:
        candidate_count, total_capacity = self._medium_group_initial_capacity(group)
        return (
            0 if candidate_count == 0 else 1,
            total_capacity - group.demand,
            candidate_count,
            -group.demand,
            group.group_id,
        )

    def _medium_group_capacity_key(self, group: BoxGroup) -> tuple[int, int, int, str]:
        candidate_count, total_capacity = self._medium_group_initial_capacity(group)
        return (total_capacity - group.demand, candidate_count, -group.demand, group.group_id)

    def _medium_group_initial_capacity(self, group: BoxGroup) -> tuple[int, int]:
        capacities = []
        for area_no in self._candidate_areas_for_group(group):
            cap = min(
                self.area_total_cap[area_no],
                self.area_size_cap[(area_no, group.size_mode)],
            )
            if cap > 0:
                capacities.append(cap)
        return len(capacities), sum(capacities)

    def _quota_weights_for_group(self, group: BoxGroup) -> dict[str, int]:
        return {
            area: qty
            for (v, f, area, size), qty in self.problem.area_size_quota.items()
            if v == group.voyage_id and f == group.status and size == group.big_plan_size_mode and qty > 0
        }

    def _initial_big_plan_deviation_unit_penalty(self) -> float:
        return max(
            1.0,
            float(getattr(self.config, "big_plan_area_deviation_penalty", 30.0) or 30.0),
            float(getattr(self.config, "big_plan_fallback_tier_penalty", 500.0) or 500.0) / 10.0,
        )

    def _medium_initial_unit_cost(self, group: BoxGroup, area_no: str) -> float:
        cost = 0.0
        if area_no in getattr(self.problem, "user_voyage_area_requirements", {}).get(group.voyage_id, set()):
            cost -= max(1000.0, float(getattr(self.config, "medium_small_feedback_cap_penalty", 20000.0)) * 0.25)
        tier = self._area_fallback_tier_for_group(group, area_no)
        if tier > 0:
            cost += self._area_fallback_tier_penalty(tier)
        berth = self._berth_key(group.voyage_id)
        if berth:
            distance = self.problem.berth_distances.get((area_no, berth))
            if distance is not None:
                cost += self.config.berth_distance_penalty * distance / 100.0
        if self._area_has_loading_during_window(group.voyage_id, area_no):
            cost += self.config.active_loading_area_penalty
        if self._area_has_loading_after_window(group.voyage_id, area_no):
            cost -= self.config.post_window_loading_area_reward
        return cost

    def _medium_initial_split_cost(self, group: BoxGroup) -> float:
        if self._prefers_concentrated_medium_group(group):
            return float(getattr(self.config, "medium_small_group_area_split_penalty", 500.0))
        return float(getattr(self.config, "group_area_split_penalty", 0.0))

    def _initial_attempt_random(self, random_offset: int):
        import random

        return random.Random(int(getattr(self.config, "seed", 7) or 7) + random_offset)

    def _log_medium_initial_assignment_success(self, method: str, details: dict) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if method == "scip":
            print(
                f"[{stamp}] medium initial assignment built by SCIP "
                f"status={details.get('status')} vars={details.get('variable_count')}",
                flush=True,
            )
            return
        print(
            f"[{stamp}] medium initial assignment built after attempt={details.get('attempt')} "
            f"strategy={details.get('strategy')}",
            flush=True,
        )

    @staticmethod
    def _log_medium_initial_assignment_failure(message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] medium initial assignment failed: {message}", flush=True)

    @staticmethod
    def _configure_initial_scip_output(model) -> None:
        try:
            model.hideOutput()
            return
        except Exception:
            pass
        try:
            model.setParam("display/verblevel", 0)
        except Exception:
            pass

    @staticmethod
    def _initial_scip_solution_count(model) -> int:
        try:
            return int(model.getNSols())
        except Exception:
            try:
                return 1 if model.getBestSol() is not None else 0
            except Exception:
                return 0

    @staticmethod
    def _free_initial_scip_model(model) -> None:
        for method_name in ("freeTransform", "freeProb"):
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                method()
            except Exception:
                pass

    def _allocate_group_to_areas(
        self,
        group: BoxGroup,
        weights: dict[str, int],
        area_load: Counter[str],
        area_size_load: Counter[tuple[str, str]],
        quota_load: Counter[tuple[str, str, str, str]],
        strategy: str = "weighted",
    ) -> Counter[str]:
        allocations: Counter[str] = Counter()
        candidates = [
            area_no for area_no in self._candidate_areas_for_group(group)
            if self._area_free_capacity(group, area_no, area_load, area_size_load, quota_load) > 0
        ]
        if not candidates:
            return allocations

        def free_with_allocations(area_no: str) -> int:
            trial_area_load = Counter(area_load)
            trial_area_size_load = Counter(area_size_load)
            trial_quota_load = Counter(quota_load)
            for allocated_area, allocated_qty in allocations.items():
                trial_area_load[allocated_area] += allocated_qty
                trial_area_size_load[(allocated_area, group.size_mode)] += allocated_qty
                trial_quota_load[(group.voyage_id, group.status, allocated_area, group.big_plan_size_mode)] += allocated_qty
            return self._hard_inherited_area_free_capacity(
                group,
                area_no,
                trial_area_load,
                trial_area_size_load,
                trial_quota_load,
            )

        def preferred_free_with_allocations(area_no: str) -> int:
            trial_quota_load = Counter(quota_load)
            for allocated_area, allocated_qty in allocations.items():
                trial_quota_load[(group.voyage_id, group.status, allocated_area, group.big_plan_size_mode)] += allocated_qty
            quota_key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
            quota_free = self.problem.area_size_quota.get(quota_key, 0) - trial_quota_load[quota_key]
            return min(free_with_allocations(area_no), max(0, quota_free))

        def fill_by_order(area_order: list[str]) -> None:
            remaining = group.demand - sum(allocations.values())
            for area_no in area_order:
                if remaining <= 0:
                    return
                take = min(remaining, free_with_allocations(area_no))
                if take > 0:
                    allocations[area_no] += take
                    remaining -= take

        strategy = strategy or "weighted"
        if strategy == "capacity":
            remaining = group.demand
            ordered = sorted(
                candidates,
                key=lambda area_no: (
                    0 if area_no in weights else 1,
                    0 if free_with_allocations(area_no) >= remaining else 1,
                    -free_with_allocations(area_no),
                    -self.area_size_cap[(area_no, group.size_mode)],
                    -weights.get(area_no, 0),
                    area_no,
                ),
            )
            fill_by_order(ordered)
            return allocations

        if strategy == "concentrated":
            ordered = sorted(
                candidates,
                key=lambda area_no: (
                    0 if area_no in weights else 1,
                    0 if free_with_allocations(area_no) >= group.demand else 1,
                    -free_with_allocations(area_no),
                    -weights.get(area_no, 0),
                    area_no,
                ),
            )
            for area_no in ordered:
                free = free_with_allocations(area_no)
                if free >= group.demand:
                    allocations[area_no] += group.demand
                    return allocations
            fill_by_order(ordered)
            return allocations

        preferred = {area: qty for area, qty in weights.items() if area in candidates and qty > 0}
        if preferred and self._prefers_concentrated_medium_group(group):
            for area_no in sorted(
                preferred,
                key=lambda area: (
                    0 if preferred_free_with_allocations(area) >= group.demand else 1,
                    -preferred_free_with_allocations(area),
                    -weights.get(area, 0),
                    area,
                ),
            ):
                free = preferred_free_with_allocations(area_no)
                take = min(group.demand, free)
                if take > 0:
                    allocations[area_no] += take
                    break
        elif preferred:
            preferred_target = min(group.demand, sum(preferred.values()))
            proportional = self._allocate_by_weights(preferred, preferred_target)
            for area_no, qty in proportional.items():
                free = preferred_free_with_allocations(area_no)
                take = min(qty, free)
                if take > 0:
                    allocations[area_no] += take

        remaining = group.demand - sum(allocations.values())
        if remaining <= 0:
            return allocations

        def fill_key(area_no: str) -> tuple[int, int, int, str]:
            in_big_plan = self._is_big_plan_area_for_group(group, area_no)
            free = self._hard_inherited_area_free_capacity(group, area_no, area_load, area_size_load, quota_load)
            return (
                0 if preferred_free_with_allocations(area_no) > 0 else 1,
                self._area_fallback_tier_for_group(group, area_no),
                0 if free >= remaining else 1,
                area_no,
            )

        for area_no in sorted(candidates, key=fill_key):
            trial_area_load = area_load + allocations
            trial_area_size_load = Counter(area_size_load)
            trial_quota_load = Counter(quota_load)
            for allocated_area, allocated_qty in allocations.items():
                trial_area_size_load[(allocated_area, group.size_mode)] += allocated_qty
                trial_quota_load[(group.voyage_id, group.status, allocated_area, group.big_plan_size_mode)] += allocated_qty
            free = self._hard_inherited_area_free_capacity(
                group,
                area_no,
                trial_area_load,
                trial_area_size_load,
                trial_quota_load,
            )
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
        return max(0, min(total_free, size_free))

    def _hard_inherited_area_free_capacity(
        self,
        group: BoxGroup,
        area_no: str,
        area_load: Counter[str],
        area_size_load: Counter[tuple[str, str]],
        quota_load: Counter[tuple[str, str, str, str]],
    ) -> int:
        free = self._area_free_capacity(group, area_no, area_load, area_size_load, quota_load)
        quota_key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
        quota = self.problem.area_size_quota.get(quota_key, 0)
        if quota > 0:
            free = min(free, max(0, quota - quota_load[quota_key]))
        return free

    def _mutate_medium(self, assignment: MediumAssignment) -> bool:
        """Move a batch of boxes for one coarse group to another feasible area."""
        area_load, area_size_load, quota_load = self._medium_loads(assignment)
        if self._try_place_unassigned_medium(assignment, area_load, area_size_load, quota_load):
            return True

        movable = [
            group for group in self.problem.groups
            if sum(assignment.get(group.group_id, Counter()).values()) > 0
        ]
        if not movable:
            return False
        group = self.random.choice(movable)
        group_assignment = assignment[group.group_id]
        source_areas = [area_no for area_no, qty in group_assignment.items() if qty > 0]
        source_areas.sort(
            key=lambda area_no: self._medium_mutation_source_key(group, area_no, quota_load)
        )
        candidates = list(self._candidate_areas_for_group(group))
        candidates.sort(
            key=lambda area_no: self._medium_mutation_target_key(group, area_no, quota_load)
        )

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

    def _medium_mutation_source_key(
        self,
        group: BoxGroup,
        area_no: str,
        quota_load: Counter[tuple[str, str, str, str]],
    ) -> tuple[int, int, float, str]:
        quota_key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
        target = self._effective_big_plan_area_size_targets().get(quota_key, 0)
        load = quota_load.get(quota_key, 0)
        excess = load - target
        return (
            0 if excess > 0 else 1,
            0 if target <= 0 else 1,
            -float(excess),
            area_no,
        )

    def _medium_mutation_target_key(
        self,
        group: BoxGroup,
        area_no: str,
        quota_load: Counter[tuple[str, str, str, str]],
    ) -> tuple[int, int, float, str]:
        quota_key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
        target = self._effective_big_plan_area_size_targets().get(quota_key, 0)
        load = quota_load.get(quota_key, 0)
        deficit = target - load
        return (
            0 if deficit > 0 else 1,
            self._area_fallback_tier_for_group(group, area_no),
            -float(deficit),
            area_no,
        )

    def _try_place_unassigned_medium(
        self,
        assignment: MediumAssignment,
        area_load: Counter[str],
        area_size_load: Counter[tuple[str, str]],
        quota_load: Counter[tuple[str, str, str, str]],
    ) -> bool:
        groups = list(self.problem.groups)
        self.random.shuffle(groups)
        for group in groups:
            assigned = sum(assignment.get(group.group_id, Counter()).values())
            shortage = group.demand - assigned
            if shortage <= 0:
                continue
            candidates = list(self._candidate_areas_for_group(group))
            self.random.shuffle(candidates)
            max_add = max(1, min(shortage, max(5, math.ceil(group.demand * 0.20))))
            add_qty = self.random.randint(1, max_add)
            for target_area in candidates[:80]:
                free = self._hard_inherited_area_free_capacity(group, target_area, area_load, area_size_load, quota_load)
                take = min(add_qty, free)
                if take <= 0:
                    continue
                assignment.setdefault(group.group_id, Counter())[target_area] += take
                area_load[target_area] += take
                area_size_load[(target_area, group.size_mode)] += take
                quota_load[(group.voyage_id, group.status, target_area, group.big_plan_size_mode)] += take
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
        quota = self.problem.area_size_quota.get(quota_key, 0)
        if quota > 0 and quota_load[quota_key] + qty > quota:
            return False
        return True

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
        voyage_area_load: Counter[tuple[str, str]] = Counter()
        for group_id, area_counts in assignment.items():
            group = groups_by_id[group_id]
            assigned = sum(area_counts.values())
            if assigned != group.demand:
                return False
            candidates = self._candidate_areas_for_group(group)
            for area_no, qty in area_counts.items():
                if qty <= 0:
                    return False
                if area_no not in candidates:
                    return False
                area_load[area_no] += qty
                voyage_area_load[(group.voyage_id, area_no)] += qty
                area_size_load[(area_no, group.size_mode)] += qty
                quota_key = (group.voyage_id, group.status, area_no, group.big_plan_size_mode)
                quota_load[quota_key] += qty
                quota = self.problem.area_size_quota.get(quota_key, 0)
                if quota > 0 and quota_load[quota_key] > quota:
                    return False
        for area_no, load in area_load.items():
            if load > self.area_total_cap[area_no]:
                return False
        for key, load in area_size_load.items():
            if load > self.area_size_cap[key]:
                return False
        for voyage_id, areas in getattr(self.problem, "user_voyage_area_requirements", {}).items():
            for area_no in areas:
                if voyage_area_load[(voyage_id, area_no)] <= 0:
                    return False
        return True

    def medium_energy(self, assignment: MediumAssignment, include_small_proxy: bool = True) -> float:
        return self.medium_energy_components(assignment, include_small_proxy=include_small_proxy)["total"]

    def medium_energy_components(self, assignment: MediumAssignment, include_small_proxy: bool = True) -> dict[str, float]:
        group_areas: defaultdict[str, set[str]] = defaultdict(set)
        group_area_count: defaultdict[tuple[str, str], int] = defaultdict(int)
        area_size_count: Counter[tuple[str, str, str, str]] = Counter()
        voyage_areas: set[tuple[str, str]] = set()
        coarse_area_count: defaultdict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
        components = {
            "big_plan_fallback_tier": 0.0,
            "strict_feedback": 0.0,
            "repair_failure_feedback": 0.0,
            "medium_small_learned_cap": 0.0,
            "group_area_split": 0.0,
            "group_area_shape": 0.0,
            "big_plan_deviation": 0.0,
            "berth_distance": 0.0,
            "active_loading_area": 0.0,
            "post_window_reward": 0.0,
            "small_plan_proxy": 0.0,
        }
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
                tier = self._area_fallback_tier_for_group(group, area_no)
                if tier > 0:
                    components["big_plan_fallback_tier"] += self._area_fallback_tier_penalty(tier) * qty
                exact_strict_feedback = getattr(self, "small_plan_strict_area_feedback", {}).get(
                    (group.voyage_id, group.status, group.port, group.size, area_no),
                    0,
                )
                broad_strict_feedback = getattr(self, "small_plan_strict_area_feedback", {}).get(
                    (group.voyage_id, group.status, "*", group.size, area_no),
                    0,
                )
                strict_feedback_count = max(exact_strict_feedback, broad_strict_feedback)
                if strict_feedback_count:
                    penalty = getattr(self.config, "small_plan_strict_feedback_penalty", 45.0)
                    components["strict_feedback"] += penalty * strict_feedback_count * qty
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
                    multiplier = getattr(self.config, "small_plan_repair_failure_feedback_multiplier", 4.0)
                    components["repair_failure_feedback"] += (
                        self.config.small_plan_feedback_penalty * multiplier * feedback_count * qty
                    )
        learned_caps = getattr(self, "medium_small_learned_area_size_caps", {})
        if learned_caps:
            learned_penalty = float(getattr(self.config, "medium_small_feedback_cap_penalty", 300000.0) or 300000.0)
            for key, cap in learned_caps.items():
                excess = area_size_count.get(key, 0) - int(cap)
                if excess > 0:
                    components["medium_small_learned_cap"] += learned_penalty * excess

        for group in self.problem.groups:
            components["group_area_split"] += self.config.group_area_split_penalty * max(
                0,
                len(group_areas[group.group_id]) - 1,
            )
            if self._prefers_concentrated_medium_group(group):
                components["group_area_shape"] += self._group_area_concentration_energy(group, group_area_count)
            else:
                components["group_area_shape"] += self._group_area_balance_energy(group, group_area_count)
        components["big_plan_deviation"] += self._big_plan_deviation_energy(area_size_count)

        for voyage_id, area_no in voyage_areas:
            berth = self._berth_key(voyage_id)
            if berth:
                distance = self.problem.berth_distances.get((area_no, berth))
                if distance is not None:
                    components["berth_distance"] += self.config.berth_distance_penalty * distance / 100.0
            if self._area_has_loading_during_window(voyage_id, area_no):
                components["active_loading_area"] += self.config.active_loading_area_penalty
            if self._area_has_loading_after_window(voyage_id, area_no):
                components["post_window_reward"] -= self.config.post_window_loading_area_reward
        required_penalty = max(
            100000.0,
            float(getattr(self.config, "medium_small_feedback_cap_penalty", 20000.0) or 20000.0) * 10.0,
        )
        for voyage_id, areas in getattr(self.problem, "user_voyage_area_requirements", {}).items():
            for area_no in areas:
                if (voyage_id, area_no) not in voyage_areas:
                    components["user_required_area_missing"] += required_penalty

        small_proxy = getattr(self, "_small_plan_proxy_energy", None) if include_small_proxy else None
        if small_proxy is not None:
            components["small_plan_proxy"] += small_proxy(coarse_area_count)
        components["total"] = sum(components.values())
        return components

    def _candidate_areas_for_group(self, group: BoxGroup) -> set[str]:
        cached = getattr(self, "_candidate_area_cache", {}).get(group.group_id)
        if cached is not None:
            return cached
        return self._compute_candidate_areas_for_group(group)

    def _compute_candidate_areas_for_group(self, group: BoxGroup) -> set[str]:
        return {
            area_no
            for area_no in self.bays_by_area
            if self.area_size_cap[(area_no, group.size_mode)] > 0
            and self.area_total_cap[area_no] > 0
            and self._user_area_policy_allows(group.voyage_id, area_no)
            and (
                self._area_supports_group_flow(group, area_no)
                or self._user_area_policy_forces_support(group.voyage_id, area_no)
            )
        }

    def _user_area_policy_allows(self, voyage_id: str, area_no: str) -> bool:
        allow = getattr(self.problem, "user_voyage_area_allowlist", {}).get(voyage_id, set())
        block = getattr(self.problem, "user_voyage_area_blocklist", {}).get(voyage_id, set())
        if area_no in block:
            return False
        if allow and area_no not in allow:
            return False
        return True

    def _user_area_policy_forces_support(self, voyage_id: str, area_no: str) -> bool:
        allow = getattr(self.problem, "user_voyage_area_allowlist", {}).get(voyage_id, set())
        required = getattr(self.problem, "user_voyage_area_requirements", {}).get(voyage_id, set())
        return area_no in allow or area_no in required

    def _area_supports_group_flow(self, group: BoxGroup, area_no: str) -> bool:
        if self._is_big_plan_area_for_group(group, area_no):
            return True
        functions = self.problem.area_functions.get(area_no, set())
        if group.status == "OF":
            return "OF" in functions
        return group.status in functions

    def _is_big_plan_area_for_group(self, group: BoxGroup, area_no: str) -> bool:
        if area_no in self.problem.assigned_areas.get((group.voyage_id, group.status), set()):
            return True
        return False

    def _is_any_big_plan_area(self, area_no: str) -> bool:
        return any(row.area_no == area_no and row.planned_boxes > 0 for row in self.problem.big_plan)

    def _area_fallback_tier_for_group(self, group: BoxGroup, area_no: str) -> int:
        big_plan_size = getattr(group, "big_plan_size_mode", None)
        if big_plan_size is None:
            size = getattr(group, "size", "40")
            big_plan_size = "40" if size == "45" else size if size in {"20", "40"} else "40"
        quota_key = (group.voyage_id, group.status, area_no, big_plan_size)
        if self.problem.area_size_quota.get(quota_key, 0) > 0:
            return 0
        if self._is_big_plan_area_for_group(group, area_no):
            return 1
        if self._is_any_big_plan_area(area_no):
            return 2
        return 3

    def _area_fallback_tier_penalty(self, tier: int) -> float:
        base = float(getattr(self.config, "big_plan_fallback_tier_penalty", 500.0) or 500.0)
        return base * max(0, int(tier))

    def _big_plan_deviation_energy(self, area_size_count: Counter[tuple[str, str, str, str]]) -> float:
        energy = 0.0
        targets = self._effective_big_plan_area_size_targets()
        keys = set(area_size_count) | set(targets)
        for key in keys:
            target = targets.get(key, 0)
            actual = area_size_count.get(key, 0)
            if target == 0 and actual == 0:
                continue
            energy += self.config.big_plan_area_deviation_penalty * abs(actual - target)
        return energy

    def _effective_big_plan_area_size_targets(self) -> Counter[tuple[str, str, str, str]]:
        cached = getattr(self, "_effective_big_plan_area_size_target_cache", None)
        if cached is not None:
            return cached

        demand_by_key: Counter[tuple[str, str, str]] = Counter()
        for group in self.problem.groups:
            demand_by_key[(group.voyage_id, group.status, group.big_plan_size_mode)] += group.demand

        quota_by_key: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        for (voyage_id, flow, area_no, size_mode), qty in self.problem.area_size_quota.items():
            if qty > 0:
                quota_by_key[(voyage_id, flow, size_mode)][area_no] += qty

        targets: Counter[tuple[str, str, str, str]] = Counter()
        for key, area_weights in quota_by_key.items():
            voyage_id, flow, size_mode = key
            demand = demand_by_key.get(key, 0)
            quota_total = sum(area_weights.values())
            if demand <= 0 or quota_total <= 0:
                continue
            target_total = min(demand, quota_total)
            if target_total == quota_total:
                allocation = dict(area_weights)
            else:
                allocation = self._allocate_by_weights(dict(area_weights), target_total)
            for area_no, qty in allocation.items():
                if qty > 0:
                    targets[(voyage_id, flow, area_no, size_mode)] = qty
        self._effective_big_plan_area_size_target_cache = targets
        return targets

    def _group_area_balance_energy(self, group: BoxGroup, group_area_count: dict[tuple[str, str], int]) -> float:
        counts = [
            (area_no, qty)
            for (group_id, area_no), qty in group_area_count.items()
            if group_id == group.group_id and qty > 0
        ]
        if not counts or group.demand <= 0:
            return 0.0

        penalty = 0.0
        min_boxes = max(0, int(getattr(self.config, "medium_large_group_min_area_boxes", 10) or 0))
        if min_boxes > 0:
            small_area_penalty = float(
                getattr(self.config, "medium_large_group_small_area_penalty", 300.0)
            ) / max(1.0, min_boxes)
            for _area_no, qty in counts:
                penalty += small_area_penalty * max(0, min_boxes - qty)

        if len(counts) <= 1:
            return penalty

        pair_penalty = self.config.group_area_balance_penalty / max(1.0, group.demand) / max(1, len(counts) - 1)
        for left_index, (_left_area, left_qty) in enumerate(counts):
            for _right_area, right_qty in counts[left_index + 1 :]:
                penalty += pair_penalty * abs(left_qty - right_qty)
        return penalty

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
        split_penalty = getattr(self.config, "medium_small_group_area_split_penalty", 500.0)
        fragment_penalty = getattr(self.config, "medium_small_group_fragment_penalty", 20.0)
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
