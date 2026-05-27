from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


type Solution = np.ndarray
type EvaluatedSolution = tuple[float, Solution]


@dataclass(frozen=True)
class BeesHyperparameters:
    n_bees: int = 40
    n_elite: int = 3
    n_best: int = 12
    elite_neigh: int = 25
    best_neigh: int = 10
    neighbourhood_depth: int = 1
    neighbourhood_type: str = "two_opt"

    def __post_init__(self) -> None:
        if self.n_bees <= 0:
            raise ValueError("n_bees must be greater than 0.")
        if self.n_elite < 0:
            raise ValueError("n_elite cannot be negative.")
        if self.n_best <= 0:
            raise ValueError("n_best must be greater than 0.")
        if self.n_elite > self.n_best:
            raise ValueError("n_elite cannot be greater than n_best.")
        if self.n_best > self.n_bees:
            raise ValueError("n_best cannot be greater than n_bees.")
        if self.elite_neigh < 0 or self.best_neigh < 0:
            raise ValueError("Neighbourhood sizes cannot be negative.")
        if self.neighbourhood_depth <= 0:
            raise ValueError("neighbourhood_depth must be greater than 0.")
        if self.neighbourhood_type not in {"two_opt", "swap"}:
            raise ValueError("neighbourhood_type must be either 'two_opt' or 'swap'.")


class BeesOptimizer:
    def __init__(
        self,
        problem: Any,
        max_fes: int,
        hyperparameters: BeesHyperparameters,
    ) -> None:
        if max_fes <= 0:
            raise ValueError("max_fes must be greater than 0.")

        self.problem = problem
        self.max_fes = max_fes
        self.hyperparameters = hyperparameters

        self.best_cost = float("inf")
        self.best_solution: Solution | None = None
        self.history: list[dict[str, Any]] = []
        self.fes_total_count = 0
        self.cycles = 0

    def _has_evaluation_budget(self) -> bool:
        return self.problem.fe_count < self.max_fes

    def _evaluate(self, solution: Solution) -> EvaluatedSolution | None:
        if not self._has_evaluation_budget():
            return None

        solution_copy = np.copy(solution)
        cost = float(self.problem.evaluate(solution_copy))
        self.fes_total_count = self.problem.fe_count

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_solution = np.copy(solution_copy)

        return cost, solution_copy

    def _create_random_site(self) -> EvaluatedSolution | None:
        if not self._has_evaluation_budget():
            return None

        solution = self.problem.create_random_solution()
        return self._evaluate(solution)

    def _create_neighbour(self, solution: Solution) -> Solution:
        neighbour = np.copy(solution)
        for _ in range(self.hyperparameters.neighbourhood_depth):
            if self.hyperparameters.neighbourhood_type == "two_opt":
                neighbour = self._get_two_opt_neighbour(neighbour)
            else:
                neighbour = self.problem.get_neighbour(neighbour)
        return np.copy(neighbour)

    def _get_two_opt_neighbour(self, solution: Solution) -> Solution:
        neighbour = np.copy(solution)
        if len(neighbour) <= 3:
            return self.problem.get_neighbour(neighbour)

        first, second = self.problem.rng.choice(
            np.arange(1, len(neighbour)),
            size=2,
            replace=False,
        )
        left, right = sorted((first, second))
        neighbour[left : right + 1] = neighbour[left : right + 1][::-1]
        return neighbour

    def _initialize_population(self) -> list[EvaluatedSolution]:
        population: list[EvaluatedSolution] = []

        while len(population) < self.hyperparameters.n_bees:
            site = self._create_random_site()
            if site is None:
                break
            population.append(site)

        return population

    def _search_site(self, site: EvaluatedSolution, recruits: int) -> EvaluatedSolution:
        best_cost, best_solution = site[0], np.copy(site[1])

        for _ in range(recruits):
            if not self._has_evaluation_budget():
                break

            neighbour = self._create_neighbour(site[1])
            evaluated_neighbour = self._evaluate(neighbour)
            if evaluated_neighbour is None:
                break

            neighbour_cost, neighbour_solution = evaluated_neighbour
            if neighbour_cost < best_cost:
                best_cost = neighbour_cost
                best_solution = np.copy(neighbour_solution)

        return best_cost, best_solution

    def _build_next_population(
        self,
        population: list[EvaluatedSolution],
    ) -> list[EvaluatedSolution]:
        params = self.hyperparameters
        population.sort(key=lambda site: site[0])
        selected_sites = population[: params.n_best]
        elite_sites = selected_sites[: params.n_elite]
        best_sites = selected_sites[params.n_elite :]

        next_population: list[EvaluatedSolution] = []

        for site in elite_sites:
            next_population.append(self._search_site(site, params.elite_neigh))

        for site in best_sites:
            next_population.append(self._search_site(site, params.best_neigh))

        while len(next_population) < params.n_bees:
            scout = self._create_random_site()
            if scout is None:
                break
            next_population.append(scout)

        return next_population

    def _save_history(self, population: list[EvaluatedSolution]) -> None:
        population_costs = [cost for cost, _ in population]
        best_solution = None
        if self.best_solution is not None:
            best_solution = np.copy(self.best_solution)

        self.history.append(
            {
                "cycle": self.cycles,
                "fes": self.problem.fe_count,
                "best_cost": self.best_cost,
                "best_solution": best_solution,
                "population_costs": population_costs,
                "population_size": len(population),
                "is_full_population": len(population) == self.hyperparameters.n_bees,
                "avg_population_cost": float(np.mean(population_costs))
                if population_costs
                else float("inf"),
                "std_population_cost": float(np.std(population_costs))
                if population_costs
                else 0.0,
            }
        )

    def solve(self) -> tuple[float, Solution | None]:
        self.problem.fe_count = 0
        self.best_cost = float("inf")
        self.best_solution = None
        self.history.clear()
        self.fes_total_count = 0
        self.cycles = 0

        population = self._initialize_population()
        if not population:
            return self.best_cost, self.best_solution

        self._save_history(population)

        while self._has_evaluation_budget():
            next_population = self._build_next_population(population)
            if not next_population:
                break

            population = next_population
            self.cycles += 1
            self._save_history(population)

        return self.best_cost, self.best_solution

    def run(self) -> tuple[Solution | None, float]:
        best_cost, best_solution = self.solve()
        return best_solution, best_cost
