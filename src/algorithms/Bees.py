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
    greedy_initial_sites: int = 8
    greedy_start_candidates: int = 32
    greedy_candidate_list_size: int = 1

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
        if self.greedy_initial_sites < 0:
            raise ValueError("greedy_initial_sites cannot be negative.")
        if self.greedy_initial_sites > self.n_bees:
            raise ValueError("greedy_initial_sites cannot be greater than n_bees.")
        if self.greedy_start_candidates < 0:
            raise ValueError("greedy_start_candidates cannot be negative.")
        if self.greedy_candidate_list_size <= 0:
            raise ValueError("greedy_candidate_list_size must be greater than 0.")


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

    def _get_distance_matrix(self) -> np.ndarray | None:
        if not hasattr(self.problem, "distance_matrix"):
            return None

        matrix = self.problem.distance_matrix
        if matrix is None:
            return None
        return np.asarray(matrix)

    def _rotate_to_start_node(self, solution: Solution) -> Solution:
        start_node = int(getattr(self.problem, "start_node", 0))
        positions = np.where(solution == start_node)[0]
        if len(positions) == 0:
            return np.copy(solution)
        return np.roll(solution, -int(positions[0])).astype(np.int32, copy=False)

    def _create_greedy_solution(
        self,
        start_city: int,
        distance_matrix: np.ndarray,
    ) -> Solution:
        num_cities = len(distance_matrix)
        unvisited = np.ones(num_cities, dtype=bool)
        unvisited[start_city] = False

        tour = np.empty(num_cities, dtype=np.int32)
        tour[0] = start_city
        current_city = start_city

        for position in range(1, num_cities):
            candidates = np.flatnonzero(unvisited)
            distances = distance_matrix[current_city, candidates]

            if self.hyperparameters.greedy_candidate_list_size == 1:
                next_city = int(candidates[np.argmin(distances)])
            else:
                nearest_order = np.argsort(distances, kind="stable")
                list_size = min(
                    self.hyperparameters.greedy_candidate_list_size,
                    len(nearest_order),
                )
                nearest_candidates = candidates[nearest_order[:list_size]]
                nearest_distances = distances[nearest_order[:list_size]]
                weights = 1.0 / np.maximum(nearest_distances, 1e-12)
                weights = weights / np.sum(weights)
                next_city = int(self.problem.rng.choice(nearest_candidates, p=weights))

            tour[position] = next_city
            unvisited[next_city] = False
            current_city = next_city

        return self._rotate_to_start_node(tour)

    def _create_greedy_sites(self) -> list[EvaluatedSolution]:
        params = self.hyperparameters
        if params.greedy_initial_sites == 0:
            return []

        distance_matrix = self._get_distance_matrix()
        if distance_matrix is None:
            return []

        num_cities = len(distance_matrix)
        if params.greedy_start_candidates == 0 or params.greedy_start_candidates >= num_cities:
            start_cities = np.arange(num_cities)
        else:
            required = params.greedy_start_candidates
            start_node = int(getattr(self.problem, "start_node", 0))
            other_cities = np.array(
                [city for city in range(num_cities) if city != start_node],
                dtype=np.int32,
            )
            self.problem.rng.shuffle(other_cities)
            sampled = other_cities[: max(0, required - 1)]
            start_cities = np.concatenate(([start_node], sampled))

        candidates: list[EvaluatedSolution] = []
        for start_city in start_cities:
            if not self._has_evaluation_budget():
                break

            solution = self._create_greedy_solution(int(start_city), distance_matrix)
            evaluated_site = self._evaluate(solution)
            if evaluated_site is not None:
                candidates.append(evaluated_site)

        candidates.sort(key=lambda site: site[0])
        return candidates[: params.greedy_initial_sites]

    def _create_neighbour(self, solution: Solution) -> Solution:
        neighbour = np.copy(solution)
        for _ in range(self.hyperparameters.neighbourhood_depth):
            if self.hyperparameters.neighbourhood_type == "two_opt":
                neighbour = self._get_two_opt_neighbour(neighbour)
            else:
                neighbour = self._get_swap_neighbour(neighbour)
        return np.copy(neighbour)

    def _get_two_opt_neighbour(self, solution: Solution) -> Solution:
        if hasattr(self.problem, "get_neighbour_swap"):
            return self.problem.get_neighbour(solution, self.problem.rng)

        neighbour = np.copy(solution)
        if len(neighbour) <= 3:
            return self._get_swap_neighbour(neighbour)

        first, second = self.problem.rng.choice(
            np.arange(1, len(neighbour)),
            size=2,
            replace=False,
        )
        left, right = sorted((first, second))
        neighbour[left : right + 1] = neighbour[left : right + 1][::-1]
        return neighbour

    def _get_swap_neighbour(self, solution: Solution) -> Solution:
        if hasattr(self.problem, "get_neighbour_swap"):
            return self.problem.get_neighbour_swap(solution, self.problem.rng)
        return self.problem.get_neighbour(solution)

    def _initialize_population(self) -> list[EvaluatedSolution]:
        population = self._create_greedy_sites()

        while len(population) < self.hyperparameters.n_bees:
            site = self._create_random_site()
            if site is None:
                break
            population.append(site)

        return population

    def _search_site(self, site: EvaluatedSolution, recruits: int) -> EvaluatedSolution:
        best_cost, best_solution = site[0], np.copy(site[1])
        current_cost, current_solution = best_cost, np.copy(best_solution)

        for _ in range(recruits):
            if not self._has_evaluation_budget():
                break

            neighbour = self._create_neighbour(current_solution)
            evaluated_neighbour = self._evaluate(neighbour)
            if evaluated_neighbour is None:
                break

            neighbour_cost, neighbour_solution = evaluated_neighbour
            if neighbour_cost < current_cost:
                current_cost = neighbour_cost
                current_solution = np.copy(neighbour_solution)

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
