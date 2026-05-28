from pathlib import Path
import sys
import time

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_DIR / "src"
CORE_DIR = SRC_DIR / "core"
for path in (PROJECT_DIR, SRC_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithms.Bees import BeesHyperparameters, BeesOptimizer
from analysis.BeesResultAnalyzer import BeesResultAnalyzer
from core.TSPProblem import TSPProblem


GENERATE_PLOTS = True


def read_optimal_tour(path: Path) -> np.ndarray:
    nodes: list[int] = []
    reading_tour = False

    with open(path, encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line == "TOUR_SECTION":
                reading_tour = True
                continue
            if not reading_tour:
                continue
            if line in {"-1", "EOF"}:
                break
            nodes.extend(int(value) for value in line.split())

    return np.array(nodes, dtype=np.int32) - 1


def run_bees() -> None:
    problem_name = "a280"
    max_fes = 50_000
    problem_path = PROJECT_DIR / "data" / "tsplib95" / "tasks" / f"{problem_name}.tsp"
    solution_path = PROJECT_DIR / "data" / "tsplib95" / "solutions" / f"{problem_name}.opt.tour"

    problem = TSPProblem(file_path=problem_path, seed=42)
    problem.file_path = problem_path
    hyperparameters = BeesHyperparameters(
        n_bees=50,
        n_elite=5,
        n_best=18,
        elite_neigh=80,
        best_neigh=30,
        neighbourhood_depth=2,
        neighbourhood_type="two_opt",
    )

    optimizer = BeesOptimizer(
        problem=problem,
        max_fes=max_fes,
        hyperparameters=hyperparameters,
    )

    start_time = time.perf_counter()
    best_cost, best_solution = optimizer.solve()
    execution_time = time.perf_counter() - start_time

    optimal_cost = None
    if solution_path.exists():
        optimal_solution = read_optimal_tour(solution_path)
        optimal_cost = problem._calculate_fitness(optimal_solution)

    analyzer = BeesResultAnalyzer(
        optimizer=optimizer,
        target_path=PROJECT_DIR / "results" / "Bees",
        exec_time=execution_time,
        series_name=problem_name,
        optimal_cost=optimal_cost,
    )
    if GENERATE_PLOTS:
        analyzer.plot_all_and_save()
    else:
        analyzer.save_results_json()

    gap_percent = None
    if optimal_cost is not None:
        gap_percent = ((best_cost - optimal_cost) / optimal_cost) * 100

    print(f"Best cost: {best_cost}")
    if optimal_cost is not None:
        print(f"Optimal cost: {optimal_cost}")
        print(f"Gap to optimum: {gap_percent:.2f}%")
    print(f"Best solution: {best_solution}")
    print(f"Function evaluations: {problem.fe_count}")
    print(f"Results saved to: {PROJECT_DIR / 'results' / 'Bees' / problem_name}")


if __name__ == "__main__":
    run_bees()
