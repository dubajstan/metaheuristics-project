from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_DIR / "src"
for path in (PROJECT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.algorithms.Bees import BeesHyperparameters, BeesOptimizer
from src.core.TSPProblem import TSPProblem


PROBLEMS = {
    "att48": {
        "tsp": PROJECT_DIR / "data" / "tsplib95" / "tasks" / "att48.tsp",
        "opt": PROJECT_DIR / "data" / "tsplib95" / "solutions" / "att48.opt.tour",
    },
    "eil101": {
        "tsp": PROJECT_DIR / "data" / "tsplib95" / "tasks" / "eil101.tsp",
        "opt": PROJECT_DIR / "data" / "tsplib95" / "solutions" / "eil101.opt.tour",
    },
    "a280": {
        "tsp": PROJECT_DIR / "data" / "tsplib95" / "tasks" / "a280.tsp",
        "opt": PROJECT_DIR / "data" / "tsplib95" / "solutions" / "a280.opt.tour",
    },
}

RUNS_COUNT = 10
MAX_FES = 50_000
PIPELINE_RESULTS_DIR = PROJECT_DIR / "pipeline_results"
PLOTS_DIR = PROJECT_DIR / "plots"

DEFAULT_HYPERPARAMETERS = BeesHyperparameters(
    n_bees=12,
    n_elite=2,
    n_best=4,
    elite_neigh=900,
    best_neigh=220,
    neighbourhood_depth=1,
    neighbourhood_type="two_opt",
    greedy_initial_sites=4,
    greedy_start_candidates=0,
    greedy_candidate_list_size=1,
)


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


def rpd(cost: float, optimal_cost: float) -> float:
    return ((cost - optimal_cost) / optimal_cost) * 100.0


def run_single_bees(
    run_id: int,
    seed: int,
    problem_path: Path,
    max_fes: int,
    hyperparameters: BeesHyperparameters,
) -> dict[str, Any]:
    problem = TSPProblem(problem_path, seed=seed)
    problem.file_path = problem_path
    optimizer = BeesOptimizer(problem, max_fes, hyperparameters)

    start_time = time.perf_counter()
    best_cost, _best_solution = optimizer.solve()
    elapsed_time = time.perf_counter() - start_time

    return {
        "run_id": run_id,
        "seed": seed,
        "best_cost": float(best_cost),
        "fes_used": int(problem.fe_count),
        "cycles": int(optimizer.cycles),
        "time_seconds": float(elapsed_time),
        "history": optimizer.history,
    }


def get_history_series(history: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in history if np.isfinite(record[key])]


def generate_pipeline_plots(
    run_history: list[dict[str, Any]],
    problem_name: str,
    run_id: int,
    optimal_cost: float,
    found_cost: float,
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fes = get_history_series(run_history, "fes")
    best_costs = get_history_series(run_history, "best_cost")
    avg_costs = get_history_series(run_history, "avg_population_cost")
    std_costs = get_history_series(run_history, "std_population_cost")

    plt.figure(figsize=(11, 6))
    plt.plot(fes, best_costs, color="forestgreen", linewidth=2.0, label="Best cost")
    plt.axhline(optimal_cost, color="black", linestyle="--", linewidth=1.4, label=f"Known optimum ({optimal_cost:.0f})")
    plt.axhline(found_cost, color="crimson", linestyle="-", linewidth=1.2, label=f"Run result ({found_cost:.0f})")
    plt.title(f"[{problem_name.upper()}] Bees Algorithm Convergence - Run {run_id:02d}")
    plt.xlabel("Function evaluations")
    plt.ylabel("Cost")
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{problem_name}_run_{run_id:02d}_convergence.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(fes, best_costs, color="forestgreen", linewidth=2.0, label="Best cost")
    axes[0].plot(fes, avg_costs, color="royalblue", linewidth=1.5, label="Average population cost")
    axes[0].axhline(optimal_cost, color="black", linestyle="--", linewidth=1.3, label=f"Known optimum ({optimal_cost:.0f})")
    axes[0].set_title(f"[{problem_name.upper()}] Bees Algorithm Diagnostics - Run {run_id:02d}")
    axes[0].set_ylabel("Cost")
    axes[0].grid(True, linestyle="--", alpha=0.55)
    axes[0].legend()

    axes[1].plot(fes, std_costs, color="darkorange", linewidth=1.8, label="Population cost std. dev.")
    axes[1].set_xlabel("Function evaluations")
    axes[1].set_ylabel("Standard deviation")
    axes[1].grid(True, linestyle="--", alpha=0.55)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / f"{problem_name}_run_{run_id:02d}_diagnostics.png", dpi=300)
    plt.close()


def generate_best_run_convergence(
    results: list[dict[str, Any]],
    problem_name: str,
    optimal_cost: float,
    save_dir: Path,
) -> None:
    best_run = min(results, key=lambda result: result["best_cost"])
    history = best_run["history"]
    fes = get_history_series(history, "fes")
    best_costs = get_history_series(history, "best_cost")

    plt.figure(figsize=(11, 6))
    plt.plot(fes, best_costs, color="forestgreen", linewidth=2.2, label=f"Best run {best_run['run_id']:02d}")
    plt.axhline(optimal_cost, color="black", linestyle="--", linewidth=1.4, label=f"Known optimum ({optimal_cost:.0f})")
    plt.title(f"[{problem_name.upper()}] Best Bees Algorithm Run")
    plt.xlabel("Function evaluations")
    plt.ylabel("Cost")
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{problem_name}_best_run_convergence.png", dpi=300)
    plt.close()


def generate_summary_plots(
    results: list[dict[str, Any]],
    problem_name: str,
    optimal_cost: float,
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    for result in results:
        fes = get_history_series(result["history"], "fes")
        best_costs = get_history_series(result["history"], "best_cost")
        plt.plot(fes, best_costs, linewidth=1.0, alpha=0.45)
    plt.axhline(optimal_cost, color="black", linestyle="--", linewidth=1.4, label=f"Known optimum ({optimal_cost:.0f})")
    plt.title(f"Bees Algorithm: Convergence of Best Cost - {problem_name}")
    plt.xlabel("Function evaluations")
    plt.ylabel("Cost")
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{problem_name}_01_convergence.png", dpi=300)
    plt.close()

    best_run = min(results, key=lambda result: result["best_cost"])
    best_history = best_run["history"]
    fes = get_history_series(best_history, "fes")
    avg_costs = get_history_series(best_history, "avg_population_cost")
    best_costs = get_history_series(best_history, "best_cost")
    std_costs = get_history_series(best_history, "std_population_cost")

    plt.figure(figsize=(11, 6))
    plt.plot(fes, avg_costs, color="royalblue", linewidth=2.0, label="Average population cost")
    plt.plot(fes, best_costs, color="forestgreen", linestyle="--", linewidth=1.7, label="Best cost")
    plt.axhline(optimal_cost, color="black", linestyle="--", linewidth=1.3, label=f"Known optimum ({optimal_cost:.0f})")
    plt.title(f"Bees Algorithm: Average Population Cost - {problem_name}")
    plt.xlabel("Function evaluations")
    plt.ylabel("Cost")
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{problem_name}_02_population_average.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(fes, std_costs, color="darkorange", linewidth=2.0, label="Population cost std. dev.")
    plt.title(f"Bees Algorithm: Population Cost Standard Deviation - {problem_name}")
    plt.xlabel("Function evaluations")
    plt.ylabel("Standard deviation")
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{problem_name}_03_population_std.png", dpi=300)
    plt.close()

    rpds = [rpd(result["best_cost"], optimal_cost) for result in results]
    plt.figure(figsize=(10, 6))
    plt.bar([result["run_id"] for result in results], rpds, color="slateblue")
    plt.title(f"Bees Algorithm: Final RPD per Run - {problem_name}")
    plt.xlabel("Run")
    plt.ylabel("RPD [%]")
    plt.grid(axis="y", linestyle="--", alpha=0.55)
    plt.tight_layout()
    plt.savefig(save_dir / f"{problem_name}_04_rpd_distribution.png", dpi=300)
    plt.close()


def generate_json_report(
    results: list[dict[str, Any]],
    problem_name: str,
    max_fes: int,
    optimal_cost: float,
    hyperparameters: BeesHyperparameters,
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    costs = [result["best_cost"] for result in results]
    times = [result["time_seconds"] for result in results]
    fes_counts = [result["fes_used"] for result in results]
    cycles = [result["cycles"] for result in results]
    rpds = [rpd(cost, optimal_cost) for cost in costs]

    report = {
        "algorithm": "Bees Algorithm",
        "problem_name": problem_name,
        "runs_count": len(results),
        "max_fes": max_fes,
        "optimal_cost": optimal_cost,
        "hyperparameters": asdict(hyperparameters),
        "summary_statistics": {
            "best_result": float(np.min(costs)),
            "best_result_rpd_percentage": float(np.min(rpds)),
            "average_result": float(np.mean(costs)),
            "average_result_rpd_percentage": float(np.mean(rpds)),
            "average_time_seconds": float(np.mean(times)),
            "average_fes_count": float(np.mean(fes_counts)),
            "average_cycles": float(np.mean(cycles)),
            "std_dev_rpd_percentage": float(np.std(rpds)),
        },
        "raw_data_from_all_runs": [
            {
                "run_id": result["run_id"],
                "seed": result["seed"],
                "best_cost": float(result["best_cost"]),
                "rpd_percentage": float(rpd(result["best_cost"], optimal_cost)),
                "time_seconds": float(result["time_seconds"]),
                "fes_used": int(result["fes_used"]),
                "cycles": int(result["cycles"]),
                "stagnation_triggered": result["fes_used"] < max_fes,
            }
            for result in results
        ],
    }

    filepath = save_dir / f"{problem_name}_bees_results.json"
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)
    print(f"Saved report JSON: {filepath}")


def run_pipeline() -> None:
    PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for problem_name, paths in PROBLEMS.items():
        print("\n=============================================")
        print(f" BEES PIPELINE START: {problem_name.upper()}")
        print("=============================================")

        reference_problem = TSPProblem(paths["tsp"])
        optimal_solution = read_optimal_tour(paths["opt"])
        optimal_cost = reference_problem._calculate_fitness(optimal_solution)

        all_results: list[dict[str, Any]] = []
        for run_id in range(1, RUNS_COUNT + 1):
            seed = 420 + run_id
            result = run_single_bees(
                run_id=run_id,
                seed=seed,
                problem_path=paths["tsp"],
                max_fes=MAX_FES,
                hyperparameters=DEFAULT_HYPERPARAMETERS,
            )
            all_results.append(result)
            print(
                f"Run {run_id:02d}/{RUNS_COUNT} | "
                f"FEs: {result['fes_used']:>6} | "
                f"Cost: {result['best_cost']:.0f} | "
                f"RPD: {rpd(result['best_cost'], optimal_cost):6.2f}% | "
                f"Time: {result['time_seconds']:.2f}s"
            )

            generate_pipeline_plots(
                run_history=result["history"],
                problem_name=problem_name,
                run_id=run_id,
                optimal_cost=optimal_cost,
                found_cost=result["best_cost"],
                save_dir=PIPELINE_RESULTS_DIR,
            )

        generate_best_run_convergence(
            results=all_results,
            problem_name=problem_name,
            optimal_cost=optimal_cost,
            save_dir=PIPELINE_RESULTS_DIR,
        )
        generate_summary_plots(
            results=all_results,
            problem_name=problem_name,
            optimal_cost=optimal_cost,
            save_dir=PLOTS_DIR / problem_name,
        )
        generate_json_report(
            results=all_results,
            problem_name=problem_name,
            max_fes=MAX_FES,
            optimal_cost=optimal_cost,
            hyperparameters=DEFAULT_HYPERPARAMETERS,
            save_dir=PIPELINE_RESULTS_DIR,
        )

    print(f"\nPipeline finished. Results: {PIPELINE_RESULTS_DIR}")
    print(f"Summary plots: {PLOTS_DIR}")


if __name__ == "__main__":
    run_pipeline()
