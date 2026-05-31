import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class BeesResultAnalyzer:
    def __init__(
        self,
        optimizer: Any,
        target_path: Path | str,
        exec_time: float,
        series_name: str = "series",
        optimal_cost: float | None = None,
    ) -> None:
        if not optimizer.history:
            raise ValueError("Optimizer history data cannot be empty.")

        self.optimizer = optimizer
        self.target_path = Path(target_path) / series_name
        self.target_path.mkdir(parents=True, exist_ok=True)
        self.exec_time = exec_time
        self.series_name = series_name
        self.optimal_cost = optimal_cost

        self.history = optimizer.history
        self.cycles = [record["cycle"] for record in self.history]
        self.fes = [record["fes"] for record in self.history]
        self.best_costs = [record["best_cost"] for record in self.history]

        self.full_population_history = [
            record
            for record in self.history
            if record.get("is_full_population", True)
        ]
        self.full_population_cycles = [
            record["cycle"] for record in self.full_population_history
        ]
        self.full_population_best_costs = [
            record["best_cost"] for record in self.full_population_history
        ]
        self.avg_costs = [
            record["avg_population_cost"] for record in self.full_population_history
        ]
        self.std_costs = [
            record["std_population_cost"] for record in self.full_population_history
        ]

        last_record = self.history[-1]
        self.best_cost = last_record["best_cost"]
        self.best_solution = last_record["best_solution"]

        problem_path = getattr(optimizer.problem, "file_path", None)
        self.problem_name = Path(problem_path).name if problem_path else "unknown"

    def plot_best_cost(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.cycles,
            self.best_costs,
            label="Best Cost",
            color="green",
            linewidth=2,
            marker="o",
        )

        if self.optimal_cost is not None:
            plt.axhline(
                y=self.optimal_cost,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Optimal Cost ({self.optimal_cost:.0f})",
            )

        plt.title(f"Bees Algorithm: Best Cost - {self.series_name}")
        plt.xlabel("Cycle Number")
        plt.ylabel("Cost")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.target_path / f"{self.series_name}_best_cost.png", dpi=300)
        plt.close()

    def plot_average_cost(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.full_population_cycles,
            self.avg_costs,
            label="Average Cost",
            color="royalblue",
            linewidth=2,
            marker="o",
        )
        plt.plot(
            self.full_population_cycles,
            self.full_population_best_costs,
            label="Current Best Cost",
            color="green",
            linestyle="--",
            linewidth=1.5,
            marker="o",
        )
        plt.title(f"Average Population Cost - {self.series_name}")
        plt.xlabel("Cycle Number")
        plt.ylabel("Cost")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.target_path / f"{self.series_name}_average_cost.png", dpi=300)
        plt.close()

    def plot_cost_std_dev(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.full_population_cycles,
            self.std_costs,
            label="Standard Deviation",
            color="darkorange",
            linewidth=2,
            marker="o",
        )
        plt.title(f"Cost Standard Deviation - {self.series_name}")
        plt.xlabel("Cycle Number")
        plt.ylabel("Standard Deviation")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.target_path / f"{self.series_name}_std_dev.png", dpi=300)
        plt.close()

    def plot_convergence_by_fes(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.fes,
            self.best_costs,
            label="Best Cost",
            color="purple",
            linewidth=2,
        )

        if self.optimal_cost is not None:
            plt.axhline(
                y=self.optimal_cost,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Optimal Cost ({self.optimal_cost:.0f})",
            )

        plt.title(f"Bees Algorithm: Convergence - {self.series_name}")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Cost")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.target_path / f"{self.series_name}_convergence_fes.png", dpi=300)
        plt.close()

    def save_results_json(self) -> None:
        solution_data = self.best_solution
        if hasattr(solution_data, "tolist"):
            solution_data = solution_data.tolist()
        elif isinstance(solution_data, list):
            solution_data = [int(node) for node in solution_data]

        params = self.optimizer.hyperparameters
        results = {
            "problem_name": self.problem_name,
            "series_name": self.series_name,
            "hyperparameters": {
                "n_bees": params.n_bees,
                "n_elite": params.n_elite,
                "n_best": params.n_best,
                "elite_neigh": params.elite_neigh,
                "best_neigh": params.best_neigh,
                "neighbourhood_depth": params.neighbourhood_depth,
                "neighbourhood_type": params.neighbourhood_type,
                "greedy_initial_sites": params.greedy_initial_sites,
                "greedy_start_candidates": params.greedy_start_candidates,
                "greedy_candidate_list_size": params.greedy_candidate_list_size,
            },
            "execution_time_seconds": self.exec_time,
            "total_fes": self.optimizer.fes_total_count,
            "cycles": self.optimizer.cycles,
            "best_cost": self.best_cost,
            "optimal_cost": self.optimal_cost,
            "gap_to_optimal": None
            if self.optimal_cost is None
            else self.best_cost - self.optimal_cost,
            "gap_to_optimal_percent": None
            if self.optimal_cost is None
            else ((self.best_cost - self.optimal_cost) / self.optimal_cost) * 100,
            "best_solution": solution_data,
        }

        with open(self.target_path / f"{self.series_name}_result.json", "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

    def plot_all_and_save(self) -> None:
        self.plot_best_cost()
        self.plot_average_cost()
        self.plot_cost_std_dev()
        self.plot_convergence_by_fes()
        self.save_results_json()
