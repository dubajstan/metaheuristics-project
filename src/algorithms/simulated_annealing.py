from src.core.Problem import Problem
import math
import numpy as np
from src.core.OptimizationAlgorithm import OptimizationAlgorithm
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import os

from src.core.TSPProblem import TSPProblem
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tsplib95

# --- Type Aliases ---
Solution = np.ndarray


# --- Core Algorithm Functions ---
def get_next_solution(current_solution: Solution, current_energy: float, temperature: float, problem: Problem, rng) -> tuple[Solution, float, bool]:
    """
    Calculates the next solution for SA chain.
    """
    neighbour = problem.get_neighbour(current_solution, rng)
    neighbour_energy = problem.evaluate(neighbour)
    delta_E = neighbour_energy - current_energy
    
    if delta_E < 0:
        return neighbour, neighbour_energy, True
    
    if temperature > 1e-12:
        accept_prob = math.exp(-delta_E / temperature)
        if rng.random() < accept_prob:
            return neighbour, neighbour_energy, True
    return current_solution, current_energy, False


def run_as_chain(chain_id: int, problem: Problem, init_temp: float, next_temp_func: Callable[[int, float], float], max_fes: int) -> dict[str, Any]:
    rng = np.random.default_rng(problem.seed + chain_id)
    current_solution = problem.create_random_solution()
    current_energy = problem.evaluate(current_solution)
    best_solution = np.copy(current_solution)
    best_energy = current_energy
    temperature = init_temp
    iteration = 0
    history = []
    
    # Store: chain_id, iteration, energy, best_energy, temperature, is_accepted, delta_E
    history.append((chain_id, iteration, current_energy, best_energy, temperature, True, 0.0))

    while problem.fe_count < max_fes:
        iteration += 1
        if iteration % 50_000 == 0:
            print(f"Chain {chain_id} : {iteration} FEs")

        # Capture old energy to calculate precise delta for logging
        old_energy = current_energy
        current_solution, current_energy, accepted = get_next_solution(
            current_solution,
            current_energy,
            temperature,
            problem,
            rng
        )
        
        delta_E = current_energy - old_energy

        if current_energy < best_energy:
            best_energy = current_energy
            best_solution = np.copy(current_solution)

        history.append((
            chain_id,
            iteration,
            current_energy,
            best_energy,
            temperature,
            accepted,
            delta_E
        ))

        temperature = next_temp_func(iteration, temperature)
    
    return {
        'chain_id': chain_id,
        'best_solution': best_solution,
        'best_energy': best_energy,
        'history': history
    }


class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(self, problem: Problem, max_fes: int, workers: int = 1):
        super().__init__(problem, max_fes)
        self.workers = workers
        self.history: list[dict[str, Any]] = []
        self.chain_results: list[dict[str, Any]] = []

    def solve(self, init_temp: float, next_temp_func: Callable[[int, float], float]) -> Solution:
        fes_per_worker = self.max_fes
        global_best_solution = None
        global_best_energy = float('inf')

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(
                    run_as_chain,
                    chain_id,
                    deepcopy(self.problem),
                    init_temp,
                    next_temp_func,
                    fes_per_worker
                )
                for chain_id in range(self.workers)
            ]

            for future in as_completed(futures):
                result = future.result()
                self.chain_results.append(result)
                self.history.extend(result['history'])
                if result['best_energy'] < global_best_energy:
                    global_best_energy = result['best_energy']
                    global_best_solution = np.copy(result['best_solution'])
        
        self.history.sort(key=lambda x: (x[0], x[1]))
        return global_best_solution


# --- Configuration Helpers ---
#class GeomTempCooling:
#    def __call__(self, iteration: int, current_temp: float) -> float:
#        return 0.95 * current_temp
    

def estimate_initial_temperature(problem, samples=1000, target_acceptance=0.8, rng=None):
    sol = problem.create_random_solution()
    energy = problem.evaluate(sol)
    deltas = []

    for _ in range(samples):
        neigh = problem.get_neighbour(sol, rng)
        neigh_energy = problem.evaluate(neigh)
        delta = neigh_energy - energy
        if delta > 0:
            deltas.append(delta)
        sol = neigh
        energy = neigh_energy

    avg_delta = np.mean(deltas) if deltas else 1.0
    return -avg_delta / np.log(target_acceptance)


# --- Advanced Visualization Engine ---
# --- Advanced Visualization Engine ---
class SAVisualizer:
    """Handles deep analytical state tracking visualizations named dynamically per instance."""
    def __init__(self, history_data: list, problem_path: str, base_dir: str = "plots"):
        # Dynamically extract file name (e.g., 'a280')
        self.problem_name = os.path.splitext(os.path.basename(problem_path))[0]
        self.output_dir = os.path.join(base_dir, self.problem_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df = pd.DataFrame(
            history_data, 
            columns=['chain_id', 'iteration', 'energy', 'best_energy', 'temperature', 'is_accepted', 'delta_E']
        )
        sns.set_theme(style="whitegrid", palette="muted")

    def plot_convergence(self, optimal_energy: float, found_energy: float):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        # Wykresy linii z jawnym label dla legendy
        sns.lineplot(data=self.df, x='iteration', y='energy', hue='chain_id', alpha=0.3, ax=ax, linewidth=1, legend=False)
        # Tworzymy dummy plot dla legendy linii ciągłej (Instantaneous Energy) i kropkowanej (Best-so-far)
        for chain in self.df['chain_id'].unique():
            chain_df = self.df[self.df['chain_id'] == chain]
            ax.plot(chain_df['iteration'], chain_df['best_energy'], linestyle=":", linewidth=2.5, label=f'Chain {chain} (Best-So-Far)')
            ax.plot([], [], color='gray', alpha=0.5, label=f'Chain {chain} (Current State)' if chain == 0 else "")

        ax.axhline(y=optimal_energy, color='black', linestyle='--', linewidth=1.5, label=f'Global Optimum Reference ({optimal_energy:.0f})')
        ax.axhline(y=found_energy, color='red', linestyle='-', linewidth=1.5, label=f'Best Found Solution ({found_energy:.0f})')
        
        ax.set_title(f'[{self.problem_name}] Energy Convergence Profile', fontsize=12, fontweight='bold')
        ax.set_xlabel('Function Evaluations (Iterations)')
        ax.set_ylabel('Tour Distance (Energy)')
        
        # Umieszczenie uporządkowanej legendy poza wykresem
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Optimization Metrics")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{self.problem_name}_01_convergence.png", dpi=300)
        plt.close()

    def plot_cooling_schedule(self):
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.lineplot(data=self.df, x='iteration', y='temperature', hue='chain_id', linewidth=2.5, ax=ax)
        
        ax.set_title(f'[{self.problem_name}] Temperature Decay Profile', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Temperature')
        ax.legend(title="Parallel Chains", loc="upper right")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{self.problem_name}_02_cooling.png", dpi=300)
        plt.close()

    def plot_acceptance_rate(self):
        fig, ax = plt.subplots(figsize=(9, 5))
        self.df['is_accepted_num'] = self.df['is_accepted'].astype(float)
        self.df['rolling_acc'] = self.df.groupby('chain_id')['is_accepted_num'].transform(
            lambda x: x.rolling(window=2000, min_periods=1).mean()
        )
        
        sns.lineplot(data=self.df, x='iteration', y='rolling_acc', hue='chain_id', ax=ax, alpha=0.8, linewidth=1.5)
        ax.set_title(f'[{self.problem_name}] Rolling Acceptance Probability', fontsize=12, fontweight='bold')
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('Acceptance Index (0.0 - 1.0)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(title="MCMC Chains", loc="upper right")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{self.problem_name}_03_acceptance_rate.png", dpi=300)
        plt.close()

    def plot_energy_distribution(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=self.df, x='energy', hue='chain_id', fill=True, common_norm=False, alpha=0.4, ax=ax)
        
        ax.set_title(f'[{self.problem_name}] Explored State Density Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Energy Space (Tour Distance)')
        ax.set_ylabel('Relative Sampling Density')
        ax.legend(title="Search Trace", loc="upper right", labels=[f"Chain {c}" for c in self.df['chain_id'].unique()])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{self.problem_name}_04_energy_dist.png", dpi=300)
        plt.close()

    def plot_degrading_moves(self):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        # Isolate instances where delta_E was strictly positive but accepted
        degrading_df = self.df[(self.df['delta_E'] > 0) & (self.df['is_accepted'] == True)]
        
        if not degrading_df.empty:
            scatter = sns.scatterplot(
                data=degrading_df, 
                x='iteration', 
                y='delta_E', 
                hue='chain_id', 
                size='temperature',
                sizes=(10, 100), 
                alpha=0.6, 
                ax=ax
            )
            ax.set_title(f'[{self.problem_name}] Accepted Uphill (Worse) Moves Over Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration Index')
            ax.set_ylabel('Positive Delta Energy (+delta_E)')
            
            # Przenosimy rozbudowaną legendę (chain i rozmiar temperatury) na bok
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Exploration Dynamics")
        else:
            ax.text(0.5, 0.5, 'No worse moves accepted during exploration.', horizontalalignment='center', verticalalignment='center')
            
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{self.problem_name}_05_uphill_escapes.png", dpi=300)
        plt.close()


import optuna
import numpy as np
import tsplib95
from copy import deepcopy

# Assuming your class structures are in the same script or imported
# from your_module import TSPProblem, SimulatedAnnealing, estimate_initial_temperature

class DynamicGeomCooling:
    def __init__(self, alpha: float):
        self.alpha = alpha
    def __call__(self, iteration: int, current_temp: float) -> float:
        return self.alpha * current_temp

def objective(trial):
    """Optuna objective function to minimize the tour distance on att48."""
    # 1. Setup the problem instance
    # Use a small FEs budget for tuning so it runs incredibly fast!
    tuning_max_fes = 50_000 
    problem = TSPProblem('data/tsplib95/tasks/att48.tsp')
    
    # 2. Define the hyperparameter search space
    # Instead of guessing T_0, we tune the target acceptance rate for our estimation function
    target_acceptance = trial.suggest_float('target_acceptance', 0.40, 0.95)
    
    # Tune the alpha parameter for geometric cooling
    # For 50k FEs, we explore a range that cools down efficiently
    alpha = trial.suggest_float('alpha', 0.999, 0.99999)
    
    # 3. Evaluate over multiple random seeds to avoid lucky/unlucky runs
    seeds = [42, 123, 999]
    trial_energies = []
    
    for seed in seeds:
        # Isolate problem environment per seed
        local_problem = deepcopy(problem)
        rng = np.random.default_rng(seed=seed)
        
        # Calculate initial temperature using the trial's target acceptance
        init_temp = estimate_initial_temperature(
            local_problem, 
            samples=500, 
            target_acceptance=target_acceptance, 
            rng=rng
        )
        
        temperature_func = DynamicGeomCooling(alpha)
        
        # Run SA using 2 workers for faster evaluation
        sa = SimulatedAnnealing(local_problem, max_fes=tuning_max_fes, workers=2)
        solution = sa.solve(init_temp, temperature_func)
        
        found_energy = local_problem.evaluate(solution)
        trial_energies.append(found_energy)
        
    # Return the mean energy across our test seeds to guide Optuna reliably
    return float(np.mean(trial_energies))

if __name__ == '__main__':
    # Disable heavy logging prints from your SA chains during optimization if necessary
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Create an Optuna study aiming to minimize our objective metric (Tour Distance)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42) # Bayesian Optimization via TPE
    )
    
    print("Starting Hyperparameter Tuning Optimization via Optuna...")
    # 30 trials is typically a sweet spot for 2 continuous parameters
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    
    print("\n" + "="*40)
    print("TUNING COMPLETE")
    print("="*40)
    print(f"Best Trial Mean Energy: {study.best_value:.2f}")
    print("Best Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f"  -> {key}: {value}")
        
    # How to leverage the results back in your main execution script:
    best_acceptance = study.best_params['target_acceptance']
    best_alpha = study.best_params['alpha']


# --- Execution Runtime ---
#if __name__ == '__main__':
#    # File configuration configuration path
#    problem_name = 'a280'
#    for problem_name in ('eil101',):
#        problem_file_path = f'data/tsplib95/tasks/{problem_name}.tsp'
#
#        problem = TSPProblem(problem_file_path)
#        max_fes = 7_000_000  
#        workers = 1          
#
#        rng = np.random.default_rng(seed=420)
#        init_temp = estimate_initial_temperature(problem, target_acceptance=0.8, rng=rng)
#        print(f"Calculated Initial Temperature ({problem_file_path}): {init_temp:.4f}")
#
#        temperature_func = GeomTempCooling()
#        sa = SimulatedAnnealing(problem, max_fes, workers)
#
#        print("Executing Chain Engine...")
#        solution = sa.solve(init_temp, temperature_func)
#
#        # Standard Validation Loading
#        opt_data = tsplib95.load(f'data/tsplib95/solutions/{problem_name}.opt.tour')
#        optimal_solution = np.array(opt_data.tours[0]) - 1
#        optimal_energy = problem.evaluate(optimal_solution)
#        found_energy = problem.evaluate(solution)
#
#        print(f"Done. Optimal Ref: {optimal_energy:.0f} | Best Discovered: {found_energy:.0f}")
#
#        # Fire Visualization Sequence
#        print(f"Generating data metrics tracking plots under root paths inside './plots/ ...")
#        viz = SAVisualizer(sa.history, problem_path=problem_file_path, base_dir="plots")
#        viz.plot_convergence(optimal_energy, found_energy)
#        viz.plot_cooling_schedule()
#        viz.plot_acceptance_rate()
#        viz.plot_energy_distribution()
#        viz.plot_degrading_moves()
#        print("Diagnostic processing output complete.")


