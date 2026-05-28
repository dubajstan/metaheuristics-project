from src.core.Problem import Problem
import math
import numpy as np
from src.core.OptimizationAlgorithm import OptimizationAlgorithm
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy


type Solution = np.ndarray


def get_next_solution(current_solution: Solution, current_energy: float, temperature: float, problem: Problem, rng) -> tuple[Solution, float, bool]:
    """
    Calculated the next solution for SA chain.
    Returns:
    - next_solution (np.ndarray): The state for the next iteration (either the neighbour or original) 
    - next_energy (float): The energy of the next state
    - accepted (bool): True if the proposed neighbour was accepted, False otherwise
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
    history.append((
        chain_id,
        iteration,
        current_energy,
        temperature,
        True
    ))

    while problem.fe_count < max_fes:
        iteration += 1

        current_solution, current_energy, accepted = get_next_solution(
            current_solution,
            current_energy,
            temperature,
            problem,
            rng
        )

        if current_energy < best_energy:
            best_energy = current_energy
            best_solution = np.copy(current_solution)

        history.append((
        chain_id,
        iteration,
        current_energy,
        temperature,
        accepted
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
        fes_per_worker = self.max_fes // self.workers
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




from src.core.TSPProblem import TSPProblem
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tsplib95


class GeomTempCooling:
    def __call__(self, iteration: int, current_temp: float) -> float:
        return 0.999999 * current_temp
    

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

    avg_delta = np.mean(deltas)

    return -avg_delta / np.log(target_acceptance)


if __name__ == '__main__':
    problem = TSPProblem('data/tsplib95/tasks/a280.tsp')
    max_fes = 5_000_000
    workers = 1
    rng = np.random.default_rng(seed=42)
    init_temp = estimate_initial_temperature(problem, rng=rng)
    print(init_temp)
    temperature_func = GeomTempCooling()

    sa = SimulatedAnnealing(problem, max_fes, workers)
    #temp_func = ExponentialCoolingWithLocalSearch(init_temp, 0.0, max_fes)
    solution = sa.solve(init_temp, temperature_func)
    
    opt_data = tsplib95.load('data/tsplib95/solutions/a280.opt.tour')
    opt_nodes = opt_data.tours[0]
    optimal_solution = np.array(opt_nodes) - 1
    optimal_energy = problem.evaluate(optimal_solution)

    df = pd.DataFrame(sa.history, columns=['chain_id', 'iteration', 'energy', 'temperature', 'is_accepted'])
    sns.set_theme(style="whitegrid", palette="husl")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # wykres energi od iteracji
    sns.lineplot(
        data=df, 
        x='iteration', 
        y='energy', 
        hue='chain_id', 
        ax=axes[0, 0], 
        alpha=0.8,
        linewidth=1.5
    )
    axes[0, 0].axhline(
        y=optimal_energy, 
        color='black', 
        linestyle='--', 
        linewidth=2, 
        label=f'Global Optimum ({optimal_energy:.0f})'
    )
    axes[0, 0].set_title('Simulated Annealing: Convergence of Energy', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Energy (Tour Distance)', fontsize=12)
    axes[0, 0].legend(title='Legend', bbox_to_anchor=(1.01, 1), loc='upper left')
    # wykres temperatury od iteracji
    sns.lineplot(
        data=df[df['chain_id'] == 0], 
        x='iteration', 
        y='temperature', 
        ax=axes[0, 1], 
        color='crimson', 
        linewidth=2.5
    )
    axes[0, 1].set_title('Cooling Schedule', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Function Evaluations (Iterations)', fontsize=12)
    axes[0, 1].set_ylabel('Temperature', fontsize=12)
    # wykres średniej akceptacji w przedzialach 1000 iteracji
    df['is_accepted'] = df['is_accepted'].astype(float)
    df['rolling_acceptance'] = df.groupby('chain_id')['is_accepted'].transform(
        lambda x: x.rolling(window = 1000, min_periods=1).mean()
    )
    sns.lineplot(
        data=df,
        x='iteration',
        y='rolling_acceptance',
        hue='chain_id',
        ax=axes[1,0],
        alpha=0.8,
        linewidth=1.5,
        legend=False
    )
    axes[1, 0].set_title('Moving average of 1000 iterations', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Function evaluations (iterations)', fontsize=11)
    axes[1, 0].set_ylabel('Acceptance rate (0.0 = 0%, 1.0 = 100%)')
    axes[1, 0].set_ylim(-0.05, 1.05)



    plt.tight_layout()
    plt.savefig('sa_history_seaborn.png', dpi=300, bbox_inches='tight')
    plt.show()

