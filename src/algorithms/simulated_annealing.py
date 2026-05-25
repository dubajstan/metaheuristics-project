from src.core.Problem import Problem
import math
import numpy as np
from src.core.OptimizationAlgorithm import OptimizationAlgorithm
from typing import Callable, Any
import concurrent.futures


type Solution = np.ndarray


def get_next_solution(current_solution: Solution, current_energy: float, temperature: float, problem: Problem) -> tuple[Solution, float, bool]:
    """
    Calculated the next solution for SA chain.
    Returns:
    - next_solution (np.ndarray): The state for the next iteration (either the neighbour or original) 
    - next_energy (float): The energy of the next state
    - accepted (bool): True if the proposed neighbour was accepted, False otherwise
    """

    if temperature < 1e-12: # zeby nie dzielilo przez 0
        raise ZeroDivisionError('Upsi Daisy')
    
    neighbour = problem.get_neighbour(current_solution)
    neighbour_energy = problem.evaluate(neighbour)
    delta_E = neighbour_energy - current_energy
    if delta_E < 0:
        return neighbour, neighbour_energy, True
    else:
        accept_prob = math.exp(-delta_E / temperature)
        if problem.rng.random() < accept_prob:
            return neighbour, neighbour_energy, True
    return current_solution, current_energy, False


def run_as_chain(chain_id: int, problem: Problem, init_temp: float, next_temp_func: Callable[[int, float], float], max_fes: int) -> dict[str, Any]:
    problem.rng = np.random.default_rng(problem.seed + chain_id)
    current_solution = problem.create_random_solution()
    current_energy = problem.evaluate(current_solution)
    best_solution = np.copy(current_solution)
    best_energy = current_energy
    temperature = init_temp
    iteration = 0
    history = []
    history.append({
        'chain_id': chain_id,
        'iteration': iteration,
        'solution': np.copy(current_solution),
        'energy': current_energy,
        'temperature': temperature,
        'accepted': True
    })

    while problem.fe_count < max_fes:
        iteration += 1
        if temperature < 1e-12:
            break

        current_solution, current_energy, accepted = get_next_solution(
            current_solution,
            current_energy,
            temperature,
            problem
        )

        if current_energy < best_energy:
            best_energy = current_energy
            best_solution = np.copy(current_solution)

        history.append({
        'chain_id': chain_id,
        'iteration': iteration,
        'solution': np.copy(current_solution),
        'energy': current_energy,
        'temperature': temperature,
        'accepted': accepted
        })

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

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(
                    run_as_chain,
                    chain_id,
                    self.problem,
                    init_temp,
                    next_temp_func,
                    fes_per_worker
                )
                for chain_id in range(self.workers)
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                self.chain_results.append(result)
                self.history.extend(result['history'])
                if result['best_energy'] < global_best_energy:
                    global_best_energy = result['best_energy']
                    global_best_solution = np.copy(result['best_solution'])
        
        self.history.sort(key=lambda x: (x['chain_id'], x['iteration']))

        return global_best_solution




from src.core.TSPProblem import TSPProblem
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tsplib95


class ExponentialCoolingWithLocalSearch:
    def __init__(self, init_temp: float, final_temp: float, max_fes: int, cooling_fraction: float = 0.85):
        """
        cooling_fraction: np. 0.85 oznacza, że przez 85% czasu stygniemy, a przez ostatnie 15% robimy Local Search.
        """
        self.init_temp = init_temp
        # Upewniamy się, że final_temp jest większa od 1e-12, żeby nie uruchomić "break"
        self.final_temp = max(final_temp, 1e-10) 
        
        # Obliczamy ile iteracji faktycznie przeznaczamy na stygnięcie
        self.cooling_steps = int(max_fes * cooling_fraction)
        
        # Wyliczamy alpha TYLKO dla kroków chłodzenia
        self.alpha = (self.final_temp / self.init_temp) ** (1.0 / self.cooling_steps)
        print(f"[Cooling Setup] Alpha: {self.alpha:.7f} | Chłodzenie: {self.cooling_steps} kroków | Local Search: {max_fes - self.cooling_steps} kroków")

    def __call__(self, iteration: int, current_temp: float) -> float:
        # Faza 1: Normalne stygnięcie
        if iteration < self.cooling_steps:
            return current_temp * self.alpha
            
        # Faza 2: Czysty Local Search
        # Zwracamy stałą, mikroskopijną temperaturę.
        # Prawdopodobieństwo akceptacji gorszego rozwiązania będzie matematycznie bliskie zeru.
        else:
            return 1e-11




if __name__ == '__main__':
    problem = TSPProblem('data/tsplib95/tasks/a280.tsp')
    sa = SimulatedAnnealing(problem, 1000000, 8)
    init_temp = 10000000.0
    max_fes = 1000000 // 8
    temp_func = ExponentialCoolingWithLocalSearch(init_temp, 0.0, max_fes)
    solution = sa.solve(1000000.0, temp_func)
    
    opt_data = tsplib95.load('data/tsplib95/solutions/a280.opt.tour')
    opt_nodes = opt_data.tours[0]
    optimal_solution = np.array(opt_nodes) - 1
    optimal_energy = problem.evaluate(optimal_solution)

    df = pd.DataFrame(sa.history)
    sns.set_theme(style="whitegrid", palette="husl")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    sns.lineplot(
        data=df, 
        x='iteration', 
        y='energy', 
        hue='chain_id', 
        ax=axes[0], 
        alpha=0.8,
        linewidth=1.5
    )
    axes[0].axhline(
        y=optimal_energy, 
        color='black', 
        linestyle='--', 
        linewidth=2, 
        label=f'Global Optimum ({optimal_energy:.0f})'
    )
    axes[0].set_title('Simulated Annealing: Convergence of Energy', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Energy (Tour Distance)', fontsize=12)
    axes[0].legend(title='Legend', bbox_to_anchor=(1.01, 1), loc='upper left')
    sns.lineplot(
        data=df[df['chain_id'] == 0], 
        x='iteration', 
        y='temperature', 
        ax=axes[1], 
        color='crimson', 
        linewidth=2.5
    )
    axes[1].set_title('Cooling Schedule', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Function Evaluations (Iterations)', fontsize=12)
    axes[1].set_ylabel('Temperature', fontsize=12)
    plt.tight_layout()
    plt.savefig('sa_history_seaborn.png', dpi=300, bbox_inches='tight')
    plt.show()