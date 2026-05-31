import math
import numpy as np
import time
import os
import json
import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tsplib95
from copy import deepcopy
from typing import Callable, Any

from src.core.Problem import Problem
from src.core.OptimizationAlgorithm import OptimizationAlgorithm
from src.core.TSPProblem import TSPProblem

Solution = np.ndarray


def get_next_solution(current_solution: Solution, current_energy: float, temperature: float, problem: Problem, rng) -> tuple[Solution, float, bool]:
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


def run_single_sa(seed: int, problem: Problem, init_temp: float, alpha: float, max_fes: int, stagnation_limit: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    current_solution = problem.create_random_solution()
    current_energy = problem.evaluate(current_solution)
    
    best_solution = np.copy(current_solution)
    best_energy = current_energy
    
    temperature = init_temp
    iteration = 0
    stagnation_counter = 0
    history = []
    
    while problem.fe_count < max_fes:
        iteration += 1
        
        current_solution, current_energy, accepted = get_next_solution(
            current_solution, current_energy, temperature, problem, rng
        )
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_solution = np.copy(current_solution)
            stagnation_counter = 0
        else:
            stagnation_counter += 1
            
        history.append({
            'iteration': iteration,
            'energy': current_energy,
            'best_energy': best_energy,
            'temperature': temperature,
            'is_accepted': accepted
        })
        
        if stagnation_counter >= stagnation_limit:
            break
            
        temperature *= alpha
        
    return {
        'best_solution': best_solution,
        'best_energy': best_energy,
        'fes_used': problem.fe_count,
        'history': history
    }


def estimate_initial_temperature(problem: Problem, samples: int = 1000, target_acceptance: float = 0.8, rng=None):
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


def tune_hyperparameters(problem_path: str, max_fes_tuning: int = 100_000, n_trials: int = 20) -> tuple[float, float]:
    print(f"\n--- Rozpoczynam strojenie Optuna dla: {os.path.basename(problem_path)} ---")
    problem = TSPProblem(problem_path)
    
    def objective(trial):
        # Zawężone, bezpieczne przedziały dla początkowej akceptacji gorszych ruchów
        target_acc = trial.suggest_float('target_acceptance', 0.7, 0.95)
        
        # Stroimy wykładnik końcowej temperatury (np. od 10^-5 do 10^-2 temperatury początkowej)
        # Zapobiega to zgadywaniu surowego alpha, które "rozjeżdża się" przy zmianie budżetu FEs
        final_temp_log_ratio = trial.suggest_float('final_temp_log_ratio', -5.0, -2.0)
        final_temp_ratio = 10 ** final_temp_log_ratio
        
        energies = []
        for s in range(3):  # 3 powtórzenia wystarczą do stabilnej oceny konfiguracji
            local_prob = deepcopy(problem)
            rng = np.random.default_rng(s)
            t0 = estimate_initial_temperature(local_prob, 1000, target_acc, rng)
            
            # Obliczenie idealnego alfa dla dedykowanego budżetu strojenia
            alpha = final_temp_ratio ** (1.0 / max_fes_tuning)
            
            res = run_single_sa(s, local_prob, t0, alpha, max_fes_tuning, stagnation_limit=max_fes_tuning)
            energies.append(res['best_energy'])
            
        return float(np.mean(energies))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_target = study.best_params['target_acceptance']
    best_ratio = 10 ** study.best_params['final_temp_log_ratio']
    
    return best_target, best_ratio


def generate_json_report(results: list, problem_name: str, max_fes: int, optimal_cost: float, save_dir: str):
    energies = [r['best_energy'] for r in results]
    times = [r['time_seconds'] for r in results]
    fes_counts = [r['fes_used'] for r in results]
    rpds = [((e - optimal_cost) / optimal_cost) * 100 for e in energies]
    
    report = {
        "algorithm": "Simulated Annealing",
        "problem_name": problem_name,
        "runs_count": len(results),
        "max_fes": max_fes,
        "optimal_cost": optimal_cost,
        "summary_statistics": {
            "best_result": float(np.min(energies)),
            "best_result_rpd_percentage": float(np.min(rpds)),
            "average_result": float(np.mean(energies)),
            "average_result_rpd_percentage": float(np.mean(rpds)),
            "average_time_seconds": float(np.mean(times)),
            "average_fes_count": float(np.mean(fes_counts)),
            "std_dev_rpd_percentage": float(np.std(rpds))
        },
        "raw_data_from_all_runs": [
            {
                "run_id": r['run_id'],
                "best_cost": float(r['best_energy']),
                "rpd_percentage": float(((r['best_energy'] - optimal_cost) / optimal_cost) * 100),
                "time_seconds": float(r['time_seconds']),
                "fes_used": int(r['fes_used']),
                "stagnation_triggered": r['fes_used'] < max_fes
            }
            for r in results
        ]
    }
    
    filepath = os.path.join(save_dir, f"{problem_name}_sa_results.json")
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Saved report JSON: {filepath}")


def generate_pipeline_plots(run_history: list, problem_name: str, run_id: int, optimal_cost: float, found_cost: float, save_dir: str):
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    df = pd.DataFrame(run_history)
    sns.set_theme(style="whitegrid", palette="muted")
    
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    
    # ----------------------------------------------------
    # PANEL 1: CONVERGENCE AND EXPLORATION PROFILE
    # ----------------------------------------------------
    axes[0].scatter(
        df['iteration'], df['energy'], 
        color='darkorange', alpha=0.3, s=2.0, 
        label='Current Solution Energy'
    )
    
    axes[0].plot(
        df['iteration'], df['best_energy'], 
        color='mediumblue', linewidth=2.5, 
        label='Best Solution Energy'
    )
    
    axes[0].axhline(y=optimal_cost, color='black', linestyle='--', linewidth=1.5, label=f'Known Optimum ({optimal_cost:.0f})')
    axes[0].axhline(y=found_cost, color='crimson', linestyle='-', linewidth=1.5, label=f'Found Optimum ({found_cost:.0f})')
    
    axes[0].set_title(f'[{problem_name.upper()}] Convergence and Exploration Profile - Run {run_id:02d}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Objective Function Value (Energy)', fontsize=10)
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    # ----------------------------------------------------
    # PANEL 2: ROLLING ACCEPTANCE RATE
    # ----------------------------------------------------
    df['is_accepted_numeric'] = df['is_accepted'].astype(float)
    df['rolling_acceptance'] = df['is_accepted_numeric'].rolling(window=1000, min_periods=1).mean()
    
    axes[1].plot(
        df['iteration'], df['rolling_acceptance'], 
        color='forestgreen', linewidth=1.8, 
        label='Rolling Acceptance Rate (Window = 1000)'
    )
    
    axes[1].set_title('Move Acceptance Rate Over Time', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Number of Evaluations / Iterations (FEs)', fontsize=10)
    axes[1].set_ylabel('Acceptance Level (0.0 - 1.0)', fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    
    filename = f"{problem_name}_run_{run_id:02d}_diagnostics.png"
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == '__main__':
    PROBLEMS = {
        'att48':  {'tsp': 'data/tsplib95/tasks/att48.tsp',  'opt': 'data/tsplib95/solutions/att48.opt.tour'},
        'eil101': {'tsp': 'data/tsplib95/tasks/eil101.tsp', 'opt': 'data/tsplib95/solutions/eil101.opt.tour'},
        'a280':   {'tsp': 'data/tsplib95/tasks/a280.tsp',   'opt': 'data/tsplib95/solutions/a280.opt.tour'}
    }
    
    RUNS_COUNT = 10
    MAX_FES = 300_000          # Zmniejszono z 5M – optymalny budżet dla płynnego wykonania w Pythonie
    STAGNATION_LIMIT = MAX_FES  # Wyłączono przedwczesny stop – pozwalamy schłodzić się algorytmowi do końca
    TUNING_BUDGET = 100_000    # Szybkie, ale reprezentatywne strojenie dla każdego problemu
    
    OUTPUT_DIR = "pipeline_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for prob_name, paths in PROBLEMS.items():
        print(f"\n=============================================")
        print(f" PIPELINE START: {prob_name.upper()}")
        print(f"=============================================")
        
        opt_data = tsplib95.load(paths['opt'])
        opt_nodes = np.array(opt_data.tours[0]) - 1
        dummy_problem = TSPProblem(paths['tsp'])
        optimal_cost = dummy_problem.evaluate(opt_nodes)
        
        # Strojenie optymalnego stosunku temperatur
        best_target, best_ratio = tune_hyperparameters(paths['tsp'], max_fes_tuning=TUNING_BUDGET, n_trials=20)
        
        # Przeliczenie współczynnika alfa dokładnie pod duży budżet (MAX_FES) eksperymentu
        best_alpha = best_ratio ** (1.0 / MAX_FES)
        print(f"Hyperparameters: Target Acceptance={best_target:.4f}, Alpha={best_alpha:.6f} (scaled to full budget)")
        
        all_runs_results = []
        
        for run_id in range(1, RUNS_COUNT + 1):
            seed = 420 + run_id
            problem_instance = TSPProblem(paths['tsp'])
            rng = np.random.default_rng(seed)
            
            t0 = estimate_initial_temperature(problem_instance, 1000, best_target, rng)
            
            start_time = time.perf_counter()
            run_result = run_single_sa(
                seed=seed, 
                problem=problem_instance, 
                init_temp=t0, 
                alpha=best_alpha, 
                max_fes=MAX_FES, 
                stagnation_limit=STAGNATION_LIMIT
            )
            elapsed_time = time.perf_counter() - start_time
            
            all_runs_results.append({
                'run_id': run_id,
                'best_energy': run_result['best_energy'],
                'fes_used': run_result['fes_used'],
                'time_seconds': elapsed_time
            })
            
            print(f"Run {run_id:02d}/{RUNS_COUNT} | FEs: {run_result['fes_used']:>7} | Cost: {run_result['best_energy']:.0f} | Time: {elapsed_time:.2f}s")
            
            generate_pipeline_plots(
                run_history=run_result['history'],
                problem_name=prob_name,
                run_id=run_id,
                optimal_cost=optimal_cost,
                found_cost=run_result['best_energy'],
                save_dir=OUTPUT_DIR
            )
                
        generate_json_report(all_runs_results, prob_name, MAX_FES, optimal_cost, OUTPUT_DIR)
        
    print(f"\n[✔] Pipeline zakończony. Pliki JSON znajdują się w: '{OUTPUT_DIR}/', a wykresy w: '{OUTPUT_DIR}/plots/'")