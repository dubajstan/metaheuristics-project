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

# ==========================================
# 1. CORE ALGORITHM MODULES
# ==========================================

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
        
        old_energy = current_energy
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

# ==========================================
# 2. TUNING & REPORT LOGIC
# ==========================================

def tune_hyperparameters(problem_path: str, max_fes_tuning: int = 50_000, n_trials: int = 20) -> tuple[float, float]:
    print(f"\n--- Rozpoczynam strojenie Optuna dla: {os.path.basename(problem_path)} ---")
    problem = TSPProblem(problem_path)
    
    def objective(trial):
        target_acc = trial.suggest_float('target_acceptance', 0.5, 0.95)
        alpha = trial.suggest_float('alpha', 0.99, 0.99999)
        
        seeds = [42, 123]
        energies = []
        
        for s in seeds:
            local_prob = deepcopy(problem)
            rng = np.random.default_rng(s)
            t0 = estimate_initial_temperature(local_prob, 500, target_acc, rng)
            res = run_single_sa(s, local_prob, t0, alpha, max_fes_tuning, stagnation_limit=10_000)
            energies.append(res['best_energy'])
            
        return float(np.mean(energies))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params['target_acceptance'], study.best_params['alpha']


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
    print(f"[✔] Zapisano raport JSON: {filepath}")


def generate_pipeline_plots(run_history: list, problem_name: str, run_id: int, optimal_cost: float, found_cost: float, save_dir: str):
    """Generuje szczegółowy wykres konwergencji dla konkretnego uruchomienia."""
    df = pd.DataFrame(run_history)
    sns.set_theme(style="whitegrid", palette="muted")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Bieżąca energia
    sns.lineplot(data=df, x='iteration', y='energy', color='gray', alpha=0.3, linewidth=1, label='Bieżąca energia (Current Energy)')
    # 2. Najlepsza znaleziona energia w czasie
    sns.lineplot(data=df, x='iteration', y='best_energy', color='blue', linewidth=2.5, label='Najlepsza dotychczasowa (Best Found Energy)')
    # 3. Teoretyczne globalne optimum
    ax.axhline(y=optimal_cost, color='black', linestyle='--', linewidth=1.5, label=f'Globalne optimum referencyjne (Global Optimum: {optimal_cost:.0f})')
    # 4. Ostatecznie znalezione optimum w tym uruchomieniu
    ax.axhline(y=found_cost, color='red', linestyle='-', linewidth=1.5, label=f'Ostateczny wynik tego uruchomienia (Found Global Optimum: {found_cost:.0f})')
    
    ax.set_title(f'[{problem_name.upper()}] Wykres zbieżności - Test {run_id:02d}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Liczba ewaluacji (Iterations / FEs)', fontsize=11)
    ax.set_ylabel('Wartość funkcji celu (Energy / Tour Distance)', fontsize=11)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    filename = f"{problem_name}_run_{run_id:02d}_convergence.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
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
    MAX_FES = 1_000_000  
    STAGNATION_LIMIT = 50_000 
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
        
        # 1. Faza Strojenia (Optuna)
        tuning_budget = 30_000 if prob_name == 'att48' else 100_000
        best_target, best_alpha = tune_hyperparameters(paths['tsp'], max_fes_tuning=tuning_budget, n_trials=15)
        print(f"Użyte parametry: Target Acceptance={best_target:.4f}, Alpha={best_alpha:.5f}")
        
        # 2. Faza Testowania (10 niezależnych uruchomień)
        print(f"\n--- Uruchamianie {RUNS_COUNT} testów z generowaniem wykresów ---")
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
            
            # Zbieranie metryk do pliku JSON
            all_runs_results.append({
                'run_id': run_id,
                'best_energy': run_result['best_energy'],
                'fes_used': run_result['fes_used'],
                'time_seconds': elapsed_time
            })
            
            print(f"Run {run_id:02d}/{RUNS_COUNT} | FEs: {run_result['fes_used']:>7} | Cost: {run_result['best_energy']:.0f} | Time: {elapsed_time:.2f}s")
            
            # GENEROWANIE WYKRESU DLA KAŻDEGO ODPALENIA Z OSOBNA
            generate_pipeline_plots(
                run_history=run_result['history'],
                problem_name=prob_name,
                run_id=run_id,
                optimal_cost=optimal_cost,
                found_cost=run_result['best_energy'],
                save_dir=OUTPUT_DIR
            )
                
        # 3. Zapis zbiorczego raportu JSON po zakończeniu serii testów dla danego problemu
        generate_json_report(all_runs_results, prob_name, MAX_FES, optimal_cost, OUTPUT_DIR)
        
    print(f"\n[✔] Pipeline zakończony. Wykresy (10 na problem) oraz pliki JSON znajdują się w: '{OUTPUT_DIR}/'")