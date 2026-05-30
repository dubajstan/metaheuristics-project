import numpy as np
import json
import time
import tsplib95
from pathlib import Path
from algorithms.ASTuner import ASTuner
from algorithms.ASOptimizer import ASOptimizer
from core.TSPProblem import TSPProblem

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
HYPERPARAMETERS_DIR = PROJECT_DIR / 'hyperparameters' / 'AS'
TASKS_DIR = PROJECT_DIR / 'data' / 'tsplib95'/ 'tasks'
OPTIMAL_SOLUTIONS_DIR = PROJECT_DIR / 'data' / 'tsplib95' / 'solutions'
RESULTS_DIR = PROJECT_DIR / 'results' / 'AS'

def run_experiments_for_report(problem_name: str, max_fes: int, start_seed : int = 42, runs: int = 10) -> None:
    '''
    Runs the Ant System algorithm a specified number of times to collect statistics 
    required for the final report. Calculates Relative Percentage Deviation
    and saves all data to a JSON file.

    Parameters
    -----
    problem_name : str - Name of the problem file without extension.
    max_fes : int - Maximum number of cost function evaluations for the run.
    runs : int - Number of algorithm executions. Default is 10.
    start_seed : int - Fixed seed. Default is 42.
    '''
    best_costs = []
    execution_times = []
    fes_counts = []
    
    # Load hyperparameters only once before the loop
    
    hyperparameters = ASTuner.load_params(
        HYPERPARAMETERS_DIR / f'{problem_name}_hyperparameters.json'
    )
  

    # Load the optimal solution to calculate RPD
    optimal_data = tsplib95.load(OPTIMAL_SOLUTIONS_DIR / f'{problem_name}.opt.tour')
    optimal_solution = np.array(optimal_data.tours[0]) - 1
    
    # Create a temporary problem instance just to evaluate the optimal cost
    tmp_problem = TSPProblem(
        file_path = TASKS_DIR / f'{problem_name}.tsp', 
        seed = 42
    )
    optimal_cost = tmp_problem.evaluate(optimal_solution)

    for i in range(runs):
        # Change the seed in each iteration to examine algorithm stability
        current_seed = 42 + i 
        
        problem = TSPProblem(
            file_path = TASKS_DIR / f'{problem_name}.tsp', 
            seed = current_seed
        )
        
        optimizer = ASOptimizer(
            problem=problem, 
            max_fes=max_fes, 
            hyperparameters=hyperparameters,
        ) 
        

        start_time = time.perf_counter()
        best_cost, best_solution = optimizer.solve()
        end_time = time.perf_counter()
        
        exec_time = end_time - start_time
        
        best_costs.append(best_cost)
        execution_times.append(exec_time)
        

        fes_counts.append(int(optimizer.problem.fe_count))

        print(f'Run {i + 1} / {runs} finished in {exec_time}')
        

    # Calculate basic statistics
    global_best = float(np.min(best_costs))
    average_best = float(np.mean(best_costs))
    average_time = float(np.mean(execution_times))
    optimal_cost = float(optimal_cost)
    average_fes = float(np.mean(fes_counts))
    
    # Calculate RPD for each run
    rpds = [float((c - optimal_cost) / optimal_cost * 100.0) for c in best_costs]
    
    # Calculate aggregated RPD statistics
    global_best_rpd = float(np.min(rpds))
    average_best_rpd = float(np.mean(rpds))
    
    # Calculate standard deviation of RPD
    std_dev_rpd = float(np.std(rpds))
    
    results_data = {
        "algorithm": "Ant System",
        "problem_name": problem_name,
        "runs_count": runs,
        "max_fes": max_fes,
        "optimal_cost": optimal_cost,
        "summary_statistics": {
            "best_result": global_best,
            "best_result_rpd_percentage": global_best_rpd,
            "average_result": average_best,
            "average_result_rpd_percentage": average_best_rpd,
            "average_time_seconds": average_time,
            "average_fes_count": average_fes,
            "std_dev_rpd_percentage": std_dev_rpd
        },
        "raw_data_from_all_runs": {
            "costs": [float(c) for c in best_costs],
            "rpd_percentages": rpds,
            "execution_times_seconds": [float(t) for t in execution_times],
            "fes_counts": fes_counts,
        }
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_filename = RESULTS_DIR / f"{problem_name}_AS_experiment_summary.json"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)

if __name__ == '__main__':
    run_experiments_for_report(
        problem_name= 'att48',
        max_fes = 16_000
    )