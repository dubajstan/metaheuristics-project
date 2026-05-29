from core.TSPProblem import TSPProblem
from algorithms.ASOptimizer import ASOptimizer
from algorithms.ASHyperparameters import ASHyperparameters
from analysis.ASResultAnalyzer import ASResultAnalyzer
from algorithms.ASTuner import ASTuner
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
HYPERPARAMETERS_DIR = PROJECT_DIR / 'hyperparameters' / 'AS'
TASKS_DIR = PROJECT_DIR / 'data' / 'tsplib95'/ 'tasks'
OPTIMAL_SOLUTIONS_DIR = PROJECT_DIR / 'data' / 'tsplib95' / 'solutions'
RESULTS_DIR = PROJECT_DIR / 'results' / 'AS'

def tune_for_problem(problem_name: str, seed: int, max_fes: int, iterations: int) -> None:
    '''
    Initializes the TSP problem and runs the hyperparameter tuning process using Optuna.

    Parameters
    -----
    problem_name : str - Name of the problem instance without extension.
    seed : int - Fixed seed.
    max_fes : int - Maximum number of cost function evaluations per trial.
    iterations : int - Number of trials for the tuner to perform.
    '''   

    problem = TSPProblem(
        file_path = TASKS_DIR /f'{problem_name}.tsp', 
        seed = seed
    )

    tuner = ASTuner(
        problem=problem, 
        max_fes=max_fes
        )
     
    tuner.tune(HYPERPARAMETERS_DIR, iterations)
  


def run_ant_system(problem_name: str, seed: int, max_fes: int, series_name: str | None = None) -> None:

    '''
    Loads tuned hyperparameters, runs the Ant System optimization, and generates analysis reports.

    Parameters
    -----
    problem_name : str - Name of the problem file without extension.
    seed : int - Random seed for reproducibility.
    max_fes : int - Maximum number of cost function evaluations for the run.
    series_name : str | None - Optional name for the output series. Defaults to problem_name if None.
    '''


    if not series_name:
        series_name = problem_name
    
    problem = TSPProblem(
        file_path = TASKS_DIR /f'{problem_name}.tsp', 
        seed = seed
    )
      
    hyperparameters = ASTuner.load_params(
        HYPERPARAMETERS_DIR/f'{problem_name}_hyperparameters.json'
    )

    optimizer = ASOptimizer(
        problem=problem, 
        max_fes=max_fes, 
        hyperparameters=hyperparameters,
    ) 
    
    start_time = time.perf_counter()
    
    optimizer.solve()

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    analyzer = ASResultAnalyzer(
        optimizer=optimizer, 
        target_dir=RESULTS_DIR,
        optimal_sol_dir = OPTIMAL_SOLUTIONS_DIR,
        series_name=series_name,
        exec_time=execution_time
    )

    analyzer.plot_all_and_save()


if __name__ == '__main__':
    
    tune_for_problem(
        problem_name='att48',
        seed=42, 
        max_fes=20_000, 
        iterations=100
        )

    run_ant_system(
        problem_name='att48',
        seed=42, 
        max_fes=10000, 
        series_name=None
        )
