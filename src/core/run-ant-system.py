from core.TSPProblem import TSPProblem
from algorithms.ASOptimizer import ASOptimizer
from algorithms.ASHyperparameters import ASHyperparameters
from analysis.ASResultAnalyzer import ASResultAnalyzer
from algorithms.ASTuner import ASTuner
from pathlib import Path
import time

def run_ant_system():
    
    project_dir = Path(__file__).resolve().parent.parent.parent
    problem_path = project_dir / 'data' / 'tsplib95'/ 'tasks' / 'a280.tsp'

    problem = TSPProblem(
        file_path=problem_path, 
        seed = 42
    )
    
    hyperparameters_path = project_dir / 'hyperparameters' / 'AS'

    
    #tuner = ASTuner(problem=problem, max_fes=5_000) 
    #hyperparameters = tuner.tune(hyperparameters_path, 10)
  
    hyperparameters = ASTuner.load_params(
        hyperparameters_path/'a280.tsp_hyperparameters.json'
    )

  
    optimizer = ASOptimizer(
        problem=problem, 
        max_fes=8_000, 
        hyperparameters=hyperparameters
    ) 
    
    
    start_time = time.perf_counter()
    
    optimizer.solve()

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    analyzer = ASResultAnalyzer(
        optimizer=optimizer, 
        target_path=project_dir / 'results' / 'AS',
        series_name='a280',
        exec_time=execution_time
    )

    analyzer.plot_all_and_save()


if __name__ == '__main__':
    run_ant_system()
