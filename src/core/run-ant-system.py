from core.TSPProblem import TSPProblem
from algorithms.AntOptimizer import AntOptimizer
from algorithms.ASHyperparameters import ASHyperparameters
from analysis.AntResultAnalyzer import AntResultAnalyzer
from pathlib import Path
import time

def run_ant_system():
    
    project_dir = Path(__file__).resolve().parent.parent.parent
    path = project_dir / 'data' / 'tsplib95'/ 'tasks' / 'a280.tsp'

    problem = TSPProblem(
        file_path=path, 
        seed = 42
        )

    config = ASHyperparameters(
        m=280, 
        c=0.2, 
        p=0.5, 
        q=10, 
        alpha=2, 
        beta=2, 
        )
    
    optimizer = AntOptimizer(
        problem=problem, 
        max_fes=2_000, 
        config=config
        ) 
    
    
    start_time = time.perf_counter()
    
    optimizer.solve()

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    analyzer = AntResultAnalyzer(
        history=optimizer.history, 
        target_path=project_dir / 'charts',
        series_name='a280.tsp'
        )

    analyzer.plot_all_and_save()
    print(f'Time: {execution_time}')


if __name__ == '__main__':
    run_ant_system()
