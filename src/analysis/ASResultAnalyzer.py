import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from algorithms.ASOptimizer import ASOptimizer

class ASResultAnalyzer:

    def __init__(self, optimizer: ASOptimizer, target_path: Path | str, exec_time: float,series_name: str = 'series') -> None:
        '''
        Initializes the analyzer with a target directory and loads data directly from an ASOptimizer instance.
        '''
        self.target_path = Path(target_path) / f'{series_name}'
        self.target_path.mkdir(parents=True, exist_ok=True)
        
        self.history = []
        self.series_name = series_name
        self.best_cost = float('inf')
        self.best_solution = None
        self.cycles = []
        self.fes_count = 0
        self.best_costs_per_cycle = []
        self.avg_costs_per_cycle = []
        self.std_costs_per_cycle = []
        
        self.exec_time = 0.0
        self.hyperparameters = {}
        self.problem_name = "unknown"
        
        self.load_series(optimizer, exec_time, series_name)

    def load_series(self, optimizer: ASOptimizer, exec_time: float, series_name: str = 'series') -> None:
        '''
        Extracts history, execution time, hyperparameters, and problem name from the optimizer object.
        '''
        if not optimizer.history:
            raise ValueError("Optimizer history data cannot be empty.")
            
        self.history = optimizer.history
        self.exec_time = exec_time
        
        self.hyperparameters = {
            "m": optimizer.m,
            "c": optimizer.c,
            "p": optimizer.p,
            "q": optimizer.q,
            "alpha": optimizer.alpha,
            "beta": optimizer.beta
        }
            
        self.problem_name = Path(optimizer.problem.file_path).name
 
            
        if series_name is not None:
            self.series_name = series_name
            
        last_record = self.history[-1]
        self.best_cost = last_record[0][0]
        self.best_solution = last_record[0][1]
        
        self.cycles = list(range(1, len(self.history) + 1))
        self.fes_count = optimizer.fes_total_count
        
        self.best_costs_per_cycle = [record[0][0] for record in self.history]
        self.avg_costs_per_cycle = [np.mean(record[1]) for record in self.history]
        self.std_costs_per_cycle = [np.std(record[1]) for record in self.history]

    def plot_best_cost(self) -> None:
        '''Displays and saves the chart of the global best cost over cycles.'''
        plt.figure(figsize=(10, 6))
        plt.plot(self.cycles, self.best_costs_per_cycle, label='Best Cost', color='green', linewidth=2, marker='o')
        plt.title(f'Best Cost  - {self.series_name}')
        plt.xlabel('Cycle Number')
        plt.ylabel('Cost')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        save_path = self.target_path / f'{self.series_name}_best_cost.png'
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def plot_cost_std_dev(self) -> None:
        '''Displays and saves the chart of the cost standard deviation within each cycle.'''
        plt.figure(figsize=(10, 6))
        plt.plot(self.cycles, self.std_costs_per_cycle, label='Standard Deviation', color='orange', linewidth=2, marker='o')
        plt.title(f'Cost Standard Deviation - {self.series_name}')
        plt.xlabel('Cycle Number')
        plt.ylabel('Standard Deviation')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        save_path = self.target_path / f'{self.series_name}_std_dev.png'
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def plot_average_cost(self) -> None:
        '''Displays and saves the chart of the average population cost along with the current best cost.'''
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.cycles, self.avg_costs_per_cycle, label='Average Cost', color='blue', linewidth=2, marker='o')
        plt.plot(self.cycles, self.best_costs_per_cycle, label='Current Best Cost', color='green', linestyle='--', linewidth=1.5, marker='o')
        
        plt.title(f'Average Population Cost - {self.series_name}')
        plt.xlabel('Cycle Number')
        plt.ylabel('Cost')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        save_path = self.target_path / f'{self.series_name}_average_cost.png'
        plt.savefig(save_path)
        plt.show()
        plt.close()

    def save_results_json(self) -> None:
        '''Saves the execution summary (problem name, hyperparameters, best cost, best solution, execution time and cycles number) to a JSON file.'''
        save_path = self.target_path / f'{self.series_name}_result.json'
        
        solution_data = self.best_solution
        if hasattr(solution_data, 'tolist'):
            solution_data = solution_data.tolist()
        elif isinstance(solution_data, (np.ndarray, list)):
            solution_data = [int(node) for node in solution_data]

        results = {
            'problem_name': self.problem_name,
            'series_name': self.series_name,
            'hyperparameters': self.hyperparameters,
            'execution_time_seconds': self.exec_time,
            'total_fes': self.fes_count,
            'cycles': len(self.cycles),
            'best_cost': self.best_cost,
            'best_solution': solution_data,
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        
    def plot_all_and_save(self) -> None:
        '''Helper method to automatically generate and save all available plots and data metrics.'''
        self.plot_best_cost()
        self.plot_cost_std_dev()
        self.plot_average_cost()
        self.save_results_json()