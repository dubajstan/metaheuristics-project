import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class AntResultAnalyzer:

    def __init__(self, history: list[tuple], target_path: Path | str, series_name: str = "series") -> None:
        '''
        Initializes the analyzer with a target directory and loads the first data series.
        '''
        self.target_path = Path(target_path) / f'ants-{series_name}'
        self.target_path.mkdir(parents=True, exist_ok=True)
        
        self.history = []
        self.series_name = series_name
        self.best_cost = float('inf')
        self.best_solution = None
        self.cycles = []
        self.best_costs_per_cycle = []
        self.avg_costs_per_cycle = []
        self.std_costs_per_cycle = []
        
        self.load_series(history, series_name)

    def load_series(self, history: list[tuple], series_name: str = None) -> None:
        '''
        Repeats the initialization process for a new history series using the existing target path.
        Optionally updates the series name.
        '''
        if not history:
            raise ValueError("History data cannot be empty.")
            
        self.history = history
        if series_name is not None:
            self.series_name = series_name
            
        last_record = self.history[-1]
        self.best_cost = last_record[0][0]
        self.best_solution = last_record[0][1]
        
        self.cycles = list(range(1, len(self.history) + 1))
        
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
        
    def plot_all_and_save(self) -> None:
        '''Helper method to automatically generate and save all available plots.'''
        self.plot_best_cost()
        self.plot_cost_std_dev()
        self.plot_average_cost()