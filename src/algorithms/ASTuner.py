import os
import json
import optuna
from pathlib import Path
from core.TSPProblem import TSPProblem
from algorithms.ASHyperparameters import ASHyperparameters
from algorithms.ASOptimizer import ASOptimizer

class ASTuner:
    def __init__(self, problem: TSPProblem, max_fes: int):
        self.problem_path = problem.file_path
        self.problem_name = self.problem_path.name
        self.problem = problem
        self.max_fes = max_fes
        
        self.save_filename = f"{self.problem_name}_hyperparameters.json"

    def _objective(self, trial: optuna.Trial) -> float:
        '''
        Objective function for Optuna. Executes a single trial with sampled hyperparameters.
        '''
        self.problem.fe_count = 0  #Reset the evaluation counter in the original problem object before each run
        
        min_ants = max(5, int(0.5 * self.problem.num_cities))
        max_ants = max(20, int(1.5 * self.problem.num_cities))

        #Define the hyperparameter search space
        m = trial.suggest_int("m", min_ants, max_ants)
        c = trial.suggest_float("c", 0.01, 10.0)
        p = trial.suggest_float("p", 0.1, 0.99)
        q = trial.suggest_float("q", 1.0, 500.0)
        alpha = trial.suggest_float("alpha", 0.5, 5.0)
        beta = trial.suggest_float("beta", 2.0, 5.0)
        
        optimizer = ASOptimizer(problem=self.problem, max_fes=self.max_fes, hyperparameters=ASHyperparameters(m=m, c=c, p=p, q=q, alpha=alpha, beta=beta))
        best_cost, _ = optimizer.solve()
        
        return best_cost

    def tune(self, save_directory: Path | str, n_trials: int = 50, save_file_name: str | None = None) -> ASHyperparameters:
        '''
        Runs the hyperparameter tuning process and saves the results.
        '''
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)
        
        best_params = study.best_params
        
        self.save_params(best_params, directory_path=save_directory)
        
        return ASHyperparameters(**best_params)

    def save_params(self, params: dict, directory_path: Path | str, save_file_name: str | None = None) -> None:
        '''
        Saves the hyperparameter dictionary to a JSON file in the specified directory.
        '''
        directory_path = Path(directory_path)
        os.makedirs(directory_path, exist_ok=True)
        full_path = directory_path / f'{save_file_name if save_file_name else self.save_filename}'
        
        data = {
            "problem_name": self.problem_name,
            "hyperparameters": params
        }
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_params(file_path: str) -> ASHyperparameters:
        '''
        Loads the hyperparameters from a given JSON file path.
        '''

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Hyperparameter file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        params = data.get("hyperparameters", {})
        
        return ASHyperparameters(**params)