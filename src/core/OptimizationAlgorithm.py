from abc import ABC, abstractmethod
from src.core.Problem import Problem

class OptimizationAlgorithm(ABC):
    def __init__(self, problem: Problem, max_fes: int):
        self.problem = problem
        self.max_fes = max_fes
        self.history = [] # trzyma historie np. (fes_count, best_fitness) albo (fes_count, rodzina calych rozwiazan) aby potem analizowac zbierznosc itp

    
    @abstractmethod
    def solve(self):
        """Executes the algorithm. Must check self.problem.fe_count < self.max_fes"""
        pass



        