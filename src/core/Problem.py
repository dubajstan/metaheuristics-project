from abc import ABC, abstractmethod
import numpy as np


class Problem(ABC):
    def __init__(self, seed: int = 42):
        self.fe_count = 0 # klasa problem automatycznie bedzie sledzic ile razy zostal obliczony fitness/cost
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    
    def evaluate(self, solution) -> float:
        """Wrapper to count evaluations and call the actual fitness function"""
        self.fe_count += 1
        return self._calculate_fitness(solution)
    

    @abstractmethod
    def _calculate_fitness(self, solution) -> float:
        """The actual math/logic of the problem, can be also a wrapper if used with lirary like tsplib95"""
        pass


    @abstractmethod
    def create_random_solution(self):
        """Generate a valid random solution (random permutation for TSP, random vector, etc.)"""
        pass


    @abstractmethod
    def get_neighbour(self, solution):
        """Returns a modified version ofthe solution, ex. swap two cities"""
        pass