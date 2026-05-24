from Problem import Problem
import tsplib95
from pathlib import Path
import numpy as np


class TSPProblem(Problem):
    def __init__(self, file_path: Path | str, seed: int = 42):
        super().__init__()
        file_path = Path(file_path)
        self.tsplib_problem = tsplib95.load(file_path) # parsuje problem z pliku
        self._raw_nodes = list(self.tsplib_problem.get_nodes())
        self.num_cities = len(self._raw_nodes)
        self.nodes = np.arange(self.num_cities)
        self.start_node = 0        
        self._distance_matrix = self._build_distance_matrix()
        np.fill_diagonal(self._distance_matrix, 0.0)
        self.rng = np.random.default_rng(seed)


    def _build_distance_matrix(self) -> np.ndarray:
        """Creates internal weight matrix in numpy"""
        matrix = np.zeros((self.num_cities, self.num_cities), dtype=np.float64)
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                raw_i = self._raw_nodes[i]
                raw_j = self._raw_nodes[j]
                matrix[i, j] = self.tsplib_problem.get_weight(raw_i, raw_j)
        return matrix


    def _calculate_fitness(self, solution: np.ndarray | list[int]) -> float:
        """Calculates total tour distance.
        Expects solution format: [city_a, city_b, ...] with start node, ex: [2,4,3,1,0]
        """
        sol = np.asarray(solution)
        internal_distance = np.sum(self._distance_matrix[sol[:-1], sol[1:]])
        loop_closure = self._distance_matrix[sol[-1], sol[0]]
        return float(internal_distance + loop_closure)


    def create_random_solution(self) -> np.ndarray:
        """Generates valid initial tour starting from start_node"""
        cities = np.copy(self.nodes)
        self.rng.shuffle(cities)
        return cities
    

    def get_neighbour(self, solution: np.ndarray | list[int]) -> np.ndarray:
        """Returns a neighbour by swapping two random cities with each other"""
        neighbour = np.copy(solution)
        idx1, idx2 = self.rng.choice(len(neighbour), size=2, replace=False)
        neighbour[idx1], neighbour[idx2] = neighbour[idx2], neighbour[idx1]
        return neighbour
    
    def get_distance(self, city_a: int, city_b: int) -> float:
        """Returns distance between city_a and city_b using internal matrix"""
        return float(self._distance_matrix[city_a, city_b])


    @property
    def cities(self) -> np.ndarray:
        """Returns all cities including start node"""
        return self.nodes

    @property
    def distance_matrix(self) -> np.ndarray:
        """Returns a safe copy of the distance matrix as a numpy array"""
        return self._distance_matrix.copy()
    

