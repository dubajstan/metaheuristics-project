from core.TSPProblem import TSPProblem
from algorithms.ASHyperparameters import ASHyperparameters
import numpy as np


class ASOptimizer:

    def __init__(self, problem: TSPProblem, max_fes: int, hyperparameters: ASHyperparameters):
        
        #general
        self.problem = problem
        self.max_fes = max_fes
        self.history = []
        self.dimension = problem.num_cities
        self.rng = problem.rng
        self.exec_time = 0

        self.problem.fe_count = 0

        #ant system specific
        self.m = hyperparameters.m
        self.c = hyperparameters.c
        self.p = hyperparameters.p
        self.q = hyperparameters.q
        self.alpha = hyperparameters.alpha
        self.beta = hyperparameters.beta
        self.tabu_lists = [[] for _ in range(self.m)]
        self.tabu_masks = np.ones((self.m, self.dimension), dtype = bool)
        self.distances = problem.distance_matrix
        self.visibilities = np.zeros_like(self.distances)
        non_zero_distances_mask = self.distances != 0
        self.visibilities[non_zero_distances_mask] = 1.0 / self.distances[non_zero_distances_mask]
        self.taus = np.full((self.dimension, self.dimension), self.c)
        self.delta_taus = np.zeros_like(self.taus)
        self.fes_total_count = 0




    def place_ants(self) -> None:
        '''
        Creates tabu list for each ant - list witch will hold visited nodes.
        First vertex in tabu list is a starting location for the ant.
        '''

        self.tabu_masks.fill(True)

        for ant_id in range(self.m):
            self.tabu_lists[ant_id].clear()
            self.tabu_lists[ant_id].append(start_node_id:=self.rng.integers(0,self.dimension))
            self.tabu_masks[ant_id, start_node_id] = False

    def choose_next_node(self, i : int) -> int:
        '''Selects next node that i-th ant will visit.'''

        tabu_list = self.tabu_lists[i]

        current_node = tabu_list[-1]

        tabu_mask = self.tabu_masks[i]

        candidate_vertices = np.where(tabu_mask)[0] #array of indexes at wich values were set to true

        candidates_taus = self.taus[current_node, tabu_mask] #trail intesities
        candidates_visibilities = self.visibilities[current_node, tabu_mask] #visibility factors for (current_node,next_node) edges
        candidates_numerators = (candidates_taus ** self.alpha) * (candidates_visibilities ** self.beta)

        numerators_sum = np.sum(candidates_numerators)

        #edge case checking to avoid 0 division
        if numerators_sum == 0.:
            candidates_probabilities = np.ones_like(candidates_numerators) / len(candidates_numerators)

        else:
            candidates_probabilities = candidates_numerators / numerators_sum

        return self.rng.choice(candidate_vertices, p = candidates_probabilities)

    def update_delta_tau_matrix(self, costs : list[float]) -> None:
        '''Updates delta tau matrix.'''


        for tabu, cost in zip(self.tabu_lists, costs):
            delta_increment = self.q / cost
            for i in range(len(tabu) - 1):
                self.delta_taus[tabu[i], tabu[i+1]] += delta_increment
                self.delta_taus[tabu[i+1], tabu[i]] += delta_increment
            
            self.delta_taus[tabu[-1], tabu[0]] += delta_increment
            self.delta_taus[tabu[0], tabu[-1]] += delta_increment

    def update_intensity_matrix(self) -> None:
        '''Updates tau (intensity) matrix.'''

        self.taus *= self.p
        self.taus += self.delta_taus

    def run_cycle(self) -> None:
        '''Runs a single cycle of the algorithm.'''

        self.place_ants()
        for _ in range(self.dimension - 1):
            for ant_id in range(self.m):
                self.tabu_lists[ant_id].append(node_id:=self.choose_next_node(ant_id))
                self.tabu_masks[ant_id, node_id] = False



    def solve(self):
        '''Runs the algorithm. Stops when cost function evaluations limit is exceeded and discards last (illegal) cycle.'''

        best_cost = float('inf')
        best_solution = None
        
        self.run_cycle()

        while not self.problem.fe_count > self.max_fes:

            self.fes_total_count = self.problem.fe_count
                
            costs = [self.problem.evaluate(route) for route in self.tabu_lists]

            if costs[min_id:=np.argmin(costs)] < best_cost:
                best_cost = costs[min_id]
                best_solution = self.tabu_lists[min_id].copy()

            self.update_delta_tau_matrix(costs)
            self.update_intensity_matrix()
            self.delta_taus.fill(0)

            self.history.append(((best_cost, best_solution), costs))

            self.run_cycle()

        return best_cost, best_solution
