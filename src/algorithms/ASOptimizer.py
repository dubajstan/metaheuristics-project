from core.TSPProblem import TSPProblem
from algorithms.ASHyperparameters import ASHyperparameters
import numpy as np


class ASOptimizer:

    '''
    Optimizer class for solving the Traveling Salesman Problem using the Ant System algorithm.
    '''

    def __init__(self, problem: TSPProblem, max_fes: int, hyperparameters: ASHyperparameters, max_cycles_without_improvement: int = 75):
        
        '''
        Initializes the Ant System optimizer.

        Parameters
        -----
        problem : TSPProblem - The TSP problem instance containing distance data.
        max_fes : int - Maximum allowed number of cost function evaluations.
        hyperparameters : ASHyperparameters - Set of hyperparameters for the Ant System.
        max_cycles_without_improvement : int - Number of cycles to wait before early stopping.
        '''

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
        self.visibilities_beta = self.visibilities ** self.beta
        self.taus = np.full((self.dimension, self.dimension), self.c)
        self.taus_alpha = np.zeros_like(self.taus)
        self.delta_taus = np.zeros_like(self.taus)
        self.fes_total_count = 0
        self.max_cycles_without_improvement = max_cycles_without_improvement




    def _place_ants(self) -> None:
        '''
        Creates tabu list for each ant - list witch will hold visited nodes.
        First node in tabu list is a starting location for the ant.
        '''

        self.tabu_masks.fill(True)

        for ant_id in range(self.m):
            self.tabu_lists[ant_id].clear()
            self.tabu_lists[ant_id].append(start_node_id:=self.rng.integers(0,self.dimension))
            self.tabu_masks[ant_id, start_node_id] = False

    def _choose_next_node(self, i : int) -> int:
        '''
        Selects the next city for a specific ant based on pheromone levels and distances.

        Parameters
        -----
        i : int - The index of the current ant.

        Returns
        -----
        int - The index of the chosen next city.
        '''

        tabu_list = self.tabu_lists[i]

        current_node = tabu_list[-1]

        tabu_mask = self.tabu_masks[i]

        numerators = self.taus_alpha[current_node] * self.visibilities_beta[current_node]

        numerators = numerators * tabu_mask

        numerators_sum = np.sum(numerators)

        #edge case checking to avoid 0 division
        if numerators_sum == 0.:
            probabilities = tabu_mask / np.sum(tabu_mask)

        else:
            probabilities = numerators / numerators_sum

        return self.rng.choice(self.dimension, p = probabilities)

    def _update_delta_tau_matrix(self, costs : list[float]) -> None:
        '''
        Calculates the quantity of new pheromones left by ants on their paths and updates delta tau matix.

        Parameters
        -----
        costs : list[float] - A list of total route costs for all ants.
        '''


        for tabu, cost in zip(self.tabu_lists, costs):
            delta_increment = self.q / cost
            
            
            from_nodes = tabu
            to_nodes = tabu[1:] + [tabu[0]]
            
            self.delta_taus[from_nodes, to_nodes] += delta_increment
            self.delta_taus[to_nodes, from_nodes] += delta_increment

    def _update_intensity_matrix(self) -> None:
        '''Updates tau (intensity) matrix.'''

        self.taus *= self.p
        self.taus += self.delta_taus

    def _run_cycle(self) -> None:
        '''Runs a single cycle of the algorithm.'''

        self._place_ants()

        self.taus_alpha = self.taus ** self.alpha

        for _ in range(self.dimension - 1):
            for ant_id in range(self.m):
                self.tabu_lists[ant_id].append(node_id:=self._choose_next_node(ant_id))
                self.tabu_masks[ant_id, node_id] = False



    def solve(self) -> tuple[float, list[int]]:
        '''
        Runs the optimization algorithm. Stops when max function evaluations are reached or stgantion occurs.

        Returns
        -----
        tuple[float, list[int]] - A tuple containing the best cost and the best solution route.
        '''

        best_cost = float('inf')
        best_solution = None
        cycles_without_improvement = 0

        self._run_cycle()

        while not self.problem.fe_count > self.max_fes:

            self.fes_total_count = self.problem.fe_count
                
            costs = [self.problem.evaluate(route) for route in self.tabu_lists]

            if costs[min_id:=np.argmin(costs)] < best_cost:
                best_cost = costs[min_id]
                best_solution = self.tabu_lists[min_id].copy()
                cycles_without_improvement = 0
            
            else:
                cycles_without_improvement += 1

            self._update_delta_tau_matrix(costs)
            self._update_intensity_matrix()
            self.delta_taus.fill(0)

            self.history.append(((best_cost, best_solution), costs))

            if cycles_without_improvement > self.max_cycles_without_improvement:
                break

            self._run_cycle()

        return best_cost, best_solution
