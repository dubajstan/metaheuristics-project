class Bees:
    def __init__(self, problem, params):
        self.problem = problem
        self.best_solution = None
        self.best_cost = float("inf")

        self.params = params
        self.n_bees = params["n_bees"]
        self.n_elite = params["n_elite"]
        self.n_best = params["n_best"]
        self.elite_neigh = params["elite_neigh"]
        self.best_neigh = params["best_neigh"]
        self.max_iter = params["max_iter"]
        #self.random_seed

    def initialize_population(self):
        population = []

        for _ in range(self.n_bees):
            solution = self.problem.random_solution()
            population.append(solution)

        return population

    def neighborhood_search(self, solution, n_neighbors):
        best_neigh = None
        best_cost = float("inf")
        for _ in range(n_neighbors):
            neighbor = self.problem.neighbor(solution)
            cost = self.problem.cost(neighbor)

            if cost < best_cost:
                best_cost = cost
                best_neigh = neighbor
        return best_neigh


    def run(self):
        population = self.initialize_population()

        for _ in range(self.max_iter):

            population.sort(key=lambda solution: self.problem.cost(solution))

            best_current = population[0]
            best_current_cost = self.problem.cost(best_current)

            if best_current_cost < self.best_cost:
                self.best_cost = best_current_cost
                self.best_solution = best_current[:] #kopia zamiast ref

            elite = population[:self.n_elite]
            best = population[self.n_elite:self.n_best]

            new_population = []

            for solution in elite:
                best_neighbor = self.neighborhood_search(solution, self.elite_neigh)
                if best_neighbor is not None:
                    new_population.append(best_neighbor)
                else:
                    new_population.append(solution)

            for solution in best:
                best_neighbor = self.neighborhood_search(solution, self.best_neigh)
                if best_neighbor is not None:
                    new_population.append(best_neighbor)
                else:
                    new_population.append(solution)

            while len(new_population) < self.n_bees:
                new_population.append(self.problem.random_solution())


            population=new_population

        return self.best_solution, self.best_cost


