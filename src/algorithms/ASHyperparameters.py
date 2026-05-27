from dataclasses import dataclass

@dataclass
class ASHyperparameters:
    m: int          # The total number of ants in the colony.
    c: float        # The initial pheromone trail intensity assigned to each edge.
    p: float        # The pheromone retention coefficient.
    q: float        # A constant used to scale the amount of pheromone laid by ants.
    alpha: float    # Controls the influence of the pheromone trail.
    beta: float     # Controls the influence of actual distance.