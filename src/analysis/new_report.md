### 3.2 Simulated Annealing

Simulated Annealing (SA) is a probabilistic, single-state metaheuristic algorithm used to locate global optima within large and complex search spaces. It is particularly effective for discrete optimization problems like the TSP, where traditional gradient-based approaches fail due to non-differentiable or highly rugged objective landscapes.

The algorithm is inspired by **metallurgy**, where a material is heated to a high temperature and then cooled slowly and systematically. Heating increases the kinetic energy of the atoms, allowing them to break out of their initial, flawed configurations. Slow cooling then permits them to settle into highly ordered, low-energy crystalline states. In the context of mathematical optimization, this physical metaphor maps directly to the algorithm's components:

* **State / Solution:** A specific permutation of cities representing a valid TSP tour ($\pi$).
* **Energy:** The value of the objective function ($F(\pi)$) evaluated at a given state. Because the objective is to minimize total distance, lower energy corresponds to a better solution.
* **Temperature ($T$):** A global control parameter that dictates the degree of randomness in the search space exploration.

#### 3.2.1 The Metropolis Acceptance Criterion

The defining feature of Simulated Annealing is its ability to escape local optima by accepting worse solutions based on a probabilistic threshold, preventing the algorithm from stalling like a purely greedy hill-climber.

When moving from a current tour with energy $E_{\text{current}}$ to a neighboring tour with energy $E_{\text{new}}$, the change in energy is computed as:

$$\Delta E = E_{\text{new}} - E_{\text{current}}$$

1. **Improving Moves ($\Delta E < 0$):** If the neighbor solution yields a shorter path, it is **always accepted**.
2. **Degrading Moves ($\Delta E \ge 0$):** If the neighbor solution is worse or equal, it is accepted with a probability $P$ determined by the **Metropolis criterion**:

$$P = \exp\left(-\frac{\Delta E}{T}\right)$$

##### Dynamic Behavior of the Criterion

* **High Temperature (Early Phase):** The denominator $T$ is large, driving $P$ close to 1. The algorithm accepts almost all moves, operating like a **Random Walk** to thoroughly explore the global search space.
* **Low Temperature (Late Phase):** As $T \to 0$, the probability $P \to 0$. The algorithm strictly rejects worse moves, morphing into a greedy **Local Search** to converge on the nearest minimum.
* **Impact of $\Delta E$:** Larger, highly damaging steps face a exponentially lower probability of acceptance than minor deviations.

#### 3.2.2 Algorithmic Workflow

The standard sequential execution loop of our Simulated Annealing implementation follows these precise steps:

1. **Temperature Initialization:** Calculate the optimal starting temperature ($T_0$) tailored to the specific scale of the TSP instance.
2. **State Initialization:** Generate a random initial candidate tour and evaluate its baseline cost (energy).
3. **Neighborhood Exploration:** Perturb the current solution using a localized transition operator to discover a neighboring tour.
4. **Metropolis Selection:** Apply the Metropolis Criterion based on a uniform random number $r \in [0, 1)$. If $\Delta E < 0$ or if $r < \exp(-\Delta E / T)$, accept the transition ($S \rightarrow S'$).
5. **Temperature Schedule:** Decrease the system's global temperature according to a predefined cooling strategy.
6. **Iterative Loop:** Repeat steps 3 through 5 continuously until the maximum function evaluations limit is reached.
7. **Result Delivery:** Return the absolute best-encountered tour tracked during the entire process.

#### 3.2.3 Implementation Choices

##### Adaptive Initial Temperature Estimation

Simulated Annealing exhibits high sensitivity to the choice of the initial temperature ($T_0$). Setting $T_0$ too high wastes an excessive number of function evaluations on a purely random walk, while setting it too low causes the algorithm to prematurely collapse into a greedy local search, getting trapped in the nearest local minimum.

To eliminate arbitrary static configurations, this implementation dynamically estimates $T_0$ based on empirical performance testing. Before starting the optimization loop, the algorithm performs preliminary sampling:

* It begins with a random solution, finds a neighbor, transitions to it, and repeats this chain for **1000 samples**.
* It calculates the mean value of the energy increases ($\overline{\Delta E}$) for degrading moves.
* It computes $T_0$ to guarantee that the initial acceptance probability of worse solutions is **approximately 80%** ($\chi_0 \approx 0.80$).

This relationship is derived by rearranging the Metropolis criterion:

$$P(\text{acceptance}) = \exp\left(-\frac{\overline{\Delta E}}{T_0}\right) \approx \chi_0 \implies T_0 = -\frac{\overline{\Delta E}}{\ln(\chi_0)}$$

##### Geometric Cooling Schedule

To lower the temperature at each iteration $k$, a **Geometric Decrease** profile is utilized:

$$T_{k+1} = \alpha \cdot T_k$$

Where the cooling rate $\alpha$ is a hyperparameter chosen based on the scale of the problem instance:

* **Large scales:** A larger $\alpha$ (e.g., $0.999999$) cools the system down slowly, giving the algorithm ample time to explore complex landscapes before freezing.
* **Small scales:** A slightly smaller $\alpha$ accelerates convergence as the search space requires fewer evaluations to be thoroughly explored.

##### Neighbor Operator

Two primary transition operators were evaluated during the development phase:

* **Swap Operator:** Interchanges the positions of two randomly chosen cities in the permutation vector.
* **2-opt Operator:** Selects two edges from a route, removes them, reverses the sub-route between them, and reconnects the edges to form a new valid cycle.

Because the **2-opt operator** proved significantly more efficient at discovering lower-energy configurations during preliminary testing, the swap operator was discarded entirely from the final implementation.

---