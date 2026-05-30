# Simulated Annealing (SA)

## Overview

**Simulated Annealing (SA)** is a probabilistic, single-state metaheuristic algrithm used to find global optima in large and complex search space. It is often used for discrete search spaces, where traditional gradient-based approaches fail due to a non-differentiable or higly rugged objective functions. 

The algorithm was inspired by **metalurgy**, where a material is heated to a high temperature and then cooled slowly ans systematically. Heating increates kinetic energy of the atoms, allowing them to break out of the initial configuration. Slow cooling allows them to settle into a more optimal and low-energy states, by allowing atoms to reorder. As a result, it minimizes the interal deffects making the material more durable. 

## Core concepts

In the context of mathematical optimization, the physical metaphor maps directly to algorithm components:

* **State / Solution:** A specific solution to the problem coded into a vector.
* **Energy:** The value of the objective function or cost function evaluated at a given state. Typically we want to minimize the objective function, meaning lower energy is better.
* **Temperature:** A global control parameter that dictates the degree of randomness in the search. High temperature allows the algorithm to explore highly, but low temperature lets it exploit local optima.

## The Metropolis Acceptance criterion

The defining feature of SA is the ability to escape local optima. It acts as a improved greedy hill-climbing approaches that only accepts solutions that improve the objective function. SA can accept worse solutions based on a probabilistic threshold. When moving from a current state with energy $E_{\text{current}}$ to a neighbouring state with energy $E_{\text{new}}$, we calculate the change in energy:

$$
\Delta E = E_{\text{new}} - E_{\text{current}}
$$

1. **Improving moves ($\Delta E < 0$):** If the neighbor solution is better, it is **always accepted**.

2. **Degrading solution ($\Delta E \ge 0$):** If the neighbor solution is worse or the same, it is accepted with a probability $P$ determied by the **Metropolis criterion**:

$$
P = \exp\left(-\frac{\Delta E}{T}\right)
$$

**Behavior of the Criterion:**
* **Early in the run (High $T$):** The denominator is large, making $P$ close to 1. The algorithm accepts almost all moves, acting like a **Random Walk** to thoroughly explore the search space.
* **Late in the run (Low $T$):** As $T \to 0$, the probability $P \to 0$. The algorithm strictly rejects worse moves, morphing into a greedy **Local Search** to converge on the nearest minimum.
* **Impact of $\Delta E$:** Larger, more damaging steps have a much lower probability of acceptance than minor deviations.

## Algorithmic Workflow

The standard sequential execution loop of a Simulated Annealing chain follows these steps:

1.  **Initialization:** Generate an initial random solution $S$ and evaluate its energy $E(S)$. Compute or set the starting temperature $T_0$.
2.  **Neighbor Generation:** Slightly perturb the current solution using a problem-specific transition operator to create a neighboring solution $S'$.
3.  **Evaluation:** Calculate $E(S')$ and determine $\Delta E$.
4.  **Acceptance Check:** Generate a uniform random number $r \in [0, 1)$. If $\Delta E < 0$ or if $r < \exp(-\Delta E / T)$, accept the transition ($S \rightarrow S'$).
5.  **Cooling:** Decrease the temperature $T$ according to a predefined cooling schedule based on the current iteration counter.
6.  **Termination:** Repeat steps 2–5 until a stopping criterion is met.

## Cooling Schedules

The choice of how the temperature updates at each iteration determines the balance between exploration and exploitation.

| Schedule Type | Equation / Update Rule | Description |
| :--- | :--- | :--- |
| **Geometric** | $T_{k+1} = \alpha \cdot T_k$ | The most common profile. Smooth, predictable decay. Usually, $0.95 \le \alpha \le 0.999999$. |
| **Linear** | $T_k = T_0 - k \cdot \beta$ | Drops the temperature by a fixed amount $\beta$ every step. Fast, but can drop too early. |
| **Logarithmic** | $T_k = \frac{T_0}{\ln(1 + k)}$ | Guaranteed mathematically to find the global optimum given infinite time, but practically too slow for real-world application. |

## Finding the best initial temperature

The Simulated Annealing algorithm exhibits a **high sensitivity to the choice of the initial temperature ($T_0$)**. 
* If $T_0$ is set too high, the algorithm wastes an excessive number of function evaluations performing a purely random walk without converging.
* If $T_0$ is set too low, the algorithm prematurely collapses into a greedy local search (hill climbing), getting trapped in the nearest local minimum.

To address this sensitivity, and based on empirical performance testing (detailed in this [GitHub repository](https://github.com/paololapo/Simulated_annealing_for_TSP)), this implementation dynamically estimates $T_0$ rather than utilizing an arbitrary static value.

Before starting the optimization process, the algorithm performs preliminary sampling of random transitions to approximate the distribution of energy differences ($\Delta E$) for degrading moves. Based on the mean value of these energy increases ($\overline{\Delta E}$), the initial temperature $T_0$ is calculated to guarantee that the initial acceptance probability of worse solutions is **approximately 80%** ($\chi_0 \approx 0.80$).

This relationship is derived directly by rearranging the Metropolis criterion:

$$P(\text{acceptance}) = \exp\left(-\frac{\overline{\Delta E}}{T_0}\right) \approx \chi_0$$

Solving for $T_0$ yields the implementation formula:

$$T_0 = -\frac{\overline{\Delta E}}{\ln(\chi_0)}$$

## Algorithm Steps

The core workflow of our Simulated Annealing algorithm follows these precise, sequential steps:

1. **Temperature Initialization:** Calculate the optimal starting temperature ($T_0$) tailored to the specific problem instance scale.
2. **State Initialization:** Generate a random initial candidate solution and evaluate its baseline cost (energy).
3. **Neighborhood Exploration:** Perturb the current solution using a localized operator to discover a neighboring solution.
4. **Metropolis Selection:** Apply the **Metropolis Criterion** to decide whether to accept the newly found neighbor or reject it and stick with the current state.
5. **Temperature Schedule:** Decrease the system's global temperature according to a predefined cooling strategy.
6. **Iterative Loop:** Repeat steps 3 through 5 continuously until the predefined termination criterion (maximum function evaluations) is met.
7. **Result Delivery:** Return the absolute best-encountered solution tracked during the entire process.

## Implementation Choices

### 1. Adaptive Initial Temperature Estimation (Step 1)
Instead of guessing a hardcoded value for $T_0$, our code automatically measures the "ruggedness" of the problem before the optimization loop begins. 

* The algorithm starts with a single random solution, finds its neighbor, moves to it, and repeats this chain for **1000 samples**. 
* It calculates the average difference in energy ($\overline{\Delta E}$).
* It plugs this average into the formula $T_0 = -\overline{\Delta E} / \ln(0.8)$. This mathematical setup guarantees that at the very beginning of the run, the algorithm has a predictable **80% acceptance rate**.

### 2. Geometric Cooling Schedule (Step 5)
To lower the temperature at each iteration $k$, we implemented a **Geometric Decrease** profile defined by the following update rule:

$$T_{k+1} = \alpha \cdot T_k$$

Where $\alpha$ (the cooling rate) is a hyperparameter chosen based on the scale of the problem instance:
* **For larger problem scales:** We use a larger $\alpha$ (e.g., $0.999999$). This cools the system down incredibly slowly, giving the algorithm ample time to explore complex landscapes before freezing.
* **For smaller problem scales:** A slightly smaller $\alpha$ can be used to accelerate convergence, as the search space requires fewer evaluations to be thoroughly explored.

### 3. Neighbor operator (Step 3)
We tested out two mainly used operators:

* **Swap operator** which simply swaps two cities.
* **2-opt operator** which selects 2 edges from a route, removes them with a route between and reconnects it in reverse order.

**Since the 2-opt operator proved significantly more efficient during preliminary tests, we discarded the swap operator right at the beginning.**