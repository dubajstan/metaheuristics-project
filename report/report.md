# Comparison of selected metaheuristic algorithms  

**Systems and Decision Methods** semester project at
Wrocław University of Science and Technology

**Selected metaheuristic algorithms:**
* 	Ant Colony Optimization
*   Simulated Annealing
*   Bees Algorithm

**Authors:** 
*   Przemysław Mazurowski
*   Bartłomiej Niedbała
*   Szymon Majerczak  

**Submission date:** 2026/05/31  

---

## 1. The Travelling Salesman Problem (TSP) characteristics

The travelling salesman problem is an NP-hard problem in the theory of computational complexity. It does ask the following question: *Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?*

### 1.1. Definition
Formally TSP is an optimization problem that consists of finding the minimum Hamiltonian cycle in a complete weighted graph. 

Two types of TSP problem can be pointed:
* STSP - Symmetric Travelling Salesman Problem
* ATSP - Asymmetric Travelling Salesman Problem

In the Symmetric Travelling Salesman Problem (STSP), the weight of edge $(i,j)$ never differs from weight of $(j,i)$ edge for every pair of nodes. In the Asymmetric Travelling Salesman Problem (ATSP), however, these distances may vary. In this report only STSP is considered and any reference to TSP refers to STSP.

### 1.2. Problem complexity
A defining characteristic of the TSP is the fact that it is the NP-hard problem - any proposed solution, specific sequence of nodes, can be efficiently verified in polynomial time. Nonetheless, no algorithm has been discovered that can solve the TSP in polynomial time. Furthermore, TSP holds a significant place in computational complexity - every single NP problem can be reduced to an instance of TSP in polynomial time.

The TSP presents a massive computational challenge because its complexity grows factorially. For a TSP with $n$ cities, the total number of unique routes that can be evaluated is given by the formula:

$$\frac{(n-1)!}{2}$$

This combinatorial explosion means that even for a small instance of just 20 cities, the scale of available combinations expands rapidly:

$$\frac{19!}{2} \approx 6 \times 10^{16}$$

As a result, finding an exact optimal solution through brute-force computation becomes practically impossible as the number of locations increases. Way of finding near-optimal solutions to the TSP is applying metaheuristic algorithms.

### 1.3 Visualisation of the TSP instance and optimal solution

<table align = center>
  <tr>
    <td align = center>
      <img src = "./images/Illustration_of_an_unsolved_travelling_salesman_problem.png" alt="Unsolved TSP instance." width = 400 />
      <br />
      <em>Unsolved TSP instance.</em>
    </td>
    <td align = center>
      <img src = "./images/GLPK_solution_of_a_travelling_salesman_problem.png" alt = "Solution of the TSP instance" width = 400 />
      <br />
      <em>Solution of the TSP instance.</em>
    </td>
  </tr>
</table>

Sources:
* https://en.wikipedia.org/wiki/Travelling_salesman_problem
* https://pl.wikipedia.org/wiki/Problem_komiwoja%C5%BCera
* https://en.wikipedia.org/wiki/NP-hardness
* https://en.wikipedia.org/wiki/Travelling_salesman_problem#/media/File:GLPK_solution_of_a_travelling_salesman_problem.svg
* https://en.wikipedia.org/wiki/Travelling_salesman_problem#/media/File:Illustration_of_an_unsolved_travelling_salesman_problem.svg

---

## 2. Solution representation

The TSP solution for $n$ vertices can be represented as a permutation $\pi$ of the vertex set, where $v_i$ denotes the $i$-th vertex in the sequence:

$$\pi = (v_1, v_2, \dots, v_n)$$

The weights of edges  - distances - are retrieved from a distance matrix $D$, where each element $d_{i,j}$ represents the cost of traveling from vertex $i$ to vertex $j$:

$$D = [d_{i,j}]_{n \times n}$$

The objective of the optimization is to find an optimal tour $\pi^*$ that minimizes the cost function $F(\pi)$. This function calculates the sum of distances between consecutive vertices and includes the distance from the last vertex back to the first one to close the cycle:

$$F(\pi) = \sum_{i=1}^{n-1} d_{v_i, v_{i+1}} + d_{v_n, v_1}$$

$$\pi^* = \arg\min_{\pi \in \Pi} F(\pi)$$

where $\Pi$ denotes the set of all possible permutations  - valid tours.

---

## 3. Overwiew of selected alghorithms

### 3.1 Ant Colony Optimization

Ant System is a general-purpose heuristic algorithm which can be used to solve various optimization problems. It is a population-based approach. It was proposed by Marco Dorigo, Vittorio Maniezzo and Alberto Colorni in 1996 in the paper *[Ant System: Optimization by a Colony of Cooperating Agents](https://ieeexplore.ieee.org/document/484436)*. 
In this system,  search activities are distributed over so-called *ants* – agents with basic capabilities based on the behaviour of real ants. In fact, those agents possess capabilities beyond those of natural ants, like memorizing their paths.

Ant System is inspired by ants establishing the shortest routes from their colonies to feeding resources and back. The medium used by ants to set up those paths consists of pheromone trails. Each moving ant lays pheromone in different quantities. Ants tend to choose paths with more pheromone left while reinforcing the chosen path with their own pheromones. The shortest paths are eventually established, since shorter paths are covered with higher frequency than the longer ones.

The implementation is based on ant-cycle algorithm introduced in the original paper. However it does differ from original version – the stopping criterion relies on the maximum number of cost function evaluations instead of a fixed number of cycles and detecting stagnation behaviour is based on exceeding limit of cycles without improvement rather than on all ants traversing the same path. While the implementation remains semantically equivalent to the original algorithm, certain traditional loops have been replaced with optimized, vectorized operations.

The hyperparameters of the ant-cycle optimization algorithm:
 * $m$ - The total number of ants in the colony.
 * $c$ - The initial pheromone trail intensity assigned to each edge.
 * $p$ - The pheromone retention coefficient, where $(1 - p)$ represents the evaporation rate.
 * $q$ - A constant used to scale the amount of pheromone laid by ants on traversed edges.
 * $\alpha$ - A parameter controlling the influence of the pheromone trail on the ant's routing decisions.
 * $\beta$ - A parameter controlling the influence of actual distance on the ant's routing decisions.

Additionalny, in the implemention examined in this report, *cost function evaluation limit* and *cycles without best cost improvement limit* are to be set.

### 3.2 Simulated Annealing

### 3.3 Bees Algorithm

---

