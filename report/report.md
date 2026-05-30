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

In the Symmetric Travelling Salesman Problem (STSP), the weight of edge (A,B) never differs from weight of (B,A) edge for every pair of nodes. In the Asymmetric Travelling Salesman Problem (ATSP), however, these distances may vary. In this report only STSP is considered and any reference to TSP refers to STSP.

### 1.2. Problem complexity
A defining characteristic of the TSP is the fact that it is the NP-hard problem - any proposed solution, specific sequence of nodes, can be efficiently verified in polynomial time. Nonetheless, no algorithm has been discovered that can solve the TSP in polynomial time. Furthermore, TSP holds a significant place in computational complexity - every single NP problem can be reduced to an instance of TSP in polynomial time.

The TSP presents a massive computational challenge because its complexity grows factorially. For a TSP with $n$ cities, the total number of unique routes that can be evaluated is given by the formula:

$$\frac{(n-1)!}{2}$$

This combinatorial explosion means that even for a small instance of just 20 cities, the scale of available combinations expands rapidly:

$$\frac{19!}{2} \approx 6 \times 10^{16}$$

As a result, finding an exact optimal solution through brute-force computation becomes practically impossible as the number of locations increases. Way of finding near-optimal solutions to the TSP is applying metaheuristic algorithms.

### 1.3 A graphical example of the TSP instance and optimal solution

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
      <em>Solution of the TSP instance</em>
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
