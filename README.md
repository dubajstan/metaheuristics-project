# metaheuristics-project
Project for System and Decision Methods class for University

szybka notatka zeby sie zgadzało: 

-w TSP musi sie znajdować metoda random_solution() ktora generuje 1 losowa permutacje z miastami (1 losowe przejscie po miastach)
(do wywoływania wielokrotnie w algorytmach)
- neighbor przyjmuje rozwiązanie neighbor(solution) --> zwraca nowe rozwiązanie (return new_solution) w formie kopii/nie modyfikuje oryginalu bo sie algorytm rozjebie
- cost: liczy poprostu z macierzy sume --> dostaje: [0,2,1,3] i liczy dist[0][2] + dist[2][1] + dist[1][3] + dist[3][0]<-- zeby wracało do 1 miasta

TSP:
__init__(self, dist_matrix)
random_solution(self)
cost(self, solution)
neighbor(self, solution)

Bees: 
init
initialize_population
neighborhood_search
run
