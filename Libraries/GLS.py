########## Libraries ##########
import numpy as np
import random
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from TabuSearch import (
    ObjFun,
    first_solution
)

def get_neighbors_2opt(Solution):
    """
    Generar vecinos usando el método 2-opt.
    """
    neighbors = []
    n = len(Solution)
    for i in range(n - 1):
        for j in range(i + 2, n):  # Asegurar que hay al menos un nodo entre i y j
            # Generar el vecino 2-opt
            new_solution = np.copy(Solution)
            new_solution[i:j+1] = np.flip(Solution[i:j+1])  # Revertir el segmento entre i y j
            neighbors.append(new_solution)
    return neighbors

def best_neighbor(Neighborhood, alpha, DistanceMatrix, penalties):
    """
    Buscar el mejor vecino según la penalización.
    """
    Best = None
    Best_f = float('inf')

    for candidate in Neighborhood:
        candidate_f = ObjFun(candidate, DistanceMatrix)
        candidate_f_p = candidate_f

        # Agregar penalización a la función objetivo
        for i in range(len(candidate) - 1):
            candidate_f_p += alpha * penalties[candidate[i]-1, candidate[i + 1]-1]

        if candidate_f_p < Best_f:
            Best = copy.deepcopy(candidate)
            Best_f = copy.deepcopy(candidate_f_p)

    return Best, Best_f


def Guided_Local_Search(DistanceMatrix, AmountNodes, MaxOFcalls=100, alpha=0.2):
    """
    Implementación de Guided Local Search para TSP
    """
    # Inicialización de la solución
    BestSolution = first_solution(AmountNodes)
    CurrentSolution = np.copy(BestSolution)

    # Matriz de penalizaciones
    penalties = np.zeros((AmountNodes, AmountNodes))  # Penalización entre todas las ciudades
    utilities = np.zeros((AmountNodes, AmountNodes))  # Utilidad de los pares de ciudades
    
    # Registro de resultados
    bests_n = []  # Historial de costos de las mejores soluciones
    best_sol = []  # Historial de las mejores soluciones
    
    calls = 1  # Contador de llamadas a la función objetivo

    # Regularización dinamica.
    regularizacion = alpha * (ObjFun(BestSolution,DistanceMatrix) / AmountNodes)

    while calls < MaxOFcalls:
        # Generar vecinos usando 2-opt
        Neighborhood = get_neighbors_2opt(CurrentSolution)
        calls += len(Neighborhood)

        # Encontrar el mejor vecino usando penalización
        BestNeighbor, BestNeighbor_f = best_neighbor(Neighborhood, alpha, DistanceMatrix, penalties)

        regularizacion = alpha * (BestNeighbor_f / AmountNodes)

        # Calcular la utilidad y actualizar las penalizaciones
        max_utility = 0
        for i in range(len(BestNeighbor) - 1):
            pair_penalty = penalties[BestNeighbor[i]-1, BestNeighbor[i + 1]-1]
            pair_distance = DistanceMatrix[BestNeighbor[i]-1, BestNeighbor[i + 1]-1]
            utilities[BestNeighbor[i]-1, BestNeighbor[i + 1]-1] = pair_distance / (1 + pair_penalty)

            # Encontrar la utilidad máxima
            max_utility = max(max_utility, utilities[BestNeighbor[i]-1, BestNeighbor[i + 1]-1])

        # Actualizar las penalizaciones basadas en la utilidad máxima
        for i in range(len(BestNeighbor) - 1):
            if utilities[BestNeighbor[i]-1, BestNeighbor[i + 1]-1] == max_utility:
                penalties[BestNeighbor[i]-1, BestNeighbor[i + 1]-1] += 1 

        # Actualizar la solución si se encuentra una mejora
        if BestNeighbor_f < ObjFun(BestSolution, DistanceMatrix):
            BestSolution = np.copy(BestNeighbor)
        
        CurrentSolution = BestNeighbor

        # Registrar las mejores soluciones y costos
        best_sol.append(ObjFun(BestSolution, DistanceMatrix))
        bests_n.append(ObjFun(BestNeighbor, DistanceMatrix))

    return bests_n, best_sol