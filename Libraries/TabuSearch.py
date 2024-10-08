########## Libraries ##########
import numpy as np

########## Functions TS ##########

def ObjFun(Solution, DistanceMatrix):
    """
    ObjFun (function)
        Input: Permutation vector and
        distance matrix.
        Output: Value in objective function.
        Description: Calculates the objective
        function using permutation vector and
        distance matrix.
        The TSP objective function is the sum
        of all edge's cost.
    """
    # Initialization of length of vector and sum.
    total_cost = 0

    # Loop through the solution to sum the costs of edges.
    for i in range(len(Solution) - 1):
        total_cost += DistanceMatrix[Solution[i] - 1][Solution[i + 1] - 1]

    # Add the cost from the last node back to the first to complete the tour.
    total_cost += DistanceMatrix[Solution[-1] - 1][Solution[0] - 1]

    return total_cost

def first_solution(AmountNodes):
    """
    first_solution (function)
        Input: Number of nodes.
        Output: Permutation vector.
        Description: Generates first solution.
    """
    # Random permutation node.
    Vector = np.random.permutation(np.arange(1, AmountNodes + 1))
    return Vector

def two_opt_swap(Solution, i, j):
    """
    two_opt_swap (function)
        Input: Solution, and two indices i and j.
        Output: New solution with 2-opt swap applied.
        Description: Reverses the tour between indices i and j.
    """
    new_solution = np.copy(Solution)
    new_solution[i:j+1] = np.flip(Solution[i:j+1])  # Reverse the segment
    return new_solution

def get_neighbors_2opt(Solution):
    """
    get_neighbors_2opt (function)
        Input: Reference solution.
        Output: List of neighbor solutions (2-opt neighborhood).
        Description: Generates a neighborhood using 2-opt swaps.
    """
    neighbors = []
    # Apply 2-opt by selecting all possible pairs of edges to swap.
    for i in range(len(Solution) - 1):
        for j in range(i + 1, len(Solution)):
            neighbor = two_opt_swap(Solution, i, j)
            neighbors.append(neighbor.tolist())
    return neighbors

def best_neighbor(Neighborhood, DistanceMatrix, TabuList):
    """
    get_neighbors (function)
        Input: Neighborhood, Distance Matrix and
        Tabu List.
        Output: Neighbor with best improvement 
        in objective function.
        Description: Evaluates Neighborhood
        of permutation solutions.
    """
    # Best solution and value in obj. function.
    Best = None
    Best_f = float('inf')

    # Searching.
    for candidate in Neighborhood:
        # Take candidate.
        Candidate_f = ObjFun(candidate, DistanceMatrix)

        # Conditions for change.
        if (Best_f > Candidate_f and candidate not in TabuList):
            Best = candidate
            Best_f = Candidate_f
    
    return Best

def TabuSearch(DistanceMatrix, AmountNodes, MaxIterations=100, TabuSize=10,
               minErrorInten=0.001):
    """
    TabuSearch (function)
        Input: Distance Matrix (TSP instance), Total number of
        nodes, Max iterations for algorithm, size of
        tabu list, minimal error for intensification and amount
        of solutions for intensification.
        Output: Unique solutions (Best found).
        Description: Implementation of Tabu Search with 2-opt
        for TSP.
    """ 
    # Setting initial variables.
    BestSolution = first_solution(AmountNodes)
    CurrentSolution = BestSolution
    tabu_list = []

    # Until Max Iterations (Stop Criteria)
    for _ in range(MaxIterations):
        # Creating Neighborhood with 2-opt swaps.
        Neighborhood = get_neighbors_2opt(CurrentSolution)

        # Search the best neighbor.
        BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        # Intensification criteria: Minimal improvement.
        BestSolution_f = ObjFun(BestSolution, DistanceMatrix)
        CurrentSolution_f = ObjFun(BestNeighbor, DistanceMatrix)
        if abs(BestSolution_f - CurrentSolution_f) < minErrorInten:
            Neighborhood = get_neighbors_2opt(CurrentSolution)
            BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        # Diversification criteria: No improvement.
        if BestNeighbor is None:
            CurrentSolution = first_solution(AmountNodes)
            Neighborhood = get_neighbors_2opt(CurrentSolution)
            BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        # Adding to tabu list and change current solution.
        CurrentSolution = BestNeighbor
        tabu_list.append(CurrentSolution)
        if len(tabu_list) > TabuSize:
            tabu_list.pop(0)

        # Change best solution.
        if ObjFun(BestNeighbor, DistanceMatrix) < ObjFun(BestSolution, DistanceMatrix):
            BestSolution = BestNeighbor

    # Return the best solution found.
    return BestSolution


########## Functions Hill Climbing ##########

def get_neighbors(Solution, DesireNum):
    """
    get_neighbors (function)
        Input: Reference Solution and Desire number
        of neighbor.
        Output: List of neighbor solutions (neighborhood).
        Description: Generates a neighborhood 
        of solutions.
    """
    # Generates solutions.
    neighbors = []
    for _ in range(DesireNum):
        # Copy input's solution.
        neighbor = np.copy(Solution)
        # Swaping.
        i, j = np.random.choice(len(Solution), 2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        # Add neighbor.
        neighbors.append(neighbor.tolist())
    return neighbors

def TabuSearch_Sample(DistanceMatrix, AmountNodes, MaxIterations=100, TabuSize=10, numDesireSolution=50,
               minErrorInten = 0.001):
    """
    get_neighbors (function)
        Input: Distance Matrix (TSP instance), Total number of
        nodes, Max iterations for algorithm, size of
        tabu list, number of solutions to generate, minimal
        error for intensification and amount of solutions for
        intensification.
        Output: Unique solutions (Best found) and objective
        finction calls.
        Description: Implementation of Tabu Search
        metaheuristic.
    """ 
    # Setting initial variables.   
    BestSolution = first_solution(AmountNodes)
    CurrentSolution = BestSolution
    tabu_list = []

    # Until Max Iterations (Stop Criteria)
    for _ in range(MaxIterations):
        # Creating Neighborhood
        Neighborhood = get_neighbors(CurrentSolution, numDesireSolution)

        # Search the best neighbor.
        BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix, tabu_list)

        # Intensification criteria: Minimal improvement.
        BestSolution_f = ObjFun(BestSolution, DistanceMatrix)
        CurrentSolution_f = ObjFun(BestNeighbor, DistanceMatrix)
        if abs(BestSolution_f - CurrentSolution_f) < minErrorInten:
            Neighborhood = get_neighbors(BestSolution, numDesireSolution)
            BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix,tabu_list)

        # Diversification criteria: There is no improvement.
        if (BestNeighbor is None):
            CurrentSolution = first_solution(AmountNodes)
            Neighborhood = get_neighbors(CurrentSolution, numDesireSolution)
            BestNeighbor = best_neighbor(Neighborhood, DistanceMatrix,tabu_list)

        # Adding on tabu list and change current solution.
        CurrentSolution = BestNeighbor
        tabu_list.append(CurrentSolution)
        if (len(tabu_list) > TabuSize):
            tabu_list.pop(0)

        # Change best solution.
        if (ObjFun(BestNeighbor,DistanceMatrix) <
            ObjFun(BestSolution,DistanceMatrix)):
            BestSolution = BestNeighbor

    return BestSolution
