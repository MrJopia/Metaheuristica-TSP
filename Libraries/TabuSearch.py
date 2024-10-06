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


########## Functions GLS ##########

def penalty_function(Solution, DistanceMatrix, influence_factor, penalties):
    """
    penalty_function (function)
        Input: Permutation vector (solution), Distance matrix, 
        influence factor of penalizations matrix and penalties.
        Output: Penalized objective function value.
        Description: Applies penalties to the solution based on visited edges.
    """
    tamVector = len(Solution)
    sum_obj = ObjFun(Solution, DistanceMatrix)
    penalties_sum = 0

    # Calculate penalized objective function
    for i in range(-1, tamVector - 1):
        penalties_sum += influence_factor * penalties[Solution[i] - 1][Solution[i + 1] - 1]

    # Final value
    penalized_sum = sum_obj + penalties_sum
    
    return penalized_sum

def best_neighbor_GLS(Neighborhood, DistanceMatrix, influence_factor, penalties):
    """
    best_neighbor_GLS (function)
        Input: Neighborhood and Distance Matrix.
        Output: Neighbor with best improvement in objective function.
        Description: Evaluates Neighborhood of permutation solutions.
    """
    # Best solution and value in obj. function.
    Best = None
    Best_f = float('inf')

    # Searching.
    for candidate in Neighborhood:
        Candidate_f = penalty_function(candidate, DistanceMatrix, influence_factor, penalties)

        # Conditions for change.
        if (Best_f > Candidate_f):
            Best = candidate
            Best_f = Candidate_f
    
    return Best

def LS_bestimprovement(DistanceMatrix, AmountNodes, MaxIterations=100, 
                       influence_factor=0.1, penalties=None):
    """
    LS_bestimprovement (function)
        Input: Distance Matrix (TSP instance), Total number of nodes, Max iterations,
        number of desired neighbors, influence factor and penalties matrix.
        Output: Best solution found.
        Description: Local Search using Best Improvement and 2-opt swaps.
    """
    # Generate an initial solution
    CurrentSolution = first_solution(AmountNodes)
    BestSolution = CurrentSolution
    BestSolution_f = penalty_function(BestSolution, DistanceMatrix, influence_factor, penalties)

    for _ in range(MaxIterations):
        # Generate neighbors using 2-opt swaps
        Neighborhood = get_neighbors_2opt(CurrentSolution)

        # Gets best neighbor
        BestNeighbor = best_neighbor_GLS(Neighborhood, DistanceMatrix, influence_factor, penalties)
        BestNeighbor_f = penalty_function(BestNeighbor, DistanceMatrix, influence_factor, penalties)

        # Update the current solution
        CurrentSolution = BestNeighbor

        # Update the best solution found so far
        if BestNeighbor_f < BestSolution_f:
            BestSolution = CurrentSolution
            BestSolution_f = BestNeighbor_f

    return BestSolution

def update_penalties(Solution, penalties, DistanceMatrix):
    """
    update_penalties (function)
        Input: Solution, penalties, distance matrix.
        Output: Updated penalties.
        Description: Increases penalties for the most frequently used edges.
    """
    tamVector = len(Solution)
    utilities = []

    for i in range(-1, tamVector - 1):
        i_node = Solution[i] - 1
        j_node = Solution[i + 1] - 1
        cost = DistanceMatrix[i_node][j_node]
        penalty = penalties[i_node][j_node]
        utility = cost / (1 + penalty)
        utilities.append(utility)

    max_utility = np.max(utilities)

    for i in range(-1, tamVector - 1):
        i_node = Solution[i] - 1
        j_node = Solution[i + 1] - 1

        if utilities[i] == max_utility:
            penalties[i_node][j_node] += 1
            penalties[j_node][i_node] += 1  # TSP is symmetric

    return penalties

def GLS(DistanceMatrix, AmountNodes, MaxIterations=100, MaxIterationsLS=10, influence_factor=0.1):
    """
    GLS (function)
        Input: Distance Matrix (TSP instance), Total number of nodes, Max iterations,
        Max iterations for LS, number of desired neighbors, influence factor.
        Output: Best solution found.
        Description: Implementation of Guided Local Search for TSP with 2-opt.
    """
    # Generate an initial solution
    CurrentSolution = first_solution(AmountNodes)
    BestSolution = CurrentSolution
    BestSolution_f = ObjFun(BestSolution, DistanceMatrix)

    # Initialize penalty matrix
    penalties = np.zeros((AmountNodes, AmountNodes))

    for _ in range(MaxIterations):
        # Use LS_bestimprovement with 2-opt
        BestNeighbor = LS_bestimprovement(DistanceMatrix, AmountNodes, MaxIterationsLS, 
                                          influence_factor, penalties)
        BestNeighbor_f = ObjFun(BestNeighbor, DistanceMatrix)

        # Update the current solution if an improvement is found
        if BestNeighbor_f < BestSolution_f:
            CurrentSolution = BestNeighbor
            BestSolution = CurrentSolution
            BestSolution_f = BestNeighbor_f

        # Update penalties using the guided local search mechanism
        penalties = update_penalties(CurrentSolution, penalties, DistanceMatrix)

    return BestSolution