########## Libraries ##########
import numpy as np

########## Counter ##########
"""
    obj_fun_calls
        Counter of every time that
        the objective function is called.
"""
obj_fun_calls = 0

########## Functions TS ##########

def ObjFun(Solution, DistanceMatrix):
    """
    ObjFun (function)
        Input: Permutation vector and
        distance matrix.
        Output: Value in objective function.
        Description: Calculates the objetctive
        function using permutation vector and
        distance matrix.
        The TSP objective function is the sum
        of all edge's cost.
    """
    # Initialization of length of vector and
    # sum.
    tamVector = len(Solution)
    sum = 0

    # Sum connetions between nodes.
    for i in range(-1,tamVector-1):
        sum = sum + DistanceMatrix[Solution[i]-1][Solution[i+1]-1]
    
    # Count calls on this function.
    global obj_fun_calls
    obj_fun_calls += 1

    return sum
    
def first_solution(AmountNodes):
    """
    first_solution (function)
        Input: Number of nodes.
        Output: Permutation vector.
        Description: Generates first solution.
    """
    # Random permutation node.
    Vector = np.random.permutation(np.arange(1, AmountNodes+1))
    return Vector

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

def best_neighbor(Neighborhood, DistanceMatrix, TabuList):
    """
    get_neighbors (function)
        Input: Neighborhood, Distance Matrix and
        Tabu List.
        Output: neighbor  with best improvement 
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

def TabuSearch(DistanceMatrix, AmountNodes, MaxIterations=100, TabuSize=10, numDesireSolution=50,
               minErrorInten = 0.001, amountIntensification = 90):
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
        Reference: https://www.geeksforgeeks.org/what-is-tabu-search/
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
            Neighborhood = get_neighbors(BestSolution, amountIntensification)
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

    # takes the count of objective function
    # calls and return it.
    global obj_fun_calls

    return BestSolution, obj_fun_calls


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
    sum_obj = 0
    penaties_sum = 0

    # Calculates objective function.
    sum_obj = ObjFun(Solution, DistanceMatrix)

    # Calculate penalized objective function
    for i in range(-1, tamVector - 1):
        penaties_sum += influence_factor * penalties[Solution[i] - 1][Solution[i + 1] - 1]

    # Final value.
    penalized_sum = sum_obj + penaties_sum
    
    return penalized_sum

def best_neighbor_GLS(Neighborhood, DistanceMatrix, influence_factor, penalties):
    """
    get_neighbors (function)
        Input: Neighborhood and Distance Matrix.
        Output: neighbor  with best improvement 
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
        Candidate_f = penalty_function(candidate, DistanceMatrix, influence_factor, penalties)

        # Conditions for change.
        if (Best_f > Candidate_f):
            Best = candidate
            Best_f = Candidate_f
    
    return Best

def LS_bestimprovement(DistanceMatrix, AmountNodes, MaxIterations=100, numDesireSolution=50, 
                       influence_factor = 0.1, penalties = None):
    """
    LS_bestimprovement (function)
        Input: Distance Matrix (TSP instance), Total number of nodes, Max iterations,
        number of desire neighbors.
        Output: Best solution found and objective function calls.
        Description: Implementation of Local Search with Best Improvement for TSP.
    """
    # Generate an initial solution
    CurrentSolution = first_solution(AmountNodes)
    BestSolution = CurrentSolution
    BestSolution_f = penalty_function(BestSolution, DistanceMatrix, influence_factor, penalties)

    for _ in range(MaxIterations):
        # Generate neighbors
        Neighborhood = get_neighbors(CurrentSolution, numDesireSolution)

        # Gets best neighbor
        BestNeighbor = best_neighbor_GLS(Neighborhood, DistanceMatrix, influence_factor, penalties)
        BestNeighbor_f = penalty_function(BestNeighbor, DistanceMatrix, influence_factor, penalties)

        # Update the current solution
        CurrentSolution = BestNeighbor

        # Update the best solution found so far
        if (BestNeighbor_f < BestSolution_f):
            BestSolution = CurrentSolution

    return BestSolution

def update_penalties(Solution, penalties, DistanceMatrix):
    """
    update_penalties (function)
        Input: Solution, penalties, distance matrix, and 
        influence factor of penalizations.
        Output: Updated penalties.
        Description: Increases penalties for the most frequently used edges.
    """
    # Initialize elements.
    tamVector = len(Solution)
    utilities = []

    for i in range(-1, tamVector - 1):
        # Obtain utility.
        i_node = Solution[i] - 1
        j_node = Solution[i + 1] - 1
        cost = DistanceMatrix[i_node][j_node]
        penalty = penalties[i_node][j_node]
        utility = cost / (1 + penalty)

        # Update list.
        utilities.append(utility)

    # Obtain máx utility.
    max_utility = np.max(utilities)

    for i in range(-1, tamVector - 1):
        i_node = Solution[i] - 1
        j_node = Solution[i + 1] - 1

        if (utilities[i] == max_utility):
            penalties[i_node][j_node] += 1
            penalties[j_node][i_node] += 1  # TSP is symmetric

    return penalties

def GLS(DistanceMatrix, AmountNodes, MaxIterations=100, MaxIterationsLS=110, numDesireSolution=50, influence_factor = 0.1):
    """
    GLS (function)
        Input: Distance Matrix (TSP instance), Total number of nodes, Max iterations,
            Max iterations for LS, number of desired neighbor solutions, lambda parameter.
        Output: Best solution found and objective function calls.
        Description: Implementation of Guided Local Search for TSP.
    """
    # Generate an initial solution
    CurrentSolution = first_solution(AmountNodes)
    BestSolution = CurrentSolution
    BestSolution_f = ObjFun(BestSolution, DistanceMatrix)

    # Initialize penalty matrix
    penalties = np.zeros((AmountNodes, AmountNodes))

    for _ in range(MaxIterations):
        # Generate neighbors
        BestNeighbor = LS_bestimprovement(DistanceMatrix, AmountNodes, MaxIterationsLS, numDesireSolution, 
                       influence_factor, penalties)
        BestNeighbor_f = ObjFun(BestNeighbor, DistanceMatrix)

        # Update the current solution if an improvement is found
        if BestNeighbor_f < BestSolution_f:
            CurrentSolution = BestNeighbor
            BestSolution = CurrentSolution
            BestSolution_f = BestNeighbor_f

        # Update penalties using the guided local search mechanism
        penalties = update_penalties(CurrentSolution, penalties, DistanceMatrix)

    global obj_fun_calls
    return BestSolution, obj_fun_calls