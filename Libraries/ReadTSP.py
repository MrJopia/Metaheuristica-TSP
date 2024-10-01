########## Libraries ##########
import numpy as np

########## Functions ##########

def EuclidianDistance(node_A , node_B):
    """
    EuclidianDistance (function) 
        Input: Two points (coordenates x, y) that
        represents two nodes (locations) in TSP.
        Output: Euclidian distance.
        Description: Calculates euclidian distance
        between two coordenates.
    """
    return np.sqrt((node_B[0]-node_A[0])**2 
                       + (node_B[1]-node_A[1])**2)

def EuclideanDistanceMatrix(NodeList, Tam_Nodes):
    """
    EuclideanDistanceMatrix (function) 
        Input: List with coordenates of
        each point (location) and tam of the
        list.
        Output: Euclidean Distance matrix for all pairs.
        Description: Calculates Euclidean distance
        Matrix for each pairs of locations.
    """
    # Create zeros matrix.
    Matrix = np.zeros((Tam_Nodes,Tam_Nodes))

    # Vector x1 - Vector x2.
    for i in range(0, Tam_Nodes):
        for j in range(i+1, Tam_Nodes):
            Matrix[i][j] = (EuclidianDistance(NodeList[i], NodeList[j]))
            Matrix[j][i] = Matrix[i][j]

    return Matrix

def ReadTsp(filename):
    """
    ReadTsp (function)
        Input: File name.
        Output: Distance Matrix.
        Description: Read a TSP instance (.tsp file)
        and creates discance matrix.
    """
    # Input.
    infile = open(filename,'r')

    # Read instance first line (last element should be
    # filename).
    Name = infile.readline().strip().split()[-1]

    # Skip comments until DIMENSION.
    for line in infile:
        # Reading line.
        broken_line = line.strip().split()

        # Verification of DIMENSION.
        if(broken_line[0] == 'DIMENSION' or broken_line[0] == 'DIMENSION:'):
            Dimension = broken_line[-1]

        # Verification of EOF or Node coordenates secction.
        elif(broken_line[0] == 'EOF' or broken_line[0] == 'NODE_COORD_SECTION'):
            break

    # Take coordenates.
    nodelist = []
    int_dim = int(Dimension)
    for i in range(0, int_dim):
        x,y = infile.readline().strip().split()[1:]
        nodelist.append([float(x), float(y)])
    
    # Close input file.
    infile.close()
    
    # Create distance matrix.
    DistanceMatrix = EuclideanDistanceMatrix(nodelist,int_dim)
    print(Name)
    return DistanceMatrix

def ReadTSP_optTour(filename):
    """
    ReadTSP_optTour (function)
        Input: File name.
        Output: Optimal tour solution of tsp instance.
        Description: Read a TSP optimal tour solution
        found and make it a list.
    """
    # Input.
    infile = open(filename,'r')
    Dimension = 0

    # Read instance first line (last element should be
    # filename).
    Name = infile.readline().strip().split()[-1]
    print(Name)

    # Skip comments until DIMENSION.
    for line in infile:
        # Reading line.
        broken_line = line.strip().split()

        # Verification of DIMENSION.
        if(broken_line[0] == 'DIMENSION' or broken_line[0] == 'DIMENSION:'):
            Dimension = int(broken_line[-1])

        # Verification of EOF or Node coordenates secction.
        elif(broken_line[0] == 'EOF' or broken_line[0] == 'TOUR_SECTION'):
            break

    # Take route.
    opts = []
    for i in range(Dimension):
        element = infile.readline()
        opts.append(int(element))

    # Close file.
    infile.close()

    return opts