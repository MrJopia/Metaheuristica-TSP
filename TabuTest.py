########## Libraries ##########
import sys
import os
import numpy as np
import json
import random
import pandas as pd

########## Globals ##########
"""
    Path_Instances 
        Path of TSP instances in your
    max_calls_obj_func (global variable)
        Minimum of calls for end parametrization.
    obj_func_calls (global variable)
        Counter of every time that
        the objective function is called.
"""
obj_func_calls = 0
Path_Instances = "Instances/Small"
#Path_Instances = "Instances/Medium"
#Path_Instances = "Instances/Big"
Path_OPT = "OptimalTours/Small"
output_directory = 'Results/Small(Parametrization)'
#output_directory_1 = 'Results/Medium'
#output_directory_2 = 'Results/Big'
best_TS_params_file = 'best_TS_params.txt'

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from ReadTSP import ReadTsp # type: ignore
from ReadTSP import ReadTSP_optTour # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import TabuSearch  # type: ignore
from TabuSearch import GLS  # type: ignore

########## Secundary functions ##########

def Read_Content(filenames_Ins, filenames_Opt):
    """
        Read_Content (function)
            Input: File names of TSP instances and
            optimal tour associated to TSP instances.
            Output: Distance Matrix and optimal solution
            permutation vector.
            Description: Read content.
    """
    # Instances and OPT_tour.
    Instances = []
    OPT_Instances = []

    # Reading files.
    for file in filenames_Ins:
        Instances.append(ReadTsp(file))

    for file in filenames_Opt:
        OPT_Instances.append(ReadTSP_optTour(file))

    return Instances, OPT_Instances

def load_best_params(file_path):
    """
    Carga los mejores parámetros desde un archivo JSON.
    
    Args:
        file_path (str): Ruta del archivo JSON que contiene los mejores parámetros.
    
    Returns:
        dict: Diccionario con los mejores parámetros cargados.
    """
    try:
        with open(file_path, 'r') as f:
            best_params = json.load(f)
        return best_params
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no existe.")
        return None
    except json.JSONDecodeError:
        print(f"Error: No se pudo decodificar el archivo {file_path}.")
        return None

def compare_opt_r(result, optimaltour, DistanceMatrix):
    # Obtain values.
    result_f = ObjFun(result, DistanceMatrix)
    optimaltour_f = ObjFun(optimaltour, DistanceMatrix)

    # Error.
    err = (optimaltour_f-result_f)/optimaltour_f

    return err


# Obtain small TSP's instances.
Content_Instances = os.listdir(Path_Instances)
files_Instances = []
for file in Content_Instances:
    if(os.path.isfile(os.path.join(Path_Instances,file))):
        files_Instances.append(Path_Instances+"/"+file)

# Obtain optimal tours small TSP's instances.
Content_OPT = os.listdir(Path_OPT)
files_OPT = []
for file in Content_OPT:
    if(os.path.isfile(os.path.join(Path_OPT,file))):
        files_OPT.append(Path_OPT+"/"+file)

# Obtain data.
Instances, OptInstances = Read_Content(files_Instances, files_OPT)

# Obtain best parameters for optimization.
params = load_best_params(output_directory+'/'+best_TS_params_file)
print(params)

# Using best parameters to obtain solutions.
random.seed(80)
n = len(Instances)
list_results = []
list_results_f = []
list_results_c = []
list_results_err = []

# Small instances test
for j in range(n):
    for _ in range(11):
        result, calls = TabuSearch(Instances[j], len(Instances[j]), 
                            params['MaxIterations'], 
                            params['TabuSize'], 
                            params['numDesireSolution'], 
                            params['ErrorTolerance'], 
                            params['amountIntensification'])
        
        list_results.append(result)
        value = ObjFun(result,Instances[j])
        list_results_f.append(value)
        list_results_c.append(calls)
        list_results_err.append(compare_opt_r(result, OptInstances[j], Instances[j]))


# Definir nombres de columnas para el CSV
columnas = ['Results', 'ObjectiveFunction', 'Error', 'Calls']

# Crear un DataFrame con los resultados
df = pd.DataFrame({
    'Results': list_results,
    'ObjectiveFunction': list_results_f,
    'Error': list_results_err,
    'Calls': list_results_c
})

# Guardar el DataFrame en un archivo CSV
df.to_csv(output_directory + '/tsp_Small_results.csv', index=False)
    