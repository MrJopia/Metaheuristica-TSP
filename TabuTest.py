########## Libraries ##########
import sys
import os
import numpy as np
import json
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
Path_Instances = "Instances/Experimental"
Path_Params = 'Results/Parameters/best_TSS_params.txt'
Path_OPT = "Optimals/Experimental/Optimals.txt"
output_directory = 'Results/Experimentals'

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from ReadTSP import ReadTsp # type: ignore
from ReadTSP import ReadTSP_optTour # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import TabuSearch  # type: ignore
from TabuSearch import TabuSearch_Sample  # type: ignore

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

    # Reading files.
    for file in filenames_Ins:
        Instances.append(ReadTsp(file))

    OPT_Instances = ReadTSP_optTour(filenames_Opt)

    return Instances, OPT_Instances

def load_best_params(file_path):
    """
    load_best_params (function)
        Input: File path to the best parameters.
        Output: Dictionary with best parameters.
        Description: Reads the best parameters from a JSON or text file.
    """
    with open(file_path, 'r') as file:
        params = json.load(file)
    return params

def write_results(file_path, results):
    """
    write_results (function)
        Input: File path to save the results and a list of results.
        Output: None.
        Description: Writes the results and corresponding errors to a text file.
    """
    with open(file_path, 'w') as file:
        for result in results:
            # Descomponer el resultado en el valor de la función objetivo y el error
            obj_value, error = result
            file.write(f"Objective Value: {obj_value}, Error: {error}\n")

########## Procedure ##########

# Obtain TSP's instances.
Content_Instances = os.listdir(Path_Instances)
files_Instances = []
for file in Content_Instances:
    if(os.path.isfile(os.path.join(Path_Instances,file))):
        files_Instances.append(Path_Instances+"/"+file)

# Obtain TSP Instances and optimal tour
# corresponding to each one.
Instances, Opt_Instances = Read_Content(files_Instances, Path_OPT)

# Params.
best_params = load_best_params(Path_Params)
results_file_path = os.path.join(output_directory, 'tabu_search_results.txt')

# Using best parameters to obtain solutions.
n = len(Instances)
results = []
for Instance, opt_value in zip(Instances, Opt_Instances):
    for i in range(11):
        # Llamar a GLS (o TabuSearch) utilizando los mejores parámetros cargados.
        result = TabuSearch(Instance, len(Instance), 
                     MaxIterations=400,
                     minErrorInten=best_params["ErrorTolerance"])
        
        # Calcular el valor de la función objetivo para la solución obtenida
        obj_value = ObjFun(result, Instance)

        # Calcular el error respecto al valor óptimo
        error = (obj_value - opt_value) / opt_value
        
        # Guardar el resultado y el error
        results.append((obj_value, error))
        
        # Imprimir el valor de la función objetivo para la solución obtenida.
        print(f"Objective Value: {obj_value}, Error: {error}")

# Escribir los resultados en un archivo
write_results(results_file_path, results)