########## Libraries ##########
import sys
import os
import json
import csv

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
Path_Params = 'Results/Parameters/best_TS_params.txt'
Path_OPT = "Optimals/Experimental/Optimals.txt"
output_directory = 'Results/Experimentals'

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from ReadTSP import ReadTsp # type: ignore
from ReadTSP import ReadTSP_optTour # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import TabuSearch  # type: ignore
from GLS import Guided_Local_Search # type: ignore

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

def write_list_to_txt(file_path, data_list):
    """
    Escribe los elementos de una lista en un archivo de texto,
    separando cada elemento por un salto de línea.
    
    Args:
        file_path (str): Ruta del archivo donde se guardarán los datos.
        data_list (list): Lista de datos a escribir en el archivo.
    """
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")

def normalized_error(opt_value, found_value):
    """
    Calcula el error normalizado entre el valor óptimo y el valor encontrado.
    
    Args:
        opt_value (float): Valor óptimo conocido.
        found_value (float): Valor de la solución encontrada.
    
    Returns:
        float: Error normalizado.
    """
    if opt_value == 0:  # Para evitar la división por cero
        return float('inf') if found_value != 0 else 0
    return abs(opt_value - found_value) / opt_value

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
results_file_path = os.path.join(output_directory, 'GLS_converge_38.csv')

# Using best parameters to obtain solutions.
n = len(Instances)
results = []
BestNeOf, BestSolOf = TabuSearch(Instances[0], len(Instances[0]), 
                    80000,
                    best_params['TabuSize'],
                    best_params['ErrorTolerance'])

for i in range(len(BestNeOf)):
    BestNeOf[i] = (BestNeOf[i]-Opt_Instances[0])/(Opt_Instances[0])
    BestSolOf[i] = (BestSolOf[i]-Opt_Instances[0])/(Opt_Instances[0])

with open(results_file_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Escribir encabezados
    csv_writer.writerow(['Mejor', 'Vecinos'])
    # Escribir datos
    for best_sol, best_ne in zip(BestSolOf, BestNeOf):
        csv_writer.writerow([best_sol, best_ne])
