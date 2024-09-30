########## Libraries ##########
import sys
import os
import numpy as np
import optuna
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
obj_func_calls = 0
Path_Instances = "Instances/Small"
Path_OPT = "OptimalTours/Small"
output_directory = 'Results/Small(Parametrization)'
best_TS_params_file = 'best_TS_params.txt'
trials_TS_file = 'trials_TS_params.csv'
best_GLS_params_file = 'best_GLS_params.txt'
trials_GLS_file = 'trials_GLS_params.csv'

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

def Parametrization_TS(trial, Instances):
    """
        Parametrization (Function)
            Input: trial (parameters for evaluation) and
            list of matrix (one each instance).
            Output: Best trial of parameters.
            Description: Do the parametrization prodecure with
            every trial.
    """
    # Define intervals.
    MaxIterations = trial.suggest_int('MaxIterations', 50, 500)
    TabuSize = trial.suggest_int('TabuSize', 5, 50)
    numDesireSolution = trial.suggest_int('numDesireSolution', 10, 100)
    minErrorInten = trial.suggest_loguniform('ErrorTolerance', 1e-5, 1e-1)
    amountIntensification = trial.suggest_int('amountIntensification', 10, 100)

    # Initializate count of obj function calls.
    total_obj_value = 0
    num_instances = len(Instances)

    for i in range(num_instances):
        # Run Tabu Search with the parameters from Optuna
        best_solution, calls = TabuSearch(Instances[i], len(Instances[i]), MaxIterations, 
                                   TabuSize, numDesireSolution, minErrorInten, amountIntensification)
        
        # Call global variable and update calls count.
        global obj_func_calls
        obj_func_calls = calls

        # Evaluate the quality of the solution
        objective_value = ObjFun(best_solution, Instances[i])

        # Sum the objective values
        total_obj_value += objective_value

    # Return the average objective value across all instances
    return total_obj_value / num_instances

def Parametrizartion_TS_capsule(Instances):
    """
        Parametrization_capsule (Function)
            Encapsulate other inputs from principal
            parametrization function.
    """
    return lambda trial: Parametrization_TS(trial, Instances)

def Parametrization_GLS(trial, Instances):
    """
        Parametrization (Function)
            Input: trial (parameters for evaluation) and
            list of matrix (one each instance).
            Output: Best trial of parameters.
            Description: Do the parametrization prodecure with
            every trial.
    """
    MaxIterations = trial.suggest_int('MaxIterations', 50, 500)
    MaxIterationsLS = trial.suggest_int('MaxIterationsLS', 50, 200)
    influence_factor = trial.suggest_float('influence_factor', 0.01, 1.0)
    numDesireSolution = trial.suggest_int('numDesireSolution', 10, 100)

    # Initializate count of obj function calls.
    total_obj_value = 0
    num_instances = len(Instances)

    for i in range(num_instances):
        # Run Tabu Search with the parameters from Optuna
        best_solution, calls = GLS(Instances[i], len(Instances[i]), MaxIterations, 
                                   MaxIterationsLS, numDesireSolution, influence_factor)
        
        # Call global variable and update calls count.
        global obj_func_calls
        obj_func_calls = calls

        # Evaluate the quality of the solution
        objective_value = ObjFun(best_solution, Instances[i])

        # Sum the objective values
        total_obj_value += objective_value

    # Return the average objective value across all instances
    return total_obj_value / num_instances

def Parametrizartion_GLS_capsule(Instances):
    """
        Parametrization_capsule (Function)
            Encapsulate other inputs from principal
            parametrization function.
    """
    return lambda trial: Parametrization_GLS(trial, Instances)

def save_TS_study_txt(study, output_dir, best_params_file, trials_file):
    """
        save_TS_study_txt (function)
            Input: 
                - study: Optuna study object for Tabu Search.
                - output_dir: Directory where the files will be saved.
                - best_params_file: Filename for the best parameters.
                - trials_file: Filename for the trials data.
            Description:
                Saves the best parameters and trials of a Tabu Search Optuna study
                in the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_params_path = os.path.join(output_dir, best_params_file)
    trials_path = os.path.join(output_dir, trials_file)

    # Save the best parameters in a readable text file (JSON format)
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    # Save the entire study's trials as CSV for detailed analysis
    with open(trials_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trial Number", "Value", "MaxIterations", "TabuSize", "numDesireSolution", "ErrorTolerance", "Average Objective Value"])
        for trial in study.trials:
            avg_obj_value = trial.value / len(Instances)  # Assuming each trial evaluates all instances
            writer.writerow([trial.number, trial.value, 
                             trial.params.get('MaxIterations', None), 
                             trial.params.get('TabuSize', None), 
                             trial.params.get('numDesireSolution', None), 
                             trial.params.get('ErrorTolerance', None),
                             avg_obj_value])  # Add average objective value
            
def save_GLS_study_txt(study, output_dir, best_params_file, trials_file):
    """
        save_GLS_study_txt (function)
            Input: 
                - study: Optuna study object for Guided Local Search.
                - output_dir: Directory where the files will be saved.
                - best_params_file: Filename for the best parameters.
                - trials_file: Filename for the trials data.
            Description:
                Saves the best parameters and trials of a Guided Local Search Optuna study
                in the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_params_path = os.path.join(output_dir, best_params_file)
    trials_path = os.path.join(output_dir, trials_file)

    # Save the best parameters in a readable text file (JSON format)
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    # Save the entire study's trials as CSV for detailed analysis
    with open(trials_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Trial Number", "Value", "MaxIterations", "MaxIterationsLS", "Influence Factor", "numDesireSolution", "Average Objective Value"])
        for trial in study.trials:
            avg_obj_value = trial.value / len(Instances)  # Assuming each trial evaluates all instances
            writer.writerow([trial.number, trial.value, 
                             trial.params.get('MaxIterations', None), 
                             trial.params.get('MaxIterationsLS', None), 
                             trial.params.get('influence_factor', None), 
                             trial.params.get('numDesireSolution', None),
                             avg_obj_value])  # Add average objective value
            

########## Procedure ##########

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

# Obtain TSP Instances and optimal tour
# corresponding to each one.
Instances, Opt_Instances = Read_Content(files_Instances,files_OPT)

# Parametrization Tabu search.
study = optuna.create_study(direction='minimize')
study.optimize(Parametrizartion_TS_capsule(Instances), n_trials=5)
best_params = study.best_params
print('Best parameters:', best_params)
save_TS_study_txt(study, output_directory ,best_TS_params_file, trials_TS_file)

# Parametrization Guided local search.
study = optuna.create_study(direction='minimize')
study.optimize(Parametrizartion_GLS_capsule(Instances), n_trials=5)
best_params = study.best_params
print('Best parameters:', best_params)
save_GLS_study_txt(study, output_directory, best_GLS_params_file, trials_GLS_file)