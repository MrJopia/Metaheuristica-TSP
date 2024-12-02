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
"""
Path_Instances = "Instances/Parametrizacion"
Path_OPT = "Optimals/Parametrizacion/Optimals.txt"
output_directory = 'Results/Parameters'
best_TS_params_file = 'best_TS_params.txt'
best_GLS_params_file = 'best_GLS_params.txt'
trials_TS_file = 'trials_TS.csv'
trials_GLS_file = 'trials_GLS.csv'

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

def Parametrization_TS(trial, Instances, Opt_Instances):
    """
        Parametrization (Function)
            Input: trial (parameters for evaluation) and
            list of matrix (one each instance).
            Output: Best trial of parameters.
            Description: Do the parametrization prodecure with
            every trial.
    """
    # Define intervals.
    TabuSize = trial.suggest_int('TabuSize', 10, 100)
    minErrorInten = trial.suggest_float('ErrorTolerance', 1e-5, 1e-1, log=True)

    # Initializate total normalized error.
    total_normalized_error = 0
    num_instances = len(Instances)

    for i in range(num_instances):
        # Run Tabu Search with the parameters from Optuna
        best_n, best_sol = TabuSearch(Instances[i], len(Instances[i]), 80000, 
                                   TabuSize, minErrorInten)

        objective_value = min(best_sol)

        # Calculate normalized error
        normalized_error = abs(objective_value - Opt_Instances[i]) / Opt_Instances[i]

        # Sum the normalized error
        total_normalized_error += normalized_error

    # Return the average objective value across all instances
    return total_normalized_error / num_instances

def Parametrizartion_TS_capsule(Instances, Opt_Instances):
    """
        Parametrization_capsule (Function)
            Encapsulate other inputs from principal
            parametrization function.
    """
    return lambda trial: Parametrization_TS(trial, Instances, Opt_Instances)

def Parametrization_GLS(trial, Instances, Opt_Instances):
    """
        Parametrization (Function)
            Input: trial (parameters for evaluation) and
            list of matrix (one each instance).
            Output: Best trial of parameters.
            Description: Do the parametrization prodecure with
            every trial.
    """
    # Define intervals.
    alpha = trial.suggest_float('alpha', 0.125, 0.5, log=True)

    # Initializate total normalized error.
    total_normalized_error = 0
    num_instances = len(Instances)

    for i in range(num_instances):
        # Run Tabu Search with the parameters from Optuna
        best_n, best_sol = Guided_Local_Search(Instances[i], len(Instances[i]), 80000, 
                                   alpha)

        objective_value = min(best_sol)

        # Calculate normalized error
        normalized_error = abs(objective_value - Opt_Instances[i]) / Opt_Instances[i]

        # Sum the normalized error
        total_normalized_error += normalized_error

    # Return the average objective value across all instances
    return total_normalized_error / num_instances

def Parametrizartion_GLS_capsule(Instances, Opt_Instances):
    """
        Parametrization_capsule (Function)
            Encapsulate other inputs from principal
            parametrization function.
    """
    return lambda trial: Parametrization_GLS(trial, Instances, Opt_Instances)

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

    # Save the trials data in a CSV file
    with open(trials_path, 'w', newline='') as csvfile:
        fieldnames = ['trial_number', 'value', 'params']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for trial in study.trials:
            writer.writerow({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params
            })


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

# Parametrization Tabu search.
"""study = optuna.create_study(direction='minimize')
study.optimize(Parametrizartion_TS_capsule(Instances, Opt_Instances), n_trials=11, n_jobs=-1)
best_params = study.best_params
print('Best parameters:', best_params)
save_TS_study_txt(study, output_directory ,best_TS_params_file, trials_TS_file)"""

study = optuna.create_study(direction='minimize')
study.optimize(Parametrizartion_GLS_capsule(Instances, Opt_Instances), n_trials=11, n_jobs=-1)
best_params = study.best_params
print('Best parameters:', best_params)
save_TS_study_txt(study, output_directory ,best_GLS_params_file, trials_GLS_file)