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
Path_Instances = "Instances/Parametrizacion"
#Path_Instances = "Instances/Medium"
#Path_Instances = "Instances/Big"
#Path_OPT = "Optimals/Parametrizacion"
output_directory = 'Results/Small(Parametrization)'
#output_directory_1 = 'Results/Medium'
#output_directory_2 = 'Results/Big'
best_TS_params_file = 'best_TS_params_51.txt'

########## Own files ##########
# Path from the workspace.
sys.path.append(os.path.join(os.path.dirname(__file__), 'Libraries'))
from ReadTSP import ReadTsp # type: ignore
from ReadTSP import ReadTSP_optTour # type: ignore
from TabuSearch import ObjFun  # type: ignore
from TabuSearch import TabuSearch  # type: ignore
from TabuSearch import GLS  # type: ignore

########## Secundary functions ##########

########## Procedure ##########
# Obtain small TSP's instances.
Content_Instances = os.listdir(Path_Instances)
files_Instances = []
for file in Content_Instances:
    if(os.path.isfile(os.path.join(Path_Instances,file))):
        files_Instances.append(Path_Instances+"/"+file)

Instances = []
for file in files_Instances:
    Instance = ReadTsp(file)
    Instances.append(Instance)

# Using best parameters to obtain solutions.
random.seed(80)
n = len(Instances)

for Instance in Instances:
    for i in range(5):
        result = GLS(Instance, len(Instance), MaxIterations=10, MaxIterationsLS=30, influence_factor=0.1)
        print(ObjFun(result,Instance))