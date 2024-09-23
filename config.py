"""
Author: 
Lukas Schynol
lschynol@nt.tu-darmstadt.de
"""
import os
import torch

DEBUG = False
CPU_THREAD_LIMIT = None
FP_DTYPE = torch.double  # torch.double (double recommended, otherwise errors due to precision can easily occur)
PREC_EPS = 1
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}".format(DEVICE))
DFILEEXT = ".pt"  # file extension for pickled results

SCENARIODIR = os.path.abspath("scenario_data_tensor")  # dir containing pickled data sets
RESULTDIR = os.path.abspath("rdat_tens")  # dir containing pickled results
EXPORTDIR = os.path.abspath("exp_tens")  # dir containing exported results in csv

# for Codeocean
# SCENARIODIR_PRECOMP = os.path.join("..", "data", "datasets_precomp")  # dir containing pickled data sets
# RESULTDIR_PRECOMP = os.path.join("..", "data", "results_precomp")  # dir containing pickled results
# EXPORTDIR_PRECOMP = os.path.join("..", "data", "export_precomp")  # dir containing exported results in csv

# SCENARIODIR = os.path.join("..", "results", "datasets")  # dir containing pickled data sets
# RESULTDIR = os.path.join("..", "results", "results")  # dir containing pickled results
# EXPORTDIR = os.path.join("..", "results", "export")  # dir containing exported results in csv

# RW_ABILENE_DIR = os.path.abspath("abilene")  # dir for raw abilene flow data
RW_ABILENE_ROUTINGTABLE_PATH = os.path.join("abilene", "A")
RW_ABILENE_FLOW_PATH = os.path.join("abilene", "X{:02d}.gz")  # format placeholder, weeks are numbered from 01 to 24