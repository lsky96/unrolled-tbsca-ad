"""
Author: 
Lukas Schynol
lschynol@nt.tu-darmstadt.de
"""
import os
import torch

DEBUG = False
CPU_THREAD_LIMIT = None
FP_DTYPE = torch.double  # torch.double (double recommended)
PREC_EPS = 1
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}".format(DEVICE))

DFILEEXT = ".pt"  # file extension for pickled results
SCENARIODIR = os.path.abspath("scenario_data_tensor")  # dir containing pickled data sets
RESULTDIR = os.path.abspath("rdat_tens")  # dir containing pickled results
EXPORTDIR = os.path.abspath("exp_tens")  # dir containing exported results in csv
RW_ABILENE_DIR = os.path.abspath("abilene")  # dir for raw abilene flow data