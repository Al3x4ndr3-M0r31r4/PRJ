#!/bin/bash

# Since the version of installation of SHAP, for some reason, is not correctly installed,
# these alterations must be made.
# First, activate the environment created by the first script
source activate "/mnt/d/ProgramFiles/envXAISetup"

# Then uninstall the SHAP version that was installed via pip
pip uninstall -y shap

# After pip uninstall, the environment is activated again
source activate "/mnt/d/ProgramFiles/envXAISetup"

# The correct installation of SHAP, using conda, is made
conda install -y -c conda-forge shap==0.42.1

read -p "Press any key to continue..."