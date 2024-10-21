#!/bin/bash

# Run this script on a terminal with conda available
# The first path is the location of the .yml file that contains the information of which libraries to install
# The second path is the path where you want to put the conda environment and its name
# Directories with spaces are a cause of problems for the environment, make sure the environment
# does not include spaces anywhere in its path.

conda env create --file "/mnt/d/Program Files/ISEL/MEIM/tese/envXAI.yml" --prefix "/mnt/d/ProgramFiles/envXAISetup"

# This command activates the environment so that modifications can be made
source activate "/mnt/d/ProgramFiles/envXAISetup"

read -p "Press any key to continue..."