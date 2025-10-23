# Data
This directory contains scripts and data for the simulated antibiotic data

## Files
* `_antibiotic_model.py`: used to generate simulations
* others are used to supply the parameters and results from the simulations


# Data
This directory contains scripts and data for the simulated chaotic consortia data. Based on the model from this paper: https://www.science.org/doi/10.1126/science.abm7841.

## Files
* `_chaotic_model.py`: used to generate simulations
* others are used to supply the parameters and results from the simulations

# Data
This directory contains scripts and data for the simulated modified logistic equation data

## Files
* `_lingchong_eqn_model.py`: used to generate simulations
* `growth_datasets.py`: used for application around predicting growth rate, not use in the paper
* others are used to supply the parameters and results from the simulations


# Data
This directory contains scripts relevant to the data used to train and build the foundation model.

## Dirs
* `antibiotic`: this directory contains code for the simulated antibiotic data
* `chaotic`: this directory contains code for the simulated chaotic consortia
* `lingchong_eqn`: this directory contains code for the modified logistic equation
* `processed`: this directory contains data processed for use in the downstream models

## Files
* `approved_files.sh`: used to copy approved data files to the `processed` directory. This will need to be updated for a given user.
