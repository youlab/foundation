# Data
This directory contains scripts relevant to the data used to train and build the foundation model.

## Dirs
* `experimental`: this directory contains the experimental data used to train the model and for analysis in the paper
* `growth`: this directory has files for predicting growth parameters, not used in the paper
* `normalization_functions`: this directory contains functions used to preprocess the data
* `processed`: this directory contains files of processed data
* `simulation`: this directory contains the simulation data used to train the model and for analysis in the paper

## Files
* These files are used to preprocess the data prior to training
    * `normalize.py`: used to preprocess and normalize all of the data for training the models
    * `compile.py`: used to preprocess and compile all of the data for training the models, this is to be run after `normalize.py`
* `config.py`: contains parameters for preprocessing the consortia datasets
* `datasets.py`: obsolete, will be deleted



# Data
This directory contains experimental data used in the paper. Please reference the paper for specific information regarding the data.

The directory `processed` contains all of the data processed for downstream use.
