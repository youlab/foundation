# SLURM Scripts
This directory contains scripts for running the code on the Duke Compute Cluster. Specific parameters such as array numbers, memory, and the bindings will need to be adjusted by the user.

## Working Scripts
* `run_applications.sh`: used to run `src/run_applications.py`, which includes all tasks in `src/applications/` apart from Karlsson dataset forecasting
* `run_data.sh`: used to run `src/run_data.py`, which preprocesses and splits data
* `run_figs.sh`: used to run `src/run_figs.py`, which runs figures used in the 1st version of the manuscript
* `run_forecast.sh`: used to run `src/run_forecast.py`, which runs analysis for Karlsson dataset forecasting tasks
* `run_model_comp.sh`: used to run `src/model_comparison.py`
* `run_train_ml.sh`: used to run `src/run_ml.py`, which runs model training
* `run_vae_posttrain_eval.sh`: used to run `run_vae_posttrain_eval.py`, which analyzes pretrained A7X models


## Defunct Scripts (in `defunct`)
* `run_app.sh`: used to run `src/app.py`
* `run_bert.sh`: used to run `src/train_simple_bert.py`
* `run_focus.sh`: used to run `src/focus_2.py`
* `run_normalize.sh`: used to run `src/normalize.py`
* `run_train_simple_pca.sh`: used to run `src/train_simple_pca.py`
* `run_train_simple.sh`: used to run `src/train_simple.py`
