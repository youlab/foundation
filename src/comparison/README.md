# Compare Models
This directory contains scripts relevant to comparing different model architectures on different tasks

## Dirs
* `utils`: this directory has code for utility functions

## Files
* `config.py`: has configuration parameters
* These files are related to the antibiotics tasks
    * `summarize_antibiotics.py`: used to summarize all of the antibiotics tasks
    * `final_antibiotics_summary.py`: used to summarize the performance across the antibiotic tasks, takes inputs from `summarize_antibiotics.py` output
* These files are related to measuring the reconstruction accuracy
    * `reconstruction_pivot_table.py`: summarizes the results of the model reconstruction
    * `reconstruction.py`: runs reconstruction on all models on all datasets and is used for comparison of model performance
    * `summarize_reconstruction.py`: summarizes the results of all of the reconstructions
    * `reconstruct_individual_datasets.py`: used to reconstruct all of the individual datasets
