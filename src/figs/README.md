# Figs
Use this directory to create the figures for the paper. Each sub-directory should have a main function that can be pulled into `main.py` to run. Use `main.py` to specify which figures to make. Run this file using `make_figs.py` in the `src` directory.

## Directories
* `antibiotics`: figures pertaining to antibiotic tasks
* `consortia`: figures pertaining to the simulated consortia tasks
* `consortia_exp`: figures pertaining to the experimental consortia tasks
* `data`: figures pertaining to the data collected for the paper
* `fig_utils`: utility functions for the figures
* `introduction`: figures pertaining to the introduction
* `model`: figures pertaining ot the model

## Files
* `main.py`: the primary script used to create figures. Modify the boolean variables here to run the corresponding script.
* `presentation.py`: figures configured specifically for use in a presentation
* `config.py`: contains some values for use throughout the figures
* `utils_torch.py`: contains the reconstruct function
* `utils.py`: contains utility functions for the figures