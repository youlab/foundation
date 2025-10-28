#!/bin/bash
#SBATCH -e /hpc/dctrl/is178/foundation/slurm_outputs/run_figs/%A.err
#SBATCH -o /hpc/dctrl/is178/foundation/slurm_outputs/run_figs/%A.out
#SBATCH -p common
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00

# Create output directory if it doesn't exist
mkdir -p /hpc/dctrl/is178/foundation/slurm_outputs/run_figs

# Change to source directory
cd /hpc/dctrl/is178/foundation/src

# Run the figure generation script
apptainer exec --bind /hpc/group/youlab/zz294/VAE/saved_sims --bind /hpc/group/youlab/zah8/foundations --bind /work/zah8/ --bind /hpc/dctrl/is178 /hpc/group/youlab/you-lab-deep-learning.sif python run_figs.py

