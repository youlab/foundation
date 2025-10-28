#!/bin/bash
#SBATCH -e /hpc/home/zah8/foundation/slurm_outputs/run_model_comp/%A_%a.err
#SBATCH -o /hpc/home/zah8/foundation/slurm_outputs/run_model_comp/%A_%a.out
#SBATCH --array=0-40
#SBATCH -p youlab-gpu
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclusive
apptainer exec --nv /hpc/group/youlab/you-lab-deep-learning.sif python src/run_comparison.py
