#!/bin/bash
#SBATCH -e /hpc/home/zah8/foundations/slurm_outputs/focus/%A_%a.err
#SBATCH -o /hpc/home/zah8/foundations/slurm_outputs/focus/%A_%a.out
#SBATCH --array=0-7
#SBATCH -p youlab-gpu
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
apptainer exec --nv --bind /hpc/group/youlab/zz294/VAE/saved_sims --bind /hpc/group/youlab/zah8/foundations --bind /work/zah8/ /hpc/group/youlab/you-lab-deep-learning.sif python src/focus2.py
