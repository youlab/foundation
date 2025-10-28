#!/bin/bash
#SBATCH -e /hpc/home/zah8/foundation/slurm_outputs/run_normalize_%A.err
#SBATCH -o /hpc/home/zah8/foundation/slurm_outputs/run_normalize_%A.out
#SBATCH -p youlab-gpu
#SBATCH --mem=16000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
apptainer exec --nv /hpc/group/youlab/you-lab-deep-learning.sif python src/normalize.py
