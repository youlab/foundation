#!/bin/bash
#SBATCH -e /hpc/home/zah8/foundations/slurm_outputs/run_train_simple/%A_%a.err
#SBATCH -o /hpc/home/zah8/foundations/slurm_outputs/run_train_simple/%A_%a.out
#SBATCH --array=0-53
#SBATCH -p youlab-gpu
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclusive
apptainer exec --nv --bind /work/zah8/ /hpc/group/youlab/you-lab-deep-learning.sif python src/train_simple.py
