#!/bin/bash
#SBATCH -e /hpc/group/youlab/xw262/foundation/slurm_outputs/run_karlsson_%A_%a.err
#SBATCH -o /hpc/group/youlab/xw262/foundation/slurm_outputs/run_karlsson_%A_%a.out

#SBATCH --partition=youlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

nvidia-smi

apptainer exec --nv --bind /hpc/group/youlab/zah8/foundations /hpc/group/youlab/you-lab-deep-learning.sif python src/run_forecast.py