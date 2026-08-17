#!/bin/bash
#SBATCH -e /hpc/group/youlab/xw262/foundation/slurm_outputs/run_data_%A_%a.err
#SBATCH -o /hpc/group/youlab/xw262/foundation/slurm_outputs/run_data_%A_%a.out

#SBATCH -p youlab-gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

nvidia-smi

apptainer exec --nv --bind /hpc/group/youlab/zah8/foundations /hpc/group/youlab/you-lab-deep-learning.sif python src/run_data.py
