#!/bin/bash
#SBATCH -e /hpc/group/youlab/xw262/foundations/slurm_outputs/run_script_%A_%a.err
#SBATCH -o /hpc/group/youlab/xw262/foundations/slurm_outputs/run_script_%A_%a.out
#SBATCH -p youlab-gpu
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclusive
python -m applications.super_resolution.main