#!/bin/bash
#SBATCH -e /hpc/group/youlab/xw262/foundation/slurm_outputs/run_train_ml_%A_%a.err
#SBATCH -o /hpc/group/youlab/xw262/foundation/slurm_outputs/run_train_ml_%A_%a.out
#SBATCH --array=0-0
#SBATCH -p youlab-gpu
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclusive

apptainer exec --nv --bind /hpc/group/youlab/zah8/foundations /hpc/group/youlab/you-lab-deep-learning.sif python src/run_ml.py A7X 8 all
