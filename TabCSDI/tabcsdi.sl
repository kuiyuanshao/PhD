#!/bin/bash -e
#SBATCH --job-name=GPUJob   # job name (shows up in the queue)
#SBATCH --gpus-per-node A100:1
#SBATCH --time=00-12:00:00  # Walltime (DD-HH:MM:SS)
#SBATCH --cpus-per-task=10   # number of CPUs per task (1 by default)
#SBATCH --mem=12G         # amount of memory per node (1 by default)

module load Python/3.10.5-gimkl-2022a cuDNN/8.6.0.163-CUDA-11.8.0

echo "Executing Python ..."

python exe_me.py

echo "Python finished."