#!/bin/bash -l

# Use the current working directory, which is the default setting

# Use the current enviroment for this job, which is the default
#SBATCH --export=ALL

# Define job name
#SBATCH -J ByteLevel_Training

# Define partition
#SBATCH -p gpu

# Request the number of node
#SBATCH -N 1

# Request the number of cores
#SBATCH -n 24

# Redirect output and error to the log file
#SBATCH -o ./Output_Files/%x_%j.out

#SBATCH --gres=gpu:4

#SBATCH -t 72:00:00

date

echo "This code is running on ${HOSTNAME}"

START_TIME=$(date +%s)


module load apps/anaconda3/2020.02-pytorch
module load libs/cuda/11.3
module load libs/cudnn/8.2.1_cuda11.3
source activate GPUtest3

python Training.py

END_TIME=$(date +%s)

ELAPSED_TIME=$((END_TIME - START_TIME))

HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Finished running"