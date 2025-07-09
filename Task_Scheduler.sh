#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH -J LLM-8bitQ-Inference_test
#SBATCH -p gpu-140s-low
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu45
#SBATCH -t 1:00:00
#SBATCH -o ./Output_Files/%x_%j.out


date

echo "This code is running on ${HOSTNAME}"

START_TIME=$(date +%s)

module load python-venv/1.0-gcc14.2.0
module load cuda/12.8.0-gcc14.2.0
source LLM-FT/bin/activate

python Training.py

END_TIME=$(date +%s)

ELAPSED_TIME=$((END_TIME - START_TIME))

HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Finished running"