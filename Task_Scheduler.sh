#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH -J huggingface_token_test
#SBATCH -p nodes
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -o ./Output_Files/%x_%j.out



date

echo "HUGGINGFACE_HUB_TOKEN is set: ${HUGGINGFACE_HUB_TOKEN:0:8}..."

echo "This code is running on ${HOSTNAME}"

START_TIME=$(date +%s)

module load python-venv/1.0-gcc14.2.0
module load cuda/12.8.0-gcc14.2.0
source LLM-FT/bin/activate

python Inference.py

END_TIME=$(date +%s)

ELAPSED_TIME=$((END_TIME - START_TIME))

HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Finished running"