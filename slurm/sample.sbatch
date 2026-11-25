#!/usr/bin/env bash
#SBATCH --job-name=treesample
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/%x-%j.out
 

#source .venv/bin/activate

mkdir -p "$SCRATCH/huggingface"
export HF_HOME=$SCRATCH/huggingface

# Print cache location for debugging
echo "Hugging Face cache: $HF_HUB_CACHE"

echo "Current directory: $PWD"

#python scripts/01_sample.py backend=vllm "$@"
#python scripts/02_parse.py "$@"
