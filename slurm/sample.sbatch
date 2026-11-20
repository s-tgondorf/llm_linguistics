#!/usr/bin/env bash
#SBATCH --job-name=treesample
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out

source .venv/bin/activate
export HF_HOME=${HF_HOME:-$PWD/hf_cache}
python scripts/01_sample.py backend=vllm "$@"
python scripts/02_parse.py "$@"
