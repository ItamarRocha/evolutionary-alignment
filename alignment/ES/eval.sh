#!/bin/bash
#SBATCH --job-name=es_alignment_eval
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/es_alignment_eval_%A_%a.log
#SBATCH --constraint=h100

# Evaluation script for ES alignment models
# Runs quadrant classification using the unified evaluation script

set -e

cd /n/home07/itamarf/es-fine-tuning-paper

export PYTHONPATH=/n/home07/itamarf/es-fine-tuning-paper:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Example usage - update paths as needed
# You can also use sweep_configs.yaml with the run_classify_reward_cost.sh script

python alignment/evaluation/classify_reward_cost_unified.py \
    --policy_model_path PKU-Alignment/alpaca-7b-reproduced \
    --model_name ES-Alignment \
    --output_dir alignment/outputs/model_comparisons \
    --bf16 \
    --policy_use_vllm
