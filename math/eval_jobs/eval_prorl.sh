#!/bin/bash
#SBATCH --job-name=eval_prorl
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/eval_prorl_%j.log
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Evaluate ProRL (Nemotron) vs Base Model
# =============================================================================

mkdir -p logs

# # Activate metrics-rl environment
# source ~/.bashrc
# conda activate metrics-rl

cd /n/home07/itamarf/es-fine-tuning-paper/evolutionary-alignment/math

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# Fix: Use spawn method to avoid "Cannot re-initialize CUDA in forked subprocess"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Configuration
NUM_SAMPLES="${1:-256}"
MAX_TOKENS="${2:-2048}"

echo "============================================================"
echo "Evaluating: ProRL (Nemotron) vs Base"
echo "Samples: ${NUM_SAMPLES}, Max tokens: ${MAX_TOKENS}"
echo "Datasets: AIME2024, OlympiadBench (no overlap with ES training)"
echo "============================================================"

python math_eval.py \
    --model_path nvidia/Nemotron-Research-Reasoning-Qwen-1.5B \
    --model_revision v1 \
    --baseline_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets AIME2024,OlympiadBench \
    --num_samples ${NUM_SAMPLES} \
    --kmax ${NUM_SAMPLES} \
    --max_new_tokens ${MAX_TOKENS} \
    --temperature 0.6 \
    --top_p 0.95 \
    --output_dir ./eval_results \
    --output_json prorl_vs_base_results.json \
    --gpu_memory_utilization 0.9 \
    --seed 42

echo "ProRL evaluation complete!"
