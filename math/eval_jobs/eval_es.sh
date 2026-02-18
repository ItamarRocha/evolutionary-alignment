#!/bin/bash
#SBATCH --job-name=eval_es
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/eval_es_%j.log
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Evaluate ES vs Base Model
# =============================================================================
#
# Supports both .pth checkpoints (from ES training) and HuggingFace format.
# Uses vLLM with WorkerExtension to load .pth weights directly.
#
# Usage:
#   sbatch eval_es.sh [num_samples] [max_tokens] [checkpoint_path]
#
#   # Using .pth checkpoint (default - iteration 800)
#   sbatch eval_es.sh 256 2048
#
#   # Using specific .pth checkpoint
#   sbatch eval_es.sh 256 2048 /path/to/tmp_iter_600.pth
#
#   # Using HuggingFace format checkpoint
#   sbatch eval_es.sh 256 2048 /path/to/hf_checkpoint_dir
#
# =============================================================================

mkdir -p logs

# # Activate metrics-rl environment (has vLLM with worker_extension_cls support)
# source ~/.bashrc
# conda activate metrics-rl

cd /n/home07/itamarf/es-fine-tuning-paper/evolutionary-alignment/math

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# Fix: Use spawn method to avoid "Cannot re-initialize CUDA in forked subprocess"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Fix: Add parent dir to PYTHONPATH so vLLM can find utils.worker_extn module
export PYTHONPATH="/n/home07/itamarf/es-fine-tuning-paper/evolutionary-alignment:${PYTHONPATH}"

# Configuration
NUM_SAMPLES="${1:-256}"
MAX_TOKENS="${2:-2048}"

# Default to latest .pth checkpoint
ES_RUN_DIR="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/math/es-ft-experiment/es_math_20260112_140425"
DEFAULT_CKPT="${ES_RUN_DIR}/model_saves/tmp_iter_800.pth"
ES_CHECKPOINT="${3:-${DEFAULT_CKPT}}"

# Base model (same one ES was trained from)
BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

echo "============================================================"
echo "Evaluating: ES vs Base"
echo "ES Checkpoint: ${ES_CHECKPOINT}"
echo "Base Model: ${BASE_MODEL}"
echo "Samples: ${NUM_SAMPLES}, Max tokens: ${MAX_TOKENS}"
echo "Datasets: AIME2024, OlympiadBench (no overlap with ES training)"
echo "============================================================"

# Check if checkpoint exists
if [ ! -e "${ES_CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${ES_CHECKPOINT}"
    echo ""
    echo "Available checkpoints:"
    ls -la ${ES_RUN_DIR}/model_saves/
    exit 1
fi

# Determine if this is a .pth file or HuggingFace directory
if [[ "${ES_CHECKPOINT}" == *.pth ]]; then
    echo "Using .pth checkpoint with vLLM WorkerExtension"
    python math_eval.py \
        --model_path ${BASE_MODEL} \
        --vllm_ckpt ${ES_CHECKPOINT} \
        --baseline_model ${BASE_MODEL} \
        --datasets AIME2024,OlympiadBench \
        --num_samples ${NUM_SAMPLES} \
        --kmax ${NUM_SAMPLES} \
        --max_new_tokens ${MAX_TOKENS} \
        --temperature 0.6 \
        --top_p 0.95 \
        --output_dir ./eval_results \
        --output_json es_vs_base_results.json \
        --gpu_memory_utilization 0.9 \
        --seed 42
else
    echo "Using HuggingFace format checkpoint"
    python math_eval.py \
        --model_path ${ES_CHECKPOINT} \
        --baseline_model ${BASE_MODEL} \
        --datasets AIME2024,OlympiadBench \
        --num_samples ${NUM_SAMPLES} \
        --kmax ${NUM_SAMPLES} \
        --max_new_tokens ${MAX_TOKENS} \
        --temperature 0.6 \
        --top_p 0.95 \
        --output_dir ./eval_results \
        --output_json es_vs_base_results.json \
        --gpu_memory_utilization 0.9 \
        --seed 42
fi

echo "ES evaluation complete!"
