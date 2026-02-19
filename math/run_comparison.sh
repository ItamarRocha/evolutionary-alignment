#!/bin/bash
#SBATCH --job-name=math_coverage_comparison
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/math_comparison_%A.log
#SBATCH --mail-type=ALL

# =============================================================================
# ES vs ProRL vs Base Model Comparison - Invisible Leash Methodology
# =============================================================================
#
# This script runs a comprehensive comparison between:
#   1. Base Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#   2. ProRL (GRPO): nvidia/Nemotron-Research-Reasoning-Qwen-1.5B
#   3. ES: Your trained checkpoints
#
# Following the methodology from "The Invisible Leash" paper (Wu et al., 2025).
#
# Key metrics computed:
#   - Pass@k curves (k=1,2,4,8,16,32,64,128,256,512,1024)
#   - Empirical-Support Categorization (Preservation, Shrinkage, Expansion)
#   - Answer-level entropy
#   - Coverage metrics (P, E, S, O)
#   - Comparison metrics (SRR, NDR, SDS, NSCR)
#
# Usage:
#   sbatch run_comparison.sh
#   sbatch run_comparison.sh 256  # Quick test with fewer samples
#
# =============================================================================

set -e

# Create logs directory
mkdir -p logs

# Load modules
module load python/3.12.11-fasrc02
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate environment
mamba deactivate
mamba activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============================================================================
# Configuration
# =============================================================================

# Number of samples (paper uses 8192 for math, 1024 is reasonable for faster runs)
NUM_SAMPLES="${1:-1024}"

# Models to compare
BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PRORL_MODEL="nvidia/Nemotron-Research-Reasoning-Qwen-1.5B"
ES_CHECKPOINT="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/math/es-ft-experiment/es_math_20260112_140425/model_saves/tmp_iter_800.pth"

# Invisible Leash paper hyperparameters
TEMPERATURE=0.6
TOP_P=0.95
MAX_NEW_TOKENS=32000

# Datasets
DATASETS="MATH500"  # Start with one dataset, expand to all for full eval

# Output directory
OUTPUT_DIR="./eval_results/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}

echo "============================================================"
echo "MATH COVERAGE COMPARISON - INVISIBLE LEASH METHODOLOGY"
echo "============================================================"
echo ""
echo "Models:"
echo "  Base:  ${BASE_MODEL}"
echo "  ProRL: ${PRORL_MODEL}"
echo "  ES:    ${ES_CHECKPOINT}"
echo ""
echo "Hyperparameters (matching IL paper):"
echo "  Temperature: ${TEMPERATURE}"
echo "  Top-p: ${TOP_P}"
echo "  Max tokens: ${MAX_NEW_TOKENS}"
echo "  Num samples: ${NUM_SAMPLES}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Evaluate Base Model
# =============================================================================
echo "============================================================"
echo "STEP 1: Evaluating Base Model"
echo "============================================================"

python math_eval.py \
    --model_path ${BASE_MODEL} \
    --datasets ${DATASETS} \
    --num_samples ${NUM_SAMPLES} \
    --kmax ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --output_dir ${OUTPUT_DIR} \
    --output_json base_model_results.json \
    --gpu_memory_utilization 0.9 \
    --seed 42

echo "Base model evaluation complete."
echo ""

# =============================================================================
# Step 2: Evaluate ProRL (GRPO) with comparison to Base
# =============================================================================
echo "============================================================"
echo "STEP 2: Evaluating ProRL (Nemotron) vs Base"
echo "============================================================"

python math_eval.py \
    --model_path ${PRORL_MODEL} \
    --model_revision v1 \
    --baseline_model ${BASE_MODEL} \
    --datasets ${DATASETS} \
    --num_samples ${NUM_SAMPLES} \
    --kmax ${NUM_SAMPLES} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --output_dir ${OUTPUT_DIR} \
    --output_json prorl_vs_base_results.json \
    --gpu_memory_utilization 0.9 \
    --seed 42

echo "ProRL evaluation complete."
echo ""

# =============================================================================
# Step 3: Evaluate ES with comparison to Base
# =============================================================================
echo "============================================================"
echo "STEP 3: Evaluating ES vs Base"
echo "============================================================"

# If a .pth checkpoint exists, note that reconstruction may still be needed in some workflows.
if [ -f "${ES_CHECKPOINT}" ]; then
    echo "Note: ES checkpoint .pth found at ${ES_CHECKPOINT}."
    echo "If needed, reconstruct with: python ES/reconstruct_checkpoint.py --replay_log_dir <path> --target_iteration 800"
    echo ""
fi

ES_HF_CKPT="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/math/es-ft-experiment/es_math_20260112_140425/model_saves/checkpoint_iter_800"
if [ -d "${ES_HF_CKPT}" ]; then
    python math_eval.py \
        --model_path ${ES_HF_CKPT} \
        --baseline_model ${BASE_MODEL} \
        --datasets ${DATASETS} \
        --num_samples ${NUM_SAMPLES} \
        --kmax ${NUM_SAMPLES} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --output_dir ${OUTPUT_DIR} \
        --output_json es_vs_base_results.json \
        --gpu_memory_utilization 0.9 \
        --seed 42
    echo "ES evaluation complete."
else
    echo "ES checkpoint not found at ${ES_HF_CKPT}"
    echo "Please provide a valid HuggingFace format checkpoint."
fi

echo ""
echo "============================================================"
echo "COMPARISON COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Key files:"
echo "  - base_model_results.json"
echo "  - prorl_vs_base_results.json"
echo "  - es_vs_base_results.json (if ES checkpoint was available)"
echo ""
echo "To analyze results, look for:"
echo "  - pass_at_k: Compare curves between models"
echo "  - support_categorization: Preservation/Shrinkage/Expansion counts"
echo "  - answer_entropy_*: Diversity metrics"
echo "============================================================"
