#!/bin/bash
#SBATCH --job-name=convert_ckpt
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/convert_checkpoint_%j.log
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Convert ES .pth checkpoint to HuggingFace format
# =============================================================================
#
# Usage:
#   sbatch convert_checkpoint.sh [iteration]
#   sbatch convert_checkpoint.sh 800  # Convert iter 800 (default)
#   sbatch convert_checkpoint.sh 600  # Convert iter 600
#
# =============================================================================

mkdir -p logs

module load python/3.12.11-fasrc02
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

mamba deactivate
mamba activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

cd /n/home07/itamarf/es-fine-tuning-paper/evolutionary-alignment/math

# Configuration
ITERATION="${1:-800}"
ES_RUN_DIR="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/math/es-ft-experiment/es_math_20260112_140425"
PTH_PATH="${ES_RUN_DIR}/model_saves/tmp_iter_${ITERATION}.pth"
OUTPUT_DIR="${ES_RUN_DIR}/model_saves/hf_iter_${ITERATION}"

echo "============================================================"
echo "Converting ES checkpoint to HuggingFace format"
echo "  Iteration: ${ITERATION}"
echo "  Input: ${PTH_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

# Check if .pth file exists
if [ ! -f "${PTH_PATH}" ]; then
    echo "ERROR: .pth file not found at ${PTH_PATH}"
    echo "Available checkpoints:"
    ls -la ${ES_RUN_DIR}/model_saves/
    exit 1
fi

# Check if already converted
if [ -d "${OUTPUT_DIR}" ] && [ -f "${OUTPUT_DIR}/config.json" ]; then
    echo "HuggingFace checkpoint already exists at ${OUTPUT_DIR}"
    echo "Delete it first if you want to reconvert."
    exit 0
fi

python ES/convert_pth_to_hf.py \
    --pth_path ${PTH_PATH} \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_dir ${OUTPUT_DIR} \
    --device cuda

echo ""
echo "============================================================"
echo "Conversion complete!"
echo "HuggingFace checkpoint saved to: ${OUTPUT_DIR}"
echo ""
echo "Now you can run ES evaluation with:"
echo "  sbatch eval_es.sh 256 2048 ${OUTPUT_DIR}"
echo "============================================================"
