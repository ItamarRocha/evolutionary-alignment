#!/bin/bash
#SBATCH --job-name=es_math_replay_test
#SBATCH --output=logs/test_replay_log_math_%j.log
#SBATCH --error=logs/test_replay_log_math_%j.err
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=2:00:00
#SBATCH --mem=300GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100

# Test script for verifying math ES checkpoint reconstruction from replay logs

set -e

export PYTHONPATH=/n/home07/itamarf/es-fine-tuning-paper/evolutionary-alignment:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "ES Math Replay Log Verification Test"
echo "=========================================="
echo "Starting at: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Math experiment directory
EXPERIMENT_DIR="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/math/es-ft-experiment/es_math_20260112_140425"
REPLAY_LOG_DIR="${EXPERIMENT_DIR}/replay_logs"
MODEL_SAVES_DIR="${EXPERIMENT_DIR}/model_saves"

echo "Experiment directory: $EXPERIMENT_DIR"
echo "Replay log directory: $REPLAY_LOG_DIR"
echo "Model saves directory: $MODEL_SAVES_DIR"

# Check that replay logs exist
if [ ! -f "${REPLAY_LOG_DIR}/metadata.json" ]; then
    echo "ERROR: Replay log metadata not found!"
    exit 1
fi

if [ ! -f "${REPLAY_LOG_DIR}/iteration_logs.jsonl" ]; then
    echo "ERROR: Replay log iteration_logs.jsonl not found!"
    exit 1
fi

echo ""
echo "Replay log files found:"
ls -la ${REPLAY_LOG_DIR}/

echo ""
echo "Model checkpoints found:"
ls -la ${MODEL_SAVES_DIR}/*.pth 2>/dev/null || echo "No .pth files found"

echo ""
echo "Running verification..."
echo ""

# Run verification script (using the alignment one since it uses shared worker_extn.py)
python alignment/ES/verify_reconstruction_vllm.py \
  --replay_log_dir ${REPLAY_LOG_DIR} \
  --checkpoint_dir ${MODEL_SAVES_DIR} \
  --output_json ${EXPERIMENT_DIR}/verification_results.json

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Verification results: ${EXPERIMENT_DIR}/verification_results.json"
echo ""

# Print storage comparison
echo "Storage comparison:"
REPLAY_SIZE=$(du -sh ${REPLAY_LOG_DIR} | cut -f1)
CKPT_SIZE=$(du -sh ${MODEL_SAVES_DIR} | cut -f1)
echo "  Replay logs: $REPLAY_SIZE"
echo "  Model checkpoints: $CKPT_SIZE"
echo ""

echo "Finished at: $(date)"
