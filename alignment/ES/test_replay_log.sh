#!/bin/bash
#SBATCH --job-name=es_replay_test
#SBATCH --output=logs/test_replay_log_%j.log
#SBATCH --error=logs/test_replay_log_%j.err
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=2:00:00
#SBATCH --mem=300GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100

# Test script for seed-based checkpoint saving (replay logs)
# Runs 10 iterations with:
#   - Full checkpoint every 5 iterations (iter 5 and 10)
#   - Replay log saved every iteration
# Then verifies reconstruction matches saved checkpoints

set -e

export PYTHONPATH=/n/home07/itamarf/es-fine-tuning-paper:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Scorer service should be running - update this hostname
SCORER_HOST=${SCORER_HOST:-holygpu8a15301}
SCORER_URL="http://${SCORER_HOST}:8000"

# echo "=========================================="
# echo "ES Replay Log Checkpoint Test"
# echo "=========================================="
# echo "Starting at: $(date)"
# echo "SLURM Job ID: $SLURM_JOB_ID"
# echo "Scorer URL: $SCORER_URL"
# echo "=========================================="

# # Run training with replay log enabled
# echo ""
# echo "Step 1: Running 10 iterations with replay log saving..."
# echo ""

# python alignment/ES/es_fine_tuning_alignment.py \
#   --policy_model_path PKU-Alignment/alpaca-7b-reproduced \
#   --num_engines 4 \
#   --cuda_devices "0,1,2,3" \
#   --scorer_url $SCORER_URL \
#   --scorer_batch_size 32 \
#   --batch_size 64 \
#   --population_size 30 \
#   --num_iterations 10 \
#   --sigma 0.001 \
#   --alpha 0.0005 \
#   --max_new_tokens 256 \
#   --lambda_adapt \
#   --cost_threshold_d 0 \
#   --lambda_pos_cost_only \
#   --eval_every 0 \
#   --wandb_project "" \
#   --lambda_cost 1 \
#   --lambda_lr 0.005 \
#   --lambda_max 5.0 \
#   --train_samples 50 \
#   --eval_samples 50 \
#   --train_jsonl /n/home07/itamarf/es-fine-tuning-paper/alignment/data/train_250_eval_500/custom_train_250.jsonl \
#   --eval_jsonl /n/home07/itamarf/es-fine-tuning-paper/alignment/data/train_250_eval_500/custom_eval_500.jsonl \
#   --save_replay_log \
#   --full_checkpoint_every 5 \
#   --global_seed 42

# echo ""
# echo "Step 1 complete!"
# echo ""

# Find the most recent experiment directory
EXPERIMENT_BASE="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/alignment/es-ft-experiment"
LATEST_DIR=$(ls -dt ${EXPERIMENT_BASE}/alignment_nccl_* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "ERROR: No experiment directory found!"
    exit 1
fi

echo "Found experiment directory: $LATEST_DIR"
REPLAY_LOG_DIR="${LATEST_DIR}/replay_logs"
MODEL_SAVES_DIR="${LATEST_DIR}/model_saves"

echo "Replay log directory: $REPLAY_LOG_DIR"
echo "Model saves directory: $MODEL_SAVES_DIR"

# Check that replay logs were created
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
echo "Step 2: Verifying checkpoint reconstruction..."
echo ""

# Run verification script
python alignment/ES/verify_reconstruction_vllm.py \
  --replay_log_dir ${REPLAY_LOG_DIR} \
  --checkpoint_dir ${MODEL_SAVES_DIR} \
  --output_json ${LATEST_DIR}/verification_results.json

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo "Experiment directory: $LATEST_DIR"
echo "Verification results: ${LATEST_DIR}/verification_results.json"
echo ""

# Print storage comparison
echo "Storage comparison:"
REPLAY_SIZE=$(du -sh ${REPLAY_LOG_DIR} | cut -f1)
CKPT_SIZE=$(du -sh ${MODEL_SAVES_DIR} | cut -f1)
echo "  Replay logs: $REPLAY_SIZE"
echo "  Model checkpoints: $CKPT_SIZE"
echo ""

echo "Finished at: $(date)"
