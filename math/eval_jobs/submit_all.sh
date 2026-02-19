#!/bin/bash
# =============================================================================
# Submit all evaluation jobs
# =============================================================================
#
# Usage:
#   ./submit_all.sh                    # Default: 256 samples, 2048 tokens
#   ./submit_all.sh 128 1024           # Quick test: 128 samples, 1024 tokens
#   ./submit_all.sh 512 2048           # More samples
#
# Jobs run in parallel on separate GPUs.
#
# ES evaluation now loads .pth checkpoints directly via vLLM WorkerExtension
# (no conversion to HuggingFace format needed!)
#
# =============================================================================

NUM_SAMPLES="${1:-256}"
MAX_TOKENS="${2:-2048}"

echo "============================================================"
echo "Submitting all evaluation jobs"
echo "  Samples per problem: ${NUM_SAMPLES}"
echo "  Max tokens: ${MAX_TOKENS}"
echo "============================================================"

cd /n/home07/itamarf/es-fine-tuning-paper/evolutionary-alignment/math/eval_jobs

# Create logs directory
mkdir -p logs

# Submit jobs
JOB_BASE=$(sbatch --parsable eval_base.sh ${NUM_SAMPLES} ${MAX_TOKENS})
echo "Submitted base model job: ${JOB_BASE}"

JOB_PRORL=$(sbatch --parsable eval_prorl.sh ${NUM_SAMPLES} ${MAX_TOKENS})
echo "Submitted ProRL job: ${JOB_PRORL}"

JOB_ES=$(sbatch --parsable eval_es.sh ${NUM_SAMPLES} ${MAX_TOKENS})
echo "Submitted ES job: ${JOB_ES}"

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/eval_base_${JOB_BASE}.log"
echo "  tail -f logs/eval_prorl_${JOB_PRORL}.log"
echo "  tail -f logs/eval_es_${JOB_ES}.log"
echo ""
echo "Results will be in: ../eval_results/"
echo "============================================================"
