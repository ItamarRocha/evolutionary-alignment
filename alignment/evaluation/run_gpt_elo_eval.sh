#!/bin/bash
#SBATCH --job-name=gpt_elo_eval
#SBATCH --output=logs/gpt_elo_eval_%j.log
#SBATCH --account=sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --partition=serial_requeue

set -euo pipefail

# cd /n/home07/itamarf/es-fine-tuning-paper

mkdir -p logs

# Make sure at least one OPENAI_API_KEY is set
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_KEY_2="${OPENAI_API_KEY_2:-}"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Report how many keys we have
if [ -n "$OPENAI_API_KEY_2" ]; then
    echo "Using 2 API keys for parallel requests (2x throughput)"
else
    echo "Using 1 API key (set OPENAI_API_KEY_2 for 2x throughput)"
fi

echo "Starting GPT ELO evaluation at $(date)"
echo "Using $(which python)"

python alignment/evaluation/gpt_elo_eval.py \
    --max-prompts 50 \
    --concurrency 8 \
    --judgments-out alignment/outputs/gpt_elo/gpt_judgments.jsonl \
    --ratings-out alignment/outputs/gpt_elo/gpt_elo_ratings.json \
    --plot-out alignment/outputs/gpt_elo/gpt_elo_scatter.png \
    --data-dir /n/home07/itamarf/es-fine-tuning-paper/evolutionary-alignment/alignment/results-fixed \
    --models Alpaca-7b Beaver-V1 Beaver-V2 Beaver-V3 ES-repro ES-repro-n50 GRPO-beta0.01 GRPO-beta0.05 ES-n50 ES-n250 \
    --normalize-to Alpaca-7b

echo "Finished at $(date)"
