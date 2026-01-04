#!/bin/bash
#SBATCH --job-name=grpo_conciseness_eval
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/grpo_conciseness_eval_%A_%a.log
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

# Load modules
module load python/3.12.11-fasrc02
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
source activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/evolutionary-alignment

cd ..

# Set PyTorch memory allocator configuration for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# iterate over betas and seeds and execute in an array job
BETAS=(0.0)
SEEDS=(11 22 33 44)
NB=${#BETAS[@]}
NS=${#SEEDS[@]}
IDX=${SLURM_ARRAY_TASK_ID}

BIDX=$(( IDX / NS ))
SIDX=$(( IDX % NS ))
BETA=${BETAS[$BIDX]}
SEED=${SEEDS[$SIDX]}

echo "[SLURM] array index=$IDX -> beta=$BETA seed=$SEED"

MODEL_PATH=/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta${BETA}_seed${SEED}/
python conciseness_eval.py \
    --model ${MODEL_PATH} \
    --baseline_model_name Qwen/Qwen2.5-7B-Instruct \
    --precision bf16 \
    --max_new_tokens 128 \
    --num_samples 20 \
    --batch_size 4 \
    --eval_data_path data/eval.jsonl \
    --print-examples \
    --output_json GRPO/evals/temp_0.7_beta${BETA}_seed${SEED}.json \
    --seed ${SEED} \
    --beta ${BETA} \
    --temperature 1.0 \
    --top_p 1.0 \
    --do_sample
