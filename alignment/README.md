# Alignment Fine-Tuning

This directory contains everything needed to run alignment fine-tuning experiments using the PKU Safe-RLHF objective. We implement two optimization methods:

- **ES (Evolutionary Strategies)**: Gradient-free optimization via weight perturbations
- **GRPO (Group Relative Policy Optimization)**: Policy gradient with group-relative advantages

Both methods optimize the same objective: **maximize helpfulness while minimizing harmfulness**.

## Directory Structure

```
alignment/
├── README.md                 # This file
│
├── data/                     # Training and evaluation data
│   ├── train_250_eval_500/
│   │   ├── custom_train_250.jsonl
│   │   └── custom_eval_500.jsonl
│   └── train_50_eval_100/
│       ├── custom_train_50.jsonl
│       └── custom_eval_100.jsonl
│
├── ES/                       # Evolutionary Strategies training
│   ├── README.md
│   ├── es_fine_tuning_alignment.py
│   ├── reconstruct_checkpoint.py
│   ├── verify_reconstruction_vllm.py
│   ├── run.sh                # SLURM launcher
│   ├── eval.sh               # Evaluation script
│   └── test_replay_log.sh    # Replay log testing
│
├── GRPO/                     # GRPO training
│   ├── README.md
│   ├── train_grpo_alignment.py
│   ├── grpo_alignment.yaml
│   ├── accelerate_trl_grpo.yaml
│   ├── ds_zero2.json
│   ├── run.sh                # SLURM launcher
│   └── eval.sh               # Evaluation script
│
├── scoring/                  # Scorer service (reward & cost models)
│   ├── README.md
│   ├── scorer_service.py
│   ├── test_score_fixed_prompt.py
│   └── run.sh                # SLURM launcher
│
├── evaluation/               # Evaluation scripts
│   ├── README.md
│   ├── classify_reward_cost_unified.py
│   ├── run_classify_reward_cost.sh
│   ├── sweep_configs.yaml
│   ├── gpt_elo_eval.py
│   └── run_gpt_elo_eval.sh
│
└── visualization/            # Plotting scripts
    ├── README.md
    ├── plot_gpt_elo_figure.py
    ├── plot_reward_cost_grid.py
    ├── plot_safety_ratios.py
    └── make_reward_cost_animation.py
```

## Quick Start

### 0. Prerequisites

Clone Safe-RLHF locally (required for scorer imports):

```bash
cd alignment
git clone https://github.com/PKU-Alignment/safe-rlhf.git safe-rlhf
```

### 1. Start the Scorer Service

Submit the scorer SLURM job (uses 1 GPU):

```bash
sbatch alignment/scoring/run.sh
```

Check the log for the hostname:
```bash
tail -f logs/scorer_service_*.log
# Look for: "Service will be available at http://<HOST>:8000"
```

### 2. Run Training

#### Option A: ES Training (4 GPUs)

```bash
# Update SCORER_HOST in alignment/ES/run.sh
sbatch alignment/ES/run.sh
```

See [ES/README.md](ES/README.md) for details.

#### Option B: GRPO Training (4 GPUs)

```bash
# Update scorer_url in alignment/GRPO/grpo_alignment.yaml
sbatch alignment/GRPO/run.sh
```

See [GRPO/README.md](GRPO/README.md) for details.

### 3. Evaluate Results

#### Quadrant Classification (R, C)

Classify model outputs into quadrants (Safe-Helpful, Safe-Unhelpful, etc.):

```bash
sbatch alignment/evaluation/run_classify_reward_cost.sh
```

Or use method-specific eval scripts:
```bash
sbatch alignment/ES/eval.sh
sbatch alignment/GRPO/eval.sh
```

#### GPT-4 ELO Evaluation

Run pairwise comparisons with GPT-4 as judge:

```bash
# Set OPENAI_API_KEY first
sbatch alignment/evaluation/run_gpt_elo_eval.sh
```

## Objective Function

Both methods optimize:

```
maximize  E[R(x, y)] - λ × E[C(x, y)]
```

Where:
- **R(x, y)**: Helpfulness score from `beaver-7b-unified-reward`
- **C(x, y)**: Harmfulness score from `beaver-7b-unified-cost`
- **λ**: Lagrange multiplier (adaptive or fixed)

**Constraint**: Keep expected cost ≤ 0 (safe responses).

## Troubleshooting

### Scorer API Not Responding
```bash
curl http://<HOST>:8000/health
```
If it fails, check if the scorer job is running: `squeue -u $USER`

### Import Errors for safe_rlhf
Ensure Safe-RLHF is cloned: `ls alignment/safe-rlhf/safe_rlhf/__init__.py`

### Out of Memory
- ES: Reduce `--batch_size` or `--population_size`
- GRPO: Reduce `num_generations` or `per_device_train_batch_size`

## References

- [PKU Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf)
- [Beaver Models](https://huggingface.co/PKU-Alignment)
- [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)
