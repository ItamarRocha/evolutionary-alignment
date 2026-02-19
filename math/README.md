# Math Reasoning: ES vs GRPO Coverage Comparison

This directory contains code for training and evaluating math reasoning models using Evolution Strategies (ES), following the methodology from ["The Invisible Leash: Why RLVR May Not Escape Its Origin"](https://arxiv.org/abs/2507.14843) (Wu et al., 2025).

## Key Hypothesis

The Invisible Leash paper shows that RLVR methods (including GRPO) have a fundamental limitation:
- **`supp(π_θ) ⊆ supp(q)`** - The trained model's support is always a subset of the base model's support
- GRPO can never discover solutions the base model assigns zero probability to
- In practice, **empirical-support shrinkage outweighs expansion**

**Our hypothesis**: ES explores in *parameter space* (noise on weights), not just sampling space. This could allow ES to:
1. Access solutions outside the base model's original support
2. Maintain broader coverage (higher pass@k at large k)
3. Avoid the "invisible leash" problem

## Directory Structure

```
math/
├── README.md                 # This file
├── math_eval.py              # Main evaluation script (Invisible Leash methodology)
├── math_verifier.py          # Answer verification logic
├── run_comparison.sh         # Full ES vs ProRL vs Base comparison
├── ES/
│   ├── es-fine-tuning_math.py    # ES training script
│   ├── eval.sh                   # SLURM script for evaluation
│   ├── run.sh                    # SLURM script for training
│   └── reconstruct_checkpoint.py # Rebuild checkpoints from replay logs
└── eval_results/             # Output directory for evaluation results
```

## Important Configuration Notes

### Token Length Mismatch

| Model | Training max_tokens | Notes |
|-------|---------------------|-------|
| ES (our training) | **1024** | Short responses |
| ProRL/Nemotron | ~8000-16000 | Long Chain-of-Thought |
| IL Paper Eval | 32000 | Full-length evaluation |

**Impact**: ES was trained with 1024 max tokens, so it may not have learned long-form reasoning. For fair comparison:
- Use `--max_new_tokens 2048` as a compromise, OR
- Retrain ES with longer sequences

### Dataset Overlap Warning

| Training Dataset | Evaluation Dataset | Overlap? |
|------------------|-------------------|----------|
| `DigitalLearningGmbH/MATH-lighteval` | `MATH500` | **YES - Same source!** |
| `DigitalLearningGmbH/MATH-lighteval` | `AIME2024` | No |
| `DigitalLearningGmbH/MATH-lighteval` | `OlympiadBench` | No |

**Recommendation**: If ES was trained on MATH-lighteval, evaluate on **AIME2024, OlympiadBench, AMC2023** only to avoid data leakage.

## Quick Start

### 1. Evaluate ProRL (GRPO) vs Base Model

```bash
# Quick test (256 samples)
python math_eval.py \
    --model_path nvidia/Nemotron-Research-Reasoning-Qwen-1.5B \
    --model_revision v1 \
    --baseline_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets MATH500 \
    --num_samples 256

# Full evaluation (matches paper)
python math_eval.py \
    --model_path nvidia/Nemotron-Research-Reasoning-Qwen-1.5B \
    --model_revision v1 \
    --baseline_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets MATH500,AIME2024,Minerva,OlympiadBench \
    --num_samples 1024
```

### 2. Evaluate ES Checkpoint

```bash
python math_eval.py \
    --model_path /path/to/es/checkpoint \
    --baseline_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets MATH500 \
    --num_samples 1024
```

### 3. Run Full Comparison (SLURM)

```bash
# Submit comparison job
sbatch run_comparison.sh

# Or with fewer samples for quick test
sbatch run_comparison.sh 256
```

## Evaluation Methodology

### Hyperparameters (Invisible Leash Paper, Appendix B.1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature | 0.6 | Paper: "temperature of 0.6" |
| Top-p | 0.95 | Paper: "top_p value of 0.95" |
| Max tokens | 32000 | Paper: "maximum response length of 32000" |
| Sampling budget | 8192 | Paper uses k=8192 for math benchmarks |

### Datasets

| Dataset | Source | Problems |
|---------|--------|----------|
| MATH500 | HuggingFaceH4/MATH-500 | 500 |
| AIME2024 | AI-MO/aime-2024 | 30 |
| AIME2025 | yentinglin/aime_2025 | 30 |
| AMC2023 | AI-MO/amc-2023 | ~100 |
| Minerva | HuggingFaceH4/Minerva-MATH | 254 |
| OlympiadBench | maxwell-jia/OlympiadBench_Dataset | ~300 |

### Metrics Computed

#### 1. Pass@k Curves
Standard pass@k metric using unbiased estimator:
```
pass@k = E[1 - C(n-c, k) / C(n, k)]
```
where `c` is the number of correct samples out of `n` total.

#### 2. Empirical-Support Categorization (Key Invisible Leash Metric)

For each correct completion, categorize as:
- **Preservation**: Found by both base model AND trained model
- **Shrinkage**: Found by base model but NOT trained model
- **Expansion**: Found by trained model but NOT base model

```
Net Coverage Change = Expansion - Shrinkage
```

If shrinkage > expansion, RLVR has narrowed the solution space (the "invisible leash" effect).

#### 3. Answer-Level Entropy

Measures diversity of final answers:
```
H = -Σ p(a) log₂ p(a)
```
Lower entropy indicates mode collapse (converging on fewer distinct answers).

#### 4. Coverage Metrics

| Metric | Definition |
|--------|------------|
| P (Precision) | Unique correct solutions / Total unique solutions |
| E (Efficiency) | Total correct / Total samples |
| S (Success) | Problems with ≥1 correct / Total problems |
| O (Overlap) | Average redundancy among correct solutions |

#### 5. Comparison Metrics

| Metric | Definition |
|--------|------------|
| SRR (Recoverability) | Can trained model reproduce baseline's correct solutions? |
| NDR (Novelty) | Fraction of trained model's solutions that are novel |
| SDS (Diversity) | Entropy-based diversity score |
| NSCR (Coverage Δ) | (Model unique correct - Baseline unique correct) / Baseline |

## Models Used

### Base Model (Shared)
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Architecture**: Qwen2ForCausalLM, 1.5B parameters
- **Note**: All three methods (Base, ProRL, ES) start from this model

### ProRL (GRPO) - Nemotron
- **Model**: `nvidia/Nemotron-Research-Reasoning-Qwen-1.5B`
- **Revision**: `v1` (2000 steps) or `main` (3000 steps)
- **Training**: GRPO with KL regularization, entropy stabilization
- **Source**: [HuggingFace](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)

### ES (Our Method)
- **Base**: Same as above
- **Training**: Evolution Strategies with parameter-space exploration
- **Checkpoints**: `/n/netscratch/.../math/es-ft-experiment/`

## Expected Results

Based on the Invisible Leash paper, we expect:

### ProRL (GRPO)
- ✅ Higher pass@1 than base model
- ❌ Lower pass@k for large k than base model
- ❌ Shrinkage > Expansion (net negative coverage change)
- ❌ Lower answer-level entropy than base model

### ES (Hypothesis)
- ✅ Improved pass@1 over base model
- ✓ Maintained or improved pass@k for large k
- ✓ Expansion ≥ Shrinkage (positive or neutral coverage change)
- ✓ Maintained answer-level entropy

## Step-by-Step Evaluation Guide

### Step 1: Quick Sanity Check

```bash
# Test with small sample size on MATH500 only
python math_eval.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets MATH500 \
    --num_samples 64 \
    --max_problems 50
```

### Step 2: Evaluate Base Model

```bash
python math_eval.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets MATH500 \
    --num_samples 1024 \
    --output_json base_results.json
```

### Step 3: Evaluate ProRL vs Base

```bash
python math_eval.py \
    --model_path nvidia/Nemotron-Research-Reasoning-Qwen-1.5B \
    --model_revision v1 \
    --baseline_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets MATH500 \
    --num_samples 1024 \
    --output_json prorl_results.json
```

### Step 4: Evaluate ES vs Base

```bash
python math_eval.py \
    --model_path /path/to/es/checkpoint \
    --baseline_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --datasets MATH500 \
    --num_samples 1024 \
    --output_json es_results.json
```

### Step 5: Full Evaluation (All Datasets)

```bash
# Use SLURM for long-running jobs
sbatch run_comparison.sh 1024
```

## Interpreting Results

### Example Output

```
--- Empirical-Support Categorization (Invisible Leash) ---
Preservation: 450 (85.71%)
Shrinkage:    52 (9.90%)
Expansion:    23 (4.38%)
Net Change:   -29
```

**Interpretation**:
- GRPO preserved 86% of solutions found by the base model
- But it lost 52 solutions (shrinkage) while only finding 23 new ones (expansion)
- Net effect: -29 solutions lost → The "invisible leash" effect

### Key Questions to Answer

1. **Does ES avoid shrinkage?**
   - Compare `shrinkage_count` between ES and ProRL
   - ES should have lower shrinkage if it escapes the invisible leash

2. **Does ES expand coverage?**
   - Compare `expansion_count` between ES and ProRL
   - ES should have higher expansion if parameter-space exploration works

3. **Does ES maintain diversity?**
   - Compare `answer_entropy_correct_normalized`
   - ES should have higher entropy than ProRL if it avoids mode collapse

## Troubleshooting

### CUDA Out of Memory
- Reduce `--num_samples` (try 256 or 512)
- Reduce `--max_new_tokens` (try 8192)
- Use `--batch_size 10` for smaller generation batches

### Dataset Loading Errors
- Some datasets require specific configs or may be gated
- Try running with just `--datasets MATH500` first

### Model Loading Issues
- For Nemotron, use `--model_revision v1` for the original checkpoint
- For ES checkpoints, ensure they're in HuggingFace format (not just .pth)

## References

1. Wu, F., et al. (2025). "The Invisible Leash: Why RLVR May Not Escape Its Origin." arXiv:2507.14843
2. Liu, M., et al. (2025). "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries." arXiv:2505.24864
3. [Nemotron-Research-Reasoning-Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)

## Citation

If you use this code, please cite:

```bibtex
@article{wu2025invisible,
  title={The Invisible Leash: Why RLVR May Not Escape Its Origin},
  author={Wu, Fang and Xuan, Weihao and Lu, Ximing and Harchaoui, Zaid and Choi, Yejin},
  journal={arXiv preprint arXiv:2507.14843},
  year={2025}
}
```
