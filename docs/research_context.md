# Research Context: Evolutionary Alignment

## Overview

This project investigates **Evolution Strategies (ES)** as an alternative to **Reinforcement Learning (RL/GRPO)** for LLM fine-tuning. ES uses weight perturbation rather than gradient-based optimization.

## Core Thesis

**Not just "ES works at scale" — ES does something fundamentally different.**

ES is not a "poor man's GRPO." While both can achieve similar reward scores, ES exhibits qualitatively different behavior: less reward hacking, more data efficiency, and exploration of a different weight space.

## Key Findings

### 1. Hyperparameter Tuning Matters
- Qiu et al. (2025) claimed ES > RL, but this was partly due to suboptimal RL hyperparameters
- After proper tuning, GRPO can match ES on raw reward metrics
- **However**, this doesn't tell the full story...

### 2. ES Hacks Reward Less
Same data, different learning rule → ES generalizes better:
- **Conciseness task**: ES avoids degenerate solutions that GRPO finds
- **Safety alignment (PKU-SafeRLHF)**:
  - GRPO hacks reward with robotic boilerplate or hallucinated advice
  - ES converges to "helpful refusals" — providing context and legal alternatives
  - ES performs better on held-out LLM judges (not just train-time judge)

### 3. ES Explores Different Weight Space
Weight space analysis reveals qualitative differences:
- GRPO concentrates updates in unembedding layer
- ES distributes updates more evenly across layers
- Linear mode connectivity analysis shows ES occupies smoother, connected basins
- Two solution regimes identified:
  - Non-hacking regime (ES-dominated, close to base model)
  - Hacked regime (reachable by both, forms shared high-reward basin)

### 4. (Potential) ES and Solution Coverage
Open question: Does ES increase solution coverage?
- Recent work shows RL doesn't increase coverage over base LLM (can match with higher k in pass@k)
- Hypothesis: ES might actually change/increase coverage
- Relevant papers:
  - https://arxiv.org/abs/2507.14843 (reproduce Figure 1 with ES)
  - https://arxiv.org/abs/2510.15020
  - https://arxiv.org/abs/2505.24864

## Open Research Questions

1. **Sampling vs. Optimization**: Does the exploration/diversity come from ES's sampling distribution or its optimization process?
   - Potential experiment: Get rollouts via ES, then optimize via backprop

2. **KL Regularization**: Schulman's KL approximation concerns
   - http://joschu.net/blog/kl-approx.html
   - Need to verify with transformers 4.30.0

3. **Weight space evidence**: How to convincingly show ES explores a fundamentally different space?

## Tasks & Benchmarks

| Task | Description | Key Metric |
|------|-------------|------------|
| Conciseness | Generate answers matching target length | R = -\|len(gen) - len(target)\| |
| PKU-SafeRLHF | Balance helpfulness and harmlessness | Reward model + LLM judge eval |
| Countdown | (Potential) Mathematical reasoning | TBD |

## Timeline & Deadlines

**Target: ICML 2026**
- Abstract submission: **January 23, 2026 AOE**
- Full paper submission: **January 28, 2026 AOE**
- Submissions open: ~January 8-9 via OpenReview
- Author instructions: https://icml.cc/Conferences/2026/AuthorInstructions

## Related Work & Literature

### Primary Reference
- Qiu et al. (2025): "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning" (arXiv:2509.24372)
  - Need to de-couple from this paper more
  - Clarify where we agree/disagree

### Other Key Papers
- **EGGROLL**: Shows ES can work at scale (supports our narrative)
- Solution coverage papers (listed above)
- Safe RLHF benchmarks

### Feedback from Final Report
- Main actionable feedback: **Improve literature review**
- OpenReview: https://openreview.net/forum?id=1zqmmcjvdN

## Team Assignments

### Joey
- Overleaf setup
- Repo cleanup and documentation
- Weight magnitude analysis
- Fair conciseness GRPO baseline (sweep more beta, ensure KL done right)
- Countdown experiments?

### Itamar
- Fair GRPO baseline on alignment task
- Reruns with reproduced Alpaca 7b (resolving multiturn issue)
- Storage savings contribution - document in repo and write up
- Solution coverage analysis (reproduce arxiv:2507.14843 Figure 1)
- Math / Countdown experiments

### Core
- Literature review
- Framing and figure design
- Consult Sham about weight space comparisons
- Evaluate "Sample with ES, optimize with GD" experiment

## Resources

| Resource | Link |
|----------|------|
| Final report (PDF) | https://drive.google.com/file/d/16McClLBCSU7pgTV-NyVssfjJYrDlq8hF/view |
| Report feedback | https://docs.google.com/document/d/1jZwWTQnekaiTZGfUPxrziBi7SCB0xx4mqq3opbr-HH4/edit |
| OpenReview | https://openreview.net/forum?id=1zqmmcjvdN |
| Fresh experiments repo | https://github.com/jbejjani2022/evolutionary-alignment |
| Working repo (final project) | https://github.com/jbejjani2022/es-fine-tuning-paper |
| Old repo (local) | /home/ubuntu/work/evolutionary-alignment/resources/es-fine-tuning-paper |

## Key Message for Paper

> Evolution Strategies is not just a scalable alternative to RL — it explores a qualitatively different solution space. While properly-tuned GRPO can match ES on reward metrics, ES exhibits less reward hacking, better generalization to held-out judges, and distributes weight updates more evenly across layers. This makes ES a promising approach for safety-critical alignment where robustness to reward hacking matters more than raw optimization performance.

## Current State (2026-01-05)

- Paper repo synced with Overleaf (`paper/` directory, gitignored)
- AI suggestions added for:
  - Title: "Evolutionary Alignment, and What Makes It Different"
  - Abstract: Two options (~150 and ~180 words)
  - Intro: Shorter version with itemized contributions
- All AI content marked with `\ai{}` (purple) for review
- Next: Literature review, finalize framing, run additional experiments
