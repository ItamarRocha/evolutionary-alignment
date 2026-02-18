#!/usr/bin/env python
"""
GRPO training for Conciseness task.
TRL 0.26.2 compatible.

This version:
- Removes the independent (out-of-TRL) generation/eval logger entirely.
- Keeps TRL's built-in evaluation.
- Logs TRL's *eval* completions to stdout (rank 0) by capturing the exact
  completions passed into the reward function during evaluation.
- Stops logging train completions to stdout.
"""
import os
import sys
import json
import argparse


def load_yaml(path: str) -> dict:
    text = open(path, "r").read()
    try:
        return json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except Exception:
            raise RuntimeError(f"Please install pyyaml or provide valid JSON at {path}.")


def read_jsonl_rows(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def format_train_eval_datasets(cfg: dict, tokenizer):
    """
    Return (train_dataset, eval_dataset_or_None) with prompts formatted for TRL GRPO.
    Applies chat template for instruction-tuned models.
    """
    from datasets import Dataset
    pkey = cfg.get("prompt_key", "prompt")
    skey = cfg.get("solution_key", "answer")

    train_rows = list(read_jsonl_rows(cfg["train_jsonl"]))
    eval_rows = list(read_jsonl_rows(cfg["eval_jsonl"])) if (
        cfg.get("eval_jsonl") and os.path.exists(cfg["eval_jsonl"])
    ) else []

    def format_prompt(question):
        """Apply chat template to raw question."""
        messages = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted

    train_prompts = [
        {"prompt": format_prompt(str(r[pkey]).strip()), "answer": str(r[skey]).strip()}
        for r in train_rows
    ]
    eval_prompts = [
        {"prompt": format_prompt(str(r[pkey]).strip()), "answer": str(r[skey]).strip()}
        for r in eval_rows
    ]

    d_train = Dataset.from_list(train_prompts)
    d_eval = Dataset.from_list(eval_prompts) if eval_prompts else None

    return d_train, d_eval


def get_eos_token_ids(tokenizer):
    """
    Get all EOS token IDs for proper generation stopping.
    For Qwen2.5-Instruct, this includes both <|im_end|> (chat end) and <|endoftext|> (model EOS).
    """
    eos_ids = set()

    # Add the standard EOS token
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    # Add <|im_end|> token (Qwen chat format end token)
    im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end_tokens) == 1:
        eos_ids.add(im_end_tokens[0])

    # Add <|endoftext|> explicitly if different from eos_token
    try:
        endoftext_tokens = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
        if len(endoftext_tokens) == 1:
            eos_ids.add(endoftext_tokens[0])
    except Exception:
        pass

    return list(eos_ids)


def verify_setup(tokenizer, train_ds, cfg, args):
    """Print verification info for debugging. Call on rank 0 only."""
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)

    # Tokenizer checks
    print(f"\n[Tokenizer]")
    print(f"  pad_token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"  eos_token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  padding_side: {tokenizer.padding_side}")

    # Check for Qwen chat tokens
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    print(f"  <|im_end|> token ids: {im_end_id}")

    # Show all EOS token IDs
    eos_ids = get_eos_token_ids(tokenizer)
    print(f"  All EOS token IDs: {eos_ids}")

    # Sample prompt check
    print(f"\n[Sample Formatted Prompt]")
    sample = train_ds[0]
    print(f"  Prompt:\n    {repr(sample['prompt'][:200])}...")
    print(f"  Answer: {repr(sample['answer'])}")

    # Verify chat template applied
    if "<|im_start|>" not in sample['prompt']:
        print("  ⚠️  WARNING: Prompt missing <|im_start|> - chat template may not be applied!")
    else:
        print("  ✓ Chat template detected")

    if "<|im_start|>assistant" in sample['prompt']:
        print("  ✓ Generation prompt added (ends with assistant turn)")
    else:
        print("  ⚠️  WARNING: Missing assistant generation prompt!")

    # Config summary
    print(f"\n[Training Config]")
    print(f"  algorithm: GRPO")
    print(f"  beta (KL coef): {args.beta}")
    print(f"  learning_rate: {cfg['learning_rate']}")
    print(f"  num_generations: {cfg['num_generations']}")
    print(f"  max_steps: {cfg.get('max_steps', 1000)}")
    print(f"  max_completion_length: {cfg['max_completion_length']}")
    print(f"  seed: {args.seed}")

    print("\n" + "=" * 60 + "\n")


def make_reward_func(cfg: dict, tokenizer, eval_log_state: dict, max_log_samples: int = 8):
    """
    Reward (training-time):
      R = -|len(y) - len(s_k)|   (length in characters), s_k is the gold answer.

    Also: during TRL evaluation, capture a small sample of the *actual* eval completions
    (as passed to the reward function) into eval_log_state so a callback can print them
    at eval end. No train printing.
    """

    def _truncate(s, n=200):
        if s is None:
            return ""
        s = str(s)
        return s if len(s) <= n else (s[:n] + "…")

    def _extract_user_question(prompt):
        """Extract just the user question from the formatted prompt for display."""
        if prompt is None:
            return ""
        display_prompt = prompt
        if '<|im_start|>user\n' in prompt:
            start = prompt.find('<|im_start|>user\n') + len('<|im_start|>user\n')
            end = prompt.find('<|im_end|>', start)
            if end > start:
                display_prompt = prompt[start:end]
        return display_prompt

    def reward_fn(completions=None, **kwargs):
        # TRL typically passes these as keyword args:
        # prompts, completions, completions_ids, trainer_state, plus dataset columns (e.g. answer)
        if completions is None:
            completions = kwargs.get("completions", None)

        answers = kwargs.get("answer", None)
        # Prefer the documented key "prompts", but keep a fallback just in case.
        prompts = kwargs.get("prompts", None) or kwargs.get("prompt", None)

        if answers is None:
            raise ValueError("No 'answer' column found in kwargs. Check dataset columns.")
        if completions is None:
            raise ValueError("No completions provided to reward function.")

        rewards = []
        decoded_lens = []

        for completion, answer in zip(completions, answers):
            text = completion if isinstance(completion, str) else str(completion)
            y = text.strip()
            a = (answer or "").strip()
            r = -abs(len(y) - len(a))
            rewards.append(float(r))
            decoded_lens.append(len(y))

        # Capture eval samples for stdout logging (rank 0 only) — no printing here.
        is_rank_0 = os.environ.get("RANK", "0") == "0"
        if is_rank_0 and eval_log_state.get("in_eval", False):
            print(f"  [DEBUG] reward_fn called during eval with {len(completions)} completions")
            # Aggregate stats
            eval_log_state["eval_rewards"].extend(rewards)
            eval_log_state["eval_lens"].extend(decoded_lens)

            # Store a small sample (prompt/target/completion/reward)
            samples = eval_log_state["eval_samples"]
            if prompts is None:
                prompts_iter = [None] * len(rewards)
            else:
                prompts_iter = prompts

            for p, c, a, r in zip(prompts_iter, completions, answers, rewards):
                if len(samples) >= max_log_samples:
                    break
                c_text = c if isinstance(c, str) else str(c)
                samples.append({
                    "prompt": _truncate(_extract_user_question(p) if isinstance(p, str) else p, 200),
                    "target": _truncate(a, 200),
                    "completion": _truncate(c_text.strip(), 300),
                    "target_len": len(a.strip()) if isinstance(a, str) else (len(str(a)) if a is not None else 0),
                    "completion_len": len(c_text.strip()),
                    "reward": float(r),
                })

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=None, help="Override LR from config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "es_conciseness")
    cfg.setdefault("entity", None)

    # ── HF cache dir (must be set before any from_pretrained calls) ──
    hf_cache_dir = cfg.get("hf_cache_dir")
    if hf_cache_dir is not None:
        os.environ["HF_HOME"] = hf_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir

    import torch
    import random
    import numpy as np

    # ── Reproducibility (before any model / data loading) ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # W&B setup
    if cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["entity"])
    os.environ["WANDB_PROJECT"] = str(cfg["project"])

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Dataset
    train_ds, eval_ds = format_train_eval_datasets(cfg, tokenizer)

    # Verification (rank 0 only)
    if os.environ.get("RANK", "0") == "0":
        verify_setup(tokenizer, train_ds, cfg, args)

    # GRPO Config
    use_vllm = bool(int(os.environ.get("USE_VLLM", "0")))
    lr = float(args.learning_rate) if args.learning_rate is not None else float(cfg["learning_rate"])

    # Reporting: configurable via YAML (default: ["wandb"])
    report_to = cfg.get("report_to", ["wandb"])
    if isinstance(report_to, str):
        report_to = [report_to]

    grpo_args = GRPOConfig(
        output_dir=os.path.join(cfg["output_dir"], f"beta{args.beta}_lr{lr}_seed{args.seed}"),
        seed=args.seed,
        data_seed=args.seed,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=lr,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.get("warmup_steps", 0),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", 1000),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 200),
        report_to=report_to,
        run_name=f"qwen2.5-7b_grpo_conciseness_beta{args.beta}_lr{lr}_seed{args.seed}",
        bf16=True,
        remove_unused_columns=False,

        # Generation settings
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        num_generations=cfg["num_generations"],

        # GRPO specifics
        beta=float(args.beta),
        loss_type=cfg.get("loss_type", "grpo"),

        # vLLM
        use_vllm=use_vllm,

        # deepspeed config path if any
        deepspeed=cfg.get("deepspeed_config") or None,

        # Evaluation
        do_eval=eval_ds is not None,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 50)) if eval_ds is not None else None,
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        num_generations_eval=int(cfg.get("num_generations_eval", 4)),
    )

    # State used to collect TRL eval completions (captured via reward_fn) and print them at eval end.
    eval_log_state = {
        "in_eval": False,
        "eval_step": None,
        "eval_samples": [],
        "eval_rewards": [],
        "eval_lens": [],
    }
    max_eval_log_samples = int(cfg.get("max_eval_log_samples", 8))

    # Reward function (no train stdout logging; captures eval completions for callback printing)
    reward_fn = make_reward_func(
        cfg,
        tokenizer,
        eval_log_state=eval_log_state,
        max_log_samples=max_eval_log_samples,
    )

    # Trainer
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Verify KL setup (rank 0)
    if os.environ.get("RANK", "0") == "0":
        print("[KL Regularization Check]")
        if args.beta > 0:
            if hasattr(trainer, 'ref_model') and trainer.ref_model is not None:
                print(f"  ✓ Reference model loaded (beta={args.beta})")
            else:
                print(f"    Reference model attribute not found directly.")
                print(f"    This may be normal - TRL 0.26.2 handles ref model internally.")
                print(f"    Monitor 'kl' in training logs to verify KL is computed.")
        else:
            print(f"    beta=0.0, KL regularization disabled")

    # Callback that:
    # - Marks the eval window so reward_fn can capture eval completions
    # - Prints captured eval completions at eval end
    from transformers import TrainerCallback

    class TRLEvalCompletionStdoutCallback(TrainerCallback):
        def __init__(self, state_dict: dict):
            self.s = state_dict

        def on_step_end(self, args, state, control, **kwargs):
            # Evaluation (in Transformers Trainer) is triggered after step_end when control.should_evaluate is True.
            if os.environ.get("RANK", "0") != "0":
                return
            if control.should_evaluate:
                self.s["in_eval"] = True
                self.s["eval_step"] = state.global_step
                self.s["eval_samples"] = []
                self.s["eval_rewards"] = []
                self.s["eval_lens"] = []

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if os.environ.get("RANK", "0") != "0":
                return

            step = self.s.get("eval_step", state.global_step)
            rewards = self.s.get("eval_rewards", [])
            lens = self.s.get("eval_lens", [])
            samples = self.s.get("eval_samples", [])

            print(f"\n[TRL Eval Completions - Step {step}]")
            print("-" * 80)

            if not rewards:
                print("  (No eval completions captured.)")
            else:
                n = len(rewards)
                mean_reward = sum(rewards) / n
                mean_len = sum(lens) / n if lens else 0.0
                print(f"  Captured {n} eval completions")
                print(f"  mean_reward={mean_reward:.2f}, mean_len={mean_len:.1f}")
                print()

                if samples:
                    print(f"  Showing up to {len(samples)} sample(s):")
                    for i, ex in enumerate(samples, 1):
                        print(f"  [{i}/{len(samples)}]")
                        print(f"    Prompt: {ex['prompt']}")
                        print(f"    Target: {ex['target']} (len={ex['target_len']})")
                        print(f"    Completion: {ex['completion']} (len={ex['completion_len']})")
                        print(f"    Reward: {ex['reward']:.2f}")
                        print()

            print("-" * 80)
            
            # Add wandb logging for eval metrics
            if rewards:
                try:
                    import wandb
                    import numpy as np
                    wandb.log({
                        "eval/reward/mean": sum(rewards) / len(rewards),
                        "eval/reward/min": min(rewards),
                        "eval/reward/max": max(rewards),
                        "eval/reward/std": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
                        "eval/decoded_length/mean": sum(lens) / len(lens) if lens else 0.0,
                        "eval/decoded_length/min": min(lens) if lens else 0,
                        "eval/decoded_length/max": max(lens) if lens else 0,
                    })
                except Exception:
                    pass

            # Reset
            self.s["in_eval"] = False
            self.s["eval_step"] = None

    trainer.add_callback(TRLEvalCompletionStdoutCallback(eval_log_state))

    if os.environ.get("RANK", "0") == "0":
        print("\n[Logging Setup]")
        print("  ✓ TRL eval completions will be printed to stdout at each eval")
        print("  ✓ No extra (independent) generation is performed")
        print(f"  ✓ max_eval_log_samples={max_eval_log_samples}")

        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60 + "\n")

    trainer.train()
    trainer.save_model()

    if os.environ.get("RANK", "0") == "0":
        print("\n" + "=" * 60)
        print(f"Training complete. Saved to: {grpo_args.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
