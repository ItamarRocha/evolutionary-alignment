#!/usr/bin/env python3
"""
Convert ES .pth checkpoint to HuggingFace format for evaluation.

Usage:
    python convert_pth_to_hf.py \
        --pth_path /path/to/tmp_iter_800.pth \
        --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --output_dir /path/to/hf_checkpoint

The base_model provides the config and tokenizer; the .pth file provides the weights.
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_pth_to_hf(pth_path: str, base_model: str, output_dir: str, device: str = "cpu"):
    """
    Convert a .pth checkpoint to HuggingFace format.

    Args:
        pth_path: Path to the .pth file containing model state dict
        base_model: Base model path/name for config and tokenizer
        output_dir: Output directory for HuggingFace checkpoint
    """
    print(f"Loading base model config from: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device,
    )

    print(f"Loading weights from: {pth_path}")
    state_dict = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    print(f"Saving HuggingFace checkpoint to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # Also copy the tokenizer
    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir)

    print(f"Done! HuggingFace checkpoint saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert ES .pth checkpoint to HuggingFace format")
    parser.add_argument("--pth_path", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--base_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Base model for config/tokenizer")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for HF checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on")

    args = parser.parse_args()
    convert_pth_to_hf(args.pth_path, args.base_model, args.output_dir, args.device)


if __name__ == "__main__":
    main()
