#!/usr/bin/env python3
"""
Verification Script for ES Replay Log Reconstruction (vLLM-based)

This script verifies that checkpoints reconstructed from replay logs match
the original saved checkpoints. It uses vLLM to ensure the same weight format
as training.

Features:
- Timing comparison: reconstruction from replay log vs direct checkpoint loading
  - Separates engine creation time from actual operation time
  - Attempts to drop cache for fair cold comparison
- Weight comparison: max/mean/RMSE differences
- Inference comparison: run both models on same prompt with temp=0
- Storage comparison: per-checkpoint size breakdown

Usage:
    python verify_reconstruction_vllm.py \
        --replay_log_dir /path/to/replay_logs \
        --checkpoint_dir /path/to/model_saves

Requires GPU access (uses vLLM).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import ray
from vllm import LLM, SamplingParams

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# Test prompt for inference comparison
TEST_PROMPT = "BEGINNING OF CONVERSATION: USER: What is the capital of France? ASSISTANT:"


def drop_cache() -> bool:
    """Try to drop the Linux page cache. Returns True if successful."""
    try:
        # First sync to flush writes
        subprocess.run(["sync"], check=True, timeout=30)
        # Try to drop caches (requires sudo)
        result = subprocess.run(
            ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass
    return False


def load_replay_metadata(replay_dir: str) -> dict:
    """Load the one-time metadata from replay log."""
    meta_path = os.path.join(replay_dir, "metadata.json")
    with open(meta_path, "r") as f:
        return json.load(f)


def load_iteration_logs(replay_dir: str, start_iter: int = 0, end_iter: int = None) -> List[dict]:
    """Load iteration logs from the JSONL file within the specified range."""
    log_path = os.path.join(replay_dir, "iteration_logs.jsonl")
    logs = []
    with open(log_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            iter_num = entry["iteration"]
            if iter_num >= start_iter and (end_iter is None or iter_num < end_iter):
                logs.append(entry)
    return logs


def load_full_checkpoint_markers(replay_dir: str) -> List[dict]:
    """Load the list of full checkpoint markers."""
    markers_path = os.path.join(replay_dir, "full_checkpoints.json")
    if not os.path.exists(markers_path):
        return []
    with open(markers_path, "r") as f:
        return json.load(f)


def find_checkpoints_to_verify(replay_dir: str, checkpoint_dir: str) -> List[dict]:
    """Find checkpoints to verify from markers or by scanning directory."""
    markers = load_full_checkpoint_markers(replay_dir)
    
    if not markers:
        # Scan for .pth files
        if os.path.isdir(checkpoint_dir):
            for fname in os.listdir(checkpoint_dir):
                if fname.startswith("tmp_iter_") and fname.endswith(".pth"):
                    try:
                        iter_num = int(fname.replace("tmp_iter_", "").replace(".pth", ""))
                        markers.append({
                            "iteration": iter_num,
                            "checkpoint_path": os.path.join(checkpoint_dir, fname),
                        })
                    except ValueError:
                        continue
            markers.sort(key=lambda x: x["iteration"])
    
    return markers


def get_file_size_mb(path: str) -> float:
    """Get file size in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def get_dir_size_kb(path: str) -> float:
    """Get directory size in KB."""
    total = 0
    if os.path.isdir(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1024


def count_parameters(state_dict: Dict[str, torch.Tensor]) -> int:
    """Count total parameters in state dict."""
    return sum(t.numel() for t in state_dict.values())


def compare_state_dicts(reconstructed: Dict[str, torch.Tensor], 
                        original: Dict[str, torch.Tensor]) -> Dict:
    """Compare two state dicts and return statistics."""
    rec_keys = set(reconstructed.keys())
    orig_keys = set(original.keys())
    common_keys = rec_keys & orig_keys
    
    if not common_keys:
        return {
            "max_abs_diff": float('nan'),
            "mean_abs_diff": float('nan'),
            "rmse": float('nan'),
            "total_params": 0,
            "status": "no_common_keys",
        }
    
    total_params = 0
    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    sum_sq_diff = 0.0
    
    for key in common_keys:
        rec = reconstructed[key].float()
        orig = original[key].float()
        
        if rec.shape != orig.shape:
            continue
        
        diff = (rec - orig).abs()
        max_abs_diff = max(max_abs_diff, diff.max().item())
        sum_abs_diff += diff.sum().item()
        sum_sq_diff += (diff ** 2).sum().item()
        total_params += diff.numel()
    
    mean_abs_diff = sum_abs_diff / total_params if total_params > 0 else 0.0
    rmse = np.sqrt(sum_sq_diff / total_params) if total_params > 0 else 0.0
    
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rmse": rmse,
        "total_params": total_params,
        "num_common_keys": len(common_keys),
        "status": "compared",
    }


class ESNcclLLM(LLM):
    """vLLM wrapper with worker extension support."""
    def __init__(self, *args, **kwargs):
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def create_engine(base_model_path: str, gpu_mem: float = 0.45):
    """Create a vLLM engine with worker extension."""
    ESNcclLLMRemote = ray.remote(num_cpus=8, num_gpus=1)(ESNcclLLM)
    engine = ESNcclLLMRemote.remote(
        model=base_model_path,
        tensor_parallel_size=1,
        distributed_executor_backend="mp",
        worker_extension_cls="utils.worker_extn.WorkerExtension",
        dtype="float16",
        enable_prefix_caching=False,
        enforce_eager=False,
        gpu_memory_utilization=gpu_mem,
    )
    return engine


def run_inference(engine, prompt: str, max_tokens: int = 64) -> str:
    """Run inference on a vLLM engine and return the generated text."""
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # Deterministic
        seed=42,
    )
    outputs = ray.get(engine.generate.remote([prompt], sampling))
    return outputs[0].outputs[0].text


def verify_single_checkpoint(
    iteration: int,
    checkpoint_path: str,
    replay_log_dir: str,
    base_model_path: str,
    can_drop_cache: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Verify a single checkpoint with fair timing comparison:
    1. Create engine (same for both)
    2. Time JUST the load operation vs JUST the reconstruction operation
    """
    result = {
        "iteration": iteration,
        "checkpoint_path": checkpoint_path,
    }
    
    if not os.path.exists(checkpoint_path):
        result["status"] = "missing"
        return result
    
    # Get checkpoint file size
    ckpt_size_mb = get_file_size_mb(checkpoint_path)
    result["checkpoint_size_mb"] = ckpt_size_mb
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Verifying iteration {iteration}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Checkpoint size: {ckpt_size_mb:.2f} MB ({ckpt_size_mb/1024:.2f} GB)")
    
    # Load iteration logs for reconstruction
    logs = load_iteration_logs(replay_log_dir, 0, iteration)
    total_perturbs = sum(len(log["seeds"]) for log in logs)
    
    # =========================================================================
    # COLD CACHE TEST (if possible)
    # =========================================================================
    cold_load_time = None
    cold_reconstruct_time = None
    
    if can_drop_cache:
        if verbose:
            print(f"\n  [COLD CACHE TEST]")
        
        # Test 1: Cold load
        if verbose:
            print(f"    Dropping cache...")
        drop_cache()
        time.sleep(1)
        
        if verbose:
            print(f"    Creating engine for cold load test...")
        engine = create_engine(base_model_path, gpu_mem=0.40)
        ray.get(engine.generate.remote(["warmup"], SamplingParams(max_tokens=1)))
        
        if verbose:
            print(f"    Timing checkpoint load (cold)...")
        t0 = time.time()
        ray.get(engine.collective_rpc.remote(
            "load_self_weights_from_disk", args=(checkpoint_path,)
        ))
        cold_load_time = time.time() - t0
        
        # Kill engine
        try:
            ray.kill(engine)
        except:
            pass
        torch.cuda.empty_cache()
        time.sleep(2)
        
        # Test 2: Cold reconstruct
        if verbose:
            print(f"    Dropping cache...")
        drop_cache()
        time.sleep(1)
        
        if verbose:
            print(f"    Creating engine for cold reconstruct test...")
        engine = create_engine(base_model_path, gpu_mem=0.40)
        ray.get(engine.generate.remote(["warmup"], SamplingParams(max_tokens=1)))
        
        if verbose:
            print(f"    Timing reconstruction (cold, {len(logs)} iterations, {total_perturbs} perturbations)...")
        t0 = time.time()
        for log_entry in logs:
            seeds = log_entry["seeds"]
            coeffs = log_entry["update_coeffs"]
            for seed, coeff in zip(seeds, coeffs):
                ray.get(engine.collective_rpc.remote(
                    "perturb_self_weights", args=(seed, coeff, False)
                ))
        cold_reconstruct_time = time.time() - t0
        
        # Kill engine
        try:
            ray.kill(engine)
        except:
            pass
        torch.cuda.empty_cache()
        time.sleep(2)
        
        if verbose:
            print(f"    Cold load time:        {cold_load_time:.2f}s")
            print(f"    Cold reconstruct time: {cold_reconstruct_time:.2f}s")
            if cold_load_time > 0:
                ratio = cold_reconstruct_time / cold_load_time
                print(f"    Ratio (reconstruct/load): {ratio:.2f}x")
        
        result["cold_load_time"] = cold_load_time
        result["cold_reconstruct_time"] = cold_reconstruct_time
    
    # =========================================================================
    # WARM CACHE TEST
    # =========================================================================
    if verbose:
        print(f"\n  [WARM CACHE TEST]")
    
    # Pre-warm the cache by reading both files
    if verbose:
        print(f"    Warming cache...")
    
    # Read checkpoint to warm cache
    _ = torch.load(checkpoint_path, map_location="cpu")
    del _
    
    # Read base model files to warm cache
    base_model_files = []
    for root, dirs, files in os.walk(base_model_path):
        for f in files:
            if f.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                base_model_files.append(os.path.join(root, f))
    for f in base_model_files[:5]:  # Read first few weight files
        try:
            with open(f, 'rb') as fp:
                _ = fp.read(1024*1024)  # Read 1MB to warm cache
        except:
            pass
    
    time.sleep(1)
    
    # Test 1: Warm load
    if verbose:
        print(f"    Creating engine for warm load test...")
    engine_load = create_engine(base_model_path, gpu_mem=0.40)
    ray.get(engine_load.generate.remote(["warmup"], SamplingParams(max_tokens=1)))
    
    if verbose:
        print(f"    Timing checkpoint load (warm)...")
    t0 = time.time()
    ray.get(engine_load.collective_rpc.remote(
        "load_self_weights_from_disk", args=(checkpoint_path,)
    ))
    warm_load_time = time.time() - t0
    
    if verbose:
        print(f"    Running inference on loaded model...")
    output_load = run_inference(engine_load, TEST_PROMPT)
    
    # Save weights for comparison
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        load_weights_path = tmp.name
    ray.get(engine_load.collective_rpc.remote(
        "save_self_weights_to_disk", args=(load_weights_path,)
    ))
    loaded_state = torch.load(load_weights_path, map_location="cpu")
    os.unlink(load_weights_path)
    
    num_params = count_parameters(loaded_state)
    
    # Kill engine
    try:
        ray.kill(engine_load)
    except:
        pass
    torch.cuda.empty_cache()
    time.sleep(2)
    
    # Test 2: Warm reconstruct
    if verbose:
        print(f"    Creating engine for warm reconstruct test...")
    engine_recon = create_engine(base_model_path, gpu_mem=0.40)
    ray.get(engine_recon.generate.remote(["warmup"], SamplingParams(max_tokens=1)))
    
    if verbose:
        print(f"    Timing reconstruction (warm, {len(logs)} iterations, {total_perturbs} perturbations)...")
    t0 = time.time()
    for log_entry in logs:
        seeds = log_entry["seeds"]
        coeffs = log_entry["update_coeffs"]
        for seed, coeff in zip(seeds, coeffs):
            ray.get(engine_recon.collective_rpc.remote(
                "perturb_self_weights", args=(seed, coeff, False)
            ))
    warm_reconstruct_time = time.time() - t0
    
    if verbose:
        print(f"    Running inference on reconstructed model...")
    output_recon = run_inference(engine_recon, TEST_PROMPT)
    
    # Save weights for comparison
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        recon_weights_path = tmp.name
    ray.get(engine_recon.collective_rpc.remote(
        "save_self_weights_to_disk", args=(recon_weights_path,)
    ))
    reconstructed_state = torch.load(recon_weights_path, map_location="cpu")
    os.unlink(recon_weights_path)
    
    # Kill engine
    try:
        ray.kill(engine_recon)
    except:
        pass
    torch.cuda.empty_cache()
    
    if verbose:
        print(f"    Warm load time:        {warm_load_time:.2f}s")
        print(f"    Warm reconstruct time: {warm_reconstruct_time:.2f}s")
        if warm_load_time > 0:
            ratio = warm_reconstruct_time / warm_load_time
            print(f"    Ratio (reconstruct/load): {ratio:.2f}x")
    
    result["warm_load_time"] = warm_load_time
    result["warm_reconstruct_time"] = warm_reconstruct_time
    result["num_iterations_replayed"] = len(logs)
    result["num_perturbations_applied"] = total_perturbs
    result["num_params"] = num_params
    
    # =========================================================================
    # WEIGHT COMPARISON
    # =========================================================================
    if verbose:
        print(f"\n  [Weight Comparison]")
    
    comparison = compare_state_dicts(reconstructed_state, loaded_state)
    result["comparison"] = comparison
    result["status"] = "verified"
    
    if verbose:
        print(f"    Max absolute difference: {comparison['max_abs_diff']:.2e}")
        print(f"    Mean absolute difference: {comparison['mean_abs_diff']:.2e}")
        print(f"    RMSE: {comparison['rmse']:.2e}")
        print(f"    Parameters compared: {comparison['total_params']:,}")
    
    # =========================================================================
    # INFERENCE COMPARISON
    # =========================================================================
    outputs_match = output_load.strip() == output_recon.strip()
    result["outputs_match"] = outputs_match
    result["output_load"] = output_load
    result["output_recon"] = output_recon
    
    if verbose:
        print(f"\n  [Inference Comparison]")
        print(f"    Prompt: \"{TEST_PROMPT[:50]}...\"")
        print(f"    Output (loaded):       \"{output_load[:80]}...\"")
        print(f"    Output (reconstructed): \"{output_recon[:80]}...\"")
        print(f"    Outputs match: {'✅ YES' if outputs_match else '❌ NO'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    if verbose:
        print(f"\n  [Summary for iteration {iteration}]")
        print(f"    Parameters: {num_params:,}")
        if can_drop_cache and cold_load_time and cold_reconstruct_time:
            print(f"    Cold: load={cold_load_time:.2f}s, reconstruct={cold_reconstruct_time:.2f}s, ratio={cold_reconstruct_time/cold_load_time:.2f}x")
        print(f"    Warm: load={warm_load_time:.2f}s, reconstruct={warm_reconstruct_time:.2f}s, ratio={warm_reconstruct_time/warm_load_time:.2f}x")
        
        if comparison['max_abs_diff'] < 1e-6:
            print(f"    Weight match: ✅ EXACT")
        elif comparison['max_abs_diff'] < 1e-2:
            print(f"    Weight match: ✅ CLOSE (fp16 numerical precision)")
        else:
            print(f"    Weight match: ⚠️  DIFFERENCES")
    
    del loaded_state, reconstructed_state
    
    return result


def verify_checkpoints_vllm(
    replay_log_dir: str,
    checkpoint_dir: str,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Verify all checkpoints using vLLM for reconstruction.
    Returns results list and storage stats.
    """
    results = []
    
    # Load metadata
    metadata = load_replay_metadata(replay_log_dir)
    base_model_path = metadata["base_model_path"]
    
    if verbose:
        print(f"Base model: {base_model_path}")
    
    # Find checkpoints to verify
    markers = find_checkpoints_to_verify(replay_log_dir, checkpoint_dir)
    if not markers:
        print("No checkpoints found to verify")
        return results, {}
    
    if verbose:
        print(f"Found {len(markers)} checkpoints to verify")
    
    # Calculate storage stats
    replay_log_size_kb = get_dir_size_kb(replay_log_dir)
    
    storage_stats = {
        "replay_log_size_kb": replay_log_size_kb,
        "checkpoints": [],
    }
    
    # Test if we can drop cache
    if verbose:
        print(f"\nTesting cache drop capability...")
    can_drop_cache = drop_cache()
    if verbose:
        if can_drop_cache:
            print(f"  ✅ Can drop cache (sudo available)")
        else:
            print(f"  ❌ Cannot drop cache (no sudo) - will only do warm cache test")
    
    # Initialize Ray
    os.environ.pop("RAY_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)
    
    # Verify each checkpoint
    for marker in markers:
        iteration = marker["iteration"]
        checkpoint_path = marker["checkpoint_path"]
        
        result = verify_single_checkpoint(
            iteration=iteration,
            checkpoint_path=checkpoint_path,
            replay_log_dir=replay_log_dir,
            base_model_path=base_model_path,
            can_drop_cache=can_drop_cache,
            verbose=verbose,
        )
        results.append(result)
        
        # Add to storage stats
        if result.get("checkpoint_size_mb"):
            storage_stats["checkpoints"].append({
                "iteration": iteration,
                "size_mb": result["checkpoint_size_mb"],
                "size_gb": result["checkpoint_size_mb"] / 1024,
            })
    
    # Cleanup Ray
    ray.shutdown()
    
    return results, storage_stats


def main():
    parser = argparse.ArgumentParser(
        description="Verify ES checkpoint reconstruction using vLLM"
    )
    parser.add_argument(
        "--replay_log_dir",
        type=str,
        required=True,
        help="Path to the replay_logs directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to model_saves directory"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save verification results as JSON"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    if args.checkpoint_dir is None:
        parent_dir = os.path.dirname(args.replay_log_dir)
        args.checkpoint_dir = os.path.join(parent_dir, "model_saves")
    
    print("="*60)
    print("ES Replay Log Verification (vLLM-based)")
    print("="*60)
    print(f"Replay log dir: {args.replay_log_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("="*60)
    
    results, storage_stats = verify_checkpoints_vllm(
        replay_log_dir=args.replay_log_dir,
        checkpoint_dir=args.checkpoint_dir,
        verbose=not args.quiet,
    )
    
    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    
    verified = [r for r in results if r["status"] == "verified"]
    missing = [r for r in results if r["status"] == "missing"]
    
    print(f"\nCheckpoints: {len(results)} total, {len(verified)} verified, {len(missing)} missing")
    
    if verified:
        # Check if we have cold cache results
        has_cold = any(r.get("cold_load_time") is not None for r in verified)
        
        # Timing comparison
        print(f"\n--- Timing Comparison (operation time only, excludes engine creation) ---")
        
        if has_cold:
            print(f"\nCOLD CACHE (no OS page cache):")
            print(f"{'Iter':<6} {'Perturbs':<10} {'Load (s)':<12} {'Reconstruct (s)':<18} {'Ratio':<10}")
            print("-" * 56)
            for r in verified:
                if r.get("cold_load_time") is not None:
                    load_t = r["cold_load_time"]
                    recon_t = r["cold_reconstruct_time"]
                    ratio = recon_t / load_t if load_t > 0 else 0
                    print(f"{r['iteration']:<6} {r['num_perturbations_applied']:<10} {load_t:<12.2f} {recon_t:<18.2f} {ratio:.2f}x")
        
        print(f"\nWARM CACHE (OS page cache active):")
        print(f"{'Iter':<6} {'Perturbs':<10} {'Load (s)':<12} {'Reconstruct (s)':<18} {'Ratio':<10}")
        print("-" * 56)
        for r in verified:
            load_t = r["warm_load_time"]
            recon_t = r["warm_reconstruct_time"]
            ratio = recon_t / load_t if load_t > 0 else 0
            print(f"{r['iteration']:<6} {r['num_perturbations_applied']:<10} {load_t:<12.2f} {recon_t:<18.2f} {ratio:.2f}x")
        
        print(f"\n(Ratio > 1 means reconstruction is slower than loading)")
        
        # Weight comparison
        max_diffs = [r["comparison"]["max_abs_diff"] for r in verified]
        print(f"\n--- Weight Comparison ---")
        print(f"Max absolute difference (across all): {max(max_diffs):.2e}")
        
        if max(max_diffs) < 1e-6:
            print("Status: ✅ All checkpoints match exactly!")
        elif max(max_diffs) < 1e-2:
            print("Status: ✅ All checkpoints match (fp16 numerical precision)")
        else:
            print("Status: ⚠️  Some differences detected")
        
        # Inference comparison
        outputs_match = all(r.get("outputs_match", False) for r in verified)
        print(f"\n--- Inference Comparison ---")
        print(f"All outputs match: {'✅ YES' if outputs_match else '❌ NO'}")
        
        # Parameters
        print(f"\n--- Parameters ---")
        for r in verified:
            print(f"Iteration {r['iteration']}: {r.get('num_params', 0):,} parameters")
    
    # Storage comparison
    print(f"\n--- Storage Comparison ---")
    replay_kb = storage_stats.get("replay_log_size_kb", 0)
    print(f"Replay log total: {replay_kb:.2f} KB ({replay_kb/1024:.4f} MB)")
    print(f"\nPer-checkpoint breakdown:")
    print(f"{'Iter':<6} {'Checkpoint':<20} {'Replay Log':<20} {'Reduction':<15}")
    print("-" * 61)
    
    total_ckpt_mb = 0
    for i, ckpt in enumerate(storage_stats.get("checkpoints", [])):
        iteration = ckpt["iteration"]
        size_mb = ckpt["size_mb"]
        size_gb = ckpt["size_gb"]
        total_ckpt_mb += size_mb
        
        # Replay log size for this iteration (proportional)
        max_iter = max(c["iteration"] for c in storage_stats["checkpoints"])
        iter_replay_kb = replay_kb * (iteration / max_iter)
        
        reduction = (size_mb * 1024) / iter_replay_kb if iter_replay_kb > 0 else 0
        print(f"{iteration:<6} {size_gb:.2f} GB            {iter_replay_kb:.2f} KB            {reduction:,.0f}x")
    
    print(f"\nTotal checkpoint storage: {total_ckpt_mb/1024:.2f} GB")
    print(f"Total replay log storage: {replay_kb:.2f} KB")
    overall_reduction = (total_ckpt_mb * 1024) / replay_kb if replay_kb > 0 else 0
    print(f"Overall storage reduction: {overall_reduction:,.0f}x")
    
    # Save results
    if args.output_json:
        output_data = {
            "results": results,
            "storage_stats": storage_stats,
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
