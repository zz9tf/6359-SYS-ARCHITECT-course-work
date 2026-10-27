#!/usr/bin/env python
"""Where does one decode step spend its time? torch.profiler breakdown for the HF backends.

After a prefill of L tokens, K decode steps (one new token per sequence) are profiled. CUDA kernel
time is grouped into categories, and compared with the wall-clock time of an un-profiled step;
the difference is time the GPU sits idle waiting for the host (Python / kernel-launch overhead).
"""
import argparse
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import configs, hf, workloads  # noqa: E402
from quantbench.results import RESULTS, append_jsonl  # noqa: E402

warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")

# Checked in order; the first match wins.
CATEGORIES = [
    ("weight dequantization", r"[Dd]equant"),
    ("4-bit GEMV (bnb)", r"gemm_4bit|gemv_4bit"),
    ("activation quant / outliers", r"[Qq]uant|[Oo]utlier|int8_vector|vectorwise|kInt8|colwise|rowwise"),
    ("INT8 GEMM", r"i8|imma|int8|s8"),
    ("BF16/FP16 GEMM", r"gemm|Gemm|GEMM|nvjet|cutlass|xmma|matmul|cublas|splitK|sm\d+_"),
    ("attention", r"flash|fmha|[Aa]ttention|attn|sdpa"),
]
OTHER = "elementwise / norm / copy"


def categorize(name):
    for cat, pat in CATEGORIES:
        if re.search(pat, name):
            return cat
    return OTHER


@torch.inference_mode()
def decode_steps(model, state, k):
    ids, mask, pkv = state
    for _ in range(k):
        mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
        out = model(input_ids=ids, attention_mask=mask, past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv = out.past_key_values
        ids = out.logits[:, -1:].argmax(-1)
    return ids, mask, pkv


@torch.inference_mode()
def prefill(model, batch, length, seed=0):
    ids = workloads.synthetic_ids(batch, length, seed=seed).cuda()
    mask = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
    return out.logits[:, -1:].argmax(-1), mask, out.past_key_values


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="7b")
    ap.add_argument("--precisions", default="bf16,int8,nf4")
    ap.add_argument("--batch-sizes", default="1,32")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", type=Path, default=RESULTS / "profile_decode.jsonl")
    args = ap.parse_args()

    for prec in args.precisions.split(","):
        model, tok, _ = hf.load(configs.hf_model_id(args.size), prec)
        for B in [int(b) for b in args.batch_sizes.split(",")]:
            state = decode_steps(model, prefill(model, B, args.input_len), 4)       # warm-up
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            state = decode_steps(model, state, 32)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - t0) / 32 * 1e3

            state = prefill(model, B, args.input_len, seed=1)
            state = decode_steps(model, state, 2)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                decode_steps(model, state, args.steps)
                torch.cuda.synchronize()
            per_cat, per_kernel, n_kernels = defaultdict(float), defaultdict(float), 0
            for evt in prof.key_averages():
                t = evt.device_time_total if hasattr(evt, "device_time_total") else evt.cuda_time_total
                if t <= 0:
                    continue
                per_cat[categorize(evt.key)] += t / 1e3 / args.steps
                per_kernel[evt.key] += t / 1e3 / args.steps
                n_kernels += evt.count
            gpu_ms = sum(per_cat.values())
            top = sorted(per_kernel.items(), key=lambda kv: -kv[1])[:12]
            append_jsonl(args.out, {
                "size": args.size, "precision": prec, "batch": B, "input_len": args.input_len,
                "wall_ms_per_step": wall_ms, "gpu_kernel_ms_per_step": gpu_ms,
                "gpu_busy_frac": gpu_ms / wall_ms, "kernels_per_step": n_kernels / args.steps,
                "categories_ms": dict(per_cat), "top_kernels_ms": top})
            print(f"{prec:5s} B={B:<3d} wall={wall_ms:7.2f}ms gpu={gpu_ms:7.2f}ms "
                  f"kernels/step={n_kernels / args.steps:6.0f}  " +
                  ", ".join(f"{k}={v:.2f}" for k, v in sorted(per_cat.items(), key=lambda kv: -kv[1])), flush=True)
            for name, ms in top[:6]:
                print(f"      {ms:7.3f} ms  {name[:110]}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
