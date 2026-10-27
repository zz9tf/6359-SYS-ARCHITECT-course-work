#!/usr/bin/env python
"""Linear-layer microbenchmark: for which batch sizes does each low-bit kernel beat BF16? (Q3)

We time y = x W^T for the four linear shapes of one Qwen2.5 decoder layer (fused QKV, O,
fused gate/up, down) and sum them, for M (tokens in the batch) from 1 to 8192. Kernels:

  bf16          torch F.linear -> cuBLAS
  bnb-int8      bitsandbytes Linear8bitLt (LLM.int8(), outlier threshold 6.0)
  bnb-nf4       bitsandbytes Linear4bit (NF4 weights, BF16 compute)
  fp8-w8a8      vLLM: dynamic per-token FP8 activation quantization + CUTLASS FP8 GEMM
                (per-tensor FP8 weights); the activation-quantization kernel is included
  marlin-w8a16  vLLM gptq_marlin, INT8 weights, group 128 (what GPTQ-Int8 runs)
  marlin-w4a16  vLLM gptq_marlin, INT4 weights, group 128 (what GPTQ/AWQ-Int4 run)

Timing uses triton.testing.do_bench, which flushes the L2 cache before every call so weights are
streamed from DRAM as they are during real decoding.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from triton.testing import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench.results import RESULTS, append_jsonl, env_info  # noqa: E402

SHAPES = {  # name -> (K in_features, N out_features), Qwen2.5-7B
    "7b": {"qkv": (3584, 4608), "o": (3584, 3584), "gate_up": (3584, 37888), "down": (18944, 3584)},
    "3b": {"qkv": (2048, 2560), "o": (2048, 2048), "gate_up": (2048, 22016), "down": (11008, 2048)},
}


def make_bf16(w):
    return lambda x: F.linear(x, w)


def make_bnb_int8(w):
    import bitsandbytes as bnb
    n, k = w.shape
    layer = bnb.nn.Linear8bitLt(k, n, bias=False, has_fp16_weights=False, threshold=6.0)
    layer.weight = bnb.nn.Int8Params(w.detach().to(torch.float16).cpu(), requires_grad=False,
                                     has_fp16_weights=False)
    layer = layer.cuda()
    return lambda x: layer(x)


def make_bnb_nf4(w):
    import bitsandbytes as bnb
    n, k = w.shape
    layer = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(w.detach().cpu(), requires_grad=False, quant_type="nf4")
    layer = layer.cuda()
    return lambda x: layer(x)


def make_fp8(w):
    from vllm import _custom_ops as ops
    w_q, w_s = ops.scaled_fp8_quant(w)          # one per-tensor weight scale, as vLLM's online FP8
    w_t = w_q.t()                                # column-major (K, N)

    def fn(x):
        x_q, x_s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)   # dynamic per-token
        return ops.cutlass_scaled_mm(x_q, w_t, scale_a=x_s, scale_b=w_s, out_dtype=torch.bfloat16)
    return fn


def make_marlin(w, bits):
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear, marlin_make_workspace_new)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types
    n, k = w.shape
    wtype = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
    _, q_w, s = marlin_quantize(w.t().contiguous().to(torch.float16), wtype, 128)
    s = s.to(torch.bfloat16)
    zp = torch.empty(0, dtype=torch.int, device=w.device)
    ws = marlin_make_workspace_new(w.device)
    return lambda x: apply_gptq_marlin_linear(x, q_w, s, zp, ws, wtype, n, k)


KERNELS = {
    "bf16": make_bf16,
    "bnb-int8": make_bnb_int8,
    "bnb-nf4": make_bnb_nf4,
    "fp8-w8a8": make_fp8,
    "marlin-w8a16": lambda w: make_marlin(w, 8),
    "marlin-w4a16": lambda w: make_marlin(w, 4),
}
WEIGHT_BYTES_PER_PARAM = {"bf16": 2, "bnb-int8": 1, "bnb-nf4": 0.5 + 4 / 64, "fp8-w8a8": 1,
                          "marlin-w8a16": 1 + 2 / 128, "marlin-w4a16": 0.5 + 2 / 128}


def make_input(m, k, outliers=True):
    """Activations ~N(0,1) plus a few large-magnitude feature columns, as in real LLM hidden states."""
    g = torch.Generator(device="cuda").manual_seed(m * 7 + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    if outliers:
        x[:, [7, k // 3, k // 2, k - 5]] *= 25.0
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="7b", choices=list(SHAPES))
    ap.add_argument("--ms", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192")
    ap.add_argument("--kernels", default=",".join(KERNELS))
    ap.add_argument("--out", type=Path, default=RESULTS / "kernel_bench.jsonl")
    args = ap.parse_args()
    ms = [int(x) for x in args.ms.split(",")]
    append_jsonl(RESULTS / "kernel_env.jsonl", {"size": args.size, "env": env_info()})

    torch.manual_seed(0)
    for kname in args.kernels.split(","):
        for sname, (k, n) in SHAPES[args.size].items():
            w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
            try:
                fn = KERNELS[kname](w)
            except Exception as e:  # kernel not available on this GPU / build
                append_jsonl(args.out, {"size": args.size, "kernel": kname, "shape": sname, "error": repr(e)})
                print(f"{kname:13s} {sname:8s} unavailable: {e!r}", flush=True)
                continue
            ref_err = None
            for m in ms:
                x = make_input(m, k)
                try:
                    if m == 4:  # numerical sanity check against BF16
                        y_ref = F.linear(x, w).float()
                        ref_err = ((fn(x).float() - y_ref).norm() / y_ref.norm()).item()
                    t_ms = do_bench(lambda: fn(x), warmup=25, rep=100, return_mode="median")
                except Exception as e:
                    append_jsonl(args.out, {"size": args.size, "kernel": kname, "shape": sname, "m": m, "error": repr(e)})
                    print(f"{kname:13s} {sname:8s} M={m:5d} failed: {e!r}", flush=True)
                    break
                flops = 2 * m * k * n
                append_jsonl(args.out, {
                    "size": args.size, "kernel": kname, "shape": sname, "k": k, "n": n, "m": m,
                    "time_ms": t_ms, "tflops": flops / t_ms / 1e9,
                    "weight_gbps": n * k * WEIGHT_BYTES_PER_PARAM[kname] / t_ms / 1e6,
                    "rel_err_vs_bf16": ref_err})
            print(f"{kname:13s} {sname:8s} done (rel err vs bf16 @M=4: {ref_err})", flush=True)
            fn = w = None
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
