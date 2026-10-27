#!/usr/bin/env python
"""Empirical roofline ceilings of the GPU: DRAM bandwidth and dense GEMM throughput.

  * bandwidth  : device-to-device copy of a 2 GiB tensor (read + write) and a read-only reduction
  * BF16 GEMM  : 8192^3 cuBLAS matmul
  * FP8 GEMM   : 8192^3 torch._scaled_mm (e4m3, per-tensor scales)
  * INT8 GEMM  : 8192^3 torch._int_mm

The ridge point peak_flops / peak_bandwidth is the arithmetic intensity (FLOP/byte) above which a
kernel is compute-bound. Decode-time GEMMs with a BF16 weight have an intensity of ~M FLOP/byte
for M tokens in the batch.
"""
import sys
from pathlib import Path

import torch
from triton.testing import do_bench

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench.monitor import GpuMonitor  # noqa: E402
from quantbench.results import RESULTS, append_jsonl, env_info  # noqa: E402


def bench(fn):
    with GpuMonitor() as mon:
        t = do_bench(fn, warmup=50, rep=500, return_mode="median")
    return t, mon.result


def main():
    out = RESULTS / "hw_peaks.jsonl"
    env = env_info()
    rows = []

    n_bytes = 2 * 2**30
    x = torch.empty(n_bytes // 2, dtype=torch.bfloat16, device="cuda").normal_()
    y = torch.empty_like(x)
    t, m = bench(lambda: y.copy_(x))
    rows.append({"metric": "dram_copy_gbps", "value": 2 * n_bytes / t / 1e6, "time_ms": t, **m})
    t, m = bench(lambda: x.sum(dtype=torch.float32))
    rows.append({"metric": "dram_read_gbps", "value": n_bytes / t / 1e6, "time_ms": t, **m})
    x = y = None          # free the 4 GiB before the GEMM tests
    torch.cuda.empty_cache()

    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    t, m = bench(lambda: a @ b)
    rows.append({"metric": "bf16_gemm_tflops", "value": 2 * n**3 / t / 1e9, "time_ms": t, **m})

    one = torch.ones((), device="cuda", dtype=torch.float32)
    try:
        a8 = a.to(torch.float8_e4m3fn)
        b8 = b.to(torch.float8_e4m3fn).t().contiguous().t()     # column-major
        t, m = bench(lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
        rows.append({"metric": "fp8_gemm_tflops", "value": 2 * n**3 / t / 1e9, "time_ms": t, **m})
    except Exception as e:
        rows.append({"metric": "fp8_gemm_tflops", "error": repr(e)})
    try:
        ai = torch.randint(-127, 127, (n, n), device="cuda", dtype=torch.int8)
        bi = torch.randint(-127, 127, (n, n), device="cuda", dtype=torch.int8).t().contiguous().t()
        t, m = bench(lambda: torch._int_mm(ai, bi))
        rows.append({"metric": "int8_gemm_tops", "value": 2 * n**3 / t / 1e9, "time_ms": t, **m})
    except Exception as e:
        rows.append({"metric": "int8_gemm_tops", "error": repr(e)})

    for r in rows:
        append_jsonl(out, {**r, "env": env})
        print({k: v for k, v in r.items() if k in ("metric", "value", "error", "avg_sm_clock_mhz", "avg_power_w")})


if __name__ == "__main__":
    main()
