#!/usr/bin/env python
"""Controlled sweep on the Hugging Face backend (Q1 memory, Q2 latency/throughput, Q3 crossover).

One model/precision is loaded once; then for every (input_len, output_len, batch) cell we run
``--warmup`` un-recorded and ``--repeats`` recorded greedy generations of exactly ``output_len``
tokens on random-token prompts. Energy is integrated over the block of recorded repeats.

    python scripts/hf_sweep.py --size 7b --precision nf4
"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import configs, hf, memory, workloads  # noqa: E402
from quantbench.monitor import GpuMonitor, device_memory_used  # noqa: E402
from quantbench.results import RESULTS, append_jsonl, env_info  # noqa: E402

warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def ints(s):
    return [int(x) for x in s.split(",") if x]


def percentile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="7b", choices=list(configs.BASE))
    ap.add_argument("--precision", default="bf16", choices=configs.HF_PRECISIONS)
    ap.add_argument("--batch-sizes", type=ints, default=ints("1,2,4,8,16,32,64"))
    ap.add_argument("--input-lens", type=ints, default=ints("128,512,1024,2048"))
    ap.add_argument("--output-lens", type=ints, default=ints("128"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--out", type=Path, default=RESULTS / "hf_sweep.jsonl")
    args = ap.parse_args()

    model_id = configs.hf_model_id(args.size)
    model, tok, load_info = hf.load(model_id, args.precision)
    cfg = model.config
    pb = memory.param_breakdown(cfg)
    kv_tok = memory.kv_bytes_per_token(cfg)
    weights = load_info["weights_bytes"]
    common = {"backend": "hf", "size": args.size, "precision": args.precision, "tag": args.tag}
    append_jsonl(RESULTS / "hf_load.jsonl", {
        **common, "model_id": model_id, **load_info,
        "params_total": pb.total, "params_linear": pb.linear, "params_embed": pb.embed,
        "params_lm_head": pb.lm_head, "params_other": pb.other,
        "theory_bytes": memory.weight_bytes(pb, args.precision),
        "theory_ideal_bytes": memory.weight_bytes(pb, args.precision, with_metadata=False),
        "naive_bytes": memory.naive_weight_bytes(pb, configs.WEIGHT_BITS[args.precision]),
        "kv_bytes_per_token": kv_tok, "env": env_info()})

    oom_cells = []   # (input_len, output_len, batch) that ran out of memory
    # Smallest cells first: bitsandbytes can abort the whole process on an allocation failure, so the
    # largest (most likely to fail) cells must come last.
    cells = sorted(((n_out, L, B) for n_out in args.output_lens for L in args.input_lens for B in args.batch_sizes),
                   key=lambda c: (c[2] * (c[1] + c[0]), c[1], c[2]))
    for n_out, L, B in cells:
        cell = {**common, "batch": B, "input_len": L, "output_len": n_out}
        if any(L >= l and n_out >= o and B >= b for l, o, b in oom_cells):
            append_jsonl(args.out, {**cell, "status": "oom_skipped"})
            continue
        try:
            runs = []
            for r in range(args.warmup):
                ids = workloads.synthetic_ids(B, L, seed=7919 * r + 31 * B + L).cuda()
                hf.timed_generate(model, ids, torch.ones_like(ids), n_out)
            hf.reset_peak()
            with GpuMonitor() as mon:
                for r in range(args.repeats):
                    ids = workloads.synthetic_ids(B, L, seed=104729 + 7919 * r + 31 * B + L).cuda()
                    res = hf.timed_generate(model, ids, torch.ones_like(ids), n_out)
                    res.pop("output_ids")
                    runs.append(res)
            peaks = hf.peak_stats(weights)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            oom_cells.append((L, n_out, B))
            append_jsonl(args.out, {**cell, "status": "oom"})
            print(f"OOM  B={B:<4d} L={L:<5d} N={n_out}", flush=True)
            continue
        mon_res = mon.result
        n_tokens_out = B * n_out * len(runs)
        for i, res in enumerate(runs):
            gaps = [b - a for a, b in zip(res["step_times"], res["step_times"][1:])]
            append_jsonl(args.out, {
                **cell, "status": "ok", "repeat": i,
                "ttft_s": res["ttft_s"], "tpot_s": res["tpot_s"], "e2e_s": res["e2e_s"],
                "itl_p50_s": percentile(gaps, 0.5) if gaps else None,
                "itl_p99_s": percentile(gaps, 0.99) if gaps else None,
                "prefill_tok_s": B * L / res["ttft_s"],
                "decode_tok_s": B / res["tpot_s"],
                "output_tok_s": B * n_out / res["e2e_s"],
                "kv_cache_bytes": B * (L + n_out) * kv_tok,
                "weights_bytes": weights,
                **peaks,
                "nvml_peak_used_bytes": mon_res["nvml_peak_used_bytes"],
                "energy_j_per_out_token": mon_res["energy_j"] / n_tokens_out,
                "block_energy_j": mon_res["energy_j"],
                "avg_power_w": mon_res["avg_power_w"],
                "avg_sm_clock_mhz": mon_res["avg_sm_clock_mhz"],
                "temp_end_c": mon_res["temp_end_c"],
            })
        ttft = sum(r["ttft_s"] for r in runs) / len(runs)
        tpot = sum(r["tpot_s"] for r in runs) / len(runs)
        print(f"ok   B={B:<4d} L={L:<5d} N={n_out:<4d} ttft={ttft * 1e3:8.1f}ms "
              f"tpot={tpot * 1e3:7.2f}ms peak={peaks['torch_peak_alloc_bytes'] / 2**30:6.2f}GiB "
              f"nvml={device_memory_used() / 2**30:6.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
