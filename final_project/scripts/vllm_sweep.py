#!/usr/bin/env python
"""Controlled sweep on the vLLM backend (same grid as hf_sweep.py, larger batches).

Methodology follows ``vllm bench latency``: B random-token prompts of length L are submitted
together. T(1) - the latency with max_tokens=1 - is the batch time-to-first-token, and
TPOT = (T(N) - T(1)) / (N - 1). Prefix caching is disabled so repeats never reuse KV blocks.

    python scripts/vllm_sweep.py --size 7b --precision awq-int4
"""
import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # avoid FlashInfer JIT (needs nvcc)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import configs, memory, workloads  # noqa: E402
from quantbench.monitor import GpuMonitor  # noqa: E402
from quantbench.results import RESULTS, append_jsonl, env_info  # noqa: E402


def ints(s):
    return [int(x) for x in s.split(",") if x]


def build_engine(model_id, quant, args):
    """Create the engine while teeing fd 1/2 into a file so vLLM's own log lines can be parsed."""
    from vllm import LLM

    log = tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False)
    saved = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    t0 = time.perf_counter()
    try:
        llm = LLM(model=model_id, quantization=quant, gpu_memory_utilization=args.gpu_util,
                  max_model_len=args.max_model_len, max_num_seqs=max(args.batch_sizes),
                  max_num_batched_tokens=args.max_batched_tokens, enable_prefix_caching=False,
                  seed=0)
        err = None
    except Exception as e:  # e.g. the checkpoint does not fit in memory
        llm, err = None, repr(e)
    init_s = time.perf_counter() - t0
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(saved[0], 1)
    os.dup2(saved[1], 2)
    text = Path(log.name).read_text(errors="replace")
    print(text[-4000:], flush=True)

    def grab(pattern, cast=float):
        m = re.findall(pattern, text)
        return cast(m[-1].replace(",", "")) if m else None

    info = {
        "init_s": init_s,
        "load_gib": grab(r"Model loading took ([\d.]+) GiB"),
        "load_s": grab(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"),
        "kv_cache_gib": grab(r"Available KV cache memory: ([\d.]+) GiB"),
        "kv_cache_tokens": grab(r"GPU KV cache size: ([\d,]+) tokens", int),
        "error": err,
    }
    return llm, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="7b", choices=list(configs.BASE))
    ap.add_argument("--precision", default="bf16", choices=list(configs.VLLM_PRECISIONS))
    ap.add_argument("--batch-sizes", type=ints, default=ints("1,2,4,8,16,32,64,128,256"))
    ap.add_argument("--input-lens", type=ints, default=ints("128,512,1024,2048"))
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--max-prompt-tokens", type=int, default=131072, help="skip cells with B*L above this")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-batched-tokens", type=int, default=16384)
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--out", type=Path, default=RESULTS / "vllm_sweep.jsonl")
    args = ap.parse_args()

    from transformers import AutoConfig
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    model_id, quant = configs.vllm_spec(args.size, args.precision)
    llm, info = build_engine(model_id, quant, args)
    cfg = AutoConfig.from_pretrained(configs.BASE[args.size])
    pb = memory.param_breakdown(cfg)
    common = {"backend": "vllm", "size": args.size, "precision": args.precision, "tag": args.tag}
    append_jsonl(RESULTS / "vllm_load.jsonl", {
        **common, "model_id": model_id, "quantization": quant, **info,
        "gpu_util": args.gpu_util, "max_model_len": args.max_model_len,
        "theory_bytes": memory.weight_bytes(pb, args.precision),
        "kv_bytes_per_token": memory.kv_bytes_per_token(cfg), "env": env_info()})
    if llm is None:
        print(f"engine failed: {info['error']}")
        return

    n = args.output_len
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    spn = SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n, ignore_eos=True)

    def prompts(B, L, seed):
        ids = workloads.synthetic_ids(B, L, seed=seed)
        return [TokensPrompt(prompt_token_ids=row.tolist()) for row in ids]

    def run(p, sp):
        t0 = time.perf_counter()
        outs = llm.generate(p, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        assert all(len(o.outputs[0].token_ids) == sp.max_tokens for o in outs)
        return dt

    for L in args.input_lens:
        for B in args.batch_sizes:
            if B * L > args.max_prompt_tokens:
                continue
            cell = {**common, "batch": B, "input_len": L, "output_len": n}
            for r in range(args.warmup):
                p = prompts(B, L, seed=7919 * r + 31 * B + L)
                run(p, sp1)
                run(p, spn)
            sets = [prompts(B, L, seed=104729 + 7919 * r + 31 * B + L) for r in range(args.repeats)]
            t1 = [run(p, sp1) for p in sets]
            with GpuMonitor() as mon:
                tn = [run(p, spn) for p in sets]
            m = mon.result
            for i, (a, b) in enumerate(zip(t1, tn)):
                tpot = (b - a) / (n - 1)
                append_jsonl(args.out, {
                    **cell, "status": "ok", "repeat": i, "ttft_s": a, "tpot_s": tpot, "e2e_s": b,
                    "prefill_tok_s": B * L / a, "decode_tok_s": B / tpot, "output_tok_s": B * n / b,
                    "energy_j_per_out_token": m["energy_j"] / (B * n * len(tn)),
                    "block_energy_j": m["energy_j"], "avg_power_w": m["avg_power_w"],
                    "avg_sm_clock_mhz": m["avg_sm_clock_mhz"], "temp_end_c": m["temp_end_c"]})
            print(f"ok   B={B:<4d} L={L:<5d} ttft={sum(t1) / len(t1) * 1e3:8.1f}ms "
                  f"tpot={(sum(tn) - sum(t1)) / len(tn) / (n - 1) * 1e3:7.2f}ms "
                  f"tok/s={B * n * len(tn) / sum(tn):9.1f}", flush=True)


if __name__ == "__main__":
    main()
