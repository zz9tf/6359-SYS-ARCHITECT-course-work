#!/usr/bin/env python
"""Real-workload replay on the HF backend with static batching (Q3).

The 300 ShareGPT requests are processed in arrival order in batches of B (left-padded raw
prompts, as `vllm bench serve` sends them). Every request generates exactly its reference
completion length (EOS suppressed), so the work is identical across precisions and identical to
the vLLM serving benchmark run with --ignore-eos. A batch runs until its longest request is done;
per-request completion times come from the per-step timestamps.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import configs, hf  # noqa: E402
from quantbench.monitor import GpuMonitor  # noqa: E402
from quantbench.results import RESULTS, ROOT, append_jsonl  # noqa: E402

warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="7b")
    ap.add_argument("--precision", default="bf16", choices=configs.HF_PRECISIONS)
    ap.add_argument("--batch-sizes", default="16")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "sharegpt_300.json")
    ap.add_argument("--out", type=Path, default=RESULTS / "hf_sharegpt.jsonl")
    args = ap.parse_args()

    conv = json.loads(args.data.read_text())
    model, tok, _ = hf.load(configs.hf_model_id(args.size), args.precision)
    tok.padding_side = "left"
    prompts = [c["conversations"][0]["value"] for c in conv]
    out_lens = [len(tok(c["conversations"][1]["value"]).input_ids) for c in conv]

    for B in [int(b) for b in args.batch_sizes.split(",")]:
        ttfts, tpots, e2es, useful = [], [], [], 0
        t_start = time.perf_counter()
        with GpuMonitor() as mon:
            for s in range(0, len(prompts), B):
                enc = tok(prompts[s:s + B], return_tensors="pt", padding=True).to("cuda")
                lens = out_lens[s:s + B]
                res = hf.timed_generate(model, enc.input_ids, enc.attention_mask, max(lens), exact=True)
                steps = res["step_times"]
                for n in lens:
                    done = steps[n - 1]
                    ttfts.append(res["ttft_s"])
                    e2es.append(done)
                    if n > 1:
                        tpots.append((done - res["ttft_s"]) / (n - 1))
                useful += sum(lens)
                print(f"  batch {s // B:3d}: {len(lens)} reqs, {max(lens):5d} steps, {res['e2e_s']:7.2f}s", flush=True)
        wall = time.perf_counter() - t_start
        m = mon.result
        row = {"backend": "hf", "size": args.size, "precision": args.precision, "batch": B,
               "num_requests": len(prompts), "duration_s": wall, "output_tokens": useful,
               "output_throughput": useful / wall, "request_throughput": len(prompts) / wall,
               "mean_ttft_ms": 1e3 * sum(ttfts) / len(ttfts), "p50_ttft_ms": 1e3 * pct(ttfts, .5),
               "p99_ttft_ms": 1e3 * pct(ttfts, .99), "mean_tpot_ms": 1e3 * sum(tpots) / len(tpots),
               "p50_tpot_ms": 1e3 * pct(tpots, .5), "p99_tpot_ms": 1e3 * pct(tpots, .99),
               "mean_e2el_ms": 1e3 * sum(e2es) / len(e2es), "p99_e2el_ms": 1e3 * pct(e2es, .99),
               "energy_j": m["energy_j"], "energy_j_per_token": m["energy_j"] / useful,
               "avg_power_w": m["avg_power_w"]}
        append_jsonl(args.out, row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
