#!/usr/bin/env python
"""Draw the fixed-seed 300-conversation ShareGPT sample used by every real-workload experiment.

Writes data/sharegpt_300.json (ShareGPT format, replayed by `vllm bench serve` and hf_sharegpt.py)
and results/sharegpt_300_lengths.jsonl (token lengths, for the workload figure).
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import configs, workloads  # noqa: E402
from quantbench.results import RESULTS, ROOT, append_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered",
                          "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")
    tok = AutoTokenizer.from_pretrained(configs.BASE["7b"])
    samples = workloads.sample_sharegpt(src, tok, n=args.n, seed=args.seed)
    (ROOT / "data").mkdir(exist_ok=True)
    workloads.write_sharegpt_subset(samples, ROOT / "data" / f"sharegpt_{args.n}.json")
    out = RESULTS / f"sharegpt_{args.n}_lengths.jsonl"
    out.unlink(missing_ok=True)
    for s in samples:
        append_jsonl(out, {"id": s["id"], "prompt_len": s["prompt_len"], "output_len": s["output_len"]})
    pl = sorted(s["prompt_len"] for s in samples)
    ol = sorted(s["output_len"] for s in samples)
    print(f"{len(samples)} samples | prompt len median {pl[len(pl) // 2]} mean {sum(pl) / len(pl):.0f} | "
          f"output len median {ol[len(ol) // 2]} mean {sum(ol) / len(ol):.0f}")


if __name__ == "__main__":
    main()
