#!/usr/bin/env python
"""WikiText-2 perplexity (Q4), same protocol as the GPTQ/AWQ papers.

The raw test split is joined with blank lines, tokenized once and cut into non-overlapping
2048-token windows; PPL = exp(mean next-token NLL over all predicted positions). The ``hf``
backend runs a forward pass per window; the ``vllm`` backend requests the log-probability of every
prompt token, which lets us score checkpoints that only vLLM can execute (FP8, GPTQ, AWQ).
"""
import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import configs  # noqa: E402
from quantbench.results import RESULTS, append_jsonl  # noqa: E402

warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def wikitext_windows(tokenizer, seqlen):
    from datasets import load_dataset
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(test["text"])).input_ids
    n = len(ids) // seqlen
    return [ids[i * seqlen:(i + 1) * seqlen] for i in range(n)], len(ids)


@torch.inference_mode()
def nll_hf(model_id, precision, windows):
    from quantbench import hf
    model, _, _ = hf.load(model_id, precision)
    per_token = []
    for w in windows:
        x = torch.tensor([w], device="cuda")
        h = model.model(input_ids=x).last_hidden_state[0, :-1]
        nll = []
        for c in range(0, h.shape[0], 256):   # LM head in chunks: full 2048 x 152K logits do not fit next to 14B BF16
            logits = model.lm_head(h[c:c + 256]).float()
            nll.append(torch.nn.functional.cross_entropy(logits, x[0, 1 + c:1 + c + 256], reduction="none"))
        per_token.append(torch.cat(nll).cpu().tolist())
    return per_token


def nll_vllm(model_id, quant, windows, seqlen, dtype="auto"):
    """Per-token NLL of every window (list of lists)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=model_id, quantization=quant, gpu_memory_utilization=0.85, max_model_len=seqlen + 16,
              enable_prefix_caching=False, seed=0, max_num_seqs=8, dtype=dtype)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in windows], sp, use_tqdm=False)
    return [[-o.prompt_logprobs[pos][w[pos]].logprob for pos in range(1, len(w))] for w, o in zip(windows, outs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    ap.add_argument("--size", default="7b")
    ap.add_argument("--precision", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--dtype", default="auto", help="vLLM activation dtype (checkpoint default if auto)")
    ap.add_argument("--quantization", default=None, help="override vLLM's quantization method (e.g. gptq instead of gptq_marlin)")
    ap.add_argument("--out", type=Path, default=RESULTS / "quality_ppl.jsonl")
    ap.add_argument("--windows-out", type=Path, default=RESULTS / "ppl_windows",
                    help="directory for the per-window NLL of each run")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(configs.BASE[args.size])
    windows, n_tokens = wikitext_windows(tok, args.seqlen)
    t0 = time.perf_counter()
    if args.backend == "hf":
        per_token = nll_hf(configs.hf_model_id(args.size), args.precision, windows)
    else:
        model_id, quant = configs.vllm_spec(args.size, args.precision)
        per_token = nll_vllm(model_id, args.quantization or quant, windows, args.seqlen, args.dtype)
    total, count = sum(map(sum, per_token)), sum(map(len, per_token))
    ppl = math.exp(total / count)
    row = {"backend": args.backend, "size": args.size, "precision": args.precision, "seqlen": args.seqlen,
           "windows": len(windows), "tokens_scored": count, "corpus_tokens": n_tokens,
           "dtype": args.dtype, "quantization": args.quantization, "ppl": ppl, "mean_nll": total / count, "time_s": time.perf_counter() - t0}
    append_jsonl(args.out, row)
    args.windows_out.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.backend}_{args.size}_{args.precision}" + ("" if args.dtype == "auto" else f"_{args.dtype}")
           + (f"_{args.quantization}" if args.quantization else ""))
    bucket = 128   # mean NLL by position in the window shows whether errors grow with context length
    by_pos = [sum(t[i] for t in per_token for i in range(b, min(b + bucket, len(t)))) /
              sum(max(0, min(b + bucket, len(t)) - b) for t in per_token) for b in range(0, args.seqlen - 1, bucket)]
    (args.windows_out / f"{tag}.json").write_text(json.dumps(
        {"window_nll": [sum(t) / len(t) for t in per_token], "position_bucket": bucket, "position_nll": by_pos}))
    print(row)
    if args.backend == "vllm":
        # vLLM's engine teardown can abort the interpreter; the result is already on disk.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
