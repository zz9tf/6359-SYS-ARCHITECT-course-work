#!/usr/bin/env python
"""WikiText-2 perplexity (Q4), same protocol as the GPTQ/AWQ papers.

The raw test split is joined with blank lines, tokenized once and cut into non-overlapping
2048-token windows; PPL = exp(mean next-token NLL over all predicted positions). The ``hf``
backend runs a forward pass per window; the ``vllm`` backend requests the log-probability of every
prompt token, which lets us score checkpoints that only vLLM can execute (FP8, GPTQ, AWQ).
"""
import argparse
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
    total, count = 0.0, 0
    for w in windows:
        x = torch.tensor([w], device="cuda")
        logits = model(input_ids=x).logits[0, :-1].float()
        total += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum").item()
        count += len(w) - 1
    return total, count


def nll_vllm(model_id, quant, windows, seqlen):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=model_id, quantization=quant, gpu_memory_utilization=0.85, max_model_len=seqlen + 16,
              enable_prefix_caching=False, seed=0, max_num_seqs=8)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in windows], sp, use_tqdm=False)
    total, count = 0.0, 0
    for w, o in zip(windows, outs):
        for pos in range(1, len(w)):
            total -= o.prompt_logprobs[pos][w[pos]].logprob
            count += 1
    return total, count


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    ap.add_argument("--size", default="7b")
    ap.add_argument("--precision", required=True)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=RESULTS / "quality_ppl.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(configs.BASE[args.size])
    windows, n_tokens = wikitext_windows(tok, args.seqlen)
    t0 = time.perf_counter()
    if args.backend == "hf":
        total, count = nll_hf(configs.hf_model_id(args.size), args.precision, windows)
    else:
        model_id, quant = configs.vllm_spec(args.size, args.precision)
        total, count = nll_vllm(model_id, quant, windows, args.seqlen)
    ppl = math.exp(total / count)
    row = {"backend": args.backend, "size": args.size, "precision": args.precision, "seqlen": args.seqlen,
           "windows": len(windows), "tokens_scored": count, "corpus_tokens": n_tokens,
           "ppl": ppl, "mean_nll": total / count, "time_s": time.perf_counter() - t0}
    append_jsonl(args.out, row)
    print(row)
    if args.backend == "vllm":
        # vLLM's engine teardown can abort the interpreter; the result is already on disk.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
