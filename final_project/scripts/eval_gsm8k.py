#!/usr/bin/env python
"""GSM8K accuracy (Q4): zero-shot chain-of-thought, greedy decoding, full 1,319-question test set.

Prompt = chat template + question + Qwen's recommended math instruction ("reason step by step,
put the final answer within \\boxed{}"). The prediction is the last \\boxed{...} (fallback: the
last number in the answer) and is scored by numeric equality with the reference after "####".
"""
import argparse
import os
import re
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
INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
NUM = r"-?\d[\d,]*\.?\d*"


def to_number(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip().rstrip(".")
    m = re.findall(NUM, s)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def extract(text):
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    if boxed:
        v = to_number(boxed[-1])
        if v is not None:
            return v
    return to_number(text)


def gold(answer):
    return float(answer.split("####")[-1].strip().replace(",", ""))


@torch.inference_mode()
def generate_hf(model_id, precision, prompts, max_new, batch):
    from quantbench import hf
    model, tok, _ = hf.load(model_id, precision)
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def generate_vllm(model_id, quant, prompts, max_new):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, quantization=quant, gpu_memory_utilization=0.85, max_model_len=2048,
              enable_prefix_caching=False, seed=0)
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new), use_tqdm=False)
    return [r.outputs[0].text for r in res]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    ap.add_argument("--size", default="7b")
    ap.add_argument("--precision", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=RESULTS / "quality_gsm8k.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(args.limit))
    tok = AutoTokenizer.from_pretrained(configs.BASE[args.size])
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + INSTRUCTION}],
                                       tokenize=False, add_generation_prompt=True) for q in ds["question"]]
    t0 = time.perf_counter()
    if args.backend == "hf":
        texts = generate_hf(configs.hf_model_id(args.size), args.precision, prompts, args.max_new_tokens, args.batch)
    else:
        model_id, quant = configs.vllm_spec(args.size, args.precision)
        texts = generate_vllm(model_id, quant, prompts, args.max_new_tokens)
    elapsed = time.perf_counter() - t0

    preds = [extract(t) for t in texts]
    golds = [gold(a) for a in ds["answer"]]
    correct = [p is not None and abs(p - g) < 1e-4 for p, g in zip(preds, golds)]
    n_tok = [len(tok(t).input_ids) for t in texts]
    detail = RESULTS / "gsm8k_outputs" / f"{args.backend}_{args.size}_{args.precision}.jsonl"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.unlink(missing_ok=True)
    for i, (t, p, g, c) in enumerate(zip(texts, preds, golds, correct)):
        append_jsonl(detail, {"idx": i, "pred": p, "gold": g, "correct": c, "output": t})
    row = {"backend": args.backend, "size": args.size, "precision": args.precision, "n": len(ds),
           "accuracy": sum(correct) / len(correct), "n_correct": sum(correct),
           "no_answer": sum(p is None for p in preds), "mean_output_tokens": sum(n_tok) / len(n_tok),
           "time_s": elapsed}
    append_jsonl(args.out, row)
    print(row)
    if args.backend == "vllm":
        # vLLM's engine teardown can abort the interpreter; the result is already on disk.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
