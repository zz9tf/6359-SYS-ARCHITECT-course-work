#!/usr/bin/env python
"""MMLU accuracy (Q4): zero-shot, full 14,042-question test set (57 subjects), chat-formatted.

Each question and its four options go into one user turn that asks for the letter of the correct
option; the prediction is the letter among A--D that the model scores highest as the first token of
its answer (HF: argmax of the four letter logits; vLLM: greedy decoding restricted to the four
letter tokens, which selects the same argmax). Per-question correctness is saved for paired tests.
"""
import argparse
import json
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
LETTERS = ["A", "B", "C", "D"]


def build_prompt(tok, ex):
    subject = ex["subject"].replace("_", " ")
    options = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, ex["choices"]))
    user = (f"The following is a multiple choice question about {subject}. Answer with the letter of the "
            f"correct option only.\n\n{ex['question']}\n{options}\nAnswer:")
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.inference_mode()
def predict_hf(model_id, precision, prompts, letter_ids, batch):
    from quantbench import hf
    model, tok, _ = hf.load(model_id, precision)
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    preds = [None] * len(prompts)
    ids = torch.tensor(letter_ids, device="cuda")
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)   # correct positions under left padding
        logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos,
                       logits_to_keep=1).logits[:, -1]
        choice = logits.index_select(-1, ids).argmax(-1).tolist()
        for j, i in enumerate(idx):
            preds[i] = choice[j]
    return preds


def predict_vllm(model_id, quant, prompts, letter_ids):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_id, quantization=quant, gpu_memory_utilization=0.85, max_model_len=4096,
              enable_prefix_caching=False, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=1, allowed_token_ids=letter_ids)
    res = llm.generate(prompts, sp, use_tqdm=False)
    return [letter_ids.index(r.outputs[0].token_ids[0]) for r in res]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    ap.add_argument("--size", default="7b")
    ap.add_argument("--precision", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=RESULTS / "quality_mmlu.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("cais/mmlu", "all", split="test")
    if args.limit:
        ds = ds.select(range(args.limit))
    tok = AutoTokenizer.from_pretrained(configs.BASE[args.size])
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    prompts = [build_prompt(tok, ex) for ex in ds]
    t0 = time.perf_counter()
    if args.backend == "hf":
        preds = predict_hf(configs.hf_model_id(args.size), args.precision, prompts, letter_ids, args.batch)
    else:
        model_id, quant = configs.vllm_spec(args.size, args.precision)
        preds = predict_vllm(model_id, quant, prompts, letter_ids)
    elapsed = time.perf_counter() - t0

    golds = list(ds["answer"])
    correct = [p == g for p, g in zip(preds, golds)]
    detail = RESULTS / "mmlu_outputs" / f"{args.backend}_{args.size}_{args.precision}.jsonl"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.unlink(missing_ok=True)
    with detail.open("w") as f:
        for i, (p, g, c, subj) in enumerate(zip(preds, golds, correct, ds["subject"])):
            f.write(json.dumps({"idx": i, "subject": subj, "pred": p, "gold": g, "correct": c}) + "\n")
    row = {"backend": args.backend, "size": args.size, "precision": args.precision, "n": len(ds),
           "accuracy": sum(correct) / len(correct), "n_correct": sum(correct), "time_s": elapsed}
    append_jsonl(args.out, row)
    print(row)
    if args.backend == "vllm":
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
