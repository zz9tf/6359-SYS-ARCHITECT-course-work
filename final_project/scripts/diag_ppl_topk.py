#!/usr/bin/env python
"""Diagnose a perplexity outlier: is the token ranking wrong, or only the probabilities?

Scores text with vLLM and records, per configuration, the mean NLL of the reference token, the
top-1 next-token accuracy (reference token ranked first) and the mean probability the model gives
its own top-1 token. A checkpoint whose ranking is intact but whose
probabilities are flattened keeps its top-1 accuracy (and greedy-decoding quality) while its
perplexity rises. ``--corpus wikitext`` scores the first WikiText-2 windows (raw text, as in the
perplexity protocol); ``--corpus gsm8k-chat`` scores human-written GSM8K solutions as the assistant
turn after the chat-formatted question, i.e. text in the format the instruct checkpoints serve.

  python scripts/diag_ppl_topk.py --size 14b --precision gptq-int4 --corpus gsm8k-chat --windows 400
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_gsm8k import INSTRUCTION  # noqa: E402
from eval_ppl import wikitext_windows  # noqa: E402
from quantbench import configs  # noqa: E402
from quantbench.results import RESULTS, append_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="14b")
    ap.add_argument("--precision", required=True)
    ap.add_argument("--corpus", choices=["wikitext", "gsm8k-chat"], default="wikitext")
    ap.add_argument("--windows", type=int, default=24, help="WikiText windows / GSM8K questions to score")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=RESULTS / "ppl_topk_diag.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(configs.BASE[args.size])
    if args.corpus == "wikitext":
        seqs, _ = wikitext_windows(tok, args.seqlen)
        seqs = [(w, 1) for w in seqs[:args.windows]]          # (token ids, first scored position)
    else:
        import re
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.windows))
        seqs = []
        for ex in ds:
            prompt = tok.apply_chat_template([{"role": "user", "content": ex["question"] + INSTRUCTION}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
            answer = re.sub(r"<<[^>]*>>", "", ex["answer"]).replace("####", "The answer is")
            p_ids = tok(prompt, add_special_tokens=False).input_ids
            seqs.append((p_ids + tok(answer, add_special_tokens=False).input_ids, len(p_ids)))
    model_id, quant = configs.vllm_spec(args.size, args.precision)
    llm = LLM(model=model_id, quantization=quant, gpu_memory_utilization=0.85, max_model_len=args.seqlen + 16,
              enable_prefix_caching=False, seed=0, max_num_seqs=8, max_logprobs=5)
    t0 = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w, _ in seqs],
                        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5), use_tqdm=False)
    nll = hit = p_top = n = 0
    for (w, start), o in zip(seqs, outs):
        for pos in range(start, len(w)):
            lp = o.prompt_logprobs[pos]
            ref = lp[w[pos]]
            top = max(lp.values(), key=lambda x: x.logprob)
            nll -= ref.logprob
            hit += ref.rank == 1
            p_top += math.exp(top.logprob)
            n += 1
    row = {"size": args.size, "precision": args.precision, "corpus": args.corpus, "sequences": len(seqs), "tokens": n,
           "mean_nll": nll / n, "ppl": math.exp(nll / n), "top1_acc": hit / n, "mean_top1_prob": p_top / n,
           "time_s": time.perf_counter() - t0}
    append_jsonl(args.out, row)
    print(row)
    sys.stdout.flush()
    try:  # stop the engine process explicitly; os._exit alone can leave it holding GPU memory
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
