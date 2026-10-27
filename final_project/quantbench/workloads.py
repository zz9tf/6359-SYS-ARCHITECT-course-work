"""Workloads: fixed-length synthetic prompts and a fixed-seed ShareGPT sample."""
import json
import random
from pathlib import Path

import torch


def synthetic_ids(batch: int, length: int, seed: int, low: int = 100, high: int = 150_000):
    """Uniform random token ids (no special tokens). Content does not affect dense-model speed."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(low, high, (batch, length), generator=g)
    return ids


def sample_sharegpt(path, tokenizer, n: int = 300, seed: int = 0, max_prompt: int = 1024,
                    max_total: int = 2048, min_len: int = 4):
    """Sample ``n`` (prompt, reference completion) pairs like vLLM's ShareGPT benchmark does.

    First human turn -> prompt, first gpt turn -> reference completion; pairs with fewer than
    ``min_len`` tokens on either side, a prompt longer than ``max_prompt`` tokens or a total longer
    than ``max_total`` tokens are skipped.
    """
    data = json.loads(Path(path).read_text())
    data = [d for d in data if len(d.get("conversations", [])) >= 2]
    random.Random(seed).shuffle(data)
    out = []
    for d in data:
        conv = d["conversations"]
        if conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
            continue
        prompt, completion = conv[0]["value"], conv[1]["value"]
        pl = len(tokenizer(prompt).input_ids)
        ol = len(tokenizer(completion).input_ids)
        if pl < min_len or ol < min_len or pl > max_prompt or pl + ol > max_total:
            continue
        out.append({"id": d["id"], "prompt": prompt, "completion": completion,
                    "prompt_len": pl, "output_len": ol})
        if len(out) == n:
            break
    return out


def write_sharegpt_subset(samples, path):
    """Write the sample back in ShareGPT format so vLLM's benchmark replays the identical set."""
    conv = [{"id": s["id"], "conversations": [{"from": "human", "value": s["prompt"]},
                                              {"from": "gpt", "value": s["completion"]}]}
            for s in samples]
    Path(path).write_text(json.dumps(conv))
