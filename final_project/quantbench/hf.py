"""Hugging Face Transformers backend: loading with bitsandbytes and instrumented generation."""
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer

from .monitor import device_memory_used

# bitsandbytes logs one line per LLM.int8() matmul call (bf16 inputs are cast to fp16).
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


def bnb_config(precision: str):
    if precision == "bf16":
        return None
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)  # LLM.int8(), outlier threshold 6.0
    if precision == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
    raise ValueError(f"unknown HF precision {precision!r}")


def load(model_id: str, precision: str):
    """Load model + tokenizer; returns (model, tokenizer, load_info)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    nvml_before = device_memory_used()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0", quantization_config=bnb_config(precision))
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    model.eval()
    # Plain greedy decoding: neutralise the sampling defaults shipped in generation_config.json.
    gc = model.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k, gc.repetition_penalty = False, None, None, None, 1.0
    gc.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    info = {
        "load_time_s": load_s,
        "weights_bytes": torch.cuda.memory_allocated() - base_alloc,
        "load_peak_bytes": torch.cuda.max_memory_allocated() - base_alloc,
        "hf_footprint_bytes": model.get_memory_footprint(),
        "nvml_after_load_bytes": device_memory_used(),
        "nvml_before_load_bytes": nvml_before,
    }
    return model, tok, info


class StepTimer(BaseStreamer):
    """Timestamps every token emission of ``generate``.

    ``generate`` first emits the prompt (before the prefill), then one emission per decoding step.
    Each emission calls ``.cpu()`` on the new tokens, so the host timestamp is taken after the GPU
    has produced that token.
    """

    def __init__(self):
        self.stamps = []

    def put(self, value):
        self.stamps.append(time.perf_counter())

    def end(self):
        pass


@torch.inference_mode()
def timed_generate(model, input_ids, attention_mask, new_tokens: int, exact: bool = True):
    """Greedy generation of ``new_tokens`` tokens with per-step timestamps.

    Returns a dict with ttft / tpot / e2e (seconds), the per-step timestamps relative to start,
    and the generated ids. With ``exact`` the EOS token is suppressed so every sequence produces
    exactly ``new_tokens`` tokens (identical work for every precision).
    """
    timer = StepTimer()
    kwargs = dict(max_new_tokens=new_tokens, do_sample=False, streamer=timer, use_cache=True)
    if exact:
        kwargs["min_new_tokens"] = new_tokens
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    steps = [s - t0 for s in timer.stamps[1:]]       # stamps[0] is the prompt emission
    n_new = out.shape[1] - input_ids.shape[1]
    ttft = steps[0]
    e2e = t_end - t0
    tpot = (e2e - ttft) / (n_new - 1) if n_new > 1 else float("nan")
    return {"ttft_s": ttft, "tpot_s": tpot, "e2e_s": e2e, "n_new": n_new, "step_times": steps,
            "output_ids": out[:, input_ids.shape[1]:]}


def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_stats(base_alloc: int = 0) -> dict:
    return {
        "torch_peak_alloc_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch_peak_over_weights_bytes": torch.cuda.max_memory_allocated() - base_alloc,
    }
