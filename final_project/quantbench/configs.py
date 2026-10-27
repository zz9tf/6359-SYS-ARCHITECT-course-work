"""Registry of models and precisions used by every experiment.

Two backends are studied:

* ``hf``   - Hugging Face Transformers (eager PyTorch). The BF16 checkpoint is loaded and
             bitsandbytes quantizes the linear layers on the fly (LLM.int8() or NF4).
* ``vllm`` - vLLM serving engine. Pre-quantized checkpoints published by the Qwen team are
             executed with fused low-bit kernels (Marlin for GPTQ/AWQ, CUTLASS for FP8).
"""

BASE = {
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
}

HF_PRECISIONS = ["bf16", "int8", "nf4"]

# precision -> (checkpoint suffix, vLLM ``quantization`` argument)
VLLM_PRECISIONS = {
    "bf16": ("", None),
    "fp8": ("", "fp8"),              # W8A8, weights quantized at load time, dynamic activation scales
    "gptq-int8": ("-GPTQ-Int8", None),  # W8A16, gptq_marlin kernel
    "awq-int4": ("-AWQ", None),         # W4A16 g128 with zero points, awq_marlin kernel
    "gptq-int4": ("-GPTQ-Int4", None),  # W4A16 g128 symmetric, gptq_marlin kernel
}

LABELS = {
    "bf16": "BF16",
    "int8": "INT8 (bnb LLM.int8)",
    "nf4": "NF4 (bnb 4-bit)",
    "fp8": "FP8 W8A8",
    "gptq-int8": "GPTQ INT8 (W8A16)",
    "awq-int4": "AWQ INT4 (W4A16)",
    "gptq-int4": "GPTQ INT4 (W4A16)",
}

WEIGHT_BITS = {"bf16": 16, "int8": 8, "nf4": 4, "fp8": 8, "gptq-int8": 8, "awq-int4": 4, "gptq-int4": 4}


def hf_model_id(size: str) -> str:
    return BASE[size]


def vllm_spec(size: str, precision: str):
    """Return (checkpoint id, quantization argument) for a vLLM configuration."""
    suffix, quant = VLLM_PRECISIONS[precision]
    return BASE[size] + suffix, quant
