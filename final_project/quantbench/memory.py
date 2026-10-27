"""Analytic memory model: parameter breakdown, theoretical weight bytes and KV-cache size.

Only the linear layers inside the decoder blocks are quantized by every method studied here.
The token embedding, the LM head, the norms and the attention biases stay in 16-bit, which is
why the real compression ratio is always below the nominal 16/b.
"""
from dataclasses import dataclass

# Bytes stored per quantized weight element, including the quantization metadata.
#   bnb NF4      : 4-bit codes + one fp32 absmax per 64-element block (no double quantization)
#   bnb INT8     : 8-bit codes + one fp32 scale per output row (negligible)
#   GPTQ/AWQ g128: packed codes + fp16 scale + packed zero point per 128-element group
#   FP8          : 1 byte + one fp32 scale per tensor (negligible)
BYTES_PER_QWEIGHT = {
    "bf16": 2.0,
    "int8": 1.0,
    "nf4": 0.5 + 4 / 64,
    "fp8": 1.0,
    "gptq-int8": 1.0 + 2 / 128 + 1 / 128,
    "awq-int4": 0.5 + 2 / 128 + 0.5 / 128,
    "gptq-int4": 0.5 + 2 / 128 + 0.5 / 128,
}
IDEAL_BYTES = {"bf16": 2.0, "int8": 1.0, "nf4": 0.5, "fp8": 1.0, "gptq-int8": 1.0, "awq-int4": 0.5, "gptq-int4": 0.5}


@dataclass
class ParamBreakdown:
    linear: int      # weights of q/k/v/o/gate/up/down in every decoder layer (quantizable)
    embed: int       # token embedding
    lm_head: int     # output projection (0 when tied to the embedding)
    other: int       # norms + attention biases

    @property
    def total(self) -> int:
        return self.linear + self.embed + self.lm_head + self.other


def param_breakdown(cfg) -> ParamBreakdown:
    h, inter, n_layers = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    head_dim = getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads
    q_dim = cfg.num_attention_heads * head_dim
    kv_dim = cfg.num_key_value_heads * head_dim
    per_layer_linear = h * q_dim + 2 * h * kv_dim + q_dim * h + 3 * h * inter
    per_layer_other = (q_dim + 2 * kv_dim) + 2 * h          # qkv biases + two RMSNorms
    embed = cfg.vocab_size * h
    lm_head = 0 if getattr(cfg, "tie_word_embeddings", False) else cfg.vocab_size * h
    return ParamBreakdown(linear=n_layers * per_layer_linear, embed=embed, lm_head=lm_head,
                          other=n_layers * per_layer_other + h)


def weight_bytes(pb: ParamBreakdown, precision: str, with_metadata: bool = True) -> float:
    per = (BYTES_PER_QWEIGHT if with_metadata else IDEAL_BYTES)[precision]
    return pb.linear * per + (pb.embed + pb.lm_head + pb.other) * 2.0


def naive_weight_bytes(pb: ParamBreakdown, bits: int) -> float:
    """The 'everything is quantized' estimate used on the proposal slide (#params x bits / 8)."""
    return pb.total * bits / 8


def kv_bytes_per_token(cfg, dtype_bytes: int = 2) -> int:
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * dtype_bytes
