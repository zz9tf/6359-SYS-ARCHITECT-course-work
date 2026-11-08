"""Shared figure style.

Colour and marker encode the quantization type, so a reader can map every series to what is
quantized: W8A8 (8-bit weights and activations: FP8, LLM.int8()) green diamonds, W8A16 (8-bit
weights: GPTQ-INT8) magenta squares, W4A16 (4-bit weights: AWQ, GPTQ-INT4, NF4) amber triangles,
and the BF16 baseline grey circles. The software stack is the second channel: solid lines with
filled markers are vLLM (fused kernels), dashed lines with hollow markers are Hugging Face +
bitsandbytes. The colours were validated as a set (all-pairs normal-vision dE >= 19, CVD dE >= 6.9,
contrast >= 3:1 on white).
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

BLUE, GREEN, MAGENTA, AMBER = "#2a78d6", "#008300", "#d55181", "#c98500"
BASE = "#5f5e5a"     # BF16 baseline
TYPE = {"bf16": "BF16", "fp8": "W8A8", "int8": "W8A8", "gptq-int8": "W8A16",
        "awq-int4": "W4A16", "gptq-int4": "W4A16", "nf4": "W4A16"}
TYPE_COLOR = {"BF16": BASE, "W8A8": GREEN, "W8A16": MAGENTA, "W4A16": AMBER}
TYPE_NAME = {"W8A8": "8-bit weights + activations", "W8A16": "8-bit weights", "W4A16": "4-bit weights"}
COLOR = {p: TYPE_COLOR[t] for p, t in TYPE.items()}
MARKER = {"bf16": "o", "fp8": "D", "int8": "D", "gptq-int8": "s", "nf4": "^", "awq-int4": "^", "gptq-int4": "v"}
LINESTYLE = {"vllm": "-", "hf": (0, (4, 2))}

LABEL = {
    ("hf", "bf16"): "BF16 (HF)",
    ("hf", "int8"): "W8A8 LLM.int8() (HF)",
    ("hf", "nf4"): "W4A16 NF4 (HF)",
    ("vllm", "bf16"): "BF16 (vLLM)",
    ("vllm", "fp8"): "W8A8 FP8 (vLLM)",
    ("vllm", "gptq-int8"): "W8A16 GPTQ (vLLM)",
    ("vllm", "awq-int4"): "W4A16 AWQ (vLLM)",
    ("vllm", "gptq-int4"): "W4A16 GPTQ (vLLM)",
}
SHORT = {
    ("hf", "bf16"): "BF16 (HF)", ("hf", "int8"): "LLM.int8() (HF)", ("hf", "nf4"): "NF4 (HF)",
    ("vllm", "bf16"): "BF16 (vLLM)", ("vllm", "fp8"): "FP8 (vLLM)", ("vllm", "gptq-int8"): "GPTQ-INT8 (vLLM)",
    ("vllm", "awq-int4"): "AWQ-INT4 (vLLM)", ("vllm", "gptq-int4"): "GPTQ-INT4 (vLLM)",
}

# Diverging map for speedup (log2 scale): red = slower than BF16, neutral grey = parity, blue = faster.
DIVERGING = LinearSegmentedColormap.from_list(
    "speedup", ["#8f1d1d", "#e34948", "#f6c4c3", "#f0efec", "#b7d3f6", "#2a78d6", "#0d366b"])


MEW = 1.2   # marker edge width; thinner for the small report figures


def setup(scale: float = 1.0):
    """scale > 1 enlarges every font (slide versions); scale < 1 sizes them for the 5.5 in report column."""
    global MEW
    MEW = 1.2 if scale >= 1 else 0.9
    for f in ("/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
    mpl.rcParams.update({
        "font.family": "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans",
        "font.size": 9 * scale, "axes.titlesize": 10 * scale, "axes.titleweight": "bold",
        "axes.labelsize": 9 * scale, "xtick.labelsize": 8.5 * scale, "ytick.labelsize": 8.5 * scale,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 1.6 if scale >= 1 else 1.2, "lines.markersize": 5 if scale >= 1 else 3.6,
        "legend.frameon": False, "legend.fontsize": 8 * scale, "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42,
    })


def style(backend, precision):
    """Matplotlib kwargs for one (backend, precision) series."""
    hollow = backend == "hf"
    c = COLOR[precision]
    return dict(color=c, marker=MARKER[precision], linestyle=LINESTYLE[backend],
                markerfacecolor="white" if hollow else c, markeredgecolor=c, markeredgewidth=MEW,
                label=LABEL[(backend, precision)])
