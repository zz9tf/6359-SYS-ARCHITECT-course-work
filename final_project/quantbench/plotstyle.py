"""Shared figure style.

Colour encodes the weight format and follows it across every figure (validated as a set:
all-pairs normal-vision dE >= 19, CVD dE >= 6.9, contrast >= 3:1 on white). Because the
amber/green pair sits in the CVD warning band, every format also has its own marker, and lines
are direct-labelled. Line style encodes the backend: solid = vLLM (fused kernels),
dashed = Hugging Face + bitsandbytes.
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

BLUE, GREEN, MAGENTA, AMBER = "#2a78d6", "#008300", "#d55181", "#c98500"
COLOR = {"bf16": BLUE, "fp8": GREEN, "int8": MAGENTA, "gptq-int8": MAGENTA,
         "nf4": AMBER, "awq-int4": AMBER, "gptq-int4": AMBER}
MARKER = {"bf16": "o", "fp8": "D", "int8": "s", "gptq-int8": "s", "nf4": "^", "awq-int4": "^", "gptq-int4": "v"}
LINESTYLE = {"vllm": "-", "hf": (0, (4, 2))}

LABEL = {
    ("hf", "bf16"): "HF BF16",
    ("hf", "int8"): "HF INT8 (bnb LLM.int8)",
    ("hf", "nf4"): "HF NF4 (bnb)",
    ("vllm", "bf16"): "vLLM BF16",
    ("vllm", "fp8"): "vLLM FP8 W8A8",
    ("vllm", "gptq-int8"): "vLLM INT8 W8A16 (GPTQ)",
    ("vllm", "awq-int4"): "vLLM INT4 W4A16 (AWQ)",
    ("vllm", "gptq-int4"): "vLLM INT4 W4A16 (GPTQ)",
}
SHORT = {
    ("hf", "bf16"): "HF BF16", ("hf", "int8"): "HF INT8", ("hf", "nf4"): "HF NF4",
    ("vllm", "bf16"): "vLLM BF16", ("vllm", "fp8"): "vLLM FP8", ("vllm", "gptq-int8"): "vLLM INT8",
    ("vllm", "awq-int4"): "vLLM AWQ-4", ("vllm", "gptq-int4"): "vLLM GPTQ-4",
}

# Diverging map for speedup (log2 scale): red = slower than BF16, neutral grey = parity, blue = faster.
DIVERGING = LinearSegmentedColormap.from_list(
    "speedup", ["#8f1d1d", "#e34948", "#f6c4c3", "#f0efec", "#b7d3f6", "#2a78d6", "#0d366b"])


def setup(scale: float = 1.0):
    """scale > 1 enlarges every font (slide versions of the figures)."""
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
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "lines.linewidth": 1.6, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8 * scale, "figure.dpi": 110, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42,
    })


def style(backend, precision):
    """Matplotlib kwargs for one (backend, precision) series."""
    hollow = backend == "hf"
    c = COLOR[precision]
    return dict(color=c, marker=MARKER[precision], linestyle=LINESTYLE[backend],
                markerfacecolor="white" if hollow else c, markeredgecolor=c, markeredgewidth=1.2,
                label=LABEL[(backend, precision)])
