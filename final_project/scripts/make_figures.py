#!/usr/bin/env python
"""Build every figure of the report from results/*.jsonl (PDF for LaTeX + PNG for slides)."""
import math
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never probe for a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import analysis as A  # noqa: E402
from quantbench import memory as M  # noqa: E402
from quantbench import plotstyle as S  # noqa: E402
from quantbench.results import FIGURES  # noqa: E402

HF_P = ["bf16", "int8", "nf4"]
VL_P = ["bf16", "fp8", "gptq-int8", "awq-int4"]
VL_ALL = ["bf16", "fp8", "gptq-int8", "awq-int4", "gptq-int4"]
SIZE_LABEL = {"3b": "Qwen2.5-3B", "7b": "Qwen2.5-7B", "14b": "Qwen2.5-14B"}


SLIDES = "--slides" in sys.argv   # larger fonts, PNG only, written to figures/slides/
FS = 1.25 if SLIDES else 1.0       # scale for the explicit font sizes below


def save(fig, name):
    out = FIGURES / "slides" if SLIDES else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    if not SLIDES:
        fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=220 if SLIDES else 300)
    plt.close(fig)
    print(f"  wrote {out.relative_to(FIGURES.parent)}/{name}")


def label_end(ax, x, y, text, color, dx=4, dy=0, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=7.5 * FS,
                color=S.INK2, **kw)


def plain_ticks(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def tokens_label(v):
    return f"{v / 1024:g}K" if v >= 1024 else f"{v:g}"


def pow2_axis(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([str(v) for v in values])
    ax.minorticks_off()


# ----------------------------------------------------------------------------- Q1 memory
def fig_weights():
    hf, vl = A.loads()
    sizes = [s for s in ("3b", "7b", "14b") if (not hf.empty and s in set(hf["size"])) or (not vl.empty and s in set(vl["size"]))]
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.3 * len(sizes), 2.9), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        pb = A.param_breakdown(size)
        bars = []
        for p in HF_P:
            r = hf[(hf["size"] == size) & (hf.precision == p)] if not hf.empty else hf
            bars.append(("hf", p, r.weights_bytes.iloc[0] / A.GB if len(r) else np.nan))
        for p in VL_ALL:
            r = vl[(vl["size"] == size) & (vl.precision == p)] if not vl.empty else vl
            v = r.load_gib.iloc[0] * A.GIB / A.GB if len(r) and r.load_gib.notna().iloc[0] else np.nan
            bars.append(("vllm", p, v))
        x = np.array([0, 1, 2, 3.6, 4.6, 5.6, 6.6, 7.6])
        bf_hf, bf_vl = bars[0][2], bars[3][2]
        for xi, (b, p, v) in zip(x, bars):
            c = S.COLOR[p]
            if np.isnan(v):
                ax.text(xi, 0.4, "n/a" if size != "14b" or p != "bf16" else "does\nnot fit", ha="center",
                        va="bottom", fontsize=6.5 * FS, color=S.MUTED, rotation=90 if p != "bf16" else 0)
                continue
            ax.bar(xi, v, 0.78, color="white" if b == "hf" else c, edgecolor=c, linewidth=1.4,
                   hatch="////" if b == "hf" else None)
            ref = bf_hf if b == "hf" else bf_vl
            if not np.isnan(ref) and p != "bf16":
                ax.text(xi, v + 0.012 * ax.get_ylim()[1] + 0.05, f"{ref / v:.2f}×", ha="center", va="bottom",
                        fontsize=6.8 * FS, color=S.INK2)
            bits = {"bf16": 16, "int8": 8, "nf4": 4, "fp8": 8, "gptq-int8": 8, "awq-int4": 4, "gptq-int4": 4}[p]
            ax.plot([xi - 0.39, xi + 0.39], [M.naive_weight_bytes(pb, bits) / A.GB] * 2, color=S.INK, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(["BF16", "INT8", "NF4", "BF16", "FP8", "GPTQ\n8-bit", "AWQ\n4-bit", "GPTQ\n4-bit"],
                           fontsize=7 * FS, rotation=0, linespacing=0.95)
        ax.set_title(SIZE_LABEL[size])
        ax.set_ylabel("weight memory (GB)" if size == sizes[0] else "")
        ax.grid(axis="x", visible=False)
        for xc, name in ((1, "HF + bitsandbytes"), (5.6, "vLLM")):
            ax.text(xc, -0.2, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8 * FS, fontweight="bold", color=S.INK)
    handles = [Patch(facecolor="white", edgecolor=S.INK2, hatch="////", label="measured, HF + bitsandbytes"),
               Patch(facecolor=S.INK2, label="measured, vLLM"),
               Line2D([], [], color=S.INK, lw=1.2, label="naive estimate: #params × bits / 8")]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    save(fig, "q1_weight_memory")


def fig_memory_scaling(size="7b"):
    agg, oom = A.sweeps()
    d = agg[(agg.backend == "hf") & (agg["size"] == size) & (agg.output_len == 128) & (agg.tag == "grid")]
    if d.empty:
        return
    o = oom[(oom.backend == "hf") & (oom["size"] == size) & (oom.output_len == 128)]
    hf, _ = A.loads()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    L = 2048
    groups = [1, 16, 64]
    parts = [("weights", "#1c5cab"), ("KV cache", "#6da7ec"), ("activations + workspace", "#b7d3f6")]
    names = {"bf16": "BF16", "int8": "INT8", "nf4": "NF4"}
    xt, xl = [], []
    for g, B in enumerate(groups):
        for j, p in enumerate(HF_P):
            xi = g * 3.6 + j
            xt.append(xi)
            xl.append(names[p])
            r = d[(d.precision == p) & (d.batch == B) & (d.input_len == L)]
            if r.empty:
                if len(o[(o.precision == p) & (o.batch == B) & (o.input_len == L)]):
                    ax1.text(xi, 0.6, "OOM", rotation=90, ha="center", va="bottom", fontsize=7 * FS, color=S.INK2)
                continue
            r = r.iloc[0]
            vals = [r.weights / A.GB, r.kv_bytes / A.GB, (r.peak_alloc - r.weights - r.kv_bytes) / A.GB]
            bottom = 0
            for (name, c), v in zip(parts, vals):
                ax1.bar(xi, v, 0.8, bottom=bottom, color=c, edgecolor="white", linewidth=1.0)
                bottom += v
            ax1.text(xi, bottom + 0.35, f"{bottom:.0f}", ha="center", fontsize=7 * FS, color=S.INK2)
        ax1.text(g * 3.6 + 1, -0.17, f"B = {B}", transform=ax1.get_xaxis_transform(), ha="center",
                 fontsize=8.5 * FS, fontweight="bold", color=S.INK)
    cap = 33647230976 / A.GB
    ax1.axhline(cap, color=S.INK2, lw=0.8, ls=":")
    ax1.text(xt[-1] + 0.4, cap + 0.4, f"usable GPU memory {cap:.1f} GB", ha="right", va="bottom", fontsize=7 * FS, color=S.INK2)
    ax1.set_xticks(xt)
    ax1.set_xticklabels(xl, fontsize=7.5 * FS)
    ax1.set_ylim(0, cap * 1.12)
    ax1.set_ylabel("peak allocated memory (GB)")
    ax1.set_title(f"(a) Peak memory, L = {L}", loc="left")
    ax1.grid(axis="x", visible=False)
    ax1.legend(handles=[Patch(color=c, label=n) for n, c in parts], loc="upper left", fontsize=7 * FS)

    sp = A.with_speedup(agg)
    sp = sp[(sp.backend == "hf") & (sp["size"] == size) & (sp.output_len == 128) & (sp.tag == "grid")]
    for p in ["int8", "nf4"]:
        for L2, ls in [(128, (0, (1, 1.5))), (2048, "-")]:
            r = sp[(sp.precision == p) & (sp.input_len == L2)].dropna(subset=["mem_saving"]).sort_values("batch")
            if r.empty:
                continue
            st = S.style("hf", p)
            st["linestyle"] = ls
            tokens = r.batch * (L2 + 128)
            ax2.plot(tokens, r.mem_saving, **st)
            label_end(ax2, tokens.iloc[-1], r.mem_saving.iloc[-1], f"{names[p]}, L={L2}", S.COLOR[p],
                      dy=6 if L2 == 128 else -6)
    if not hf.empty:
        w = hf[hf["size"] == size].set_index("precision").weights_bytes
        for p in ["int8", "nf4"]:
            if p in w and "bf16" in w:
                ax2.axhline(w["bf16"] / w[p], color=S.COLOR[p], lw=0.9, ls=(0, (6, 3)), alpha=0.6)
                ax2.annotate(f"{names[p]} weights alone: {w['bf16'] / w[p]:.2f}×", (0.98, w["bf16"] / w[p]),
                             xycoords=ax2.get_yaxis_transform(), xytext=(0, -3), textcoords="offset points",
                             fontsize=7 * FS, color=S.INK2, ha="right", va="top")
    ax2.set_xscale("log", base=2)
    ticks = [2 ** k for k in range(7, 18, 2)]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([tokens_label(t) for t in ticks])
    ax2.set_xlabel("tokens in flight  B × (L + 128)")
    ax2.set_ylabel("peak-memory saving vs BF16 (×)")
    ax2.set_title("(b) Saving vs. tokens in flight", loc="left")
    ax2.set_xlim(right=ax2.get_xlim()[1] * 8)
    fig.tight_layout()
    save(fig, "q1_memory_scaling")


# ----------------------------------------------------------------------------- Q2 latency
def _series(ax, d, backend, precisions, x, y, scale=1.0, label_last=True, fmt=None):
    for p in precisions:
        r = d[(d.backend == backend) & (d.precision == p)].sort_values(x)
        if r.empty:
            continue
        ax.plot(r[x], r[y] * scale, **S.style(backend, p))
        if label_last:
            label_end(ax, r[x].iloc[-1], r[y].iloc[-1] * scale, S.SHORT[(backend, p)], S.COLOR[p])


def fig_latency(size="7b"):
    agg, _ = A.sweeps()
    d = agg[(agg["size"] == size) & (agg.output_len == 128) & (agg.tag == "grid")]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    a = d[d.batch == 1]
    for b, ps in (("hf", HF_P), ("vllm", VL_P)):
        _series(axes[0], a, b, ps, "input_len", "ttft", 1e3, label_last=False)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks([128, 512, 1024, 2048])
    axes[0].set_xticklabels(["128", "512", "1024", "2048"])
    axes[0].set_xlabel("prompt length L (tokens)")
    axes[0].set_ylabel("TTFT (ms)")
    plain_ticks(axes[0])
    axes[0].set_title("(a) Time to first token, B = 1")
    axes[0].set_xlim(100, 2048 * 1.25)

    b = d[d.input_len == 512]
    for bk, ps in (("hf", HF_P), ("vllm", VL_P)):
        _series(axes[1], b, bk, ps, "batch", "tpot", 1e3, label_last=False)
    pow2_axis(axes[1], [1, 4, 16, 64, 256])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("batch size B")
    axes[1].set_ylabel("TPOT (ms per output token)")
    plain_ticks(axes[1])
    axes[1].set_title("(b) Decode latency, L = 512")
    axes[1].set_xlim(0.8, 256 * 1.3)

    for bk, ps in (("hf", HF_P), ("vllm", VL_P)):
        _series(axes[2], b, bk, ps, "batch", "out_tok_s", label_last=False)
    pow2_axis(axes[2], [1, 4, 16, 64, 256])
    axes[2].set_yscale("log")
    axes[2].set_xlabel("batch size B")
    axes[2].set_ylabel("output tokens / s")
    plain_ticks(axes[2])
    axes[2].set_title("(c) Generation throughput, L = 512")
    axes[2].set_xlim(0.8, 256 * 1.3)
    fig.tight_layout()
    legend_configs(fig, y=-0.13)
    save(fig, "q2_latency_throughput" + ("" if size == "7b" else f"_{size}"))


def legend_configs(fig, y=-0.1, ncol=4, configs=(("vllm", "bf16"), ("vllm", "fp8"), ("vllm", "gptq-int8"),
                                                ("vllm", "awq-int4"), ("hf", "bf16"), ("hf", "int8"), ("hf", "nf4"))):
    handles = []
    for b, p in configs:
        st = S.style(b, p)
        handles.append(Line2D([], [], **st))
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), handlelength=3.2)


# ----------------------------------------------------------------------------- Q3 crossover
def fig_heatmaps(size="7b", metric="speedup_e2e"):
    agg, oom = A.sweeps()
    sp = A.with_speedup(agg)
    sp = sp[(sp["size"] == size) & (sp.output_len == 128) & (sp.tag == "grid")]
    panels = [("vllm", "fp8"), ("vllm", "gptq-int8"), ("vllm", "awq-int4"), ("hf", "int8"), ("hf", "nf4")]
    panels = [p for p in panels if not sp[(sp.backend == p[0]) & (sp.precision == p[1])].empty]
    if not panels:
        return
    ncol = 3
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.2, 2.55 * nrow), squeeze=False)
    lens = [128, 512, 1024, 2048]
    vmax = 2.0
    for ax, (bk, p) in zip(axes.flat, panels):
        r = sp[(sp.backend == bk) & (sp.precision == p)]
        batches = sorted(r.batch.unique())
        grid = np.full((len(lens), len(batches)), np.nan)
        for i, L in enumerate(lens):
            for j, B in enumerate(batches):
                v = r[(r.input_len == L) & (r.batch == B)][metric]
                if len(v) and not np.isnan(v.iloc[0]):
                    grid[i, j] = v.iloc[0]
        im = ax.imshow(np.log2(grid), cmap=S.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        for i in range(len(lens)):
            for j in range(len(batches)):
                v = grid[i, j]
                if SLIDES and not np.isnan(v):
                    continue          # slide version: colour only, the key numbers are on the slide
                if np.isnan(v):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=6.5 * FS, color=S.MUTED)
                    continue
                dark = abs(math.log2(v)) > 1.1
                ax.text(j, i, f"{v:.2f}" if v < 9.95 else f"{v:.0f}", ha="center", va="center", fontsize=6.3 * FS,
                        color="white" if dark else S.INK)
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels([str(b) for b in batches], fontsize=7 * FS)
        ax.set_yticks(range(len(lens)))
        ax.set_yticklabels([str(L) for L in lens], fontsize=7 * FS)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(S.LABEL[(bk, p)], fontsize=9 * FS)
        ax.set_xlabel("batch size B", fontsize=8 * FS)
        if list(axes.flat).index(ax) % ncol == 0:
            ax.set_ylabel("prompt length L", fontsize=8 * FS)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
        if SLIDES:
            continue
        ax.text(0.04, 0.5, "How to read: each cell is the end-to-end\nlatency of a batch of B requests (prompt L,\n"
                "128 generated tokens) on BF16 divided by\nthe same on the quantized model, for the\n"
                "same backend. Blue: quantization is faster;\nred: it is slower; \u2013: out of memory\nor not measured. "
                "GPTQ-INT4 behaves like\nAWQ-INT4 (same Marlin kernel).", transform=ax.transAxes, va="center",
                fontsize=7.5 * FS, color=S.INK2, linespacing=1.5)
    fig.subplots_adjust(wspace=0.18, hspace=0.55)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015, aspect=30)
    ticks = [-2, -1, 0, 1, 2]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{2 ** t:g}×" for t in ticks])
    cb.set_label("end-to-end speedup vs BF16 (same backend)\n< 1 slower (red) · > 1 faster (blue)", fontsize=8 * FS)
    cb.outline.set_visible(False)
    save(fig, "q3_speedup_heatmaps" + ("" if size == "7b" else f"_{size}"))


def fig_kernels(size="7b"):
    raw, layer = A.kernels(size)
    if layer.empty:
        return
    peaks = A.hw_peaks()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.1))
    spec = [("bnb-int8", "hf", "int8", "bnb LLM.int8"), ("bnb-nf4", "hf", "nf4", "bnb NF4"),
            ("fp8-w8a8", "vllm", "fp8", "FP8 W8A8 (CUTLASS)"), ("marlin-w8a16", "vllm", "gptq-int8", "Marlin W8A16"),
            ("marlin-w4a16", "vllm", "awq-int4", "Marlin W4A16")]
    for k, bk, p, name in spec:
        r = layer[layer.kernel == k].sort_values("m")
        if r.empty:
            continue
        st = S.style(bk, p)
        st["label"] = name
        ax1.plot(r.m, r.speedup, **st)
        s = np.log(r.speedup.to_numpy() / 1.05)
        for i in range(1, len(s)):
            if s[i - 1] > 0 >= s[i]:
                m0, m1 = math.log2(r.m.iloc[i - 1]), math.log2(r.m.iloc[i])
                mx = 2 ** (m0 + (0 - s[i - 1]) * (m1 - m0) / (s[i] - s[i - 1]))
                ax1.plot([mx], [1.05], marker="|", ms=10 * FS, color=S.COLOR[p], mew=1.5)
    ax1.axhline(1, color=S.BLUE, lw=1.0)
    ax1.text(1.1, 1.04, "BF16 cuBLAS", color=S.INK2, fontsize=7.5 * FS)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
    ax1.set_xticklabels(["1", "4", "16", "64", "256", "1K", "4K"])
    ax1.set_yticks([0.125, 0.25, 0.5, 1, 2, 4])
    ax1.set_yticklabels(["⅛×", "¼×", "½×", "1×", "2×", "4×"])
    ax1.set_xlim(0.8, 8192 * 1.5)
    ax1.set_xlabel("M = tokens per GEMM (decode batch size)")
    ax1.set_ylabel("speedup vs BF16 (one decoder layer)")
    ax1.set_title(f"(a) Linear-layer kernels, {SIZE_LABEL[size]} shapes")

    # (b) roofline: achieved TFLOP/s against arithmetic intensity (FLOP per byte of weights+activations)
    bw = peaks.get("dram_read_gbps", 1700) / 1e3      # TB/s (decode streams weights: read bandwidth)
    peak_bf16 = peaks.get("bf16_gemm_tflops", 200)
    peak_fp8 = peaks.get("fp8_gemm_tflops", np.nan)
    ai = np.logspace(-0.5, 4, 200)
    ax2.plot(ai, np.minimum(ai * bw, peak_bf16), color=S.INK2, lw=1.0)
    ax2.text(2.0, 2.0 * bw * 1.35, f"DRAM {bw:.2f} TB/s", rotation=38, fontsize=7 * FS, color=S.INK2)
    ax2.text(8e3, peak_bf16 * 0.8, f"BF16 peak {peak_bf16:.0f}", ha="right", va="top", fontsize=7 * FS, color=S.INK2)
    if not np.isnan(peak_fp8):
        ax2.plot(ai, np.minimum(ai * bw, peak_fp8), color=S.INK2, lw=0.8, ls=":")
        ax2.text(8e3, peak_fp8 * 1.1, f"FP8 peak {peak_fp8:.0f} TFLOP/s", ha="right", fontsize=7 * FS, color=S.INK2)
    bpp = {"bf16": 2, "bnb-int8": 1, "bnb-nf4": 0.5625, "fp8-w8a8": 1, "marlin-w8a16": 1.016, "marlin-w4a16": 0.516}
    kshape = raw[raw.m.isin(layer.m.unique())]
    for k, bk, p, name in [("bf16", "vllm", "bf16", "BF16")] + spec:
        r = kshape[kshape.kernel == k]
        if r.empty:
            continue
        r = r.assign(flop=2 * r.m * r.k * r.n, nbytes=r.n * r.k * bpp[k] + r.m * r.k * 2 + r.m * r.n * 2)
        agg_ = r.groupby("m")[["flop", "nbytes", "time_ms"]].sum()
        g = agg_.flop / agg_.nbytes
        tf = agg_.flop / agg_.time_ms / 1e9
        st = S.style(bk, p)
        st["label"] = name
        st["markersize"] = 3.5
        ax2.plot(g.values, tf.values, **st)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("arithmetic intensity (FLOP / byte moved)")
    ax2.set_ylabel("achieved TFLOP/s")
    ax2.set_title("(b) Roofline: fewer weight bytes = more FLOP/byte")
    ax2.set_ylim(top=peak_fp8 * 2.2 if not np.isnan(peak_fp8) else None)
    fig.tight_layout()
    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=7.5 * FS)
    save(fig, "q3_kernel_crossover")


def fig_profile(size="7b"):
    df = A.profiles()
    if df.empty:
        return
    df = df[df["size"] == size].drop_duplicates(["precision", "batch"], keep="last")
    groups = [("Matrix multiply (GEMM / GEMV)", ["BF16/FP16 GEMM", "INT8 GEMM", "4-bit GEMV (bnb)"], "#1c5cab"),
              ("Quantization overhead (dequant, act. quant, outliers)", ["weight dequantization", "activation quant / outliers"], "#e34948"),
              ("Attention + elementwise + norm", ["attention", "elementwise / norm / copy"], "#86b6ef"),
              ("GPU idle (host / launch overhead)", None, "#e1e0d9")]
    rows = []
    for B in sorted(df.batch.unique()):
        for p in HF_P:
            r = df[(df.precision == p) & (df.batch == B)]
            if r.empty:
                continue
            r = r.iloc[0]
            cats = r.categories_ms
            vals = [sum(cats.get(c, 0.0) for c in cs) if cs else max(r.wall_ms_per_step - r.gpu_kernel_ms_per_step, 0)
                    for _, cs, _ in groups]
            rows.append((f"{ {'bf16': 'BF16', 'int8': 'INT8', 'nf4': 'NF4'}[p]}, B={B}", vals, r.wall_ms_per_step))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.0))
    y = np.arange(len(rows))[::-1]
    for yi, (name, vals, wall) in zip(y, rows):
        left = 0
        for (gname, _, c), v in zip(groups, vals):
            ax.barh(yi, v, 0.62, left=left, color=c, edgecolor="white", linewidth=1.0,
                    hatch="//" if c == "#e1e0d9" else None)
            left += v
        ax.text(left + 0.6, yi, f"{wall:.1f} ms", va="center", fontsize=7.5 * FS, color=S.INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("time per decode step (ms)")
    ax.set_title(f"Where a decode step goes — {SIZE_LABEL[size]}, HF backend, L = 512")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=c, hatch="//" if c == "#e1e0d9" else None, edgecolor="white", label=g)
                       for g, _, c in groups], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7 * FS)
    ax.set_xlim(0, max(r[2] for r in rows) * 1.15)
    fig.tight_layout()
    save(fig, "mech_decode_breakdown")


# ----------------------------------------------------------------------------- real workload
def fig_serving(size="7b"):
    vl, hf = A.serving()
    if vl.empty:
        return
    vl = vl[vl["size"] == size]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    shown = []
    for p in VL_ALL:
        r = vl[vl.precision == p].sort_values("concurrency")
        if r.empty:
            continue
        st = S.style("vllm", p)
        for ax, col in ((axes[0], "output_throughput"), (axes[1], "median_tpot_ms"), (axes[2], "median_ttft_ms")):
            ax.plot(r.concurrency, r[col], **st)
        shown.append(("vllm", p))
    if not hf.empty:
        h = hf[hf["size"] == size]
        for p in HF_P:
            r = h[h.precision == p]
            if r.empty:
                continue
            st = S.style("hf", p)
            st["linestyle"] = "none"
            for ax, col in ((axes[0], "output_throughput"), (axes[1], "p50_tpot_ms"), (axes[2], "p50_ttft_ms")):
                ax.plot(r.concurrency, r[col], **st)
            shown.append(("hf", p))
    for ax, t, yl in zip(axes, ["(a) Output throughput", "(b) Median TPOT", "(c) Median TTFT"],
                         ["output tokens / s", "ms per output token", "ms"]):
        pow2_axis(ax, [1, 4, 16, 64])
        ax.set_xlabel("concurrent requests (HF: batch size)")
        ax.set_ylabel(yl)
        ax.set_title(t)
    for ax in (axes[0], axes[2]):   # HF's static-batch TTFT is an order of magnitude above vLLM's
        ax.set_yscale("log")
        plain_ticks(ax)
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.05)
    handles = []
    for bk, p in shown:
        st = S.style(bk, p)
        if bk == "hf":
            st["linestyle"] = "none"
        handles.append(Line2D([], [], **st))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), bbox_to_anchor=(0.5, -0.16),
               handlelength=2.6)
    fig.tight_layout()
    save(fig, "real_sharegpt_serving")


def fig_workload():
    d = A.sharegpt_lengths()
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2))
    for ax, col, t in ((axes[0], "prompt_len", "prompt length"), (axes[1], "output_len", "reference output length")):
        ax.hist(d[col], bins=np.logspace(np.log10(4), np.log10(2048), 28), color=S.BLUE, edgecolor="white", linewidth=0.6)
        ax.set_xscale("log")
        ax.axvline(d[col].median(), color=S.INK, lw=1)
        ax.text(d[col].median() * 1.08, ax.get_ylim()[1] * 0.92, f"median {d[col].median():.0f}", fontsize=7.5 * FS, color=S.INK2)
        ax.set_xlabel(f"{t} (tokens)")
        ax.set_ylabel("requests")
        ax.grid(axis="x", visible=False)
    fig.suptitle("ShareGPT sample used for the real-workload replay (n = 300, seed 0)", fontsize=9 * FS)
    fig.tight_layout()
    save(fig, "workload_sharegpt")


def outlen_speedups(size="7b", L=512):
    """End-to-end speedup over BF16 at one prompt length for 32, 128 and 512 generated tokens."""
    agg, _ = A.sweeps()
    d = agg[(agg["size"] == size) & (agg.input_len == L) & (agg.batch.isin([1, 16]))
            & (((agg.tag == "grid") & (agg.output_len == 128)) | (agg.tag == "outlen"))]
    return A.with_speedup(d)


def fig_outlen(size="7b"):
    sp = outlen_speedups(size)
    sp = sp[sp.precision != "bf16"].dropna(subset=["speedup_e2e"])
    if sp.empty or sp.output_len.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    shown = []
    for ax, B in zip(axes, (1, 16)):
        d = sp[sp.batch == B]
        for bk, ps in (("hf", HF_P), ("vllm", VL_P)):
            for p in ps:
                r = d[(d.backend == bk) & (d.precision == p)].sort_values("output_len")
                if len(r) < 2:
                    continue
                ax.plot(r.output_len, r.speedup_e2e, **S.style(bk, p))
                if (bk, p) not in shown:
                    shown.append((bk, p))
        ax.axhline(1, color=S.INK2, lw=0.8, ls=":")
        pow2_axis(ax, [32, 128, 512])
        ax.set_yscale("log", base=2)
        ax.set_xlabel("generated tokens N")
        ax.set_title(f"B = {B}, L = 512")
    axes[0].set_ylabel("end-to-end speedup over BF16")
    axes[0].set_ylim(0.22, 3.0)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    axes[0].yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\u00d7"))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    legend_configs(fig, y=-0.2, ncol=3, configs=tuple(shown))
    fig.tight_layout()
    save(fig, "q3_output_length")


def fig_generalization():
    """Qwen2.5-7B vs Qwen3-8B: decode speedup at B=1 and MMLU change, per format (slides)."""
    fq, fr = A.family_pair("7b", "q3-8b")
    if not fr:
        return
    cfg = [("awq-int4", "AWQ-INT4\n(vLLM)"), ("fp8", "FP8\n(vLLM)"), ("nf4", "NF4\n(HF)"), ("int8", "LLM.int8()\n(HF)")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.0))
    x = np.arange(len(cfg))
    for ax, key, yl, title in ((ax1, "tpot_b1_{}", "decode speedup over BF16", "(a) Decode speedup, B = 1"),
                               (ax2, "d_mmlu_{}", "MMLU change (points)", "(b) MMLU change vs BF16")):
        for j, (fs, lab, col) in enumerate(((fq, "Qwen2.5-7B", S.MUTED), (fr, "Qwen3-8B", S.BLUE))):
            v = [fs.get(key.format(c), np.nan) for c, _ in cfg]
            bars = ax.bar(x + (j - 0.5) * 0.38, v, 0.36, color=col, label=lab)
            for b_, val in zip(bars, v):
                if np.isnan(val):
                    continue
                txt = f"{val:.2f}\u00d7" if key.startswith("tpot") else f"{val:+.1f}".replace("-", "\u2212")
                ax.annotate(txt, (b_.get_x() + b_.get_width() / 2, val), xytext=(0, 3 if val >= 0 else -3),
                            textcoords="offset points", ha="center", va="bottom" if val >= 0 else "top",
                            fontsize=7 * FS, color=S.INK2)
        ax.axhline(1 if key.startswith("tpot") else 0, color=S.INK2, lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in cfg])
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    vals = [fs.get(f"d_mmlu_{c}", np.nan) for fs in (fq, fr) for c, _ in cfg]
    vals = [v for v in vals if not np.isnan(v)]
    ax2.set_ylim(min(min(vals), -1) * 1.45, max(max(vals), 0.3) * 1.8)
    ax1.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save(fig, "general_qwen3")


# ----------------------------------------------------------------------------- Q4 quality
def fig_quality():
    ppl, gsm = A.quality()
    if gsm.empty and ppl.empty:
        return
    hf, vl = A.loads()

    def mem(backend, size, p):
        if backend == "hf":
            r = hf[(hf["size"] == size) & (hf.precision == p)] if not hf.empty else hf
            return r.weights_bytes.iloc[0] / A.GB if len(r) else np.nan
        r = vl[(vl["size"] == size) & (vl.precision == p)] if not vl.empty else vl
        return r.load_gib.iloc[0] * A.GIB / A.GB if len(r) and r.load_gib.notna().iloc[0] else np.nan

    mm = A.mmlu()
    panels = [(gsm, "accuracy", "GSM8K accuracy (%)"), (ppl, "ppl", "WikiText-2 perplexity")]
    if not mm.empty:
        panels.insert(1, (mm, "accuracy", "MMLU accuracy (%)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels) - 0.2, 3.2))
    for (df, col, yl), ax in zip(panels, axes):
        if df.empty:
            continue
        df = df[df["size"].isin(list(SIZE_LABEL))].copy()   # Qwen2.5 family; Qwen3-8B has its own table
        df["mem"] = [mem(b, s, p) for b, s, p in zip(df.backend, df["size"], df.precision)]
        df = df[~((df.backend == "vllm") & (df.precision == "bf16") & df["size"].isin(["3b", "7b"]))]  # HF row is the BF16 reference
        cap = None
        if col == "ppl":   # keep the axis readable; an off-scale checkpoint is drawn as an arrow at the top edge
            med = df.groupby("size")[col].transform("median")
            off = df[col] > 3 * med
            if off.any():
                cap = df.loc[~off, col].max() * 1.12
                for _, r in df[off].iterrows():
                    st = S.style(r.backend, r.precision)
                    ax.annotate(f"{SIZE_LABEL[r['size']].split('-')[1]} {S.SHORT[(r.backend, r.precision)]}: "
                                f"{r[col]:.1f}\n(off scale)", xy=(r["mem"], cap), xytext=(r["mem"] * 0.96, cap * 0.9),
                                ha="right", va="top", fontsize=7.5 * FS, color=st["color"],
                                arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=1.0))
                df = df[~off]
        for size, g in df.groupby("size"):
            g = g.sort_values("mem")
            yv = g[col] * (100 if col == "accuracy" else 1)
            ax.plot(g["mem"], yv, color=S.GRID, lw=1.2, zorder=1)
            for _, r in g.iterrows():
                st = S.style(r.backend, r.precision)
                ax.plot(r["mem"], r[col] * (100 if col == "accuracy" else 1), marker=st["marker"], color=st["color"],
                        markerfacecolor=st["markerfacecolor"], markeredgecolor=st["markeredgecolor"], ms=6.5 * FS, ls="none", zorder=3)
            ax.annotate(SIZE_LABEL[size].split("-")[1], (g["mem"].iloc[-1], yv.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=8 * FS, fontweight="bold", color=S.INK2, va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32])
        ax.set_xticklabels(["2", "4", "8", "16", "32"])
        ax.set_xlabel("weight memory (GB, log scale)")
        ax.set_ylabel(yl)
        if cap is not None:
            ax.set_ylim(top=cap)
    acc = {(b, s_, p): a for b, s_, p, a in zip(gsm.backend, gsm["size"], gsm.precision, gsm.accuracy)} if not gsm.empty else {}
    a14, a7 = acc.get(("vllm", "14b", "awq-int4")), acc.get(("hf", "7b", "bf16"))
    wins = a14 is not None and a7 is not None and a14 > a7
    titles = ["GSM8K: bigger-but-quantized wins" if wins else "GSM8K accuracy"] + \
        (["MMLU accuracy"] if not mm.empty else []) + ["Perplexity (lower is better)"]
    for i, (ax, t) in enumerate(zip(axes, titles)):
        ax.set_title(f"({'abc'[i]}) {t}")
    handles = [Line2D([], [], color=S.COLOR[p], marker=S.MARKER[p], markerfacecolor="white" if b == "hf" else S.COLOR[p],
                      ls="none", label=S.LABEL[(b, p)])
               for b, p in (("hf", "bf16"), ("hf", "int8"), ("hf", "nf4"), ("vllm", "fp8"), ("vllm", "gptq-int8"),
                            ("vllm", "awq-int4"), ("vllm", "gptq-int4"))]
    fig.legend(handles=handles, loc="lower center", ncol=7 if len(panels) == 3 else 4,
               bbox_to_anchor=(0.5, -0.08 if len(panels) == 3 else -0.13), fontsize=7.5 * FS)
    fig.tight_layout()
    save(fig, "q4_quality_vs_memory")


def fig_energy(size="7b"):
    agg, _ = A.sweeps()
    d = agg[(agg["size"] == size) & (agg.output_len == 128) & (agg.tag == "grid") & (agg.input_len == 512)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for bk, ps in (("hf", HF_P), ("vllm", VL_P)):
        _series(ax, d, bk, ps, "batch", "j_per_tok", label_last=False)
    pow2_axis(ax, [1, 4, 16, 64, 256])
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel("GPU energy per output token (J)")
    ax.set_title(f"Energy efficiency, {SIZE_LABEL[size]}, L = 512")
    plain_ticks(ax)
    ax.set_xlim(0.8, 256 * 1.3)
    fig.tight_layout()
    legend_configs(fig, y=-0.32, ncol=2)
    save(fig, "energy_per_token")


def fig_appendix_3b():
    """3B versions of the latency and crossover figures (report appendix)."""
    fig_latency("3b")
    fig_heatmaps("3b")


def fig_capacity():
    _, vl = A.loads()
    if vl.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    sizes = [s for s in ("3b", "7b", "14b") if s in set(vl["size"])]
    w = 0.16
    for j, p in enumerate(VL_ALL):
        for i, s in enumerate(sizes):
            r = vl[(vl["size"] == s) & (vl.precision == p)]
            x = i + (j - 2) * w
            if r.empty or r.kv_cache_tokens.isna().iloc[0]:
                ax.text(x, 8, "does not fit" if not r.empty else "", rotation=90, fontsize=6.5 * FS, ha="center", va="bottom", color=S.INK2)
                continue
            v = r.kv_cache_tokens.iloc[0]
            ax.bar(x, v / 1e3, w * 0.9, color=S.COLOR[p], edgecolor="white", linewidth=0.8,
                   label=S.LABEL[("vllm", p)] if i == 0 else None)
            ax.text(x, v / 1e3 + 8, f"{v / 1e3:.0f}K", ha="center", va="bottom", fontsize=6.3 * FS, color=S.INK2, rotation=90)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([SIZE_LABEL[s] for s in sizes])
    ax.set_ylabel("KV-cache capacity (thousand tokens)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_title("Memory freed by quantization becomes KV cache (vLLM, 85% of 32 GB)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=7 * FS)
    save(fig, "q1_kv_capacity")


def main():
    S.setup(scale=1.25 if SLIDES else 1.0)
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or [n for n in globals() if n.startswith("fig_")]
    for name in which:
        print(name)
        try:
            globals()[name]()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
