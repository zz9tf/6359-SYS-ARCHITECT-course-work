"""Result slides. Every number on them is computed from results/ at build time."""
import re
from pathlib import Path

import numpy as np

import uta as U
from layouts import figure_with_tiles_below, figure_with_tiles_right
from quantbench import analysis as A


def macros():
    """The report's generated numbers (report/numbers.tex) as plain text, so slides and report agree."""
    f = Path(__file__).resolve().parents[1] / "report" / "numbers.tex"
    out = {}
    if f.exists():
        for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", f.read_text()):
            val = (val.replace("$\\times$", "\u00d7").replace("{,}", ",").replace("\\,", " ").replace("--", "\u2013")
                   .replace("\\%", "%").replace("$<$", "<"))
            val = re.sub(r"\$([^$]*)\$", r"\1", val).replace("vs.\\ ", "vs. ").replace("<", " < ").replace("  ", " ")
            val = re.sub(r"(?<![\w.])-(?=\d)", "\u2212", val)   # numeric minus signs only
            out[name] = val.strip()
    return out


def _x(v, nd=1):
    return "n/a" if v is None or np.isnan(v) else f"{v:.{nd}f}×"


def facts_from_results():
    f = {}
    agg, oom = A.sweeps()
    sp = A.with_speedup(agg)
    hf, vl = A.loads()

    def sv(backend, prec, size="7b", B=1, L=512, metric="speedup_tpot"):
        r = sp[(sp.backend == backend) & (sp["size"] == size) & (sp.precision == prec) & (sp.batch == B)
               & (sp.input_len == L) & (sp.output_len == 128) & (sp.tag == "grid")]
        return r[metric].iloc[0] if len(r) else np.nan

    f["sv"] = sv
    w = hf[hf["size"] == "7b"].set_index("precision").weights_bytes if not hf.empty else {}
    f["w_ratio_int8"] = w["bf16"] / w["int8"] if "int8" in w else np.nan
    f["w_ratio_nf4"] = w["bf16"] / w["nf4"] if "nf4" in w else np.nan
    m = sp[(sp.backend == "hf") & (sp["size"] == "7b") & (sp.tag == "grid") & (sp.input_len == 2048) & (sp.output_len == 128)]
    big = m.dropna(subset=["mem_saving"]).sort_values("batch")
    f["mem_big_nf4"] = big[big.precision == "nf4"].mem_saving.iloc[-1] if len(big[big.precision == "nf4"]) else np.nan
    f["mem_big_int8"] = big[big.precision == "int8"].mem_saving.iloc[-1] if len(big[big.precision == "int8"]) else np.nan
    f["mem_big_batch"] = int(big[big.precision == "nf4"].batch.iloc[-1]) if len(big[big.precision == "nf4"]) else 0
    if not vl.empty:
        kv = vl[vl["size"] == "7b"].set_index("precision").kv_cache_tokens
        f["kv_ratio_awq"] = kv.get("awq-int4", np.nan) / kv.get("bf16", np.nan)
        f["kv_bf16"], f["kv_awq"] = kv.get("bf16", np.nan), kv.get("awq-int4", np.nan)
    f["agg"], f["sp"] = agg, sp
    return f


def slide_q1(prs, fig_dir, f):
    s = U.content_slide(prs, "Q1 — Memory: Smaller, but Not 4× Smaller",
                        "Only the linear layers shrink; embeddings, LM head, KV cache and activations do not",
                        notes=("Weights: NF4 saves {:.2f}x instead of 4x, INT8 {:.2f}x instead of 2x, because the "
                               "embedding and LM head stay in 16-bit and NF4 stores a scale per 64 weights. At run time "
                               "the KV cache and activations are not quantized, so the end-to-end saving shrinks toward "
                               "1x as batch and context grow.").format(f["w_ratio_nf4"], f["w_ratio_int8"]))
    tiles = [(_x(f["w_ratio_nf4"], 2), "NF4 weight saving on Qwen2.5-7B", "nominal 4× — LM head + embeddings stay 16-bit"),
             (_x(f["mem_big_nf4"], 2), f"NF4 end-to-end saving at B={f['mem_big_batch']}, L=2048",
              "KV cache and activations are not quantized"),
             (_x(f.get("kv_ratio_awq", np.nan), 2), "more KV-cache tokens in vLLM with AWQ-INT4",
              "freed memory = more concurrent requests")]
    tiles = [(t[0], {0: "NF4 weight saving (7B), not 4\u00d7", 1: f"NF4 total saving, B={f['mem_big_batch']}, L=2048",
                     2: "more KV-cache tokens (vLLM AWQ)"}[i], t[2]) for i, t in enumerate(tiles)]
    figure_with_tiles_below(s, fig_dir / "q1_memory_main.png", tiles,
                            alt="Weight memory against the naive estimate, and memory saving versus tokens in flight")


def slide_q2(prs, fig_dir, f):
    sv = f["sv"]
    awq, int8 = sv("vllm", "awq-int4"), sv("hf", "int8")
    agg = f["agg"]
    thr = agg[(agg["size"] == "7b") & (agg.backend == "vllm") & (agg.batch == 256) & (agg.input_len == 512)
              & (agg.tag == "grid")].set_index("precision").out_tok_s
    fp_ratio = thr.get("fp8", np.nan) / thr.get("bf16", np.nan)
    s = U.content_slide(prs, "Q2 \u2014 Same Bits, Different Speed",
                        f"Qwen2.5-7B decode: from {_x(awq, 1)} faster to {_x(1 / int8, 1)} slower than BF16 "
                        "\u2014 the implementation decides",
                        notes=("Left: prefill is compute-bound, so only formats that quantize activations (FP8) cut TTFT. "
                               "Middle: decode at small batch is memory-bound; vLLM's fused kernels turn fewer bytes into "
                               "speed, bitsandbytes does not. Right: at high batch only FP8 still raises throughput."))
    tiles = [(_x(awq, 2) + " faster", "AWQ-INT4 on vLLM, batch 1", "Marlin kernel: fewer bytes = less time"),
             (_x(1 / int8, 1) + " slower", "LLM.int8() on HF, batch 1",
              "different method, activation path and engine, not only a different kernel"),
             (_x(fp_ratio, 2), "FP8 throughput at batch 256", "AWQ-INT4 there: no faster than BF16")]
    figure_with_tiles_below(s, fig_dir / "teaser_speedup.png", tiles,
                            alt="Decode speedup against weight memory saved, and against batch size")


def slide_q3(prs, fig_dir, f):
    sp = f["sp"]
    e7 = sp[(sp["size"] == "7b") & (sp.tag == "grid") & (sp.output_len == 128)]
    cx = A.crossover(e7, "speedup_e2e")
    awq = cx[(cx.backend == "vllm") & (cx.precision == "awq-int4") & (cx.input_len == 512)]
    awq_c = float(awq.crossover_batch.iloc[0]) if len(awq) else np.nan
    fp = e7[(e7.backend == "vllm") & (e7.precision == "fp8")].speedup_e2e
    i8 = e7[(e7.backend == "hf") & (e7.precision == "int8")].speedup_e2e
    s = U.content_slide(prs, "Q3 \u2014 When Does Lower Precision Pay Off?",
                        "End-to-end speedup over BF16 across batch size \u00d7 prompt length (Qwen2.5-7B)",
                        notes=("Blue cells: quantized is faster; red: slower. Weight-only 4-bit wins big at small batches "
                               "and short prompts and fades out with batch size; FP8 wins everywhere; LLM.int8 never wins."))
    tiles = [(f"B \u2248 {awq_c:.0f}" if not np.isnan(awq_c) else "n/a", "AWQ-INT4 stops winning",
              "prompt 512; decode turns compute-bound, dequant no longer hidden"),
             (f"{fp.min():.1f}\u2013{fp.max():.1f}\u00d7", "FP8 faster in every cell",
              "halves bytes and raises the compute roof"),
             (f"{i8.min():.2f}\u2013{i8.max():.2f}\u00d7", "LLM.int8() slower in every cell", "never pays off on this GPU")]
    figure_with_tiles_right(s, fig_dir / "q3_speedup_heatmaps.png", tiles,
                            alt="Heatmaps of end-to-end speedup versus BF16 over batch size and prompt length", fig_w=6.9)


def _kernel_facts():
    _, layer = A.kernels("7b")
    out = {}
    if layer.empty or "kernel" not in layer:
        return out
    for k in layer.kernel.unique():
        r = layer[layer.kernel == k].sort_values("m")
        s_, m_ = r.speedup.to_numpy(), r.m.to_numpy()
        cross = np.nan
        for i in range(1, len(s_)):
            if s_[i - 1] > 1.05 >= s_[i]:          # advantage falls within 5% of BF16
                x0, x1 = np.log2(m_[i - 1]), np.log2(m_[i])
                y0, y1 = np.log(s_[i - 1] / 1.05), np.log(s_[i] / 1.05)
                cross = 2 ** (x0 - y0 * (x1 - x0) / (y1 - y0))
        out[k] = {"m1": float(r[r.m == 1].speedup.iloc[0]) if (r.m == 1).any() else np.nan,
                  "mmax": float(r.speedup.iloc[-1]), "cross": cross}
    return out


def slide_kernels(prs, fig_dir, f):
    fig = fig_dir / "q3_kernel_crossover.png"
    k = _kernel_facts()
    if not fig.exists() or not k:
        return
    w4, fp = k.get("marlin-w4a16", {}), k.get("fp8-w8a8", {})
    s = U.content_slide(prs, "Why 4-bit Stops Winning",
                        "One decoder layer's GEMMs vs. tokens in the batch, and the same kernels on the measured roofline",
                        notes=("Weight-only 4-bit cuts the bytes 4x, so at small M it climbs the bandwidth roof 4x faster. "
                               "It also reaches the compute roof at a 4x smaller M; from there the math still runs in "
                               "BF16 and dequantization is pure overhead. FP8 W8A8 raises the compute roof itself."))
    tiles = [(_x(w4.get("m1", np.nan), 1), "Marlin W4A16 vs BF16 at M = 1", "memory-bound: 4x fewer bytes"),
             (f"M \u2248 {w4.get('cross', np.nan):.0f}" if not np.isnan(w4.get("cross", np.nan)) else "never",
              "W4A16 gain falls below 5%", "compute-bound: dequant no longer hidden"),
             (_x(fp.get("mmax", np.nan), 1), "FP8 W8A8 at M = 8192", "higher compute roof than BF16 (645 vs 218 TFLOP/s)")]
    figure_with_tiles_below(s, fig, tiles, alt="Linear-layer kernel speedup versus batch size and roofline plot")


def slide_profile(prs, fig_dir, f):
    fig = fig_dir / "mech_decode_breakdown.png"
    prof = A.profiles()
    if not fig.exists() or prof.empty:
        return
    p7 = prof[prof["size"] == "7b"].drop_duplicates(["precision", "batch"], keep="last").set_index(["precision", "batch"])

    def pv(prec, col, B=1):
        try:
            return float(p7.loc[(prec, B)][col])
        except KeyError:
            return np.nan

    s = U.content_slide(prs, "Why bitsandbytes Is Slow",
                        "One HF decode step of Qwen2.5-7B: GPU kernel time vs. time the GPU waits for the host",
                        notes=("LLM.int8() splits every matmul into casts, outlier extraction, INT8 quantization and two GEMMs. "
                               "The GPU work is small; the Python and kernel-launch overhead dominates the step."))
    tiles = [(f"{pv('bf16', 'gpu_busy_frac') * 100:.0f}% \u2192 {pv('int8', 'gpu_busy_frac') * 100:.0f}%",
              "GPU busy: BF16 \u2192 INT8", "the rest of the step the GPU waits for the CPU"),
             (_x(pv("int8", "kernels_per_step") / pv("bf16", "kernels_per_step"), 1),
              "more kernel launches (INT8)", "casts, outliers, quantize, 2 GEMMs per layer"),
             (_x(f["sv"]("hf", "nf4"), 2), "NF4 vs BF16, whole model",
              f"its GEMV kernel alone: {_kernel_facts().get('bnb-nf4', {}).get('m1', np.nan):.1f}\u00d7; GPU busy {pv('nf4', 'gpu_busy_frac') * 100:.0f}%")]
    figure_with_tiles_below(s, fig, tiles, alt="Decode step time breakdown by kernel category", tile_h=0.95)


def slide_serving(prs, fig_dir, f):
    fig = fig_dir / "real_sharegpt_serving.png"
    vl, hf = A.serving()
    if not fig.exists() or vl.empty:
        return
    v7 = vl[vl["size"] == "7b"]

    def thr(prec, c):
        r = v7[(v7.precision == prec) & (v7.concurrency == c)]
        return float(r.output_throughput.iloc[0]) if len(r) else np.nan

    def tpot(prec, c):
        r = v7[(v7.precision == prec) & (v7.concurrency == c)]
        return float(r.median_tpot_ms.iloc[0]) if len(r) else np.nan

    hf_bf = hf[(hf["size"] == "7b") & (hf.precision == "bf16")] if not hf.empty else hf
    hf_thr = float(hf_bf.output_throughput.iloc[0]) if len(hf_bf) else np.nan
    s = U.content_slide(prs, "Real Workload: 300 ShareGPT Requests",
                        "Qwen2.5-7B served by vLLM (1\u201364 concurrent clients) vs. HF static batching (B = 16)",
                        notes=("Same 300 requests, same output lengths for every configuration. Continuous batching and "
                               "fused kernels matter more than the number format; among vLLM formats FP8 scales best."))
    tiles = [(_x(tpot("bf16", 1) / tpot("awq-int4", 1), 2), "lower TPOT, AWQ-INT4, 1 client",
              f"{tpot('awq-int4', 1):.1f} vs {tpot('bf16', 1):.1f} ms per token"),
             (f"{thr('fp8', 64):,.0f}", "tokens/s, FP8, 64 clients", f"BF16: {thr('bf16', 64):,.0f} \u00b7 AWQ: {thr('awq-int4', 64):,.0f}"),
             (_x(thr("bf16", 16) / hf_thr, 1), "vLLM vs HF, both BF16",
              "a serving-stack gap, not a quantization effect; the two stacks are also timed differently (HF has no "
              "warm-up and its TTFT excludes queueing)")]
    figure_with_tiles_below(s, fig, tiles, alt="ShareGPT serving throughput, TPOT and TTFT versus concurrency")


def slide_quality(prs, fig_dir, f):
    fig = fig_dir / "q4_quality_vs_memory.png"
    ppl, gsm = A.quality()
    if not fig.exists() or gsm.empty:
        return
    m = macros()
    qf = quality_facts()
    tiles = qf["tiles"]
    if "MFourMin" in m and "EightKernMax" in m:
        d_g = qf["a14"] - qf["a7"]
        d_m = float(m.get("MMLUAWQFourteen", "nan")) - float(m.get("MMLUBFSeven", "nan"))
        tiles = [(f"\u2264 {m['EightKernMax']} pp", "8-bit (GPTQ-INT8, FP8) vs BF16",
                  f"all sizes, GSM8K and MMLU; only LLM.int8() on 3B loses ({m.get('IntMMLUThree', '?')} MMLU pp)"),
                 (f"{m['MFourMin']} to {m['MFourMax']} pp", "MMLU change, 4-bit formats",
                  f"3B {m.get('MDFourMinThree')}\u2026{m.get('MDFourMaxThree')} \u00b7 7B {m.get('MDFourMinSeven')}\u2026"
                  f"{m.get('MDFourMaxSeven')} \u00b7 14B {m.get('MDFourMinFourteen')}\u2026{m.get('MDFourMaxFourteen')}"),
                 (f"+{d_g:.1f} / +{d_m:.1f} pp", "14B 4-bit vs 7B BF16", "GSM8K / MMLU; " + qf["tiles"][2][2])]
    s = U.content_slide(prs, "Q4 \u2014 Quality: What Do We Lose?",
                        "GSM8K (1,319 q.) \u00b7 MMLU (14,042 q.) \u00b7 WikiText-2 perplexity, with paired McNemar tests",
                        notes=qf.get("notes", ""))
    figure_with_tiles_below(s, fig, tiles, alt="GSM8K, MMLU and perplexity versus weight memory")


def quality_facts():
    ppl, gsm = A.quality()
    hf, vl = A.loads()
    g = gsm.set_index(["backend", "size", "precision"]).accuracy * 100 if not gsm.empty else {}

    def acc(b, s, p):
        try:
            return float(g.loc[(b, s, p)])
        except KeyError:
            return np.nan

    eight = (("hf", "int8"), ("vllm", "gptq-int8"), ("vllm", "fp8"))
    four = (("hf", "nf4"), ("vllm", "awq-int4"), ("vllm", "gptq-int4"))
    d8 = [acc(b, "7b", p) - acc("hf", "7b", "bf16") for b, p in eight]
    d4 = [acc(b, "7b", p) - acc("hf", "7b", "bf16") for b, p in four]
    d8 = [v for v in d8 if not np.isnan(v)]
    d4 = [v for v in d4 if not np.isnan(v)]

    def tests(size, group):
        return [(b, p, t) for b, p in group if (t := A.gsm8k_paired(("hf", size, "bf16"), (b, size, p)))]

    n_fam = len(tests("3b", eight + four) + tests("7b", eight + four))
    alpha = 0.05 / n_fam if n_fam else 0.05

    def m(v, nd=1):
        return f"{v:+.{nd}f}".replace("-", "\u2212")

    def rng(vals):
        return m(min(vals)) if round(min(vals), 1) == round(max(vals), 1) else f"{m(min(vals))} to {m(max(vals))}"

    def verdict8():
        t = tests("7b", eight)
        sig = [(b, p, x) for b, p, x in t if x["p_value"] < 0.05]
        if not t:
            return ""
        if not sig:
            return "none significant (McNemar p \u2265 0.05)"
        up = all(x["delta_pp"] > 0 for _, _, x in sig)
        surv = any(x["p_value"] < alpha for _, _, x in sig)
        return (f"{len(sig)} of {len(t)} nominally significant" + (" (an improvement)" if up else "")
                + ("" if surv else f", none after Bonferroni over {n_fam} tests"))

    def verdict4():
        t7, t3 = tests("7b", four), tests("3b", four)
        s7 = sum(x["p_value"] < 0.05 for _, _, x in t7)
        d3 = [x["delta_pp"] for _, _, x in t3]
        s3 = sum(x["p_value"] < alpha for _, _, x in t3)
        txt = "none significant on 7B" if s7 == 0 else f"{s7} of {len(t7)} significant on 7B"
        if d3:
            txt += f"; 3B: {rng(d3)} pp ({s3} of {len(t3)} significant after correction)"
        return txt

    a14, a7 = acc("vllm", "14b", "awq-int4"), acc("hf", "7b", "bf16")
    t14 = A.gsm8k_paired(("hf", "7b", "bf16"), ("vllm", "14b", "awq-int4"))
    p14 = "" if not t14 else (", McNemar p < 0.001" if t14["p_value"] < 0.001 else f", McNemar p = {t14['p_value']:.3f}")
    m14 = vl[(vl["size"] == "14b") & (vl.precision == "awq-int4")].load_gib.mean() * A.GIB if not vl.empty else np.nan
    m7 = hf[(hf["size"] == "7b") & (hf.precision == "bf16")].weights_bytes.mean() if not hf.empty else np.nan
    mem_note = f"{m14 / m7:.2f}\u00d7 the weight memory{p14}" if not np.isnan(m14 / m7) else ""
    tiles = [(f"{rng(d8)} pp" if d8 else "n/a", "GSM8K change with 8-bit formats (7B)", verdict8()),
             (f"{rng(d4)} pp" if d4 else "n/a", "GSM8K change with 4-bit formats (7B)", verdict4()),
             (f"{a14:.1f}% vs {a7:.1f}%" if not np.isnan(a14) and not np.isnan(a7) else "n/a",
              "14B AWQ-INT4 vs 7B BF16 on GSM8K", mem_note)]
    return {"tiles": tiles, "d8": d8, "d4": d4, "a14": a14, "a7": a7}


def slide_general(prs, fig_dir):
    fig = fig_dir / "general_qwen3.png"
    m = macros()
    if not fig.exists() or "FamRAWQone" not in m:
        return
    s = U.content_slide(prs, "Does It Generalize? Qwen3-8B and 32B",
                        "Qwen3-8B reproduces the systems picture; Qwen3-32B runs on this GPU only at 4 bits",
                        notes=("Qwen3-8B reproduces the decode speedups of AWQ and FP8 and the LLM.int8() slowdown within a few "
                               "percent. NF4 turns from a small gain into a loss because the 36-layer model is host-bound at "
                               "batch 1, exactly like the 36-layer Qwen2.5-3B."))
    tiles = [(f"{m['FamRAWQone']} vs {m['FamQAWQone']}", "AWQ speedup: Qwen3 vs Qwen2.5",
              f"Qwen3-8B vs Qwen2.5-7B at B=1; FP8 {m.get('FamRFPone')} vs {m.get('FamQFPone')}"),
             (m.get("FamRNFone", "n/a"), "NF4 on Qwen3-8B (36 layers)",
              f"{m.get('FamRLayers')} layers \u2192 host-bound, as on Qwen2.5-3B"),
             (f"{m.get('BigMMLUawq', '?')}% vs {m.get('SmallMMLUbf', '?')}%", "MMLU: 32B 4-bit vs 8B BF16",
              f"{m.get('BigAWQw', '?')} vs {m.get('SmallBFw', '?')} GB; BF16 32B needs {m.get('BigBFw', '?')} GB")]
    if "BigMMLUawq" not in m:
        tiles[2] = (f"{m.get('FamKVratio')}", "KV-cache bytes per token vs Qwen2.5-7B", "B=256 overflows the cache in every format")
    figure_with_tiles_below(s, fig, tiles, alt="Decode speedup and MMLU change for Qwen2.5-7B and Qwen3-8B")


def slide_takeaways(prs, f):
    sv = f["sv"]
    agg = f["agg"]
    thr = agg[(agg["size"] == "7b") & (agg.backend == "vllm") & (agg.batch == 256) & (agg.input_len == 512)
              & (agg.tag == "grid")].set_index("precision").out_tok_s
    s = U.content_slide(prs, "Takeaways: When Does Quantization Pay Off?",
                        "Bit-width alone predicts little \u2014 the implementation and the regime decide",
                        notes=("Six rules: plan memory with the real ratio; for interactive serving use 4-bit weights with a "
                               "fused kernel; for throughput use FP8 W8A8; treat bitsandbytes as a memory tool, not a speed-up; "
                               "8-bit formats with good kernels preserve quality while 4-bit costs one to three MMLU points; "
                               "and at a fixed memory budget a larger 4-bit model beats a smaller 16-bit one."))
    m = macros()
    cards = [("LuMemoryStick", "Plan with the real ratio", f"{f['w_ratio_nf4']:.1f}\u00d7",
              "4-bit weight saving, not 4\u00d7"),
             ("LuZap", "Few users: 4-bit + Marlin", f"{sv('vllm', 'awq-int4'):.1f}\u00d7",
              "faster decode at batch 1"),
             ("LuTrendingUp", "Many users: FP8 W8A8", f"{thr.get('fp8', np.nan) / thr.get('bf16', np.nan):.1f}\u00d7",
              "BF16 throughput at batch 256"),
             ("LuTriangleAlert", "bitsandbytes: memory only", f"{1 / sv('hf', 'int8'):.1f}\u00d7",
              "slower decode with LLM.int8()"),
             ("LuScale", "Quality: check per model", f"{m.get('MFourMin', '?')} pp",
              "worst 4-bit MMLU loss (3B)"),
             ("LuLayers", "Spend memory on parameters", f"{m.get('MMLUAWQFourteen', '?')}%",
              f"MMLU, 14B 4-bit (vs {m.get('MMLUBFSeven', '?')}% 7B BF16)")]
    w, h = 2.93, 1.55
    for i, (ic, head, big, body) in enumerate(cards):
        x = 0.5 + (i % 3) * (w + 0.13)
        y = U.TOP + (i // 3) * (h + 0.14)
        U.box(s, x, y, w, h, fill=U.LIGHT, radius=0.08)
        U.icon(s, x + 0.3, y + 0.32, 0.36, ic)
        U.text(s, x + 0.56, y + 0.1, w - 0.64, 0.44, [head], size=13, bold=True, color=U.NAVY, anchor="m")
        U.text(s, x + 0.14, y + 0.58, 1.3, 0.85, [big], size=22, bold=True, color=U.BLUE_TXT, anchor="m")
        U.text(s, x + 1.46, y + 0.58, w - 1.56, h - 0.66, [body], size=12, color=U.TEXT, anchor="m")


def add_results_slides(prs, fig_dir, facts, setup):
    f = facts_from_results()
    setup()
    slide_q1(prs, fig_dir, f)
    slide_q2(prs, fig_dir, f)
    slide_q3(prs, fig_dir, f)
    slide_kernels(prs, fig_dir, f)
    slide_profile(prs, fig_dir, f)
    slide_serving(prs, fig_dir, f)
    slide_quality(prs, fig_dir, f)
    slide_general(prs, fig_dir)
    slide_takeaways(prs, f)
