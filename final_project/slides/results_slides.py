"""Result slides. Every number on them is computed from results/ at build time."""
import numpy as np

import uta as U
from layouts import figure_with_tiles_below, figure_with_tiles_right
from quantbench import analysis as A


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
    figure_with_tiles_right(s, fig_dir / "q1_memory_scaling.png", tiles,
                            alt="Peak memory breakdown and memory saving versus tokens in flight", fig_w=6.25)


def slide_q2(prs, fig_dir, f):
    sv = f["sv"]
    awq, int8 = sv("vllm", "awq-int4"), sv("hf", "int8")
    agg = f["agg"]
    thr = agg[(agg["size"] == "7b") & (agg.backend == "vllm") & (agg.batch == 256) & (agg.input_len == 512)
              & (agg.tag == "grid")].set_index("precision").out_tok_s
    fp_ratio = thr.get("fp8", np.nan) / thr.get("bf16", np.nan)
    s = U.content_slide(prs, "Q2 \u2014 Speed Depends on the Kernel",
                        "Qwen2.5-7B: same 4-bit weights, opposite outcomes on two software stacks",
                        notes=("Left: prefill is compute-bound, so only formats that quantize activations (FP8) cut TTFT. "
                               "Middle: decode at small batch is memory-bound; vLLM's fused kernels turn fewer bytes into "
                               "speed, bitsandbytes does not. Right: at high batch only FP8 still raises throughput."))
    tiles = [(_x(awq, 2) + " faster", "AWQ-INT4 decode on vLLM (B=1)", "Marlin kernel: fewer bytes = less time"),
             (_x(1 / int8, 1) + " slower", "LLM.int8() decode on HF (B=1)", "outliers + kernel-launch overhead"),
             (_x(fp_ratio, 2), "FP8 W8A8 throughput at B=256", "AWQ-INT4 there: no faster than BF16")]
    figure_with_tiles_below(s, fig_dir / "q2_latency_throughput.png", tiles,
                            alt="TTFT, decode latency and throughput of eight configurations for Qwen2.5-7B")


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
    tiles = [(f"B \u2248 {awq_c:.0f}" if not np.isnan(awq_c) else "n/a", "AWQ-INT4 stops winning (prompt 512)",
              "decode turns compute-bound; dequant no longer hidden"),
             (f"{fp.min():.2f}\u2013{fp.max():.2f}\u00d7", "FP8 W8A8 faster in every cell",
              "halves bytes and doubles tensor-core rate"),
             (f"{i8.min():.2f}\u2013{i8.max():.2f}\u00d7", "bnb LLM.int8(): slower in every cell", "never pays off on this GPU")]
    figure_with_tiles_right(s, fig_dir / "q3_speedup_heatmaps.png", tiles,
                            alt="Heatmaps of end-to-end speedup versus BF16 over batch size and prompt length", fig_w=6.35)


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
    tiles = [(_x(w4.get("m1", np.nan), 1), "Marlin W4A16 vs cuBLAS at M = 1", "memory-bound: 4x fewer bytes"),
             (f"M \u2248 {w4.get('cross', np.nan):.0f}" if not np.isnan(w4.get("cross", np.nan)) else "never",
              "W4A16 advantage shrinks below 5%", "compute-bound: dequant no longer hidden"),
             (_x(fp.get("mmax", np.nan), 1), "FP8 W8A8 at M = 8192", "8-bit tensor cores: 2x the BF16 rate")]
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
              "GPU busy time per step: BF16 \u2192 INT8 (B=1)", "the rest is host overhead"),
             (_x(pv("int8", "kernels_per_step") / pv("bf16", "kernels_per_step"), 1),
              "more kernel launches per step with LLM.int8()", "casts, outliers, quantize, 2 GEMMs per layer"),
             (_x(f["sv"]("hf", "nf4"), 2), "NF4 end-to-end vs BF16 (7B, B=1)",
              f"its GEMV kernel alone: {_kernel_facts().get('bnb-nf4', {}).get('m1', np.nan):.1f}\u00d7; GPU busy {pv('nf4', 'gpu_busy_frac') * 100:.0f}%")]
    figure_with_tiles_right(s, fig, tiles, alt="Decode step time breakdown by kernel category", fig_w=6.1)


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
    tiles = [(_x(tpot("bf16", 1) / tpot("awq-int4", 1), 2), "lower median TPOT with AWQ-INT4 (1 client)",
              f"{tpot('awq-int4', 1):.1f} vs {tpot('bf16', 1):.1f} ms per token"),
             (f"{thr('fp8', 64):,.0f}", "tokens/s with FP8 at 64 clients", f"BF16: {thr('bf16', 64):,.0f} \u00b7 AWQ: {thr('awq-int4', 64):,.0f}"),
             (_x(thr("bf16", 16) / hf_thr, 1), "vLLM BF16 vs HF BF16 throughput at 16 in flight",
              "the serving stack outweighs the bit-width")]
    figure_with_tiles_below(s, fig, tiles, alt="ShareGPT serving throughput, TPOT and TTFT versus concurrency")


def slide_quality(prs, fig_dir, f):
    fig = fig_dir / "q4_quality_vs_memory.png"
    ppl, gsm = A.quality()
    if not fig.exists() or gsm.empty:
        return
    qf = quality_facts()
    s = U.content_slide(prs, "Q4 \u2014 Quality: What Do We Lose?",
                        "GSM8K accuracy (1,319 questions) and WikiText-2 perplexity vs. weight memory",
                        notes=qf.get("notes", ""))
    figure_with_tiles_right(s, fig, qf["tiles"], alt="GSM8K accuracy and perplexity versus weight memory", fig_w=6.2)


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


def slide_takeaways(prs, f):
    sv = f["sv"]
    agg = f["agg"]
    thr = agg[(agg["size"] == "7b") & (agg.backend == "vllm") & (agg.batch == 256) & (agg.input_len == 512)
              & (agg.tag == "grid")].set_index("precision").out_tok_s
    s = U.content_slide(prs, "Takeaways: When Does Quantization Pay Off?",
                        "Bit-width alone predicts little \u2014 the kernel and the regime decide",
                        notes=("Four rules: plan memory with the real ratio; for interactive serving use 4-bit weights with a "
                               "fused kernel; for throughput use FP8 W8A8; treat bitsandbytes as a memory tool, not a speed-up."))
    cards = [("LuMemoryStick", "Plan memory with the real ratio", f"{f['w_ratio_nf4']:.1f}\u00d7",
              "4-bit weight saving on Qwen2.5-7B, not 4\u00d7; KV cache and activations stay 16-bit"),
             ("LuZap", "Few users: 4-bit + fused kernel", f"{sv('vllm', 'awq-int4'):.1f}\u00d7",
              "faster decode with AWQ-INT4 + Marlin at batch size 1"),
             ("LuTrendingUp", "High throughput: FP8 W8A8", f"{thr.get('fp8', np.nan) / thr.get('bf16', np.nan):.1f}\u00d7",
              "the BF16 throughput at batch 256, where 4-bit weight-only gains have vanished"),
             ("LuTriangleAlert", "bitsandbytes: memory, not speed", f"{1 / sv('hf', 'int8'):.1f}\u00d7",
              "slower decode with LLM.int8() at batch size 1; NF4 wins only tiny batches of large models")]
    w, h = 4.42, 1.55
    for i, (ic, head, big, body) in enumerate(cards):
        x = 0.5 + (i % 2) * (w + 0.16)
        y = U.TOP + (i // 2) * (h + 0.16)
        U.box(s, x, y, w, h, fill=U.LIGHT, radius=0.08)
        U.icon(s, x + 0.36, y + 0.38, 0.44, ic)
        U.text(s, x + 0.72, y + 0.14, w - 0.86, 0.48, [head], size=12, bold=True, color=U.NAVY, anchor="m")
        U.text(s, x + 0.2, y + 0.72, 1.55, 0.62, [big], size=22, bold=True, color=U.BLUE_TXT, anchor="m")
        U.text(s, x + 1.8, y + 0.7, w - 1.95, h - 0.78, [body], size=9.5, color=U.TEXT, anchor="m")


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
    slide_takeaways(prs, f)
