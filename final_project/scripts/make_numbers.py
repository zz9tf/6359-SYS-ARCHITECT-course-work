#!/usr/bin/env python
"""Write report/numbers.tex: every number quoted in the report text, computed from the results.

Keeping the prose numbers in generated macros means the text can never drift from the data.
Missing inputs produce "n/a" so a partial run still compiles.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import analysis as A  # noqa: E402
from quantbench.results import ROOT  # noqa: E402

NUMS = {}


def ok(v):
    return v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def put(name, value, fmt="{:.1f}"):
    NUMS[name] = value if isinstance(value, str) else (fmt.format(value) if ok(value) else "n/a")


def putx(name, value, nd=1):
    put(name, f"{value:.{nd}f}$\\times$" if ok(value) else "n/a")


def first(df, col):
    return float(df[col].iloc[0]) if len(df) else float("nan")


def main():
    agg, oom = A.sweeps()
    sp = A.with_speedup(agg)
    hf_load, vl_load = A.loads()
    peaks = A.hw_peaks()
    grid = sp[(sp.tag == "grid") & (sp.output_len == 128)]

    def cell(backend, size, prec, B, L, col):
        return first(grid[(grid.backend == backend) & (grid["size"] == size) & (grid.precision == prec)
                          & (grid.batch == B) & (grid.input_len == L)], col)

    # ------------------------------------------------------------------ platform / protocol
    bw = peaks.get("dram_read_gbps", float("nan")) / 1e3
    bf = peaks.get("bf16_gemm_tflops", float("nan"))
    put("BWmeas", bw, "{:.2f}")
    put("BFmeas", bf, "{:.0f}")
    put("RidgeBF", bf / bw if ok(bw) else float("nan"), "{:.0f}")
    put("FPeightmeas", peaks.get("fp8_gemm_tflops", float("nan")), "{:.0f}")
    runs = pd.concat([d[d.status == "ok"] for d in (A.raw("hf_sweep.jsonl"), A.raw("vllm_sweep.jsonl")) if not d.empty])
    put("NumMeasurements", f"{len(runs):,}".replace(",", "{,}"))
    cv = runs.groupby(A.KEYS).e2e_s.agg(lambda s: s.std() / s.mean() * 100).dropna()
    put("CVmax", cv.quantile(0.95), "{:.1f}")
    put("CVmed", cv.median(), "{:.1f}")
    put("ClockAvg", runs.avg_sm_clock_mhz.mean(), "{:.0f}")

    # ------------------------------------------------------------------ Q1 memory
    hfw = hf_load.set_index(["size", "precision"]) if not hf_load.empty else pd.DataFrame()
    errs, excess = [], []
    for (size, prec), r in hfw.iterrows():
        errs.append(abs(r.weights_bytes - r.theory_bytes) / r.weights_bytes * 100)
        excess.append((r.weights_bytes / r.naive_bytes - 1) * 100)
    put("ModelErrMax", max(errs) if errs else float("nan"), "{:.1f}")
    put("NaiveExcessMax", max(excess) if excess else float("nan"), "{:.0f}")
    pb7 = A.param_breakdown("7b")
    put("EmbedShareSeven", (pb7.embed + pb7.lm_head) / pb7.total * 100, "{:.0f}")

    def wratio(size, prec):
        try:
            return hfw.loc[(size, "bf16")].weights_bytes / hfw.loc[(size, prec)].weights_bytes
        except KeyError:
            return float("nan")

    putx("WtNFSeven", wratio("7b", "nf4"), 2)
    putx("WtIntEightSeven", wratio("7b", "int8"), 2)
    putx("WtNFThree", wratio("3b", "nf4"), 2)
    vlw = vl_load.set_index(["size", "precision"]) if not vl_load.empty else pd.DataFrame()

    def vget(size, prec, col):
        try:
            return float(vlw.loc[(size, prec)][col])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    putx("WtAWQSeven", vget("7b", "bf16", "load_gib") / vget("7b", "awq-int4", "load_gib"), 2)
    hfm = sp[(sp.backend == "hf") & (sp["size"] == "7b") & (sp.tag == "grid") & (sp.output_len == 128)]
    putx("MemSmallNF", first(hfm[(hfm.precision == "nf4") & (hfm.batch == 1) & (hfm.input_len == 128)], "mem_saving"), 2)
    putx("MemBigNF", first(hfm[(hfm.precision == "nf4") & (hfm.batch == 32) & (hfm.input_len == 2048)], "mem_saving"), 2)
    putx("MemBigInt", first(hfm[(hfm.precision == "int8") & (hfm.batch == 32) & (hfm.input_len == 2048)], "mem_saving"), 2)
    a7 = agg[(agg.backend == "hf") & (agg["size"] == "7b") & (agg.batch == 32) & (agg.input_len == 2048) & (agg.tag == "grid")]
    act = {r.precision: r.peak_alloc - r.weights - r.kv_bytes for r in a7.itertuples()}
    putx("ActIntRatio", act.get("int8", np.nan) / act.get("bf16", np.nan), 1)
    put("KVawqSeven", f"{vget('7b', 'awq-int4', 'kv_cache_tokens') / 1e3:.0f}K" if ok(vget('7b', 'awq-int4', 'kv_cache_tokens')) else "n/a")
    put("KVbfSeven", f"{vget('7b', 'bf16', 'kv_cache_tokens') / 1e3:.0f}K" if ok(vget('7b', 'bf16', 'kv_cache_tokens')) else "n/a")
    putx("KVratioAWQ", vget("7b", "awq-int4", "kv_cache_tokens") / vget("7b", "bf16", "kv_cache_tokens"), 2)
    lt = {p: hfw.loc[("7b", p)].load_time_s for p in ("bf16", "int8") if ("7b", p) in hfw.index}
    put("LoadIntEightSeven", lt.get("int8", np.nan), "{:.1f}")
    putx("LoadIntRatio", lt.get("int8", np.nan) / lt.get("bf16", np.nan), 0)
    # 14B: does BF16 fit in vLLM?
    k14 = vget("14b", "awq-int4", "kv_cache_tokens")
    bf14 = vl_load[(vl_load["size"] == "14b") & (vl_load.precision == "bf16")] if not vl_load.empty else vl_load
    if len(bf14) and bf14.kv_cache_tokens.notna().iloc[0]:
        note = (f"In BF16 the 14B model (29.5\\,GB of weights) only starts when vLLM may use 95\\% of the GPU, "
                f"and then has room for just {bf14.kv_cache_tokens.iloc[0] / 1e3:.0f}K cached tokens, "
                f"against {k14 / 1e3:.0f}K with AWQ-INT4 at the default 85\\%.")
    elif len(bf14):
        note = (f"In BF16 the 14B model's 29.5\\,GB of weights leave no room for any KV cache even when vLLM may "
                f"use 95\\% of the GPU, so it cannot be served at all, while its AWQ-INT4 version runs with "
                f"{k14 / 1e3:.0f}K cached tokens.")
    else:
        note = ""
    put("FourteenNote", note)

    # ------------------------------------------------------------------ Q2 latency
    ttft_wo = grid[(grid.backend == "vllm") & (grid["size"] == "7b") & grid.precision.isin(["awq-int4", "gptq-int4", "gptq-int8"])
                   & (grid.input_len >= 512)].speedup_ttft
    put("TTFTwoMax", (abs(1 - ttft_wo)).max() * 100 if len(ttft_wo) else float("nan"), "{:.0f}")
    putx("TTFTnfSlow", 1 / cell("hf", "7b", "nf4", 1, 512, "speedup_ttft"), 2)
    putx("TTFTfpLong", cell("vllm", "7b", "fp8", 1, 2048, "speedup_ttft"), 2)
    putx("TTFTintLong", cell("hf", "7b", "int8", 16, 2048, "speedup_ttft"), 2)
    tb = cell("vllm", "7b", "bf16", 1, 512, "tpot")
    put("TPOTbfVllmSeven", tb * 1e3, "{:.2f}")
    floor = hfw.loc[("7b", "bf16")].weights_bytes / (bw * 1e12) if ok(bw) and ("7b", "bf16") in hfw.index else float("nan")
    put("BWutilVllm", floor / tb * 100 if ok(floor) else float("nan"), "{:.0f}")
    putx("AWQspeedup", cell("vllm", "7b", "awq-int4", 1, 512, "speedup_tpot"), 2)
    putx("GPTQeightspeedup", cell("vllm", "7b", "gptq-int8", 1, 512, "speedup_tpot"), 2)
    putx("FPeightspeedup", cell("vllm", "7b", "fp8", 1, 512, "speedup_tpot"), 2)
    putx("INTeightslowdown", 1 / cell("hf", "7b", "int8", 1, 512, "speedup_tpot"), 1)
    putx("NFslowdown", 1 / cell("hf", "3b", "nf4", 1, 512, "speedup_tpot"), 2)
    putx("NFsevenB", cell("hf", "7b", "nf4", 1, 512, "speedup_tpot"), 2)
    putx("NFthreeslow", 1 / cell("hf", "3b", "nf4", 1, 512, "speedup_tpot"), 2)
    thr = {p: cell("vllm", "7b", p, 256, 512, "out_tok_s") for p in ("bf16", "awq-int4", "fp8")}
    put("ThrBFmax", f"{thr['bf16']:,.0f}" if ok(thr["bf16"]) else "n/a")
    put("ThrAWQmax", f"{thr['awq-int4']:,.0f}" if ok(thr["awq-int4"]) else "n/a")
    put("ThrFPmax", f"{thr['fp8']:,.0f}" if ok(thr["fp8"]) else "n/a")
    putx("ThrFPratio", thr["fp8"] / thr["bf16"], 2)
    e3 = grid[(grid.backend == "hf") & (grid["size"] == "3b") & (grid.precision == "int8")].speedup_e2e
    putx("IntThreeWorst", e3.min() if len(e3) else float("nan"), 2)
    # reproducibility re-check: vLLM 7B BF16 re-measured after the shared-GPU interruption
    rc = agg[(agg.tag == "recheck") & (agg.backend == "vllm") & (agg["size"] == "7b") & (agg.precision == "bf16")]
    if len(rc):
        g0 = agg[(agg.tag == "grid") & (agg.backend == "vllm") & (agg["size"] == "7b") & (agg.precision == "bf16")]
        mm = rc.merge(g0, on=["batch", "input_len", "output_len"], suffixes=("", "_0"))
        dev = (abs(mm.tpot / mm.tpot_0 - 1)).max() * 100
        put("RecheckSentence", f"Re-measuring vLLM BF16 on the 7B model after the interruption reproduced the "
            f"original TPOT within {dev:.1f}\\%.")
    else:
        put("RecheckSentence", "")
    put("TPOTbfHFThree", cell("hf", "3b", "bf16", 1, 512, "tpot") * 1e3, "{:.1f}")
    w3 = hfw.loc[("3b", "bf16")].weights_bytes if ("3b", "bf16") in hfw.index else float("nan")
    put("FloorThree", w3 / (bw * 1e12) * 1e3 if ok(bw) else float("nan"), "{:.1f}")
    wins = A.excluded_windows()
    mins = sum((b - a_).total_seconds() for a_, b in wins) / 60
    put("ExcludedWindows", f"{len(wins)} window{'s' if len(wins) != 1 else ''}, {mins:.0f} minutes in total" if wins
        else "none occurred")
    tint3, tint7 = cell("hf", "3b", "int8", 1, 512, "tpot"), cell("hf", "7b", "int8", 1, 512, "tpot")
    putx("IntThreeOverSeven", tint3 / tint7, 2)

    # ------------------------------------------------------------------ Q3 crossover
    e7 = grid[(grid["size"] == "7b")]

    def rng(backend, prec, fn):
        v = e7[(e7.backend == backend) & (e7.precision == prec)].speedup_e2e.dropna()
        return fn(v) if len(v) else float("nan")

    putx("AWQmaxEtoE", rng("vllm", "awq-int4", max), 2)
    putx("FPminEtoE", rng("vllm", "fp8", min), 2)
    putx("FPmaxEtoE", rng("vllm", "fp8", max), 2)
    putx("INTminEtoE", rng("hf", "int8", min), 2)
    putx("INTmaxEtoE", rng("hf", "int8", max), 2)
    cx = A.crossover(e7, "speedup_e2e")

    def cross(backend, prec, L=512):
        r = cx[(cx.backend == backend) & (cx.precision == prec) & (cx.input_len == L)]
        return float(r.crossover_batch.iloc[0]) if len(r) else float("nan")

    put("AWQcross", cross("vllm", "awq-int4"), "{:.0f}")
    put("NFcross", cross("hf", "nf4"), "{:.0f}")

    # kernels
    _, layer = A.kernels("7b")

    have_k = not layer.empty and "kernel" in layer

    def kern(name, m=None, fn=None):
        if not have_k:
            return float("nan")
        r = layer[layer.kernel == name].sort_values("m")
        if r.empty:
            return float("nan")
        if fn:
            return fn(r)
        return float(r[r.m == m].speedup.iloc[0]) if (r.m == m).any() else float("nan")

    def kcross(name):
        if not have_k:
            return float("nan")
        r = layer[layer.kernel == name].sort_values("m")
        s, m = r.speedup.to_numpy(), r.m.to_numpy()
        for i in range(1, len(s)):
            if s[i - 1] > 1.05 >= s[i]:     # advantage shrinks to within 5% of BF16
                x0, x1 = math.log2(m[i - 1]), math.log2(m[i])
                y0, y1 = math.log(s[i - 1] / 1.05), math.log(s[i] / 1.05)
                return 2 ** (x0 - y0 * (x1 - x0) / (y1 - y0))
        return float("nan")

    putx("MarlinFourMone", kern("marlin-w4a16", 1), 1)
    put("MarlinFourCross", kcross("marlin-w4a16"), "{:.0f}")
    putx("MarlinEightMone", kern("marlin-w8a16", 1), 1)
    put("MarlinEightCross", kcross("marlin-w8a16"), "{:.0f}")
    putx("FPkernMone", kern("fp8-w8a8", 1), 1)
    putx("FPkernMmax", kern("fp8-w8a8", fn=lambda r: float(r.speedup.iloc[-1])), 1)
    putx("NFkernMone", kern("bnb-nf4", 1), 2)
    putx("NFkernMmax", kern("bnb-nf4", fn=lambda r: float(r[r.m >= 16].speedup.min())), 2)
    lo, hi = kern("bnb-int8", fn=lambda r: float(r.speedup.min())), kern("bnb-int8", fn=lambda r: float(r.speedup.max()))
    put("INTkernRange", f"{lo:.2f}--{hi:.2f}$\\times$" if ok(lo) else "n/a")

    # profile
    prof = A.profiles()
    if not prof.empty:
        p7 = prof[prof["size"] == "7b"].drop_duplicates(["precision", "batch"], keep="last").set_index(["precision", "batch"])

        def pv(prec, B, col):
            try:
                return float(p7.loc[(prec, B)][col])
            except KeyError:
                return float("nan")

        put("BusyBFone", pv("bf16", 1, "gpu_busy_frac") * 100, "{:.0f}")
        put("BusyIntone", pv("int8", 1, "gpu_busy_frac") * 100, "{:.0f}")
        put("BusyNFone", pv("nf4", 1, "gpu_busy_frac") * 100, "{:.0f}")
        putx("KernIntRatio", pv("int8", 1, "kernels_per_step") / pv("bf16", 1, "kernels_per_step"), 1)
    else:
        for k in ("BusyBFone", "BusyIntone", "BusyNFone", "KernIntRatio"):
            put(k, "n/a")

    # energy
    en = grid[(grid["size"] == "7b") & (grid.input_len == 512)]

    def ev(backend, prec, B, col="j_per_tok"):
        return first(en[(en.backend == backend) & (en.precision == prec) & (en.batch == B)], col)

    put("EnAWQone", ev("vllm", "awq-int4", 1), "{:.1f}")
    put("EnBFone", ev("vllm", "bf16", 1), "{:.1f}")
    putx("EnRatioAWQone", ev("vllm", "bf16", 1) / ev("vllm", "awq-int4", 1), 1)
    put("EnFPmax", ev("vllm", "fp8", 256), "{:.2f}")
    put("EnBFmax", ev("vllm", "bf16", 256), "{:.2f}")
    put("PowerInt", ev("hf", "int8", 1, "power"), "{:.0f}")
    pw = en[en.backend == "vllm"].power
    put("PowerRange", f"{pw.min():.0f}--{pw.max():.0f}" if len(pw) else "n/a")

    # ------------------------------------------------------------------ real workload (ShareGPT)
    sv_, sh_ = A.serving()

    def srv(prec, c, col):
        r = sv_[(sv_["size"] == "7b") & (sv_.precision == prec) & (sv_.concurrency == c)] if not sv_.empty else sv_
        return float(r[col].iloc[0]) if len(r) else float("nan")

    def hfs(prec, col="output_throughput"):
        r = sh_[(sh_["size"] == "7b") & (sh_.precision == prec)] if not sh_.empty else sh_
        return float(r[col].iloc[0]) if len(r) else float("nan")

    put("SrvTPOTbfOne", srv("bf16", 1, "median_tpot_ms"), "{:.1f}")
    put("SrvTPOTawqOne", srv("awq-int4", 1, "median_tpot_ms"), "{:.1f}")
    put("SrvTPOTfpOne", srv("fp8", 1, "median_tpot_ms"), "{:.1f}")
    for k, prec in (("SrvThrFP", "fp8"), ("SrvThrAWQ", "awq-int4"), ("SrvThrBF", "bf16")):
        v = srv(prec, 64, "output_throughput")
        put(k, f"{v:,.0f}" if ok(v) else "n/a")
    putx("SrvGap", srv("bf16", 16, "output_throughput") / hfs("bf16"), 1)
    put("SrvTTFTvlSixteen", srv("bf16", 16, "median_ttft_ms"), "{:.0f}")
    put("SrvTTFThf", hfs("bf16", "p50_ttft_ms"), "{:.0f}")
    for k, prec in (("SrvHFnf", "nf4"), ("SrvHFint", "int8")):
        v = hfs(prec)
        put(k, f"{v:,.0f}" if ok(v) else "n/a")

    # ------------------------------------------------------------------ output length (decode share)
    ol = agg[(agg["size"] == "7b") & (agg.input_len == 512) & (agg.batch.isin([1, 16]))
             & (((agg.tag == "grid") & (agg.output_len == 128)) | (agg.tag == "outlen"))]
    ol = A.with_speedup(ol).set_index(["backend", "precision", "batch", "output_len"]).speedup_e2e

    def osp(b, p_, B, n):
        return float(ol.get((b, p_, B, n), float("nan")))

    for tag, (b, p_) in {"AWQ": ("vllm", "awq-int4"), "FP": ("vllm", "fp8"), "GQE": ("vllm", "gptq-int8"),
                         "INT": ("hf", "int8"), "NF": ("hf", "nf4")}.items():
        for B, bt in ((1, "One"), (16, "Sixteen")):
            putx(f"Out{tag}{bt}Short", osp(b, p_, B, 32), 2)
            putx(f"Out{tag}{bt}Long", osp(b, p_, B, 512), 2)

    # ------------------------------------------------------------------ Q4 quality
    ppl, gsm = A.quality()
    acc = (gsm.set_index(["backend", "size", "precision"]).accuracy * 100) if not gsm.empty else pd.Series(dtype=float)
    pp = ppl.set_index(["backend", "size", "precision"]).ppl if not ppl.empty else pd.Series(dtype=float)

    def g(b, s_, p):
        return float(acc.get((b, s_, p), float("nan")))

    def q(b, s_, p):
        return float(pp.get((b, s_, p), float("nan")))

    base = g("hf", "7b", "bf16")
    eight = [(b, p) for b, p in (("hf", "int8"), ("vllm", "gptq-int8"), ("vllm", "fp8"))]
    four = [(b, p) for b, p in (("hf", "nf4"), ("vllm", "awq-int4"), ("vllm", "gptq-int4"))]
    d8 = [g(b, "7b", p) - base for b, p in eight if ok(g(b, "7b", p))]
    d4 = [g(b, "7b", p) - base for b, p in four if ok(g(b, "7b", p))]
    p8 = [q(b, "7b", p) - q("hf", "7b", "bf16") for b, p in eight if ok(q(b, "7b", p))]
    p4 = [q(b, "7b", p) - q("hf", "7b", "bf16") for b, p in four if ok(q(b, "7b", p))]
    sig = []
    for b, p in eight + four:
        r = A.gsm8k_paired(("hf", "7b", "bf16"), (b, "7b", p))
        if r:
            sig.append((p, r["p_value"]))
    n_sig = sum(pv < 0.05 for _, pv in sig)
    a14, a7 = g("vllm", "14b", "awq-int4"), base
    m14 = vget("14b", "awq-int4", "load_gib") * 2**30 / 1e9
    m7 = hfw.loc[("7b", "bf16")].weights_bytes / 1e9 if ("7b", "bf16") in hfw.index else float("nan")
    put("AccBFSeven", base, "{:.1f}")

    def sg(v, nd=1, tex=True):
        """Signed number with a real minus sign."""
        t = f"{v:+.{nd}f}"
        return f"${t}$" if tex else t.replace("-", "\u2212")

    def span(vals, nd=1, tex=True):
        lo, hi = min(vals), max(vals)
        return sg(lo, nd, tex) if round(lo, nd) == round(hi, nd) else f"{sg(lo, nd, tex)} to {sg(hi, nd, tex)}"

    def four_cost(size, tex=True):
        """Describe the 4-bit GSM8K change of one model size, honouring the McNemar tests."""
        rows = []
        for b, p in four:
            t = A.gsm8k_paired(("hf", size, "bf16"), (b, size, p))
            if t:
                rows.append((t["delta_pp"], t["p_value"]))
        if not rows:
            return None
        d = [r[0] for r in rows]
        if all(r[1] >= 0.05 for r in rows):
            return f"nothing measurable on the {size.upper()} model ({span(d, 1, tex)} points, none significant)"
        return f"{span([-v for v in d], 1, tex).replace('+', '')} points on the {size.upper()} model".replace(" to ", "--" if tex else "\u2013") \
            if all(v < 0 for v in d) else f"{span(d, 1, tex)} points on the {size.upper()} model"

    if d8 and d4:
        put("DeltaEightMax", max(abs(v) for v in d8), "{:.1f}")
        put("DeltaFourMin", sg(min(d4)))
        put("DeltaFourMax", sg(max(d4)))
        put("PPLEightMax", max(p8) if p8 else float("nan"), "{:.2f}")
        put("PPLFourMax", max(p4) if p4 else float("nan"), "{:.2f}")
        put("NumSig", f"{n_sig} of {len(sig)}")
        t14 = A.gsm8k_paired(("hf", "7b", "bf16"), ("vllm", "14b", "awq-int4"))
        p14 = ("$p<0.001$" if t14["p_value"] < 0.001 else f"$p={t14['p_value']:.3f}$") if t14 else ""
        p14_txt = p14.replace("$", "")
        c7, c3 = four_cost("7b"), four_cost("3b")
        c7_t, c3_t = four_cost("7b", False), four_cost("3b", False)
        d8_max = max(abs(v) for v in d8)
        better = ok(a14) and ok(a7) and a14 > a7 and t14 and t14["p_value"] < 0.05
        cost = f"4-bit formats cost {c7}" + (f" but {c3}" if c3 else "")
        cost_t = f"4-bit formats cost {c7_t}" + (f" but {c3_t}" if c3_t else "")
        if better:
            put("QualityAbstract", f"On GSM8K, 8-bit formats stay within {d8_max:.1f} points of BF16 and {cost}; a 14B "
                f"model in AWQ-INT4 ({m14:.1f}\\,GB) outscores the 7B model in BF16 ({m7:.1f}\\,GB).")
            put("QualityGuide", f"Qwen2.5-14B in AWQ-INT4 needs {m14:.1f}\\,GB of weights, less than the {m7:.1f}\\,GB "
                f"of Qwen2.5-7B in BF16, and scores {a14:.1f}\\% on GSM8K against {a7:.1f}\\% (McNemar {p14}).")
            put("QualityConclusion", "8-bit formats preserve quality, 4-bit formats cost little on a 7B model, and a larger "
                "quantized model is the better use of a fixed memory budget.")
            put("QualityReadme", f"On GSM8K, 8-bit formats stay within {d8_max:.1f} points of BF16 and {cost_t}; "
                f"Qwen2.5-14B in AWQ-INT4 ({m14:.1f} GB) scores {a14:.1f}% vs {a7:.1f}% for Qwen2.5-7B in BF16 "
                f"({m7:.1f} GB, McNemar {p14_txt}).")
        else:
            put("QualityAbstract", f"On GSM8K, 8-bit formats stay within {d8_max:.1f} points of BF16 and {cost}.")
            put("QualityGuide", (f"In our runs Qwen2.5-14B in AWQ-INT4 ({m14:.1f}\\,GB) scores {a14:.1f}\\% on GSM8K "
                                 f"against {a7:.1f}\\% for Qwen2.5-7B in BF16 ({m7:.1f}\\,GB), so the larger quantized "
                                 "model is not clearly better; measure task quality before switching.")
                if ok(a14) else "Quality must be checked per task before switching formats.")
            put("QualityConclusion", "8-bit formats preserve quality; 4-bit formats need a per-task check.")
            put("QualityReadme", f"On GSM8K, 8-bit formats stay within {d8_max:.1f} points of BF16 and {cost_t}.")
    else:
        for k in ("QualityAbstract", "QualityGuide", "QualityConclusion", "QualityReadme"):
            put(k, "")

    # per-size quality deltas, backend noise and significance (McNemar, paired GSM8K answers)
    def deltas(size, group):
        rows = []
        for b, p in group:
            a = g(b, size, p)
            if not ok(a):
                continue
            t = A.gsm8k_paired(("hf", size, "bf16"), (b, size, p))
            rows.append((b, p, a - g("hf", size, "bf16"), q(b, size, p) - q("hf", size, "bf16"),
                         t["p_value"] if t else float("nan")))
        return rows

    for size, tag in (("3b", "Three"), ("7b", "Seven")):
        r8, r4 = deltas(size, eight), deltas(size, four)
        if r4:
            put(f"DFourMin{tag}", sg(min(r[2] for r in r4)))
            put(f"DFourMax{tag}", sg(max(r[2] for r in r4)))
            put(f"PPLFourMin{tag}", sg(min(r[3] for r in r4), 2))
            put(f"PPLFourMax{tag}", sg(max(r[3] for r in r4), 2))
        if r8:
            put(f"DEightAbs{tag}", max(abs(r[2]) for r in r8), "{:.1f}")
            put(f"PPLEightAbs{tag}", max(abs(r[3]) for r in r8), "{:.2f}")
        tested = [r for r in r8 + r4 if ok(r[4])]
        if tested:
            sig8 = [r for r in r8 if ok(r[4]) and r[4] < 0.05]
            sig4 = [r for r in r4 if ok(r[4]) and r[4] < 0.05]
            put(f"NumSigEight{tag}", f"{len(sig8)} of {len([r for r in r8 if ok(r[4])])}")
            put(f"NumSigFour{tag}", f"{len(sig4)} of {len([r for r in r4 if ok(r[4])])}")
            put(f"MinPEight{tag}", min(r[4] for r in r8 if ok(r[4])), "{:.2f}")
        vb = g("vllm", size, "bf16") - g("hf", size, "bf16")
        put(f"BackendNoise{tag}", abs(vb), "{:.1f}")
        t = A.gsm8k_paired(("hf", size, "bf16"), ("vllm", size, "bf16"))
        if t:
            put(f"BackendFlips{tag}", t["a_only"] + t["b_only"], "{:d}")
    # significance of the 8-bit differences, Bonferroni over every quantized-vs-BF16 test on 3B and 7B
    fam = []
    for size in ("3b", "7b"):
        for b, p_ in eight + four:
            t = A.gsm8k_paired(("hf", size, "bf16"), (b, size, p_))
            if t:
                fam.append((size, b, p_, t["delta_pp"], t["p_value"]))
    if fam:
        alpha = 0.05 / len(fam)
        put("NumTests", len(fam), "{:d}")
        put("BonfAlpha", alpha, "{:.4f}")
        sig8 = [r for r in fam if (r[1], r[2]) in eight and r[4] < 0.05]
        p8min = min(r[4] for r in fam if (r[1], r[2]) in eight)
        if not sig8:
            put("EightSentence", f"none of these differences is significant (McNemar $p\\ge{p8min:.2f}$)")
        else:
            names = {("hf", "int8"): "LLM.int8()", ("vllm", "gptq-int8"): "GPTQ-INT8", ("vllm", "fp8"): "FP8"}
            desc = "; ".join(f"{names[(r[1], r[2])]} on {r[0].upper()}, ${r[3]:+.1f}$ points, $p={r[4]:.3f}$" for r in sig8)
            surv = [r for r in sig8 if r[4] < alpha]
            what = "an improvement" if all(r[3] > 0 for r in sig8) else "a change"
            put("EightSentence", (f"the only nominally significant difference is {what} ({desc})" if len(sig8) == 1 else
                                  f"nominally significant differences: {desc}") +
                (f", which does not survive a Bonferroni correction over the {len(fam)} comparisons"
                 if not surv else f", significant even after a Bonferroni correction over the {len(fam)} comparisons"))
        sig4_3 = [r for r in fam if r[0] == "3b" and (r[1], r[2]) in four]
        put("FourThreeSurvive", f"{sum(r[4] < alpha for r in sig4_3)} of {len(sig4_3)}")
    put("AccBFThree", g("hf", "3b", "bf16"), "{:.1f}")
    for k, (b, sz, p) in {"AccAWQFourteen": ("vllm", "14b", "awq-int4"), "AccFPFourteen": ("vllm", "14b", "fp8"),
                          "AccGPTQFourFourteen": ("vllm", "14b", "gptq-int4"), "AccBFFourteen": ("hf", "14b", "bf16"),
                          "AccAWQSeven": ("vllm", "7b", "awq-int4"), "PPLAWQFourteen": ("vllm", "14b", "awq-int4"),
                          "PPLBFSeven": ("hf", "7b", "bf16")}.items():
        put(k, g(b, sz, p) if k.startswith("Acc") else q(b, sz, p), "{:.1f}" if k.startswith("Acc") else "{:.2f}")
    put("GuideHead", "Prefer a larger quantized model to a smaller full-precision one." if ok(a14) and ok(a7) and a14 > a7
        else "Check task quality before trading model size for bit-width.")

    # the 14B GPTQ-INT4 perplexity outlier: ranking on raw text vs chat-formatted text
    tk = A.raw("ppl_topk_diag.jsonl")
    if not tk.empty:
        tk = tk.assign(corpus=tk.get("corpus", pd.Series(index=tk.index, dtype=object)).fillna("wikitext"))
        tk = tk.drop_duplicates(["size", "precision", "corpus"], keep="last").set_index(["size", "precision", "corpus"])
        others = ("awq-int4", "gptq-int8", "fp8")
        for corpus, tag in (("wikitext", "Wiki"), ("gsm8k-chat", "Chat")):
            if ("14b", "gptq-int4", corpus) in tk.index:
                put(f"TopOneGPTQ{tag}", tk.loc[("14b", "gptq-int4", corpus)].top1_acc * 100, "{:.0f}")
                o = [tk.loc[("14b", p_, corpus)].top1_acc * 100 for p_ in others if ("14b", p_, corpus) in tk.index]
                if o:
                    put(f"TopOneOthers{tag}", f"{min(o):.0f}--{max(o):.0f}" if round(min(o)) != round(max(o)) else f"{min(o):.0f}")
    q14 = [q("vllm", "14b", p_) for p_ in ("awq-int4", "gptq-int8", "fp8")]
    q14 = [v for v in q14 if ok(v)]
    put("PPLGPTQFourFourteen", q("vllm", "14b", "gptq-int4"), "{:.1f}")
    if q14:
        put("PPLOthersFourteen", f"{min(q14):.2f}--{max(q14):.2f}")
    put("AccGPTQFourFourteenPct", g("vllm", "14b", "gptq-int4"), "{:.1f}")

    def tex_safe(v):
        return re.sub(r"(?<!\\)%", r"\\%", str(v))   # a bare % would comment out the closing brace

    out = ROOT / "report" / "numbers.tex"
    out.write_text("% generated by scripts/make_numbers.py -- do not edit\n" +
                   "".join(f"\\newcommand{{\\{k}}}{{{tex_safe(v)}}}\n" for k, v in sorted(NUMS.items())))
    missing = [k for k, v in NUMS.items() if v == "n/a"]
    print(f"wrote {out} ({len(NUMS)} macros, {len(missing)} n/a: {', '.join(missing)})")


if __name__ == "__main__":
    main()
