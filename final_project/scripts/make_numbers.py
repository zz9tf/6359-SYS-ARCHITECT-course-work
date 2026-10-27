#!/usr/bin/env python
"""Write report/numbers.tex: every number quoted in the report text, computed from the results.

Keeping the prose numbers in generated macros means the text can never drift from the data.
Missing inputs produce "n/a" so a partial run still compiles.
"""
import math
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
    bw = peaks.get("dram_copy_gbps", float("nan")) / 1e3
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
        note = (f"In BF16 the 14B model (29.5\\,GB of weights) does not start at all, while its AWQ-INT4 version "
                f"runs with {k14 / 1e3:.0f}K cached tokens.")
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
            if s[i - 1] > 1 >= s[i]:
                x0, x1, y0, y1 = math.log2(m[i - 1]), math.log2(m[i]), math.log(s[i - 1]), math.log(s[i])
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
        putx("KernIntRatio", pv("int8", 1, "kernels_per_step") / pv("bf16", 1, "kernels_per_step"), 1)
    else:
        for k in ("BusyBFone", "BusyIntone", "KernIntRatio"):
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

    out = ROOT / "report" / "numbers.tex"
    out.write_text("% generated by scripts/make_numbers.py -- do not edit\n" +
                   "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(NUMS.items())))
    missing = [k for k, v in NUMS.items() if v == "n/a"]
    print(f"wrote {out} ({len(NUMS)} macros, {len(missing)} n/a: {', '.join(missing)})")


if __name__ == "__main__":
    main()
