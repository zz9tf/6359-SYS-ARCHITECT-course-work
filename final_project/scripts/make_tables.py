#!/usr/bin/env python
"""Summary tables: CSV (results/summary), LaTeX (report/tables) and Markdown (results/summary/*.md)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import analysis as A  # noqa: E402
from quantbench import configs  # noqa: E402
from quantbench import memory as M  # noqa: E402
from quantbench.results import RESULTS, ROOT  # noqa: E402

TEX = ROOT / "report" / "tables"
MD = RESULTS / "summary"
ORDER = [("hf", "bf16"), ("hf", "int8"), ("hf", "nf4"), ("vllm", "bf16"), ("vllm", "fp8"),
         ("vllm", "gptq-int8"), ("vllm", "awq-int4"), ("vllm", "gptq-int4")]
TEXNAME = {("hf", "bf16"): "HF BF16", ("hf", "int8"): "HF INT8 (LLM.int8)", ("hf", "nf4"): "HF NF4",
           ("vllm", "bf16"): "vLLM BF16", ("vllm", "fp8"): "vLLM FP8 W8A8", ("vllm", "gptq-int8"): "vLLM GPTQ W8A16",
           ("vllm", "awq-int4"): "vLLM AWQ W4A16", ("vllm", "gptq-int4"): "vLLM GPTQ W4A16"}


def write(df, name, caption, label, colfmt=None, note=None, wide=False):
    TEX.mkdir(parents=True, exist_ok=True)
    MD.mkdir(parents=True, exist_ok=True)
    A.save_csv(df, f"{name}.csv")
    (MD / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")
    colfmt = colfmt or "l" + "r" * (len(df.columns) - 1)
    body = df.to_latex(index=False, escape=False, column_format=colfmt, na_rep="--")
    body = re.sub(r"(?<!\\)%", r"\\%", body)
    env = "table*" if wide else "table"
    tex = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\setlength{\\tabcolsep}{4pt}" if wide else "\\setlength{\\tabcolsep}{3.2pt}", body.strip()]
    if note:
        tex.append(f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    tex.append(f"\\end{{{env}}}")
    (TEX / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"  wrote {name}")


SHORT = {("hf", "bf16"): "HF BF16", ("hf", "int8"): "HF INT8", ("hf", "nf4"): "HF NF4",
         ("vllm", "bf16"): "vLLM BF16", ("vllm", "fp8"): "vLLM FP8", ("vllm", "gptq-int8"): "vLLM GPTQ-8",
         ("vllm", "awq-int4"): "vLLM AWQ-4", ("vllm", "gptq-int4"): "vLLM GPTQ-4"}


def signed(v, nd):
    """Signed delta with a real minus sign; '--' when either side is missing."""
    if v is None or np.isnan(v):
        return "--"
    return f"${v:+.{nd}f}$"


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def fmt(x, nd=1, suffix=""):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.{nd}f}{suffix}"


def table_weights():
    hf, vl = A.loads()
    rows = []
    for size in ("3b", "7b", "14b"):
        pb = A.param_breakdown(size)
        ref = {}
        for b, p in ORDER:
            if b == "hf":
                r = hf[(hf["size"] == size) & (hf.precision == p)] if not hf.empty else hf
                meas = r.weights_bytes.iloc[0] / A.GB if len(r) else np.nan
            else:
                r = vl[(vl["size"] == size) & (vl.precision == p)] if not vl.empty else vl
                meas = r.load_gib.iloc[0] * A.GIB / A.GB if len(r) and r.load_gib.notna().iloc[0] else np.nan
            if p == "bf16":
                ref[b] = meas
            if np.isnan(meas) and size == "14b" and p in ("int8", "nf4"):
                continue
            bits = {"bf16": 16, "int8": 8, "nf4": 4, "fp8": 8, "gptq-int8": 8, "awq-int4": 4, "gptq-int4": 4}[p]
            naive = M.naive_weight_bytes(pb, bits) / A.GB
            theory = M.weight_bytes(pb, p) / A.GB
            rows.append({"Model": f"Qwen2.5-{size.upper()}", "Config": TEXNAME[(b, p)],
                         "Naive (GB)": fmt(naive, 2), "Model (GB)": fmt(theory, 2), "Measured (GB)": fmt(meas, 2),
                         "Ratio vs BF16": fmt(ref.get(b, np.nan) / meas, 2, "$\\times$") if p != "bf16" else "1.00$\\times$",
                         "Naive ratio": f"{16 / bits:.2f}$\\times$"})
    full = pd.DataFrame(rows)
    A.save_csv(full, "t_weight_memory_all.csv")
    (MD / "t_weight_memory_all.md").write_text(full.to_markdown(index=False) + "\n")
    write(full[full.Model != "Qwen2.5-3B"], "t_weight_memory",
          "Q1 --- weight memory: naive estimate ($\\#$params$\\times$bits/8), our analytic model (unquantized "
          "embedding/LM head/norms + quantization metadata), and measured on the GPU.", "tab:weights", colfmt="llrrrrr", wide=True)


def table_speed(size="7b", L=512):
    agg, _ = A.sweeps()
    d = agg[(agg["size"] == size) & (agg.output_len == 128) & (agg.tag == "grid") & (agg.input_len == L)]
    sp = A.with_speedup(agg)
    sp = sp[(sp["size"] == size) & (sp.output_len == 128) & (sp.tag == "grid") & (sp.input_len == L)]
    rows = []
    for b, p in ORDER:
        r = d[(d.backend == b) & (d.precision == p)].set_index("batch")
        s = sp[(sp.backend == b) & (sp.precision == p)].set_index("batch")
        if r.empty:
            continue
        bmax = 64 if 64 in r.index else r.index.max()
        rows.append({"Config": TEXNAME[(b, p)],
                     "TTFT B=1 (ms)": fmt(r.ttft.get(1, np.nan) * 1e3, 1),
                     "TPOT B=1 (ms)": fmt(r.tpot.get(1, np.nan) * 1e3, 2),
                     f"TPOT B={bmax} (ms)": fmt(r.tpot.get(bmax, np.nan) * 1e3, 2),
                     f"tok/s B={bmax}": fmt(r.out_tok_s.get(bmax, np.nan), 0),
                     "Decode speedup B=1": fmt(s.speedup_tpot.get(1, np.nan), 2, "$\\times$"),
                     "mJ/token B=1": fmt(r.j_per_tok.get(1, np.nan) * 1e3, 0),
                     f"mJ/token B={bmax}": fmt(r.j_per_tok.get(bmax, np.nan) * 1e3, 0)})
    write(pd.DataFrame(rows), "t_speed",
          f"Q2 --- latency, throughput and energy for Qwen2.5-{size.upper()} at prompt length {L} and 128 output "
          "tokens (mean of 3 runs). Speedup is relative to BF16 on the same backend.", "tab:speed", colfmt="lrrrrrrr", wide=True)


def table_crossover(size="7b"):
    agg, _ = A.sweeps()
    sp = A.with_speedup(agg)
    sp = sp[(sp["size"] == size) & (sp.output_len == 128) & (sp.tag == "grid")]
    cx = {m: A.crossover(sp, m) for m in ("speedup_tpot", "speedup_e2e")}

    def cross_txt(c):
        return "never" if c == 0 else ("all $B$" if np.isinf(c) else f"$B\\approx{c:.0f}$")

    rows = []
    for b, p in ORDER:
        if p == "bf16":
            continue
        row = {"Config": TEXNAME[(b, p)]}
        for m, name in (("speedup_tpot", "Decode"), ("speedup_e2e", "E2E")):
            r = cx[m][(cx[m].backend == b) & (cx[m].precision == p) & (cx[m].input_len == 512)] if not cx[m].empty else cx[m]
            if r.empty:
                continue
            r = r.iloc[0]
            row[f"{name} $B{{=}}1$"] = fmt(r.speedup_b1, 2, "$\\times$")
            row[f"{name} max $B$"] = f"{fmt(r.speedup_bmax, 2, '$\\times$')} ({r.bmax})"
            row[f"{name} wins up to"] = cross_txt(r.crossover_batch)
        rows.append(row)
    write(pd.DataFrame(rows), "t_crossover",
          f"Q3 --- speedup over BF16 on the same backend at batch size 1 and at the largest batch measured, and "
          f"the batch size at which the quantized configuration stops winning (Qwen2.5-{size.upper()}, $L=512$, 128 "
          "output tokens; linear interpolation in $\\log B$).", "tab:crossover", colfmt="lrrlrrl", wide=True)


def table_quality():
    ppl, gsm = A.quality()
    mm = A.mmlu()
    names = {"3b": "Qwen2.5-3B", "7b": "Qwen2.5-7B", "14b": "Qwen2.5-14B", "q3-8b": "Qwen3-8B"}

    def pick(df, b, size, p, col, scale=1.0):
        if df.empty:
            return np.nan
        r = df[(df.backend == b) & (df["size"] == size) & (df.precision == p)]
        return r[col].iloc[0] * scale if len(r) else np.nan

    rows = []
    for size in ("3b", "7b", "14b", "q3-8b"):
        first = True
        base_ppl, base_g, base_m = (pick(ppl, "hf", size, "bf16", "ppl"), pick(gsm, "hf", size, "bf16", "accuracy", 100),
                                    pick(mm, "hf", size, "bf16", "accuracy", 100))
        same = ppl[(ppl["size"] == size)].ppl if not ppl.empty else pd.Series(dtype=float)
        for b, p in ORDER:
            pv, gv, mv = pick(ppl, b, size, p, "ppl"), pick(gsm, b, size, p, "accuracy", 100), pick(mm, b, size, p, "accuracy", 100)
            if np.isnan(pv) and np.isnan(gv) and np.isnan(mv):
                continue
            ref = b == "hf" and p == "bf16"
            tg = None if ref else A.paired("gsm8k", ("hf", size, "bf16"), (b, size, p))
            tm = None if ref else A.paired("mmlu", ("hf", size, "bf16"), (b, size, p))
            outlier = len(same) > 2 and pv > 3 * same.median()
            rows.append({"Model": names[size] if first else "", "Config": SHORT[(b, p)],
                         "PPL": fmt(pv, 2) + ("$^{\\ddagger}$" if outlier else ""),
                         "$\\Delta$PPL": "--" if ref else signed(pv - base_ppl, 2),
                         "GSM8K": fmt(gv, 1), "$\\Delta$": "--" if ref else signed(gv - base_g, 1),
                         "$p$": fmt_p(tg["p_value"]) if tg else "--",
                         "MMLU": fmt(mv, 1), "$\\Delta$\\,": "--" if ref else signed(mv - base_m, 1),
                         "$p$\\,": fmt_p(tm["p_value"]) if tm else "--"})
            first = False
    write(pd.DataFrame(rows), "t_quality",
          "Q4 --- quality. WikiText-2 perplexity (2048-token windows), GSM8K accuracy (\\%, 1,319 questions, zero-shot "
          "CoT, greedy) and MMLU accuracy (\\%, 14,042 questions, zero-shot). Deltas are against HF BF16 of the same "
          "model; $p$ is the two-sided exact McNemar test of the paired answers against HF BF16.", "tab:quality",
          colfmt="llrrrrrrrr", wide=True,
          note="$^{\\ddagger}$Degraded on raw text but not on chat-formatted text; see \\cref{sec:q4}.")


def table_serving(size="7b"):
    vl, hf = A.serving()
    rows = []
    for (b, p) in ORDER:
        src = vl if b == "vllm" else hf
        if src.empty:
            continue
        r = src[(src["size"] == size) & (src.precision == p)]
        for _, x in r.sort_values("concurrency").iterrows():
            if b == "vllm" and x.concurrency not in (1, 64):
                continue
            ttft = x.get("median_ttft_ms", x.get("p50_ttft_ms"))
            tpot = x.get("median_tpot_ms", x.get("p50_tpot_ms"))
            rows.append({"Config": SHORT[(b, p)] + (" (static)" if b == "hf" else ""), "$C$": int(x.concurrency),
                         "req/s": fmt(x.get("request_throughput"), 2),
                         "tok/s": fmt(x.output_throughput, 0),
                         "TTFT p50 / p99": f"{fmt(ttft, 0)} / {fmt(x.get('p99_ttft_ms'), 0)}",
                         "TPOT p50 / p99": f"{fmt(tpot, 1)} / {fmt(x.get('p99_tpot_ms'), 1)}"})
    if rows:
        write(pd.DataFrame(rows), "t_serving",
              f"Real workload --- 300 ShareGPT requests replayed against Qwen2.5-{size.upper()} (reference output "
              "lengths). $C$: concurrent closed-loop clients for vLLM, static batch size for HF. Throughput in "
              "requests/s and output tokens/s; TTFT and TPOT in ms.", "tab:serving", colfmt="lrrrrr")


def table_generalization(a="7b", b="q3-8b"):
    sa, sb = A.family_pair(a, b)
    if not sb:
        return
    bc = sb["b_cmp"]

    def x(v):
        return "--" if v is None or np.isnan(v) else ("all $B$" if np.isinf(v) else f"{v:.2f}$\\times$")

    def dp(d, p_):
        return "--" if np.isnan(d) else f"{signed(d, 1)} ({fmt_p(p_)})"

    spec = [("Weight saving, AWQ-INT4 (vLLM)", "wsave_awq-int4", x), ("Weight saving, NF4 (HF)", "wsave_nf4", x),
            ("Decode speedup $B=1$, AWQ-INT4", "tpot_b1_awq-int4", x), ("Decode speedup $B=1$, FP8", "tpot_b1_fp8", x),
            ("Decode speedup $B=1$, NF4", "tpot_b1_nf4", x), ("Decode speedup $B=1$, LLM.int8()", "tpot_b1_int8", x),
            (f"Throughput $B={bc}$, AWQ-INT4", "thr_bfit_awq-int4", x), (f"Throughput $B={bc}$, FP8", "thr_bfit_fp8", x)]
    rows = [{"Metric (vs.\\ BF16)": n, configs.MODEL_LABEL[a]: f(sa.get(k, np.nan)), configs.MODEL_LABEL[b]: f(sb.get(k, np.nan))}
            for n, k, f in spec]
    for bench, label in (("gsm8k", "GSM8K"), ("mmlu", "MMLU")):
        for p, pl in (("fp8", "FP8"), ("int8", "LLM.int8()"), ("awq-int4", "AWQ-INT4"), ("nf4", "NF4")):
            rows.append({"Metric (vs.\\ BF16)": f"{label} $\\Delta$ pp ($p$), {pl}",
                         configs.MODEL_LABEL[a]: dp(sa.get(f"d_{bench}_{p}", np.nan), sa.get(f"p_{bench}_{p}", np.nan)),
                         configs.MODEL_LABEL[b]: dp(sb.get(f"d_{bench}_{p}", np.nan), sb.get(f"p_{bench}_{p}", np.nan))})
    write(pd.DataFrame(rows), "t_generalization",
          f"Generalization --- the headline results for {configs.MODEL_LABEL[a]} and {configs.MODEL_LABEL[b]} "
          "(non-thinking mode) on the same grid ($L=512$, 128 output tokens). Throughput is compared at the largest batch "
          f"at which both models' KV caches hold every request ($B={bc}$). Qwen3-8B has no official GPTQ checkpoint.",
          "tab:general", colfmt="lrr")


def main():
    for f in (table_weights, table_speed, table_crossover, table_quality, table_serving, table_generalization):
        try:
            f()
        except Exception as e:  # keep going so partial results still produce tables
            print(f"  {f.__name__} failed: {e!r}")


if __name__ == "__main__":
    main()
