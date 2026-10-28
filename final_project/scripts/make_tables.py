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


def write(df, name, caption, label, colfmt=None, note=None, wide=False, pos="htbp", tex_names=None):
    """CSV and Markdown keep readable column names; tex_names renames columns for the LaTeX table only."""
    MD.mkdir(parents=True, exist_ok=True)
    A.save_csv(df, f"{name}.csv")
    (MD / f"{name}.md").write_text(df.to_markdown(index=False) + "\n")
    colfmt = colfmt or "l" + "r" * (len(df.columns) - 1)
    body = df.rename(columns=tex_names or {}).to_latex(index=False, escape=False, column_format=colfmt, na_rep="--")
    body = re.sub(r"(?<!\\)%", r"\\%", body)
    write_tex(name, body.strip(), caption, label, note, size="\\footnotesize" if wide else "\\small",
              sep="4pt" if wide else "3.2pt", pos=pos)


def write_tex(name, body, caption, label, note=None, size="\\small", sep="4pt", pos="htbp"):
    """One single-column table float (NeurIPS layout: the caption goes above the table).

    Appendix tables float anywhere (htbp); main-text tables pass pos="t".
    """
    TEX.mkdir(parents=True, exist_ok=True)
    tex = [f"\\begin{{table}}[{pos}]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\setlength{{\\tabcolsep}}{{{sep}}}", body]
    if note:
        tex.append(f"\\par\\vspace{{3pt}}\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    tex.append("\\end{table}")
    (TEX / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"  wrote {name}")


SHORT = {("hf", "bf16"): "HF BF16", ("hf", "int8"): "HF INT8", ("hf", "nf4"): "HF NF4",
         ("vllm", "bf16"): "vLLM BF16", ("vllm", "fp8"): "vLLM FP8", ("vllm", "gptq-int8"): "vLLM GPTQ-8",
         ("vllm", "awq-int4"): "vLLM AWQ-4", ("vllm", "gptq-int4"): "vLLM GPTQ-4"}


def signed(v, nd):
    """Signed delta with a real minus sign; '--' when either side is missing."""
    if v is None or np.isnan(v):
        return "--"
    if round(v, nd) == 0:
        return f"${0:.{nd}f}$"
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
          "Weight memory: the naive estimate ($\\#$params$\\times$bits/8), our model (embedding, LM head and norms "
          "stay in 16 bits; each format adds its scales), and the value measured on the GPU.", "tab:weights",
          colfmt="llrrrrr", wide=True)


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
    bm = 64   # short LaTeX headers: subscripts give the batch size
    short = {"TTFT B=1 (ms)": "TTFT$_{1}$", "TPOT B=1 (ms)": "TPOT$_{1}$", f"TPOT B={bm} (ms)": f"TPOT$_{{{bm}}}$",
             f"tok/s B={bm}": f"tok/s$_{{{bm}}}$", "Decode speedup B=1": "Speedup$_{1}$", "mJ/token B=1": "mJ/tok$_{1}$",
             f"mJ/token B={bm}": f"mJ/tok$_{{{bm}}}$"}
    write(pd.DataFrame(rows), "t_speed",
          f"Latency, throughput and energy of Qwen2.5-{size.upper()} with {L}-token prompts and 128 output tokens "
          "(mean of 3 runs). Subscripts give the batch size. TTFT and TPOT in ms; Speedup$_1$ is the decode speedup "
          "over BF16 on the same backend; energy is GPU energy per output token.", "tab:speed",
          colfmt="lrrrrrrr", wide=True, tex_names=short)


def table_crossover(size="7b"):
    agg, _ = A.sweeps()
    sp = A.with_speedup(agg)
    sp = sp[(sp["size"] == size) & (sp.output_len == 128) & (sp.tag == "grid")]
    cx = {m: A.crossover(sp, m) for m in ("speedup_tpot", "speedup_e2e")}

    def cross_txt(c):
        return "never" if c == 0 else ("all $B$" if np.isinf(c) else f"$B\\approx{c:.0f}$")

    rows, lines = [], []
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
        lines.append(" & ".join(row.values()) + " \\\\")
    df = pd.DataFrame(rows)
    A.save_csv(df, "t_crossover.csv")
    (MD / "t_crossover.md").write_text(df.to_markdown(index=False) + "\n")
    body = "\n".join(["\\begin{tabular}{lrrlrrl}", "\\toprule",
                      " & \\multicolumn{3}{c}{Decode step (TPOT)} & \\multicolumn{3}{c}{Whole request (E2E)} \\\\",
                      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                      "Config & $B{=}1$ & largest $B$ & faster up to & $B{=}1$ & largest $B$ & faster up to \\\\",
                      "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"])
    write_tex("t_crossover", body,
              f"Where each format stops paying off (Qwen2.5-{size.upper()}, 512-token prompts, 128 output tokens). "
              "Speedup over BF16 on the same backend at batch size 1 and at the largest batch we ran (in brackets), "
              "and the batch size up to which the format is faster (linear interpolation in $\\log B$).",
              "tab:crossover", size="\\footnotesize", sep="3.5pt")


def table_quality():
    ppl, gsm = A.quality()
    mm = A.mmlu()
    names = {"3b": "Qwen2.5-3B", "7b": "Qwen2.5-7B", "14b": "Qwen2.5-14B", "q3-8b": "Qwen3-8B", "q3-32b": "Qwen3-32B"}

    def pick(df, b, size, p, col, scale=1.0):
        if df.empty:
            return np.nan
        r = df[(df.backend == b) & (df["size"] == size) & (df.precision == p)]
        return r[col].iloc[0] * scale if len(r) else np.nan

    rows = []
    for size in ("3b", "7b", "14b", "q3-8b", "q3-32b"):
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
            r_ppl = ppl[(ppl.backend == b) & (ppl["size"] == size) & (ppl.precision == p)] if not ppl.empty else ppl
            forced = len(r_ppl) and "dtype_used" in r_ppl and r_ppl.dtype_used.iloc[0] not in ("auto", None)
            rows.append({"Model": names[size] if first else "", "Config": SHORT[(b, p)],
                         "PPL": fmt(pv, 2) + ("$^{\\ddagger}$" if outlier else "") + ("$^{\\S}$" if forced else ""),
                         "$\\Delta$PPL": "--" if ref else signed(pv - base_ppl, 2),
                         "GSM8K": fmt(gv, 1), "$\\Delta$": "--" if ref else signed(gv - base_g, 1),
                         "$p$": fmt_p(tg["p_value"]) if tg else "--",
                         "MMLU": fmt(mv, 1), "$\\Delta$\\,": "--" if ref else signed(mv - base_m, 1),
                         "$p$\\,": fmt_p(tm["p_value"]) if tm else "--"})
            first = False
    write(pd.DataFrame(rows), "t_quality",
          "All quality results. WikiText-2 perplexity (2048-token windows), GSM8K accuracy (\\%, 1,319 questions, "
          "zero-shot chain of thought, greedy) and MMLU accuracy (\\%, 14,042 questions, zero-shot). $\\Delta$ is the "
          "change against HF BF16 of the same model (Qwen3-32B has no BF16 run: it does not fit the GPU); $p$ is the "
          "two-sided exact McNemar test on the paired answers.", "tab:quality",
          colfmt="llrrrrrrrr", wide=True,
          note="$^{\\ddagger}$High only on raw text, not on chat-formatted text (\\cref{sec:q4}). "
               "$^{\\S}$Run with BF16 activations: with the checkpoint's FP16 activations, some 2048-token windows "
               "overflow (\\cref{sec:general}).")


def table_quality_main():
    """Main-text quality table: change against HF BF16, formats down (8-bit, then 4-bit), model sizes across."""
    ppl, gsm = A.quality()
    mm = A.mmlu()
    sizes = ["3b", "7b", "14b"]
    spec = [("hf", "int8", "LLM.int8()", "HF", "8/8"), ("vllm", "fp8", "FP8", "vLLM", "8/8"),
            ("vllm", "gptq-int8", "GPTQ-INT8", "vLLM", "8/16"), ("hf", "nf4", "NF4", "HF", "4/16"),
            ("vllm", "awq-int4", "AWQ-INT4", "vLLM", "4/16"), ("vllm", "gptq-int4", "GPTQ-INT4", "vLLM", "4/16")]

    def pick(df, b, size, p, col, scale=1.0):
        if df.empty:
            return np.nan
        r = df[(df.backend == b) & (df["size"] == size) & (df.precision == p)]
        return r[col].iloc[0] * scale if len(r) else np.nan

    tests = {bench: {(b, size, p): A.paired(bench, ("hf", size, "bf16"), (b, size, p))
                     for size in sizes for b, p, *_ in spec} for bench in ("gsm8k", "mmlu")}
    n_tests = {bench: sum(t is not None for t in d.values()) for bench, d in tests.items()}
    alpha = {bench: 0.05 / n for bench, n in n_tests.items() if n}

    def acc_delta(bench, df, b, size, p):
        v, base = pick(df, b, size, p, "accuracy", 100), pick(df, "hf", size, "bf16", "accuracy", 100)
        if np.isnan(v) or np.isnan(base):
            return "--"
        d = round(v - base, 1)
        t = tests[bench].get((b, size, p))
        txt = f"{d:+.1f}" if d != 0 else "0.0"
        return f"$\\mathbf{{{txt}}}$" if t and t["p_value"] < alpha[bench] else f"${txt}$"

    def ppl_delta(b, size, p):
        v, base = pick(ppl, b, size, p, "ppl"), pick(ppl, "hf", size, "bf16", "ppl")
        if np.isnan(v) or np.isnan(base):
            return "--"
        d = v - base
        return (f"${d:+.2f}$" if round(d, 2) != 0 else "$0.00$") + ("$^{\\ddagger}$" if d > 3 * base else "")

    lines = ["BF16 (value) & HF & 16/16 & " + " & ".join(
        f"{pick(ppl, 'hf', z, 'bf16', 'ppl'):.2f} & {pick(gsm, 'hf', z, 'bf16', 'accuracy', 100):.1f} & "
        f"{pick(mm, 'hf', z, 'bf16', 'accuracy', 100):.1f}" for z in sizes) + " \\\\", "\\midrule"]
    for i, (b, p, name, bk, wa) in enumerate(spec):
        if i == 3:
            lines.append("\\midrule")
        cells = [f"{ppl_delta(b, z, p)} & {acc_delta('gsm8k', gsm, b, z, p)} & {acc_delta('mmlu', mm, b, z, p)}"
                 for z in sizes]
        lines.append(f"{name} & {bk} & {wa} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    cells = [f"{ppl_delta('vllm', z, 'bf16')} & {acc_delta('gsm8k', gsm, 'vllm', z, 'bf16')} & "
             f"{acc_delta('mmlu', mm, 'vllm', z, 'bf16')}" for z in sizes]
    lines.append("BF16 (noise) & vLLM & 16/16 & " + " & ".join(cells) + " \\\\")
    body = "\n".join(["\\begin{tabular}{lllrrrrrrrrr}", "\\toprule",
                      " & & & \\multicolumn{3}{c}{Qwen2.5-3B} & \\multicolumn{3}{c}{Qwen2.5-7B} & "
                      "\\multicolumn{3}{c}{Qwen2.5-14B} \\\\",
                      "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
                      "Format & Backend & W/A & PPL & GSM8K & MMLU & PPL & GSM8K & MMLU & PPL & GSM8K & MMLU \\\\",
                      "\\midrule", *lines, "\\bottomrule", "\\end{tabular}"])
    write_tex("t_quality_main", body,
              "Quality cost of each format. First row: BF16 values (WikiText-2 perplexity; GSM8K and MMLU accuracy "
              "in \\%). Other rows: change against HF BF16 of the same model (accuracy in points). \\textbf{Bold}: "
              f"significant after Bonferroni correction over the {n_tests.get('mmlu', 0)} comparisons per benchmark "
              "(exact McNemar test). Last row: BF16 on vLLM, the noise floor.",
              "tab:quality_main", size="\\footnotesize", sep="2.9pt", pos="t",
              note="$^{\\ddagger}$Large only on raw text; on chat-formatted text this checkpoint matches the other "
                   "14B formats (\\cref{sec:q4}).")


def table_serving_main(size="7b"):
    """Main-text serving table: throughput and latency at 1, 16 and 64 concurrent requests."""
    vl, hf = A.serving()
    if vl.empty:
        return

    def get(src, p, c, col, alt=None):
        r = src[(src["size"] == size) & (src.precision == p) & (src.concurrency == c)] if not src.empty else src
        if r.empty:
            return np.nan
        x = r.iloc[0]
        v = x.get(col, np.nan)
        return x.get(alt, np.nan) if (v is None or (isinstance(v, float) and np.isnan(v))) and alt else v

    def n0(v):
        return "--" if v is None or np.isnan(v) else f"{v:,.0f}"

    lines = []
    for b, p in ORDER:
        if b == "vllm":
            cells = [n0(get(vl, p, 1, "output_throughput")), fmt(get(vl, p, 1, "median_tpot_ms"), 1),
                     n0(get(vl, p, 16, "output_throughput")), n0(get(vl, p, 16, "median_ttft_ms")),
                     n0(get(vl, p, 64, "output_throughput")), fmt(get(vl, p, 64, "median_tpot_ms"), 1)]
        else:
            cells = ["--", "--", n0(get(hf, p, 16, "output_throughput")),
                     n0(get(hf, p, 16, "p50_ttft_ms", "median_ttft_ms")), "--", "--"]
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{SHORT[(b, p)]}{' (static)' if b == 'hf' else ''} & " + " & ".join(cells) + " \\\\")
        if (b, p) == ("hf", "nf4"):
            lines.append("\\midrule")
    body = "\n".join(["\\begin{tabular}{lrrrrrr}", "\\toprule",
                      " & \\multicolumn{2}{c}{$C=1$} & \\multicolumn{2}{c}{$C=16$} & \\multicolumn{2}{c}{$C=64$} \\\\",
                      "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
                      "Config & tok/s & TPOT & tok/s & TTFT & tok/s & TPOT \\\\", "\\midrule",
                      *lines, "\\bottomrule", "\\end{tabular}"])
    write_tex("t_serving_main", body,
              f"Real traffic: 300 ShareGPT requests replayed against Qwen2.5-{size.upper()} with their reference answer "
              "lengths. $C$ is the number of concurrent requests (HF runs static batches of 16). Output tokens per "
              "second, and median TTFT and TPOT in ms.", "tab:serving_main", size="\\small", sep="5pt")


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
              f"All serving results: 300 ShareGPT requests replayed against Qwen2.5-{size.upper()} (reference answer "
              "lengths). $C$: concurrent clients for vLLM, static batch size for HF. Throughput in requests/s and "
              "output tokens/s; TTFT and TPOT in ms (median / 99th percentile).", "tab:serving", colfmt="lrrrrr")


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
          f"The main results for {configs.MODEL_LABEL[a]} and {configs.MODEL_LABEL[b]} (non-thinking mode) on the "
          "same grid (512-token prompts, 128 output tokens). Throughput is compared at the largest batch at which both "
          f"models' KV caches hold every request ($B={bc}$). Qwen3-8B has no official GPTQ checkpoint.",
          "tab:general", colfmt="lrr")


def main():
    for f in (table_weights, table_speed, table_crossover, table_quality, table_quality_main, table_serving,
              table_serving_main, table_generalization):
        try:
            f()
        except Exception as e:  # keep going so partial results still produce tables
            print(f"  {f.__name__} failed: {e!r}")


if __name__ == "__main__":
    main()
