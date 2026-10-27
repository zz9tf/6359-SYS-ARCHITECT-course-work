#!/usr/bin/env python
"""Summary tables: CSV (results/summary), LaTeX (report/tables) and Markdown (results/summary/*.md)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench import analysis as A  # noqa: E402
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
    body = body.replace("\\toprule", "\\toprule").replace("%", "\\%")
    env = "table*" if wide else "table"
    tex = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\setlength{\\tabcolsep}{4pt}" if wide else "\\setlength{\\tabcolsep}{3.2pt}", body.strip()]
    if note:
        tex.append(f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    tex.append(f"\\end{{{env}}}")
    (TEX / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"  wrote {name}")


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
    hf, vl = A.loads()
    rows = []
    for size in ("3b", "7b", "14b"):
        base_ppl = base_acc = None
        for b, p in ORDER:
            q = ppl[(ppl.backend == b) & (ppl["size"] == size) & (ppl.precision == p)] if not ppl.empty else ppl
            g = gsm[(gsm.backend == b) & (gsm["size"] == size) & (gsm.precision == p)] if not gsm.empty else gsm
            if q.empty and g.empty:
                continue
            pv = q.ppl.iloc[0] if len(q) else np.nan
            av = g.accuracy.iloc[0] * 100 if len(g) else np.nan
            if b == "hf" and p == "bf16":
                base_ppl, base_acc = pv, av
            rows.append({"Model": f"Qwen2.5-{size.upper()}", "Config": TEXNAME[(b, p)],
                         "WikiText-2 PPL": fmt(pv, 3),
                         "$\\Delta$PPL": fmt(pv - base_ppl, 3) if base_ppl is not None and p != "bf16" else "--",
                         "GSM8K (\\%)": fmt(av, 1),
                         "$\\Delta$acc (pp)": fmt(av - base_acc, 1) if base_acc is not None and p != "bf16" else "--"})
    write(pd.DataFrame(rows), "t_quality",
          "Q4 --- quality. WikiText-2 perplexity (2048-token windows) and GSM8K accuracy (1,319 questions, zero-shot "
          "CoT, greedy). Deltas are against HF BF16 of the same model.", "tab:quality", colfmt="llrrrr")


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
            rows.append({"Config": TEXNAME[(b, p)] + (" (static)" if b == "hf" else ""), "Conc.": int(x.concurrency),
                         "Output tok/s": fmt(x.output_throughput, 0), "p50 TTFT (ms)": fmt(ttft, 0),
                         "p99 TTFT (ms)": fmt(x.get("p99_ttft_ms"), 0), "p50 TPOT (ms)": fmt(tpot, 1),
                         "p99 TPOT (ms)": fmt(x.get("p99_tpot_ms"), 1)})
    if rows:
        write(pd.DataFrame(rows), "t_serving",
              f"Real workload --- 300 ShareGPT requests replayed against Qwen2.5-{size.upper()} (exact reference output "
              "lengths). vLLM: closed-loop clients at the given concurrency; HF: static batches of 16.", "tab:serving",
              colfmt="lrrrrrr")


def main():
    for f in (table_weights, table_speed, table_crossover, table_quality, table_serving):
        try:
            f()
        except Exception as e:  # keep going so partial results still produce tables
            print(f"  {f.__name__} failed: {e!r}")


if __name__ == "__main__":
    main()
