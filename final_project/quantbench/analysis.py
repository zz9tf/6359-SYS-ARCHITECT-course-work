"""Aggregate raw JSONL measurements into tidy pandas tables shared by figures and tables."""
import json
import math

import numpy as np
import pandas as pd

from .results import RESULTS, read_jsonl

KEYS = ["backend", "size", "precision", "tag", "batch", "input_len", "output_len"]
CELL = ["backend", "size", "tag", "batch", "input_len", "output_len"]
GIB = 2**30
GB = 1e9


def excluded_windows():
    """Time windows in which another user's process shared the GPU (logged by the job queue)."""
    return [(pd.Timestamp(w["start"]), pd.Timestamp(w["end"])) for w in read_jsonl(RESULTS / "excluded_windows.jsonl")]


def in_excluded_window(ts):
    t = pd.to_datetime(ts)
    bad = pd.Series(False, index=t.index)
    for a, b in excluded_windows():
        bad |= (t >= a) & (t <= b)
    return bad


def raw(name, keep_excluded=False):
    """Rows of a results file, minus rows written while the GPU was shared with another user."""
    df = pd.DataFrame(read_jsonl(RESULTS / name))
    if df.empty or keep_excluded or "timestamp" not in df:
        return df
    return df[~in_excluded_window(df.timestamp)].reset_index(drop=True)


def sweeps():
    """Mean/std per sweep cell for both backends, plus the list of out-of-memory cells."""
    frames = [df for df in (raw("hf_sweep.jsonl"), raw("vllm_sweep.jsonl")) if not df.empty]
    df = pd.concat(frames, ignore_index=True)
    for col in ("torch_peak_alloc_bytes", "kv_cache_bytes", "weights_bytes"):
        if col not in df:
            df[col] = np.nan
    ok = df[df.status == "ok"]
    agg = ok.groupby(KEYS).agg(
        ttft=("ttft_s", "mean"), ttft_std=("ttft_s", "std"),
        tpot=("tpot_s", "mean"), tpot_std=("tpot_s", "std"),
        e2e=("e2e_s", "mean"), e2e_std=("e2e_s", "std"),
        out_tok_s=("output_tok_s", "mean"), decode_tok_s=("decode_tok_s", "mean"),
        prefill_tok_s=("prefill_tok_s", "mean"), j_per_tok=("energy_j_per_out_token", "mean"),
        power=("avg_power_w", "mean"), clock=("avg_sm_clock_mhz", "mean"),
        peak_alloc=("torch_peak_alloc_bytes", "max"), kv_bytes=("kv_cache_bytes", "first"),
        weights=("weights_bytes", "first"), n=("ttft_s", "size")).reset_index()
    oom = df[df.status.isin(["oom", "oom_skipped"])][KEYS].drop_duplicates()
    # vLLM admits a batch in waves when its tokens exceed the KV-cache capacity; such cells do not
    # measure one concurrent batch, so they are flagged (and excluded from the comparisons).
    _, vl = loads()
    agg["fits_kv"] = True
    if not vl.empty and "kv_cache_tokens" in vl:
        cap = vl.set_index(["size", "precision"]).kv_cache_tokens.to_dict()
        is_vllm = agg.backend == "vllm"
        need = agg.batch * (agg.input_len + agg.output_len)
        lim = [cap.get((s, p), float("inf")) for s, p in zip(agg["size"], agg.precision)]
        agg.loc[is_vllm, "fits_kv"] = (need <= lim)[is_vllm]
    return agg, oom


def with_speedup(agg):
    """Join every row with the BF16 row of the same backend/size/cell; ratios > 1 mean faster."""
    agg = agg[agg.fits_kv] if "fits_kv" in agg else agg
    base = agg[agg.precision == "bf16"][CELL + ["ttft", "tpot", "e2e", "out_tok_s", "j_per_tok", "peak_alloc"]]
    m = agg.merge(base, on=CELL, suffixes=("", "_bf16"), how="left")
    m["speedup_ttft"] = m.ttft_bf16 / m.ttft
    m["speedup_tpot"] = m.tpot_bf16 / m.tpot
    m["speedup_e2e"] = m.e2e_bf16 / m.e2e
    m["energy_ratio"] = m.j_per_tok / m.j_per_tok_bf16
    m["mem_saving"] = m.peak_alloc_bf16 / m.peak_alloc
    return m


def crossover(sp, metric="speedup_tpot"):
    """Batch size at which a quantized config stops beating BF16 (log-linear interpolation)."""
    rows = []
    for (backend, size, prec, L, n), g in sp[sp.precision != "bf16"].groupby(
            ["backend", "size", "precision", "input_len", "output_len"]):
        g = g.dropna(subset=[metric]).sort_values("batch")
        if g.empty:
            continue
        b, s = g.batch.to_numpy(), g[metric].to_numpy()
        if (s > 1).all():
            cross, verdict = math.inf, "faster at every batch"
        elif (s <= 1).all():
            cross, verdict = 0.0, "never faster"
        else:
            i = int(np.argmax(s <= 1)) if s[0] > 1 else int(np.argmax(s > 1))
            x0, x1 = math.log2(b[i - 1]), math.log2(b[i])
            y0, y1 = math.log(s[i - 1]), math.log(s[i])
            cross = 2 ** (x0 + (0 - y0) * (x1 - x0) / (y1 - y0))
            verdict = "faster below" if s[0] > 1 else "faster above"
        rows.append({"backend": backend, "size": size, "precision": prec, "input_len": L, "output_len": n,
                     "crossover_batch": cross, "verdict": verdict, "speedup_b1": s[0], "speedup_bmax": s[-1],
                     "bmax": int(b[-1])})
    return pd.DataFrame(rows)


def loads():
    hf = raw("hf_load.jsonl")
    vl = raw("vllm_load.jsonl")
    if not hf.empty:
        hf = hf.drop(columns=["env"], errors="ignore").drop_duplicates(["size", "precision"], keep="last")
    if not vl.empty:
        vl = vl.drop(columns=["env"], errors="ignore").drop_duplicates(["size", "precision"], keep="last")
    return hf, vl


def serving():
    rows = []
    windows = excluded_windows()
    for f in sorted((RESULTS / "serve").glob("*.json")):
        d = json.loads(f.read_text())
        when = pd.to_datetime(d.get("date", ""), format="%Y%m%d-%H%M%S", errors="coerce")
        if pd.notna(when) and any(a <= when <= b for a, b in windows):
            continue
        size, prec, conc = f.stem.rsplit("_", 2)
        rows.append({"backend": "vllm", "size": size, "precision": prec, "concurrency": int(conc[1:]),
                     **{k: v for k, v in d.items() if isinstance(v, (int, float))}})
    vl = pd.DataFrame(rows)
    hf = raw("hf_sharegpt.jsonl")
    if not hf.empty:
        hf = hf.drop_duplicates(["size", "precision", "batch"], keep="last").rename(columns={"batch": "concurrency"})
    return vl, hf


def quality():
    ppl = raw("quality_ppl.jsonl")
    gsm = raw("quality_gsm8k.jsonl")
    if not ppl.empty:
        ppl = ppl.drop_duplicates(["backend", "size", "precision"], keep="last")
    if not gsm.empty:
        gsm = gsm.drop_duplicates(["backend", "size", "precision"], keep="last")
    return ppl, gsm


def hw_peaks():
    df = raw("hw_peaks.jsonl")
    if df.empty:
        return {}
    df = df.dropna(subset=["value"]) if "value" in df else df
    return df.drop_duplicates("metric", keep="last").set_index("metric")["value"].to_dict()


def kernels(size="7b"):
    df = raw("kernel_bench.jsonl")
    if df.empty:
        return df, df
    df = df[(df["size"] == size) & df.time_ms.notna()] if "time_ms" in df else df.iloc[0:0]
    layer = df.groupby(["kernel", "m"]).agg(time_ms=("time_ms", "sum"), n_shapes=("shape", "nunique")).reset_index()
    layer = layer[layer.n_shapes == layer.n_shapes.max()]
    base = layer[layer.kernel == "bf16"][["m", "time_ms"]].rename(columns={"time_ms": "bf16_ms"})
    layer = layer.merge(base, on="m")
    layer["speedup"] = layer.bf16_ms / layer.time_ms
    return df, layer


def profiles():
    return raw("profile_decode.jsonl")


def sharegpt_lengths():
    return raw("sharegpt_300_lengths.jsonl")


def param_breakdown(size):
    """Parameter breakdown recorded by the HF sweep; falls back to the model config on the Hub."""
    from .memory import ParamBreakdown, param_breakdown as from_config
    hf, _ = loads()
    r = hf[hf["size"] == size] if not hf.empty else hf
    if len(r):
        r = r.iloc[0]
        return ParamBreakdown(linear=int(r.params_linear), embed=int(r.params_embed),
                              lm_head=int(r.params_lm_head), other=int(r.params_other))
    from transformers import AutoConfig
    return from_config(AutoConfig.from_pretrained(f"Qwen/Qwen2.5-{size.upper()}-Instruct"))


def save_csv(df, name):
    out = RESULTS / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / name, index=False, float_format="%.6g")
    return out / name


def gsm8k_paired(a, b):
    """McNemar exact test between two GSM8K runs on the same questions.

    a, b: (backend, size, precision). Returns accuracy difference (b - a, percentage points), the
    discordant counts and the two-sided exact p-value.
    """
    from scipy.stats import binomtest

    def load(cfg):
        rows = read_jsonl(RESULTS / "gsm8k_outputs" / f"{cfg[0]}_{cfg[1]}_{cfg[2]}.jsonl")
        return {r["idx"]: bool(r["correct"]) for r in rows}

    ra, rb = load(a), load(b)
    common = sorted(set(ra) & set(rb))
    if not common:
        return None
    n10 = sum(ra[i] and not rb[i] for i in common)   # a right, b wrong
    n01 = sum(rb[i] and not ra[i] for i in common)   # b right, a wrong
    n = n10 + n01
    p = binomtest(n01, n, 0.5).pvalue if n else 1.0
    acc_a = sum(ra[i] for i in common) / len(common)
    acc_b = sum(rb[i] for i in common) / len(common)
    return {"n": len(common), "delta_pp": (acc_b - acc_a) * 100, "a_only": n10, "b_only": n01, "p_value": p}
