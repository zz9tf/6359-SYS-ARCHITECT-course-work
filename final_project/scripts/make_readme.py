#!/usr/bin/env python
"""Fill the KEY-FINDINGS and RESULTS blocks of README.md from the results (numbers stay in sync)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantbench.results import RESULTS, ROOT  # noqa: E402


def macros():
    text = (ROOT / "report" / "numbers.tex").read_text()
    out = {}
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text):
        val = (val.replace("$\\times$", "×").replace("{,}", ",").replace("\\,", " ").replace("--", "–")
               .replace("\\%", "%"))
        out[name] = val
    return out


def md_table(name):
    f = RESULTS / "summary" / f"{name}.md"
    if not f.exists():
        return "_(not available)_"
    return f.read_text().replace("$\\times$", "×").replace("$B\\approx", "B≈").replace("$B{=}1$", "B=1") \
        .replace("$B=", "B=").replace("$B$", "B").replace("$", "")


def main():
    m = macros()
    findings = f"""## Key findings

1. **Memory — smaller, but not 4× smaller.** Only the decoder's linear layers shrink; the embedding,
   LM head, norms and quantization scales stay 16-bit. On Qwen2.5-7B, 4-bit weights save
   **{m['WtNFSeven']}** (NF4) to **{m['WtAWQSeven']}** (AWQ) instead of 4×, and INT8 **{m['WtIntEightSeven']}** instead of 2×.
   At run time the KV cache and activations are not quantized: at 32 × 2048-token prompts the
   end-to-end saving falls to **{m['MemBigNF']}** (NF4) and **{m['MemBigInt']}** (LLM.int8(), whose activation
   workspace is {m['ActIntRatio']} that of BF16).
2. **Speed depends on the kernel, not the bit-width.** At batch size 1, AWQ-INT4 with vLLM's Marlin
   kernel decodes **{m['AWQspeedup']} faster** than BF16, while bitsandbytes LLM.int8() is **{m['INTeightslowdown']} slower**
   and NF4 is {m['NFsevenB']} faster on 7B but {m['NFthreeslow']} slower on 3B (whose BF16 step is host-bound).
3. **Weight-only gains vanish in the compute-bound regime.** AWQ-INT4 stops winning around
   **B ≈ {m['AWQcross']}** (512-token prompts) and never shortens prefill; **FP8 W8A8 is faster in every
   cell** ({m['FPminEtoE']}–{m['FPmaxEtoE']} end to end) and delivers **{m['ThrFPratio']}** the BF16 throughput at B = 256.
4. **The mechanism is visible at kernel level.** On the measured roofline, 4-bit weights quadruple
   arithmetic intensity, so Marlin W4A16 is {m['MarlinFourMone']} faster than cuBLAS at M = 1 and crosses over at
   M ≈ {m['MarlinFourCross']}; LLM.int8() keeps the GPU busy only {m['BusyIntone']}% of a decode step (BF16: {m['BusyBFone']}%).
5. **Energy follows latency.** At batch size 1, AWQ-INT4 needs {m['EnAWQone']} J per token against
   {m['EnBFone']} J for BF16; at B = 256 the GPU is power-capped and only FP8 still saves energy.
6. **Quality.** {m.get('QualityReadme', 'See the report.')}
"""
    results = f"""## Results at a glance

**Q1 — weight memory (GB)**

{md_table('t_weight_memory_all')}

![weights](figures/q1_weight_memory.png)
![memory scaling](figures/q1_memory_scaling.png)

**Q2 — latency and throughput, Qwen2.5-7B, prompt 512, 128 output tokens**

{md_table('t_speed')}

![latency](figures/q2_latency_throughput.png)

**Q3 — where each format stops winning (speedup over BF16 on the same backend)**

{md_table('t_crossover')}

![heatmaps](figures/q3_speedup_heatmaps.png)
![kernels](figures/q3_kernel_crossover.png)
![profile](figures/mech_decode_breakdown.png)

**Real workload — 300 ShareGPT requests**

{md_table('t_serving')}

![serving](figures/real_sharegpt_serving.png)

**Q4 — quality**

{md_table('t_quality')}

![quality](figures/q4_quality_vs_memory.png)
"""
    readme = ROOT / "README.md"
    text = readme.read_text()
    text = re.sub(r"<!-- KEY-FINDINGS -->.*?<!-- /KEY-FINDINGS -->|<!-- KEY-FINDINGS -->",
                  lambda _: "<!-- KEY-FINDINGS -->\n" + findings + "<!-- /KEY-FINDINGS -->", text, flags=re.S)
    text = re.sub(r"<!-- RESULTS -->.*?<!-- /RESULTS -->|<!-- RESULTS -->",
                  lambda _: "<!-- RESULTS -->\n" + results + "<!-- /RESULTS -->", text, flags=re.S)
    readme.write_text(text)
    missing = [k for k, v in m.items() if v == "n/a"]
    print(f"README updated; macros n/a: {missing}")


if __name__ == "__main__":
    main()
