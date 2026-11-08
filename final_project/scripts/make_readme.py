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

![What each quantization configuration does to each metric](figures/impact_matrix.png)

The figure reads from left to right, and each finding limits the next one.

1. **Memory: quantization saves bytes, but fewer than the bit-width suggests.** Only the decoder's
   linear layers shrink; the embedding, LM head and scales stay 16-bit. So on Qwen2.5-7B, 4-bit weights
   save **{m['WtNFSeven']}** (NF4) to **{m['WtAWQSeven']}** (AWQ), not 4×, and 8-bit weights **{m['WtIntEightSeven']}**, not 2×. At run time
   the KV cache and activations are not quantized, so with 32 × 2048-token prompts the total saving falls
   to **{m['MemBigNF']}** (NF4) and **{m['MemBigInt']}** (LLM.int8()). The saved memory pays off as capacity: vLLM holds
   **{m['KVratioAWQ']}** more KV-cache tokens with AWQ-INT4.
2. **Small batches: the saved bytes become speed only with a fused kernel.** At batch size 1, decoding is
   memory-bound, so fewer bytes should mean less time. On vLLM they do: AWQ-INT4 decodes **{m['AWQspeedup']}**
   faster, GPTQ-INT8 {m['GPTQeightspeedup']}, FP8 {m['FPeightspeedup']} ({m['IdealAWQ']}–{m['IdealGQE']}% of their byte savings). With bitsandbytes they
   do not: NF4 is only {m['NFsevenB']} faster, and LLM.int8() is **{m['INTeightslowdown']} slower**.
3. **Large batches: the gain lasts only while decoding is memory-bound.** Batching makes decoding
   compute-bound, so weight-only gains fade and vanish near **B ≈ {m['AWQcross']}** (AWQ-INT4, 512-token prompts).
   Only **FP8 W8A8**, which also halves the compute, stays faster in every cell ({m['FPminEtoE']}–{m['FPmaxEtoE']}) and
   gives **{m['ThrFPratio']}** the BF16 throughput at B = 256.
4. **Why: the roofline and the host.** Marlin W4A16 is {m['MarlinFourMone']} faster than cuBLAS at M = 1 and loses its
   gain at M ≈ {m['MarlinFourCross']}, where it hits the same BF16 compute roof. On HF, kernel launches on the CPU
   dominate: the GPU is busy only {m['BusyIntone']}% of an LLM.int8() decode step (BF16: {m['BusyBFone']}%).
5. **Quality: small cost at 8 bits, moderate at 4 bits.** GPTQ-INT8 and FP8 stay within {m['EightKernMax']} points of BF16
   on GSM8K and MMLU. 4-bit formats cost {m['MFourAbsMin']}–{m['MFourAbsMax']} MMLU points, most on the 3B model, where GSM8K
   drops as well ({m['DFourMinThree']} to {m['DFourMaxThree']} points).
6. **So spend the saved memory on parameters.** Qwen2.5-14B in AWQ-INT4 ({m['QualityGuide'].split('needs ')[1].split(' of')[0]}) beats Qwen2.5-7B
   in BF16, and Qwen3-32B, which fits this GPU only in 4 bits ({m.get('BigAWQw', '?')} GB), scores
   **{m.get('BigGSMawq', '?')}% / {m.get('BigMMLUawq', '?')}%** on GSM8K / MMLU against {m.get('SmallGSMbf', '?')}% / {m.get('SmallMMLUbf', '?')}% for Qwen3-8B in BF16 ({m.get('SmallBFw', '?')} GB).
7. **The systems effects generalize; the quality order does not.** On Qwen3-8B, AWQ-INT4 decodes
   {m.get('FamRAWQone', '?')} faster (Qwen2.5-7B: {m.get('FamQAWQone', '?')}), and NF4 falls to {m.get('FamRNFone', '?')} on the 36-layer model (host-bound).
   But there FP8 costs as much MMLU as AWQ-INT4, so quality must be checked per model, and with more than
   one metric: the 14B GPTQ-INT4 checkpoint keeps its task scores but has a perplexity of {m.get('PPLGPTQFourFourteen', '?')}.
"""
    results = f"""## Results at a glance

**Overview — effect of each quantized configuration on each metric, Qwen2.5-7B (vs. BF16 on the same stack)**

{md_table('quantization_effects')}

**Q1 — weight memory (GB)**

{md_table('t_weight_memory_all')}

![weights](figures/q1_weight_memory.png)
![memory](figures/q1_memory_main.png)

**Q2 — latency and throughput, Qwen2.5-7B, prompt 512, 128 output tokens**

{md_table('t_speed')}

![speed](figures/speed_effects.png)\n![latency](figures/q2_latency_throughput.png)

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

**Does it generalize? Qwen2.5-7B vs Qwen3-8B**

{md_table('t_generalization')}

**Output length — end-to-end speedup vs generated tokens**

![output length](figures/q3_output_length.png)
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
