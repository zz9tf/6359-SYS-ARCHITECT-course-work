# Smaller Is Not Always Faster
### The memory–performance trade-off of LLM quantization on an RTX 5090

CSE 6359 Advanced Topics in Systems & Architecture — Final Project, Fall 2026
**Sijia Zhong · Zheng Zheng · Haofei Sun**

📄 **Report:** [`report/main.pdf`](report/main.pdf) (NeurIPS 2026 format: 9-page main text + appendix) · 🎞️ **Slides:** [`slides/final_presentation.pptx`](slides/final_presentation.pptx) · 📝 Proposal: [`proposal/`](proposal/)

<!-- KEY-FINDINGS -->
## Key findings

![What each quantization configuration does to each metric](figures/impact_matrix.png)

The figure reads from left to right, and each finding limits the next one.

1. **Memory: quantization saves bytes, but fewer than the bit-width suggests.** Only the decoder's
   linear layers shrink; the embedding, LM head and scales stay 16-bit. So on Qwen2.5-7B, 4-bit weights
   save **2.60×** (NF4) to **2.70×** (AWQ), not 4×, and 8-bit weights **1.75×**, not 2×. At run time
   the KV cache and activations are not quantized, so with 32 × 2048-token prompts the total saving falls
   to **1.49×** (NF4) and **1.04×** (LLM.int8()). The saved memory pays off as capacity: vLLM holds
   **1.90×** more KV-cache tokens with AWQ-INT4.
2. **Small batches: the saved bytes become speed only with a fused kernel.** At batch size 1, decoding is
   memory-bound, so fewer bytes should mean less time. On vLLM they do: AWQ-INT4 decodes **2.44×**
   faster, GPTQ-INT8 1.64×, FP8 1.59× (90–95% of their byte savings). With bitsandbytes they
   do not: NF4 is only 1.13× faster, and LLM.int8() is **3.4× slower**.
3. **Large batches: the gain lasts only while decoding is memory-bound.** Batching makes decoding
   compute-bound, so weight-only gains fade and vanish near **B ≈ 175** (AWQ-INT4, 512-token prompts).
   Only **FP8 W8A8**, which also halves the compute, stays faster in every cell (1.59×–1.80×) and
   gives **1.73×** the BF16 throughput at B = 256.
4. **Why: the roofline and the host.** Marlin W4A16 is 3.2× faster than cuBLAS at M = 1 and loses its
   gain at M ≈ 333, where it hits the same BF16 compute roof. On HF, kernel launches on the CPU
   dominate: the GPU is busy only 28% of an LLM.int8() decode step (BF16: 97%).
5. **Quality: small cost at 8 bits, moderate at 4 bits.** GPTQ-INT8 and FP8 stay within 0.9 points of BF16
   on GSM8K and MMLU. 4-bit formats cost 0.6–2.8 MMLU points, most on the 3B model, where GSM8K
   drops as well ($-3.8$ to $-3.1$ points).
6. **So spend the saved memory on parameters.** Qwen2.5-14B in AWQ-INT4 (10.1 GB) beats Qwen2.5-7B
   in BF16, and Qwen3-32B, which fits this GPU only in 4 bits (19.6 GB), scores
   **94.2% / 78.1%** on GSM8K / MMLU against 91.4% / 69.6% for Qwen3-8B in BF16 (16.4 GB).
7. **The systems effects generalize; the quality order does not.** On Qwen3-8B, AWQ-INT4 decodes
   2.33× faster (Qwen2.5-7B: 2.44×), and NF4 falls to 0.90× on the 36-layer model (host-bound).
   But there FP8 costs as much MMLU as AWQ-INT4, so quality must be checked per model, and with more than
   one metric: the 14B GPTQ-INT4 checkpoint keeps its task scores but has a perplexity of 37.5.
<!-- /KEY-FINDINGS -->

## What we measured

| | |
|---|---|
| **GPU** | NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32 GB GDDR7, 170 SMs), driver 595.91 |
| **Software** | PyTorch 2.13 (CUDA 13.0), Transformers 5.19, bitsandbytes 0.50.2, vLLM 0.32 — pinned in [`requirements.txt`](requirements.txt) |
| **Models** | Qwen2.5-3B / 7B / 14B-Instruct; Qwen3-8B and Qwen3-32B (non-thinking mode) as a second model family — Qwen3-32B fits the 32 GB GPU only at 4 bits |
| **Configurations** | HF + bitsandbytes: BF16, INT8 (LLM.int8()), NF4 · vLLM: BF16, FP8 W8A8, GPTQ-INT8 (W8A16), AWQ-INT4 and GPTQ-INT4 (W4A16, Marlin kernels) |
| **Workloads** | controlled sweep (batch 1–256 × prompt 128–2048 tokens, 128 generated tokens) · output-length sweep (32–512 generated tokens) · 300 ShareGPT requests replayed at 1–64 concurrent clients · WikiText-2 perplexity · GSM8K (all 1,319 questions) · MMLU (all 14,042 questions) |
| **Metrics** | weight + peak memory, KV-cache capacity, TTFT, TPOT, throughput, GPU energy per token, perplexity, accuracy; kernel microbenchmarks, measured roofline, per-step profiles |

<!-- RESULTS -->
## Results at a glance

**Overview — effect of each quantized configuration on each metric, Qwen2.5-7B (vs. BF16 on the same stack)**

| type   | method     | backend   |   weight saving (x) |   KV-cache capacity (x, vLLM) |   peak-memory saving, 32x2048 tokens (x, HF) |   prefill speedup, B=1 L=2048 (x) |   decode speedup, B=1 (x) |   throughput ratio, B=64 (x) |   energy saving per token, B=1 (x) |   perplexity change |   GSM8K change (points) |   GSM8K McNemar p |   MMLU change (points) |   MMLU McNemar p |
|:-------|:-----------|:----------|--------------------:|------------------------------:|---------------------------------------------:|----------------------------------:|--------------------------:|-----------------------------:|-----------------------------------:|--------------------:|------------------------:|------------------:|-----------------------:|-----------------:|
| W8A8   | FP8        | vllm      |               1.749 |                         1.623 |                                      nan     |                             1.773 |                     1.59  |                        1.759 |                              1.946 |               0.054 |                   0     |             1     |                 -0.142 |            0.327 |
| W8A8   | LLM.int8() | hf        |               1.749 |                       nan     |                                        1.041 |                             1.051 |                     0.291 |                        0.601 |                              0.823 |               0.027 |                  -0.227 |             0.771 |                 -0.207 |            0.134 |
| W8A16  | GPTQ-INT8  | vllm      |               1.724 |                         1.605 |                                      nan     |                             0.933 |                     1.641 |                        1.119 |                              1.885 |               0.002 |                   0.91  |             0.043 |                  0.043 |            0.61  |
| W4A16  | AWQ-INT4   | vllm      |               2.701 |                         1.902 |                                      nan     |                             0.939 |                     2.438 |                        1.152 |                              2.541 |               0.39  |                  -0.455 |             0.576 |                 -1.588 |            0     |
| W4A16  | GPTQ-INT4  | vllm      |               2.712 |                         1.857 |                                      nan     |                             0.988 |                     2.453 |                        1.196 |                              2.651 |               0.318 |                   0.227 |             0.828 |                 -1.41  |            0     |
| W4A16  | NF4        | hf        |               2.602 |                       nan     |                                        1.493 |                             0.926 |                     1.134 |                        0.929 |                              1.597 |               0.312 |                  -0.531 |             0.5   |                 -0.62  |            0.002 |


**Q1 — weight memory (GB)**

| Model       | Config               |   Naive (GB) |   Model (GB) |   Measured (GB) | Ratio vs BF16   | Naive ratio   |
|:------------|:---------------------|-------------:|-------------:|----------------:|:----------------|:--------------|
| Qwen2.5-3B  | BF16 (HF)            |         6.17 |         6.17 |            6.17 | 1.00×    | 1.00×  |
| Qwen2.5-3B  | BF16 (vLLM)          |         6.17 |         6.17 |            6.22 | 1.00×    | 1.00×  |
| Qwen2.5-3B  | W8A8 FP8 (vLLM)      |         3.09 |         3.4  |            3.47 | 1.79×    | 2.00×  |
| Qwen2.5-3B  | W8A8 LLM.int8() (HF) |         3.09 |         3.4  |            3.4  | 1.81×    | 2.00×  |
| Qwen2.5-3B  | W8A16 GPTQ (vLLM)    |         3.09 |         3.46 |            3.51 | 1.77×    | 2.00×  |
| Qwen2.5-3B  | W4A16 AWQ (vLLM)     |         1.54 |         2.06 |            2.09 | 2.97×    | 4.00×  |
| Qwen2.5-3B  | W4A16 GPTQ (vLLM)    |         1.54 |         2.06 |            2.08 | 2.98×    | 4.00×  |
| Qwen2.5-3B  | W4A16 NF4 (HF)       |         1.54 |         2.18 |            2.2  | 2.81×    | 4.00×  |
| Qwen2.5-7B  | BF16 (HF)            |        15.23 |        15.23 |           15.23 | 1.00×    | 1.00×  |
| Qwen2.5-7B  | BF16 (vLLM)          |        15.23 |        15.23 |           15.34 | 1.00×    | 1.00×  |
| Qwen2.5-7B  | W8A8 FP8 (vLLM)      |         7.62 |         8.71 |            8.77 | 1.75×    | 2.00×  |
| Qwen2.5-7B  | W8A8 LLM.int8() (HF) |         7.62 |         8.71 |            8.71 | 1.75×    | 2.00×  |
| Qwen2.5-7B  | W8A16 GPTQ (vLLM)    |         7.62 |         8.86 |            8.9  | 1.72×    | 2.00×  |
| Qwen2.5-7B  | W4A16 AWQ (vLLM)     |         3.81 |         5.57 |            5.68 | 2.70×    | 4.00×  |
| Qwen2.5-7B  | W4A16 GPTQ (vLLM)    |         3.81 |         5.57 |            5.66 | 2.71×    | 4.00×  |
| Qwen2.5-7B  | W4A16 NF4 (HF)       |         3.81 |         5.85 |            5.85 | 2.60×    | 4.00×  |
| Qwen2.5-14B | BF16 (HF)            |        29.54 |        29.54 |           29.54 | 1.00×    | 1.00×  |
| Qwen2.5-14B | BF16 (vLLM)          |        29.54 |        29.54 |           29.6  | 1.00×    | 1.00×  |
| Qwen2.5-14B | W8A8 FP8 (vLLM)      |        14.77 |        16.33 |           16.51 | 1.79×    | 2.00×  |
| Qwen2.5-14B | W8A8 LLM.int8() (HF) |        14.77 |        16.33 |           16.34 | 1.81×    | 2.00×  |
| Qwen2.5-14B | W8A16 GPTQ (vLLM)    |        14.77 |        16.64 |           16.73 | 1.77×    | 2.00×  |
| Qwen2.5-14B | W4A16 AWQ (vLLM)     |         7.39 |         9.98 |           10.07 | 2.94×    | 4.00×  |
| Qwen2.5-14B | W4A16 GPTQ (vLLM)    |         7.39 |         9.98 |           10.02 | 2.95×    | 4.00×  |
| Qwen2.5-14B | W4A16 NF4 (HF)       |         7.39 |        10.55 |           10.58 | 2.79×    | 4.00×  |


![weights](figures/q1_weight_memory.png)
![memory](figures/q1_memory_main.png)

**Q2 — latency and throughput, Qwen2.5-7B, prompt 512, 128 output tokens**

| Config               |   TTFT B=1 (ms) |   TPOT B=1 (ms) |   TPOT B=64 (ms) |   tok/s B=64 | Decode speedup B=1   |   mJ/token B=1 |   mJ/token B=64 |
|:---------------------|----------------:|----------------:|-----------------:|-------------:|:---------------------|---------------:|----------------:|
| BF16 (HF)            |            38.4 |           10.9  |            20.04 |         1669 | 1.00×         |           5092 |             294 |
| BF16 (vLLM)          |            37   |            9.52 |            12.39 |         2209 | 1.00×         |           4878 |             249 |
| W8A8 FP8 (vLLM)      |            22.8 |            5.99 |             7.88 |         3886 | 1.59×         |           2507 |             141 |
| W8A8 LLM.int8() (HF) |            57.8 |           37.45 |            47.14 |         1003 | 0.29×         |           6187 |             305 |
| W8A16 GPTQ (vLLM)    |            38.5 |            5.8  |             8.41 |         2471 | 1.64×         |           2588 |             234 |
| W4A16 AWQ (vLLM)     |            38.6 |            3.9  |             7.99 |         2544 | 2.44×         |           1919 |             224 |
| W4A16 GPTQ (vLLM)    |            36.9 |            3.88 |             7.89 |         2642 | 2.45×         |           1840 |             219 |
| W4A16 NF4 (HF)       |            48.3 |            9.62 |            22.87 |         1550 | 1.13×         |           3188 |             340 |


![speed](figures/speed_effects.png)
![latency](figures/q2_latency_throughput.png)

**Q3 — where each format stops winning (speedup over BF16 on the same backend)**

| Config               | Decode B=1   | Decode max B     | Decode wins up to   | E2E B=1   | E2E max B        | E2E wins up to   |
|:---------------------|:-----------------|:-------------------|:--------------------|:--------------|:-------------------|:-----------------|
| W8A8 FP8 (vLLM)      | 1.59×     | 1.36× (256) | all B             | 1.59×  | 1.73× (256) | all B          |
| W8A8 LLM.int8() (HF) | 0.29×     | 0.43× (64)  | never               | 0.30×  | 0.60× (64)  | never            |
| W8A16 GPTQ (vLLM)    | 1.64×     | 1.00× (256) | B≈251       | 1.61×  | 0.97× (256) | B≈123    |
| W4A16 AWQ (vLLM)     | 2.44×     | 1.02× (256) | all B             | 2.33×  | 0.98× (256) | B≈175    |
| W4A16 GPTQ (vLLM)    | 2.45×     | 1.05× (256) | all B             | 2.35×  | 1.03× (256) | all B          |
| W4A16 NF4 (HF)       | 1.13×     | 0.88× (64)  | B≈7         | 1.12×  | 0.93× (64)  | B≈7      |


![heatmaps](figures/q3_speedup_heatmaps.png)
![kernels](figures/q3_kernel_crossover.png)
![profile](figures/mech_decode_breakdown.png)

**Real workload — 300 ShareGPT requests**

| Config                   |   C |   req/s |   tok/s | TTFT p50 / p99   | TPOT p50 / p99   |
|:-------------------------|------:|--------:|--------:|:-----------------|:-----------------|
| BF16 (HF) (static)       |    16 |    0.84 |     254 | 797 / 1244       | 23.5 / 29.7      |
| BF16 (vLLM)              |     1 |    0.35 |     104 | 16 / 62          | 9.6 / 9.6        |
| BF16 (vLLM)              |    64 |   11.76 |    3537 | 46 / 1654        | 13.6 / 16.7      |
| FP8 (vLLM)               |     1 |    0.54 |     164 | 11 / 44          | 6.1 / 6.2        |
| FP8 (vLLM)               |    64 |   17.92 |    5388 | 33 / 1485        | 8.9 / 15.1       |
| LLM.int8() (HF) (static) |    16 |    0.39 |     119 | 691 / 1158       | 54.9 / 64.1      |
| GPTQ-8 (vLLM)            |     1 |    0.55 |     166 | 11 / 63          | 6.0 / 6.2        |
| GPTQ-8 (vLLM)            |    64 |   15.74 |    4733 | 41 / 1743        | 10.5 / 18.0      |
| AWQ-4 (vLLM)             |     1 |    0.8  |     240 | 11 / 65          | 4.1 / 4.5        |
| AWQ-4 (vLLM)             |    64 |   17.23 |    5180 | 40 / 1747        | 10.1 / 14.8      |
| GPTQ-4 (vLLM)            |     1 |    0.8  |     241 | 11 / 62          | 4.1 / 4.4        |
| GPTQ-4 (vLLM)            |    64 |   17.23 |    5181 | 40 / 1722        | 10.0 / 19.3      |
| NF4 (HF) (static)        |    16 |    0.72 |     215 | 842 / 1271       | 27.8 / 35.4      |


![serving](figures/real_sharegpt_serving.png)

**Q4 — quality**

| Model       | Config          | PPL                | \DeltaPPL   |   GSM8K | \Delta   | p      |   MMLU | \Delta\,   | p\,    |
|:------------|:----------------|:-------------------|:--------------|--------:|:-----------|:---------|-------:|:-------------|:---------|
| Qwen2.5-3B  | BF16 (HF)       | 8.56               | --            |    85.8 | --         | --       |   63.5 | --           | --       |
|             | BF16 (vLLM)     | 8.56               | 0.00        |    86   | +0.2     | 0.871    |   63.7 | +0.1       | 0.075    |
|             | FP8 (vLLM)      | 8.64               | +0.09       |    85.4 | -0.5     | 0.610    |   63.2 | -0.4       | 0.038    |
|             | LLM.int8() (HF) | 8.66               | +0.10       |    85.5 | -0.3     | 0.777    |   62.4 | -1.1       | <0.001 |
|             | GPTQ-8 (vLLM)   | 8.56               | 0.00        |    86.4 | +0.6     | 0.322    |   63.6 | +0.1       | 0.261    |
|             | AWQ-4 (vLLM)    | 9.10               | +0.54       |    82.7 | -3.1     | 0.002    |   61.4 | -2.2       | <0.001 |
|             | GPTQ-4 (vLLM)   | 9.22               | +0.66       |    82   | -3.8     | <0.001 |   61.1 | -2.5       | <0.001 |
|             | NF4 (HF)        | 9.11               | +0.55       |    82.6 | -3.3     | <0.001 |   60.8 | -2.8       | <0.001 |
| Qwen2.5-7B  | BF16 (HF)       | 7.46               | --            |    90.8 | --         | --       |   71.4 | --           | --       |
|             | BF16 (vLLM)     | 7.46               | 0.00        |    91.2 | +0.4     | 0.359    |   71.4 | 0.0        | 0.704    |
|             | FP8 (vLLM)      | 7.51               | +0.05       |    90.8 | 0.0      | 1.000    |   71.3 | -0.1       | 0.327    |
|             | LLM.int8() (HF) | 7.49               | +0.03       |    90.6 | -0.2     | 0.771    |   71.2 | -0.2       | 0.134    |
|             | GPTQ-8 (vLLM)   | 7.46               | 0.00        |    91.7 | +0.9     | 0.043    |   71.5 | 0.0        | 0.610    |
|             | AWQ-4 (vLLM)    | 7.85               | +0.39       |    90.4 | -0.5     | 0.576    |   69.9 | -1.6       | <0.001 |
|             | GPTQ-4 (vLLM)   | 7.78               | +0.32       |    91.1 | +0.2     | 0.828    |   70   | -1.4       | <0.001 |
|             | NF4 (HF)        | 7.77               | +0.31       |    90.3 | -0.5     | 0.500    |   70.8 | -0.6       | 0.002    |
| Qwen2.5-14B | BF16 (HF)       | 5.70               | --            |    93.9 | --         | --       |   77   | --           | --       |
|             | FP8 (vLLM)      | 5.77               | +0.07       |    94.4 | +0.5     | 0.371    |   76.9 | 0.0        | 0.862    |
|             | LLM.int8() (HF) | 5.78               | +0.08       |    94.5 | +0.6     | 0.280    |   76.6 | -0.3       | 0.025    |
|             | GPTQ-8 (vLLM)   | 5.71               | +0.01       |    94.7 | +0.8     | 0.013    |   76.9 | -0.1       | 0.416    |
|             | AWQ-4 (vLLM)    | 6.16               | +0.46       |    93.9 | +0.1     | 1.000    |   75.1 | -1.9       | <0.001 |
|             | GPTQ-4 (vLLM)   | 37.46^{\ddagger} | +31.77      |    93.7 | -0.2     | 0.899    |   76.4 | -0.6       | 0.005    |
|             | NF4 (HF)        | 6.07               | +0.37       |    93.2 | -0.7     | 0.281    |   76.4 | -0.6       | 0.003    |
| Qwen3-8B    | BF16 (HF)       | 9.73               | --            |    91.4 | --         | --       |   69.6 | --           | --       |
|             | BF16 (vLLM)     | 9.72               | 0.00        |    92   | +0.6     | 0.215    |   69.6 | 0.0        | 0.610    |
|             | FP8 (vLLM)      | 9.73               | 0.00        |    91.8 | +0.5     | 0.488    |   68.9 | -0.7       | <0.001 |
|             | LLM.int8() (HF) | 9.72               | 0.00        |    92   | +0.6     | 0.382    |   70.3 | +0.7       | <0.001 |
|             | AWQ-4 (vLLM)    | 10.18              | +0.46       |    91.1 | -0.2     | 0.818    |   68.9 | -0.7       | 0.003    |
|             | NF4 (HF)        | 9.98               | +0.25       |    91.5 | +0.2     | 0.909    |   69.3 | -0.3       | 0.156    |
| Qwen3-32B   | AWQ-4 (vLLM)    | 7.88^{\S}        | --            |    94.2 | --         | --       |   78.1 | --           | --       |
|             | NF4 (HF)        | 7.93               | --            |    92.6 | --         | --       |   79.4 | --           | --       |


![quality](figures/q4_quality_vs_memory.png)

**Does it generalize? Qwen2.5-7B vs Qwen3-8B**

| Metric (vs.\ BF16)                  | Qwen2.5-7B        | Qwen3-8B          |
|:------------------------------------|:------------------|:------------------|
| Weight saving, AWQ-INT4 (vLLM)      | 2.70×      | 2.67×      |
| Weight saving, NF4 (HF)             | 2.60×      | 2.55×      |
| Decode speedup B=1, AWQ-INT4      | 2.44×      | 2.33×      |
| Decode speedup B=1, FP8           | 1.59×      | 1.54×      |
| Decode speedup B=1, NF4           | 1.13×      | 0.90×      |
| Decode speedup B=1, LLM.int8()    | 0.29×      | 0.28×      |
| Throughput B=64, AWQ-INT4         | 1.15×      | 1.12×      |
| Throughput B=64, FP8              | 1.76×      | 1.67×      |
| GSM8K \Delta pp (p), FP8        | 0.0 (1.000)     | +0.5 (0.488)    |
| GSM8K \Delta pp (p), LLM.int8() | -0.2 (0.771)    | +0.6 (0.382)    |
| GSM8K \Delta pp (p), AWQ-INT4   | -0.5 (0.576)    | -0.2 (0.818)    |
| GSM8K \Delta pp (p), NF4        | -0.5 (0.500)    | +0.2 (0.909)    |
| MMLU \Delta pp (p), FP8         | -0.1 (0.327)    | -0.7 (<0.001) |
| MMLU \Delta pp (p), LLM.int8()  | -0.2 (0.134)    | +0.7 (<0.001) |
| MMLU \Delta pp (p), AWQ-INT4    | -1.6 (<0.001) | -0.7 (0.003)    |
| MMLU \Delta pp (p), NF4         | -0.6 (0.002)    | -0.3 (0.156)    |


**Output length — end-to-end speedup vs generated tokens**

![output length](figures/q3_output_length.png)
<!-- /RESULTS -->

## Repository layout

```
final_project/
├── quantbench/            measurement library (loading, timing, NVML energy/memory monitor,
│                          analytic memory model, workloads, aggregation, plot style)
├── scripts/               one script per experiment + figure/table generation
│   ├── hf_sweep.py        precision × batch × prompt-length sweep, Hugging Face + bitsandbytes
│   ├── vllm_sweep.py      the same sweep on vLLM
│   ├── kernel_bench.py    linear-layer kernels vs. batch size (cuBLAS, bnb, CUTLASS FP8, Marlin)
│   ├── hw_peaks.py        measured DRAM bandwidth and BF16/FP8/INT8 GEMM ceilings
│   ├── profile_decode.py  torch.profiler breakdown of one decode step
│   ├── serve_bench.sh     ShareGPT replay against a vLLM server (`vllm bench serve`)
│   ├── hf_sharegpt.py     the same requests with HF static batching
│   ├── eval_ppl.py        WikiText-2 perplexity (HF or vLLM)
│   ├── eval_gsm8k.py      GSM8K accuracy (HF or vLLM)
│   ├── eval_mmlu.py       MMLU accuracy (HF or vLLM)
│   ├── diag_ppl_topk.py   next-token ranking on raw vs chat text (perplexity-outlier diagnosis)
│   └── make_figures.py · make_tables.py · make_numbers.py · make_readme.py
├── results/               raw measurements (*.jsonl), serving results, summaries (CSV/Markdown), env info
├── figures/               all figures (PDF for the report, PNG for the slides and this README)
├── report/                LaTeX source (NeurIPS 2026 style) and PDF of the final report
├── slides/                final presentation (UTA template) and its build script
└── proposal/              proposal slides and write-up
```

## Reproduce

```bash
cd final_project
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
bash run_all.sh                      # every experiment, about twelve GPU-hours on an RTX 5090
python scripts/make_figures.py       # figures/*.pdf|png
python scripts/make_tables.py        # results/summary/* and report/tables/*
python scripts/make_numbers.py       # report/numbers.tex (every number quoted in the report)
python scripts/make_readme.py        # key findings and result tables in this README
(cd report && tectonic main.tex)     # report/main.pdf
python slides/build_slides.py        # slides/final_presentation.pptx
```

Notes: vLLM's FlashInfer sampler needs `nvcc` for JIT compilation, which our machine does not have, so
all scripts set `VLLM_USE_FLASHINFER_SAMPLER=0` (we decode greedily, so the sampler is never on the
critical path). Raw JSONL rows carry the full environment (`env`) of the run that produced them.
