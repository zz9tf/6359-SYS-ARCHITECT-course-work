| Model       | Config               |   Naive (GB) |   Model (GB) |   Measured (GB) | Ratio vs BF16   | Naive ratio   |
|:------------|:---------------------|-------------:|-------------:|----------------:|:----------------|:--------------|
| Qwen2.5-7B  | BF16 (HF)            |        15.23 |        15.23 |           15.23 | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-7B  | BF16 (vLLM)          |        15.23 |        15.23 |           15.34 | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-7B  | W8A8 FP8 (vLLM)      |         7.62 |         8.71 |            8.77 | 1.75$\times$    | 2.00$\times$  |
| Qwen2.5-7B  | W8A8 LLM.int8() (HF) |         7.62 |         8.71 |            8.71 | 1.75$\times$    | 2.00$\times$  |
| Qwen2.5-7B  | W8A16 GPTQ (vLLM)    |         7.62 |         8.86 |            8.9  | 1.72$\times$    | 2.00$\times$  |
| Qwen2.5-7B  | W4A16 AWQ (vLLM)     |         3.81 |         5.57 |            5.68 | 2.70$\times$    | 4.00$\times$  |
| Qwen2.5-7B  | W4A16 GPTQ (vLLM)    |         3.81 |         5.57 |            5.66 | 2.71$\times$    | 4.00$\times$  |
| Qwen2.5-7B  | W4A16 NF4 (HF)       |         3.81 |         5.85 |            5.85 | 2.60$\times$    | 4.00$\times$  |
| Qwen2.5-14B | BF16 (HF)            |        29.54 |        29.54 |           29.54 | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-14B | BF16 (vLLM)          |        29.54 |        29.54 |           29.6  | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-14B | W8A8 FP8 (vLLM)      |        14.77 |        16.33 |           16.51 | 1.79$\times$    | 2.00$\times$  |
| Qwen2.5-14B | W8A8 LLM.int8() (HF) |        14.77 |        16.33 |           16.34 | 1.81$\times$    | 2.00$\times$  |
| Qwen2.5-14B | W8A16 GPTQ (vLLM)    |        14.77 |        16.64 |           16.73 | 1.77$\times$    | 2.00$\times$  |
| Qwen2.5-14B | W4A16 AWQ (vLLM)     |         7.39 |         9.98 |           10.07 | 2.94$\times$    | 4.00$\times$  |
| Qwen2.5-14B | W4A16 GPTQ (vLLM)    |         7.39 |         9.98 |           10.02 | 2.95$\times$    | 4.00$\times$  |
| Qwen2.5-14B | W4A16 NF4 (HF)       |         7.39 |        10.55 |           10.58 | 2.79$\times$    | 4.00$\times$  |
