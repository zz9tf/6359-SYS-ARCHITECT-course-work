| Model       | Config             |   Naive (GB) |   Model (GB) |   Measured (GB) | Ratio vs BF16   | Naive ratio   |
|:------------|:-------------------|-------------:|-------------:|----------------:|:----------------|:--------------|
| Qwen2.5-7B  | HF BF16            |        15.23 |        15.23 |           15.23 | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-7B  | HF INT8 (LLM.int8) |         7.62 |         8.71 |            8.71 | 1.75$\times$    | 2.00$\times$  |
| Qwen2.5-7B  | HF NF4             |         3.81 |         5.85 |            5.85 | 2.60$\times$    | 4.00$\times$  |
| Qwen2.5-7B  | vLLM BF16          |        15.23 |        15.23 |           15.34 | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-7B  | vLLM FP8 W8A8      |         7.62 |         8.71 |            8.77 | 1.75$\times$    | 2.00$\times$  |
| Qwen2.5-7B  | vLLM GPTQ W8A16    |         7.62 |         8.86 |            8.9  | 1.72$\times$    | 2.00$\times$  |
| Qwen2.5-7B  | vLLM AWQ W4A16     |         3.81 |         5.57 |            5.68 | 2.70$\times$    | 4.00$\times$  |
| Qwen2.5-7B  | vLLM GPTQ W4A16    |         3.81 |         5.57 |            5.66 | 2.71$\times$    | 4.00$\times$  |
| Qwen2.5-14B | HF BF16            |        29.54 |        29.54 |           29.54 | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-14B | HF INT8 (LLM.int8) |        14.77 |        16.33 |           16.34 | 1.81$\times$    | 2.00$\times$  |
| Qwen2.5-14B | HF NF4             |         7.39 |        10.55 |           10.58 | 2.79$\times$    | 4.00$\times$  |
| Qwen2.5-14B | vLLM BF16          |        29.54 |        29.54 |           29.6  | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-14B | vLLM FP8 W8A8      |        14.77 |        16.33 |           16.51 | 1.79$\times$    | 2.00$\times$  |
| Qwen2.5-14B | vLLM GPTQ W8A16    |        14.77 |        16.64 |           16.73 | 1.77$\times$    | 2.00$\times$  |
| Qwen2.5-14B | vLLM AWQ W4A16     |         7.39 |         9.98 |           10.07 | 2.94$\times$    | 4.00$\times$  |
| Qwen2.5-14B | vLLM GPTQ W4A16    |         7.39 |         9.98 |           10.02 | 2.95$\times$    | 4.00$\times$  |
