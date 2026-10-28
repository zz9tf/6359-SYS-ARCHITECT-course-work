| Model       | Config             |   Naive (GB) |   Model (GB) |   Measured (GB) | Ratio vs BF16   | Naive ratio   |
|:------------|:-------------------|-------------:|-------------:|----------------:|:----------------|:--------------|
| Qwen2.5-3B  | HF BF16            |         6.17 |         6.17 |            6.17 | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-3B  | HF INT8 (LLM.int8) |         3.09 |         3.4  |            3.4  | 1.81$\times$    | 2.00$\times$  |
| Qwen2.5-3B  | HF NF4             |         1.54 |         2.18 |            2.2  | 2.81$\times$    | 4.00$\times$  |
| Qwen2.5-3B  | vLLM BF16          |         6.17 |         6.17 |            6.22 | 1.00$\times$    | 1.00$\times$  |
| Qwen2.5-3B  | vLLM FP8 W8A8      |         3.09 |         3.4  |            3.47 | 1.79$\times$    | 2.00$\times$  |
| Qwen2.5-3B  | vLLM GPTQ W8A16    |         3.09 |         3.46 |            3.51 | 1.77$\times$    | 2.00$\times$  |
| Qwen2.5-3B  | vLLM AWQ W4A16     |         1.54 |         2.06 |            2.09 | 2.97$\times$    | 4.00$\times$  |
| Qwen2.5-3B  | vLLM GPTQ W4A16    |         1.54 |         2.06 |            2.08 | 2.98$\times$    | 4.00$\times$  |
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
