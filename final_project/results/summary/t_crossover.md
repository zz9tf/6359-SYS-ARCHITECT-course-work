| Config             | Decode $B{=}1$   | Decode max $B$     | Decode wins up to   | E2E $B{=}1$   | E2E max $B$        | E2E wins up to   |
|:-------------------|:-----------------|:-------------------|:--------------------|:--------------|:-------------------|:-----------------|
| HF INT8 (LLM.int8) | 0.29$\times$     | 0.43$\times$ (64)  | never               | 0.30$\times$  | 0.60$\times$ (64)  | never            |
| HF NF4             | 1.13$\times$     | 0.88$\times$ (64)  | $B\approx7$         | 1.12$\times$  | 0.93$\times$ (64)  | $B\approx7$      |
| vLLM FP8 W8A8      | 1.59$\times$     | 1.36$\times$ (256) | all $B$             | 1.59$\times$  | 1.73$\times$ (256) | all $B$          |
| vLLM GPTQ W8A16    | 1.64$\times$     | 1.00$\times$ (256) | $B\approx251$       | 1.61$\times$  | 0.97$\times$ (256) | $B\approx123$    |
| vLLM AWQ W4A16     | 2.44$\times$     | 1.02$\times$ (256) | all $B$             | 2.33$\times$  | 0.98$\times$ (256) | $B\approx175$    |
| vLLM GPTQ W4A16    | 2.45$\times$     | 1.05$\times$ (256) | all $B$             | 2.35$\times$  | 1.03$\times$ (256) | all $B$          |
