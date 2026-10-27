# Course Project Proposal

## Smaller Is Not Always Faster: LLM Quantization  Memory–Performance Trade-off

### 1. 背景与动机
大语言模型的参数量越来越大，推理时模型权重会占用大量 GPU 显存。Quantization 通过把 BF16/FP16 权重压缩为 INT8、INT4 等低精度表示，可以显著降低显存需求和模型大小。

但“模型更小”并不一定意味着“推理一定更快”。低比特量化还会引入反量化、kernel 支持、memory/compute bottleneck 等额外因素，因此实际 latency 和 throughput 可能依赖模型大小、batch size 和量化方法。

本项目希望回答一个简单但有实际意义的问题：**量化能省多少显存，以及它什么时候真的能带来推理加速？**

---

### 2. 研究问题
- **RQ1：** BF16/FP16、INT8、INT4 分别能节省多少 GPU memory？
- **RQ2：** 不同精度下，TTFT、decode latency 和 throughput 如何变化？
- **RQ3：** 更低 bit 是否总是更快？不同 batch size / prompt length 下是否存在不同结论？
- **RQ4（可选）：** 量化对模型质量（perplexity / 简单任务准确率）影响多大？

---

### 3. 实施方案

#### 模型
主模型建议：
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)（资源允许时作为第二个模型）

比较配置：
1. BF16 / FP16 baseline
2. bitsandbytes INT8
3. bitsandbytes INT4（NF4 或 4-bit）
4. **可选 extension：** AWQ 4-bit

#### 数据
- Serving workload：从 [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) 中随机采样约 100–300 个 prompts，测试推理速度：latency，ttft，throughput，gpt memory
- Quality evaluation：
  - [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) 做 perplexity
  - 或只选一个小型 benchmark（如 GSM8K 子集 （数学推理正确性），可以抽取100-200题）作为补充

#### Framework / 工具
- Python 3.10+
- PyTorch
- Hugging Face Transformers
- bitsandbytes
- 可选：vLLM / AWQ
- `nvidia-smi` 或 PyTorch CUDA memory API 记录 GPU memory

Hugging Face bitsandbytes 文档：
- https://huggingface.co/docs/transformers/quantization/bitsandbytes

AWQ repo：
- https://github.com/mit-han-lab/llm-awq

#### 实验变量
控制以下因素：
- Precision：BF16/FP16、INT8、INT4
- Batch size：1 / 4 / 8 / 16（显存允许范围内）
- Prompt length：短 / 中 / 长，例如约 128 / 512 / 1024 tokens
- Model size：3B，资源允许时加 7B

每组实验运行多次，去掉 warm-up 后取平均值。

---

### 4. Metrics
主要系统指标：
- Peak GPU memory / VRAM
- Model loading time
- End-to-end latency
- TTFT (Time To First Token)
- TPOT / decode latency
- Throughput (tokens/s)
- Requests/s（如果做 batching）

质量指标：
- Perplexity（WikiText-2）
- 或简单任务 accuracy（可选）

最终重点观察：
> **Memory saving 与实际 speedup 是否一致，以及什么情况下 INT4/INT8 反而不比 FP16 更快。**

---

### 5. 预期图表
- Precision vs. Peak GPU Memory
- Precision vs. Throughput
- Precision vs. TTFT
- Batch Size vs. Throughput（不同 precision 多条曲线）
- Prompt Length vs. Latency
- Quality vs. Compression Ratio（可选）

---

### 6. 参考文献
1. Frantar et al., **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**, ICLR 2023.  
   https://arxiv.org/abs/2210.17323

2. Lin et al., **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**, MLSys 2024.  
   https://arxiv.org/abs/2306.00978  
   Repo: https://github.com/mit-han-lab/llm-awq

3. Xiao et al., **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**, ICML 2023.  
   https://arxiv.org/abs/2211.10438

4. Hugging Face Transformers — bitsandbytes Quantization.  
   https://huggingface.co/docs/transformers/quantization/bitsandbytes

---

### 7. 环境需求
建议：
- Linux + NVIDIA GPU
- CUDA 12.x（按实验服务器版本调整）
- 24 GB GPU（RTX 4090 / A5000 / A10 / A100 等）基本可以完成 3B/7B 小规模实验
- 如果只有较小 GPU，可只做 3B 模型

本项目不需要训练模型，主要工作是 **加载不同 precision 的模型 + benchmark + 分析结果**，因此整体计算成本较低。

---

### 8. 三人分工建议
- **成员 A：** BF16/FP16 + INT8 实验、环境搭建
- **成员 B：** INT4 / AWQ 实验、不同 batch/prompt length benchmark
- **成员 C：** quality evaluation、结果可视化与统计分析

三人共同完成结果解释、报告和 presentation。

---

### 9. 时间线（约 6 周）
**Week 1：** 阅读 GPTQ / AWQ，确定模型、环境和 benchmark scripts  
**Week 2：** 跑通 BF16 / INT8 / INT4 pipeline  
**Week 3：** 完成 batch size、prompt length、模型大小实验  
**Week 4：** 补充 quality evaluation / AWQ extension  
**Week 5：** 画图、分析结果、总结 trade-off  
**Week 6：** 写报告、制作 slides、复现实验并整理代码

---

### 10. 预期成果
最终希望得到一个清晰结论：

> Quantization 能稳定降低模型 memory footprint，但 inference speedup 并不只由 bit-width 决定；不同 workload、batch size 和 kernel/backend 下可能出现明显不同的 performance trade-off。
