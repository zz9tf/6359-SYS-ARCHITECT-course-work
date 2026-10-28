#!/usr/bin/env bash
# Reproduces every measurement of the study on one GPU, in the order we ran them.
# Total GPU time on an RTX 5090: roughly 8-9 hours (the bitsandbytes INT8 runs dominate).
#
#   uv venv .venv --python 3.12 && uv pip install -r requirements.txt
#   source .venv/bin/activate && ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1

# 0. Workload sample + hardware ceilings
python scripts/prepare_sharegpt.py                      # data/sharegpt_300.json
python scripts/hw_peaks.py                              # DRAM bandwidth, BF16/FP8/INT8 GEMM peaks

# 1. Controlled sweeps: precision x batch x prompt length (Q1-Q3)
for size in 3b 7b; do
  for p in bf16 int8 nf4; do python scripts/hf_sweep.py --size $size --precision $p; done
  for p in bf16 fp8 gptq-int8 awq-int4 gptq-int4; do python scripts/vllm_sweep.py --size $size --precision $p; done
done
for p in awq-int4 gptq-int4 gptq-int8 fp8; do
  python scripts/vllm_sweep.py --size 14b --precision $p --batch-sizes 1,4,16,64 --input-lens 512,2048
done
python scripts/vllm_sweep.py --size 14b --precision bf16 --gpu-util 0.95 --batch-sizes 1,4,16,64 --input-lens 512,2048
for p in bf16 nf4 int8; do
  python scripts/hf_sweep.py --size 14b --precision $p --batch-sizes 1,4,16,64 --input-lens 512,2048
done
# output-length sub-sweep (Q2)
for p in bf16 int8 nf4; do
  python scripts/hf_sweep.py --size 7b --precision $p --input-lens 512 --batch-sizes 1,16 --output-lens 32,512 --tag outlen
done
for p in bf16 awq-int4 fp8 gptq-int8; do for n in 32 512; do
  python scripts/vllm_sweep.py --size 7b --precision $p --input-lens 512 --batch-sizes 1,16 --output-len $n --tag outlen
done; done

# 2. Mechanism: linear-layer kernels and per-step profile (Q3)
python scripts/kernel_bench.py --size 7b
python scripts/profile_decode.py --size 7b

# 3. Real workload: 300 ShareGPT requests (Q2/Q3)
for p in bf16 fp8 gptq-int8 awq-int4 gptq-int4; do scripts/serve_bench.sh 7b $p "1 4 16 64"; done
for p in bf16 int8 nf4; do python scripts/hf_sharegpt.py --size 7b --precision $p --batch-sizes 16; done

# 4. Quality (Q4)
for size in 3b 7b; do
  for p in bf16 int8 nf4; do
    python scripts/eval_ppl.py --backend hf --size $size --precision $p
    python scripts/eval_gsm8k.py --backend hf --size $size --precision $p
  done
  for p in bf16 fp8 gptq-int8 awq-int4 gptq-int4; do
    python scripts/eval_ppl.py --backend vllm --size $size --precision $p
    python scripts/eval_gsm8k.py --backend vllm --size $size --precision $p
  done
done
for p in fp8 gptq-int8 awq-int4 gptq-int4; do
  python scripts/eval_ppl.py --backend vllm --size 14b --precision $p
  python scripts/eval_gsm8k.py --backend vllm --size 14b --precision $p
done
python scripts/eval_ppl.py --backend hf --size 14b --precision bf16
python scripts/eval_gsm8k.py --backend hf --size 14b --precision bf16 --batch 8
for p in nf4 int8; do
  python scripts/eval_ppl.py --backend hf --size 14b --precision $p
  python scripts/eval_gsm8k.py --backend hf --size 14b --precision $p --batch 32
done
# MMLU (all 14,042 questions)
for size in 3b 7b 14b; do for p in bf16 fp8 gptq-int8 awq-int4 gptq-int4; do
  [ "$size" = 14b ] && [ "$p" = bf16 ] && continue      # 14B BF16 does not fit vLLM's cache; HF runs it below
  python scripts/eval_mmlu.py --backend vllm --size $size --precision $p
done; done
for size in 3b 7b; do
  python scripts/eval_mmlu.py --backend hf --size $size --precision bf16 --batch 16
  python scripts/eval_mmlu.py --backend hf --size $size --precision nf4 --batch 16
  python scripts/eval_mmlu.py --backend hf --size $size --precision int8 --batch 8
done
python scripts/eval_mmlu.py --backend hf --size 14b --precision bf16 --batch 8
python scripts/eval_mmlu.py --backend hf --size 14b --precision nf4 --batch 16
python scripts/eval_mmlu.py --backend hf --size 14b --precision int8 --batch 4
# Diagnosing the 14B GPTQ-INT4 perplexity outlier: activation dtype, and next-token ranking on raw vs chat text
python scripts/eval_ppl.py --backend vllm --size 14b --precision gptq-int4 --dtype bfloat16 --out results/quality_ppl_diag.jsonl
for s in 14b 7b; do for p in gptq-int4 awq-int4 gptq-int8 fp8; do
  python scripts/diag_ppl_topk.py --size $s --precision $p
  python scripts/diag_ppl_topk.py --size $s --precision $p --corpus gsm8k-chat --windows 400
done; done

# 5. Second model family: Qwen3-8B (non-thinking mode; no official GPTQ checkpoint)
for p in bf16 fp8 awq-int4; do
  python scripts/vllm_sweep.py --size q3-8b --precision $p --batch-sizes 1,4,16,64,256 --input-lens 512,2048
  python scripts/eval_ppl.py --backend vllm --size q3-8b --precision $p
  python scripts/eval_gsm8k.py --backend vllm --size q3-8b --precision $p
  python scripts/eval_mmlu.py --backend vllm --size q3-8b --precision $p
done
for p in bf16 nf4 int8; do
  python scripts/hf_sweep.py --size q3-8b --precision $p --batch-sizes 1,4,16 --input-lens 512
  python scripts/hf_sweep.py --size q3-8b --precision $p --batch-sizes 1,4 --input-lens 2048
  python scripts/eval_ppl.py --backend hf --size q3-8b --precision $p
  python scripts/eval_gsm8k.py --backend hf --size q3-8b --precision $p --batch $([ $p = int8 ] && echo 16 || echo 32)
  python scripts/eval_mmlu.py --backend hf --size q3-8b --precision $p --batch $([ $p = int8 ] && echo 8 || echo 16)
done

# 6. Figures and tables
python scripts/make_figures.py
python scripts/make_tables.py
