#!/usr/bin/env bash
# ShareGPT serving benchmark on vLLM (Q2/Q3, real workload).
# Starts one OpenAI-compatible vLLM server for the configuration, replays the fixed 300-request
# ShareGPT sample with `vllm bench serve` at several concurrency levels (closed loop: a new request
# is sent as soon as one finishes), then shuts the server down.
#
#   scripts/serve_bench.sh 7b awq-int4 "1 4 16 64"
set -euo pipefail
cd "$(dirname "$0")/.."
SIZE=$1
PREC=$2
CONC=${3:-"1 4 16 64"}
PORT=${PORT:-8123}
export VLLM_USE_FLASHINFER_SAMPLER=0

read -r MODEL QUANT < <(python -c "from quantbench import configs; m, q = configs.vllm_spec('$SIZE', '$PREC'); print(m, q or '')")
QARG=()
[ -n "${QUANT:-}" ] && QARG=(--quantization "$QUANT")
mkdir -p results/serve results/serve_logs

vllm serve "$MODEL" "${QARG[@]}" --port "$PORT" --gpu-memory-utilization 0.85 --max-model-len 4096 \
    --max-num-seqs 256 --no-enable-prefix-caching --seed 0 > "results/serve_logs/${SIZE}_${PREC}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
until curl -sf "localhost:$PORT/health" > /dev/null; do
    kill -0 "$SERVER" 2>/dev/null || { echo "server died"; tail -20 "results/serve_logs/${SIZE}_${PREC}.log"; exit 1; }
    sleep 5
done

for C in $CONC; do
    vllm bench serve --backend vllm --base-url "http://localhost:$PORT" --model "$MODEL" \
        --dataset-name sharegpt --dataset-path data/sharegpt_300.json --num-prompts 300 \
        --max-concurrency "$C" --ignore-eos --seed 0 --num-warmups 8 \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --result-dir results/serve --result-filename "${SIZE}_${PREC}_c${C}.json"
done
