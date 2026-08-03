#!/bin/bash
#SBATCH --job-name=qwen35-serve
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/bl4363/slurm_logs/qwen35-serve-%j.out
#SBATCH --error=/scratch/bl4363/slurm_logs/qwen35-serve-%j.out

# Serves Qwen3.5-122B-A10B over an OpenAI-compatible HTTP API for the
# open-ended task/rubric pipeline.
#
#   sbatch serving/serve_qwen35.sbatch
#
# Once healthy, the reachable URL is written to serving/endpoint.txt, which
# env.local.sh picks up automatically.
#
# Context length matches OpenDR-eval's deploy/vllm_server.sh (262144). This is
# affordable here because only 12 of the 48 layers use full attention (the other
# 36 are linear attention, whose state does not grow with sequence length) and
# num_key_value_heads is 2:
#
#   2 (K,V) * 2 heads * 256 head_dim * 2 bytes * 12 layers = 24 KiB / token
#   262144 tokens -> 6.3 GB for one full-length sequence
#
# 234 GB of weights over TP=2 is 117 GB/GPU; at 0.92 utilization that leaves
# ~13 GB/GPU, i.e. room for roughly 1.1M tokens of KV.
#
# --no-async-scheduling works around a vLLM 0.26.0 crash: with async scheduling
# the engine takes the step_with_batch_queue path, and an aborted request leaves
# a stale id that scheduler.py:1670 looks up in req_id_to_index, raising KeyError
# and killing the engine. The agent aborts requests routinely (stop sequences,
# retries), so this fires within minutes. The workload is a handful of concurrent
# sessions, not throughput-bound, so the scheduling gap costs us nothing.
#
# No --reasoning-parser or --tool-call-parser: generation_agent_longform.py
# parses <think> and <tool_call> out of the raw message content itself, and a
# parser would move them into separate response fields.
#
# If this OOMs, raise --gres to gpu:h200:4 and TP_SIZE to 4, or lower MAX_LEN.

set -euo pipefail

REPO_ROOT="/scratch/bl4363/QUEST"
MODEL_PATH="${MODEL_PATH:-/scratch/bl4363/models/Qwen3.5-122B-A10B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"
# Each concurrent sequence holds a resident linear-attention ("Mamba") state
# block of roughly 38 MB/GPU, drawn from the same pool as the KV cache. vLLM's
# default max_num_seqs of 1024 needs more blocks than fit and aborts CUDA graph
# capture; only ~362 fit here. The pipeline runs a handful of workers, so 64 is
# ample and leaves the rest of the pool for KV.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ENDPOINT_FILE="${REPO_ROOT}/serving/endpoint.txt"

mkdir -p /scratch/bl4363/slurm_logs "${REPO_ROOT}/serving"

source "${REPO_ROOT}/.venv-vllm/bin/activate"

# Keep vLLM's compile cache off /home: it writes ~5400 files per run and the
# home inode quota is only 30k.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/bl4363/vllm_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/bl4363/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/bl4363/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/bl4363/torchinductor_cache}"
mkdir -p "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

export HF_HOME="${HF_HOME:-/scratch/bl4363/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/bl4363/hf_cache}"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/scratch/bl4363/triton_cache/outlines}"

HOST_ADDR="$(hostname -f)"
echo "=============================================================="
echo "node        : ${HOST_ADDR}"
echo "model       : ${MODEL_PATH}"
echo "served as   : ${SERVED_MODEL_NAME}"
echo "tp / maxlen : ${TP_SIZE} / ${MAX_LEN}"
echo "gpus        : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --no-async-scheduling \
  --trust-remote-code &
VLLM_PID=$!

# Publish the endpoint only after /health answers, so a client that finds the
# file can trust it. Clean it up on exit so nothing points at a dead server.
cleanup() {
  rm -f "${ENDPOINT_FILE}"
  kill "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for attempt in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "http://${HOST_ADDR}:${PORT}/v1" > "${ENDPOINT_FILE}"
    echo "READY after ~$((attempt * 10))s -> $(cat "${ENDPOINT_FILE}")"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "FATAL: vllm exited during startup; see the log above." >&2
    exit 1
  fi
  sleep 10
done

if [[ ! -f "${ENDPOINT_FILE}" ]]; then
  echo "FATAL: server did not become healthy within 30 minutes." >&2
  exit 1
fi

wait "${VLLM_PID}"
