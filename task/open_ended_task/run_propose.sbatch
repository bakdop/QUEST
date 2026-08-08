#!/bin/bash
#SBATCH --job-name=quest-propose
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/bl4363/slurm_logs/quest-propose-%j.out
#SBATCH --error=/scratch/bl4363/slurm_logs/quest-propose-%j.out

# One call per question: keyword -> research -> question.
# See generation_prompt_propose.py for the four steps and why the order matters.
#
#   NUM_ITERATIONS=512 WORKERS=32 sbatch task/open_ended_task/run_propose.sbatch
#
# Trajectories are written as each finishes, so a walltime kill still leaves
# everything completed up to that point; re-running the extractor over the
# directory picks them all up.

set -euo pipefail

REPO_ROOT="/scratch/bl4363/QUEST"
MODEL_PATH="${MODEL_PATH:-/scratch/bl4363/models/Qwen3.5-122B-A10B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

export NUM_ITERATIONS="${NUM_ITERATIONS:-512}"
export WORKERS="${WORKERS:-32}"
export PROMPT_VARIANT=propose
export FIXED_PAIRS_FILE="${FIXED_PAIRS_FILE:-}"

RUN_TAG="${RUN_TAG:-propose${NUM_ITERATIONS}_${SLURM_JOB_ID:-manual}}"
OUT_DIR="${REPO_ROOT}/outputs/${RUN_TAG}"
export TRAJ_DIR="${OUT_DIR}/trajectories"
mkdir -p "${TRAJ_DIR}" /scratch/bl4363/slurm_logs

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/bl4363/vllm_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/bl4363/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/bl4363/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/bl4363/torchinductor_cache}"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/scratch/bl4363/triton_cache/outlines}"
mkdir -p "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

export HF_HOME="${HF_HOME:-/scratch/bl4363/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/bl4363/hf_cache}"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO

echo "=============================================================="
echo "node        : $(hostname -f)"
echo "run tag     : ${RUN_TAG}"
echo "questions   : ${NUM_ITERATIONS} (workers=${WORKERS}, port=${PORT})"
echo "seeds       : ${FIXED_PAIRS_FILE:-<sampled at random>}"
echo "output      : ${OUT_DIR}"
echo "=============================================================="

# --- 1. serve ------------------------------------------------------------------
(
  source "${REPO_ROOT}/.venv-vllm/bin/activate"
  exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --no-async-scheduling --trust-remote-code
) &
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for the model to load ..."
READY=0
for attempt in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "READY after ~$((attempt * 10))s"; READY=1; break
  fi
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "FATAL: vllm exited during startup." >&2; exit 1; }
  sleep 10
done
[[ "${READY}" -eq 1 ]] || { echo "FATAL: server not healthy within 30 minutes." >&2; exit 1; }

cd "${REPO_ROOT}/task/open_ended_task"
source "${REPO_ROOT}/.venv/bin/activate"
export LLM_API_BASE="http://127.0.0.1:${PORT}/v1"
source "${REPO_ROOT}/env.local.sh"

echo "checking outbound connectivity ..."
jina=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://r.jina.ai/ || echo 000)
serper=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://google.serper.dev/search || echo 000)
echo "  r.jina.ai -> ${jina}, google.serper.dev -> ${serper}"
[[ "${jina}" != "000" && "${serper}" != "000" ]] || { echo "FATAL: no outbound internet." >&2; exit 1; }

# --- 2. generate ---------------------------------------------------------------
echo; echo "===== generating ${NUM_ITERATIONS} questions ====="
python generate_longform_tasks.py || echo "generator exited non-zero; extracting what completed"

traj=$(find "${TRAJ_DIR}" -name '*.json' | wc -l)
echo "trajectories written: ${traj}"
[[ "${traj}" -gt 0 ]] || { echo "FATAL: no trajectories produced." >&2; exit 1; }

# --- 3. extract + audit --------------------------------------------------------
echo; echo "===== extracting and auditing ====="
python extract_propose.py --input_dir "${TRAJ_DIR}" --output_file "${OUT_DIR}/proposed_qa.jsonl"

n=$(wc -l < "${OUT_DIR}/proposed_qa.jsonl" 2>/dev/null || echo 0)
echo
echo "===== DONE ====="
echo "  ${OUT_DIR}/proposed_qa.jsonl   (${n} questions from ${traj} trajectories)"
