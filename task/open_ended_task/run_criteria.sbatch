#!/bin/bash
# Site-specific: edit --account / --partition / --gres for your cluster.
#SBATCH --job-name=quest-criteria
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/quest-criteria-%A_%a.out
#SBATCH --error=logs/quest-criteria-%A_%a.out

# Rubric generation for evidence_first runs, Slurm version. One run directory per
# array task; each task serves the model (TP=2) and writes <run>/criteria.jsonl.
# Submit from the repo root:
#
#   export MODEL_PATH=/path/to/Qwen3.5-122B-A10B
#   RUN_DIRS="outputs/ev512_123_0 outputs/ev512_123_1" sbatch --array=0-1 task/open_ended_task/run_criteria.sbatch
#   LIMIT=16 RUN_DIRS="outputs/ev512_123_0" sbatch --array=0-0 ...   # smoke test -> criteria_sample16.jsonl
#
# No internet needed: the corpus is read from the run's own trajectories.
# The output file is opened with "w", so a re-run starts that directory over.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Qwen3.5-122B-A10B weights}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b}"
: "${RUN_DIRS:?set RUN_DIRS to space-separated run directories (relative to the repo root)}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
LIMIT="${LIMIT:-0}"
read -ra DIRS <<< "${RUN_DIRS}"
if [[ "${TASK}" -ge "${#DIRS[@]}" ]]; then
  echo "FATAL: task ${TASK} but only ${#DIRS[@]} run dirs; pass --array=0-$(( ${#DIRS[@]} - 1 ))" >&2
  exit 1
fi
RUN_DIR="${DIRS[${TASK}]}"

PORT="${PORT:-$(( 8000 + TASK * 100 ))}"
TP_SIZE="${TP_SIZE:-2}"
# The prompt is ~25k tokens (mostly corpus) plus up to 16k of output.
MAX_LEN="${MAX_LEN:-131072}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

# Matched to --max-num-seqs: more workers than slots just queue inside vLLM.
export CRITERIA_MAX_WORKERS="${CRITERIA_MAX_WORKERS:-16}"
export CRITERIA_MAX_TOKENS="${CRITERIA_MAX_TOKENS:-16000}"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-${CACHE_ROOT}/outlines}"
mkdir -p "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" "${REPO_ROOT}/logs"
export HF_HUB_OFFLINE=1

OUT="${REPO_ROOT}/${RUN_DIR}/criteria.jsonl"
[[ "${LIMIT}" -gt 0 ]] && OUT="${REPO_ROOT}/${RUN_DIR}/criteria_sample${LIMIT}.jsonl"

echo "=============================================================="
echo "node     : $(hostname -f)"
echo "task     : ${TASK} of ${#DIRS[@]}  (port ${PORT})"
echo "run dir  : ${RUN_DIR} -> ${OUT}"
echo "limit    : ${LIMIT}  (0 = all)"
echo "=============================================================="

# Checked before model loading, not after.
for f in "${REPO_ROOT}/${RUN_DIR}/proposed_qa.jsonl" "${REPO_ROOT}/${RUN_DIR}/trajectories"; do
  [[ -e "${f}" ]] || { echo "FATAL: missing ${f}" >&2; exit 1; }
done

# --- 1. serve --------------------------------------------------------------
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
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

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

# --- 2. generate rubrics ---------------------------------------------------
cd "${REPO_ROOT}/task/open_ended_task"
source "${REPO_ROOT}/.venv/bin/activate"

export CRITERIA_MODEL_NAME="hosted_vllm/${SERVED_MODEL_NAME}"
export API_BASE="http://127.0.0.1:${PORT}/v1"
export API_KEY=EMPTY

rc=0
python longform_rubric/generate_criteria_evidence.py \
    --input_file  "${REPO_ROOT}/${RUN_DIR}/proposed_qa.jsonl" \
    --traj_dir    "${REPO_ROOT}/${RUN_DIR}/trajectories" \
    --output_file "${OUT}" \
    --limit "${LIMIT}" || rc=1

echo
echo "===== DONE (exit ${rc}) ====="
echo "  $(wc -l < "${OUT}" 2>/dev/null || echo 0) rubrics  ${OUT}"
exit "${rc}"
