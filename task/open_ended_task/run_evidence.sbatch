#!/bin/bash
# Site-specific: edit --account / --partition / --gres for your cluster.
#SBATCH --job-name=quest-ev
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/quest-ev-%A_%a.out
#SBATCH --error=logs/quest-ev-%A_%a.out

# Long-form question generation (PROMPT_VARIANT=evidence_first), Slurm version.
# Submit from the repo root (logs/ must exist there):
#
#   mkdir -p logs
#   export SERPER_API_KEY=... JINA_API_KEY=... MODEL_PATH=/path/to/Qwen3.5-122B-A10B
#   NUM_ITERATIONS=512 WORKERS=32 sbatch task/open_ended_task/run_evidence.sbatch
#
# Each array task serves its own copy of the model (TP=2) and writes
# outputs/ev<N>_<jobid>_<task>/{trajectories/,proposed_qa.jsonl}. The port and the
# tool caches are offset by the array index so two tasks can share a node.
#
# Trajectories are written as each question finishes, so a walltime kill keeps
# everything completed so far; re-run extract_evidence.py + fix_leaks.py over the
# directory to pick them up.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Qwen3.5-122B-A10B weights}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
PORT="${PORT:-$((8000 + TASK * 100))}"
TP_SIZE="${TP_SIZE:-2}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

: "${SERPER_API_KEY:?export SERPER_API_KEY (search tool)}"
: "${JINA_API_KEY:?export JINA_API_KEY (visit tool)}"

export NUM_ITERATIONS="${NUM_ITERATIONS:-512}"
export WORKERS="${WORKERS:-32}"
export PROMPT_VARIANT=evidence_first

RUN_TAG="${RUN_TAG:-ev${NUM_ITERATIONS}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}_${TASK}"
export OUT_DIR="${REPO_ROOT}/outputs/${RUN_TAG}"
export TRAJ_DIR="${OUT_DIR}/trajectories"
mkdir -p "${TRAJ_DIR}" "${REPO_ROOT}/logs" "${REPO_ROOT}/database"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-${CACHE_ROOT}/outlines}"
mkdir -p "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO

echo "=============================================================="
echo "node        : $(hostname -f)"
echo "array task  : ${TASK}  (port ${PORT})"
echo "questions   : ${NUM_ITERATIONS} (workers=${WORKERS})"
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

# --- 2. generate ---------------------------------------------------------------
cd "${REPO_ROOT}/task/open_ended_task"
source "${REPO_ROOT}/.venv/bin/activate"

LLM_API_BASE="http://127.0.0.1:${PORT}/v1"
export SERVED_MODEL_NAME TOKENIZER_PATH="${MODEL_PATH}" MODEL_PATH
export SERPER_KEY_ID="${SERPER_KEY_ID:-$SERPER_API_KEY}"
export JINA_API_KEYS="${JINA_API_KEYS:-$JINA_API_KEY}"
export DEEPRESEARCH_MODEL_NAME="vllm/${SERVED_MODEL_NAME}"
export DEEPRESEARCH_API_BASE="${LLM_API_BASE}" DEEPRESEARCH_OPENAI_API_KEY=EMPTY
export SUMMARY_MODEL_NAME="${SERVED_MODEL_NAME}" SUMMARY_OPENAI_API_KEY=EMPTY
export API_BASE="${LLM_API_BASE}" API_KEY=EMPTY
export SAVE_TRAJ=true
export VISIT_CACHE_ENABLED=true VISIT_CACHE_RESUME=true
export SEARCH_CACHE_ENABLED=true SEARCH_CACHE_RESUME=true
export VISIT_CACHE_FILE="${VISIT_CACHE_FILE:-${REPO_ROOT}/database/visit_${TASK}.db}"
export SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-${REPO_ROOT}/database/search_${TASK}.db}"

echo "checking outbound connectivity ..."
jina=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://r.jina.ai/ || echo 000)
serper=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://google.serper.dev/search || echo 000)
echo "  r.jina.ai -> ${jina}, google.serper.dev -> ${serper}"
[[ "${jina}" != "000" && "${serper}" != "000" ]] || { echo "FATAL: no outbound internet." >&2; exit 1; }

echo; echo "===== generating ${NUM_ITERATIONS} questions ====="
python generate_longform_tasks.py || echo "generator exited non-zero; extracting what completed"

traj=$(find "${TRAJ_DIR}" -name '*.json' | wc -l)
echo "trajectories written: ${traj}"
[[ "${traj}" -gt 0 ]] || { echo "FATAL: no trajectories produced." >&2; exit 1; }

# --- 3. extract, then rewrite figures leaked into the question -----------------
echo; echo "===== extracting ====="
python extract_evidence.py --input_dir "${TRAJ_DIR}" \
                           --output_file "${OUT_DIR}/proposed_qa.jsonl"

echo; echo "===== removing leaked figures (server still up) ====="
python fix_leaks.py --input "${OUT_DIR}/proposed_qa.jsonl" \
                    --output "${OUT_DIR}/proposed_qa.jsonl" \
                    --workers "${WORKERS}"

echo
echo "===== DONE ====="
echo "  ${OUT_DIR}/proposed_qa.jsonl   ($(wc -l < "${OUT_DIR}/proposed_qa.jsonl") questions from ${traj} trajectories)"
