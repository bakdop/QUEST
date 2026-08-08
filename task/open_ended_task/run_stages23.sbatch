#!/bin/bash
#SBATCH --job-name=quest-s23
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/bl4363/slurm_logs/quest-s23-%j.out
#SBATCH --error=/scratch/bl4363/slurm_logs/quest-s23-%j.out

# STAGES 2 and 3, over an existing stage-1 run.
#
#   SPINE_RUN=spinefix8_15469596 sbatch task/open_ended_task/run_stages23.sbatch
#
# Deliberately separate from run_spine.sbatch. Stage 1 is the expensive half
# (all the search and page fetching) and its corpus is reusable, so iterating on
# the stage 2/3 prompts should not re-pay for it. Reads
# outputs/${SPINE_RUN}/spines.jsonl + trajectories, writes proposed_qa.jsonl
# next to them.
#
# RUBRICS=1 also runs the existing rubric generator on the result; the schema
# generate_stages23.py writes is the one it already expects.

set -euo pipefail

REPO_ROOT="/scratch/bl4363/QUEST"
MODEL_PATH="${MODEL_PATH:-/scratch/bl4363/models/Qwen3.5-122B-A10B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

: "${SPINE_RUN:?set SPINE_RUN=<stage-1 run tag under outputs/>}"
OUT_DIR="${REPO_ROOT}/outputs/${SPINE_RUN}"
SPINES="${OUT_DIR}/spines.jsonl"
TRAJ_DIR="${OUT_DIR}/trajectories"
export WORKERS="${WORKERS:-4}"
REQUIRE_QC="${REQUIRE_QC:-0}"
RUBRICS="${RUBRICS:-0}"

[[ -f "${SPINES}" ]]     || { echo "FATAL: no ${SPINES}" >&2; exit 1; }
[[ -d "${TRAJ_DIR}" ]]   || { echo "FATAL: no ${TRAJ_DIR}" >&2; exit 1; }

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/bl4363/vllm_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/bl4363/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/bl4363/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/bl4363/torchinductor_cache}"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/scratch/bl4363/triton_cache/outlines}"
mkdir -p "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}" /scratch/bl4363/slurm_logs

export HF_HOME="${HF_HOME:-/scratch/bl4363/.huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/bl4363/hf_cache}"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO

echo "=============================================================="
echo "node        : $(hostname -f)"
echo "stage-1 run : ${SPINE_RUN}   ($(wc -l < "${SPINES}") spines)"
echo "workers     : ${WORKERS}   require_qc=${REQUIRE_QC}  rubrics=${RUBRICS}"
echo "=============================================================="

# --- 1. start the server ------------------------------------------------------
(
  source "${REPO_ROOT}/.venv-vllm/bin/activate"
  exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --no-async-scheduling \
    --trust-remote-code
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
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "FATAL: vllm exited during startup." >&2; exit 1
  fi
  sleep 10
done
[[ "${READY}" -eq 1 ]] || { echo "FATAL: server not healthy within 30 minutes." >&2; exit 1; }

cd "${REPO_ROOT}/task/open_ended_task"
source "${REPO_ROOT}/.venv/bin/activate"
export LLM_API_BASE="http://127.0.0.1:${PORT}/v1"
source "${REPO_ROOT}/env.local.sh"

# Stage 2 searches, so the tools must be reachable. Stage 3 does not.
echo "checking outbound connectivity ..."
jina_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://r.jina.ai/ || echo 000)
serper_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://google.serper.dev/search || echo 000)
echo "  r.jina.ai -> ${jina_code}, google.serper.dev -> ${serper_code}"
if [[ "${jina_code}" == "000" || "${serper_code}" == "000" ]]; then
  echo "FATAL: no outbound internet from $(hostname)." >&2; exit 1
fi

# --- 2. stages 2 and 3 --------------------------------------------------------
QC_FLAG=""
[[ "${REQUIRE_QC}" == "1" ]] && QC_FLAG="--require_qc_pass"

echo; echo "===== STAGES 2+3 ====="
python generate_stages23.py \
  --spines   "${SPINES}" \
  --traj_dir "${TRAJ_DIR}" \
  --out_dir  "${OUT_DIR}" \
  --workers  "${WORKERS}" ${QC_FLAG}

qa_count=$(wc -l < "${OUT_DIR}/proposed_qa.jsonl" 2>/dev/null || echo 0)
[[ "${qa_count}" -gt 0 ]] || { echo "FATAL: proposed_qa.jsonl is empty." >&2; exit 1; }

# --- 3. rubrics (optional) ----------------------------------------------------
if [[ "${RUBRICS}" == "1" ]]; then
  echo; echo "===== rubrics ====="
  python longform_rubric/generate_criteria_findings.py \
    --input_file  "${OUT_DIR}/proposed_qa.jsonl" \
    --output_file "${OUT_DIR}/criteria.jsonl"
fi

echo
echo "===== DONE ====="
echo "  ${OUT_DIR}/proposed_qa.jsonl        (${qa_count} questions)"
echo "  ${OUT_DIR}/stage2_trajectories/"
echo "  ${OUT_DIR}/stage3_trajectories/"
