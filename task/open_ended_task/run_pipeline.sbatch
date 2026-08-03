#!/bin/bash
#SBATCH --job-name=quest-pipeline
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/bl4363/slurm_logs/quest-pipeline-%j.out
#SBATCH --error=/scratch/bl4363/slurm_logs/quest-pipeline-%j.out

# Serves Qwen3.5-122B and runs the full open-ended pipeline against it in one
# job, then exits.
#
#   NUM_ITERATIONS=10 sbatch task/open_ended_task/run_pipeline.sbatch
#
# This supersedes the serve + run_generation pair. Splitting them left the GPU
# idle between stages while the server waited to be fed work, and a serving job
# that sat idle was reaped by the scheduler (CANCELLED by uid 0) mid-run,
# killing two in-flight rubric requests. Here the GPU is busy from the moment
# the model finishes loading until the last rubric is written, and the job
# releases its allocation as soon as the work is done.
#
# The pipeline talks to 127.0.0.1 directly, so no endpoint file is involved.
# GPU nodes do have outbound internet (verified on gh109: r.jina.ai 200,
# google.serper.dev 403 i.e. reachable but unauthenticated), which the search
# and visit tools need.
#
# Serving flags are explained in serving/serve_qwen35.sbatch, which is still the
# right entry point when you want a long-lived server for interactive use.

set -euo pipefail

REPO_ROOT="/scratch/bl4363/QUEST"
MODEL_PATH="${MODEL_PATH:-/scratch/bl4363/models/Qwen3.5-122B-A10B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

export NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
export WORKERS="${WORKERS:-4}"
export CRITERIA_MAX_WORKERS="${CRITERIA_MAX_WORKERS:-4}"

# default  -> the original QUEST proposer + four-dimension rubrics
# findings -> investigate for structure first, record findings, then write the
#             question from them; rubrics are grounded in those findings and
#             shaped like ResearchRubrics (flat list, per-item weight, axis).
export PROMPT_VARIANT="${PROMPT_VARIANT:-default}"

RUN_TAG="${RUN_TAG:-${PROMPT_VARIANT}${NUM_ITERATIONS}_${SLURM_JOB_ID:-manual}}"
OUT_DIR="${REPO_ROOT}/outputs/${RUN_TAG}"
export TRAJ_DIR="${OUT_DIR}/trajectories"
mkdir -p "${TRAJ_DIR}" /scratch/bl4363/slurm_logs

# Keep regenerable caches off /home: vLLM's compile cache alone writes ~5400
# files per run and the home inode quota is only 30k.
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
echo "iterations  : ${NUM_ITERATIONS} (workers=${WORKERS})"
echo "output      : ${OUT_DIR}"
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
    echo "READY after ~$((attempt * 10))s"
    READY=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "FATAL: vllm exited during startup; see the log above." >&2
    exit 1
  fi
  sleep 10
done
[[ "${READY}" -eq 1 ]] || { echo "FATAL: server not healthy within 30 minutes." >&2; exit 1; }

# --- 2. point the pipeline at the local server --------------------------------
cd "${REPO_ROOT}/task/open_ended_task"
source "${REPO_ROOT}/.venv/bin/activate"
export LLM_API_BASE="http://127.0.0.1:${PORT}/v1"
source "${REPO_ROOT}/env.local.sh"

# The search and visit tools call Serper and Jina. Fail here rather than after a
# long partial run. A 4xx means reachable-but-unauthenticated, which is fine;
# only a connection failure (000) is fatal.
echo "checking outbound connectivity ..."
jina_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://r.jina.ai/ || echo 000)
serper_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://google.serper.dev/search || echo 000)
echo "  r.jina.ai -> ${jina_code}, google.serper.dev -> ${serper_code}"
if [[ "${jina_code}" == "000" || "${serper_code}" == "000" ]]; then
  echo "FATAL: no outbound internet from $(hostname)." >&2
  exit 1
fi

# --- 3. generate --------------------------------------------------------------
echo; echo "===== generating ${NUM_ITERATIONS} tasks ====="
python generate_longform_tasks.py

traj_count=$(find "${TRAJ_DIR}" -name '*.json' | wc -l)
echo "trajectories written: ${traj_count}"
[[ "${traj_count}" -gt 0 ]] || { echo "FATAL: no trajectories produced." >&2; exit 1; }

# A trajectory whose tool calls never parsed produced its question from the
# model's own knowledge rather than from search, which the prompt forbids.
# Report it here; these are worth discarding before the data is used.
python - "${TRAJ_DIR}" <<'PY'
import glob, json, os, sys
ungrounded = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    d = json.load(open(path))
    asst = ' '.join((m.get('content') or '') for m in d['messages'] if m.get('role') == 'assistant')
    resp = ' '.join((m.get('content') or '') for m in d['messages'] if m.get('role') == 'user')
    if asst.count('<tool_call>') - resp.count('"tool_name": "unknown"') <= 0:
        ungrounded.append(os.path.basename(path))
print(f"trajectories with no successful tool result: {len(ungrounded)}")
for name in ungrounded:
    print(f"  UNGROUNDED {name}")
PY

# --- 4. extract ---------------------------------------------------------------
echo; echo "===== extracting proposed QAs ====="
python extract_proposed_qa.py \
  --input_dir "${TRAJ_DIR}" \
  --output_file "${OUT_DIR}/proposed_qa.jsonl"
qa_count=$(wc -l < "${OUT_DIR}/proposed_qa.jsonl" 2>/dev/null || echo 0)
[[ "${qa_count}" -gt 0 ]] || { echo "FATAL: proposed_qa.jsonl is empty." >&2; exit 1; }

# --- 5. rubrics ---------------------------------------------------------------
echo; echo "===== generating rubrics (${PROMPT_VARIANT}) ====="
if [[ "${PROMPT_VARIANT}" == "findings" ]]; then
  CRITERIA_SCRIPT=longform_rubric/generate_criteria_findings.py
else
  CRITERIA_SCRIPT=longform_rubric/generate_criteria.py
fi
python "${CRITERIA_SCRIPT}" \
  --input_file "${OUT_DIR}/proposed_qa.jsonl" \
  --output_file "${OUT_DIR}/criteria.jsonl"
crit_count=$(wc -l < "${OUT_DIR}/criteria.jsonl" 2>/dev/null || echo 0)

echo
echo "===== DONE ====="
echo "  ${OUT_DIR}/proposed_qa.jsonl   (${qa_count} questions)"
echo "  ${OUT_DIR}/criteria.jsonl      (${crit_count} rubric sets)"
if [[ "${crit_count}" -ne "${qa_count}" ]]; then
  echo "  NOTE: $((qa_count - crit_count)) question(s) have no rubric; see the failures above."
fi
