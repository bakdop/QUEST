#!/bin/bash
#SBATCH --job-name=quest-ev
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/bl4363/slurm_logs/quest-ev-%A_%a.out
#SBATCH --error=/scratch/bl4363/slurm_logs/quest-ev-%A_%a.out

# PROMPT_VARIANT=evidence_first: one ReAct loop per question, one kind of object.
#
#   NUM_ITERATIONS=512 WORKERS=32 sbatch task/open_ended_task/run_evidence.sbatch
#
# A job array, so one submission is two parallel jobs of NUM_ITERATIONS each —
# 512+512 by default. The gpu48 QOS allows 24 GPUs per user and each task takes
# 2, so more is fine: `sbatch --array=0-3` gives four.
#
# Tasks may land on the same node, so the port and the tool caches are offset by
# the array index; sharing either would have two servers fighting over :8000 and
# two runs writing the same diskcache.
#
# Trajectories are written as each question finishes, so a walltime kill leaves
# everything completed up to that point; re-run extract_evidence.py over the
# directory to pick them up.

set -euo pipefail

REPO_ROOT="/scratch/bl4363/QUEST"
MODEL_PATH="${MODEL_PATH:-/scratch/bl4363/models/Qwen3.5-122B-A10B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-122b}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
PORT="${PORT:-$((8000 + TASK * 100))}"
TP_SIZE="${TP_SIZE:-2}"
MAX_LEN="${MAX_LEN:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

export NUM_ITERATIONS="${NUM_ITERATIONS:-512}"
export WORKERS="${WORKERS:-32}"
export PROMPT_VARIANT=evidence_first
export FIXED_PAIRS_FILE="${FIXED_PAIRS_FILE:-}"

RUN_TAG="${RUN_TAG:-ev${NUM_ITERATIONS}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}_${TASK}"
export OUT_DIR="${REPO_ROOT}/outputs/${RUN_TAG}"
export TRAJ_DIR="${OUT_DIR}/trajectories"
mkdir -p "${TRAJ_DIR}" /scratch/bl4363/slurm_logs

# separate tool caches per task — two runs on one node would otherwise write the
# same diskcache files
export VISIT_CACHE_FILE="${VISIT_CACHE_FILE:-${REPO_ROOT}/database/visit_${TASK}.db}"
export SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-${REPO_ROOT}/database/search_${TASK}.db}"
mkdir -p "${REPO_ROOT}/database"

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
export SAVE_TRAJ=true

echo "=============================================================="
echo "node        : $(hostname -f)"
echo "array task  : ${TASK}  (port ${PORT})"
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

# --- 3. extract, then take the answer back out of the questions ----------------
echo; echo "===== extracting ====="
python extract_evidence.py --input_dir "${TRAJ_DIR}" \
                           --output_file "${OUT_DIR}/proposed_qa.jsonl"

echo; echo "===== removing leaked figures (server still up) ====="
python fix_leaks.py --input "${OUT_DIR}/proposed_qa.jsonl" \
                    --output "${OUT_DIR}/proposed_qa.jsonl" \
                    --workers "${WORKERS}"

echo; echo "===== after the rewrite ====="
python - <<'PY2'
import json, os, statistics as st
p = os.path.join(os.environ["OUT_DIR"], "proposed_qa.jsonl")
rows = [json.loads(l) for l in open(p)]
sig = lambda k: sum(1 for r in rows if (r.get("_signals") or {}).get(k))
print(f"  questions            {len(rows)}")
print(f"  rewritten            {sig('leak_rewritten')}")
print(f"  still leaking        {sig('leak_unfixed')}")
print(f"  situated             {sum(1 for r in rows if (r.get('_signals') or {}).get('situated'))}")
asks = [(r.get('_signals') or {}).get('coordinate_asks', 0) for r in rows]
print(f"  coordinate asks      median {st.median(asks) if asks else 0}   (ResearchRubrics: 1)")
PY2

echo
echo "===== DONE ====="
echo "  ${OUT_DIR}/proposed_qa.jsonl   ($(wc -l < "${OUT_DIR}/proposed_qa.jsonl") questions from ${traj} trajectories)"
