#!/bin/bash
#SBATCH --job-name=quest-gen10
#SBATCH --account=torch_pr_520_general
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/bl4363/slurm_logs/quest-gen10-%j.out
#SBATCH --error=/scratch/bl4363/slurm_logs/quest-gen10-%j.out

# Generates open-ended long-form tasks and their rubrics against the vLLM server
# started by serving/serve_qwen35.sbatch.
#
#   NUM_ITERATIONS=10 sbatch task/open_ended_task/run_generation.sbatch
#
# This job is pure HTTP: it calls the local model, Serper, and Jina. It waits for
# the serving job to publish its endpoint before starting.

set -euo pipefail

REPO_ROOT="/scratch/bl4363/QUEST"
cd "${REPO_ROOT}/task/open_ended_task"

source "${REPO_ROOT}/.venv/bin/activate"

export NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
export WORKERS="${WORKERS:-4}"

RUN_TAG="${RUN_TAG:-gen${NUM_ITERATIONS}_${SLURM_JOB_ID:-manual}}"
OUT_DIR="${REPO_ROOT}/outputs/${RUN_TAG}"
export TRAJ_DIR="${OUT_DIR}/trajectories"
mkdir -p "${TRAJ_DIR}"

# --- wait for the model server ------------------------------------------------
ENDPOINT_FILE="${REPO_ROOT}/serving/endpoint.txt"
echo "waiting for ${ENDPOINT_FILE} ..."
for i in $(seq 1 360); do
  [[ -f "${ENDPOINT_FILE}" ]] && break
  sleep 30
done
[[ -f "${ENDPOINT_FILE}" ]] || { echo "FATAL: serving job never published an endpoint (waited 3h)." >&2; exit 1; }

source "${REPO_ROOT}/env.local.sh"
echo "endpoint: ${LLM_API_BASE}"

curl -sf --max-time 20 "${LLM_API_BASE%/v1}/health" >/dev/null \
  || { echo "FATAL: endpoint published but not healthy: ${LLM_API_BASE}" >&2; exit 1; }

# --- preflight: this node must reach the outside world ------------------------
# The search and visit tools call Serper and Jina. If compute nodes have no
# outbound route, fail here rather than after a long partial run.
echo "checking outbound connectivity ..."
jina_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://r.jina.ai/ || echo 000)
serper_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 https://google.serper.dev/search || echo 000)
echo "  r.jina.ai -> ${jina_code}, google.serper.dev -> ${serper_code}"
if [[ "${jina_code}" == "000" || "${serper_code}" == "000" ]]; then
  echo "FATAL: no outbound internet from $(hostname). Run this stage on the login node instead." >&2
  exit 1
fi

# --- 1. generate --------------------------------------------------------------
echo; echo "===== generating ${NUM_ITERATIONS} tasks (workers=${WORKERS}) ====="
python generate_longform_tasks.py

traj_count=$(find "${TRAJ_DIR}" -name '*.json' | wc -l)
echo "trajectories written: ${traj_count}"
[[ "${traj_count}" -gt 0 ]] || { echo "FATAL: no trajectories produced." >&2; exit 1; }

# --- 2. extract ---------------------------------------------------------------
echo; echo "===== extracting proposed QAs ====="
python extract_proposed_qa.py \
  --input_dir "${TRAJ_DIR}" \
  --output_file "${OUT_DIR}/proposed_qa.jsonl"
qa_count=$(wc -l < "${OUT_DIR}/proposed_qa.jsonl" 2>/dev/null || echo 0)
echo "questions extracted: ${qa_count}"
[[ "${qa_count}" -gt 0 ]] || { echo "FATAL: proposed_qa.jsonl is empty." >&2; exit 1; }

# --- 3. rubrics ---------------------------------------------------------------
echo; echo "===== generating rubrics ====="
python longform_rubric/generate_criteria.py \
  --input_file "${OUT_DIR}/proposed_qa.jsonl" \
  --output_file "${OUT_DIR}/criteria.jsonl"
crit_count=$(wc -l < "${OUT_DIR}/criteria.jsonl" 2>/dev/null || echo 0)

echo
echo "===== DONE ====="
echo "  ${OUT_DIR}/proposed_qa.jsonl   (${qa_count} questions)"
echo "  ${OUT_DIR}/criteria.jsonl      (${crit_count} rubric sets)"
