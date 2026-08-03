#!/bin/bash
# End-to-end smoke test for the open-ended question + rubric pipeline.
#
#   bash task/open_ended_task/smoke_test.sh
#
# Runs a deliberately tiny generation (2 tasks) through all three stages and
# fails loudly if any stage produces nothing. Use this to validate a fresh
# endpoint before committing to a full-scale run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

source "${REPO_ROOT}/.venv/bin/activate"
source "${REPO_ROOT}/env.local.sh"

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/smoke}"
export TRAJ_DIR="${OUT_DIR}/trajectories"
export NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
export WORKERS="${WORKERS:-2}"
export CRITERIA_PROCESS_LIMIT="${CRITERIA_PROCESS_LIMIT:-2}"
export CRITERIA_MAX_WORKERS="${CRITERIA_MAX_WORKERS:-2}"
mkdir -p "${TRAJ_DIR}"

fail() { echo "SMOKE TEST FAILED: $*" >&2; exit 1; }

step() { echo; echo "===== $* ====="; }

# --- 0. endpoint reachable ----------------------------------------------------
step "0/3 endpoint check: ${LLM_API_BASE}"
curl -sf "${LLM_API_BASE%/v1}/health" >/dev/null \
  || fail "no healthy server at ${LLM_API_BASE}. Start it with: sbatch serving/serve_qwen35.sbatch"
echo "served models: $(curl -sf "${LLM_API_BASE}/models" | head -c 400)"

# --- 1. generate long-form tasks ---------------------------------------------
step "1/3 generate ${NUM_ITERATIONS} long-form tasks"
python generate_longform_tasks.py
traj_count=$(find "${TRAJ_DIR}" -name '*.json' | wc -l)
[[ "${traj_count}" -gt 0 ]] || fail "no trajectories written to ${TRAJ_DIR}"
echo "trajectories: ${traj_count}"

# --- 2. extract proposed QAs --------------------------------------------------
step "2/3 extract proposed QAs"
python extract_proposed_qa.py \
  --input_dir "${TRAJ_DIR}" \
  --output_file "${OUT_DIR}/proposed_qa.jsonl"
qa_count=$(wc -l < "${OUT_DIR}/proposed_qa.jsonl" 2>/dev/null || echo 0)
[[ "${qa_count}" -gt 0 ]] || fail "proposed_qa.jsonl is empty — the model likely did not emit the expected QA format"
echo "proposed QAs: ${qa_count}"

# --- 3. generate rubrics ------------------------------------------------------
step "3/3 generate rubrics"
python longform_rubric/generate_criteria.py \
  --input_file "${OUT_DIR}/proposed_qa.jsonl" \
  --output_file "${OUT_DIR}/criteria.jsonl"
crit_count=$(wc -l < "${OUT_DIR}/criteria.jsonl" 2>/dev/null || echo 0)
[[ "${crit_count}" -gt 0 ]] || fail "criteria.jsonl is empty"
echo "rubrics: ${crit_count}"

echo
echo "===== SMOKE TEST PASSED ====="
echo "  ${OUT_DIR}/proposed_qa.jsonl   (${qa_count} questions)"
echo "  ${OUT_DIR}/criteria.jsonl      (${crit_count} rubric sets)"
