#!/usr/bin/env bash
# Rubric generation for evidence_first runs on a single 8-GPU node without Slurm.
# Two TP=4 servers (GPUs 0-3 :8000, GPUs 4-7 :8100); the run directories are dealt
# across them and each gets <run>/criteria.jsonl.
#
# Usage:
#   QUEST=/path/to/QUEST MODEL=/path/to/Qwen3.5-122B-A10B \
#   RUN_DIRS="outputs/ev0809_1519_a outputs/ev0809_1519_b" bash run_criteria_node.sh
#
#   LIMIT=16 ...      # smoke test -> criteria_sample16.jsonl
#   SERVERS=a ...     # one server only (4 GPUs)
#
# No internet needed: the corpus is read from the run's own trajectories.
# The output file is opened with "w", so a re-run starts that directory over.
set -uo pipefail

QUEST="${QUEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL="${MODEL:?set MODEL to the Qwen3.5-122B-A10B weights}"
VENV_VLLM="${VENV_VLLM:-$QUEST/.venv-vllm}"
VENV_GEN="${VENV_GEN:-$QUEST/.venv}"
RUN_DIRS="${RUN_DIRS:?set RUN_DIRS to space-separated run directories (relative to QUEST)}"

SERVERS="${SERVERS:-a b}"
LIMIT="${LIMIT:-0}"                # 0 = every question
TP="${TP:-4}"
# The prompt is ~25k tokens (mostly corpus) plus up to 16k of output.
CTX="${CTX:-65536}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
SERVED="${SERVED:-qwen3.5-122b}"
TAG="${TAG:-crit$(date +%m%d_%H%M)}"

# Matched to --max-num-seqs: more workers than slots just queue inside vLLM.
export CRITERIA_MAX_WORKERS="${CRITERIA_MAX_WORKERS:-16}"
export CRITERIA_MAX_TOKENS="${CRITERIA_MAX_TOKENS:-16000}"

SRC="$QUEST/task/open_ended_task"
LOGS="$QUEST/logs"

# vLLM is served on 127.0.0.1; keep it off any configured proxy.
export no_proxy="${no_proxy:-},127.0.0.1,localhost,0.0.0.0,::1"
export NO_PROXY="$no_proxy"

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

# --------------------------------------------------------------- preflight ---
[[ -d "$SRC" ]]                    || die "no $SRC (set QUEST=...)"
[[ -d "$MODEL" ]]                  || die "model not at $MODEL"
[[ -f "$VENV_VLLM/bin/activate" ]] || die "no vllm venv at $VENV_VLLM (set VENV_VLLM=...)"
[[ -f "$VENV_GEN/bin/activate" ]]  || die "no generation venv at $VENV_GEN (set VENV_GEN=...)"

dirs=()
for d in $RUN_DIRS; do
  [[ -f "$QUEST/$d/proposed_qa.jsonl" && -d "$QUEST/$d/trajectories" ]] \
    || die "$d has no proposed_qa.jsonl + trajectories/"
  dirs+=("$d")
done

declare -A PLAN
servers=($SERVERS)
for i in "${!dirs[@]}"; do
  s=${servers[$(( i % ${#servers[@]} ))]}
  PLAN[$s]+=" ${dirs[$i]}"
done
active=()
for s in "${servers[@]}"; do [[ -n "${PLAN[$s]:-}" ]] && active+=("$s"); done

ngpu=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
need=$(( TP * ${#active[@]} ))
[[ "$ngpu" -ge "$need" ]] || die "${active[*]} needs $need GPUs at TP=$TP but nvidia-smi sees $ngpu"
mkdir -p "$LOGS"
for s in "${active[@]}"; do say "server $s:${PLAN[$s]}"; done

# ------------------------------------------------------------------ serving ---
declare -A PORT GPUS PIDS
PORT[a]=8000; GPUS[a]="0,1,2,3"
PORT[b]=8100; GPUS[b]="4,5,6,7"

start_server() {
  local r=$1 p=${PORT[$1]} g=${GPUS[$1]}
  say "server $r: serving $MODEL on GPUs $g, port $p (TP=$TP, ctx=$CTX)"
  (
    source "$VENV_VLLM/bin/activate"
    export CUDA_VISIBLE_DEVICES="$g"
    exec vllm serve "$MODEL" \
      --served-model-name "$SERVED" \
      --host 127.0.0.1 --port "$p" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$CTX" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --no-async-scheduling --trust-remote-code
  ) >"$LOGS/${TAG}_${r}_vllm.log" 2>&1 &
  PIDS[$r]=$!
}

cleanup() {
  say "shutting servers down"
  for r in "${active[@]}"; do kill "${PIDS[$r]:-}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for r in "${active[@]}"; do start_server "$r"; done

for r in "${active[@]}"; do
  p=${PORT[$r]}
  say "server $r: waiting for :$p (log: $LOGS/${TAG}_${r}_vllm.log)"
  ok=0
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "${PIDS[$r]}" 2>/dev/null || die "server $r: vllm died during startup — see $LOGS/${TAG}_${r}_vllm.log"
    sleep 10
  done
  [[ "$ok" -eq 1 ]] || die "server $r: :$p not healthy after 40 min"
  say "server $r: ready"
done

# ---------------------------------------------------------------- rubrics ---
gen_pids=()
for r in "${active[@]}"; do
  (
    cd "$SRC"
    source "$VENV_GEN/bin/activate"
    export CRITERIA_MODEL_NAME="hosted_vllm/$SERVED"
    export API_BASE="http://127.0.0.1:${PORT[$r]}/v1"
    export API_KEY=EMPTY
    for RUN_DIR in ${PLAN[$r]}; do
      OUT="$QUEST/$RUN_DIR/criteria.jsonl"
      [[ "$LIMIT" -gt 0 ]] && OUT="$QUEST/$RUN_DIR/criteria_sample${LIMIT}.jsonl"
      echo "########## $RUN_DIR -> $OUT ##########"
      python longform_rubric/generate_criteria_evidence.py \
          --input_file  "$QUEST/$RUN_DIR/proposed_qa.jsonl" \
          --traj_dir    "$QUEST/$RUN_DIR/trajectories" \
          --output_file "$OUT" \
          --limit "$LIMIT" || echo "FAILED: $RUN_DIR" >&2
    done
  ) >"$LOGS/${TAG}_${r}_criteria.log" 2>&1 &
  gen_pids+=($!)
done

say "follow with: tail -f $LOGS/${TAG}_*_criteria.log"
for pid in "${gen_pids[@]}"; do wait "$pid" || say "a criteria process exited non-zero"; done

say "DONE"
for d in "${dirs[@]}"; do
  f="$QUEST/$d/criteria.jsonl"
  [[ "$LIMIT" -gt 0 ]] && f="$QUEST/$d/criteria_sample${LIMIT}.jsonl"
  printf "  %6s rubrics  %s\n" "$(wc -l < "$f" 2>/dev/null || echo 0)" "$f"
done
