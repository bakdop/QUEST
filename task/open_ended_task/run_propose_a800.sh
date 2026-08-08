#!/usr/bin/env bash
# Two genuinely parallel question-generation runs on one 8x A800 (80GB) box.
#
# The SLURM cluster caps each user at 2 GPUs, and Qwen3.5-122B-A10B is 234GB so
# it needs TP=4 — one server saturates the quota and a "parallel" pair of jobs
# just serialises. Eight A800s fit two TP=4 servers side by side, so the two runs
# actually overlap:
#
#   run A   GPUs 0-3   :8000   -> outputs/<TAG>_a/
#   run B   GPUs 4-7   :8100   -> outputs/<TAG>_b/
#
# Usage (from anywhere, on the A800 box):
#   export SERPER_API_KEY=...  JINA_API_KEY=...
#   bash run_propose_a800.sh
#
#   N=512 WORKERS=16 bash run_propose_a800.sh        # per run
#   RUNS=a           bash run_propose_a800.sh        # only the first half
#   FIXED_PAIRS_FILE=fixtures/pairs_findings8.jsonl N=8 bash run_propose_a800.sh
#
# Trajectories are written one per completed question, so killing this at any
# point keeps everything finished so far; re-run extract_propose.py over the
# directory to pick them up.
set -uo pipefail

# ===================== EDIT THESE IF YOUR PATHS DIFFER =====================
QUEST="${QUEST:-/ssd/boyan/QUEST}"
MODEL="${MODEL:-/ssd/boyan/models/Qwen3.5-122B-A10B}"
VENV_VLLM="${VENV_VLLM:-$QUEST/.venv-vllm}"
VENV_GEN="${VENV_GEN:-$QUEST/.venv}"
# ===========================================================================

N="${N:-512}"                 # questions per run
WORKERS="${WORKERS:-16}"      # in-flight trajectories per run
TP="${TP:-4}"                 # 234GB bf16 across 4x80GB, ~86GB left for KV
CTX="${CTX:-65536}"           # what deploy/eval_drb_a800.sh uses for this model
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
SERVED="${SERVED:-qwen3.5-122b}"
RUNS="${RUNS:-a b}"
TAG="${TAG:-propose$(date +%m%d_%H%M)}"
FIXED_PAIRS_FILE="${FIXED_PAIRS_FILE:-}"

SRC="$QUEST/task/open_ended_task"
LOGS="$QUEST/logs"

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

# --------------------------------------------------------------- preflight ---
say "checking prerequisites"
[[ -d "$QUEST" ]]                  || die "QUEST repo not at $QUEST (set QUEST=...)"
[[ -d "$SRC" ]]                    || die "no $SRC — is this the right repo?"
[[ -f "$SRC/generation_prompt_propose.py" ]] || die "generation_prompt_propose.py missing — pull the branch that has it"
[[ -f "$SRC/extract_propose.py" ]] || die "extract_propose.py missing"
[[ -d "$MODEL" ]]                  || die "model not at $MODEL (set MODEL=...)"
[[ -f "$VENV_VLLM/bin/activate" ]] || die "no vllm venv at $VENV_VLLM (set VENV_VLLM=...)"
[[ -f "$VENV_GEN/bin/activate" ]]  || die "no generation venv at $VENV_GEN (set VENV_GEN=...)"
[[ -n "${SERPER_API_KEY:-}" ]]     || die "export SERPER_API_KEY=... (the search tool needs it)"
[[ -n "${JINA_API_KEY:-}" ]]       || die "export JINA_API_KEY=... (the visit tool needs it)"

ngpu=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
need=$(( TP * $(echo "$RUNS" | wc -w) ))
[[ "$ngpu" -ge "$need" ]] || die "$RUNS needs $need GPUs at TP=$TP but nvidia-smi sees $ngpu"
say "$ngpu GPUs visible, need $need — ok"
mkdir -p "$LOGS"

for host in google.serper.dev r.jina.ai; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 "https://$host/" || echo 000)
  [[ "$code" != "000" ]] || die "no outbound route to $host"
  say "  $host -> $code"
done

# ------------------------------------------------------------------ serving ---
declare -A PORT GPUS PIDS
PORT[a]=8000; GPUS[a]="0,1,2,3"
PORT[b]=8100; GPUS[b]="4,5,6,7"

start_server() {
  local r=$1 p=${PORT[$1]} g=${GPUS[$1]}
  say "run $r: serving $MODEL on GPUs $g, port $p (TP=$TP, ctx=$CTX)"
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
  for r in $RUNS; do kill "${PIDS[$r]:-}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for r in $RUNS; do start_server "$r"; done

for r in $RUNS; do
  p=${PORT[$r]}
  say "run $r: waiting for :$p to load (log: $LOGS/${TAG}_${r}_vllm.log)"
  ok=0
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "${PIDS[$r]}" 2>/dev/null || die "run $r: vllm died during startup — see $LOGS/${TAG}_${r}_vllm.log"
    sleep 10
  done
  [[ "$ok" -eq 1 ]] || die "run $r: :$p not healthy after 40 min"
  say "run $r: ready"
done

# --------------------------------------------------------------- generation ---
gen_pids=()
for r in $RUNS; do
  p=${PORT[$r]}
  out="$QUEST/outputs/${TAG}_${r}"
  mkdir -p "$out/trajectories" "$QUEST/database"
  say "run $r: generating $N questions with $WORKERS workers -> $out"
  (
    cd "$SRC"
    source "$VENV_GEN/bin/activate"
    # Set before sourcing env.local.sh: every var there uses :- defaults, so
    # anything exported here wins. The two runs must not share an endpoint, a
    # trajectory dir, or a cache file.
    export LLM_API_BASE="http://127.0.0.1:$p/v1"
    export SERVED_MODEL_NAME="$SERVED"
    export TOKENIZER_PATH="$MODEL" MODEL_PATH="$MODEL"
    export TRAJ_DIR="$out/trajectories"
    export VISIT_CACHE_FILE="$QUEST/database/visit_${r}.db"
    export SEARCH_CACHE_FILE="$QUEST/database/search_${r}.db"
    export NUM_ITERATIONS="$N" WORKERS="$WORKERS"
    export PROMPT_VARIANT=propose
    export FIXED_PAIRS_FILE="$FIXED_PAIRS_FILE"
    source "$QUEST/env.local.sh"
    python generate_longform_tasks.py
  ) >"$LOGS/${TAG}_${r}_gen.log" 2>&1 &
  gen_pids+=($!)
done

say "both runs going; follow with: tail -f $LOGS/${TAG}_*_gen.log"
say "progress:  ls $QUEST/outputs/${TAG}_*/trajectories | wc -l"
for pid in "${gen_pids[@]}"; do wait "$pid" || say "a generation process exited non-zero; extracting what completed"; done

# ------------------------------------------------------------------ extract ---
for r in $RUNS; do
  out="$QUEST/outputs/${TAG}_${r}"
  n=$(find "$out/trajectories" -name '*.json' 2>/dev/null | wc -l)
  say "run $r: $n trajectories"
  [[ "$n" -gt 0 ]] || { say "run $r: nothing to extract"; continue; }
  ( cd "$SRC"; source "$VENV_GEN/bin/activate"
    python extract_propose.py --input_dir "$out/trajectories" \
                              --output_file "$out/proposed_qa.jsonl" ) | tee "$LOGS/${TAG}_${r}_qc.log"
done

say "DONE"
for r in $RUNS; do
  f="$QUEST/outputs/${TAG}_${r}/proposed_qa.jsonl"
  [[ -f "$f" ]] && echo "  $f  ($(wc -l < "$f") questions)"
done
