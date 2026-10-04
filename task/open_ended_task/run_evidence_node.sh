#!/usr/bin/env bash
# Long-form question generation (PROMPT_VARIANT=evidence_first) on a single 8-GPU
# node without Slurm. Two independent runs side by side, one TP=4 server each:
#
#   run a   GPUs 0-3   :8000   -> outputs/<TAG>_a/
#   run b   GPUs 4-7   :8100   -> outputs/<TAG>_b/
#
# Usage (from anywhere):
#   export SERPER_API_KEY=... JINA_API_KEY=...
#   QUEST=/path/to/QUEST MODEL=/path/to/Qwen3.5-122B-A10B bash run_evidence_node.sh
#
#   N=512 WORKERS=16 ...      # questions / in-flight trajectories per run
#   RUNS=a ...                # only the first run (4 GPUs)
#   PROXY=http://host:port    # if outbound traffic needs an HTTP proxy
#
# Trajectories are written one per completed question, so killing this keeps
# everything finished so far; re-run extract_evidence.py + fix_leaks.py over the
# directory to pick them up.
set -uo pipefail

QUEST="${QUEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL="${MODEL:?set MODEL to the Qwen3.5-122B-A10B weights}"
VENV_VLLM="${VENV_VLLM:-$QUEST/.venv-vllm}"
VENV_GEN="${VENV_GEN:-$QUEST/.venv}"

N="${N:-512}"                 # questions per run
WORKERS="${WORKERS:-16}"      # in-flight trajectories per run
TP="${TP:-4}"                 # 234GB bf16 across 4x80GB
CTX="${CTX:-65536}"
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
SERVED="${SERVED:-qwen3.5-122b}"
RUNS="${RUNS:-a b}"
TAG="${TAG:-ev$(date +%m%d_%H%M)}"

SRC="$QUEST/task/open_ended_task"
LOGS="$QUEST/logs"

# vLLM is served on 127.0.0.1, so no_proxy is required whenever a proxy is set.
PROXY="${PROXY:-}"
if [[ -n "$PROXY" ]]; then
  export no_proxy="${no_proxy:-127.0.0.1,localhost}" NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
  export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
fi

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

# --------------------------------------------------------------- preflight ---
[[ -d "$SRC" ]]                    || die "no $SRC (set QUEST=...)"
[[ -d "$MODEL" ]]                  || die "model not at $MODEL"
[[ -f "$VENV_VLLM/bin/activate" ]] || die "no vllm venv at $VENV_VLLM (set VENV_VLLM=...)"
[[ -f "$VENV_GEN/bin/activate" ]]  || die "no generation venv at $VENV_GEN (set VENV_GEN=...)"
[[ -n "${SERPER_API_KEY:-}" ]]     || die "export SERPER_API_KEY=... (search tool)"
[[ -n "${JINA_API_KEY:-}" ]]       || die "export JINA_API_KEY=... (visit tool)"

ngpu=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
need=$(( TP * $(echo "$RUNS" | wc -w) ))
[[ "$ngpu" -ge "$need" ]] || die "RUNS='$RUNS' needs $need GPUs at TP=$TP but nvidia-smi sees $ngpu"
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
  say "run $r: waiting for :$p (log: $LOGS/${TAG}_${r}_vllm.log)"
  ok=0
  for i in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "${PIDS[$r]}" 2>/dev/null || die "run $r: vllm died during startup — see $LOGS/${TAG}_${r}_vllm.log"
    sleep 10
  done
  [[ "$ok" -eq 1 ]] || die "run $r: :$p not healthy after 40 min"
  say "run $r: ready"
done

# The generator reads its endpoint, model names, tool keys and caches from env.
gen_env() {
  local r=$1
  export LLM_API_BASE="http://127.0.0.1:${PORT[$r]}/v1"
  export SERVED_MODEL_NAME="$SERVED" TOKENIZER_PATH="$MODEL" MODEL_PATH="$MODEL"
  export SERPER_KEY_ID="${SERPER_KEY_ID:-$SERPER_API_KEY}"
  export JINA_API_KEYS="${JINA_API_KEYS:-$JINA_API_KEY}"
  export DEEPRESEARCH_MODEL_NAME="vllm/$SERVED"
  export DEEPRESEARCH_API_BASE="$LLM_API_BASE" DEEPRESEARCH_OPENAI_API_KEY=EMPTY
  export SUMMARY_MODEL_NAME="$SERVED" SUMMARY_OPENAI_API_KEY=EMPTY
  export API_BASE="$LLM_API_BASE" API_KEY=EMPTY
  export PROMPT_VARIANT=evidence_first
}

# --------------------------------------------------------------- generation ---
gen_pids=()
for r in $RUNS; do
  out="$QUEST/outputs/${TAG}_${r}"
  mkdir -p "$out/trajectories" "$QUEST/database"
  say "run $r: generating $N questions with $WORKERS workers -> $out"
  (
    cd "$SRC"
    source "$VENV_GEN/bin/activate"
    gen_env "$r"
    export TRAJ_DIR="$out/trajectories" SAVE_TRAJ=true
    export VISIT_CACHE_FILE="$QUEST/database/visit_${r}.db"
    export SEARCH_CACHE_FILE="$QUEST/database/search_${r}.db"
    export VISIT_CACHE_ENABLED=true VISIT_CACHE_RESUME=true
    export SEARCH_CACHE_ENABLED=true SEARCH_CACHE_RESUME=true
    export NUM_ITERATIONS="$N" WORKERS="$WORKERS"
    python generate_longform_tasks.py
  ) >"$LOGS/${TAG}_${r}_gen.log" 2>&1 &
  gen_pids+=($!)
done

say "follow with: tail -f $LOGS/${TAG}_*_gen.log"
say "progress:    ls $QUEST/outputs/${TAG}_*/trajectories | wc -l"
for pid in "${gen_pids[@]}"; do wait "$pid" || say "a generation process exited non-zero; extracting what completed"; done

# ------------------------------------------------- extract + fix leaked figures ---
for r in $RUNS; do
  out="$QUEST/outputs/${TAG}_${r}"
  n=$(find "$out/trajectories" -name '*.json' 2>/dev/null | wc -l)
  say "run $r: $n trajectories"
  [[ "$n" -gt 0 ]] || { say "run $r: nothing to extract"; continue; }
  ( cd "$SRC"; source "$VENV_GEN/bin/activate"; gen_env "$r"
    python extract_evidence.py --input_dir "$out/trajectories" \
                               --output_file "$out/proposed_qa.jsonl"
    # needs the server, so it runs before cleanup
    python fix_leaks.py --input "$out/proposed_qa.jsonl" \
                        --output "$out/proposed_qa.jsonl" --workers "$WORKERS"
  ) | tee "$LOGS/${TAG}_${r}_qc.log"
done

say "DONE"
for r in $RUNS; do
  f="$QUEST/outputs/${TAG}_${r}/proposed_qa.jsonl"
  [[ -f "$f" ]] && echo "  $f  ($(wc -l < "$f") questions)"
done
