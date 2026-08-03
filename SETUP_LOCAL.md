# Local Setup — Long-form Question / Rubric Generation

Runs `task/open_ended_task/` against a locally served Qwen3.5-122B-A10B. No cloud
provider credentials required.

## What is here

| Path | Purpose |
| --- | --- |
| `.venv` | Client env (Python 3.10). Pipeline code, HTTP only — no torch/vLLM. |
| `.venv-vllm` | Serving env (Python 3.12, vLLM 0.26.0, torch 2.11+cu130). |
| `env.local.sh` | All credentials, the live endpoint, and run scale. Gitignored. |
| `serving/serve_qwen35.sbatch` | Slurm job that serves the model and publishes its URL. |
| `task/open_ended_task/smoke_test.sh` | Tiny end-to-end run across all three stages. |

## Running it

```bash
# 1. Bring the model up (2x H200, tensor parallel 2).
sbatch serving/serve_qwen35.sbatch
squeue -u $USER

# 2. Wait for the endpoint to be published. This appears only after /health passes.
cat serving/endpoint.txt        # e.g. http://gh133.hpc.nyu.edu:8000/v1

# 3. Validate the whole chain on 2 tasks before scaling up.
bash task/open_ended_task/smoke_test.sh

# 4. Full run.
bash task/open_ended_task/run_generate_tasks_longform.sh
python task/open_ended_task/extract_proposed_qa.py \
  --input_dir outputs/openended_trajectories \
  --output_file outputs/proposed_qa.jsonl
python task/open_ended_task/longform_rubric/generate_criteria.py \
  --input_file outputs/proposed_qa.jsonl \
  --output_file outputs/criteria.jsonl
```

Scale knobs live in `env.local.sh`: `NUM_ITERATIONS` (tasks to generate),
`WORKERS` (generation concurrency), `CRITERIA_MAX_WORKERS` (rubric concurrency).

## Serving notes

234 GB of bf16 weights against 2 x 141 GB leaves roughly 25 GB for KV cache, so
`MAX_LEN` defaults to 65536. If the job OOMs:

```bash
# Either shorten the context...
MAX_LEN=32768 sbatch serving/serve_qwen35.sbatch
# ...or take four GPUs.
sed -i 's/gpu:h200:2/gpu:h200:4/' serving/serve_qwen35.sbatch
TP_SIZE=4 sbatch serving/serve_qwen35.sbatch
```

## Why the environments are split

`requirements.txt` pins `torch`, `vllm`, and `sandbox_fusion` for the
`inference/` and `evaluation/` trees. Task generation only issues HTTP calls, so
`.venv` omits them; that keeps the client light and lets the serving env track
vLLM's own torch pin independently.

## Modifications to upstream code

Five targeted edits, all backward compatible — existing Azure/Bedrock configs
behave exactly as before.

| File | Change | Why |
| --- | --- | --- |
| `run_generate_tasks_longform.sh` | Prefer `.venv`, fall back to conda; source `env.local.sh` | No conda on this cluster |
| `generate_longform_tasks.py` | `NUM_ITERATIONS` / `WORKERS` env reads | Were hardcoded in `main()`; a smoke test otherwise needs a source edit |
| `longform_rubric/generate_criteria.py` | Env-driven model + credentials; `reasoning_effort` opt-out; env-driven worker/limit | Azure keys were hardcoded at module level; local servers reject `reasoning_effort` |
| `generation_agent_longform.py` | Rewrite `vllm/` to `hosted_vllm/` | LiteLLM routes a bare `vllm/` prefix to its **offline** backend, which imports the vllm package and loads weights in-process — it is not an HTTP client |
| `generation_agent_longform.py` | `TOKENIZER_PATH` override in `count_tokens`, drop `return_tensors="pt"` | The fallback loaded a tokenizer from the LiteLLM model name, which is not a path, and required torch |

`boto3` is also required but missing from `requirements.txt`:
`generation_agent_longform.py:21` imports it unconditionally, so the module fails
to import even on a vLLM-only run. It is installed in `.venv`.
