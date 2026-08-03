# Long-form (Open-Ended) Question/Rubric Generation — Experiment Environment

Date: 2026-08-02
Scope: `task/open_ended_task/` — task generation + rubric generation only.
Out of scope: reference-answer generation (`ref_gen/`), answer polishing, and the
`evaluation/` + `training_scripts/` trees.

## Goal

Make this pipeline runnable end-to-end on the NYU Greene-style Slurm cluster,
backed by a locally served Qwen3.5-122B-A10B, with no cloud provider credentials:

```text
run_generate_tasks_longform.sh -> extract_proposed_qa.py -> longform_rubric/generate_criteria.py
```

## Environment Facts

| Fact | Value |
| --- | --- |
| Cluster | Slurm; `h200` partition, `gh[101-134]`, 8x H200 per node |
| Login node | No GPU, no conda, system Python 3.12, `uv` 0.9.30 available |
| Model weights | `/scratch/bl4363/models/Qwen3.5-122B-A10B` (234 GB, bf16) |
| Model arch | `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe` (hybrid linear + full attention MoE) |
| Existing keys in shell | `OPENAI_API_KEY`, `SERPER_API_KEY`, `JINA_API_KEY` |

Blocking constraint discovered during exploration: the existing vLLM install at
`/scratch/bl4363/rubric/.venv` is 0.12.0 and its model registry does **not**
contain `Qwen3_5MoeForConditionalGeneration`. Serving requires a newer vLLM in a
dedicated environment.

## Architecture

### Two isolated Python environments

| Env | Path | Contents | Runs on |
| --- | --- | --- | --- |
| serving | `.venv-vllm` | vLLM version with `qwen3_5_moe` support, GPU torch | H200 compute node |
| client | `.venv` (Python 3.10) | litellm, openai, qwen-agent, requests, tqdm, json5, transformers, tiktoken, pandas, numpy | Login node |

The pipeline talks to the model purely over HTTP, so the client environment does
not need torch or vLLM. Keeping them separate avoids a multi-gigabyte install on
the side that only issues API calls, and lets the serving env track vLLM's own
torch pin without disturbing the pipeline.

`requirements.txt` is deliberately not used as-is: it pins `torch==2.10.0`,
`vllm==0.19.0`, and `sandbox_fusion==0.3.7`, which serve the `inference/` and
`evaluation/` trees, not task generation.

### vLLM serving job

`serving/serve_qwen35.sbatch`:

- `--partition=h200 --gres=gpu:h200:2`, tensor parallel 2
- `--served-model-name qwen3.5-122b`
- On startup, writes `http://<hostname>:8000/v1` to `serving/endpoint.txt` so the
  client side never needs the compute node name hand-copied.

Memory budget: 234 GB of weights against 2 x 141 GB = 282 GB. That leaves roughly
25 GB for KV cache and activations, so `--max-model-len` starts at 65536. If the
job OOMs, the fallback is `--gres=gpu:h200:4` with tensor parallel 4.

### Configuration layer

`env.local.sh` at the repo root (gitignored) is the single place holding
credentials and the endpoint. Each run script sources it. It maps the variables
that already exist in the user's shell onto the names the repo expects:

| Shell has | Repo wants |
| --- | --- |
| `SERPER_API_KEY` | `SERPER_KEY_ID` |
| `JINA_API_KEY` | `JINA_API_KEYS` (plural) |

It also points `DEEPRESEARCH_MODEL_NAME`, `SUMMARY_MODEL_NAME`, and the criteria
model at the local endpoint.

One subtlety: `tool_visit.py:462` reads `SUMMARY_AZURE_API_BASE or API_BASE`, and
setting `SUMMARY_AZURE_API_BASE` sends it down the `AzureOpenAI` branch at
`tool_visit.py:475`. The local endpoint must therefore be set as `API_BASE`, not
`SUMMARY_AZURE_API_BASE`.

### Code changes

Three targeted edits. No unrelated refactoring.

1. `run_generate_tasks_longform.sh:8-9` — replace `conda activate deepresearch`
   with sourcing `.venv/bin/activate`, and source `env.local.sh`.
2. `longform_rubric/generate_criteria.py:14-17` — module-level hardcoded Azure
   keys and `model="azure/gpt-5"` become environment reads
   (`CRITERIA_MODEL_NAME`, `API_BASE`, `API_KEY`), with the previous values kept
   as defaults so existing Azure users are unaffected.
3. `generate_longform_tasks.py:381,384` — `num_iterations = 10` and `workers = 1`
   are hardcoded inside `main()` with no CLI surface. They become
   `NUM_ITERATIONS` / `WORKERS` environment reads with the same defaults.
   Without this, a smoke test requires editing source.

### Smoke test

`smoke_test.sh` runs `NUM_ITERATIONS=2 WORKERS=2` through all three stages and
asserts each output file exists and is non-empty. This validates the chain before
committing to a full-scale run.

## Success Criteria

1. `serving/serve_qwen35.sbatch` brings up a healthy vLLM server and
   `serving/endpoint.txt` contains a reachable URL.
2. `smoke_test.sh` completes and produces a non-empty `proposed_qa.jsonl` and a
   non-empty `criteria.jsonl`.
3. No cloud provider credentials are required anywhere in the path.

## Risks

| Risk | Status |
| --- | --- |
| No released vLLM supports `qwen3_5_moe` | **Resolved.** vLLM 0.26.0 registers `Qwen3_5MoeForConditionalGeneration`; verified in `.venv-vllm`. |
| 234 GB weights do not fit 2x H200 with usable KV cache | **Open until first run.** Starts at `--max-model-len 65536`; fall back to tensor parallel 4. |
| Qwen3.5 tool-calling format differs from what `qwen-agent` emits | **Open until first run.** Surfaced by the smoke test: the agent parses tool calls from raw text, so a format mismatch shows up as an empty `proposed_qa.jsonl`. |

## Findings During Implementation

Two problems that the original design did not anticipate, both fixed:

1. **LiteLLM's `vllm/` prefix is not an HTTP client.** `get_llm_provider("vllm/x")`
   resolves to provider `vllm`, whose handler
   (`litellm/llms/vllm/completion/handler.py`) does `from vllm import LLM` and
   loads weights in-process. The OpenAI-compatible prefix is `hosted_vllm/`.
   `generation_agent_longform.py`'s "Local vLLM (OpenAI compatible)" branch would
   therefore have failed on import in the client env. It now rewrites the prefix.
2. **`boto3` is an undeclared hard dependency.** `generation_agent_longform.py:21`
   imports it at module level, so the module cannot be imported without it even
   on a vLLM-only run. Not present in `requirements.txt`.

Additionally, `count_tokens`'s fallback path called
`AutoTokenizer.from_pretrained(self.llm_local_path)`, where `llm_local_path` is
the LiteLLM model name rather than a filesystem path, and used
`return_tensors="pt"`, which requires torch. Both are now handled via
`TOKENIZER_PATH`.

Slurm on this cluster rejects the default account; the job specifies
`--account=torch_pr_520_tandon_advanced --partition=h200_tandon`, matching the
convention in the user's other scripts. `sbatch --test-only` allocates.

## Verification Performed

All three LLM call sites were exercised against a mock OpenAI-compatible server
before consuming any GPU time, confirming each routes to the local endpoint with
the correct model name and no provider-specific parameters leaking:

| Stage | Call site | Result |
| --- | --- | --- |
| Rubric | `generate_criteria.AIClient.generate` (LiteLLM `hosted_vllm/`) | passes, `reasoning_effort` correctly omitted |
| Visit summary | `tool_visit.Visit.call_server` (raw `OpenAI` client) | passes, uses the non-Azure branch |
| Main generation | agent `vllm/` → `hosted_vllm/` rewrite | passes |
