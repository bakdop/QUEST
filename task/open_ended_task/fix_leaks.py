"""Take the answer back out of the question.

    python fix_leaks.py --input proposed_qa.jsonl --output proposed_qa.jsonl

A question that carries the figures the investigation found asks to be
transcribed rather than researched. It is also the one failure in this pipeline
that can be detected exactly — a number in the question either appears in a
statement or it does not — so it gets a mechanism instead of another sentence in
the prompt, which has been tried six times on adjacent problems and tied every
time.

Detection is the intersection of the numbers in the statements with the numbers
in the question, which is what `leak_tokens` in analyze_run.py could not do: its
regex catches currency, percentages and magnitude words, so of the three figures
in

    "…offering courses to 13 million online learners while maintaining only 38
    students in exclusive online degree programs out of 15,500 total students…"

it would find `13 million` and miss `38` and `15,500`.

Bare four-digit years are excluded. A year in a question is usually pinning the
subject — "Carrie (1988)", "the 2025 fiscal year" — and naming what you are
asking about is the opposite of giving the answer away.

The rewrite is a single call with no tools and one job: drop those tokens and
change nothing else. It is then re-checked, so success is decided by the detector
rather than by the model's say-so. A question that still leaks is marked and, with
--drop-unfixed, removed.
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from generation_agent_longform import MultiTurnReactAgent

model = os.environ.get("DEEPRESEARCH_MODEL_NAME", "openai/qwen3.5-122b")
model_type = ('bedrock' if model.startswith('bedrock/') else
              'azure' if model.startswith('azure/') else
              'vllm' if model.startswith('vllm/') else 'openai')
generate_cfg = {'max_tokens': 4000, 'max_retries': 10, 'temperature': 0.3}
if model_type != 'openai':
    generate_cfg['top_p'] = 0.95
llm_cfg = {'model': model, 'generate_cfg': generate_cfg, 'model_type': model_type}

YEAR = re.compile(r"^(?:19|20)\d{2}$")
# A number with at least two digits, optionally grouped, optionally followed by a
# magnitude word. Two digits because "3 of 8" collides with everything.
NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:%|percent|million|billion|trillion|k\b)?",
                 re.I)


def numbers(text):
    out = set()
    for m in NUM.finditer(text or ""):
        # [\d,]* eats a sentence comma, so "$12,518,415," comes back with it
        tok = m.group(0).strip().rstrip(",.")
        digits = re.sub(r"\D", "", tok)
        if len(digits) < 2 or YEAR.match(tok):
            continue
        out.add(tok)
    return out


def leaked(row):
    """Numbers the question shares with its statements."""
    q = row.get("prompt") or ""
    qn = numbers(q)
    if not qn:
        return []
    said = set()
    for s in (row.get("statements") or []):
        if not isinstance(s, dict):
            continue
        said |= numbers(s.get("claim"))
        for e in (s.get("evidence") or []):
            if isinstance(e, dict):
                said |= numbers(e.get("quote"))
    # compare on digits, so "15,500" in one and "15500" in the other still match
    dig = {re.sub(r"\D", "", t): t for t in said}
    return sorted({t for t in qn if re.sub(r"\D", "", t) in dig})


SYSTEM = """You rewrite a research question to take the answer back out of it.

You are given a question and a list of figures that appear both in it and in the
material it was built from. Those figures are the answer: whoever answers the
question is supposed to go and find them. Rewrite the question so that none of
them appears.

Change nothing else. Keep the subject, the deliverable, the period, the named
entities and the shape of the request exactly as they are — the question should
read as though it had been written this way. Where a figure was doing work in the
sentence, say what it was about rather than what it was: "13 million online
learners" becomes "its open-courseware audience", "38 students" becomes "its
online degree enrolment".

Emit the rewritten question inside <answer></answer> and nothing else.

Current date:
"""


def rewrite(agent, row, tokens):
    uc = (f"QUESTION\n{row['prompt']}\n\n"
          f"FIGURES THAT MUST NOT APPEAR\n" + "\n".join(f"  {t}" for t in tokens))
    r = agent._run({"item": {"question": row.get("topic") or "", "answer": "1"},
                    "system_prompt": SYSTEM, "user_content": uc,
                    "iteration_id": f"leak{row.get('id')}",
                    "subcategory": row.get("topic")}, model)
    text = (r.get("prediction") or {}).get("text") or ""
    hit = re.search(r"<answer>(.*?)</answer>", text, re.S)
    return hit.group(1).strip() if hit else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 8)))
    ap.add_argument("--drop-unfixed", action="store_true",
                    help="remove questions still leaking after the rewrite")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    todo = [(r, leaked(r)) for r in rows]
    dirty = [(r, t) for r, t in todo if t]
    print(f"{len(dirty)}/{len(rows)} questions carry figures from their own statements")
    if not dirty:
        if args.output != args.input:
            with open(args.output, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        return

    traj = os.path.join(os.path.dirname(os.path.abspath(args.output)), "leak_trajectories")
    os.makedirs(traj, exist_ok=True)

    def one(pair):
        row, tokens = pair
        agent = MultiTurnReactAgent(llm=llm_cfg, function_list=[])
        agent.traj_dir = traj
        try:
            new = rewrite(agent, row, tokens)
        except Exception as e:
            print(f"  #{row.get('id')}: rewrite failed ({type(e).__name__}: {e})")
            return
        if not new:
            print(f"  #{row.get('id')}: no answer returned")
            return
        before = row["prompt"]
        row["prompt"] = new
        still = leaked(row)
        if still:
            row["prompt"] = before          # a partial fix is not a fix
            row.setdefault("_signals", {})["leak_unfixed"] = still
            print(f"  #{row.get('id')}: still leaks {still}")
        else:
            row.setdefault("_signals", {})["leak_rewritten"] = tokens
            print(f"  #{row.get('id')}: removed {tokens}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, dirty))

    fixed = sum(1 for r in rows if (r.get("_signals") or {}).get("leak_rewritten"))
    unfixed = [r for r in rows if (r.get("_signals") or {}).get("leak_unfixed")]
    if args.drop_unfixed and unfixed:
        keep = {id(r) for r in unfixed}
        rows = [r for r in rows if id(r) not in keep]

    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"\nrewritten {fixed}/{len(dirty)}; {len(unfixed)} still leaking"
          f"{' (dropped)' if args.drop_unfixed else ' (kept, marked _signals.leak_unfixed)'}")
    print(f"-> {args.output}  ({len(rows)} questions)")


if __name__ == "__main__":
    main()
