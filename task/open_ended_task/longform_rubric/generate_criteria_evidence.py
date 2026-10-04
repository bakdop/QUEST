"""Rubric generation from an evidence_first run, grounded in that run's own corpus.

    python longform_rubric/generate_criteria_evidence.py \
        --input_file  outputs/<run>/proposed_qa.jsonl \
        --traj_dir    outputs/<run>/trajectories \
        --output_file outputs/<run>/criteria.jsonl

Input is the `proposed_qa.jsonl` written by extract_evidence.py (question,
subtopics, statements, research path) plus the run's own trajectories, whose tool
responses are the corpus every rubric item must be grounded in.

Three things decide the shape of this one.

POSITIVE ONLY. No penalties, no negative weights. RubricHub (arXiv 2601.08430)
ablates exactly this: "positive-only weights consistently outperform those with
negative penalties", HealthBench 66.2 vs 63.2 and LLMEval-Med 75.3 vs 74.2, which
they attribute to "the grader's low accuracy on negative criteria, which hinders
optimization". Their limitations section repeats it — incorporating pitfalls adds
noise that degrades RL. QUEST's own older generator never had negative weights
either.

GROUNDED IN THE CORPUS, NOT IN THE STATEMENTS. Checking the five runs made under
this prompt, only 19 of 39 statement quotes appear verbatim in the run's own tool
responses after canonicalising quotes, dashes and whitespace; another 10 share a
contiguous run of at least 60% of their words, and the remaining quarter are
reconstructed, the worst at 0.13. Every url is present. So the model searched and
then paraphrased while writing the statement down. tool_visit.py also summarises a
page before it enters context, so "verbatim from a tool response" already means
verbatim from a summary. The corpus is therefore passed in whole and every item
must point at the span in it that supports the item.

THE META-RULES GO IN THE PROMPT. RubricHub's Appendix A is a rubric for rubrics —
Consistency, Stability, Alignment; Coverage, Criteria Num. 3-25, Independence,
Atomicity; Clarity, Conciseness, Language Consistency; Distinguishability, Weight
Rationality, Verifiability. Running it as a second scoring pass would pay twice
for something the generator can be told once. Distinguishability is the one that
matters most here and it is stated as a precondition for writing an item at all:
an item that a competent report could skip without being worse is the item that
does the most damage as a reward.

Kept from the ResearchRubrics register: the writing rules (standalone,
acceptance sets, one fact per item, no prompt echo, operational soft axes).
`why_it_matters` carries the argument for an item, never the content being graded.
"""
import argparse
import json
import os
import re
import threading
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import litellm

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# The five axes: QUEST's four evaluation dimensions plus Implicit Criteria. Synthesis
# is not its own axis -- reconciling sources sits inside Insight -- and References &
# Citation Quality is not used.
AXES = [
    "Comprehensiveness",
    "Insight",
    "Instruction Following",
    "Readability",
    "Implicit Criteria",
]

MODEL = os.environ.get("CRITERIA_MODEL_NAME", "azure/gpt-5")
KW = {}
if os.environ.get("API_BASE"):
    KW["api_base"] = os.environ["API_BASE"]
if os.environ.get("API_KEY"):
    KW["api_key"] = os.environ["API_KEY"]
_re = os.environ.get("CRITERIA_REASONING_EFFORT", "none")
if _re.lower() != "none":
    KW["reasoning_effort"] = _re

MAX_WORKERS = int(os.environ.get("CRITERIA_MAX_WORKERS", 4))
MAX_TOKENS = int(os.environ.get("CRITERIA_MAX_TOKENS", 16000))
# A run's tool responses run to ~19k tokens at the median. The cap is a guard on
# the tail, not a budget: cutting the corpus costs items that could have been
# grounded in it, which is the one thing this generator cannot get anywhere else.
CORPUS_CHARS = int(os.environ.get("CRITERIA_CORPUS_CHARS", 160_000))
RETRIES = 3

_print_lock = threading.Lock()


# ------------------------------------------------------------------- corpus ---
def canon(s):
    s = unicodedata.normalize("NFKC", s or "")
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), ("…", "..."), ("\xa0", " ")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def corpus_of(traj):
    """Everything the tools returned, in the order the run saw it.

    The user turns after the first are tool results; the first is the task. The
    assistant turns are the model's own reasoning and are left out — grounding an
    item in something the proposer said about a page, rather than in the page,
    would defeat the point.
    """
    msgs = [m for m in (traj.get("messages") or []) if m.get("role") == "user"]
    return [canon(str(m.get("content", ""))) for m in msgs[1:]]


def render_corpus(chunks):
    out, used = [], 0
    for i, c in enumerate(chunks, 1):
        if used >= CORPUS_CHARS:
            out.append(f"[{len(chunks) - i + 1} further tool responses omitted]")
            break
        room = CORPUS_CHARS - used
        body = c if len(c) <= room else c[:room] + " […truncated]"
        used += len(body)
        out.append(f"--- tool response {i} ---\n{body}")
    return "\n\n".join(out) or "(no tool responses recorded)"


# -------------------------------------------------------------- run material ---
def render_subtopics(subs):
    out = []
    for s in subs:
        if not isinstance(s, dict):
            continue
        out.append(
            f"- [{s.get('exposure', '?')}] {s.get('handle', '')}\n"
            f"    the section has to answer: {s.get('query', '')}\n"
            f"    what the evidence showed: {s.get('what_it_shows', '')}"
        )
    return "\n".join(out) or "(none recorded)"


def render_statements(stmts):
    out = []
    for s in stmts:
        if not isinstance(s, dict):
            continue
        line = [f"{s.get('id', '?')} [{s.get('subtopic', '')}] {s.get('claim', '')}"]
        for e in (s.get("evidence") or [])[:3]:
            if isinstance(e, dict):
                line.append(f"      {e.get('source', '')}  \"{(e.get('quote') or '')[:200]}\"")
        out.append("\n".join(line))
    return "\n".join(out) or "(none recorded)"


def render_path(path):
    out = []
    for i, s in enumerate(path or [], 1):
        if not isinstance(s, dict):
            continue
        qs = ", ".join(str(q) for q in (s.get("queries") or []))
        out.append(
            f"step {i}: {qs}\n"
            f"    needed first: {s.get('from') or '(nothing — answerable from the question)'}\n"
            f"    produced: {s.get('yields') or []}\n"
            f"    the gap it closed: {s.get('why', '')}"
        )
    return "\n".join(out) or "(none recorded)"


from prompt.criteria_prompt_evidence import CRITERIA_PROMPT_EVIDENCE as PROMPT



def parse_items(text):
    """Thinking models mention the tag names while reasoning, so drop the reasoning
    block first and take the last tagged span rather than the first."""
    t = text.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"<json_output>(.*?)</json_output>", t, re.DOTALL | re.IGNORECASE)
    raw = blocks[-1].strip() if blocks else t.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start = raw.find("[")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        items = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    keep = []
    for i in items:
        if not isinstance(i, dict) or not i.get("criterion") or i.get("axis") not in AXES:
            continue
        w = i.get("weight")
        if not isinstance(w, (int, float)):
            continue
        # A negative weight is not a formatting slip, it is the penalty item this
        # generator exists to not produce. Drop it rather than flip its sign.
        if w < 0:
            continue
        g = i.get("grounding")
        if isinstance(g, str):
            g = [g]
        if not isinstance(g, list) or not [x for x in g if str(x).strip()]:
            continue
        i["grounding"] = [str(x).strip() for x in g if str(x).strip()]
        keep.append(i)
    return keep or None


def one(item, traj_dir, lo, hi):
    traj_name = item.get("trajectory")
    chunks = []
    if traj_name:
        p = os.path.join(traj_dir, traj_name)
        if os.path.exists(p):
            try:
                chunks = corpus_of(json.load(open(p)))
            except Exception:
                chunks = []
    prompt = PROMPT.format(
        question=item["prompt"],
        subtopics=render_subtopics(item.get("subtopics") or []),
        statements=render_statements(item.get("statements") or []),
        path=render_path(item.get("research_path")),
        corpus=render_corpus(chunks),
        lo=lo, hi=hi,
    )
    for attempt in range(RETRIES):
        try:
            r = litellm.completion(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                   max_tokens=MAX_TOKENS, **KW)
            items = parse_items(r.choices[0].message.content or "")
            if items:
                return {**item, "rubrics": items,
                        "_rubric_corpus_chunks": len(chunks)}
        except Exception as e:
            if attempt == RETRIES - 1:
                with _print_lock:
                    print(f"  id={item.get('id')} failed: {type(e).__name__}: {str(e)[:120]}")
    return None


def grounding_hits(row, traj_dir):
    """Does each item's `grounding` actually point at something in that run's corpus?

    The one mechanical check on the grounding requirement. Matched on a contiguous
    run of eight words rather than on the whole reference, because a reference that
    stitches two sentences together is still pointing at something real. An item
    counts as grounded when ANY of its references resolves: an item synthesising
    across two pages has no single span to cite, and the first real one found this
    way was a markup range built from "Popcorn $1 $6.5 550%" in an SSRN table and a
    "900%" on another page.

    A reference written as the researcher's paraphrase rather than as corpus text
    fails here, and should — that paraphrase is the layer this generator is meant
    to see past.
    """
    p = os.path.join(traj_dir, row.get("trajectory") or "")
    if not os.path.exists(p):
        return None
    try:
        blob = " ".join(corpus_of(json.load(open(p)))).lower()
    except Exception:
        return None
    ids = {str(s.get("id")) for s in (row.get("statements") or []) if isinstance(s, dict)}

    def resolves(ref):
        if str(ref).strip() in ids or any(m in ids for m in re.findall(r"\bS\d+\b", str(ref))):
            return True
        w = canon(str(ref)).lower().split()
        return any(" ".join(w[i:i + 8]) in blob for i in range(max(1, len(w) - 7)))

    hit = sum(1 for it in row["rubrics"]
              if any(resolves(r) for r in (it.get("grounding") or [])))
    return hit, len(row["rubrics"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--traj_dir", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--min_items", type=int, default=20)
    ap.add_argument("--max_items", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0)
    # Sharding exists for the queue, not for the GPU. A 500-question run is ~40
    # minutes of compute but h200_tandon has been making 12-hour asks wait 4-13
    # hours and 4-hour asks wait 1, so the wall-clock win comes from splitting
    # into jobs short enough to backfill. Parallel jobs wait concurrently.
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input_file)]
    if args.limit:
        rows = rows[:args.limit]
    total = len(rows)
    if args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            ap.error(f"--shard must be in [0, {args.num_shards})")
        # Strided, not contiguous: corpus length varies several-fold between
        # questions and a contiguous block can land most of the long ones in one
        # shard, which would leave that job running long after the others exit.
        rows = rows[args.shard::args.num_shards]
    shard_note = f" | shard {args.shard}/{args.num_shards} of {total}" if args.num_shards > 1 else ""
    print(f"{len(rows)} questions | model {MODEL} "
          f"| {args.min_items}-{args.max_items} items{shard_note}")

    # Written as each rubric lands, not collected and dumped at the end. A
    # thousand-question shard is hours of GPU time and a walltime kill on a
    # collect-then-write would throw all of it away; this way the file holds
    # everything finished up to the kill, the same property run_evidence.sbatch
    # relies on for trajectories.
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    results = []
    with open(args.output_file, "w") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        jobs = ex.map(lambda it: one(it, args.traj_dir, args.min_items, args.max_items), rows)
        for i, r in enumerate(jobs, 1):
            if r:
                results.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            print(f"[{i}/{len(rows)}] {'ok' if r else 'FAILED'}", flush=True)

    if not results:
        print("nothing written")
        return
    items = [i for r in results for i in r["rubrics"]]
    print(f"\nwrote {len(results)}/{len(rows)} rubrics, {len(items)} items")
    counts = sorted(len(r["rubrics"]) for r in results)
    print(f"  items per rubric   median {counts[len(counts)//2]}  "
          f"min {counts[0]}  max {counts[-1]}   (ResearchRubrics median 24)")
    c = Counter(i["axis"] for i in items)
    RR = {"Implicit Criteria": 39.4, "Explicit Criteria": 27.7, "Synthesis of Information": 15.8,
          "Communication Quality": 7.8, "Instruction Following": 5.8,
          "References & Citation Quality": 3.5}
    for a, v in c.most_common():
        print(f"  {a:>30}: {v:>4}  {100*v/len(items):>5.1f}%   (RR {RR.get(a, 0):.1f}%)")
    neg = sum(1 for i in items if i["weight"] < 0)
    dig = sum(1 for i in items if re.search(r"\d", i["criterion"]))
    print(f"  negative weights   {neg}   (must be 0)")
    print(f"  items with a digit {dig}  {100*dig/len(items):.1f}%   (ResearchRubrics 29.3%)")
    print(f"  weights            {dict(sorted(Counter(i['weight'] for i in items).items()))}")

    tot_hit = tot_n = 0
    for r in results:
        gh = grounding_hits(r, args.traj_dir)
        if gh:
            tot_hit += gh[0]
            tot_n += gh[1]
    if tot_n:
        print(f"  grounding found in the corpus  {tot_hit}/{tot_n}  {100*tot_hit/tot_n:.0f}%")


if __name__ == "__main__":
    main()
