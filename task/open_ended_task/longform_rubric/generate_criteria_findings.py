"""Rubric generation grounded in the proposer's findings, in ResearchRubrics form.

The default generator (generate_criteria.py) is handed the question text and
nothing else, so the only criteria it can write are generic writing standards -
"assesses whether the analysis has depth". Those are not checkable, and they are
not what ResearchRubrics measures: 39% of its rubric items are Implicit Criteria,
concrete things a competent answer must contain that the question never asked for
("uses boats where the Darien Gap prevents overland travel").

This generator reads the findings the proposer recorded, and turns each into a
binary-checkable item. It also mirrors ResearchRubrics' schema - a flat list of
{criterion, weight, axis} with 1-5 weights and negative weights for penalties -
and few-shots on real rubric items drawn from ResearchRubrics_data.jsonl.

  python longform_rubric/generate_criteria_findings.py \
      --input_file  outputs/<run>/proposed_qa.jsonl \
      --output_file outputs/<run>/criteria.jsonl
"""
import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import litellm

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
RETRIES = 3
RR_PATH = os.environ.get("RESEARCH_RUBRICS_PATH", "./longform_utils/ResearchRubrics_data.jsonl")

AXES = [
    "Implicit Criteria",
    "Explicit Criteria",
    "Synthesis of Information",
    "Communication Quality",
    "Instruction Following",
    "References & Citation Quality",
]

_print_lock = threading.Lock()


def load_rubric_examples(k_tasks=2, k_items=7):
    """Few-shot on real ResearchRubrics items so the output matches their register."""
    try:
        rows = [json.loads(l) for l in open(RR_PATH)]
    except OSError:
        return ""
    rows = [r for r in rows if len(r.get("rubrics", [])) >= k_items]
    if not rows:
        return ""
    out = []
    for r in random.sample(rows, k=min(k_tasks, len(rows))):
        items = random.sample(r["rubrics"], k=k_items)
        out.append(
            "Question: " + " ".join(r["prompt"].split())[:400] + "\nRubric items:\n"
            + "\n".join(
                f'  {{"criterion": {json.dumps(" ".join(i["criterion"].split()), ensure_ascii=False)}, '
                f'"weight": {i.get("weight", 3)}, "axis": {json.dumps(i.get("axis", "Implicit Criteria"))}}}'
                for i in items
            )
        )
    return "\n\n".join(out)


PROMPT = """You write grading rubrics for long-form deep-research reports, in the style of the ResearchRubrics benchmark.

A rubric item is a single statement that a grader can mark satisfied or not satisfied by reading the report. It must be concrete enough that two graders would agree. "The response analyses the topic in depth" is not a rubric item. "The response states that a 32GB DDR5-6000 kit rose from roughly $100 in October 2025 to $389" is.

AXES
  Implicit Criteria             Something a competent answer must contain that the question never asked for. Domain knowledge the asker did not know to request. This is the most important axis and should be the largest.
  Explicit Criteria             Something the question asked for directly.
  Synthesis of Information      Requires combining several sources or sections - a comparison, a reconciliation, a conclusion that no single source supports.
  Instruction Following         Format, length, scope, audience constraints stated in the question.
  Communication Quality         Structure, tables, headings, readability.
  References & Citation Quality Sourcing and attribution.

WEIGHTS
  5  essential - the report fails its purpose without this
  3  important
  1  minor
  Negative weights are penalties. Use -2 to -5 for a specific factual error or a
  specific damaging omission that a shallow report is likely to commit. Write
  these so that they are satisfied ONLY when the report does the bad thing.

HERE ARE REAL RUBRIC ITEMS IN THE TARGET STYLE
{examples}

NOW WRITE A RUBRIC FOR THIS TASK

Question given to the report writer:
{question}

The researcher who designed this question already investigated the topic. These are the findings they established - each is specific, drawn from at least two sources, and NOT stated outright by any single source. The question deliberately does not mention them; a good answer has to arrive at them independently.

{findings}

INSTRUCTIONS
1. Turn every finding into at least one Implicit Criteria item stating the specific content the report must contain - the named entity, the actual number, the date, the unit. Do not soften a finding into a theme.
2. Where a finding records what a shallow answer says instead, add a negative-weight item that fires when the report gives only that vaguer version.
3. Add Explicit Criteria for what the question asked for directly.
4. Add Synthesis items where the answer must reconcile or combine findings rather than list them.
5. Add a few Communication Quality, Instruction Following and References items.
6. 20-30 items total. Vary the weights; do not give everything a 3.

WRITING RULES - these are what separate a usable rubric from a checklist

STANDALONE. Every item must be gradeable by someone holding only the report and
that one line. Never write "the finding", "neither study", "the exclusion
criteria", "as established above", "the observed drop". Restate the referent
inside the item, even if it makes the item longer.

ACCEPTANCE SETS. If the item's verb is a judgment verb - explains, analyses,
compares, reconciles, considers, details, discusses - it MUST end with a
parenthetical listing 2-4 concrete things that count as satisfying it. This is
what makes such an item gradeable at all:

  weak:   "The response explains how the two fee schedules differ."
  usable: "The response explains how the two fee schedules differ (e.g., that the
           $89 tier excludes weekend pickup, that the annual plan is billed in
           advance, that cancellation forfeits the remaining term)."

ONE FACT PER ITEM. No item may assert more than two numbers or two named
entities. If a finding contains four figures, write four items. A report that
gets three of four must not leave two graders splitting on one line.

NO PROMPT ECHO. An Explicit Criteria item may never be the question's own wording
prefixed with "The response addresses...". Every on-topic report would pass it, so
it measures nothing. Bind the requirement to an observable value or artifact:

  echo:   "The response addresses the hidden costs of the service."
  usable: "The response states the certificate refresh obligation and names at
           least one recurring cost beyond bandwidth."

SOFT AXES MUST BE OPERATIONAL. In Communication Quality and Instruction
Following, the words clear, clearly, professional, suitable, appropriate,
excessive, generic, throughout, where available and key are banned unless
immediately followed by a test that can be run: an exact section name, a word
count, "defined at first use", "no first-person pronouns", "as a table with one
row per option".

SPREAD THE NEGATIVES. Do not park every penalty on Implicit Criteria. A report
can also fail by ignoring a sub-question that was asked (Instruction Following),
by presenting a derived figure with no source (References), or by burying a
comparison in prose that was asked for as a table (Communication Quality). Write
penalties on at least three different axes. Avoid hinging a penalty on "implies"
or "suggests" - a penalty must fire on something the report actually says.

BALANCE. Not every item should be a number-recall check. Roughly a third of the
Implicit items should demand an explanation, a mechanism, or a caveat rather than
a figure.

Output a JSON array and nothing else. Wrap it in <json_output></json_output> tags.

<json_output>
[{{"criterion": "...", "weight": 5, "axis": "Implicit Criteria"}}]
</json_output>"""


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
    keep = [
        i for i in items
        if isinstance(i, dict) and i.get("criterion") and i.get("axis") in AXES
        and isinstance(i.get("weight"), (int, float))
    ]
    return keep or None


def render_findings(item):
    fs = item.get("findings")
    if not fs:
        return "(none recorded - write the rubric from the question alone)"
    out = []
    for f in fs:
        if not isinstance(f, dict):
            continue
        parts = [f"- CLAIM: {f.get('claim','')}"]
        if f.get("operation"):
            parts.append(f"  operation: {f['operation']}")
        if f.get("derivation"):
            parts.append(f"  how it was established: {f['derivation']}")
        if f.get("shallow_miss"):
            parts.append(f"  what a shallow answer says instead: {f['shallow_miss']}")
        # evidence[] is the current shape; sources[] was the first-run shape.
        ev = f.get("evidence")
        if isinstance(ev, list) and ev:
            for e in ev[:4]:
                if isinstance(e, dict):
                    q = (e.get("quote") or "")[:200]
                    parts.append(f"  source {e.get('url','')}\n    quotes: {q}")
        elif f.get("sources"):
            parts.append(f"  sources: {', '.join(map(str, f['sources'][:4]))}")
        out.append("\n".join(parts))
    return "\n\n".join(out)


def one(item, examples):
    prompt = PROMPT.format(examples=examples or "(none available)",
                           question=item["prompt"],
                           findings=render_findings(item))
    for attempt in range(RETRIES):
        try:
            r = litellm.completion(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                   max_tokens=MAX_TOKENS, **KW)
            items = parse_items(r.choices[0].message.content or "")
            if items:
                return {**{k: v for k, v in item.items() if k != "solution"},
                        "rubrics": items}
        except Exception as e:
            if attempt == RETRIES - 1:
                with _print_lock:
                    print(f"  id={item.get('id')} failed: {type(e).__name__}: {str(e)[:120]}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input_file)]
    examples = load_rubric_examples()
    print(f"{len(rows)} questions | few-shot examples: {'yes' if examples else 'NO'} | model {MODEL}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, r in enumerate(ex.map(lambda it: one(it, examples), rows), 1):
            if r:
                results.append(r)
            print(f"[{i}/{len(rows)}] {'ok' if r else 'FAILED'}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = sum(len(r["rubrics"]) for r in results)
    print(f"\nwrote {len(results)}/{len(rows)} rubrics, {n} items total")
    if results:
        from collections import Counter
        c = Counter(i["axis"] for r in results for i in r["rubrics"])
        tot = sum(c.values())
        for a, v in c.most_common():
            print(f"  {a:>30}: {v:>4}  {100*v/tot:>5.1f}%")
        neg = sum(1 for r in results for i in r["rubrics"] if i["weight"] < 0)
        print(f"  negative-weight items: {neg}")


if __name__ == "__main__":
    main()
