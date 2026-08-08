"""Run the three-agent question chain: explore -> deepen -> ask.

    NUM_ITERATIONS=8 WORKERS=4 python generate_chain.py --out_dir outputs/<tag>

Three independent ReAct loops per question. Stage 1 and 2 have search and visit;
stage 3 has neither, so the question it writes cannot drift off the evidence
behind it.

What moves between the stages is structured, never a corpus. Handing stage 2 the
raw tool responses of stage 1 (what generate_stages23.py did) breaks the only
mechanical anti-fabrication check there is: a quote has to be verbatim from a
tool response, and that means anything if and only if the tool responses are the
quoting agent's own. Text pasted into a user turn is not. The search/visit disk
cache makes stage 2's re-retrieval of the same urls close to free, and stage 1's
search log travels along so stage 2 knows which directions came back empty.

Stage 1 may return `unusable`, in which case stages 2 and 3 do not run. That
verdict was removed from the single-call version because bailing there saved
nothing and so was pure escape hatch; split three ways it saves both tool-using
calls. There is no gate on it — the search and visit counts are recorded next to
it, so abuse shows up in the data rather than being pre-empted by a threshold.

Output, all appended as each question finishes so a kill keeps what completed:

    <out_dir>/chain.jsonl      one line per completed question, all three stages
    <out_dir>/unusable.jsonl   stage 1 said the keyword leads nowhere
    <out_dir>/failed.jsonl     a stage did not return usable JSON
    <out_dir>/a{1,2,3}_trajectories/

extract_chain.py turns chain.jsonl into proposed_qa.jsonl. Shaping is kept out of
here so a bug in it can be fixed without re-running the model.
"""
import argparse
import asyncio
import json
import os
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

from generation_agent_longform import MultiTurnReactAgent
from generation_prompt_explore import SYSTEM_PROMPT as EXPLORE_PROMPT
from generation_prompt_deepen import SYSTEM_PROMPT as DEEPEN_PROMPT
from generation_prompt_ask import SYSTEM_PROMPT as ASK_PROMPT
from extract_proposed_qa import coerce_prediction_json
from generate_longform_tasks import (
    DOMAIN_TO_CSV,
    category_structure,
    find_main_category,
    load_fixed_pairs,
    sample_keywords_from_csv,
    sample_subcategory_with_weights,
    subcategory_counts,
    subcategory_lock,
)

model = os.environ.get("DEEPRESEARCH_MODEL_NAME", "openai/qwen3.5-122b")
model_type = ('bedrock' if model.startswith('bedrock/') else
              'azure' if model.startswith('azure/') else
              'vllm' if model.startswith('vllm/') else 'openai')
generate_cfg = {'max_tokens': 20000, 'max_retries': 10, 'temperature': 1}
if model_type != 'openai':
    generate_cfg['top_p'] = 0.95
llm_cfg = {'model': model, 'generate_cfg': generate_cfg, 'model_type': model_type}

executor = None
write_lock = threading.Lock()

# A quote is trimmed only for the stage-3 turn, which has no tools and reads the
# whole chain at once. The full text stays in chain.jsonl.
MAX_QUOTE_CHARS = 300


def append(path, record):
    with write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rounds(traj):
    """(searches, visits) before the answer — the same count extract_spines.py
    uses, so numbers from the two pipelines are comparable."""
    s = v = 0
    for m in (traj or {}).get("messages", []):
        if m.get("role") != "assistant":
            continue
        c = m.get("content") or ""
        if "<answer>" in c:
            break
        for raw in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", c, re.S):
            try:
                name = json.loads(raw).get("name")
            except Exception:
                continue
            if name == "search":
                s += 1
            elif name == "visit":
                v += 1
    return s, v


# --------------------------------------------------------------------------
# the three user turns
# --------------------------------------------------------------------------
def explore_user_content(topic, keyword):
    return (
        f"Topic: {topic}\n\n"
        f"Initial Keyword: {keyword}\n\n"
        "Note: the keyword above is a starting point sampled from a trending-search "
        "list, not a requirement. Search wide around it first, then report in "
        "`keyword_verdict` what you did with it."
    )


def deepen_user_content(p1, topic):
    out = [f"TOPIC AREA\n{topic}\n", f"SPINE\n{p1.get('spine')}\n"]
    subs = [s for s in (p1.get("subtopics") or []) if isinstance(s, dict)]
    # The system prompt already says what these notes are and are not; the user
    # turn just labels them.
    out.append("WHERE A FIRST PASS FOUND MATERIAL — its notes, with how much each turned up\n"
               "and what it showed.\n")
    for i, s in enumerate(subs, 1):
        out.append(f"  {i}. [{s.get('material')}] {s.get('handle')} — {s.get('query')}")
        out.append(f"     shows : {s.get('what_it_shows')}")
        if s.get("source"):
            out.append(f"     source: {s.get('source')}")
    searched = [q for q in (p1.get("searched") or []) if isinstance(q, dict)]
    if searched:
        out.append("\nALREADY SEARCHED — what that pass ran and what came back. The pages behind\n"
                   "these are not in your context: retrieve anything you intend to quote.\n")
        for q in searched:
            out.append(f"  {q.get('query')!r} -> {q.get('returned')}")
    if p1.get("keyword_note"):
        out.append(f"\nHOW THE TOPIC WAS ARRIVED AT\n{p1['keyword_note']}")
    return "\n".join(out)


def ask_user_content(p2, topic):
    stmts = [s for s in (p2.get("statements") or []) if isinstance(s, dict)]
    finds = [f for f in (p2.get("findings") or []) if isinstance(f, dict)]
    smap = {s.get("id"): s for s in stmts}
    out = [f"TOPIC AREA\n{topic}\n", f"SPINE\n{p2.get('spine')}\n", "SUBTOPICS"]
    for s in (p2.get("subtopics") or []):
        if not isinstance(s, dict):
            continue
        out.append(f"  {s.get('handle')} — {s.get('query')}   [{s.get('material')}]")
        out.append(f"      shows: {s.get('what_it_shows')}")

    out.append("\nSTATEMENTS")
    for s in stmts:
        out.append(f"\n  [{s.get('id')}] ({s.get('subtopic')}) {s.get('claim')}")
        for e in (s.get("evidence") or []):
            if not isinstance(e, dict):
                continue
            q = (e.get("quote") or "")[:MAX_QUOTE_CHARS]
            out.append(f"        \"{q}\"")
            out.append(f"          — {e.get('source')}")

    out.append("\nFINDINGS")
    for f in finds:
        src = f.get("from") or []
        out.append(f"\n  [{f.get('id')}] from {src}")
        for ref in src:
            if ref in smap:
                out.append(f"        {ref}: {smap[ref].get('claim')}")
            else:
                out.append(f"        {ref}: (a finding — keeping this one keeps that one)")
        out.append(f"    analysis    : {f.get('analysis')}")
        out.append(f"    conclusion  : {f.get('conclusion')}")
        out.append(f"    shallow_miss: {f.get('shallow_miss')}")

    open_gaps = [g for g in (p2.get("gaps") or [])
                 if isinstance(g, dict) and str(g.get("closed")).lower() != "yes"]
    if open_gaps:
        out.append("\nSTILL OPEN — the previous pass could not close these. A question that "
                   "turns on one of them cannot be answered from this material.")
        for g in open_gaps:
            out.append(f"  - {g.get('gap')} ({g.get('how')})")
    return "\n".join(out)


# --------------------------------------------------------------------------
# one question, three stages
# --------------------------------------------------------------------------
def _stage(loop, prompt, user_content, topic, iid, traj_dir, tools):
    a = MultiTurnReactAgent(llm=llm_cfg, function_list=tools)
    # traj_dir is read from the environment at construction, which would put all
    # three stages in one directory. Set it per instance — safe under
    # concurrency, unlike setting the env var.
    a.traj_dir = traj_dir
    os.makedirs(traj_dir, exist_ok=True)
    return loop.run_in_executor(executor, lambda: a._run({
        "item": {"question": topic, "answer": "1"},
        "system_prompt": prompt,
        "user_content": user_content,
        "iteration_id": iid,
        "subcategory": topic,
    }, model))


async def run_one(i, sem, args):
    async with sem:
        fixed = load_fixed_pairs()
        if fixed:
            pair = fixed[i % len(fixed)]
            topic = pair["topic"]
            main_category = find_main_category(category_structure, topic)
            keyword = pair["keyword"]
        else:
            main_category, topic = sample_subcategory_with_weights(
                category_structure, subcategory_counts, subcategory_lock)
            kws = sample_keywords_from_csv(topic, num_keywords=1)
            if not kws:
                print(f"[#{i+1}] no keyword available for {topic!r}")
                return None
            keyword = kws[0]

        # Which pool the keyword came from. Six subcategories share
        # entertainment.csv and the four worst questions of findings8_15312041
        # all came out of it; with the complexity axes gone this seed is the only
        # sampled input left, so its provenance has to be recoverable per row.
        seed = {"id": i + 1, "topic": topic, "main_category": main_category,
                "keyword": keyword, "source_csv": DOMAIN_TO_CSV.get(topic)}
        print(f"[#{i+1}] {topic} / {keyword!r} (from {seed['source_csv']})")
        try:
            return await _chain(i, seed, topic, keyword, args)
        except Exception as e:
            # gather(return_exceptions=True) would otherwise swallow this into a
            # count, and a whole run's worth of tool calls disappears with it.
            append(args.failed, {**seed, "stage": "exception",
                                 "error": f"{type(e).__name__}: {e}",
                                 "traceback": traceback.format_exc()})
            print(f"[#{i+1}] EXCEPTION {type(e).__name__}: {e}")
            return None


async def _chain(i, seed, topic, keyword, args):
        loop = asyncio.get_event_loop()

        # ---- stage 1: explore --------------------------------------------
        r1 = await _stage(loop, EXPLORE_PROMPT, explore_user_content(topic, keyword),
                          topic, f"{i+1}a1", os.path.join(args.out_dir, "a1_trajectories"),
                          ["search", "visit"])
        p1 = coerce_prediction_json(r1.get("prediction"))
        s1, v1 = rounds(r1)
        if not p1:
            append(args.failed, {**seed, "stage": "explore",
                                 "termination": r1.get("termination"),
                                 "searches": s1, "visits": v1})
            print(f"[#{i+1}] EXPLORE unparseable (termination={r1.get('termination')})")
            return None

        if (p1.get("keyword_verdict") or "").strip().lower() == "unusable":
            append(args.unusable, {**seed,
                                   "keyword_note": p1.get("keyword_note"),
                                   "searched": p1.get("searched"),
                                   "searches": s1, "visits": v1})
            print(f"[#{i+1}] unusable after {s1} searches / {v1} visits — stopping here")
            return None
        if not (p1.get("spine") or "").strip() or not p1.get("subtopics"):
            append(args.failed, {**seed, "stage": "explore", "reason": "no spine or no subtopics",
                                 "searches": s1, "visits": v1})
            print(f"[#{i+1}] EXPLORE produced no spine/subtopics")
            return None
        print(f"[#{i+1}] explore -> {len(p1['subtopics'])} subtopics "
              f"({s1} searches, {v1} visits)")

        # ---- stage 2: deepen ---------------------------------------------
        r2 = await _stage(loop, DEEPEN_PROMPT, deepen_user_content(p1, topic),
                          topic, f"{i+1}a2", os.path.join(args.out_dir, "a2_trajectories"),
                          ["search", "visit"])
        p2 = coerce_prediction_json(r2.get("prediction"))
        s2, v2 = rounds(r2)
        if not p2 or not p2.get("findings"):
            append(args.failed, {**seed, "stage": "deepen",
                                 "termination": r2.get("termination"),
                                 "searches": s2, "visits": v2})
            print(f"[#{i+1}] DEEPEN FAILED (termination={r2.get('termination')})")
            return None
        n_sub1 = len(p1.get("subtopics") or [])
        n_sub2 = len(p2.get("subtopics") or [])
        print(f"[#{i+1}] deepen -> subtopics {n_sub1} -> {n_sub2}, "
              f"{len(p2.get('statements') or [])} statements, "
              f"{len(p2['findings'])} findings ({s2} searches, {v2} visits)")

        # ---- stage 3: ask, no tools --------------------------------------
        r3 = await _stage(loop, ASK_PROMPT, ask_user_content(p2, topic),
                          topic, f"{i+1}a3", os.path.join(args.out_dir, "a3_trajectories"),
                          [])
        p3 = coerce_prediction_json(r3.get("prediction"))
        if not p3 or not p3.get("proposed_question"):
            append(args.failed, {**seed, "stage": "ask",
                                 "termination": r3.get("termination")})
            print(f"[#{i+1}] ASK FAILED (termination={r3.get('termination')})")
            return None
        centre = p3.get("centre") or {}
        print(f"[#{i+1}] ask -> kept {len(centre.get('kept') or [])} / "
              f"discarded {len(centre.get('discarded') or [])}")

        record = {**seed, "explore": p1, "deepen": p2, "ask": p3,
                  "rounds": {"explore": {"searches": s1, "visits": v1},
                             "deepen": {"searches": s2, "visits": v2}}}
        append(args.chain, record)
        return record


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=int(os.environ.get("NUM_ITERATIONS", 10)))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 4)))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.chain = os.path.join(args.out_dir, "chain.jsonl")
    args.unusable = os.path.join(args.out_dir, "unusable.jsonl")
    args.failed = os.path.join(args.out_dir, "failed.jsonl")

    global executor
    executor = ThreadPoolExecutor(max_workers=args.workers)
    sem = asyncio.Semaphore(args.workers)
    print(f"chain: {args.n} questions, workers={args.workers}, model={model}\n")

    results = await asyncio.gather(*[run_one(i, sem, args) for i in range(args.n)],
                                   return_exceptions=True)
    ok = errs = 0
    for r in results:
        if isinstance(r, Exception):
            errs += 1
            print(f"  ERROR: {type(r).__name__}: {r}")
        elif r:
            ok += 1

    def count(p):
        return sum(1 for _ in open(p)) if os.path.exists(p) else 0

    print(f"\n{ok}/{args.n} complete -> {args.chain}")
    print(f"  unusable : {count(args.unusable)}")
    print(f"  failed   : {count(args.failed)}")
    print(f"  errored  : {errs}")
    executor.shutdown(wait=True)


if __name__ == "__main__":
    asyncio.run(main())
