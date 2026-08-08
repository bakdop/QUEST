"""Run STAGE 2 and STAGE 3 over the spines produced by stage 1.

  python generate_stages23.py --spines outputs/<run>/spines.jsonl \
                              --traj_dir outputs/<run>/trajectories \
                              --out_dir  outputs/<run>

Stage 2 gets the spine plus the corpus stage 1 already collected — the tool
responses out of the stage-1 trajectory, not its reasoning. The reasoning is
deliberately withheld: it contains stage 1's own subject-selection rationalising
and there is nothing downstream to gain from it.

Stage 3 gets the refined spine plus stage 2's findings and essentials, and no
tools at all.

Writes proposed_qa.jsonl in the shape longform_rubric/generate_criteria_findings.py
already expects (prompt / findings / essentials / centre / the four axes), so
the existing rubric stage runs on top of this unchanged.
"""
import argparse
import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from generation_agent_longform import MultiTurnReactAgent
from generation_prompt_stage2 import SYSTEM_PROMPT as STAGE2_PROMPT
from generation_prompt_stage3 import SYSTEM_PROMPT as STAGE3_PROMPT
from extract_proposed_qa import coerce_prediction_json

model = os.environ.get("DEEPRESEARCH_MODEL_NAME", "openai/qwen3.5-122b")
model_type = ('bedrock' if model.startswith('bedrock/') else
              'azure' if model.startswith('azure/') else
              'vllm' if model.startswith('vllm/') else 'openai')
generate_cfg = {'max_tokens': 20000, 'max_retries': 10, 'temperature': 1}
if model_type != 'openai':
    generate_cfg['top_p'] = 0.95
llm_cfg = {'model': model, 'generate_cfg': generate_cfg, 'model_type': model_type}

executor = None


# --------------------------------------------------------------------------
# corpus assembly
# --------------------------------------------------------------------------
def build_corpus(traj, max_chars):
    """The tool responses from a stage-1 trajectory, in order, capped.

    Kept as raw tool output rather than a summary: stage 2's first required act
    is to inventory figures and put same-quantity values side by side, and a
    summary is exactly where those values stop being comparable.
    """
    blocks, total = [], 0
    for m in traj.get("messages", []):
        if m.get("role") != "user":
            continue
        c = m.get("content") or ""
        if "<tool_response>" not in c:
            continue
        if total + len(c) > max_chars:
            blocks.append(f"\n[corpus truncated at {max_chars} chars; "
                          f"{len(traj['messages'])} messages in the source trajectory]")
            break
        blocks.append(c)
        total += len(c)
    return "\n\n".join(blocks)


def stage2_user_content(spine, corpus):
    lines = [f"SPINE\n{spine.get('spine')}\n"]
    subs = [s for s in (spine.get("subtopics") or []) if isinstance(s, dict)]
    if subs:
        lines.append("SUBTOPICS STAGE 1 MAPPED, with how much material each has and what it shows.\n"
                     "This is a map of where stage 1 found material, not a set of boxes to fill. "
                     "Add, split, merge or drop as your own investigation requires, and give the "
                     "final list in `subtopics_refined`.")
        for i, s in enumerate(subs, 1):
            lines.append(f"  {i}. [{s.get('material')}] {s.get('subtopic')}")
            lines.append(f"     shows : {s.get('what_it_shows')}")
            lines.append(f"     source: {s.get('source')}")
        lines.append("")
    if spine.get("tension"):
        lines.append(f"WHY THIS CANNOT BE SETTLED FROM ONE ANGLE\n{spine['tension']}\n")
    lines.append(
        "CORPUS — everything the tools returned while stage 1 arrived at that spine.\n"
        "It is yours; do not re-fetch what is already here. Inventory it first, then "
        "search only for what the inventory opens up.\n")
    lines.append(corpus)
    return "\n".join(lines)


def stage3_user_content(spine_refined, statements, findings, essentials, topic):
    smap = {s.get("id"): s for s in statements if isinstance(s, dict)}
    ess_ids = {e.get("id") for e in essentials if isinstance(e, dict)}
    out = [f"TOPIC AREA\n{topic}\n", f"SPINE\n{spine_refined}\n", "STATEMENTS"]
    for s in statements:
        if not isinstance(s, dict):
            continue
        mark = " [ESSENTIAL]" if s.get("id") in ess_ids else ""
        ev = [e for e in (s.get("evidence") or []) if isinstance(e, dict)]
        out.append(f"  [{s.get('id')}]{mark} {s.get('claim')}")
        # show how many independent sources carry the claim — a statement backed
        # by a regulator and a study is not the same object as one backed by a
        # single vendor blog, and stage 3 should be able to see the difference
        for e in ev:
            out.append(f"        backed by: {e.get('source')}")
    why = {e.get("id"): e.get("why_expected") for e in essentials if isinstance(e, dict)}
    if why:
        out.append("\nWHY THE ESSENTIALS ARE ESSENTIAL")
        for sid, w in why.items():
            out.append(f"  [{sid}] {w}")
    out.append("\nFINDINGS")
    for f in findings:
        src = f.get("from") or []
        out.append(f"\n  [{f.get('id')}] from {src}")
        # spell the chain out so the dependency is legible without cross-referencing
        for ref in src:
            if ref in smap:
                out.append(f"        {ref}: {smap[ref].get('claim')}")
            else:
                out.append(f"        {ref}: (a finding — keeping this one keeps that one)")
        out.append(f"    analysis    : {f.get('analysis')}")
        out.append(f"    conclusion  : {f.get('conclusion')}")
        out.append(f"    shallow_miss: {f.get('shallow_miss')}")
    return "\n".join(out)


def to_legacy(statements, findings, essentials):
    """Expand the compact id form into the shape the rubric generator expects.

    longform_rubric/generate_criteria_findings.py reads findings with
    observation/evidence and essentials with point/source/why_expected. Keeping
    that contract means the rubric stage runs unchanged on top of the new
    ontology; the structured `statements` and `from` chains are written
    alongside, not instead.
    """
    smap = {s.get("id"): s for s in statements if isinstance(s, dict)}

    def ev_of(s):
        return [e for e in (s.get("evidence") or []) if isinstance(e, dict)]

    out_f = []
    for f in findings:
        refs = [smap[r] for r in (f.get("from") or []) if r in smap]
        out_f.append({
            "id": f.get("id"),
            "from": f.get("from"),
            "observation": " ".join(r.get("claim") or "" for r in refs),
            "analysis": f.get("analysis"),
            "conclusion": f.get("conclusion"),
            "shallow_miss": f.get("shallow_miss"),
            # a statement may now be corroborated, so this is a flatten, not a
            # 1:1 map — one cited statement can contribute several evidence rows
            "evidence": [{"url": e.get("source"), "quote": e.get("quote"),
                          "contributes": r.get("claim")}
                         for r in refs for e in ev_of(r)],
        })
    out_e = []
    for e in essentials:
        if not isinstance(e, dict):
            continue
        s = smap.get(e.get("id")) or {}
        ev = ev_of(s)
        out_e.append({
            "id": e.get("id"),
            "point": s.get("claim"),
            "source": ev[0].get("source") if ev else None,
            "sources": [x.get("source") for x in ev],
            "why_expected": e.get("why_expected"),
        })
    return out_f, out_e


# --------------------------------------------------------------------------
# one spine, both stages
# --------------------------------------------------------------------------
async def run_one(spine, traj_dir, sem, args):
    async with sem:
        sid = spine["id"]
        # Gate on the spine itself, not on a verdict. `unusable` used to be a
        # verdict stage 1 could emit; it was removed because the model reached
        # for it instead of widening the keyword. Any record without a spine is
        # nothing for stage 2 to work from, whatever it calls itself.
        if not (spine.get("spine") or "").strip():
            print(f"[#{sid}] skipped: stage 1 produced no spine "
                  f"(verdict={spine.get('keyword_verdict')!r})")
            return None
        traj_path = os.path.join(traj_dir, spine["trajectory"])
        traj = json.load(open(traj_path))
        corpus = build_corpus(traj, args.max_corpus_chars)
        if not corpus.strip():
            print(f"[#{sid}] skipped: stage-1 trajectory has no tool responses")
            return None

        loop = asyncio.get_event_loop()

        # ---- stage 2: findings + essentials, with tools -------------------
        # traj_dir is read from the environment at construction, which would put
        # these next to stage 1's and break extract_spines.py (it sorts on
        # int(filename.split('_')[1]), and these ids are "3s2"). Set it per
        # instance instead — safe under concurrency, unlike setting the env var.
        a2 = MultiTurnReactAgent(llm=llm_cfg, function_list=["search", "visit"])
        a2.traj_dir = os.path.join(args.out_dir, "stage2_trajectories")
        os.makedirs(a2.traj_dir, exist_ok=True)
        r2 = await loop.run_in_executor(executor, lambda: a2._run({
            "item": {"question": spine.get("topic") or "", "answer": "1"},
            "system_prompt": STAGE2_PROMPT,
            "user_content": stage2_user_content(spine, corpus),
            "iteration_id": f"{sid}s2",
            "subcategory": spine.get("topic"),
        }, model))
        p2 = coerce_prediction_json(r2.get("prediction"))
        if not p2 or not p2.get("findings"):
            print(f"[#{sid}] STAGE 2 FAILED (termination={r2.get('termination')})")
            return None
        statements = [s for s in (p2.get("statements") or []) if isinstance(s, dict)]
        findings = [f for f in p2["findings"] if isinstance(f, dict)]
        essentials = [e for e in (p2.get("essentials") or []) if isinstance(e, dict)]
        spine_refined = p2.get("spine_refined") or spine.get("spine")
        subs_refined = p2.get("subtopics_refined") or [
            s.get("subtopic") for s in (spine.get("subtopics") or []) if isinstance(s, dict)]
        chained = sum(1 for f in findings
                      if any(str(r).startswith("F") for r in (f.get("from") or [])))
        n0 = len(spine.get("subtopics") or [])
        print(f"[#{sid}] stage 2 -> {len(statements)} statements, {len(findings)} findings "
              f"({chained} chained), {len(essentials)} essentials, "
              f"subtopics {n0} -> {len(subs_refined)}")

        # ---- stage 3: the question, no tools -------------------------------
        a3 = MultiTurnReactAgent(llm=llm_cfg, function_list=[])
        a3.traj_dir = os.path.join(args.out_dir, "stage3_trajectories")
        os.makedirs(a3.traj_dir, exist_ok=True)
        r3 = await loop.run_in_executor(executor, lambda: a3._run({
            "item": {"question": spine.get("topic") or "", "answer": "1"},
            "system_prompt": STAGE3_PROMPT,
            "user_content": stage3_user_content(spine_refined, statements, findings,
                                                essentials, spine.get("topic")),
            "iteration_id": f"{sid}s3",
            "subcategory": spine.get("topic"),
        }, model))
        p3 = coerce_prediction_json(r3.get("prediction"))
        if not p3 or not p3.get("proposed_question"):
            print(f"[#{sid}] STAGE 3 FAILED (termination={r3.get('termination')})")
            return None
        print(f"[#{sid}] stage 3 -> kept {len((p3.get('centre') or {}).get('kept') or [])}"
              f" / discarded {len((p3.get('centre') or {}).get('discarded') or [])}")

        legacy_f, legacy_e = to_legacy(statements, findings, essentials)
        return {
            "id": sid,
            "topic": spine.get("topic"),
            "keyword": spine.get("keyword"),
            "keyword_verdict": spine.get("keyword_verdict"),
            "spine": spine.get("spine"),
            "spine_refined": spine_refined,
            "spine_change_note": p2.get("spine_change_note"),
            "subtopics": subs_refined,
            "subtopic_change_note": p2.get("subtopic_change_note"),
            "gaps_found": p2.get("gaps_found"),
            "prompt": p3["proposed_question"],
            # `findings`/`essentials` carry the legacy shape the rubric generator
            # reads; the compact id form is kept alongside under *_compact so the
            # chain survives into anything that wants it.
            "statements": statements,
            "findings": legacy_f,
            "essentials": legacy_e,
            "findings_compact": findings,
            "essentials_compact": essentials,
            "independence_note": p2.get("independence_note"),
            "centre": p3.get("centre"),
            "conceptual_breadth": p3.get("conceptual_breadth"),
            "logical_nesting": p3.get("logical_nesting"),
            "exploration": p3.get("exploration"),
            "analysis_load": p3.get("analysis_load"),
            "solution": p3.get("solution"),
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spines", required=True)
    ap.add_argument("--traj_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 4)))
    ap.add_argument("--max_corpus_chars", type=int, default=200_000)
    ap.add_argument("--require_qc_pass", action="store_true",
                    help="skip spines that failed stage-1 compliance checks")
    args = ap.parse_args()

    global executor
    executor = ThreadPoolExecutor(max_workers=args.workers)
    sem = asyncio.Semaphore(args.workers)

    spines = [json.loads(l) for l in open(args.spines) if l.strip()]
    if args.require_qc_pass:
        before = len(spines)
        spines = [s for s in spines if (s.get("_qc") or {}).get("pass")]
        print(f"require_qc_pass: {len(spines)}/{before} spines kept")
    print(f"running stages 2+3 over {len(spines)} spines (workers={args.workers})")

    results = await asyncio.gather(*[run_one(s, args.traj_dir, sem, args) for s in spines],
                                   return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        elif r:
            out.append(r)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "proposed_qa.jsonl")
    with open(path, "w") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(out)}/{len(spines)} -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
