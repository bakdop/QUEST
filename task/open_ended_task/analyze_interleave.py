"""Did the reasoning/retrieval interleave actually happen? Measured off the trajectory.

    python analyze_interleave.py --input_dir outputs/<tag>/trajectories

Since 2026-08-09 nothing in the output records the session at all. `key_queries`
was a retrospective self-report of what the run did; `research_path` replaced it
and answers a different question — the path someone holding only the finished
question would walk to reach the centre. Neither the generator's wandering nor its
change of topic appears anywhere now, by design.

So this script is the only measurement of whether the run itself interleaved, and
it takes it off the trajectory, where the model has no say.

The trajectory has the real sequence — which round issued which query, and which
round first returned each url — so the interleave can be counted rather than
asked for:

  rounds            how many search/visit rounds happened at all
  source_rounds     how many distinct rounds produced a url that a statement
                    ends up citing. 1 means everything came from one sweep.
  carried_queries   queries containing a term that first appeared in an earlier
                    round's tool output and is not in the topic or keyword —
                    a query that could only have been written after reading.

Read against the `research_path` depth that extract_evidence.py reports, the two
now separate a different pair of cases:

  interleave high, path shallow  the run did the work and the question does not
                                 need it. The material is deeper than what was
                                 asked for — tighten the question, not the search.
  both low                       the run stopped early. A prompt change is the
                                 wrong first move — this project has tied on six
                                 of them; look at why it stopped.
  interleave low, path deep      the path is a story. Someone holding only the
                                 question could not have walked it, because the
                                 run never walked anything like it either.
"""
import argparse
import json
import os
import re
import statistics as st
from collections import Counter

from extract_proposed_qa import coerce_prediction_json

STOP = set("the a an of to in for and or is are was were be been on at by with that this "
           "it its as from than more most less not no you your which what how when where "
           "why who vs versus about into over under between across per new".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9$%.\-]+", (s or "").lower())
            if w not in STOP and len(w) > 3}


def rounds_of(traj):
    """[(queries_issued, text_returned)] per round, in order."""
    out, pending = [], None
    for m in traj.get("messages", []):
        c = m.get("content") or ""
        if m.get("role") == "assistant":
            if "<answer>" in c:
                break
            qs = []
            for raw in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", c, re.S):
                try:
                    call = json.loads(raw)
                except Exception:
                    continue
                a = call.get("arguments") or {}
                q = a.get("query") or a.get("goal") or a.get("url") or ""
                qs.extend(q if isinstance(q, list) else [str(q)])
            pending = qs
        elif m.get("role") == "user" and "<tool_response>" in c and pending is not None:
            out.append((pending, c))
            pending = None
    return out


def analyse(traj, pred):
    rs = rounds_of(traj)
    seed = words(traj.get("subcategory"))
    for m in traj.get("messages", []):
        hit = re.search(r"Initial Keyword:\s*(.+)", m.get("content") or "")
        if hit:
            seed |= words(hit.group(1))
            break

    # a query is "carried" when it uses a term that only became available by
    # reading an earlier round
    seen, carried, total_q = set(seed), 0, 0
    for qs, resp in rs:
        for q in qs:
            total_q += 1
            if words(q) - seen - seed:
                carried += 1
        seen |= words(resp)

    stmts = [s for s in (pred.get("statements") or []) if isinstance(s, dict)]
    srcs = {e.get("source") for s in stmts for e in (s.get("evidence") or [])
            if isinstance(e, dict)}
    src_rounds = set()
    for i, (_, resp) in enumerate(rs):
        if any(u and u in resp for u in srcs):
            src_rounds.add(i + 1)
    path = [q for q in (pred.get("research_path") or pred.get("key_queries") or [])
            if isinstance(q, dict)]
    recorded = sum(1 for q in path if q.get("from"))
    return {
        "rounds": len(rs),
        "queries": total_q,
        "carried_queries": carried,
        "carried_share": round(carried / total_q, 2) if total_q else 0.0,
        "source_rounds": len(src_rounds),
        "statements": len(stmts),
        "recorded": recorded,
        "recorded_share": round(recorded / len(path), 2) if path else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(os.listdir(args.input_dir)):
        if not f.endswith(".json"):
            continue
        traj = json.load(open(os.path.join(args.input_dir, f)))
        pred = coerce_prediction_json(traj.get("prediction"))
        if not pred or not pred.get("statements"):
            continue
        rows.append((f, analyse(traj, pred)))

    if not rows:
        print("no parseable trajectories")
        return

    print(f"{'file':<34} {'rnds':>4} {'qs':>4} {'carried':>8} {'srcR':>5} "
          f"{'stmts':>6} {'recorded':>9}")
    for f, a in rows:
        print(f"{f[:34]:<34} {a['rounds']:>4} {a['queries']:>4} "
              f"{a['carried_queries']:>3}/{a['carried_share']:<4} {a['source_rounds']:>5} "
              f"{a['statements']:>6} {a['recorded']:>4}/{a['recorded_share']:<4}")

    def med(k):
        return st.median([a[k] for _, a in rows])

    print(f"\nmedians over {len(rows)} runs")
    print(f"  rounds                {med('rounds')}")
    print(f"  carried-query share   {med('carried_share')}"
          f"   <- retrieval driven by what was read")
    print(f"  rounds that fed a cited source  {med('source_rounds')}"
          f"   (1 = everything from one sweep)")
    print(f"  recorded carried      {med('recorded_share')}   <- the model's own account")

    flat = sum(1 for _, a in rows if a["source_rounds"] <= 1)
    gap = sum(1 for _, a in rows if a["carried_share"] >= 0.4 and a["recorded_share"] < 0.2)
    print(f"\n  {flat}/{len(rows)} drew every cited source from a single round")
    print(f"  {gap}/{len(rows)} carried queries but recorded almost none of it"
          f"   <- annotation gap, not behaviour")


if __name__ == "__main__":
    main()
