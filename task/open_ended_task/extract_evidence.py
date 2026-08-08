"""PROMPT_VARIANT=evidence_first trajectories -> proposed_qa.jsonl.

    python extract_evidence.py --input_dir outputs/<tag>/trajectories \
                               --output_file outputs/<tag>/proposed_qa.jsonl

There is one kind of object upstream now — a statement, with a verbatim quote and
the url that returned it — so the two things this has to do are shape it for the
rubric generator and measure the investigation.

MEASURE. Nothing is injected upstream, so every axis is counted here. Depth is
the length of the `built_on` chain: how far reading had to go before the next
thing to search for was known. Note this counts differently from
extract_chain.py, which required a node to merge two parents before it deepened
the chain. That rule existed to stop an inference ladder scoring as depth —
`0F+2S -> 1F+1S -> 1F+1S` was one conclusion annotated three times. It does not
apply here: a single-parent link means a round of reading determined a round of
searching, and the statement at the end of it carries its own quote and url, so
the chain cannot be padded without actually retrieving. The stricter count is
still reported, as `chain_depth_merged`.

SHAPE. longform_rubric/generate_criteria_findings.py reads `findings` with
`conclusion` and `evidence[{url, quote}]`. Until it is rewritten to read
statements directly, the kept statements are written into that shape as well —
`conclusion` is the claim, `evidence` is its own evidence. Nothing is invented in
the translation, and no `analysis` or `shallow_miss` is fabricated to fill fields
that no longer have a source.
"""
import argparse
import json
import os
import re
import statistics as st
from collections import Counter
from urllib.parse import urlparse

from extract_proposed_qa import coerce_prediction_json

BREADTH = (("Simple", 2, 2), ("Moderate", 3, 5), ("High", 6, 10 ** 6))
LOAD = (("Low", 1, 3), ("Medium", 4, 9), ("High", 10, 10 ** 6))
NESTING = (("Shallow", 1, 1), ("Intermediate", 2, 3), ("Deep", 4, 10 ** 6))


def band(n, table):
    for name, lo, hi in table:
        if lo <= n <= hi:
            return name
    return None


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def keyword_of(traj):
    for m in traj.get("messages", []):
        if m.get("role") != "user":
            continue
        hit = re.search(r"Initial Keyword:\s*(.+)", m.get("content") or "")
        if hit:
            return hit.group(1).strip().splitlines()[0]
    return None


def rounds(traj):
    """(searches, visits) before the answer."""
    s = v = 0
    for m in traj.get("messages", []):
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


def chain_depth(stmts, require_merge=False):
    """How far the investigation had to go.

    A statement with no `built_on` is depth 1 — it was found without needing to
    read anything first. Each link adds one. With require_merge, only a statement
    resting on two or more others counts as a step, which is the count
    extract_chain.py used for inference chains.
    """
    smap = {s.get("id"): s for s in stmts if isinstance(s, dict) and s.get("id")}
    if not smap:
        return 0

    def d(sid, seen=()):
        if sid in seen or sid not in smap:
            return 0
        parents = [p for p in (smap[sid].get("built_on") or []) if p in smap]
        deeper = max([d(p, seen + (sid,)) for p in parents] or [0])
        step = (1 if len(parents) >= 2 else 0) if require_merge else (1 if parents else 0)
        return step + deeper

    return max(d(i) for i in smap) + 1


def to_legacy(stmts, kept):
    """The findings shape the rubric generator still reads.

    A kept statement becomes one item whose `conclusion` is its claim and whose
    `evidence` is its own. `analysis` and `shallow_miss` are left off rather than
    invented — both are `.get()`-guarded downstream.
    """
    out = []
    for s in stmts:
        if kept and s.get("id") not in kept:
            continue
        ev = [e for e in (s.get("evidence") or []) if isinstance(e, dict)]
        out.append({
            "id": s.get("id"),
            "conclusion": s.get("claim"),
            "subtopic": s.get("subtopic"),
            "built_on": s.get("built_on"),
            "evidence": [{"url": e.get("source"), "quote": e.get("quote"),
                          "contributes": s.get("claim")} for e in ev],
        })
    return out


def signals(p, stmts, subs, traj):
    handles = {norm(s.get("handle")) for s in subs}
    per_sub = Counter(norm(s.get("subtopic")) for s in stmts)
    counts = [per_sub.get(h, 0) for h in handles]
    doms = sorted({urlparse(e["source"]).netloc.replace("www.", "")
                   for s in stmts for e in (s.get("evidence") or [])
                   if isinstance(e, dict) and (e.get("source") or "").startswith("http")})
    centre = p.get("centre") or {}
    ids = {s.get("id") for s in stmts}
    s_rounds, v_rounds = rounds(traj)
    built = [s for s in stmts if s.get("built_on")]
    return {
        "built_on_share": round(len(built) / len(stmts), 3) if stmts else 0.0,
        "chain_depth_merged": chain_depth(stmts, require_merge=True),
        "statements_per_subtopic": {"median": st.median(counts) if counts else 0,
                                    "min": min(counts) if counts else 0,
                                    "empty": sum(1 for c in counts if c == 0)},
        "subtopic_words": [len((s.get("query") or "").split()) for s in subs],
        "material": Counter(s.get("material") for s in subs),
        "domains": doms, "n_domains": len(doms),
        "no_evidence": [s.get("id") for s in stmts if not (s.get("evidence") or [])],
        "statements_off_map": [s.get("id") for s in stmts
                               if norm(s.get("subtopic")) not in handles],
        "dangling_built_on": sorted({p_ for s in stmts
                                     for p_ in (s.get("built_on") or []) if p_ not in ids}),
        "kept": len(centre.get("kept") or []),
        "discarded": len(centre.get("discarded") or []),
        "n_key_queries": len(p.get("key_queries") or []),
        "searches": s_rounds, "visits": v_rounds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    files = sorted((f for f in os.listdir(args.input_dir) if f.endswith(".json")),
                   key=lambda x: int(re.sub(r"\D", "", x.split("_")[1]) or 0))
    out, unparseable = [], []
    for f in files:
        traj = json.load(open(os.path.join(args.input_dir, f)))
        p = coerce_prediction_json(traj.get("prediction"))
        if not p or not p.get("proposed_question"):
            unparseable.append((f, traj.get("termination")))
            continue
        stmts = [s for s in (p.get("statements") or []) if isinstance(s, dict)]
        subs = [s for s in (p.get("subtopics") or []) if isinstance(s, dict)]
        kept = {i for i in ((p.get("centre") or {}).get("kept") or [])}

        n_sub, n_st = len(subs), len(stmts)
        depth = chain_depth(stmts)
        implicit = sum(1 for s in subs if str(s.get("exposure")).lower() == "implicit")
        share = implicit / n_sub if n_sub else 0.0
        out.append({
            "id": len(out) + 1,
            "topic": traj.get("subcategory"),
            "keyword": keyword_of(traj),
            "prompt": p["proposed_question"],
            "spine": p.get("spine"),
            "subtopics": subs,
            "statements": stmts,
            "findings": to_legacy(stmts, kept),
            "essentials": [],
            "centre": p.get("centre"),
            "key_queries": p.get("key_queries"),
            "keyword_verdict": p.get("keyword_verdict"),
            "keyword_note": p.get("keyword_note"),
            "trajectory": f,
            "_axes": {
                "conceptual_breadth": n_sub, "conceptual_breadth_band": band(n_sub, BREADTH),
                "analysis_load": n_st, "analysis_load_band": band(n_st, LOAD),
                "logical_nesting": depth, "logical_nesting_band": band(depth, NESTING),
                "exploration": round(share, 3),
                "exploration_band": ("Low" if share < 1 / 3 else
                                     "Medium" if share <= 2 / 3 else "High"),
            },
            "_signals": signals(p, stmts, subs, traj),
        })

    with open(args.output_file, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"extracted {len(out)}/{len(files)} -> {args.output_file}")
    for f, term in unparseable[:8]:
        print(f"  unparseable: {f}  (termination={term})")
    if not out:
        return

    def med(fn):
        return st.median([fn(r) for r in out])

    print(f"  subtopics        median {med(lambda r: r['_axes']['conceptual_breadth'])}"
          f"   (propose baseline: 2-5)")
    print(f"  statements       median {med(lambda r: r['_axes']['analysis_load'])}"
          f"   (propose baseline: 6)")
    words = [w for r in out for w in r['_signals']['subtopic_words']]
    print(f"  subtopic words   median {st.median(words) if words else 0}"
          f"   (propose baseline: 4)")
    per = [r['_signals']['statements_per_subtopic']['median'] for r in out]
    print(f"  statements/sub   median {st.median(per)}   (propose baseline: 2)")
    print(f"  built_on share   median {med(lambda r: r['_signals']['built_on_share'])}"
          f"   (no baseline — this is the new one)")
    print(f"  chain depth      median {med(lambda r: r['_axes']['logical_nesting'])}"
          f"   (>=2 in {sum(1 for r in out if r['_axes']['logical_nesting'] >= 2)}/{len(out)})")
    print(f"  implicit share   median {med(lambda r: r['_axes']['exploration'])}")
    print(f"  searches/visits  median {med(lambda r: r['_signals']['searches'])}"
          f" / {med(lambda r: r['_signals']['visits'])}")
    for k, label in (("no_evidence", "statements with no evidence"),
                     ("statements_off_map", "statements off the subtopic list"),
                     ("dangling_built_on", "built_on ids that do not exist")):
        n = sum(len(r["_signals"][k]) for r in out)
        if n:
            rows = sum(1 for r in out if r["_signals"][k])
            print(f"  ! {label}: {n} across {rows} questions")
    for name, table in (("conceptual_breadth", BREADTH), ("analysis_load", LOAD),
                        ("logical_nesting", NESTING)):
        print(f"  {name:20} {dict(Counter(r['_axes'][f'{name}_band'] for r in out))}")


if __name__ == "__main__":
    main()
