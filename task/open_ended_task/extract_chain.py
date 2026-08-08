"""chain.jsonl -> proposed_qa.jsonl, with the four axes measured rather than read.

    python extract_chain.py --input outputs/<tag>/chain.jsonl \
                            --output outputs/<tag>/proposed_qa.jsonl

Three jobs.

REJOIN. Stage 3 emits statements without their evidence blocks, so the quotes and
urls are never retyped by a stage that has no tools and could not check them. The
evidence is put back here, by id, out of stage 2's output. An id stage 3 invented
therefore arrives with nothing behind it, which is a detectable error rather than
a silent fabrication — #11 of propose512_15497121 invented a statement id and #21
produced a url with a space in it, both while retyping.

MEASURE. Nothing is injected upstream any more, so every axis is counted off the
structures here. Logical Nesting is counted as *synthesis* depth: only a finding
whose `from` names two or more other findings deepens the chain. Under the old
count the ladder `0F+2S -> 1F+1S -> 1F+1S -> 1F+1S` scored Deep, and both runs
seeded Deep in propose512_15497121 produced exactly that shape — a single
conclusion annotated four times. It scores 1 here.

RECORD. Nothing is gated after stage 1, by design: a judgement about how good a
chain is belongs on the whole batch, where a threshold can be picked against a
distribution. That only works if the signals a later filter would need are on
every row, so they are written into `_signals` whether or not anything reads them
yet.

The output keeps the contract longform_rubric/generate_criteria_findings.py
already reads — `prompt`, and `findings`/`essentials`/`centre` in the legacy
shape — so the existing rubric stage runs on top of this unchanged.
"""
import argparse
import json
import os
import re
import statistics as st
from collections import Counter
from urllib.parse import urlparse

# The RR vocabulary, kept so measured values can be compared with the levels the
# old runs were seeded with. Exploration is a share, not a count.
BREADTH = (("Simple", 2, 2), ("Moderate", 3, 5), ("High", 6, 10 ** 6))
LOAD = (("Low", 1, 1), ("Medium", 2, 3), ("High", 4, 10 ** 6))
NESTING = (("Shallow", 1, 1), ("Intermediate", 2, 3), ("Deep", 4, 10 ** 6))


def band(n, table):
    for name, lo, hi in table:
        if lo <= n <= hi:
            return name
    return None


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def synthesis_depth(findings):
    """Longest chain, counting only nodes that actually merge two findings.

    A finding built from statements alone is depth 1. Standing one finding on
    another adds nothing — that is annotation, not derivation. A finding drawing
    on two findings is what makes the next level.
    """
    fmap = {f.get("id"): f for f in findings if isinstance(f, dict) and f.get("id")}
    if not fmap:
        return 0

    def d(fid, seen=()):
        if fid in seen or fid not in fmap:
            return 0
        kids = [r for r in (fmap[fid].get("from") or []) if r in fmap]
        deeper = max([d(k, seen + (fid,)) for k in kids] or [0])
        return (1 if len(kids) >= 2 else 0) + deeper

    return max(d(i) for i in fmap) + 1


def shapes(findings):
    """(#findings, #statements) in each finding's `from` — the ladder is visible
    as a run of 1F+1S."""
    fmap = {f.get("id"): f for f in findings if isinstance(f, dict) and f.get("id")}
    out = []
    for f in fmap.values():
        refs = f.get("from") or []
        nf = sum(1 for r in refs if r in fmap)
        out.append(f"{nf}F+{len(refs) - nf}S")
    return out


def rejoin(row):
    """Stage 3's kept material, with stage 2's evidence put back by id."""
    p2, p3 = row.get("deepen") or {}, row.get("ask") or {}
    ev_by_id = {s.get("id"): [e for e in (s.get("evidence") or []) if isinstance(e, dict)]
                for s in (p2.get("statements") or []) if isinstance(s, dict)}
    sub_by_handle = {norm(s.get("handle")): s
                     for s in (p2.get("subtopics") or []) if isinstance(s, dict)}

    statements, unknown_ids = [], []
    for s in (p3.get("statements") or []):
        if not isinstance(s, dict):
            continue
        sid = s.get("id")
        if sid not in ev_by_id:
            unknown_ids.append(sid)
            continue
        statements.append({**s, "evidence": ev_by_id[sid]})

    findings = [f for f in (p3.get("findings") or []) if isinstance(f, dict)]

    subtopics, added_subtopics = [], []
    for s in (p3.get("subtopics") or []):
        if not isinstance(s, dict):
            continue
        prev = sub_by_handle.get(norm(s.get("handle")))
        if prev is None:
            added_subtopics.append(s.get("handle"))
            subtopics.append({**s, "material": None, "what_it_shows": None})
        else:
            subtopics.append({"handle": s.get("handle"), "query": s.get("query"),
                              "exposure": s.get("exposure"),
                              "material": prev.get("material"),
                              "what_it_shows": prev.get("what_it_shows")})
    return statements, findings, subtopics, unknown_ids, added_subtopics


def to_legacy(statements, findings):
    """The findings/essentials shape the rubric generator reads.

    Essentials are the statements no finding uses — the ordinary content an
    answer still fails without.
    """
    smap = {s.get("id"): s for s in statements}
    used = {r for f in findings for r in (f.get("from") or [])}

    out_f = []
    for f in findings:
        refs = [smap[r] for r in (f.get("from") or []) if r in smap]
        out_f.append({
            "id": f.get("id"), "from": f.get("from"),
            "observation": " ".join(r.get("claim") or "" for r in refs),
            "analysis": f.get("analysis"), "conclusion": f.get("conclusion"),
            "shallow_miss": f.get("shallow_miss"),
            "evidence": [{"url": e.get("source"), "quote": e.get("quote"),
                          "contributes": r.get("claim")}
                         for r in refs for e in (r.get("evidence") or [])],
        })
    out_e = []
    for sid, s in smap.items():
        if sid in used:
            continue
        ev = s.get("evidence") or []
        out_e.append({"id": sid, "point": s.get("claim"),
                      "source": ev[0].get("source") if ev else None,
                      "sources": [e.get("source") for e in ev],
                      "why_expected": "kept without a finding drawing on it"})
    return out_f, out_e


def signals(row, statements, findings, subtopics, unknown_ids, added_subtopics):
    p1, p2 = row.get("explore") or {}, row.get("deepen") or {}
    handles = {norm(s.get("handle")) for s in subtopics}
    per_sub = Counter(norm(s.get("subtopic")) for s in statements)
    misaligned = [s.get("id") for s in statements if norm(s.get("subtopic")) not in handles]
    doms = sorted({urlparse(e["source"]).netloc.replace("www.", "")
                   for s in statements for e in (s.get("evidence") or [])
                   if isinstance(e, dict) and (e.get("source") or "").startswith("http")})
    centre = (row.get("ask") or {}).get("centre") or {}
    open_gaps = [g for g in (p2.get("gaps") or [])
                 if isinstance(g, dict) and str(g.get("closed")).lower() != "yes"]
    counts = [per_sub.get(h, 0) for h in handles]
    return {
        "statements_per_subtopic": {"median": st.median(counts) if counts else 0,
                                    "min": min(counts) if counts else 0,
                                    "empty": sum(1 for c in counts if c == 0)},
        "subtopic_words": [len((s.get("query") or "").split()) for s in subtopics],
        "material": Counter(s.get("material") for s in subtopics),
        "domains": doms,
        "n_domains": len(doms),
        "finding_shapes": shapes(findings),
        "open_gaps": len(open_gaps),
        "discarded": len(centre.get("discarded") or []),
        "kept": len(centre.get("kept") or []),
        "rounds": row.get("rounds"),
        "subtopics_added_by_deepen": len(p2.get("subtopics") or []) - len(p1.get("subtopics") or []),
        # Stage 3 has no tools. Anything here means it wrote material it could not
        # have checked, which is the one thing that stage is not allowed to do.
        "ask_unknown_statement_ids": unknown_ids,
        "ask_added_subtopics": added_subtopics,
        "statements_off_map": misaligned,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="chain.jsonl from generate_chain.py")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    out = []
    for row in rows:
        p3 = row.get("ask") or {}
        if not p3.get("proposed_question"):
            continue
        stmts, finds, subs, unknown, added = rejoin(row)
        legacy_f, legacy_e = to_legacy(stmts, finds)

        n_sub, n_find = len(subs), len(finds)
        depth = synthesis_depth(finds)
        implicit = sum(1 for s in subs if str(s.get("exposure")).lower() == "implicit")
        share = implicit / n_sub if n_sub else 0.0
        out.append({
            "id": row.get("id"), "topic": row.get("topic"),
            "main_category": row.get("main_category"),
            "keyword": row.get("keyword"), "source_csv": row.get("source_csv"),
            "prompt": p3["proposed_question"],
            "spine": (row.get("deepen") or {}).get("spine"),
            "subtopics": subs,
            "statements": stmts,
            "findings": legacy_f, "essentials": legacy_e,
            "findings_compact": finds,
            "centre": p3.get("centre"),
            "map_change_note": (row.get("deepen") or {}).get("map_change_note"),
            "keyword_verdict": (row.get("explore") or {}).get("keyword_verdict"),
            # Measured, not seeded. The band names are the RR vocabulary so these
            # line up against runs that were seeded with a level.
            "_axes": {
                "conceptual_breadth": n_sub, "conceptual_breadth_band": band(n_sub, BREADTH),
                "analysis_load": n_find, "analysis_load_band": band(n_find, LOAD),
                "logical_nesting": depth, "logical_nesting_band": band(depth, NESTING),
                "exploration": round(share, 3),
                "exploration_band": ("Low" if share < 1 / 3 else
                                     "Medium" if share <= 2 / 3 else "High"),
            },
            "_signals": signals(row, stmts, finds, subs, unknown, added),
        })

    with open(args.output, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"extracted {len(out)}/{len(rows)} -> {args.output}")
    if not out:
        return

    def med(f):
        return st.median([f(r) for r in out])

    print(f"  subtopics        median {med(lambda r: r['_axes']['conceptual_breadth'])}")
    print(f"  statements       median {med(lambda r: len(r['statements']))}")
    print(f"  findings         median {med(lambda r: r['_axes']['analysis_load'])}")
    print(f"  synthesis depth  median {med(lambda r: r['_axes']['logical_nesting'])}"
          f"   (>=2 in {sum(1 for r in out if r['_axes']['logical_nesting'] >= 2)}/{len(out)})")
    words = [w for r in out for w in r['_signals']['subtopic_words']]
    print(f"  subtopic words   median {st.median(words) if words else 0}"
          f"   (propose baseline: 4)")
    per = [r['_signals']['statements_per_subtopic']['median'] for r in out]
    print(f"  statements/sub   median {st.median(per)}   (propose baseline: 2)")
    print(f"  implicit share   median {med(lambda r: r['_axes']['exploration'])}")
    for k, label in (("ask_unknown_statement_ids", "invented statement ids"),
                     ("ask_added_subtopics", "subtopics stage 3 added"),
                     ("statements_off_map", "statements off the map")):
        n = sum(len(r["_signals"][k]) for r in out)
        rowsn = sum(1 for r in out if r["_signals"][k])
        if n:
            print(f"  ! {label}: {n} across {rowsn} questions")
    for name, table in (("conceptual_breadth", BREADTH), ("analysis_load", LOAD),
                        ("logical_nesting", NESTING)):
        c = Counter(r["_axes"][f"{name}_band"] for r in out)
        print(f"  {name:20} {dict(c)}")
    csvs = Counter(r.get("source_csv") for r in out)
    print(f"  seed pools       {dict(csvs)}")


if __name__ == "__main__":
    main()
