"""Extract questions from PROMPT_VARIANT=propose trajectories.

Writes proposed_qa.jsonl in the shape longform_rubric/generate_criteria_findings.py
already reads: `findings` expanded from the compact `from` form, and `essentials`
synthesised as the kept statements no kept finding uses.

No pass/fail gate — everything that parses is kept. The per-row `_stats` block is
descriptive only (how many statements and findings, how deep the longest chain
runs, how many domains), so a run can be characterised without anything being
filtered on it.
"""
import argparse
import json
import os
import re
from urllib.parse import urlparse

from extract_proposed_qa import coerce_prediction_json

VERDICTS = ("kept", "narrowed", "replaced")
BREADTH = {"simple": (2, 2), "moderate": (3, 5), "high": (6, 99)}
NESTING = {"shallow": (1, 1), "intermediate": (2, 3), "deep": (4, 99)}
LOAD = {"low": (1, 1), "medium": (2, 3), "high": (4, 99)}
STOP = set("the a an of to in for and or is are was were be been on at by with that "
           "this it its as from than more most less not no you your which".split())


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def words(s):
    return {w for w in re.findall(r"[a-z0-9$%.]+", norm(s)) if w not in STOP and len(w) > 2}


def jaccard(a, b):
    A, B = words(a), words(b)
    return len(A & B) / len(A | B) if (A or B) else 0.0


def corpus_of(traj):
    return norm(" ".join(m.get("content") or "" for m in traj.get("messages", [])
                         if m.get("role") == "user"))


def figures_and_names(s):
    """The tokens that would constitute a leak if they reached the question."""
    out = set()
    for m in re.finditer(r'\b\d[\d,]*\.?\d*\s*%|\b(?:19|20)\d{2}\b|[$£€]\s?[\d,]+(?:\.\d+)?'
                         r'|\b\d[\d,]*\.?\d*\s*(?:million|billion|percent)\b', s or ""):
        out.add(m.group(0).strip())
    for m in re.finditer(r'\b[A-Z][A-Za-z0-9&.\-]{3,}(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,2}\b', s or ""):
        out.add(m.group(0).strip())
    return out


def ask_count(q):
    """Coordinate asks in the question. The enumerating failure shows up as a
    list of dimensions joined by commas after 'including'/'accounting for', or as
    numbered sub-tasks."""
    q = q or ""
    n = 1
    n += len(re.findall(r'\(\s*\d\s*\)|\b\d\.\s+[A-Z]', q))          # numbered sub-tasks
    for m in re.finditer(r'\b(including|accounting for|addressing|covering|analyz\w+|'
                         r'examining|considering)\b(.{0,400})', q, re.I):
        tail = m.group(2)
        cut = re.split(r'(?<=[.;])\s', tail)[0]
        n += cut.count(",") + len(re.findall(r'\band\b', cut, re.I))
    return n


def levels(subs, findings, kept):
    """(breadth, nesting, load) counted off the kept structures."""
    fmap = {f.get("id"): f for f in findings if isinstance(f, dict)}
    kept_f = [i for i in kept if i in fmap] or list(fmap)
    kept_subs = {norm(s.get("subtopic")) for s in subs if isinstance(s, dict)}

    def depth(fid, seen=()):
        if fid in seen or fid not in fmap:
            return 0
        refs = [r for r in (fmap[fid].get("from") or []) if r in fmap]
        return 1 + max([depth(r, seen + (fid,)) for r in refs] or [0])

    return len(kept_subs), max([depth(f) for f in kept_f] or [0]), len(kept_f)


def band(n, table):
    for k, (lo, hi) in table.items():
        if lo <= n <= hi:
            return k
    return None


def stats(p):
    """Descriptive counts only."""
    subs = [x for x in (p.get("subtopics") or []) if isinstance(x, dict)]
    stmts = [x for x in (p.get("statements") or []) if isinstance(x, dict)]
    finds = [f for f in (p.get("findings") or []) if isinstance(f, dict)]
    fmap = {f.get("id"): f for f in finds}
    kept = list((p.get("centre") or {}).get("kept") or []) or list(fmap)

    def depth(fid, seen=()):
        if fid in seen or fid not in fmap:
            return 0
        refs = [r for r in (fmap[fid].get("from") or []) if r in fmap]
        return 1 + max([depth(r, seen + (fid,)) for r in refs] or [0])

    doms = sorted({urlparse(e["source"]).netloc.replace("www.", "")
                   for x in stmts for e in (x.get("evidence") or [])
                   if isinstance(e, dict) and (e.get("source") or "").startswith("http")})
    return {"n_subtopics": len(subs), "n_statements": len(stmts), "n_findings": len(finds),
            "chain_depth": max([depth(f) for f in fmap] or [0]),
            "n_kept": len([k for k in kept if k in fmap]),
            "asks": ask_count(p.get("proposed_question")),
            "n_domains": len(doms), "domains": doms}


def to_legacy(p):
    """The findings/essentials shape the rubric generator reads. Essentials are
    the kept statements no kept finding uses — the ordinary content."""
    smap = {s.get("id"): s for s in (p.get("statements") or []) if isinstance(s, dict)}
    finds = [f for f in (p.get("findings") or []) if isinstance(f, dict)]
    kept = set((p.get("centre") or {}).get("kept") or [])
    used = {r for f in finds if f.get("id") in kept for r in (f.get("from") or [])}
    out_f = []
    for f in finds:
        refs = [smap[r] for r in (f.get("from") or []) if r in smap]
        out_f.append({**f,
                      "observation": " ".join(r.get("claim") or "" for r in refs),
                      "evidence": [{"url": e.get("source"), "quote": e.get("quote"),
                                    "contributes": r.get("claim")}
                                   for r in refs for e in (r.get("evidence") or [])
                                   if isinstance(e, dict)]})
    out_e = []
    for sid in kept:
        s = smap.get(sid)
        if not s or sid in used:
            continue
        ev = [e for e in (s.get("evidence") or []) if isinstance(e, dict)]
        out_e.append({"id": sid, "point": s.get("claim"),
                      "source": ev[0].get("source") if ev else None,
                      "why_expected": "kept in the centre without a finding using it"})
    return out_f, out_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json"))
    out, unparseable = [], []
    for f in files:
        traj = json.load(open(os.path.join(args.input_dir, f)))
        p = coerce_prediction_json(traj.get("prediction"))
        if not p or not p.get("proposed_question"):
            unparseable.append(f)
            continue
        lf, le = to_legacy(p)
        out.append({
            "id": len(out) + 1, "topic": traj.get("subcategory"),
            "prompt": p["proposed_question"],
            "spine": p.get("spine"), "subtopics": p.get("subtopics"),
            "tension": p.get("tension"), "statements": p.get("statements"),
            "findings": lf, "essentials": le, "findings_compact": p.get("findings"),
            "centre": p.get("centre"),
            "conceptual_breadth": p.get("conceptual_breadth"),
            "logical_nesting": p.get("logical_nesting"),
            "exploration": p.get("exploration"),
            "analysis_load": p.get("analysis_load"),
            "level_note": p.get("level_note"),
            "keyword_verdict": p.get("keyword_verdict"),
            "trajectory": f,
            "_stats": stats(p),
        })

    with open(args.output_file, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nextracted {len(out)}/{len(files)} -> {args.output_file}")
    if unparseable:
        print(f"  unparseable: {len(unparseable)}")
        for f in unparseable[:5]:
            print(f"    {f}")
    if out:
        import statistics as st
        for k in ("n_subtopics", "n_statements", "n_findings", "chain_depth",
                  "n_kept", "asks", "n_domains"):
            v = [r["_stats"][k] for r in out]
            print(f"  {k:14} median {st.median(v):>5}  range {min(v)}-{max(v)}")


if __name__ == "__main__":
    main()
