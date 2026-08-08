"""Extract questions from PROMPT_VARIANT=propose trajectories, and audit them.

Writes proposed_qa.jsonl (in the shape longform_rubric/generate_criteria_findings.py
already reads) plus a `_qc` block per row, and a summary on stdout. Rows that fail
are still written — stage-downstream decides what to drop, rather than material
disappearing silently, which is how the old pipeline ended up with an empty
`centre.discarded` in 8/8 runs.

Every check here exists because a real run failed it:

  grounded          a source url that never came back from a tool (#3 of
                    spinefix8_15469596 cited wttt.org after visiting wttc.org);
                    a quote that is not in the corpus
  statements_atomic two statements making the same claim, which would let a
                    finding cite "two sources" without having combined anything
  findings_derived  a finding whose `from` has fewer than two distinct claims, or
                    whose conclusion is already sitting in a statement it cites
  ids_resolve       `from` / `centre` referencing ids that do not exist
  chain_kept        a finding kept while something it stands on was discarded
  question_one_ask  the enumerating failure — 4 of the old pipeline's 8 questions
                    listed their own subtopics
  question_no_leak  a figure or name from a finding's analysis/conclusion
                    appearing in the question
  levels_match      the four declared levels against the levels counted off the
                    structures
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


def audit(p, traj):
    fails, warns = [], []
    blob = corpus_of(traj)

    if norm(p.get("keyword_verdict")) not in VERDICTS:
        fails.append(f"keyword_accounted: {p.get('keyword_verdict')!r}")
    if not norm(p.get("spine")):
        fails.append("keyword_accounted: no spine")

    subs = [s for s in (p.get("subtopics") or []) if isinstance(s, dict)]
    sub_names = {norm(s.get("subtopic")) for s in subs}
    if len(subs) < 2:
        fails.append(f"subtopic_count: {len(subs)}")

    stmts = [s for s in (p.get("statements") or []) if isinstance(s, dict)]
    smap = {s.get("id"): s for s in stmts}
    for s in stmts:
        sid = s.get("id")
        if sub_names and norm(s.get("subtopic")) not in sub_names:
            warns.append(f"subtopic_unlisted: {sid} -> {s.get('subtopic')!r}")
        ev = [e for e in (s.get("evidence") or []) if isinstance(e, dict)]
        if not ev:
            fails.append(f"grounded: {sid} has no evidence")
        for e in ev:
            u = (e.get("source") or "").strip()
            if not u or norm(u) not in blob:
                fails.append(f"grounded: {sid} cites {u[:60]!r} which never came back from a tool")
            q = norm(e.get("quote"))
            if q and q not in blob:
                fails.append(f"grounded: {sid} quote not in corpus — {q[:50]!r}")
    for i in range(len(stmts)):
        for j in range(i + 1, len(stmts)):
            if jaccard(stmts[i].get("claim"), stmts[j].get("claim")) >= 0.7:
                warns.append(f"statements_atomic: {stmts[i].get('id')} ~ {stmts[j].get('id')}")

    finds = [f for f in (p.get("findings") or []) if isinstance(f, dict)]
    fmap = {f.get("id"): f for f in finds}
    for f in finds:
        fid = f.get("id")
        refs = f.get("from") or []
        missing = [r for r in refs if r not in smap and r not in fmap]
        if missing:
            fails.append(f"ids_resolve: {fid} from {missing}")
        claims = [smap[r].get("claim") for r in refs if r in smap]
        claims += [fmap[r].get("conclusion") for r in refs if r in fmap]
        distinct = [c for k, c in enumerate(claims)
                    if all(jaccard(c, claims[m]) < 0.7 for m in range(k))]
        if len(distinct) < 2:
            fails.append(f"findings_derived: {fid} has {len(distinct)} distinct input claim(s)")
        concl = norm(f.get("conclusion"))
        for r in refs:
            if r in smap and concl and jaccard(concl, smap[r].get("claim")) >= 0.7:
                fails.append(f"findings_derived: {fid} conclusion restates {r}")
        if not norm(f.get("analysis")):
            fails.append(f"findings_derived: {fid} has no analysis")
        if not norm(f.get("shallow_miss")):
            warns.append(f"findings_derived: {fid} has no shallow_miss")
    for i in range(len(finds)):
        for j in range(i + 1, len(finds)):
            if jaccard(finds[i].get("shallow_miss"), finds[j].get("shallow_miss")) >= 0.6:
                warns.append(f"findings_independent: {finds[i].get('id')} ~ {finds[j].get('id')}")

    centre = p.get("centre") if isinstance(p.get("centre"), dict) else {}
    kept = [k for k in (centre.get("kept") or [])]
    disc = set(centre.get("discarded") or [])
    for k in kept + list(disc):
        if k not in smap and k not in fmap:
            fails.append(f"ids_resolve: centre references {k}")
    for k in kept:
        for r in (fmap.get(k, {}).get("from") or []):
            if r in disc:
                fails.append(f"chain_kept: {k} kept but {r} discarded")
            elif r not in kept:
                warns.append(f"chain_kept: {k} kept but {r} not in kept")

    q = p.get("proposed_question") or ""
    if not q.strip():
        fails.append("question_one_ask: empty question")
    n_ask = ask_count(q)
    if n_ask >= 4:
        fails.append(f"question_one_ask: ~{n_ask} coordinate asks")
    elif n_ask == 3:
        warns.append(f"question_one_ask: ~{n_ask} coordinate asks")
    leaked = set()
    for f in finds:
        if f.get("id") not in kept and kept:
            continue
        for tok in figures_and_names(f"{f.get('analysis')} {f.get('conclusion')}"):
            if len(tok) > 3 and norm(tok) in norm(q):
                leaked.add(tok)
    if leaked:
        warns.append(f"question_no_leak: {sorted(leaked)[:6]}")

    b, n, l = levels(subs, finds, kept)
    got = {"conceptual_breadth": band(b, BREADTH), "logical_nesting": band(n, NESTING),
           "analysis_load": band(l, LOAD)}
    for k, v in got.items():
        said = norm(p.get(k))
        if v and said and said != v:
            warns.append(f"levels_match: {k} says {said} but counts {v}")

    return fails, warns, {"n_subtopics": b, "n_statements": len(stmts), "n_findings": len(finds),
                          "chain_depth": n, "n_kept": len(kept), "n_discarded": len(disc),
                          "asks": n_ask, "counted": got,
                          "domains": sorted({urlparse(e["source"]).netloc.replace("www.", "")
                                             for s in stmts for e in (s.get("evidence") or [])
                                             if isinstance(e, dict) and (e.get("source") or "").startswith("http")})}


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
        fails, warns, info = audit(p, traj)
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
            "_qc": {"pass": not fails, "failures": fails, "warnings": warns, **info},
        })

    with open(args.output_file, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = sum(1 for r in out if r["_qc"]["pass"])
    print(f"\nextracted {len(out)}/{len(files)} -> {args.output_file}")
    print(f"  unparseable: {len(unparseable)}")
    print(f"  QC pass    : {ok}/{len(out)}")
    tally = {}
    for r in out:
        for x in r["_qc"]["failures"] + r["_qc"]["warnings"]:
            tally[x.split(":")[0]] = tally.get(x.split(":")[0], 0) + 1
    for k, n in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"    {k:24} {n}")
    if out:
        import statistics as st
        for k in ("n_subtopics", "n_statements", "n_findings", "chain_depth", "asks"):
            v = [r["_qc"][k] for r in out]
            print(f"  {k:14} median {st.median(v):>4}  range {min(v)}-{max(v)}")
        dom = [len(r["_qc"]["domains"]) for r in out]
        print(f"  {'domains':14} median {st.median(dom):>4}  range {min(dom)}-{max(dom)}")


if __name__ == "__main__":
    main()
