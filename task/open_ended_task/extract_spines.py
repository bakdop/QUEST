"""Extract STAGE 1 spines from trajectories, and audit them.

Writes spines.jsonl (one record per spine, plus a `_qc` block) and a QC report on
stdout. A spine that fails is still written, with its failures recorded, so stage
2 can decide what to do rather than have the material silently disappear — the
old pipeline's `centre.discarded` was empty 8/8 for exactly that reason.

The checks map one-to-one onto what stage 1 is for.

  grounded — it was researched, not guessed
    keyword_accounted    the old runs silently welded a dead keyword into the
                         subject (eddie kaspbrak) or silently dropped one
                         (robert irwin); the verdict field makes it visible
    searched_before_spine  the old pipeline chose its subject in round 1
    source_traceable     every cited url must have actually come back from a
                         tool, and every subtopic must say what it shows

  enough to work with
    subtopic_count       >=2
    material_depth       >=3 subtopics whose material is not "thin"

  has tension
    subtopics_differ     soft: if every subtopic shows the same thing there is
                         nothing to weigh. Lexical overlap only, so it warns
                         rather than fails
    tension_not_hedge    "it's more nuanced" is what the model writes when it is
                         guessing rather than reading

Only the current single-`subtopics` schema is read. Two earlier schemas existed
(`frictions[]`, and `parts` + `tension.considerations`) but no run was ever
produced under either, so there is nothing to stay compatible with.
"""
import argparse
import json
import os
import re
from urllib.parse import urlparse

from extract_proposed_qa import coerce_prediction_json

HEDGES = (
    "more complicated than", "more nuanced", "there are trade-offs",
    "there are tradeoffs", "it depends on", "not as simple as",
    "a mixed picture", "varies widely", "no one-size-fits-all",
    "both have pros and cons", "more complex than people",
)
# `unusable` is allowed again, but it now has an entry bar rather than being the
# cheapest exit. On the fixed-seed run the model declared `kristin cabot`
# unusable after 8 searches and ZERO page visits, and `eddie kaspbrak` unusable
# without ever widening from the character to Stephen King's stage adaptations —
# the topic the old pipeline did build a question from. Both were abandonments
# for lack of searching, not judgements that the topic was bad, so an
# `unusable` record is audited *harder* than a normal one: see audit().
VERDICTS = ("kept", "narrowed", "replaced", "unusable")
STOP = set("the a an of to in for and or is are was were be been on at by with "
           "that this it its as from than more most less not no you your which "
           "show shows toward towards push pushes".split())


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def content_words(s):
    """Crudely stemmed content words. Catches near-verbatim paraphrase, not
    semantic paraphrase, so its verdicts are warnings for a human to arbitrate,
    never hard failures."""
    out = set()
    for w in re.findall(r"[a-z0-9$%.]+", norm(s)):
        if w in STOP or len(w) <= 2:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def jaccard(a, b):
    A, B = content_words(a), content_words(b)
    return len(A & B) / len(A | B) if (A or B) else 0.0


def corpus_of(traj):
    return norm(" ".join(
        m.get("content") or "" for m in traj.get("messages", [])
        if m.get("role") == "user"
    ))


def rounds_before_answer(traj):
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


def audit(spine, traj, args):
    fails, warns = [], []

    verdict = norm(spine.get("keyword_verdict"))
    has_spine = bool(norm(spine.get("spine")))
    if verdict not in VERDICTS:
        fails.append(f"keyword_accounted: verdict={verdict!r} not one of {VERDICTS}")
    if verdict == "unusable" and has_spine:
        fails.append("keyword_accounted: verdict is 'unusable' but a spine was given")
    if verdict != "unusable" and not has_spine:
        fails.append("keyword_accounted: no spine was produced")

    s, v = rounds_before_answer(traj)

    if verdict == "unusable":
        # Audited harder than a normal record, not skipped. Abandonment is a
        # claim about the corpus and has to be earned; the old early-return made
        # these the least-scrutinised trajectories in the run.
        if s < args.unusable_min_search or v < args.unusable_min_visit:
            fails.append(f"abandoned_too_early: {s} search / {v} visit rounds before "
                         f"giving up (want >={args.unusable_min_search}/"
                         f"{args.unusable_min_visit})")
        if len(norm(spine.get("keyword_note"))) < args.unusable_min_note:
            fails.append(f"abandoned_too_early: keyword_note is "
                         f"{len(norm(spine.get('keyword_note')))} chars — it must name the "
                         f"directions tried and why each failed")
        return fails, warns, {"unusable": True, "search_rounds": s, "visit_rounds": v}

    if s < args.min_search or v < args.min_visit:
        fails.append(f"searched_before_spine: {s} search / {v} visit rounds "
                     f"(want >={args.min_search}/{args.min_visit})")

    subs = [x for x in (spine.get("subtopics") or []) if isinstance(x, dict)]
    if len(subs) < args.min_subtopics:
        fails.append(f"subtopic_count: {len(subs)} (want >={args.min_subtopics})")

    blob = corpus_of(traj)
    for k, x in enumerate(subs, 1):
        if not norm(x.get("what_it_shows")):
            fails.append(f"source_traceable: subtopic {k} does not say what it shows")
        u = (x.get("source") or "").strip()
        if not u:
            fails.append(f"source_traceable: subtopic {k} cites no source")
        elif norm(u) not in blob:
            fails.append(f"source_traceable: subtopic {k} cites {u[:70]} "
                         f"which never came back from a tool")

    solid = [x for x in subs if norm(x.get("material")) in ("rich", "adequate")]
    if len(solid) < args.min_solid:
        warns.append(f"material_depth: {len(solid)} non-thin subtopic(s) of {len(subs)} "
                     f"(want >={args.min_solid})")

    same = 0
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            sim = jaccard(subs[i].get("what_it_shows"), subs[j].get("what_it_shows"))
            if sim >= args.overlap:
                same += 1
                warns.append(f"subtopics_differ: {i+1} vs {j+1} overlap {sim:.2f} "
                             f"— may be one subtopic reworded")

    tension = norm(spine.get("tension"))
    joined = tension + " " + " ".join(norm(x.get("what_it_shows")) for x in subs)
    hit = [h for h in HEDGES if h in joined]
    if hit:
        fails.append(f"tension_not_hedge: {hit}")
    if not tension:
        fails.append("tension_not_hedge: `tension` is empty")

    doms = sorted({urlparse(x["source"]).netloc.replace("www.", "")
                   for x in subs if (x.get("source") or "").startswith("http")})
    return fails, warns, {"search_rounds": s, "visit_rounds": v,
                          "n_subtopics": len(subs), "solid_subtopics": len(solid),
                          "reworded_pairs": same, "domains": doms}


def keyword_of(traj):
    for m in traj.get("messages", []):
        if m.get("role") == "user" and "Initial Keyword" in (m.get("content") or ""):
            mm = re.search(r"Initial Keyword:\s*(.+)", m["content"])
            if mm:
                return mm.group(1).strip()
    return ""


def main():
    ap = argparse.ArgumentParser(description="Extract and audit STAGE 1 spines")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--min_search", type=int, default=2)
    ap.add_argument("--min_visit", type=int, default=1)
    ap.add_argument("--min_subtopics", type=int, default=2)
    ap.add_argument("--min_solid", type=int, default=3)
    ap.add_argument("--overlap", type=float, default=0.5)
    # abandonment is audited harder than a normal spine, not skipped
    ap.add_argument("--unusable_min_search", type=int, default=3)
    ap.add_argument("--unusable_min_visit", type=int, default=2)
    ap.add_argument("--unusable_min_note", type=int, default=120)
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".json")],
                   key=lambda x: int(re.sub(r"\D", "", x.split("_")[1]) or 0))
    out, unparseable = [], []
    for f in files:
        traj = json.load(open(os.path.join(args.input_dir, f)))
        pred = coerce_prediction_json(traj.get("prediction"))
        if pred is None or "spine" not in pred:
            unparseable.append(f)
            continue
        fails, warns, info = audit(pred, traj, args)
        out.append({
            "id": len(out) + 1,
            "topic": traj.get("subcategory"),
            "keyword": keyword_of(traj),
            "keyword_verdict": pred.get("keyword_verdict"),
            "keyword_note": pred.get("keyword_note"),
            "spine": pred.get("spine"),
            "subtopics": pred.get("subtopics"),
            "tension": pred.get("tension"),
            "trajectory": f,
            "_qc": {"pass": not fails, "failures": fails, "warnings": warns, **info},
        })

    with open(args.output_file, "w") as fh:
        for item in out:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    ok = sum(1 for o in out if o["_qc"]["pass"])
    clean = sum(1 for o in out if o["_qc"]["pass"] and not o["_qc"]["warnings"])
    print(f"\nextracted {len(out)}/{len(files)} spines -> {args.output_file}")
    for f in unparseable:
        print(f"  unparseable: {f}")
    print(f"compliance pass : {ok}/{len(out)}")
    print(f"no warnings too : {clean}/{len(out)}")
    tally = {}
    for o in out:
        for fl in o["_qc"]["failures"] + o["_qc"]["warnings"]:
            tally[fl.split(":")[0]] = tally.get(fl.split(":")[0], 0) + 1
    for k, n in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"  {k:24} {n}")
    print()
    for o in out:
        q = o["_qc"]
        print(f"[{'ok  ' if q['pass'] else 'FAIL'}] #{o['id']} {o['topic']}  "
              f"kw={o['keyword']!r} -> {o['keyword_verdict']}")
        print(f"       spine  : {norm(o['spine'])[:150]}")
        print(f"       {q.get('n_subtopics')} subtopics ({q.get('solid_subtopics')} non-thin) "
              f"| {q.get('search_rounds')}S/{q.get('visit_rounds')}V before answer "
              f"| {len(q.get('domains') or [])} domains")
        for k, x in enumerate(o.get("subtopics") or [], 1):
            print(f"         {k}. [{x.get('material')}] {norm(x.get('subtopic'))[:70]}")
            print(f"            -> {norm(x.get('what_it_shows'))[:88]}")
        print(f"       tension: {norm(o.get('tension'))[:150]}")
        for fl in q["failures"]:
            print(f"       !! {fl}")
        for w in q["warnings"]:
            print(f"       ~  {w}")


if __name__ == "__main__":
    main()
