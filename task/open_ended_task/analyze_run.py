"""Compare findings-first runs on the things that actually went wrong before.

  python analyze_run.py outputs/findings10_15230324 [outputs/other_run ...]

The first audit of this pipeline found the failure mode was not shallow search —
search depth tripled — but leakage: six of eight questions restated their own
findings, turning content the answerer was meant to discover into a checklist.
That is invisible in the structural counts, so measure it directly and track it
across runs.
"""
import json
import os
import re
import statistics as st
import sys
from collections import Counter

OPERATIONS = {
    "ADJUDICATE", "NORMALIZE", "DERIVE", "ATTRIBUTE", "QUANTIFY_IMPACT",
    "EXTRAPOLATE", "CHECK_FEASIBILITY", "FRAME_METRIC", "BOUND", "SEGMENT",
    "COUNTERFACTUAL", "RULE_OUT",
}
# Operations whose whole purpose is to produce a figure no source states, so a
# number missing from the corpus is expected rather than suspicious.
COMPUTES_NEW_FIGURES = {"DERIVE", "NORMALIZE", "QUANTIFY_IMPACT", "EXTRAPOLATE", "SEGMENT"}

# Register tells: phrases absent from all 101 ResearchRubrics prompts that the
# generator reaches for when it slips into report-brief voice.
BLEED = ["structural", "aggregate", "affects the interpretation",
         "specific numbers", "not just the headline", "comprehensive analysis covering",
         "adjudicat", "normaliz", "trade-off analysis"]


def leak_tokens(claim):
    """Split into two buckets, because they mean different things.

    A figure or date from a finding appearing in the question is unambiguous
    leakage — the asker could not have known it. A proper noun is weaker
    evidence: the subject of the question necessarily appears in its findings
    too, so "SpaceX" in a question about SpaceX is not a leak. Report figures as
    the headline number and names separately as a softer signal.
    """
    figures, names = set(), set()
    for m in re.finditer(r'\b\d[\d,]*\.?\d*\s*%|\b(?:19|20)\d{2}\b|[$£€]\s?[\d,]+(?:\.\d+)?'
                         r'|\b\d[\d,]*\.?\d*\s*(?:million|billion|percent)\b', claim):
        figures.add(m.group(0).strip())
    for m in re.finditer(r'\b([A-Z][A-Za-z0-9&.\-]{2,}(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,3})\b', claim):
        w = m.group(1).strip()
        if w.lower() not in {"the", "this", "these", "that"} and len(w) > 3:
            names.add(w)
    return figures, names


def norm(s):
    return re.sub(r"\s+", " ", s or "").lower()


def _corpus_by_iteration(run):
    """Tool-response text per trajectory, keyed by iteration_id."""
    tdir = os.path.join(run, "trajectories")
    out = {}
    if not os.path.isdir(tdir):
        return out
    for name in os.listdir(tdir):
        if not name.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(tdir, name)))
        except (OSError, json.JSONDecodeError):
            continue
        text = " ".join((m.get("content") or "") for m in d.get("messages", [])
                        if m.get("role") == "user")
        out[d.get("iteration_id")] = norm(text)
    return out


def grounding_report(run, qa):
    """Do the figures in each claim actually appear in what was retrieved?

    An audit of the first run found 3 of 39 findings carried numbers that existed
    nowhere in the trajectory, and one cited an outlet that was never fetched. A
    fabricated figure becomes a grading criterion demanding a false fact, so it is
    the most damaging defect the pipeline can emit — and the cheapest to detect.

    Rows are matched to trajectories positionally: extract_proposed_qa.py numbers
    rows in sorted-filename order, so row N is the Nth trajectory by that order.
    """
    corp = _corpus_by_iteration(run)
    if not corp:
        return
    iters = sorted(corp)
    ungrounded, quoted_ok, quoted_tot, checked = [], 0, 0, 0
    for pos, q in enumerate(qa):
        text = corp.get(iters[pos]) if pos < len(iters) else None
        if text is None:
            continue
        for f in (q.get("findings") or []):
            if not isinstance(f, dict):
                continue
            # DERIVE and friends exist precisely to produce a number no source
            # states, so a figure missing from the corpus is expected there and
            # proves nothing. Only quoting operations are checked this way.
            if f.get("operation") in COMPUTES_NEW_FIGURES:
                continue
            checked += 1
            figs = re.findall(r'\b\d[\d,]*\.?\d*\s*%|[$£€]\s?[\d,]+(?:\.\d+)?'
                              r'|\b\d[\d,]*\.?\d*\s*(?:million|billion)\b',
                              f.get("claim") or "")
            missing = [x for x in figs if norm(x) not in text
                       and norm(x.replace(",", "")) not in text]
            if missing:
                ungrounded.append((q.get("id"), f.get("id"), missing[:4]))
    for q in qa:
        for f in (q.get("findings") or []):
            if not isinstance(f, dict):
                continue
            for e in (f.get("evidence") or []):
                if isinstance(e, dict) and e.get("quote"):
                    quoted_tot += 1
    if checked:
        print(f"  figures absent from corpus  {len(ungrounded)}/{checked} "
              f"(quoting operations only; heuristic)")
        for qid, fid, miss in ungrounded[:6]:
            print(f"    id={qid} {fid}: {', '.join(miss)}")
    # The reliable check. Once findings carry verbatim quotes, a quote that is
    # not in the corpus is fabrication outright, with no false positives from
    # figures the model legitimately computed.
    if quoted_tot:
        corp_all = " ".join(corp.values())
        ok = 0
        for q in qa:
            for f in (q.get("findings") or []):
                if not isinstance(f, dict):
                    continue
                for e in (f.get("evidence") or []):
                    if isinstance(e, dict) and e.get("quote"):
                        frag = norm(e["quote"])[:60]
                        if len(frag) > 20 and frag in corp_all:
                            ok += 1
        print(f"  evidence quotes found verbatim  {ok}/{quoted_tot}")
    else:
        print("  evidence quotes: none recorded (pre-quote schema)")


def analyse(run):
    qa_path = os.path.join(run, "proposed_qa.jsonl")
    if not os.path.exists(qa_path):
        print(f"{run}: no proposed_qa.jsonl")
        return
    qa = [json.loads(l) for l in open(qa_path)]
    cr_path = os.path.join(run, "criteria.jsonl")
    cr = [json.loads(l) for l in open(cr_path)] if os.path.exists(cr_path) else []

    n_traj = len([f for f in os.listdir(os.path.join(run, "trajectories"))
                  if f.endswith(".json")]) if os.path.isdir(os.path.join(run, "trajectories")) else 0

    print(f"\n{'='*72}\n{run}\n{'='*72}")
    print(f"trajectories {n_traj} -> questions {len(qa)} -> rubrics {len(cr)}")

    findings = [f for q in qa for f in (q.get("findings") or []) if isinstance(f, dict)]
    if findings:
        bad_ops = Counter(f.get("operation") for f in findings
                          if f.get("operation") not in OPERATIONS)

        def n_sources(f):
            ev = f.get("evidence")
            if isinstance(ev, list):
                return len({e.get("url") for e in ev if isinstance(e, dict) and e.get("url")})
            return len({u for u in (f.get("sources") or [])})

        print(f"\nfindings {len(findings)} "
              f"({st.mean(len(q.get('findings') or []) for q in qa):.1f}/question)")
        print(f"  >=2 distinct sources  {sum(1 for f in findings if n_sources(f) >= 2)}/{len(findings)}")
        print(f"  has shallow_miss      {sum(1 for f in findings if f.get('shallow_miss'))}/{len(findings)}")
        print(f"  has no_single_source  {sum(1 for f in findings if f.get('no_single_source'))}/{len(findings)}")
        if bad_ops:
            print(f"  INVALID operation     {dict(bad_ops)}")
        grounding_report(run, qa)

    # --- leakage: finding content restated in the question ---
    fig_q, name_q, per_q = 0, 0, []
    for q in qa:
        figs, names = set(), set()
        for f in (q.get("findings") or []):
            if isinstance(f, dict):
                a, b = leak_tokens(f.get("claim") or "")
                figs |= a
                names |= b
        prompt = norm(q.get("prompt"))
        fhit = sorted({t for t in figs if norm(t) in prompt})
        nhit = sorted({t for t in names if norm(t) in prompt})
        per_q.append((q.get("id"), fhit, nhit))
        fig_q += bool(fhit)
        name_q += bool(nhit)
    if qa:
        print(f"\nLEAKAGE (figures/dates — unambiguous)  {fig_q}/{len(qa)} questions")
        for qid, fhit, _ in per_q:
            if fhit:
                print(f"  id={qid}: {', '.join(fhit[:8])}")
        print(f"leakage (proper nouns — softer; subject of the question counts here) "
              f"{name_q}/{len(qa)}")

    words = [len((q.get("prompt") or "").split()) for q in qa]
    if words:
        print(f"\nquestion length  median {int(st.median(words))} words "
              f"(ResearchRubrics median 68)   range {min(words)}-{max(words)}")
    bleeds = Counter()
    for q in qa:
        for b in BLEED:
            if b in norm(q.get("prompt")):
                bleeds[b] += 1
    if bleeds:
        print(f"register bleed   {dict(bleeds)}")

    if cr:
        items = [i for c in cr for i in (c.get("rubrics") or [])]
        if items:
            ax = Counter(i.get("axis") for i in items)
            tot = sum(ax.values())
            print(f"\nrubric items {len(items)} ({len(items)/len(cr):.1f}/question), "
                  f"{sum(1 for i in items if (i.get('weight') or 0) < 0)} negative")
            for a, v in ax.most_common():
                print(f"  {a:>30} {v:>4}  {100*v/tot:>5.1f}%")


if __name__ == "__main__":
    for run in sys.argv[1:] or ["outputs/findings10_15230324"]:
        analyse(run)
