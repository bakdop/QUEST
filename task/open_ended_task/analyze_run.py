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


ASK_RE = re.compile(
    r"\b(what|how|whether|which|why|when|should I|can you|please"
    r"|I need to (?:know|understand)|I(?:'d| would) like to (?:know|understand)"
    r"|I want to know)\b", re.I)

# Words that open a sentence or are grammatically capitalised, so they say nothing
# about whether the subject was actually pinned to a named thing.
_NOT_A_SPECIFIC = {
    "I", "The", "A", "An", "In", "What", "How", "Write", "Create", "Analyze",
    "Analyse", "Using", "Include", "Your", "This", "It", "Also", "For", "As",
    "My", "We", "Some", "Provide", "Compose", "Examine", "Explain", "Compare",
    "Generate", "Find", "Help", "Propose", "Make", "Prove", "Critically",
    "Develop", "Design", "Give", "Please", "Assume", "If", "When", "Based",
    "Consider", "Start", "Then", "Add", "Use", "Each", "Do", "Be", "Focus",
}


def pinned_tokens(text):
    """Named entities and years in a question - how hard its subject is pinned.

    A question that names nothing specific cannot make a finding obligatory: any
    competent essay on the general topic satisfies it. ResearchRubrics questions
    carry a median of five.
    """
    toks = {t for t in re.findall(r"\b[A-Z][a-zA-Z0-9&.\-]{1,}\b", text or "")
            if t not in _NOT_A_SPECIFIC}
    return toks | set(re.findall(r"\b(?:19|20)\d{2}\b", text or ""))


def arithmetic_errors(text, tol=0.02):
    """Verify 'a / b = c' claims written into an analysis field.

    A derivation whose sum is wrong has no value at all, and the grounding check
    cannot see it: every figure involved is present in the corpus, only the
    operation on them is wrong.

    This catches miscomputation only. It does NOT catch the mispairing seen in
    findings8_15267682, where "1,558/309 = 5.04" divides an unfunded-patent count
    by a funded-publication count: the division is arithmetically correct, the
    operands are the wrong two numbers. Nothing short of reading the observation
    catches that.
    """
    bad = []
    for m in re.finditer(r"([\d,]+(?:\.\d+)?)\s*/\s*([\d,]+(?:\.\d+)?)\s*=\s*([\d,]+(?:\.\d+)?)",
                         text or ""):
        try:
            a, b, c = (float(g.replace(",", "")) for g in m.groups())
        except ValueError:
            continue
        if b == 0:
            continue
        got = a / b
        if abs(got - c) > max(tol * abs(c), tol):
            bad.append(f"{m.group(0)} (actually {got:.2f})")
    return bad


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


def restatement_report(findings):
    """Is the conclusion doing work the observation did not already do?

    Across two runs, 0 of 24 findings tagged ADJUDICATE named which conflicting
    source wins or why — the conclusion restated the observation with the word
    "conflicting" attached. Splitting the fields makes that measurable: a
    conclusion whose content words are a subset of the observation's added nothing.
    """
    split = [f for f in findings if f.get("observation") and f.get("conclusion")]
    if not split:
        print("  observation/conclusion: not split (pre-split schema)")
        return
    thin = []
    for f in split:
        obs = set(re.findall(r"[a-z]{4,}", norm(f["observation"])))
        con = set(re.findall(r"[a-z]{4,}", norm(f["conclusion"])))
        new = con - obs
        if len(new) <= 3:
            thin.append((f.get("id"), sorted(new)))
    substantive = len(split) - len(thin)
    print(f"  conclusion adds nothing new  {len(thin)}/{len(split)} "
          f"(<=3 content words absent from its own observation)")
    print(f"  substantive conclusions      {substantive}   "
          f"-> analysis load {'High' if substantive >= 3 else 'Medium' if substantive else 'Low'} "
          f"by the prompt's own definition")
    for fid, new in thin[:4]:
        print(f"    {fid}: new words = {new}")

    # ADJUDICATE has to name a winner; the others have their own tells, but this
    # is the one that failed outright, so track it explicitly.
    adj = [f for f in split if f.get("operation") == "ADJUDICATE"]
    if adj:
        VERDICT = re.compile(r"\b(more reliable|authoritative|supersede|takes precedence|"
                             r"should be used|is the relevant|is correct|applies to|"
                             r"methodolog|prefer|rather than the)\b", re.I)
        ok = sum(1 for f in adj if VERDICT.search(f["conclusion"]))
        print(f"  ADJUDICATE names a winner    {ok}/{len(adj)}")


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
                              f.get("conclusion") or f.get("claim") or "")
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
    loads = Counter(q.get("analysis_load") for q in qa if q.get("analysis_load"))
    if loads:
        print(f"analysis load    {dict(loads)}")

    findings = [f for q in qa for f in (q.get("findings") or []) if isinstance(f, dict)]
    if findings:
        # `operation` is no longer a recorded field - the analysis text describes
        # the move instead. Old runs still carry it, so only flag when present.
        bad_ops = Counter(f.get("operation") for f in findings
                          if f.get("operation") and f.get("operation") not in OPERATIONS)

        def n_sources(f):
            ev = f.get("evidence")
            if isinstance(ev, list):
                return len({e.get("url") for e in ev if isinstance(e, dict) and e.get("url")})
            return len({u for u in (f.get("sources") or [])})

        counts = [len(q.get('findings') or []) for q in qa]
        print(f"\nfindings {len(findings)} "
              f"({st.mean(counts):.1f}/question, range {min(counts)}-{max(counts)})")
        print(f"  >=2 distinct sources  {sum(1 for f in findings if n_sources(f) >= 2)}/{len(findings)}")
        dup = sum(1 for f in findings
                  if isinstance(f.get("evidence"), list)
                  and len(f["evidence"]) > n_sources(f))
        if dup:
            print(f"  DUPLICATE evidence URL  {dup}/{len(findings)} "
                  f"(two quotes from one page is one source)")
        print(f"  has shallow_miss      {sum(1 for f in findings if f.get('shallow_miss'))}/{len(findings)}")
        if bad_ops:
            print(f"  INVALID operation     {dict(bad_ops)}")
        arith = [(f.get("id", "?"), e)
                 for f in findings for e in arithmetic_errors(f.get("analysis"))]
        print(f"  ARITHMETIC errors     {len(arith)}"
              f"{' (a/b=c checked in analysis)' if not arith else ''}")
        for fid, e in arith[:8]:
            print(f"    {fid}: {e}")
        restatement_report(findings)
        grounding_report(run, qa)

    centres = [q.get("centre") for q in qa if isinstance(q.get("centre"), dict)]
    if centres:
        kept = sum(len(c.get("kept") or []) for c in centres)
        disc = sum(len(c.get("discarded") or []) for c in centres)
        none_disc = sum(1 for c in centres if not (c.get("discarded") or []))
        print(f"\ncentre {len(centres)}/{len(qa)} recorded  kept {kept}  discarded {disc}"
              f"  ({none_disc} questions discarded nothing)")

    ess = [e for q in qa for e in (q.get("essentials") or []) if isinstance(e, dict)]
    if ess:
        print(f"\nessentials {len(ess)} ({len(ess)/max(len(qa),1):.1f}/question)"
              f"  with why_expected {sum(1 for e in ess if e.get('why_expected'))}/{len(ess)}")
    elif findings:
        print("\nessentials: none recorded")

    # --- leakage: finding content restated in the question ---
    fig_q, name_q, per_q = 0, 0, []
    for q in qa:
        figs, names = set(), set()
        for f in (q.get("findings") or []):
            if isinstance(f, dict):
                a, b = leak_tokens(" ".join(
                    str(f.get(k) or "") for k in
                    ("observation", "analysis", "conclusion", "claim", "derivation")))
                figs |= a
                names |= b
        prompt = norm(q.get("prompt"))
        # "Aug. 18" in a finding and "august 18" in the question are the same
        # leak; strip punctuation before comparing.
        flat = re.sub(r"[.,$£€%]", "", prompt)
        fhit = sorted({t for t in figs
                       if norm(t) in prompt or re.sub(r"[.,$£€%]", "", norm(t)) in flat})
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

    # The first findings-first run wrote questions that asked for each finding
    # separately - 7 asks against 3 kept findings - which hands the answerer a
    # checklist without leaking a single figure. Ask density is what catches it.
    asks = [len(ASK_RE.findall(q.get("prompt") or "")) for q in qa]
    marks = [(q.get("prompt") or "").count("?") for q in qa]
    if asks:
        print(f"ask density      median {int(st.median(asks))} asks "
              f"(ResearchRubrics median 1)   range {min(asks)}-{max(asks)}")
        print(f"                 median {int(st.median(marks))} question marks "
              f"(ResearchRubrics median 0)")
        loud = [(q.get("id", i), a) for i, (q, a) in enumerate(zip(qa, asks)) if a > 3]
        for qid, a in loud[:6]:
            print(f"    id={qid}: {a} asks")

    # A question that never pins its subject cannot make findings obligatory.
    specifics = [len(pinned_tokens(q.get("prompt") or "")) for q in qa]
    if specifics:
        print(f"subject pinning  median {int(st.median(specifics))} named specifics "
              f"(ResearchRubrics median 5)   range {min(specifics)}-{max(specifics)}")
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
            imp = [i for i in items if i.get("axis") == "Implicit Criteria"]
            digit = sum(1 for i in imp if re.search(r"\d", i.get("criterion", "")))
            print(f"\nrubric items {len(items)} ({len(items)/len(cr):.1f}/question), "
                  f"{sum(1 for i in items if (i.get('weight') or 0) < 0)} negative")
            if imp:
                print(f"  Implicit items hinging on a figure  {digit}/{len(imp)} "
                      f"({100*digit/len(imp):.0f}%; ResearchRubrics 29%)")
            for a, v in ax.most_common():
                print(f"  {a:>30} {v:>4}  {100*v/tot:>5.1f}%")


if __name__ == "__main__":
    for run in sys.argv[1:] or ["outputs/findings10_15230324"]:
        analyse(run)
