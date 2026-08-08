"""PROMPT_VARIANT=evidence_first trajectories -> proposed_qa.jsonl.

    python extract_evidence.py --input_dir outputs/<tag>/trajectories \
                               --output_file outputs/<tag>/proposed_qa.jsonl

There is one kind of object upstream now — a statement, with a verbatim quote and
the url that returned it — so the two things this has to do are shape it for the
rubric generator and measure the investigation.

MEASURE. Nothing is injected upstream, so every axis is counted here. Depth is
read off `key_queries`, where each recorded search names the statements whose
reading sent the run looking for it and the statements it produced: the chain is
statement -> query -> statement, and its length is how far the investigation had
to go. A single-parent step counts, unlike in extract_chain.py, where the merge
requirement existed to stop an inference ladder scoring as depth. It does not
apply here — every link is a retrieval carrying its own quote and url, so the
chain cannot be padded without actually going and looking.

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


# ---------------------------------------------------------------- question ---
STOP = set("the a an of to in for and or is are was were be been on at by with that "
           "this it its as from than more most less not no you your which what how "
           "when where why who vs about into over under between across per new".split())


def content_words(s):
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", norm(s))
            if w not in STOP and len(w) > 2}


def leak_tokens(claim):
    """Figures and proper nouns in a claim, as two buckets.

    Lifted from analyze_run.py, which is imported by nothing. A figure or date
    turning up in the question is unambiguous leakage — the asker could not have
    known it. A proper noun is weaker: the subject of a question about SpaceX
    necessarily appears in its statements too. `or ""` added because the original
    took its argument raw and `to_legacy` can hand it a None claim.
    """
    claim = claim or ""
    figures, names = set(), set()
    for m in re.finditer(r'\b\d[\d,]*\.?\d*\s*%|\b(?:19|20)\d{2}\b|[$£€]\s?[\d,]+(?:\.\d+)?'
                         r'|\b\d[\d,]*\.?\d*\s*(?:million|billion|percent)\b', claim):
        figures.add(m.group(0).strip())
    for m in re.finditer(r'\b([A-Z][A-Za-z0-9&.\-]{2,}(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,3})\b', claim):
        w = m.group(1).strip()
        if w.lower() not in {"the", "this", "these", "that"} and len(w) > 3:
            names.add(w)
    return figures, names


def coordinate_asks(q):
    """Coordinate asks in a question — the checklist shape.

    From extract_propose.py. A question that hands over its own contents does it
    as a run of commas and `and`s after `including` / `examining` / `accounting
    for`.
    """
    q = q or ""
    n = 1 + len(re.findall(r"\(\s*\d\s*\)|\b\d\.\s+[A-Z]", q))
    for m in re.finditer(r"\b(including|accounting for|addressing|covering|analyz\w+|"
                         r"examining|considering|compare|comparing)\b(.{0,400})", q, re.I):
        cut = re.split(r"(?<=[.;])\s", m.group(2))[0]
        n += cut.count(",") + len(re.findall(r"\band\b", cut, re.I))
    return n


# Ways a ResearchRubrics question narrows without naming what the answer must
# contain. 33 of its 101 questions open in first person or a persona, and the
# third-person ones still carry an audience, a length or an exclusion.
SITUATED = (
    (r"\b(i|i'm|i am|i've|i need|i want|my|we|we're|our)\b", "asker"),
    (r"\b(act as|pretend (that )?you|you are a|assume you are|imagine (that )?you)\b", "persona"),
    (r"\b(audience|for readers|for a reader|assume the reader|accessible to|"
     r"not familiar with|unfamiliar with|for a graduate|for a seminar)\b", "audience"),
    (r"\b(less than|no more than|under|at most|keep (it|your answer)|word|page)s?\s*\d",
     "length"),
    (r"\b(talk less about|focus less on|leave out|do not (cover|discuss)|"
     r"rather than the|not just the)\b", "exclusion"),
)


def situated(q):
    return [name for pat, name in SITUATED if re.search(pat, q or "", re.I)]


def question_signals(q, subs, stmts):
    """Is the question a contents list of its own subtopics?"""
    qw = content_words(q)
    named, disagree = [], []
    for s in subs:
        hw = content_words(s.get("handle"))
        if hw and len(hw & qw) / len(hw) >= 0.5:
            named.append(s.get("handle"))
            if str(s.get("exposure")).lower() == "implicit":
                disagree.append(s.get("handle"))
    figs = set()
    for s in stmts:
        surface = " ".join([str(s.get("claim") or ""), str(s.get("subtopic") or "")]
                           + [str(e.get("quote") or "") for e in (s.get("evidence") or [])
                              if isinstance(e, dict)])
        figs |= leak_tokens(surface)[0]
    qn = norm(q)
    flat = re.sub(r"[.,$£€%]", "", qn)
    leaked = sorted({t for t in figs
                     if norm(t) in qn or re.sub(r"[.,$£€%]", "", norm(t)) in flat})
    return {
        "subtopics_named_in_question": named,
        "exposure_disagreement": disagree,
        "leaked_figures": leaked,
        "coordinate_asks": coordinate_asks(q),
        "situated": situated(q),
        "question_words": len((q or "").split()),
    }


# ------------------------------------------------------------------ sources ---
# Descriptive only — never a gate. A Reddit thread is the right source for
# "audience reactions were mixed" and the wrong one for "the show uses an
# in-house band"; whether a source fits its claim is a judgement, so this only
# says which statements are worth reading.
#
# Matched on the host, not by substring. Substring matching put AWS's own
# post-mortem at aws.amazon.com into `commerce` (it contains "amazon.") and
# nytix.com into `ugc` (it contains "x.com"). UGC matches subdomains too, since
# old.reddit.com is still Reddit; commerce does not, since aws.amazon.com is not
# a shop.
UGC = {"reddit.com", "linkedin.com", "x.com", "twitter.com", "facebook.com",
       "instagram.com", "quora.com", "medium.com", "substack.com", "tiktok.com",
       "youtube.com", "pinterest.com"}
COMMERCE = {"amazon.com", "target.com", "walmart.com", "goodreads.com", "ebay.com",
            "etsy.com", "bestbuy.com"}
PRIMARY = {"doi.org", "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
           "ncbi.nlm.nih.gov", "arxiv.org", "sec.gov", "ssrn.com", "jstor.org"}
OFFICIAL = {"europa.eu", "who.int", "oecd.org", "imf.org", "worldbank.org",
            "iso.org", "ietf.org", "un.org"}


def _under(host, domains):
    return any(host == d or host.endswith("." + d) for d in domains)


def tier(dom):
    host = (dom or "").lower().lstrip(".")
    if _under(host, PRIMARY):
        return "primary"
    if host.endswith((".gov", ".edu", ".int", ".mil")) or ".gov." in host \
            or ".edu." in host or _under(host, OFFICIAL):
        return "official"
    if _under(host, UGC):
        return "ugc"
    if host in COMMERCE:
        return "commerce"
    return "other"


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


def parents_of(stmts, queries):
    """statement id -> the statements whose reading sent the run looking for it.

    Read off `key_queries`, not off the statements. The relation the
    investigation actually has is statement -> query -> statement: reading S5 and
    S8 prompts a search, and that search returns S6. An earlier version asked each
    statement for a `built_on` list and got the wrong thing — the run with the
    strongest investigation left it empty on all nine statements, correctly, since
    every statement was a direct retrieval and none was derived from another.
    """
    ids = {s.get("id") for s in stmts if isinstance(s, dict)}
    par = {}
    for q in queries or []:
        if not isinstance(q, dict):
            continue
        src = [f for f in (q.get("from") or []) if f in ids]
        for y in (q.get("yielded") or []):
            if y in ids:
                par.setdefault(y, set()).update(src)
    return par


def chain_depth(stmts, queries):
    """How far the investigation had to go before it could ask for this.

    A statement nothing sent the run looking for is depth 1. Each link adds one.
    Every link is a real retrieval carrying its own quote and url, so unlike an
    inference chain it cannot be padded — which is why a single-parent step counts
    here where extract_chain.py required a merge.
    """
    ids = [s.get("id") for s in stmts if isinstance(s, dict) and s.get("id")]
    if not ids:
        return 0
    par = parents_of(stmts, queries)

    def d(sid, seen=()):
        if sid in seen:
            return 0
        ps = [p for p in par.get(sid, ()) if p in ids]
        return (1 if ps else 0) + max([d(p, seen + (sid,)) for p in ps] or [0])

    return max(d(i) for i in ids) + 1


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
    kq = [q for q in (p.get("key_queries") or []) if isinstance(q, dict)]
    carried = [q for q in kq if q.get("from")]
    reached = {y for q in kq for y in (q.get("yielded") or []) if y in ids}
    named = {i for q in kq for i in (q.get("from") or []) + (q.get("yielded") or [])}
    return {
        # what share of the recorded searches came out of something already read
        "carried_query_share": round(len(carried) / len(kq), 3) if kq else 0.0,
        # what share of statements a recorded search claims to have produced
        "reached_by_query_share": round(len(reached) / len(stmts), 3) if stmts else 0.0,
        "n_key_queries": len(kq),
        "statements_per_subtopic": {"median": st.median(counts) if counts else 0,
                                    "min": min(counts) if counts else 0,
                                    "empty": sum(1 for c in counts if c == 0)},
        "subtopic_words": [len((s.get("query") or "").split()) for s in subs],
        "material": Counter(s.get("material") for s in subs),
        "domains": doms, "n_domains": len(doms),
        "no_evidence": [s.get("id") for s in stmts if not (s.get("evidence") or [])],
        "statements_off_map": [s.get("id") for s in stmts
                               if norm(s.get("subtopic")) not in handles],
        "dangling_query_ids": sorted(named - ids),
        "source_tiers": Counter(tier(d) for d in doms),
        # statements whose every source is user-generated — the batch to read
        "ugc_only": [s.get("id") for s in stmts
                     if (s.get("evidence") or [])
                     and all(tier(urlparse(e.get("source") or "").netloc.replace("www.", ""))
                             == "ugc"
                             for e in (s.get("evidence") or []) if isinstance(e, dict))],
        "kept": len(centre.get("kept") or []),
        "discarded": len(centre.get("discarded") or []),
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
        depth = chain_depth(stmts, p.get("key_queries"))
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
            "_signals": {**signals(p, stmts, subs, traj),
                         **question_signals(p["proposed_question"], subs, stmts)},
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
    print(f"  carried queries  median {med(lambda r: r['_signals']['carried_query_share'])}"
          f"   <- searches that came out of something already read")
    print(f"  reached by query median {med(lambda r: r['_signals']['reached_by_query_share'])}"
          f"   <- statements a recorded search produced")
    print(f"  chain depth      median {med(lambda r: r['_axes']['logical_nesting'])}"
          f"   (>=2 in {sum(1 for r in out if r['_axes']['logical_nesting'] >= 2)}/{len(out)})")
    print(f"  implicit share   median {med(lambda r: r['_axes']['exploration'])}")
    print(f"  question words   median {med(lambda r: r['_signals']['question_words'])}"
          f"   (ResearchRubrics median: 68)")
    print(f"  coordinate asks  median {med(lambda r: r['_signals']['coordinate_asks'])}"
          f"   (ResearchRubrics ask density median: 1)")
    sit = sum(1 for r in out if r['_signals']['situated'])
    print(f"  situated         {sit}/{len(out)}"
          f"   (ResearchRubrics: 33/101 open in first person or a persona)")
    named = sum(len(r['_signals']['subtopics_named_in_question']) for r in out)
    dis = sum(len(r['_signals']['exposure_disagreement']) for r in out)
    subs_n = sum(r['_axes']['conceptual_breadth'] for r in out)
    print(f"  subtopics named in the question   {named}/{subs_n}"
          f"   of which marked implicit: {dis}")
    tiers = Counter()
    for r in out:
        tiers.update(r['_signals']['source_tiers'])
    print(f"  source tiers     {dict(tiers)}")
    ugc = sum(len(r['_signals']['ugc_only']) for r in out)
    if ugc:
        print(f"  ! {ugc} statements sourced only from user-generated pages — read these")
    print(f"  searches/visits  median {med(lambda r: r['_signals']['searches'])}"
          f" / {med(lambda r: r['_signals']['visits'])}")
    for k, label in (("no_evidence", "statements with no evidence"),
                     ("statements_off_map", "statements off the subtopic list"),
                     ("dangling_query_ids", "key_queries ids that do not exist"),
                     ("leaked_figures", "figures from a statement restated in the question")):
        n = sum(len(r["_signals"][k]) for r in out)
        if n:
            rows = sum(1 for r in out if r["_signals"][k])
            print(f"  ! {label}: {n} across {rows} questions")
    for name, table in (("conceptual_breadth", BREADTH), ("analysis_load", LOAD),
                        ("logical_nesting", NESTING)):
        print(f"  {name:20} {dict(Counter(r['_axes'][f'{name}_band'] for r in out))}")


if __name__ == "__main__":
    main()
