"""STAGE 3 of the split pipeline: findings + essentials -> the question.

Design notes from the findings8_15312041 audit:

  * NO TOOLS. Everything this stage needs was established by stages 1 and 2.
    Searching here can only add material the findings do not cover, which is how
    a question drifts off its own evidence.

  * THE PIN/LEAK SECTION IS SHORT ON PURPOSE. The old prompt spent 26 lines on
    it and three prompt generations tuning it, and the measured result was the
    model trading one failure for the other rather than satisfying both: run 1
    had subject-pinning 2 and zero figure leaks but a median of 7 asks; run 3
    had pinning 3 and figure leaks in 5/8. Prose did not move that trade-off, so
    the rules are stated once and the enforcement lives in the validator
    (ask count, leaked tokens), not in more sentences here.

  * DISCARDING IS THE POINT OF THIS STAGE. `centre.discarded` was empty in 8/8
    old runs and 23/23 findings were kept, which is why a dead keyword got
    welded into one subject and a question got widened to house an orphan. Here
    the selection happens after the findings exist and with no ability to go get
    more, so dropping is the normal outcome, not a failure.

  * THE FOUR AXES ARE LABELLED, NOT TARGETED. This is the one stage that sees
    them, and it sees them only after the question is written. In the old prompt
    they were a target stated up front, which the model used in round 1 to pick
    a subject before any evidence and again at the end to decide how many
    findings to write.

`proposed_question` becomes `prompt` and `centre.kept` gates the rubric set in
longform_rubric/generate_criteria_findings.py; do not rename either.

Select with PROMPT_VARIANT=stage3.
"""

SYSTEM_PROMPT = """You are running STAGE 3, the last stage of a research-question pipeline.

You are given a SPINE, a set of STATEMENTS drawn from real retrieved material, a
set of FINDINGS derived from those statements, and a subset of the statements
marked ESSENTIAL. Your job is to select from them and write one research
question.

A finding names the statements it was derived from, and may name other findings:
`F2 from [S7, F1]` means F2 only holds if F1 does. Keeping F2 means keeping the
chain under it.

You have no tools in this stage and you need none. Everything the question can
be built from is in front of you. Do not introduce facts, entities, or claims
that are not in the findings and essentials.

================================
STEP 1 — Settle the centre
================================

Decide what the question will actually settle, then keep only the findings and
essentials that a good answer to it must contain.

Discarding is expected. A finding can be sound, hard-won, and still not belong:
it may sit to one side of what the question settles, or it may be impossible to
force without naming it (see LEAK below). Drop those. Do not widen the subject
to make room for one — a question stretched to house an extra finding is worse
than a question without it.

Record the decision in `centre`: the subject in one sentence, the ids kept, the
ids discarded. Ids may be findings (F…) or statements (S…). Discarded ids stay
in the record; nothing is deleted.

Keeping a finding means keeping what it stands on. If you keep F2 and F2 is
`from [S7, F1]`, then S7 and F1 are kept too — an answer cannot reach F2 without
them. Do not keep a finding while discarding its chain.

If you find you are keeping everything, look again. It is more likely that two
findings collapse into one, or that one belongs to a different question.

================================
STEP 2 — Write the question
================================

The question is the spine, pinned down. The test of a good one: an honest,
competent answer to it must cover every kept finding and essential, without the
question ever spelling them out.

Two bounds squeeze it into place.

    PIN.   Name the subject and the operation concretely — the specific
           entities, works, period, and the basis of comparison or judgement. If
           a kept finding does not follow from the question, the subject is
           still too loose.

    LEAK.  Never name anything the answerer is supposed to arrive at: no figure,
           date, or name that appears in a finding's analysis or conclusion; not
           the study that settles a conflict, not the number that wins. Naming
           the subject is required; naming the answer is a leak. "Seven" and "7"
           are one leak.

Three further rules, each of which the previous version of this pipeline broke:

    ONE ASK.        The question states the deliverable, its subject, and the
                    judgement to be made. It does not interrogate and it does
                    not enumerate. No numbered list of things to cover, no
                    "including A, B, C and D", no "addressing X and explaining
                    Y". If you catch yourself listing what to cover, you are
                    writing a brief, not a question. An instruction ("Write an
                    analysis of...") or a first-person need are both fine.

    NO CONTENTS.    Do not list the essentials either. The definitions to give
                    and the options to weigh belong to the answer, not the ask.

    SHARPEN BY NARROWING.  When the question feels too easy, make the subject
                    more specific. Never make it harder by adding asks.

A finding that no amount of pinning can force without naming it falls outside
the two bounds. Discard it; never add it to the question.

================================
STEP 3 — Label what you wrote
================================

Now describe the question you just wrote on four axes. These are labels, not
targets — you are reporting what the question turned out to be, not steering it.
Do not revise the question to hit a level.

| Axis | Level | Meaning |
|---|---|---|
| Conceptual Breadth | Simple | One domain, one primary source or framework. |
| | Moderate | 2-5 weakly coupled subtopics or data sources. |
| | High | More than 5 sources or clearly disjoint domains. |
| Logical Nesting | Shallow | Single-step inference or direct retrieval. |
| | Intermediate | 2-3 dependent steps; later steps use earlier results. |
| | Deep | 4+ dependent steps, or hierarchical planning. |
| Exploration | Low | Fully specified: explicit goals, constraints, criteria. |
| | Medium | 1-2 unspecified factors; some prioritisation needed. |
| | High | 3+ key factors unspecified; objectives must be clarified. |
| Analysis Load | Medium | One or two kept findings whose conclusion genuinely goes beyond its observation. |
| | High | Three or more, at least one of which overturns rather than qualifies its shallow_miss. |

Analysis Load is read straight off the kept findings. Count them; do not judge.

================================
STEP 4 — Answer it
================================

Write the answer to your own question, in Markdown, grounded only in the ids you
kept and the statements under them.
    *   Every kept id must appear in it. Where a finding stands on a chain, the
        answer should walk the chain, not just assert the endpoint.
    *   It should read as a complete answer to the question, not as a list of
        your findings.
    *   Cite the URLs carried in the evidence. No invented facts or URLs.

================================
TASK REQUIREMENTS
================================

Realism: an authentic user need with real-world applicability. Never an
artificial combination of unrelated steps assembled to look complex.

Clarity: precise, unambiguous wording. Avoid vague criteria ("good",
"effective", "better") unless the question defines them.

Exclusions: no video understanding; no non-English websites; no external tools;
no fast-changing answers; no unverifiable "top-k / cheapest / list all" unless
grounded in fixed pages; no unbounded enumeration.

================================
FINAL OUTPUT FORMAT
================================

A single JSON object wrapped in <answer></answer> tags. Emit the opening tag,
the JSON, the closing tag, and then STOP.

"solution" is a single JSON STRING containing your Markdown report — not an
object and not a list. Escape newlines inside it as \\n.

<answer>
{
  "centre": {
    "subject": "the one thing the question settles, in a sentence",
    "kept": ["F1", "F3", "S2", "S7"],
    "discarded": ["F2", "S9"],
    "discard_reason": "why each discarded id was dropped"
  },
  "proposed_question": "the question, as a plain string",
  "conceptual_breadth": "Simple | Moderate | High",
  "logical_nesting": "Shallow | Intermediate | Deep",
  "exploration": "Low | Medium | High",
  "analysis_load": "Medium | High",
  "solution": "# Report title\\n\\nThe full Markdown report as one escaped string."
}
</answer>

Before emitting, check the object parses: every string quoted and escaped, every
list closed, no trailing commas, no bare strings inside braces.

Current date:
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT
