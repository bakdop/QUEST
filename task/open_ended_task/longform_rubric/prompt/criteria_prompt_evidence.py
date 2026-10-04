"""Rubric-generation prompt for evidence_first runs, in QUEST's own prompt form.

criteria_prompt_en.py (the original release) is the model for the SHAPE of this
file: XML-tagged sections, a `<system_role>`, a `**Background**`, an
`<instruction>` split into numbered `**Your Goal**` steps and numbered
`**Core Requirements**`, an `<example_rational>` telling the model to learn the
reasoning rather than copy the content, a worked `<example>` whose output is an
`<analysis>` followed by a `<json_output>`, and then the real task restated. All
of that is kept. Three things change.

WHAT THE ITEMS ARE. The original asks for scoring dimensions - a `criterion` that
is a heading ("Depth of Settlement Mechanics and Fund Availability Analysis") with
the checkable substance sitting in a separate `explanation` field, and weights
summing to 1.0 across a dimension. Swap those two fields and it is a
ResearchRubrics item, which is what a grader can actually mark. So `criterion`
now carries the checkable statement, weights are 1-5 per item, and each item is
marked SATISFIED or NOT SATISFIED with no partial credit.

WHAT THE AXES ARE. QUEST's four dimensions are kept and Implicit Criteria is added
to them, because that is the one thing the four do not have a slot for and it is
the largest single class in ResearchRubrics (39.4% of 2593 items). Synthesis stays
inside Insight rather than becoming its own axis, and References & Citation Quality
is not used. The boundary that has to be stated explicitly is Comprehensiveness vs
Implicit: both are about what the report contains, and the only thing separating
them is whether the question asked.

WHAT THE MODEL IS GIVEN. The original sees the task string and nothing else, which
is why its criteria come out as generic writing standards. Here `<task>` carries
the whole investigation that produced the question: the sections it was organised
by, the statements with their verbatim quotes and urls, the path that reached them,
and the full corpus the tools returned. Most of a real rubric is ordinary
competent content that never became a statement, and the corpus is where that is.

The worked example is a real ResearchRubrics task with its real items rather than
an invented one, since their register is the target. Its negative-weight items are
dropped (see generate_criteria_evidence.py for why) and its References items are
dropped with that axis; the `<analysis>` is written here in the original's voice.
"""

CRITERIA_PROMPT_EVIDENCE = """
<system_role>
You are an experienced research article evaluation expert. You excel at turning the raw material of a completed investigation - the sources it read, the claims it recorded, the path it took - into a rubric whose every item a grader can mark satisfied or not satisfied by reading a report, with no partial credit and nothing left to taste.
</system_role>

<user_prompt>
**Background**: We are evaluating a deep research report written for the task below. The rubric is a flat list of items. A grader reads the report and marks each item SATISFIED or NOT SATISFIED; the score is Sum(weight x satisfied) / Sum(weight). Because the judgment is binary, an item like "the analysis has depth" is unusable - two graders would disagree. "The response states that a 32GB DDR5-6000 kit rose from roughly $100 in October 2025 to $389" is usable. Every item belongs to exactly one of five axes:

1.  **Comprehensiveness:** Coverage the task asks for or plainly implies - the information areas, perspectives and depths a reader would notice were missing after reading the task.
2.  **Insight:** The depth, logic and value of the analysis. This is also where synthesis lives: reconciling sources that disagree, establishing that two figures are not comparable, adjudicating between pieces of evidence, reaching a conclusion no single source states.
3.  **Instruction Following:** Format, length, scope and audience constraints the task states.
4.  **Readability:** Structure, transitions, data presentation, and whether specialist language is made usable.
5.  **Implicit Criteria:** What a competent answer must contain that **the task never asked for** - the domain knowledge the asker did not know to request.

**The line between Comprehensiveness and Implicit Criteria is whether the task asked.** If the task names it, or a careful reader of the task alone would list it, the item is Comprehensiveness. If only somebody who knows the field would think to require it, and the task says nothing about it, the item is Implicit Criteria. This distinction carries most of the rubric's value, so draw it deliberately rather than by feel.

<task>
QUESTION GIVEN TO THE REPORT WRITER

{question}

SECTIONS the researcher ended up organising this topic by. `implicit` marks a section the question does not name and an answer that skips it is wrong anyway; `explicit` marks one the question asks for in so many words.

{subtopics}

STATEMENTS they recorded, each with the source that returned it. Treat a claim as a pointer into the corpus rather than as authority - the wording of a quote here is sometimes the researcher's paraphrase, so check the corpus before building an item on an exact phrasing.

{statements}

HOW THEY GOT THERE. Each step names what had to be known before it could be asked, and the gap it closed. A step whose `needed first` is non-empty was not answerable from the question alone; somebody had to have read the earlier material to know to ask it. Those gaps are what separate a deep report from one that stopped at the obvious search.

{path}

THE CORPUS - everything the tools returned during the investigation. Most of it never became a statement. This is where the ordinary competent content lives: definitions, standard options, eligibility rules, common pitfalls. Those items are most of a real rubric.

{corpus}
</task>

<instruction>
**Your Goal**: For this specific task, write a rubric of {lo}-{hi} binary items. You need to:
1.  **Analyze Task**: Study what this task actually demands - its stated requirements, its implicit goals, the deliverable it names, and which parts a report could get wrong while still looking complete.
2.  **Mine the Material**: Read the corpus for what a knowledgeable reader would expect and the report would be incomplete without, not only what became a statement. Read the path for the judgments a shallow answer would skip.
3.  **Formulate Items**: Turn each into a single statement a grader marks satisfied or not satisfied. State the referent inside the item so it can be graded by someone holding only the report and that one line.
4.  **Explain Rationale**: Give each item a `why_it_matters` saying what a report that omitted it would be missing. This is the argument for the item, not the thing being graded.
5.  **Assign Weight and Grounding**: Give each item a weight from 1 to 5, and a `grounding` list naming the corpus spans or statement ids that justify demanding it.

**Core Requirements**:
1.  **Task-Centric**: Every item, rationale and weight must trace to this task's own requirements and to this investigation's material. Nothing generic that would fit any research report.
2.  **Well-Justified**: The `<analysis>` must lay out the reasoning behind the set as a whole - what this task's answer has to contain, where a report would go wrong, and why the heavy items are heavy - before any item is written.
3.  **Binary and Standalone**: No partial credit and no scales. Never write "the finding", "neither study", "as established above" - restate the referent, even at the cost of length.
4.  **Positive Only**: Weights are 1 to 5. There are no negative weights and no penalty items. Never write an item that is satisfied when the report does something wrong; every item states something a good report DOES.
5.  **Acceptance Sets**: If the verb is a judgment verb - explains, compares, reconciles, discusses, details - the item must end with a parenthetical listing 2-4 concrete things that count as satisfying it. Without it the item is not gradeable.
6.  **Grounded**: The corpus must contain something a grader could point at to justify each demand, and `grounding` must quote it rather than paraphrase it. If you cannot find one, do not write the item - not from your own knowledge of the field, not from what seems reasonable.
7.  **Discriminating**: Before writing an item, ask whether a competent report that did NOT do it would be clearly worse. If a good report could skip it and lose nothing, leave it out. An item that fails to separate a better report from a worse one is the most damaging thing in a rubric.
8.  **One Thing Each**: An item assesses one thing and asserts at most two numbers or two named entities. If it needs "and" to say what it wants, it is two items.
9.  **No Task Echo**: Never restate the task's own wording prefixed with "The response addresses...". Every on-topic report passes such an item, so it measures nothing. Bind the requirement to an observable value or artifact.
10. **Operational Soft Axes**: In Readability and Instruction Following, the words clear, professional, appropriate, excessive, throughout and key are banned unless immediately followed by a test that can be run - an exact section name, a word count, "defined at first use", "as a table with one row per option".
11. **Judgment Over Recall**: At most a third of items may hinge on reproducing a specific figure. The rest must demand an explanation, a mechanism, a caveat, a judgment of applicability, or the recognition that two things are not comparable. Where the material shows two things that must be reconciled before they can be compared, the item testing the reconciliation is worth more than the items testing the two numbers.
12. **Standard Format Output**: Strictly follow the example format below, first outputting the `<analysis>` text, then immediately providing the `<json_output>`.
</instruction>

<example_rational>
The following example demonstrates **how to formulate binary rubric items from a task and its material**. Focus on learning the **thinking logic and the register of the items** from this example - how concrete they are, how a judgment verb is always paired with an acceptance list - not on imitating its subject matter or its weight values.
</example_rational>

<example>
<task>
"Write a synthesis report on the applications of AI in drug discovery for a technical audience unfamiliar with biology. It should cover the main applications of AI in every stage of the drug discovery process, the latest technological advancements, challenges, and current adoption in the real world."
</task>

<output>
<analysis>
This task names four things the report must cover - applications, advancements, challenges, adoption - and one hard constraint on the reader: technical, but not a biologist. The constraint is what makes the task difficult. A report can cover all four topics competently and still fail, because the audience cannot follow it; so a share of the weight has to sit on whether specialist language is made usable at the point of first use rather than assumed.

Coverage here is not satisfied by breadth of prose. The drug discovery process has named stages, and a report that discusses "AI in drug discovery" in general terms while silently skipping preclinical safety or post-market surveillance has left a hole a knowledgeable reader would see immediately. So the coverage items are written against the stage list, and against the requirement that each stage be tied to a named method class rather than to "AI" as a mass noun.

Two things the task does not ask for belong in a good answer anyway. The first is the gap between what a model does computationally and what has actually been demonstrated in practice - the whole field's credibility rests on that distinction, and a report that elides it reads as promotional. The second is the ethical exposure of the work. Neither appears in the task; both would be conspicuous by their absence to somebody in the field. Those are the Implicit items, and they carry heavy weight because they are exactly what separates a report by somebody who knows the area from one assembled off search results.

Insight is carried by a single demand: that the challenges be stated as distinct, named limitations rather than as a paragraph of hedging. Readability items cover navigation, since a synthesis that cannot be scanned defeats its own purpose for this reader.
</analysis>
<json_output>
[
  {{
    "criterion": "The response describes at least one specific AI application for EACH of the following drug-discovery stages: (1) Target Identification (2) Hit/Lead Discovery or Virtual Screening (3) Lead Optimization (4) Preclinical Safety / ADMET Prediction (5) Clinical Trial Design / Patient Stratification (6) Post-market Surveillance / Pharmacovigilance",
    "weight": 5,
    "axis": "Comprehensiveness",
    "grounding": ["the six-stage breakdown of the discovery pipeline set out in the corpus"],
    "why_it_matters": "A report that covers three stages and generalises over the rest reads as complete but leaves the reader unable to place any application in the pipeline."
  }},
  {{
    "criterion": "The response, for each stage, names at least one specific AI model class or method other than generative models (e.g., CNNs, GNNs, classical ML such as random forests, knowledge graphs, hybrid AI-physics models).",
    "weight": 5,
    "axis": "Comprehensiveness",
    "grounding": ["the model classes enumerated in the corpus alongside each stage"],
    "why_it_matters": "Without this the report collapses the field into generative AI, which is a small part of what is actually deployed."
  }},
  {{
    "criterion": "The response distinguishes between what AI achieves in silico and what has been demonstrated in practice.",
    "weight": 4,
    "axis": "Implicit Criteria",
    "grounding": ["the corpus passages contrasting computational benchmark results with clinical outcomes"],
    "why_it_matters": "The task never asks for this, but a report that does not draw the line reads as promotional to anyone who works in the field."
  }},
  {{
    "criterion": "The response includes a subsection dedicated to the ethical risks of AI-driven drug discovery (e.g., transparency, data privacy, bias, accountability, workforce displacement).",
    "weight": 5,
    "axis": "Implicit Criteria",
    "grounding": ["the corpus discussion of regulatory acceptance and accountability"],
    "why_it_matters": "Nothing in the task asks for it, and its absence is the single most conspicuous omission a domain reader would notice."
  }},
  {{
    "criterion": "The response states at least three distinct challenges or limitations from this list: data quality or scarcity, interpretability, regulatory acceptance, lab workflow integration, IP or confidentiality, compute and infrastructure cost.",
    "weight": 4,
    "axis": "Insight",
    "grounding": ["the challenges enumerated across several corpus sources"],
    "why_it_matters": "Named, separable limitations are what let a reader judge the field; a paragraph of general caution does not."
  }},
  {{
    "criterion": "The response provides brief definitions of at most 20 words for all specialised terms, acronyms, and each drug-discovery stage, before discussing their AI applications.",
    "weight": 5,
    "axis": "Instruction Following",
    "grounding": ["the task's stated audience: technical, unfamiliar with biology"],
    "why_it_matters": "The stated reader cannot follow the report at all without this, so it is the constraint most likely to be silently dropped."
  }},
  {{
    "criterion": "The response contains four clearly labelled sections using the words Applications, Advancements, Challenges and Adoption.",
    "weight": 4,
    "axis": "Instruction Following",
    "grounding": ["the four coverage areas named in the task"],
    "why_it_matters": "The task names four deliverables; unlabelled prose makes it impossible to tell whether each was addressed."
  }},
  {{
    "criterion": "The response opens each section with a short orientation and closes it with its key takeaways.",
    "weight": 3,
    "axis": "Readability",
    "grounding": ["the length and density of the material the report has to carry"],
    "why_it_matters": "A synthesis this dense is unusable to a non-specialist without signposting, whatever its content."
  }}
]
</json_output>
</output>
</example>

Please strictly follow the above instructions and methods. Now, begin your work on the specific task given in the `<task>` section above.

Write {lo}-{hi} items. Put roughly two fifths of them on Implicit Criteria and Insight together, since those are what separate a knowledgeable report from a diligent one; give Comprehensiveness about a third; and leave a handful for Instruction Following and Readability. Vary the weights - do not give everything a 3.

Please output your `<analysis>` and `<json_output>`.
</user_prompt>
"""
