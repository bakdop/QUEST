import json
import os
import re
import argparse


def _strip_trailing_closers(text):
    """The model occasionally emits one closing brace too many."""
    while text.count('{') < text.count('}'):
        idx = text.rfind('}')
        text = text[:idx] + text[idx + 1:]
        try:
            return json.loads(text), text
        except json.JSONDecodeError:
            continue
    return None, text


def _rescue_bare_solution(text):
    """Recover `"solution": { "...markdown..." }` — an object holding a bare
    string, which is not valid JSON.

    The output template in both prompts used to show `"solution": {
    "PLACEHOLDER_SOLUTION" }`, and the model reproduced that shape faithfully.
    The template is fixed, but the report is the longest and most escape-prone
    field in the object, so keep salvaging it rather than losing the question and
    findings along with it.
    """
    m = re.search(r'"solution"\s*:\s*\{\s*(".*")\s*\}\s*\}?\s*$', text, re.DOTALL)
    if not m:
        return None
    patched = text[:m.start()] + '"solution": ' + m.group(1) + '}'
    try:
        return json.loads(patched)
    except json.JSONDecodeError:
        return None


def coerce_prediction_json(pred):
    """Return prediction['json'] as a dict, or None if it cannot be recovered.

    The agent stores the raw model text when its own JSON parse fails, so a
    trajectory whose report merely had a formatting slip would otherwise be
    dropped entirely.
    """
    raw = pred.get('json') if isinstance(pred, dict) else None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    parsed, text = _strip_trailing_closers(text)
    if parsed is not None:
        return parsed
    return _rescue_bare_solution(text)


def main():
    parser = argparse.ArgumentParser(description='Extract prediction fields from JSON files')
    parser.add_argument('--input_dir', type=str, default="./outputs/openended_trajectories/", help='Input directory containing JSON files')
    parser.add_argument('--output_file', type=str, default='./outputs/extracted_questions.jsonl', help='Output JSONL file path')
    args = parser.parse_args()

    filelist = [f for f in os.listdir(args.input_dir) if f.endswith('.json')]
    filelist.sort(key=lambda x: int(x.split('_')[1]))
    out = []
    skipped = []
    for idx, file in enumerate(filelist):
        with open(os.path.join(args.input_dir, file), "r") as f:
            data = json.load(f)
        # One malformed trajectory must not discard the rest of the batch.
        pred = coerce_prediction_json(data.get("prediction"))
        required = ('conceptual_breadth', 'logical_nesting', 'exploration', 'proposed_question')
        if pred is None or any(k not in pred for k in required):
            skipped.append(file)
            continue
        item = {
            'id': len(out) + 1,
            'topic': "_".join(file.split('_')[7:]).strip("json").strip('.'),
            "conceptual_breadth": pred['conceptual_breadth'],
            "logical_nesting": pred['logical_nesting'],
            "exploration": pred['exploration'],
            'prompt': pred['proposed_question']
        }
        # Carry the proposer's evidence forward. Rubric generation used to see the
        # question and nothing else, so it could only produce generic writing
        # standards - it had no way to know which specific facts a good answer
        # must contain. The findings-first prompt records them; keep them.
        for key in ('findings', 'essentials', 'analysis_load', 'solution'):
            if key in pred:
                item[key] = pred[key]
        # analysis_load is sampled by the pipeline, so trust the trajectory over
        # whatever the model echoed back into its answer.
        if data.get('analysis_load'):
            item['analysis_load'] = data['analysis_load']
        out.append(item)
    print(f"extracted {len(out)}/{len(filelist)}")
    for file in skipped:
        print(f"  skipped (unparseable prediction): {file}")
    with open(args.output_file, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
