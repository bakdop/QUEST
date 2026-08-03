import json
import os
import argparse


def coerce_prediction_json(pred):
    """Return prediction['json'] as a dict, or None if it cannot be recovered.

    The agent stores the raw model text when its own JSON parse fails. In
    practice the model occasionally emits one closing brace too many, which is
    otherwise valid JSON, so strip trailing unbalanced closers before giving up.
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
    while text.count('{') < text.count('}'):
        idx = text.rfind('}')
        text = text[:idx] + text[idx + 1:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return None


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
        for key in ('spine', 'findings', 'solution'):
            if key in pred:
                item[key] = pred[key]
        out.append(item)
    print(f"extracted {len(out)}/{len(filelist)}")
    for file in skipped:
        print(f"  skipped (unparseable prediction): {file}")
    with open(args.output_file, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
