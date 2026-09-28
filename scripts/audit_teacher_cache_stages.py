"""Read-only audit of SD cache extraction provenance; does not load a model."""
import argparse
import json
from collections import Counter
from pathlib import Path


def audit(cache, *, layers_per_step=5):
    counts = Counter()
    invalid = []
    checked = 0
    for path in sorted(cache.glob('*.json')):
        row = json.loads(path.read_text())
        if 'attention_tensors' not in row:
            continue
        checked += 1
        count = row['attention_tensors']
        counts[str(count)] += 1
        expected = row['inversion_steps'] * layers_per_step
        if count != expected:
            invalid.append({'metadata': str(path), 'actual': count, 'expected': expected,
                            'reason': 'not_reconstruction_only_capture'})
    return {'cache': str(cache), 'checked': checked, 'tensor_counts': dict(counts),
            'invalid': invalid, 'stage_count_gate': bool(checked) and not invalid,
            'limitation': 'Matching counts are necessary, not sufficient; code path and numerical parity remain required.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.cache)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'invalid'}))
    return 0 if result['stage_count_gate'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
