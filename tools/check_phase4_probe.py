"""Validate the recorded diagnostic against the immutable Phase 3 F32 reference."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
a = json.loads((root / 'docs/measurements/phase4/ferrum-probe-f32.json').read_text())
b = json.loads((root / 'docs/measurements/phase3/reference-f32.json').read_text())
assert a['prompt_ids'] == b['prompt_ids']
maximum = 0
for x, y in zip(a['steps'], b['steps'], strict=True):
    assert x['token'] == y['token']
    assert [p[0] for p in x['top10']] == [p[0] for p in y['top10']]
    for key in x['selected']:
        maximum = max(maximum, abs(x['selected'][key] - y['selected'][key]))
    for p, q in zip(x['top10'], y['top10'], strict=True):
        maximum = max(maximum, abs(p[1] - q[1]))
assert maximum <= 8e-5, maximum
print(json.dumps({'steps': len(a['steps']), 'ordered_top10_match': True, 'max_abs': maximum, 'tolerance': 8e-5}))
