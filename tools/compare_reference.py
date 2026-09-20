"""Check the saved Ferrum/Transformers diagnostics; no inference dependency."""
import json
from pathlib import Path

root = Path('docs/measurements/phase3')
ferrum = json.loads((root / 'ferrum-probe-f32.json').read_text())
reference = json.loads((root / 'reference-f32.json').read_text())
assert ferrum['prompt_ids'] == reference['prompt_ids']
assert ferrum['generated_ids'] == reference['generated_ids']
errors = []
for actual, expected in zip(ferrum['steps'], reference['steps'], strict=True):
    assert [x[0] for x in actual['top10']] == [x[0] for x in expected['top10']]
    errors.extend(abs(x[1] - y[1]) for x, y in zip(actual['top10'], expected['top10'], strict=True))
    errors.extend(abs(value - expected['selected'][key]) for key, value in actual['selected'].items())
assert max(errors) < 1e-4
lines = [f'F32: all 8 tokens and ordered top10 IDs match. Maximum absolute error over selected/top10 logits: {max(errors):.10g}']
ferrum = json.loads((root / 'ferrum-probe.json').read_text())
reference = json.loads((root / 'reference-bf16.json').read_text())
assert ferrum['prompt_ids'] == reference['prompt_ids']
assert ferrum['generated_ids'][:5] == reference['generated_ids'][:5]
for actual, expected in zip(ferrum['steps'][:5], reference['steps'][:5], strict=True):
    assert max(abs(v - expected['selected'][k]) for k, v in actual['selected'].items()) <= 0.5
lines.append('BF16: first 5 tokens match; sixth differs at a 25.75 tie in Transformers (1492 help / 7789 assist). Ferrum: 25.625 help / 25.75 assist.')
text = '\n'.join(lines) + '\n'
(root / 'reference-comparison.txt').write_text(text)
print(text)
