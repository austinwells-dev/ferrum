"""External BF16 reference on exact recorded Ferrum matrix IDs; no prompt cache reuse."""
import importlib.metadata
import json
import statistics
import sys
import time
import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import generate_step

model, _ = load(sys.argv[1])
mx.eval(model.parameters())
weights = tree_flatten(model.parameters())
print(json.dumps({'mlx': importlib.metadata.version('mlx'), 'mlx_lm': importlib.metadata.version('mlx-lm'),
                  'dtypes': sorted({str(v.dtype) for _, v in weights}),
                  'weight_bytes': sum(v.nbytes for _, v in weights),
                  'prompt_cache': 'fresh per invocation', 'kv_bits': None}), flush=True)
seen = set()
with open(sys.argv[2]) as source:
    cases = [json.loads(line) for line in source]
for case in cases:
    label = case['case']
    if label in seen:
        continue
    seen.add(label)
    ids = case['prompt_ids']
    count = len(case['generated_ids'])
    for run in range(2):
        mx.synchronize()
        mx.reset_peak_memory()
        tokens = []
        durations = []
        start = previous = time.perf_counter()
        for token, _ in generate_step(mx.array(ids), model, max_tokens=count, prompt_cache=None, kv_bits=None):
            now = time.perf_counter()
            durations.append(now - previous)
            tokens.append(token)
            previous = now
        mx.synchronize()
        elapsed = time.perf_counter() - start
        if run:
            print(json.dumps({'case': label, 'prompt_ids': ids, 'generated_ids': tokens,
                              'first_token_ms': 1000 * durations[0],
                              'prefill_including_selection_tps': len(ids) / durations[0],
                              'generation_ms': 1000 * elapsed, 'generation_tps': len(tokens) / elapsed,
                              'post_first_token_tps': (len(tokens) - 1) / (elapsed - durations[0]),
                              'yield_interval_median_tps': 1 / statistics.median(durations[1:]),
                              'active_bytes': mx.get_active_memory(), 'cache_bytes': mx.get_cache_memory(),
                              'peak_active_bytes': mx.get_peak_memory()}), flush=True)
