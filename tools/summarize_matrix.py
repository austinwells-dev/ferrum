"""Summarize runtime_matrix JSONL; retain raw per-step samples separately."""
import json
import statistics as stats
import sys
from collections import defaultdict


def millis(value):
    return value['secs'] * 1000 + value['nanos'] / 1e6


for path in sys.argv[1:]:
    groups = defaultdict(list)
    with open(path) as source:
        for line in source:
            value = json.loads(line)
            groups[value['case']].append(value)
    results = {}
    for case, runs in groups.items():
        result = {key: stats.median(x[key] for x in runs) for key in
                  ('prefill_ms', 'prefill_tps', 'first_token_ms', 'decode_median_ms', 'decode_tps')}
        for phase in ('prefill', 'decode'):
            counters = [x['prefill_counters'] for x in runs] if phase == 'prefill' else [c for x in runs for c in x['decode_counters']]
            result[phase] = {key: stats.mean(millis(x[key]) for x in counters)
                             for key in ('gpu', 'encode', 'wait', 'allocation_time')}
            result[phase].update({key: stats.mean(x[key] for x in counters)
                                 for key in ('allocations', 'allocated_bytes', 'reused_bytes', 'command_buffers', 'completion_waits', 'dispatches')})
            result[phase]['peak_transient_bytes'] = max(x['transient_peak_bytes'] for x in counters)
            result[phase]['max_arena_capacity'] = max(x['arena_capacity'] for x in counters)
        results[case] = result
    print(json.dumps({'source': path, 'cases': results}, indent=2))
