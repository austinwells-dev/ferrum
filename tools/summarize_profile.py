"""Read opt-in Ferrum CLI profile records; this never runs inference."""
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('log')
parser.add_argument('output')
args = parser.parse_args()
profiles = [json.loads(line.split(': ', 1)[1])
            for line in Path(args.log).read_text().splitlines()
            if line.startswith('PROFILE ')]

def ms(value):
    return value['secs'] * 1000 + value['nanos'] / 1e6

rows = []
zero = {'calls': 0, **{field: {'secs': 0, 'nanos': 0} for field in
        ['wall', 'allocation', 'submission', 'synchronized', 'gpu']}}
for name in sorted(set().union(*(profile.keys() for profile in profiles))):
    entry = profiles[0].get(name, zero)
    row = {'operation': name, 'prefill_calls': entry['calls'],
           'prefill_wall_ms': ms(entry['wall']), 'prefill_gpu_ms': ms(entry['gpu'])}
    if len(profiles) > 1:
        row['decode_calls'] = profiles[1].get(name, zero)['calls']
        for field in ['wall', 'allocation', 'submission', 'synchronized', 'gpu']:
            row['decode_mean_' + field + '_ms'] = statistics.mean(
                ms(profile.get(name, zero)[field]) for profile in profiles[1:])
    rows.append(row)
Path(args.output).write_text(json.dumps(rows, indent=2) + '\n')
