"""Optional external-engine measurement; never imported by Ferrum."""
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
model, tokenizer = load(sys.argv[1])
reference=json.loads((Path(__file__).resolve().parents[1]/'docs/measurements/phase3/reference-bf16.json').read_text())
ids=reference['prompt_ids']
assert tokenizer.encode(reference['raw'],add_special_tokens=False)==ids
from mlx.utils import tree_flatten
weights=tree_flatten(model.parameters())
print(json.dumps({'mlx':importlib.metadata.version('mlx'),'mlx_lm':importlib.metadata.version('mlx-lm'),'dtypes':sorted(set(str(v.dtype) for _,v in weights)),'weight_bytes':sum(v.nbytes for _,v in weights)}))
for run in range(4):
    durations=[];tokens=[]
    mx.reset_peak_memory()
    start=time.perf_counter()
    for token,_ in generate_step(mx.array(ids),model,max_tokens=10):
        now=time.perf_counter();durations.append(now-start);tokens.append(token);start=time.perf_counter()
    mx.synchronize()
    if run:
        print(json.dumps({'run':run,'prompt_tokens':len(ids),'generated_ids':tokens,'first_token_ms':1000*durations[0],'prefill_including_first_selection_tps':len(ids)/durations[0],'median_decode_tps':1/statistics.median(durations[1:]),'active_bytes':mx.get_active_memory(),'cache_bytes':mx.get_cache_memory(),'peak_active_bytes':mx.get_peak_memory()}))
