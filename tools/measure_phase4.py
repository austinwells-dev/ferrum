"""Repeat the pinned Hello benchmark. Requires a release build and local checkpoint."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
out=root/'docs/measurements/phase4'
model=sys.argv[1]
prefix=sys.argv[2] if len(sys.argv)>2 else "g"
base=[str(root/'target/release/ferrum'),'run','--model',model,'--prompt','Hello!','--max-new-tokens','32','--temperature','0','--warmup']
records=[]
for i in range(3):
    text=subprocess.run(['/usr/bin/time','-l',*base],capture_output=True,text=True,check=True)
    log=text.stdout+text.stderr
    (out/f'{prefix}-final-{i+1}.txt').write_text(log)
    pre=re.search(r'prefill: ([\d.]+) ms \(([\d.]+) tok/s\); first token: ([\d.]+)',log)
    dec=re.search(r'decode: median ([\d.]+) ms \(([\d.]+) tok/s\)',log)
    rss=re.search(r'(\d+)\s+maximum resident set size',log)
    records.append(dict(run=i+1,prefill_ms=float(pre[1]),prefill_tps=float(pre[2]),first_token_ms=float(pre[3]),decode_ms=float(dec[1]),decode_tps=float(dec[2]),peak_rss_bytes=int(rss[1])))
for label,env in [(f'{prefix}-profile',os.environ.copy()),(f'{prefix}-kernel-profile',dict(os.environ,FERRUM_BATCH_LIMIT='1'))]:
    command=base.copy(); command[command.index('32')]='8';command.append('--profile')
    text=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
    (out/f'{label}.txt').write_text(text.stdout+text.stderr)
(out/f'{prefix}-summary.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records))
