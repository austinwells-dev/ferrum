#!/usr/bin/env python3
"""Journal table for a control/candidate A/B with saved llama.cpp references.

Cells: control->candidate / llama.cpp tok/s, then
[control/llama -> candidate/llama; median paired candidate/control].
"""
import argparse, json, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("ab")
ap.add_argument("reference")
ap.add_argument("--label", default="")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.ab) if l.startswith("{")]
rows = [r for r in rows if not r.get("warmup") and "error" not in r]
ref = [json.loads(l) for l in open(a.reference) if l.startswith("{")]
llama = defaultdict(list)
for r in ref:
    if r.get("runtime") == "llama.cpp":
        llama[r["case"]].append(r)
names = {"short": "Short", "128": "128-token prompt", "512": "512-token prompt",
         "1024": "1,024-token prompt", "sustained-128-decode": "Sustained decode"}
fmt = lambda x: f"{x:,.1f}"
cases = list(dict.fromkeys(r["case"] for r in rows))
for case in cases:
    by = {v: {r["pair"]: r for r in rows if r["case"] == case and r["ferrum_variant"] == v}
          for v in ("control", "candidate")}
    pairs = sorted(set(by["control"]) & set(by["candidate"]))
    ll = llama.get(case, [])
    cells = []
    for f in ("prefill_tps", "decode_tps", "generation_tps"):
        c = st.median(by["control"][p][f] for p in pairs)
        n = st.median(by["candidate"][p][f] for p in pairs)
        ratio = st.median(by["candidate"][p][f] / by["control"][p][f] for p in pairs)
        if ll:
            l = st.median(r[f] for r in ll)
            cells.append(f"{fmt(c)}->{fmt(n)} / {fmt(l)} `[{c/l:.3f}->{n/l:.3f}; {ratio:.3f}]`")
        else:
            cells.append(f"{fmt(c)}->{fmt(n)} `[{ratio:.3f}]`")
    ttft = f"{fmt(st.median(by['control'][p]['first_token_ms'] for p in pairs))}->" \
           f"{fmt(st.median(by['candidate'][p]['first_token_ms'] for p in pairs))}"
    if ll:
        ttft += f" / {fmt(st.median(r['first_token_ms'] for r in ll))}"
    same = sum(by["control"][p]["generated_ids"] == by["candidate"][p]["generated_ids"] for p in pairs)
    ids = f"{same}/{len(pairs)}"
    if ll:
        lid = ll[0]["generated_ids"]
        ids += f"; {sum(by['candidate'][p]['generated_ids'] == lid for p in pairs)}/{len(pairs)}" \
               f"; {sum(by['control'][p]['generated_ids'] == lid for p in pairs)}/{len(pairs)}"
    label = f"{a.label} / " if a.label else ""
    print(f"| {label}{names.get(case, case)} | " + " | ".join(cells) + f" | {ttft} | {ids} |")
