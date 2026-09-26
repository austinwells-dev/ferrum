#!/usr/bin/env python3
"""Interleaved A/B between two phase55_bench executables (control, candidate).

Either side may add request fields (--control-opts/--candidate-opts, JSON) or
environment variables (--control-env/--candidate-env, JSON). The same binary
may be passed twice for an in-process option A/B. Every request runs through
the Phase 7A CPU preflight; rows carry ferrum_variant=control|candidate."""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase55_matched_matrix as m  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cases", required=True, type=Path)
ap.add_argument("--output", required=True, type=Path)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--control", required=True)
ap.add_argument("--candidate", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--control-opts", default="{}")
ap.add_argument("--candidate-opts", default="{}")
ap.add_argument("--only", default="", help="comma-separated case names")
ap.add_argument("--control-env", default="{}")
ap.add_argument("--candidate-env", default="{}")
a = ap.parse_args()

opts = {"control": json.loads(a.control_opts), "candidate": json.loads(a.candidate_opts)}
engines = {
    "control": m.JsonLineProcess([a.control, a.model], {**os.environ, **json.loads(a.control_env)}),
    "candidate": m.JsonLineProcess([a.candidate, a.model], {**os.environ, **json.loads(a.candidate_env)}),
}
cases = m.source_cases(a.cases)
if a.only:
    keep = set(a.only.split(","))
    cases = [c for c in cases if c["case"] in keep]
names = ["control", "candidate"]
try:
    for i, row in enumerate(cases):
        for n in m.pair_order(names, i, 0):
            m.cpu_preflight()
            req = {"case": row["case"], "pair": 0, "pair_order": "warmup", "warmup": True,
                   "prompt_ids": row["prompt_ids"], "max_new_tokens": 2, **m.case_request_options(row), **opts[n]}
            engines[n].request(req)
    with a.output.open("w") as out:
        for i, row in enumerate(cases):
            mx = row.get("max_new_tokens", len(row.get("generated_ids", [])))
            for p in range(a.repeats):
                order = m.pair_order(names, i, p)
                for n in order:
                    m.cpu_preflight()
                    req = {"case": row["case"], "pair": p, "pair_order": ">".join(order), "warmup": False,
                           "prompt_ids": row["prompt_ids"], "max_new_tokens": mx,
                           **m.case_request_options(row), **opts[n]}
                    r = engines[n].request(req)
                    r["ferrum_variant"] = n
                    out.write(json.dumps(r, separators=(",", ":")) + "\n")
                    out.flush()
finally:
    for e in engines.values():
        e.close()
