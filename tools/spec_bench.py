#!/usr/bin/env python3
"""Speculative-decoding benchmark against an OpenAI-compatible server.

Sends the same chat prompts as examples/hybrid_spec.rs (greedy by default)
to ferrum-server or llama-server and reports decode tok/s and draft
acceptance from each response's `timings` (both servers report
predicted_n / predicted_ms / draft_n / draft_n_accepted).

usage: spec_bench.py [--url http://127.0.0.1:8080] [--tokens 128]
                     [--prompts 8] [--temperature 0] [--no-think] [--out FILE]
"""
import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using memoization, then explain its complexity.",
    "A train leaves at 3:40 pm and travels 210 km at 84 km/h. At what time does it arrive? Show your steps.",
    "Explain the difference between a mutex and a semaphore, with a short example of each in C.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Implement binary search in Rust on a sorted slice of i32 and add unit tests.",
    "What is the derivative of x^3 * ln(x)? Explain each step.",
    "Write a haiku about autumn leaves, then explain the imagery you chose.",
    'Convert this JSON to YAML: {"name": "ferrum", "version": 9, "features": ["metal", "speculation"]}',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--prompts", type=int, default=len(PROMPTS))
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--out", help="write per-prompt results as JSON lines")
    a = p.parse_args()
    total_n = total_ms = drafted = accepted = 0
    rows = []
    for i, prompt in enumerate(PROMPTS[: a.prompts]):
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.tokens,
            "temperature": a.temperature,
            "top_k": 20 if a.temperature > 0 else 1,
            "top_p": 0.95 if a.temperature > 0 else 1.0,
            "seed": 7,
            "stream": False,
        }
        if a.no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(
            a.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.load(r)
        t = resp.get("timings", {})
        n, ms = t.get("predicted_n", 0), t.get("predicted_ms", 0.0)
        dn, da = t.get("draft_n", 0) or 0, t.get("draft_n_accepted", 0) or 0
        total_n += n
        total_ms += ms
        drafted += dn
        accepted += da
        rows.append({"prompt": i, "tokens": n, "ms": ms, "draft_n": dn, "draft_accepted": da})
        print(
            f"prompt {i}: {n} tokens, {n / max(ms, 1e-9) * 1e3:.2f} tok/s, "
            f"drafts {da}/{dn}",
            flush=True,
        )
    print(
        f"\ndecode {total_n / max(total_ms, 1e-9) * 1e3:.2f} tok/s over {total_n} tokens; "
        f"draft acceptance {accepted}/{drafted} "
        f"({100 * accepted / max(drafted, 1):.1f}%)"
    )
    if a.out:
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    sys.exit(main())
