"""Concurrency sweep: N users at once against one ferrum-server --parallel 8.

usage: conc.py BIN MODEL PORT OUT.json [-- extra server args]
env: USERS (default 1,2,3,4,5,6,7,8), ROUNDS (2), TOKENS (128)

Each user streams one chat completion; a round sends N at once. Every
round's prompts start with a fresh tag so nothing is served from cache.
"""
import json, os, subprocess, sys, threading, time, urllib.request

TOPICS = [
    "how a refrigerator moves heat out of its interior",
    "the history of the printing press in Europe",
    "how TCP recovers from packet loss",
    "why bread dough rises and what gluten does",
    "how a bill becomes law in a parliamentary system",
    "the life cycle of a star like the Sun",
    "how vaccines train the immune system",
    "how a compiler turns source code into machine code",
]


def stream(port, prompt, tokens):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens, "temperature": 0, "seed": 1, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    usage = None
    timings = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            for c in ev.get("choices", []):
                if c.get("delta", {}).get("content") and first is None:
                    first = time.time() - t0
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
    end = time.time() - t0
    n = usage["completion_tokens"] if usage else None
    return {"ttft": first, "total": end, "tokens": n,
            "decode_tps": timings.get("predicted_per_second") if timings else None}


def main():
    args = sys.argv[1:]
    extra = args[args.index("--") + 1:] if "--" in args else []
    binary, model, port, out = args[:4]
    users = [int(x) for x in os.environ.get("USERS", "1,2,3,4,5,6,7,8").split(",")]
    rounds = int(os.environ.get("ROUNDS", "2"))
    tokens = int(os.environ.get("TOKENS", "128"))
    server = subprocess.Popen([binary, "--model", model, "--port", port] + extra,
                              stdout=subprocess.DEVNULL, stderr=open(out + ".log", "w"))
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                pass
            if server.poll() is not None:
                sys.exit("server exited")
            time.sleep(0.5)
        else:
            sys.exit("server never became healthy")
        # Warm-up, so the first measured round doesn't pay for pipeline setup.
        stream(port, "Say hello.", 8)
        runs = []
        tag = 0
        for n in users:
            for r in range(rounds):
                tag += 1
                res = [None] * n
                def go(i):
                    res[i] = stream(port, f"[session {tag}-{i}] Explain in detail {TOPICS[i % len(TOPICS)]}. "
                                    "Write at least 400 words.", tokens)
                t0 = time.time()
                th = [threading.Thread(target=go, args=(i,)) for i in range(n)]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                wall = time.time() - t0
                total = sum(x["tokens"] for x in res)
                run = {"users": n, "round": r, "wall": wall, "tokens": total,
                       "aggregate_tps": total / wall, "requests": res}
                runs.append(run)
                ttft = [x["ttft"] for x in res]
                print(f"users {n} round {r}: {total} tokens in {wall:.1f} s = {total / wall:.1f} tok/s; "
                      f"TTFT {min(ttft):.2f}-{max(ttft):.2f} s; per-user {[round(x['decode_tps'], 1) for x in res]}",
                      flush=True)
        json.dump(runs, open(out, "w"), indent=1)
    finally:
        server.terminate()
        server.wait()


main()
