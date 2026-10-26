"""Per-user cache state: 8 users, each with its own conversation.

usage: cache.py BIN MODEL PORT OUT.json [-- extra server args]

Turn 1: all users send their first message at once. Turn 2: all send a
follow-up (their history plus a new question) at once. Turn 3: user 0
alone asks again, after everyone else's turn 2. Records cached/prompt
tokens per user per turn, and /slots after each turn.
"""
import json, subprocess, sys, threading, time, urllib.request

USERS = 8
TOKENS = 48
# A long shared-style system prompt per user, distinct per user, so a cache
# miss costs a visible prefill.
def system(u):
    facts = " ".join(f"Rule {i}: user {u} prefers option {(i * 7 + u) % 13} for case {i}." for i in range(60))
    return f"You are assistant number {u}. {facts}"

QUESTIONS = ["Which option does rule 5 give?", "And rule 12?", "Summarise rules 1 to 3."]


def post(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def ask(port, history):
    r = post(port, "/v1/chat/completions", {
        "messages": history, "max_tokens": TOKENS, "temperature": 0, "seed": 1,
        "chat_template_kwargs": {"enable_thinking": False}})
    u = r["usage"]
    return r["choices"][0]["message"]["content"], u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"], r["timings"]["prompt_ms"]


def slots(port):
    out = []
    for s in post(port, "/slots"):
        out.append({k: s.get(k) for k in ("id", "is_processing", "n_cached_tokens", "prompt_cache_key") if k in s})
    return out


def main():
    args = sys.argv[1:]
    extra = args[args.index("--") + 1:] if "--" in args else []
    binary, model, port, out = args[:4]
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
        hist = [[{"role": "system", "content": system(u)}] for u in range(USERS)]
        result = {"turns": [], "slots": []}

        def turn(name, users, q):
            rows = [None] * USERS
            def go(u):
                h = hist[u] + [{"role": "user", "content": q}]
                text, p, c, ms = ask(port, h)
                hist[u] = h + [{"role": "assistant", "content": text}]
                rows[u] = {"user": u, "prompt": p, "cached": c, "prefill_ms": round(ms, 1)}
            th = [threading.Thread(target=go, args=(u,)) for u in users]
            for t in th:
                t.start()
            for t in th:
                t.join()
            rows = [r for r in rows if r]
            result["turns"].append({"turn": name, "rows": rows})
            result["slots"].append({"after": name, "slots": slots(port)})
            print(f"{name}: " + "; ".join(f"u{r['user']} {r['cached']}/{r['prompt']}" for r in rows), flush=True)

        turn("turn 1 (all at once)", range(USERS), QUESTIONS[0])
        turn("turn 2 (all at once)", range(USERS), QUESTIONS[1])
        turn("turn 3 (user 0 alone)", [0], QUESTIONS[2])
        json.dump(result, open(out, "w"), indent=1)
    finally:
        server.terminate()
        server.wait()


main()
