"""Greedy outputs of a ferrum-server: sequential or concurrent requests.

usage: check.py BIN MODEL PORT OUT.json [--concurrent] [-- extra server args]
"""
import json, subprocess, sys, threading, time, urllib.request

PROMPTS = [
    "Write a Python function that checks whether a number is prime.",
    "Explain in three sentences why the sky is blue.",
    "List five capital cities in Europe and one fact about each.",
    "What is 17 * 23? Show your working.",
    "Write a haiku about a lighthouse.",
    "Translate 'good morning, how are you?' into French, German and Spanish.",
    "Give me a JSON object describing a cat with name, age and colour.",
    "Summarise the plot of Romeo and Juliet in two sentences.",
    # Several prompt chunks, so prefill interleaves with the others' decoding.
    " ".join(f"Item {i}: the {['red','green','blue','amber'][i % 4]} crate holds {i * 7 % 31} bolts." for i in range(160))
    + " Which crate numbers hold exactly 12 bolts? Answer with the numbers only.",
]


def post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        return json.loads(r.read())


def ask(port, prompt, tokens, extra=None, history=()):
    body = {
        "messages": list(history) + [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "temperature": 0,
        "seed": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body.update(extra or {})
    r = post(port, "/v1/chat/completions", body)
    return {
        "content": r["choices"][0]["message"]["content"],
        "prompt_tokens": r["usage"]["prompt_tokens"],
        "cached": r["usage"]["prompt_tokens_details"]["cached_tokens"],
        "tps": r["timings"]["predicted_per_second"],
    }


def main():
    args = sys.argv[1:]
    extra_args = args[args.index("--") + 1 :] if "--" in args else []
    args = args[: args.index("--")] if "--" in args else args
    binary, model, port, out = args[:4]
    concurrent = "--concurrent" in args
    tokens = int(__import__("os").environ.get("TOKENS", "96"))
    server = subprocess.Popen(
        [binary, "--model", model, "--port", port] + extra_args,
        stdout=subprocess.DEVNULL,
        stderr=open(out + ".log", "w"),
    )
    try:
        for _ in range(600):
            try:
                if get(port, "/health").get("status") == "ok":
                    break
            except Exception:
                pass
            if server.poll() is not None:
                sys.exit("server exited")
            time.sleep(0.5)
        results = [None] * len(PROMPTS)
        start = time.time()
        if concurrent:
            def run(i):
                results[i] = ask(port, PROMPTS[i], tokens)
            threads = [threading.Thread(target=run, args=(i,)) for i in range(len(PROMPTS))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for i, p in enumerate(PROMPTS):
                results[i] = ask(port, p, tokens)
        wall = time.time() - start
        # A retry of the first prompt should reuse its cache, and so should a
        # second turn of the long one.
        retry = ask(port, PROMPTS[0], tokens)
        follow = ask(port, "Now list the amber crates among them.", tokens, history=[
            {"role": "user", "content": PROMPTS[-1]},
            {"role": "assistant", "content": results[-1]["content"]}])
        retry["follow"] = follow
        slots = get(port, "/slots")
        json.dump(
            {"results": results, "wall": wall, "retry": retry, "slots": slots},
            open(out, "w"),
            indent=1,
        )
        print(f"{out}: {len(PROMPTS)} requests in {wall:.1f} s; retry cached {retry['cached']}/{retry['prompt_tokens']}; "
              f"second turn cached {follow['cached']}/{follow['prompt_tokens']}")
    finally:
        server.terminate()
        server.wait()


main()
