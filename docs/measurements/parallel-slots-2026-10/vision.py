"""Per-request image sizes: the same picture at several budgets.

usage: vision.py BIN MODEL MMPROJ PORT OUT.json [--base]
--base sends only the default request (the base branch has no budgets).
"""
import base64, json, struct, subprocess, sys, time, urllib.error, urllib.request, zlib


def png(w, h):
    # A red disc on a blue-and-white grid, so the picture has detail to lose.
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            if (x - w * 0.4) ** 2 + (y - h * 0.5) ** 2 < (h * 0.3) ** 2:
                row += bytes((220, 30, 30))
            elif (x // 64 + y // 64) % 2:
                row += bytes((40, 70, 200))
            else:
                row += bytes((250, 250, 250))
        rows.append(bytes(row))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


URL = "data:image/png;base64," + base64.b64encode(png(1600, 1200)).decode()

CASES = {
    "default": ({}, {}),
    "detail low": ({"detail": "low"}, {}),
    "detail high": ({"detail": "high"}, {}),
    "max_tokens 300": ({"max_tokens": 300}, {}),
    "request image_detail low": ({}, {"image_detail": "low"}),
    "request image_max_tokens 400": ({}, {"image_max_tokens": 400}),
    "part wins over request": ({"detail": "high"}, {"image_detail": "low"}),
    "above cap clamps": ({"max_tokens": 100000}, {}),
    "invalid detail": ({"detail": "huge"}, {}),
    "zero max_tokens": ({"max_tokens": 0}, {}),
}


def ask(port, part_extra, body_extra):
    image = {"url": URL}
    image.update(part_extra)
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": image},
            {"type": "text", "text": "Describe this picture in one sentence."}]}],
        "max_tokens": 40, "temperature": 0, "seed": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body.update(body_extra)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r = json.loads(r.read())
        return {"prompt_tokens": r["usage"]["prompt_tokens"], "content": r["choices"][0]["message"]["content"]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode()[:200]}


def main():
    binary, model, mmproj, port, out = sys.argv[1:6]
    cases = {"default": CASES["default"]} if "--base" in sys.argv else CASES
    server = subprocess.Popen([binary, "--model", model, "--mmproj", mmproj, "--port", port, "-c", "8192"],
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
        results = {name: ask(port, *args) for name, args in cases.items()}
        json.dump(results, open(out, "w"), indent=1)
        for name, r in results.items():
            print(f"{name:30} {r}")
    finally:
        server.terminate()
        server.wait()


main()
