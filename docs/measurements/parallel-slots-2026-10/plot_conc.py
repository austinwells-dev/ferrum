"""Three panels from conc.py results: total tok/s, per-user tok/s, worst TTFT.

usage: plot_conc.py NODRAFT.json DFLASH.json OUT.svg
For each user count the round with the higher total tok/s is plotted.
"""
import json, statistics, sys

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("No drafter", "#2a78d6", "circle"), ("DFlash drafter", "#eb6834", "square")]


def best(path):
    runs = json.load(open(path))
    out = {}
    for r in runs:
        if r["users"] not in out or r["aggregate_tps"] > out[r["users"]]["aggregate_tps"]:
            out[r["users"]] = r
    pts = []
    for n in sorted(out):
        r = out[n]
        pts.append((n, r["aggregate_tps"],
                    statistics.mean(x["decode_tps"] for x in r["requests"]),
                    max(x["ttft"] for x in r["requests"])))
    return pts


def nice(v):
    for step in (1, 2, 5, 10, 20, 25, 50, 100):
        if v / step <= 5:
            return step
    return 200


def panel(x0, y0, w, h, title, unit, series, idx):
    s = []
    top = max(p[idx] for pts in series for p in pts) * 1.12
    step = nice(top)
    top = step * (int(top / step) + 1)
    xs = sorted({p[0] for pts in series for p in pts})
    px = lambda n: x0 + (n - xs[0]) / (xs[-1] - xs[0]) * w
    py = lambda v: y0 + h - v / top * h
    s.append(f'<text x="{x0}" y="{y0 - 28}" font-size="15" font-weight="600" fill="{INK}">{title}</text>')
    s.append(f'<text x="{x0}" y="{y0 - 10}" font-size="12" fill="{INK2}">{unit}</text>')
    v = 0
    while v <= top + 1e-9:
        y = py(v)
        s.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        label = f"{v:g}"
        s.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" font-size="11" fill="{INK2}" text-anchor="end">{label}</text>')
        v += step
    for n in xs:
        s.append(f'<text x="{px(n):.1f}" y="{y0 + h + 18}" font-size="11" fill="{INK2}" text-anchor="middle">{n}</text>')
    s.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 36}" font-size="12" fill="{INK2}" text-anchor="middle">users at once</text>')
    for (name, color, shape), pts in zip(SERIES, series):
        path = " ".join(f"{px(p[0]):.1f},{py(p[idx]):.1f}" for p in pts)
        s.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for p in pts:
            x, y = px(p[0]), py(p[idx])
            if shape == "circle":
                s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
            else:
                s.append(f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1.5" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
    return s


def main():
    series = [best(sys.argv[1]), best(sys.argv[2])]
    W, H = 1290, 430
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="-apple-system, Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
           f'<text x="40" y="34" font-size="18" font-weight="700" fill="{INK}">Tiel-Coder-35B-A3B IQ4_XS, ferrum-server --parallel 8: concurrent users</text>',
           f'<text x="40" y="56" font-size="12" fill="{INK2}">128 greedy tokens per user, best of 2 rounds, Apple M5 32 GB. The drafter runs on slot 0 only; other users decode without it.</text>']
    lx = 40
    for name, color, shape in SERIES:
        if shape == "circle":
            out.append(f'<circle cx="{lx + 6}" cy="80" r="5" fill="{color}"/>')
        else:
            out.append(f'<rect x="{lx + 1}" y="75" width="10" height="10" rx="1.5" fill="{color}"/>')
        out.append(f'<text x="{lx + 18}" y="84" font-size="12" fill="{INK}">{name}</text>')
        lx += 150
    pw, ph, top = 330, 220, 140
    for i, (title, unit, idx) in enumerate([
            ("Total throughput", "tokens/s, all users together", 1),
            ("Each user's speed", "decode tokens/s, mean over users", 2),
            ("Slowest first token", "seconds, worst user", 3)]):
        out += panel(80 + i * (pw + 85), top, pw, ph, title, unit, series, idx)
    out.append("</svg>")
    open(sys.argv[3], "w").write("\n".join(out))


main()
