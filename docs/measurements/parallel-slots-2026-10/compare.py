import json, sys
base = json.load(open(sys.argv[1]))["results"]
for n in sys.argv[2:]:
    r = json.load(open(n))
    same = [x["content"] == y["content"] for x, y in zip(base, r["results"])]
    diff = [i for i, s in enumerate(same) if not s]
    print(f"{n}: identical to {sys.argv[1]} {sum(same)}/{len(same)} {diff or ''}; wall {r['wall']:.1f} s")
