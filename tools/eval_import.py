#!/usr/bin/env python3
"""Install the question sets for the TUI's intelligence benchmark.

Downloads a fixed, seeded sample of each dataset through the Hugging Face
datasets-server API (standard library only) and writes JSON lines in the
format read by src/bin/ferrum-tui/eval.rs, into ~/.config/ferrum/eval (the
folder the TUI reads; FERRUM_EVAL_DIR overrides it there):

  math.jsonl          GSM8K (MIT)            check "number"
  knowledge.jsonl     MMLU (MIT)             check "choice"
  instructions.jsonl  IFEval (Apache-2.0)    check "rules"
  code.jsonl          CRUXEval (MIT)         check "literal" (output prediction)

Items are shuffled with a fixed seed, so the first QUICK items of each file
form the quick run and a rerun of this script reproduces the same files.

usage: eval_import.py [--per-set 100] [--seed 7] [--out ~/.config/ferrum/eval]
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"

# IFEval instruction ids eval.rs can verify, with the kwargs it reads.
RULES = {
    "punctuation:no_comma": [],
    "change_case:english_lowercase": [],
    "change_case:english_capital": [],
    "length_constraints:number_words": ["relation", "num_words"],
    "length_constraints:number_paragraphs": ["num_paragraphs"],
    "detectable_format:number_bullet_lists": ["num_bullets"],
    "detectable_format:number_highlighted_sections": ["num_highlights"],
    "detectable_format:title": [],
    "detectable_format:json_format": [],
    "keywords:existence": ["keywords"],
    "keywords:forbidden_words": ["forbidden_words"],
    "keywords:letter_frequency": ["letter", "let_relation", "let_frequency"],
    "startend:end_checker": ["end_phrase"],
    "startend:quotation": [],
    "detectable_content:postscript": ["postscript_marker"],
}


def rows(dataset, config, split):
    """Every row of a split, 100 at a time."""
    out, offset = [], 0
    while True:
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split,
             "offset": offset, "length": 100})
        page = {}
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{API}?{query}", timeout=60) as r:
                    page = json.load(r)
                break
            except Exception as e:  # rate limits and transient errors
                if attempt == 4:
                    raise
                print(f"  retrying {dataset} @ {offset}: {e}", file=sys.stderr)
                time.sleep(2 + attempt * 3)
        batch = [r["row"] for r in page.get("rows", [])]
        out.extend(batch)
        offset += len(batch)
        if not batch or offset >= page.get("num_rows_total", 0):
            return out


def gsm8k():
    for i, r in enumerate(rows("openai/gsm8k", "main", "test")):
        answer = r["answer"].split("####")[-1].strip().replace(",", "")
        yield {"id": f"gsm8k-{i}", "cat": "math", "check": "number",
               "answer": answer, "prompt": r["question"].strip()}


def mmlu():
    for i, r in enumerate(rows("cais/mmlu", "all", "test")):
        options = "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"]))
        subject = r["subject"].replace("_", " ")
        yield {"id": f"mmlu-{i}", "cat": "knowledge", "check": "choice",
               "answer": "ABCD"[r["answer"]],
               "prompt": f"A question about {subject}.\n\n{r['question'].strip()}\n\n{options}"}


def ifeval():
    for r in rows("google/IFEval", "default", "train"):
        ids, kwargs = r["instruction_id_list"], r["kwargs"]
        if not all(i in RULES for i in ids):
            continue
        rules = []
        for rid, kw in zip(ids, kwargs):
            rule = {"id": rid}
            for k in RULES[rid]:
                if kw.get(k) is None:
                    break
                rule[k] = kw[k]
            else:
                rules.append(rule)
                continue
            break
        if len(rules) == len(ids):
            yield {"id": f"ifeval-{r['key']}", "cat": "instructions",
                   "check": "rules", "rules": rules, "prompt": r["prompt"]}


def cruxeval():
    for r in rows("cruxeval-org/cruxeval", "default", "test"):
        yield {"id": f"crux-{r['id']}", "cat": "code", "check": "literal",
               "answer": r["output"],
               "prompt": f"Here is a Python function:\n\n```python\n{r['code']}\n```\n\n"
                         f"What does `f({r['input']})` return?"}


SETS = {"math": gsm8k, "knowledge": mmlu, "instructions": ifeval, "code": cruxeval}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-set", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=os.path.expanduser("~/.config/ferrum/eval"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, source in SETS.items():
        print(f"{name}: downloading", file=sys.stderr)
        items = list(source())
        random.Random(f"{a.seed}-{name}").shuffle(items)
        items = items[: a.per_set]
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{name}: wrote {len(items)} items", file=sys.stderr)


if __name__ == "__main__":
    main()
