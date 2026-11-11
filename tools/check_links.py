#!/usr/bin/env python3
"""Check every relative link and #anchor in the repository's markdown.

usage: python3 tools/check_links.py [ROOT]

Scans tracked and untracked (not ignored) *.md files. External links
(anything with a scheme) are not fetched. Anchors use GitHub's heading
slugs, including the -1, -2 suffixes for repeated headings. Exits 1 if any
link points at a missing file or heading.
"""
import os
import re
import subprocess
import sys


def slug(heading):
    heading = heading.strip().replace("`", "").lower()
    heading = re.sub(r"[^\w\- ]", "", heading)
    return heading.replace(" ", "-")


def headings(path, cache={}):
    if path not in cache:
        found, counts, in_code = set(), {}, False
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("```"):
                    in_code = not in_code
                if in_code:
                    continue
                m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
                if m:
                    s = slug(m.group(1))
                    n = counts.get(s, 0)
                    counts[s] = n + 1
                    found.add(s if n == 0 else f"{s}-{n}")
        cache[path] = found
    return cache[path]


def main():
    os.chdir(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    problems = []
    for f in files:
        if not os.path.exists(f):
            continue  # deleted in the working tree
        with open(f, encoding="utf-8") as fh:
            text = re.sub(r"```.*?```", "", fh.read(), flags=re.S)
        for m in re.finditer(r"\]\(([^)\s]+)\)", text):
            target = m.group(1)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue
            path, _, anchor = target.partition("#")
            resolved = os.path.normpath(os.path.join(os.path.dirname(f), path)) if path else f
            if not os.path.exists(resolved):
                problems.append(f"{f}: missing {target}")
            elif anchor and resolved.endswith(".md") and anchor not in headings(resolved):
                problems.append(f"{f}: no heading for {target}")
    for p in sorted(problems):
        print(p)
    print(f"{len(problems)} broken links in {len(files)} markdown files")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
