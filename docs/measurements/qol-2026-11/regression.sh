#!/bin/sh
# Regression checks for feat/smooth-mode-and-qol against experimental.
# usage: regression.sh BASE_DIR   (a checkout of experimental, built with
#   cargo build --release --bins --example hybrid_bench)
# Run from the branch's checkout after
#   cargo build --release --bins --example hybrid_bench \
#     --example hybrid_session_check --example hybrid_spec_check
set -u
BASE=$1
SPARK=$HOME/Models/Spark-X2.6-4B-GGUF/Spark-X2.6-4B-Q4_K_M.gguf
MOE=$HOME/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf
PROMPT='Write a Python function that merges two sorted lists, then explain its time complexity in one paragraph.'
clean() { sed -e 's/\x1b\[[0-9;]*m//g' -e 's/^> //' | grep -v '^$'; }
for M in "$SPARK" "$MOE"; do
  echo "== greedy, $(basename "$M")"
  echo "$PROMPT" | "$BASE/target/release/ferrum-cli" -m "$M" --context 8192 --no-think --temperature 0 --max-tokens 200 2>/dev/null | clean > /tmp/ferrum-greedy-base.txt
  echo "$PROMPT" | target/release/ferrum-cli -m "$M" --context 8192 --no-think --temperature 0 --max-tokens 200 2>/dev/null | clean > /tmp/ferrum-greedy-branch.txt
  echo "base   md5 $(md5 -q /tmp/ferrum-greedy-base.txt) ($(wc -c < /tmp/ferrum-greedy-base.txt) bytes)"
  echo "branch md5 $(md5 -q /tmp/ferrum-greedy-branch.txt) ($(wc -c < /tmp/ferrum-greedy-branch.txt) bytes)"
  cmp -s /tmp/ferrum-greedy-base.txt /tmp/ferrum-greedy-branch.txt && echo IDENTICAL || echo DIFFERENT
  echo "== hybrid_session_check, $(basename "$M")"
  target/release/examples/hybrid_session_check "$M" 2>&1 | tail -12
  echo "== hybrid_spec_check, $(basename "$M")"
  target/release/examples/hybrid_spec_check "$M" 2>&1 | tail -12
done
echo DONE
