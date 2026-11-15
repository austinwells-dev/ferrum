#!/bin/sh
# Interleaved A-B-B-A timing of feat/smooth-mode-and-qol against experimental.
# usage: abba.sh BASE_DIR   (both checkouts built with
#   cargo build --release --bins --example hybrid_bench)
# Two measurements per run:
#   bench: examples/hybrid_bench --pp 512,2048 --tg 128 (forward passes alone)
#   chat:  ferrum-cli on a piped ~1000-token prompt, greedy, 200 tokens: the
#          Session path the branch changes (its stats line: prefill and
#          decode tok/s).
set -u
BASE=$1
SPARK=$HOME/Models/Spark-X2.6-4B-GGUF/Spark-X2.6-4B-Q4_K_M.gguf
MOE=$HOME/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf
PROMPT="Continue this story for a long time, in the same style, without stopping. $(printf 'The quick brown fox jumps over the lazy dog near the river bank at dawn. %.0s' $(seq 60))"
PHASE7A_PREFLIGHT_IGNORE=dasd perl -e 'alarm 600; exec @ARGV' python3 scripts/phase7a_cpu_preflight.py --wait || echo "preflight did not clear"
run() { # label dir model
  echo "$1 bench: $("$2/target/release/examples/hybrid_bench" "$3" --pp 512,2048 --tg 128 2>&1 | tr '\n' ' ')"
  echo "$1 chat:  $(echo "$PROMPT" | "$2/target/release/ferrum-cli" -m "$3" --context 8192 --no-think --temperature 0 --max-tokens 200 2>&1 >/dev/null | grep -o 'prompt [0-9]* tokens.*stop [A-Za-z]*')"
}
for M in "$SPARK" "$MOE"; do
  echo "== $(basename "$M")"
  for round in 1 2; do
    run "$round base  " "$BASE" "$M"
    run "$round branch" . "$M"
    run "$round branch" . "$M"
    run "$round base  " "$BASE" "$M"
  done
done
echo DONE
