#!/bin/sh
# Interleaved A-B-B-A check of the allocation change (weights and K/V are no
# longer zeroed a second time after Metal hands out zero-filled buffers).
# usage: load-abba.sh BIN_DIR   holding bench-base, cli-base (716e9b4) and
#   bench-new, cli-new (the change), from
#   cargo build --release --bin ferrum-cli --example hybrid_bench
# Per run:
#   bench: hybrid_bench --pp 512,2048 --tg 128 in a fresh process, so every
#          K/V page it writes is written for the first time
#   chat:  ferrum-cli at the default (largest) context on a ~1000-token
#          prompt, greedy, 200 tokens: reported load time, the stats line,
#          the whole process's wall time and the md5 of the reply
set -u
D=$1
SPARK=$HOME/Models/Spark-X2.6-4B-GGUF/Spark-X2.6-4B-Q4_K_M.gguf
MOE=$HOME/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf
PROMPT="Continue this story for a long time, in the same style, without stopping. $(printf 'The quick brown fox jumps over the lazy dog near the river bank at dawn. %.0s' $(seq 60))"
PHASE7A_PREFLIGHT_IGNORE=dasd perl -e 'alarm 600; exec @ARGV' python3 scripts/phase7a_cpu_preflight.py --wait || echo "preflight did not clear"
run() { # label suffix model
  echo "$1 bench: $("$D/bench-$2" "$3" --pp 512,2048 --tg 128 2>&1 | tr '\n' ' ')"
  start=$(perl -MTime::HiRes=time -e 'printf "%.2f", time')
  "$D/cli-$2" -m "$3" --prompt "$PROMPT" --no-think --temperature 0 --max-tokens 200 </dev/null >"$D/out.txt" 2>"$D/err.txt"
  end=$(perl -MTime::HiRes=time -e 'printf "%.2f", time')
  echo "$1 chat:  $(cat "$D/out.txt" "$D/err.txt" | grep -o 'loaded in [0-9.]* s') | $(cat "$D/out.txt" "$D/err.txt" | grep -o 'prompt [0-9]* tokens.*stop [A-Za-z]*') | wall $(echo "$end - $start" | bc) s | reply md5 $(sed 's/\x1b\[[0-9;]*m//g' "$D/out.txt" | grep -v 'loading\|loaded in\|^sampling\|^\[prompt' | md5)"
}
for M in "$SPARK" "$MOE"; do
  echo "== $(basename "$M")"
  for round in 1 2; do
    run "$round base  " base "$M"
    run "$round branch" new "$M"
    run "$round branch" new "$M"
    run "$round base  " base "$M"
  done
done
echo DONE
