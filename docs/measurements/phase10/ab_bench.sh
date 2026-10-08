#!/bin/zsh
# A/B: main (F16) vs branch (F16, Q8). Output: ab_*.txt in the scratchpad.
SP=$SCRATCH
source $SP/env.sh
MAIN=$SP/main-target/release/examples
BR=$REPO/target/release/examples
run() { # label bin kv args...
  local label=$1 bin=$2 kv=$3; shift 3
  echo "### $label (kv $kv)"
  FERRUM_KV_CACHE=$kv $bin "$@" 2>&1 | grep -vE "^\s*$"
}
{
for rep in 1 2; do
  run "main rep$rep" $MAIN/hybrid_bench f16 "$TIEL" --pp 512 --tg 0 --reps 1 --depth 0,8192,32768,65536 --pp-depth 32768
  run "branch rep$rep" $BR/hybrid_bench f16 "$TIEL" --pp 512 --tg 0 --reps 1 --depth 0,8192,32768,65536 --pp-depth 32768
  run "branch-q8 rep$rep" $BR/hybrid_bench q8 "$TIEL" --pp 512 --tg 0 --reps 1 --depth 0,8192,32768,65536 --pp-depth 32768
done
} > $SP/ab_bench.txt 2>&1
{
for depth in 0 32768; do
  ctx=$((depth + 8192))
  run "main depth $depth" $MAIN/hybrid_spec f16 "$TIEL" --drafter mtp --prompts 3 --tokens 192 --baseline --context $ctx --depth $depth --filler $SP/corpus.txt
  run "branch depth $depth" $BR/hybrid_spec f16 "$TIEL" --drafter mtp --prompts 3 --tokens 192 --baseline --context $ctx --depth $depth --filler $SP/corpus.txt
  run "branch-q8 depth $depth" $BR/hybrid_spec q8 "$TIEL" --drafter mtp --prompts 3 --tokens 192 --baseline --context $ctx --depth $depth --filler $SP/corpus.txt
done
} > $SP/ab_spec.txt 2>&1
echo done > $SP/ab_done
