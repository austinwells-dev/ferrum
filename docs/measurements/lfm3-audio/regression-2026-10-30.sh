#!/bin/zsh
# LFM3-Audio input regression vs experimental 8cb5110: greedy, memory, KLD, A-B-B-A bench, LFM3-VL image and text.
# This is the script as it ran on 2026-10-30.
# Run under: caffeinate -dims zsh regression-2026-10-30.sh > regression-2026-10-30.txt 2>&1
#
# Setup it expects in its own directory W (it creates none of it):
#   git worktree add --detach $W/base 8cb5110
#   git worktree add --detach $W/branch 4a2fae3
#   (cd $W/<each> && cargo build --release --bin ferrum --bin ferrum-cli --example hybrid_bench --example hybrid_kld)
#   mkdir -p $W/out; cp docs/measurements/parallel-slots-2026-10/concurrency-tiel.png $W/picture.png
#   corpus.txt, in this order (chunks are positional, so order matters):
#     (git show 8cb5110:README.md; for f in $(git ls-tree --name-only 8cb5110 docs/ | grep '\.md$' | sort); do git show 8cb5110:$f; done) > $W/corpus.txt
# Model and repo paths below are this machine's.
#
set -u
W=${0:A:h}
REPO=/Users/austinwells/Documents/ferrum
SWIFT=/Users/austinwells/models/hub/models--ukisai--Swift-1.5-Qwen3.8-27B-GGUF/snapshots/a1614465cfa35d04d3e8575d713fa779662b5eab/Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf
TIEL=/Users/austinwells/models/hub/models--peculiar-ragdoll--Tiel-Coder-35B-A3B-GGUF-MTP/snapshots/bbe9e566f39e4fc9652ac66b71968289a03c520a/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf
MODELS=($SWIFT $TIEL)
PROMPT='The capital of France is'

lock() { ioreg -n Root -d1 | grep -o '"IOConsoleLocked" = [A-Za-z]*'; }
check_unlocked() {
  local l=$(lock)
  echo "   ($l)"
  if [[ $l == *Yes* ]]; then echo "SCREEN LOCKED: run invalid, stopping"; exit 3; fi
}
bin() { echo $W/$1/target/release/$2; }

echo "# start $(date -u '+%Y-%m-%d %H:%M:%S UTC'); macOS $(sw_vers -productVersion); $(lock)"
echo "base = experimental 8cb5110, branch = feat/lfm3-audio-input 4a2fae3"
sysctl -n machdep.cpu.brand_string hw.memsize
echo "## preflight"
PHASE7A_PREFLIGHT_IGNORE=dasd perl -e 'alarm shift; exec @ARGV' 900 python3 $REPO/scripts/phase7a_cpu_preflight.py
echo "preflight exit $?"

echo "## greedy text: ferrum hybrid --model MODEL --raw --prompt '$PROMPT' --max-new-tokens 48"
for m in $MODELS; do
  for t in base branch; do
    echo "== $t ${m:t}"; check_unlocked
    $(bin $t ferrum) hybrid --model $m --raw --prompt "$PROMPT" --max-new-tokens 48 \
      > $W/out/greedy-$t-${m:t:r}.txt 2> $W/out/greedy-$t-${m:t:r}.err
    echo "exit $?"
    grep 'measured:' $W/out/greedy-$t-${m:t:r}.err $W/out/greedy-$t-${m:t:r}.txt | sed 's/^[^:]*://'
    echo "stdout md5 $(md5 -q $W/out/greedy-$t-${m:t:r}.txt) bytes $(wc -c < $W/out/greedy-$t-${m:t:r}.txt)"
  done
  cmp -s $W/out/greedy-{base,branch}-${m:t:r}.txt && echo "greedy output identical" || echo "GREEDY OUTPUT DIFFERS"
done

echo "## KLD against llama.cpp ($(llama-perplexity --version 2>&1 | grep -m1 version)), corpus = README.md + docs/*.md at 8cb5110, -c 512, 8 chunks"
for m in $MODELS; do
  k=$W/out/${m:t:r}.kld
  check_unlocked
  llama-perplexity -m $m -f $W/corpus.txt -c 512 --chunks 8 -ngl 99 --kl-divergence-base $k > $W/out/llama-${m:t:r}.log 2>&1
  echo "llama-perplexity exit $? for ${m:t:r}"
  for t in base branch; do
    echo "== $t ${m:t:r}"; check_unlocked
    $(bin $t examples/hybrid_kld) $m $k 8 2>&1 | grep -E '^(chunk|positions)'
  done
  rm -f $k
done

echo "## hybrid_bench --pp 512,2048 --tg 128, A-B-B-A x2"
for m in $MODELS; do
  echo "### ${m:t}"
  for t in base branch branch base base branch branch base; do
    echo "-- $t"; check_unlocked
    $(bin $t examples/hybrid_bench) $m --pp 512,2048 --tg 128 2>&1 | grep -E '\| (pp|tg)[0-9]+ \|'
  done
done

# The branch rewrote how image rows are spliced into the prompt (expand_media), so check an
# LFM3-VL picture and plain LFM3 text through the same front ends on both builds.
VL=$HOME/models/hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots/main
echo "## LFM3-VL image, greedy: ferrum-cli --mmproj ... --temperature 0, /image picture.png, 'Describe this picture.'"
for t in base branch; do
  echo "== $t"; check_unlocked
  printf '/image %s\nDescribe this picture.\n/exit\n' $W/picture.png | \
    $(bin $t ferrum-cli) --model $VL/LFM3-VL-2B-Q4_K_M.gguf --mmproj $VL/mmproj-LFM3-VL-2B-Q8_0.gguf \
      --temperature 0 --max-tokens 96 > $W/out/vl-$t.txt 2> $W/out/vl-$t.err
  echo "exit $?"
  grep -v -E 'tok/s|[0-9]+(\.[0-9]+)? ?(ms|s)\b' $W/out/vl-$t.txt > $W/out/vl-$t.clean
  echo "answer md5 $(md5 -q $W/out/vl-$t.clean) lines $(wc -l < $W/out/vl-$t.clean)"
done
cmp -s $W/out/vl-{base,branch}.clean && echo "VL output identical" || echo "VL OUTPUT DIFFERS"
echo "## LFM3 text greedy: ferrum hybrid --model LFM3-VL-2B-Q4_K_M.gguf --raw --prompt '$PROMPT' --max-new-tokens 48"
for t in base branch; do
  echo "== $t"; check_unlocked
  $(bin $t ferrum) hybrid --model $VL/LFM3-VL-2B-Q4_K_M.gguf --raw --prompt "$PROMPT" --max-new-tokens 48 \
    > $W/out/lfm3-$t.txt 2> $W/out/lfm3-$t.err
  echo "exit $?; stdout md5 $(md5 -q $W/out/lfm3-$t.txt)"
done
cmp -s $W/out/lfm3-{base,branch}.txt && echo "LFM3 text identical" || echo "LFM3 TEXT DIFFERS"
echo "# end $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
