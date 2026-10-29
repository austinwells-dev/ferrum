#!/bin/zsh
# Session reuse (snapshots keep sliding-window rings, pinned reply-header checkpoint)
# against experimental ab8f57e: greedy, memory, KLD, chat reuse per turn, A-B-B-A bench.
# Run under: caffeinate -dims zsh regression-2026-10-29.sh WORK > regression-2026-10-29.txt 2>&1
#
# Setup it expects in WORK (it creates none of it):
#   git worktree add --detach WORK/base ab8f57e      (plus examples/hybrid_chat_reuse.rs copied from the branch)
#   git worktree add --detach WORK/branch BRANCH_COMMIT
#   (cd WORK/<each> && cargo build --release --bin ferrum --example hybrid_bench --example hybrid_kld --example hybrid_chat_reuse)
#   mkdir -p WORK/out
#   corpus.txt, in this order (chunks are positional, so order matters):
#     (git show ab8f57e:README.md; for f in $(git ls-tree --name-only ab8f57e docs/ | grep '\.md$' | sort); do git show ab8f57e:$f; done) > WORK/corpus.txt
# Model and repo paths below are this machine's. The LFM3 KLD base file was written by
# llama.cpp v0.8.0-dev (Homebrew's 0.7.0 cannot load lfm3) for docs/measurements/lfm3.
set -u
W=$1
REPO=/Users/austinwells/Documents/ferrum
SWIFT=/Users/austinwells/models/hub/models--ukisai--Swift-1.5-Qwen3.8-27B-GGUF/snapshots/a1614465cfa35d04d3e8575d713fa779662b5eab/Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf
TIEL=/Users/austinwells/models/hub/models--peculiar-ragdoll--Tiel-Coder-35B-A3B-GGUF-MTP/snapshots/bbe9e566f39e4fc9652ac66b71968289a03c520a/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf
QWEN36=/Users/austinwells/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf
LFM3VL=/Users/austinwells/Models/hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots/main
LFM3TEXT=/Users/austinwells/Models/hub/models--LiquidAI--LFM3-1.7B-GGUF/snapshots/main/LFM3-1.7B-Q4_K_M.gguf
LFM3KLD=/private/tmp/lfm3-measure/base.kld
SPARK=/Users/austinwells/Models/Spark-X2.6-4B-GGUF/Spark-X2.6-4B-Q4_K_M.gguf
DFLASH2=/Users/austinwells/.cache/huggingface/hub/models--incoai--Qwen3.6-35B-A3B-DFlash2
PICTURE=$(ls ~/Pictures/Ferrum/*.png | head -1)
PROMPT='The capital of France is'

lock() { ioreg -n Root -d1 | grep -o '"IOConsoleLocked" = [A-Za-z]*'; }
check_unlocked() {
  local l=$(lock)
  echo "   ($l)"
  if [[ $l == *Yes* ]]; then echo "SCREEN LOCKED: run invalid, stopping"; exit 3; fi
}
bin() { echo $W/$1/target/release/$2; }

echo "# start $(date -u '+%Y-%m-%d %H:%M:%S UTC'); macOS $(sw_vers -productVersion); $(lock)"
echo "base = experimental ab8f57e, branch = $(git -C $W/branch rev-parse --short HEAD)"
sysctl -n machdep.cpu.brand_string hw.memsize
echo "## preflight"
PHASE7A_PREFLIGHT_IGNORE=dasd perl -e 'alarm shift; exec @ARGV' 900 python3 $REPO/scripts/phase7a_cpu_preflight.py
echo "preflight exit $?"

echo "## greedy text and memory: ferrum hybrid --model MODEL --raw --prompt '$PROMPT' --max-new-tokens 48"
for m in $SWIFT $TIEL $LFM3VL/LFM3-VL-2B-Q4_K_M.gguf $SPARK; do
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

echo "## KLD against llama.cpp ($(llama-perplexity --version 2>&1 | grep -m1 version)), corpus = README.md + docs/*.md at ab8f57e, -c 512, 8 chunks"
for m in $SWIFT $TIEL; do
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
echo "## KLD LFM3-1.7B against the docs/measurements/lfm3 base file (1 chunk)"
for t in base branch; do
  echo "== $t ${LFM3TEXT:t:r}"; check_unlocked
  $(bin $t examples/hybrid_kld) $LFM3TEXT $LFM3KLD 1 2>&1 | grep -E '^(chunk|positions)'
done

echo "## chat reuse per turn: hybrid_chat_reuse (5 turns, tool round, 600 new tokens per turn, temperature 0)"
reuse() {
  local name=$1; shift
  for t in base branch; do
    echo "== $t $name"; check_unlocked
    $(bin $t examples/hybrid_chat_reuse) "$@" 2>&1 | grep -E '^\||^lost'
  done
}
reuse "LFM3-VL-2B with a picture" $LFM3VL/LFM3-VL-2B-Q4_K_M.gguf --mmproj $LFM3VL/mmproj-LFM3-VL-2B-Q8_0.gguf --image $PICTURE
reuse "LFM3-VL-2B text" $LFM3VL/LFM3-VL-2B-Q4_K_M.gguf
reuse "Qwen3.6-35B-A3B" $QWEN36
reuse "Tiel-Coder-35B-A3B + DFlash2 (--draft-max 2)" $TIEL --draft $DFLASH2 --draft-max 2
reuse "Swift-1.5-Qwen3.8-27B" $SWIFT --turns 4 --max-tokens 300

echo "## hybrid_bench --pp 512,2048 --tg 128, A-B-B-A x2"
for m in $SWIFT $TIEL $LFM3VL/LFM3-VL-2B-Q4_K_M.gguf; do
  echo "### ${m:t}"
  for t in base branch branch base base branch branch base; do
    echo "-- $t"; check_unlocked
    $(bin $t examples/hybrid_bench) $m --pp 512,2048 --tg 128 2>&1 | grep -E '\| (pp|tg)[0-9]+ \|'
  done
done
echo "# end $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
