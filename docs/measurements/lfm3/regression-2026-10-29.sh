#!/bin/zsh
# LFM3 (#8) regression vs experimental ec14e94: greedy, memory, KLD, A-B-B-A bench.
# Run under: caffeinate -dims zsh run.sh > record.txt 2>&1
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
echo "base = experimental ec14e94 (before #8), branch = experimental d691ca6 (#8 merged)"
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

echo "## KLD against llama.cpp ($(llama-perplexity --version 2>&1 | grep -m1 version)), corpus = README.md + docs/*.md at ec14e94, -c 512, 8 chunks"
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
echo "# end $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
