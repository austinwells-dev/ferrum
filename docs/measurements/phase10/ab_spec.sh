#!/bin/zsh
SP=$SCRATCH
source $SP/env.sh
MAIN=$SP/main-target/release/examples
BR=$REPO/target/release/examples
run() { local label=$1 bin=$2 kv=$3; shift 3; echo "### $label (kv $kv)"; FERRUM_KV_CACHE=$kv $bin "$@" 2>&1 | grep -vE "^\s*$"; }
for cfg in "0 0" "32768 0" "32768 8"; do
  depth=${cfg% *}; drafts=${cfg#* }; ctx=$((depth + 8192))
  for who in main branch branch-q8; do
    case $who in main) bin=$MAIN; kv=f16;; branch) bin=$BR; kv=f16;; branch-q8) bin=$BR; kv=q8;; esac
    run "$who depth $depth drafts $drafts" $bin/hybrid_spec $kv "$TIEL" --drafter "$DFLASH" --drafts $drafts --prompts 2 --tokens 192 --baseline --context $ctx --depth $depth --filler $SP/corpus.txt
  done
done > $SP/ab_spec.txt 2>&1
