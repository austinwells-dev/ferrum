#!/bin/zsh
# Tiel-Coder-35B-A3B with its own MTP block (routed experts, Q3_K experts loaded
# as Q6_K) against plain decoding and the DFlash2 drafter, on hybrid_spec's chat
# prompts, greedy, 256 tokens. Two interleaved rounds; each run is checked
# against plain decoding (--baseline).
# Run from the repo root after cargo build --release --example hybrid_spec:
#   caffeinate -dims zsh docs/measurements/mtp-moe/tiel-2026-10-29.sh > docs/measurements/mtp-moe/tiel-2026-10-29.txt 2>&1
set -u
TIEL=/Users/austinwells/models/hub/models--peculiar-ragdoll--Tiel-Coder-35B-A3B-GGUF-MTP/snapshots/bbe9e566f39e4fc9652ac66b71968289a03c520a/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf
DFLASH2=/Users/austinwells/.cache/huggingface/hub/models--incoai--Qwen3.6-35B-A3B-DFlash2
lock() { ioreg -n Root -d1 | grep -o '"IOConsoleLocked" = [A-Za-z]*'; }
echo "# start $(date -u '+%Y-%m-%d %H:%M:%S UTC'); macOS $(sw_vers -productVersion); commit $(git rev-parse --short HEAD) + working tree"
sysctl -n machdep.cpu.brand_string hw.memsize
PHASE7A_PREFLIGHT_IGNORE=dasd perl -e 'alarm shift; exec @ARGV' 900 python3 scripts/phase7a_cpu_preflight.py
echo "preflight exit $?"
for round in 1 2; do
  for run in "mtp 1" "mtp 2" "mtp 3" "$DFLASH2 2"; do
    set -- ${=run}
    echo "== round $round: --drafter ${1:t} --drafts $2   ($(lock))"
    if [[ $(lock) == *Yes* ]]; then echo "SCREEN LOCKED: run invalid, stopping"; exit 3; fi
    target/release/examples/hybrid_spec $TIEL --drafter $1 --drafts $2 --tokens 256 --baseline 2>&1 | grep -vE '^(load|plan)'
  done
done
echo "# end $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
