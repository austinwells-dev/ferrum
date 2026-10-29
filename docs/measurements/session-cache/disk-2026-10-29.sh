#!/bin/zsh
# Session cache on disk: hybrid_session_disk with filler for 2k, 8k and 32k tokens on
# every model family the TUI chats with. Each row: prefill, save, reload of the
# model, load of the cache, the same request again (identical output, full
# reuse), then turn 2 from disk against turn 2 in memory, the incremental
# save after it, and a save after the conversation got shorter (it must load).
# Then q8 K/V on four models.
# Run from the repo root after cargo build --release --example hybrid_session_disk:
#   caffeinate -dims zsh docs/measurements/session-cache/disk-2026-10-29.sh > docs/measurements/session-cache/disk-2026-10-29.txt 2>&1
set -u
LFM3=/Users/austinwells/Models/hub/models--LiquidAI--LFM3-VL-2B-GGUF/snapshots/main/LFM3-VL-2B-Q4_K_M.gguf
QWEN36=/Users/austinwells/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf
TIEL=/Users/austinwells/models/hub/models--peculiar-ragdoll--Tiel-Coder-35B-A3B-GGUF-MTP/snapshots/bbe9e566f39e4fc9652ac66b71968289a03c520a/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf
SWIFT=/Users/austinwells/models/hub/models--ukisai--Swift-1.5-Qwen3.8-27B-GGUF/snapshots/a1614465cfa35d04d3e8575d713fa779662b5eab/Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf
SPARK=/Users/austinwells/Models/Spark-X2.6-4B-GGUF/Spark-X2.6-4B-Q4_K_M.gguf
DFLASH2=/Users/austinwells/.cache/huggingface/hub/models--incoai--Qwen3.6-35B-A3B-DFlash2
DIR=${TMPDIR:-/tmp}/ferrum-session-disk
lock() { ioreg -n Root -d1 | grep -o '"IOConsoleLocked" = [A-Za-z]*'; }
echo "# start $(date -u '+%Y-%m-%d %H:%M:%S UTC'); macOS $(sw_vers -productVersion); commit $(git rev-parse --short HEAD) + working tree"
sysctl -n machdep.cpu.brand_string hw.memsize
diskutil info / | grep -E 'Solid State|Device / Media Name'
PHASE7A_PREFLIGHT_IGNORE=dasd perl -e 'alarm shift; exec @ARGV' 900 python3 scripts/phase7a_cpu_preflight.py
echo "preflight exit $?"
run() {
  local name=$1; shift
  echo "== $name   ($(lock))"
  if [[ $(lock) == *Yes* ]]; then echo "SCREEN LOCKED: run invalid, stopping"; exit 3; fi
  target/release/examples/hybrid_session_disk "$@" --dir $DIR 2>&1 | grep -E '^\||PASS|FAIL|memory:|disk:|Error'
  echo "exit ${pipestatus[1]}"
}
run "LFM3-VL-2B" $LFM3
run "Spark-X2.6-4B" $SPARK
run "Qwen3.6-35B-A3B" $QWEN36
run "Tiel-Coder-35B-A3B" $TIEL
run "Tiel-Coder-35B-A3B + DFlash2 (--draft-max 2)" $TIEL --draft $DFLASH2 --draft-max 2
run "Tiel-Coder-35B-A3B + MTP" $TIEL --draft mtp
run "Swift-1.5-Qwen3.8-27B" $SWIFT
# The same with the 8-bit K/V cache, whose storage is planar (values, then scales).
for m in $LFM3 $SPARK $QWEN36 $SWIFT; do
  FERRUM_KV_CACHE=q8 run "${m:t:r} with FERRUM_KV_CACHE=q8" $m --lengths 2000,8000
done
rm -rf $DIR
echo "# end $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
