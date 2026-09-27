#!/bin/bash
# Speculative decoding matrix: llama.cpp 9710a32 vs ferrum-server, same prompts.
SP=$SCRATCH
LS=~/Library/Caches/ferrum-phase7a/llama.cpp/build-phase9/bin/llama-server
FS=/Users/austinwells/Documents/ferrum/target/release/ferrum-server
BENCH="python3 /Users/austinwells/Documents/ferrum/tools/spec_bench.py --prompts 8 --tokens 128"
SWIFT=$(ls ~/models/hub/models--ukisai--Swift-1.5-Qwen3.8-27B-GGUF/snapshots/*/Swift*.gguf)
TIEL=$(ls ~/models/hub/models--peculiar-ragdoll--Tiel-Coder-35B-A3B-GGUF-MTP/snapshots/bbe*/Tiel*IQ4_XS.gguf)
H=~/.cache/huggingface/hub
run() { # name port cmd...
  local name=$1; shift; local port=$1; shift
  "$@" > $SP/m-$name.log 2>&1 &
  local pid=$!
  for i in $(seq 1 300); do curl -s localhost:$port/health | grep -q '"ok"' && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
  echo "== $name" | tee -a $SP/matrix.txt
  $BENCH --url http://127.0.0.1:$port --out $SP/m-$name.jsonl 2>&1 | tail -1 | tee -a $SP/matrix.txt
  kill $pid; wait $pid 2>/dev/null; sleep 3
}
: > $SP/matrix.txt
run llama-swift-base 8094 $LS -m "$SWIFT" --port 8094 -c 8192
run llama-swift-mtp 8094 $LS -m "$SWIFT" --port 8094 -c 8192 --spec-type draft-mtp --spec-draft-n-max 3
run llama-swift-dflash2 8094 $LS -m "$SWIFT" -md $SP/zlab-dflash2-q8_0.gguf --port 8094 -c 8192 --spec-type draft-dflash --spec-draft-n-max 7
run llama-swift-radix 8094 $LS -m "$SWIFT" -md $SP/radix-dspark-q8_0.gguf --port 8094 -c 8192 --spec-type draft-dspark --spec-draft-n-max 7
run llama-swift-redhat 8094 $LS -m "$SWIFT" -md $SP/redhat-dspark-q8_0.gguf --port 8094 -c 8192 --spec-type draft-dspark --spec-draft-n-max 8
run ferrum-swift-base 8093 $FS --model "$SWIFT" --port 8093 -c 8192
run ferrum-swift-mtp 8093 $FS --model "$SWIFT" --port 8093 -c 8192 --draft mtp --draft-max 3
run ferrum-swift-dflash2 8093 $FS --model "$SWIFT" --port 8093 -c 8192 --draft $H/models--z-lab--Qwen3.8-27B-DFlash2
run ferrum-swift-radix 8093 $FS --model "$SWIFT" --port 8093 -c 8192 --draft $H/models--RadixArk--Qwen3.8-27B-DSpark
run ferrum-swift-redhat 8093 $FS --model "$SWIFT" --port 8093 -c 8192 --draft $H/models--RedHatAI--Qwen3.8-27B-speculator.dspark
run llama-tiel-base 8094 $LS -m "$TIEL" --port 8094 -c 8192
run llama-tiel-dflash2 8094 $LS -m "$TIEL" -md $SP/jzinno-dflash2-q8_0.gguf --port 8094 -c 8192 --spec-type draft-dflash --spec-draft-n-max 2
run llama-tiel-dflash2-8 8094 $LS -m "$TIEL" -md $SP/jzinno-dflash2-q8_0.gguf --port 8094 -c 8192 --spec-type draft-dflash --spec-draft-n-max 8
run ferrum-tiel-base 8093 $FS --model "$TIEL" --port 8093 -c 8192
run ferrum-tiel-dflash2 8093 $FS --model "$TIEL" --port 8093 -c 8192 --draft $H/models--jzinno--Ornith-1.5-35B-A3B-DFlash2 --draft-max 2
run ferrum-tiel-dflash2-8 8093 $FS --model "$TIEL" --port 8093 -c 8192 --draft $H/models--jzinno--Ornith-1.5-35B-A3B-DFlash2 --draft-max 8
echo DONE >> $SP/matrix.txt
