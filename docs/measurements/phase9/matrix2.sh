#!/bin/bash
SP=$SCRATCH
LS=~/Library/Caches/ferrum-phase7a/llama.cpp/build-phase9/bin/llama-server
BENCH="python3 /Users/austinwells/Documents/ferrum/tools/spec_bench.py --prompts 8 --tokens 128"
SWIFT=$(ls ~/models/hub/models--ukisai--Swift-1.5-Qwen3.8-27B-GGUF/snapshots/*/Swift*.gguf)
run() { local name=$1; shift; local port=$1; shift
  "$@" > $SP/m-$name.log 2>&1 & local pid=$!
  for i in $(seq 1 300); do curl -s localhost:$port/health | grep -q '"ok"' && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
  echo "== $name" >> $SP/matrix.txt
  $BENCH --url http://127.0.0.1:$port --out $SP/m-$name.jsonl 2>&1 | tail -1 >> $SP/matrix.txt
  kill $pid; wait $pid 2>/dev/null; sleep 3; }
run llama-swift-dflash2-np1 8094 $LS -m "$SWIFT" -md $SP/zlab-dflash2-q8_0.gguf --port 8094 -c 8192 -np 1 --spec-type draft-dflash --spec-draft-n-max 7
run llama-swift-radix-np1 8094 $LS -m "$SWIFT" -md $SP/radix-dspark-q8_0.gguf --port 8094 -c 8192 -np 1 --spec-type draft-dspark --spec-draft-n-max 7
run llama-swift-redhat-np1 8094 $LS -m "$SWIFT" -md $SP/redhat-dspark-q8_0.gguf --port 8094 -c 8192 -np 1 --spec-type draft-dspark --spec-draft-n-max 8
echo DONE2 >> $SP/matrix.txt
