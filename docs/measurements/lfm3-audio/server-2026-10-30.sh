#!/bin/zsh
# ferrum-server with the LFM3-Audio pair: /props modalities and one input_audio transcription.
# Run from the repo root after cargo build --release --bin ferrum-server, with S as in parity-2026-10-30.sh:
#   zsh docs/measurements/lfm3-audio/server-2026-10-30.sh > docs/measurements/lfm3-audio/server-2026-10-30.txt 2>&1
set -u
S=${S:-/private/tmp/claude-501/-Users-austinwells-Documents-ferrum/63c705ed-5e6d-4036-805c-1f0616b0bc55/scratchpad}
M=$HOME/models/hub/models--LiquidAI--LFM3-Audio-2B-GGUF/snapshots/main
PORT=18089
echo "# $(date -u '+%Y-%m-%d %H:%M:%S UTC'); commit $(git rev-parse --short HEAD)"
target/release/ferrum-server --model $M/LFM3-Audio-2B-Q4_K_M.gguf --mmproj $M/mmproj-LFM3-Audio-2B-Q8_0.gguf \
  --port $PORT > $S/server.log 2>&1 &
PID=$!
# /props answers while the model is still loading; /health says ok once it is ready.
until curl -sf localhost:$PORT/health > /dev/null; do sleep 1; done
echo "## /props modalities"
curl -s localhost:$PORT/props | python3 -c 'import json,sys; print(json.load(sys.stdin)["modalities"])'
echo "## chat completion: system 'Perform ASR in English.', user = weather.wav as input_audio"
python3 - $S/audio/weather.wav > $S/req-asr.json <<'PY'
import base64, json, sys
data = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
print(json.dumps({"messages": [
    {"role": "system", "content": "Perform ASR in English."},
    {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]}],
  "temperature": 0, "max_tokens": 96}))
PY
curl -s localhost:$PORT/v1/chat/completions -H 'content-type: application/json' -d @$S/req-asr.json | \
  python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"]); print("prompt tokens", r["usage"]["prompt_tokens"])'
kill $PID; wait $PID 2>/dev/null
echo "## server log lines about audio"
grep -i audio $S/server.log
