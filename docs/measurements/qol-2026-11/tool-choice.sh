#!/bin/sh
# tool_choice enforcement through ferrum-server.
# usage: tool-choice.sh MODEL.gguf   (server binary in target/release)
set -u
PORT=18394
./target/release/ferrum-server -m "$1" --port $PORT >"${TMPDIR:-/tmp}/tool-choice.log" 2>&1 &
PID=$!
for i in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 0.5; done
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Current weather in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},{"type":"function","function":{"name":"get_time","description":"Current time in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
show='import sys,json
d=json.load(sys.stdin)
if "error" in d: print("ERROR", d["error"].get("code"), d["error"]["message"]); sys.exit()
m=d["choices"][0]["message"]
calls=[(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []]
print("finish", d["choices"][0]["finish_reason"], "| reasoning", d["usage"].get("completion_tokens_details",{}).get("reasoning_tokens"), "tok | content", repr((m.get("content") or "")[:60]), "| calls", calls)'
chat() { # label extra-json
  printf '%s\n   ' "$1"
  curl -s localhost:$PORT/v1/chat/completions -H 'content-type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello to me in French, nothing else.\"}],\"tools\":$TOOLS,\"max_tokens\":400,\"temperature\":0$2}" | python3 -c "$show"
}
OFF=',"chat_template_kwargs":{"enable_thinking":false}'
chat "1 auto, thinking off" "$OFF"
chat "2 required, thinking off" ",\"tool_choice\":\"required\"$OFF"
chat "3 required, thinking on" ',"tool_choice":"required"'
chat "4 named get_time, thinking off" ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}}$OFF"
chat "5 named get_time, thinking on" ',"tool_choice":{"type":"function","function":{"name":"get_time"}}'
chat "6 required, thinking budget 16" ',"tool_choice":"required","reasoning_budget":16'
chat "7 named tool that is not in tools" ',"tool_choice":{"type":"function","function":{"name":"nope"}}'
printf '8 required, streamed, thinking off\n   '
curl -sN localhost:$PORT/v1/chat/completions -H 'content-type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Say hello to me in French, nothing else.\"}],\"tools\":$TOOLS,\"max_tokens\":400,\"temperature\":0,\"stream\":true,\"tool_choice\":\"required\"$OFF}" | python3 -c '
import sys,json
name="";args="";fin=None
for l in sys.stdin:
    if not l.startswith("data: {"): continue
    c=json.loads(l[6:])["choices"]
    if not c: continue
    for t in c[0]["delta"].get("tool_calls") or []:
        name+=t["function"].get("name") or ""; args+=t["function"].get("arguments") or ""
    fin=c[0].get("finish_reason") or fin
print("finish", fin, "| call", name, args)'
printf '9 Anthropic /v1/messages, tool_choice any\n   '
curl -s localhost:$PORT/v1/messages -H 'content-type: application/json' -d '{"max_tokens":1500,"temperature":0,"messages":[{"role":"user","content":"Say hello to me in French, nothing else."}],"tools":[{"name":"get_weather","description":"Current weather in a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":{"type":"any"}}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print("stop", d.get("stop_reason"), "|", [(b["type"], b.get("name"), b.get("input")) for b in d.get("content",[]) if b["type"]!="thinking"])'
kill -TERM $PID; wait $PID 2>/dev/null
grep -o 'WRN.*' "${TMPDIR:-/tmp}/tool-choice.log" | sort | uniq -c
