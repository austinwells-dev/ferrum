#!/bin/sh
# /v1/responses through ferrum-server.
# usage: responses.sh MODEL.gguf   (server binary in target/release)
set -u
PORT=18397
./target/release/ferrum-server -m "$1" --port $PORT >"${TMPDIR:-/tmp}/responses.log" 2>&1 &
PID=$!
for i in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 0.5; done
show='import sys,json
d=json.load(sys.stdin)
if "error" in d and d["error"]: print("ERROR", d["error"].get("code"), d["error"]["message"]); sys.exit()
items=[]
for o in d["output"]:
    if o["type"]=="reasoning": items.append(("reasoning", len(o["content"][0]["text"]), "chars"))
    elif o["type"]=="message": items.append(("message", o["content"][0]["text"][:70]))
    else: items.append((o["type"], o["name"], o["arguments"], o["call_id"]))
print(d["status"], d["incomplete_details"], "|", items, "|", d["usage"])'
post() { printf '%s\n   ' "$1"; curl -s localhost:$PORT/v1/responses -H 'content-type: application/json' -d "$2" | python3 -c "$show"; }
TOOLS='[{"type":"function","name":"get_time","description":"Current time in a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
post "1 string input, instructions, thinking on" '{"input":"What is 17 * 3?","instructions":"Answer with the number only.","temperature":0,"max_output_tokens":600}'
post "2 same again (cache)" '{"input":"What is 17 * 3?","instructions":"Answer with the number only.","temperature":0,"max_output_tokens":600}'
post "3 reasoning effort none, cut at 5 tokens" '{"input":"Write a long poem about the sea.","reasoning":{"effort":"none"},"temperature":0,"max_output_tokens":5}'
post "4 text.format json_schema" '{"input":"Give the capital of France.","reasoning":{"effort":"none"},"temperature":0,"text":{"format":{"type":"json_schema","name":"c","schema":{"type":"object","properties":{"capital":{"type":"string"}},"required":["capital"],"additionalProperties":false}}}}'
post "5 tool call (required)" "{\"input\":\"What time is it in Oslo?\",\"reasoning\":{\"effort\":\"none\"},\"temperature\":0,\"tools\":$TOOLS,\"tool_choice\":\"required\"}"
post "6 tool result fed back" "{\"input\":[{\"role\":\"user\",\"content\":\"What time is it in Oslo?\"},{\"type\":\"function_call\",\"call_id\":\"call_5_0\",\"name\":\"get_time\",\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\"}\"},{\"type\":\"function_call_output\",\"call_id\":\"call_5_0\",\"output\":\"14:35\"}],\"reasoning\":{\"effort\":\"none\"},\"temperature\":0,\"tools\":$TOOLS}"
post "7 previous_response_id" '{"input":"hi","previous_response_id":"resp_1"}'
printf '8 streamed, thinking on: event types in order (deltas counted), then the text\n   '
curl -sN localhost:$PORT/v1/responses -H 'content-type: application/json' -d '{"input":"What is 17 * 3?","instructions":"Answer with the number only.","temperature":0,"max_output_tokens":600,"stream":true}' | python3 -c '
import sys,json
order=[];text="";seq=[]
for l in sys.stdin:
    if not l.startswith("data: "): continue
    e=json.loads(l[6:]); seq.append(e["sequence_number"]); t=e["type"]
    if order and order[-1][0]==t: order[-1][1]+=1
    else: order.append([t,1])
    if t=="response.output_text.delta": text+=e["delta"]
    if t=="response.completed": final=e["response"]
print(" > ".join(t if n==1 else f"{t} x{n}" for t,n in order))
print("   sequence numbers consecutive:", seq==list(range(len(seq))), "| text", repr(text), "| final status", final["status"], [o["type"] for o in final["output"]])'
printf '9 streamed tool call\n   '
curl -sN localhost:$PORT/v1/responses -H 'content-type: application/json' -d "{\"input\":\"What time is it in Oslo?\",\"reasoning\":{\"effort\":\"none\"},\"temperature\":0,\"tools\":$TOOLS,\"tool_choice\":\"required\",\"stream\":true}" | python3 -c '
import sys,json
for l in sys.stdin:
    if not l.startswith("data: "): continue
    e=json.loads(l[6:])
    if e["type"]=="response.function_call_arguments.done": print("arguments", e["arguments"], end=" | ")
    if e["type"]=="response.completed": print("final", [(o["type"], o.get("name")) for o in e["response"]["output"]])'
kill -TERM $PID; wait $PID 2>/dev/null
