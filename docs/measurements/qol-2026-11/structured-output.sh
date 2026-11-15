#!/bin/sh
# Structured-output requests against ferrum-server on Spark-X2.6-4B Q4_K_M.
# usage: structured-output.sh   (from the repository root, after cargo build --release)
M=$HOME/Models/Spark-X2.6-4B-GGUF/Spark-X2.6-4B-Q4_K_M.gguf
PORT=18733
URL=localhost:$PORT/v1/chat/completions
CITY='{"type":"object","properties":{"city":{"type":"string"},"population_millions":{"type":"number"},"landmarks":{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":3},"continent":{"enum":["Europe","Asia","Africa","Americas","Oceania"]}},"required":["city","population_millions","landmarks","continent"]}'
SUM='{"type":"object","properties":{"answer":{"type":"integer"},"explanation":{"type":"string"}},"required":["answer","explanation"]}'
show() { python3 -c 'import sys,json
d=json.load(sys.stdin)
if "choices" not in d:
    print("  error:", d["error"]["message"]); sys.exit()
m=d["choices"][0]["message"]; c=m.get("content") or ""
try: json.loads(c); ok="valid JSON"
except Exception as e: ok="NOT JSON (%s)" % e
t=d.get("timings",{})
print("  content:", c[:400]); print("  %s; finish %s; %d tokens (%d reasoning chars) at %.1f tok/s; drafts %s accepted %s" % (ok, d["choices"][0]["finish_reason"], d["usage"]["completion_tokens"], len(m.get("reasoning_content") or ""), t.get("predicted_per_second",0), t.get("draft_n"), t.get("draft_n_accepted")))'; }
serve() { target/release/ferrum-server -m "$M" --port $PORT --context 8192 "$@" >/dev/null 2>&1 & PID=$!
  for i in $(seq 120); do curl -s localhost:$PORT/health | grep -q '"ok"' && break; sleep 0.5; done; }
serve
echo "1. schema (city), greedy, thinking off"
curl -s $URL -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Tell me about Paris.\"}],\"max_tokens\":200,\"temperature\":0,\"enable_thinking\":false,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"city\",\"schema\":$CITY}}}" | show
echo "2. json_object, greedy, thinking off"
curl -s $URL -d '{"messages":[{"role":"user","content":"Write a haiku about rust. Reply as JSON with keys title and lines."}],"max_tokens":200,"temperature":0,"enable_thinking":false,"response_format":{"type":"json_object"}}' | show
echo "3. schema (sum), thinking on, temperature 0.6 seed 5"
curl -s $URL -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Think very briefly.\"}],\"max_tokens\":900,\"temperature\":0.6,\"seed\":5,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"a\",\"schema\":$SUM}}}" | show
echo "4. unconstrained, greedy, thinking off (speed reference)"
curl -s $URL -d '{"messages":[{"role":"user","content":"Tell me about Paris in 60 words."}],"max_tokens":120,"temperature":0,"enable_thinking":false}' | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  %d tokens at %.1f tok/s" % (d["usage"]["completion_tokens"], d["timings"]["predicted_per_second"]))'
echo "5. schema with allOf (refused)"
curl -s $URL -d '{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"schema":{"allOf":[{"type":"string"}]}}}}' | show
echo "6. streamed schema (sum), greedy, thinking off"
curl -sN $URL -d "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"What is 9*9?\"}],\"max_tokens\":100,\"temperature\":0,\"enable_thinking\":false,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"a\",\"schema\":$SUM}}}" | python3 -c 'import sys,json
out=""
for l in sys.stdin:
    if l.startswith("data: ") and "[DONE]" not in l:
        for c in json.loads(l[6:]).get("choices",[]): out+=c["delta"].get("content") or ""
json.loads(out); print("  content:", out); print("  valid JSON")'
kill -INT $PID; wait $PID 2>/dev/null
serve --draft lookup
echo "7. schema (sum), thinking on, reasoning_budget 120, greedy, --draft lookup"
curl -s $URL -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Then repeat the phrase: the quick brown fox jumps over the lazy dog, the quick brown fox jumps over the lazy dog.\"}],\"max_tokens\":500,\"temperature\":0,\"reasoning_budget\":120,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"a\",\"schema\":$SUM}}}" | show
kill -INT $PID; wait $PID 2>/dev/null
echo DONE
