#!/bin/sh
# GBNF grammars (`grammar`) through ferrum-server.
# usage: gbnf.sh MODEL.gguf   (server binary in target/release)
set -u
PORT=18396
./target/release/ferrum-server -m "$1" --port $PORT >"${TMPDIR:-/tmp}/gbnf.log" 2>&1 &
PID=$!
for i in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 0.5; done
chat() { # label prompt grammar extra
  printf '%s\n   ' "$1"
  python3 - "$2" "$3" "$4" <<'PY' | curl -s localhost:18396/v1/chat/completions -H 'content-type: application/json' -d @- | python3 -c 'import sys,json
d=json.load(sys.stdin)
if "error" in d: print("ERROR", d["error"].get("code"), d["error"]["message"]); sys.exit()
print(d["choices"][0]["finish_reason"], d["usage"]["completion_tokens"], "tok |", repr(d["choices"][0]["message"]["content"]))'
import sys, json
body = {"messages": [{"role": "user", "content": sys.argv[1]}], "max_tokens": 80, "grammar": sys.argv[2]}
body.update(json.loads(sys.argv[3]))
print(json.dumps(body))
PY
}
OFF='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
chat "1 yes/no" "Is Paris the capital of Germany? Explain in detail." 'root ::= "yes" | "no"' "$OFF"
chat "2 a date" "When did the Apollo 11 mission land on the Moon?" 'root ::= [0-9]{4} "-" [0-9]{2} "-" [0-9]{2}' "$OFF"
chat "3 a list of three colours" "Name some colours." 'root ::= item{3}
item ::= "- " [a-z]+ "\n"' "$OFF"
chat "4 thinking on, sampled (seed 7)" "Pick a chess opening move for white." 'root ::= [KQRBN]? [a-h] [1-8]' '{"temperature":0.8,"seed":7,"max_tokens":800}'
chat "5 left recursion" "hi" 'root ::= root "a" | "a"' "$OFF"
chat "6 with response_format" "hi" 'root ::= "a"' '{"response_format":{"type":"json_object"}}'
chat "7 no grammar" "Is Paris the capital of Germany? One sentence." '' "$OFF"
kill -TERM $PID; wait $PID 2>/dev/null
grep -o 'generated [0-9]* tok in [0-9.]* s ([0-9.]* tok/s)' "${TMPDIR:-/tmp}/gbnf.log" | tr '\n' ';'; echo
