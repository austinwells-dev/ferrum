#!/bin/sh
# --models-dir: switching the loaded model by a request's `model`.
# usage: models-dir.sh MAIN.gguf OTHER.gguf   (server binary in target/release)
set -u
PORT=18399
DIR=$(mktemp -d "${TMPDIR:-/tmp}/models.XXXXXX")
ln -s "$1" "$DIR/small.gguf"; ln -s "$2" "$DIR/big.gguf"
./target/release/ferrum-server -m "$DIR/small.gguf" --models-dir "$DIR" --port $PORT >"$DIR/server.log" 2>&1 &
PID=$!
for i in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null && break; sleep 0.5; done
list() { curl -s localhost:$PORT/v1/models | python3 -c 'import sys,json; print("   models:", [(m["id"], m["meta"]["status"], m["meta"].get("n_ctx")) for m in json.load(sys.stdin)["data"]])'; }
chat() { # label model-json
  printf '%s\n   ' "$1"
  curl -s -w '\n%{time_total}' localhost:$PORT/v1/chat/completions -H 'content-type: application/json' -d "{$2\"messages\":[{\"role\":\"user\",\"content\":\"Name the capital of France in one word.\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | python3 -c 'import sys,json; t=sys.stdin.read().rsplit("\n",1); d=json.loads(t[0]); print("served by", d["model"], "|", repr(d["choices"][0]["message"]["content"]), "|", t[1], "s")'
}
list
chat "1 no model field" ""
chat "2 model big (switch)" '"model":"big",'
list
chat "3 model big again (no switch)" '"model":"big",'
chat "4 an unknown model name (back to the --model)" '"model":"gpt-4o",'
list
printf '5 two requests at once, big then small\n'
chat "  a" '"model":"big",' & sleep 0.3; chat "  b" '"model":"small",'; wait %2 2>/dev/null
printf '6 /tokenize "Hello world" per model\n   small %s big %s\n' "$(curl -s localhost:$PORT/tokenize -H 'content-type: application/json' -d '{"content":"Hello world","model":"small"}')" "$(curl -s localhost:$PORT/tokenize -H 'content-type: application/json' -d '{"content":"Hello world","model":"big"}')"
kill -TERM $PID; wait $PID 2>/dev/null
grep -o 'INF \(loaded\|switching\|also offered\).*' "$DIR/server.log" | sed "s#$DIR#DIR#g" | cut -c1-150
rm -rf "$DIR"
