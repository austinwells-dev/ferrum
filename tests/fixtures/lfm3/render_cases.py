"""Reference renderings of the LFM3-1.7B chat template, from Jinja2 as Hugging Face's
`apply_chat_template` uses it (trim_blocks, lstrip_blocks, and json.dumps for
`tojson`). `{% generation %}` / `{% endgeneration %}` are stripped first, the
same way `ChatTemplate` does: they carry no text. Writes render_cases.json
next to this file; the Rust test in src/hybrid/chat.rs renders the same cases
and compares.

usage: python3 -I render_cases.py   (needs jinja2)

bos_token and eos_token are the GGUF vocabulary strings of
tokenizer.ggml.bos_token_id (124894) and eos_token_id (124900).
"""
import json
import os

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
BOS = "<|startoftext|>"
EOS = "<|im_end|>"


def strip_generation_tags(source: str) -> str:
    out = source
    for tag in ("generation", "endgeneration"):
        rest = out
        next_ = []
        while True:
            start = rest.find("{%")
            if start < 0:
                next_.append(rest)
                break
            next_.append(rest[:start])
            end = rest.find("%}", start)
            if end < 0:
                next_.append(rest[start:])
                break
            body = rest[start + 2 : end].strip().strip("-").strip()
            if body == tag:
                rest = rest[end + 2 :]
            else:
                next_.append(rest[start : end + 2])
                rest = rest[end + 2 :]
        out = "".join(next_)
    return out


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def render(template, case):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        jinja2.exceptions.TemplateError(message)
    )
    return env.from_string(strip_generation_tags(template)).render(
        messages=case["messages"],
        tools=case.get("tools"),
        add_generation_prompt=case["add_generation_prompt"],
        bos_token=BOS,
        eos_token=EOS,
        **case["extra"],
    )


weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

cases = [
    {
        "name": "plain_chat_thinking_on",
        "messages": [{"role": "user", "content": "Hi there"}],
        "tools": None,
        "add_generation_prompt": True,
        "extra": {},
    },
    {
        "name": "thinking_off",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": None,
        "add_generation_prompt": True,
        "extra": {"enable_thinking": False},
    },
    {
        "name": "system_and_user",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi there"},
        ],
        "tools": None,
        "add_generation_prompt": True,
        "extra": {},
    },
    {
        "name": "tool_call_and_result",
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "18C and sunny"},
        ],
        "tools": [weather_tool],
        "add_generation_prompt": True,
        "extra": {"enable_thinking": False},
    },
    {
        "name": "history_without_generation_prompt",
        "messages": [
            {"role": "user", "content": "Count"},
            {"role": "assistant", "content": "1, 2, 3"},
        ],
        "tools": None,
        "add_generation_prompt": False,
        "extra": {},
    },
]

template = open(os.path.join(HERE, "chat_template.jinja"), encoding="utf-8").read()
for case in cases:
    case["expect"] = render(template, case)

with open(os.path.join(HERE, "render_cases.json"), "w", encoding="utf-8") as out:
    json.dump(cases, out, ensure_ascii=False, indent=1)
    out.write("\n")
print(f"wrote {len(cases)} cases")
