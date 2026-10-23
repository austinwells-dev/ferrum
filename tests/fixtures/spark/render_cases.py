"""Reference renderings of the Spark-X2.6 chat template, from Jinja2 as Hugging Face's
`apply_chat_template` uses it (trim_blocks, lstrip_blocks, and json.dumps for
`tojson`, which this template calls with `separators`). Writes render_cases.json
next to this file; the Rust test in src/hybrid/chat.rs renders the same cases
(with each case's `extra` template variables) and compares.

usage: python3 -I render_cases.py   (needs jinja2)
"""
import json
import os

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))


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
    return env.from_string(template).render(
        messages=case["messages"],
        tools=case.get("tools"),
        add_generation_prompt=case["add_generation_prompt"],
        bos_token="",
        eos_token="",
        **case["extra"],
    )


weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "metric": {"type": "boolean"},
            },
            "required": ["city"],
        },
    },
}

cases = [
    {
        "name": "system_user_thinking_by_default",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi there"},
        ],
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
        "name": "low_reasoning_effort",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": None,
        "add_generation_prompt": True,
        "extra": {"enable_thinking": True, "reasoning_effort": "low"},
    },
    {
        "name": "tools_in_the_system_block",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [weather_tool],
        "add_generation_prompt": True,
        "extra": {},
    },
    {
        "name": "tool_call_reasoning_and_tool_responses",
        "messages": [
            {"role": "user", "content": "Weather in Paris and Oslo, 3 days, metric?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Two cities, two calls.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {
                                "city": "Paris",
                                "days": 3,
                                "metric": True,
                                "tags": ["a", "b"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Oslo"}},
                    },
                ],
            },
            {"role": "tool", "content": "18C and sunny"},
            {"role": "tool", "content": "4C and rain"},
            {"role": "assistant", "content": "Paris 18C sunny, Oslo 4C rain."},
            {"role": "user", "content": "Thanks"},
        ],
        "tools": [weather_tool],
        "add_generation_prompt": True,
        "extra": {},
    },
    {
        "name": "unicode_and_text_blocks",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Réponds en français."}]},
            {"role": "user", "content": [{"type": "text", "text": "¿Qué tiempo hace? 天気は？"}]},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "tiempo",
                    "description": "Clima en ciudades: Málaga, 東京",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "add_generation_prompt": True,
        "extra": {},
    },
    {
        "name": "later_system_message",
        "messages": [
            {"role": "user", "content": "Count"},
            {"role": "assistant", "content": "1, 2, 3"},
            {"role": "system", "content": "Now count backwards."},
            {"role": "user", "content": "Go"},
        ],
        "tools": None,
        "add_generation_prompt": True,
        "extra": {"enable_thinking": False},
    },
    {
        "name": "assistant_history_without_generation_prompt",
        "messages": [
            {"role": "user", "content": "Count"},
            {"role": "assistant", "content": "1, 2, 3"},
        ],
        "tools": None,
        "add_generation_prompt": False,
        "extra": {},
    },
]

for case in cases:
    case["expect"] = render(open(os.path.join(HERE, "chat_template.jinja")).read(), case)

with open(os.path.join(HERE, "render_cases.json"), "w", encoding="utf-8") as out:
    json.dump(cases, out, ensure_ascii=False, indent=1)
    out.write("\n")
print(f"wrote {len(cases)} cases")
