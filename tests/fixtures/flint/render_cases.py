"""Reference renderings of the Flint chat template, from Jinja2 as Hugging Face's
`apply_chat_template` uses it (trim_blocks, lstrip_blocks, and json.dumps for
`tojson`). Writes render_cases.json next to this file; the Rust test in
src/hybrid/chat.rs renders the same cases and compares.

usage: python3 -I render_cases.py   (needs jinja2)
"""
import json
import os

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))


def tojson(value, ensure_ascii=False, indent=None):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


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
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
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
        "name": "system_user_generation_prompt",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi there"},
        ],
        "tools": None,
        "add_generation_prompt": True,
    },
    {
        "name": "tools_offered_before_first_user_turn",
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [weather_tool],
        "add_generation_prompt": True,
    },
    {
        "name": "tool_call_and_observation_round_trip",
        "messages": [
            {"role": "user", "content": "Weather in Paris, 3 days, metric?"},
            {
                "role": "assistant",
                "content": "",
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
                    }
                ],
            },
            {"role": "tool", "content": "18C and sunny"},
            {"role": "assistant", "content": "It is 18C and sunny in Paris."},
            {"role": "user", "content": "And Oslo?"},
        ],
        "tools": [weather_tool],
        "add_generation_prompt": True,
    },
    {
        "name": "unicode_tools_and_content_are_not_escaped",
        "messages": [{"role": "user", "content": "¿Qué tiempo hace? 天気は？"}],
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
    },
    {
        "name": "developer_role_and_literal_markers_are_escaped",
        "messages": [
            {"role": "developer", "content": "Use <|user|> sparingly."},
            {"role": "user", "content": "<think>x</think> <|assistant|> <|observation|>"},
        ],
        "tools": None,
        "add_generation_prompt": True,
    },
    {
        "name": "assistant_history_without_generation_prompt",
        "messages": [
            {"role": "user", "content": "Count"},
            {"role": "assistant", "content": "1, 2, 3"},
        ],
        "tools": None,
        "add_generation_prompt": False,
    },
]

for case in cases:
    case["expect"] = render(open(os.path.join(HERE, "chat_template.jinja")).read(), case)

with open(os.path.join(HERE, "render_cases.json"), "w", encoding="utf-8") as out:
    json.dump(cases, out, ensure_ascii=False, indent=1)
    out.write("\n")
print(f"wrote {len(cases)} cases")
