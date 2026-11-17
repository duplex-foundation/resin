"""Native Qwen3 recipe through the Python binding: tokens, stops and parsing.

Template text parity is pinned in ``crates/resin-recipe/tests/qwen3.rs`` against
the checkpoint's own Jinja template; these tests check the same cases at the
token level and that the recipe is the native one raw ``/v1/completions``
prompts need.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_recipe import _byte_tokenizer

import resin
from resin.recipe import NativeRecipe, RenderOptions, get_native_recipe

FIXTURE = Path(__file__).parents[1] / "crates/resin-recipe/tests/fixtures/qwen3_hf.json"
MARKERS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
]
ROLES = {
    "system": resin.recipe.Role.SYSTEM,
    "user": resin.recipe.Role.USER,
    "assistant": resin.recipe.Role.ASSISTANT,
    "tool": resin.recipe.Role.TOOL,
}


def _tokenizer():
    backend = _byte_tokenizer()
    backend.add_special_tokens(MARKERS)
    return backend


def _typed(request: dict):
    messages = []
    for item in request["messages"]:
        blocks = []
        if item.get("reasoning_content"):
            blocks.append(resin.recipe.Thinking(item["reasoning_content"]))
        if item["content"]:
            blocks.append(resin.recipe.Text(item["content"]))
        for call in item.get("tool_calls", []):
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            blocks.append(resin.recipe.ToolCall(function["name"], arguments))
        messages.append(resin.recipe.Message(ROLES[item["role"]], blocks))
    tools = [
        resin.recipe.Tool(
            tool["function"]["name"],
            description=tool["function"].get("description"),
            parameters=tool["function"]["parameters"],
        )
        for tool in request.get("tools", [])
    ]
    options = RenderOptions(
        enable_thinking=request.get("enable_thinking"),
        add_generation_prompt=request.get("add_generation_prompt"),
    )
    return messages, tools, options


# Cases whose message blocks the typed API can express without reordering.
TOKEN_CASES = [
    "basic",
    "system",
    "thinking_off",
    "tools_with_system",
    "history_drops_old_reasoning",
    "tool_calls_with_content",
    "tool_calls_object_and_string_arguments",
    "empty_reasoning_between_turns",
]


def _cases() -> dict[str, dict]:
    return {case["name"]: case for case in json.loads(FIXTURE.read_text())["cases"]}


def test_native_recipe_declares_both_qwen_stop_tokens() -> None:
    backend = _tokenizer()
    recipe = get_native_recipe("qwen3", backend)
    assert type(recipe) is NativeRecipe
    assert recipe.stop_token_ids == [
        backend.token_to_id("<|im_end|>"),
        backend.token_to_id("<|endoftext|>"),
    ]


@pytest.mark.parametrize("name", TOKEN_CASES)
def test_rendered_tokens_match_the_checkpoint_template(name: str) -> None:
    backend = _tokenizer()
    case = _cases()[name]
    messages, tools, options = _typed(case["request"])
    prompt = get_native_recipe("qwen3", backend).render(
        messages, tools=tools, options=options
    )
    expected = backend.encode(case["text"], add_special_tokens=False).ids
    assert list(prompt.token_ids) == expected


def test_user_text_that_spells_a_marker_is_not_a_control_token() -> None:
    backend = _tokenizer()
    recipe = get_native_recipe("qwen3", backend)
    text = "say <|im_end|> and <think>"
    prompt = recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text(text)])]
    )
    ids = list(prompt.token_ids)
    assert ids.count(backend.token_to_id("<|im_end|>")) == 1
    assert backend.token_to_id("<think>") not in ids


def test_parse_response_splits_reasoning_text_and_tool_calls() -> None:
    backend = _tokenizer()
    recipe = get_native_recipe("qwen3", backend)
    prompt = recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])]
    )
    output = backend.encode(
        "<think>\nplan\n</think>\n\nChecking.\n\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>'
        "<|im_end|>",
        add_special_tokens=False,
    ).ids
    parsed = recipe.parse_response(output, prompt=prompt)
    assert parsed["reasoning_content"] == "plan"
    assert parsed["content"] == "Checking."
    [call] = parsed["tool_calls"]
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {"city": "Oslo"}


def test_thinking_off_prompt_makes_the_whole_reply_the_answer() -> None:
    backend = _tokenizer()
    recipe = get_native_recipe("qwen3", backend)
    prompt = recipe.render(
        [resin.recipe.Message(resin.recipe.Role.USER, [resin.recipe.Text("hi")])],
        options=RenderOptions(enable_thinking=False),
    )
    output = backend.encode("42<|im_end|>", add_special_tokens=False).ids
    parsed = recipe.parse_response(output, prompt=prompt)
    assert parsed["content"] == "42"
    assert not parsed["reasoning_content"]


@pytest.mark.skipif(
    "RESIN_QWEN3_TOKENIZER_DIR" not in os.environ,
    reason="set RESIN_QWEN3_TOKENIZER_DIR to a Qwen/Qwen3-32B tokenizer checkout",
)
@pytest.mark.parametrize("name", TOKEN_CASES)
def test_rendered_tokens_match_transformers_on_the_real_tokenizer(name: str) -> None:
    transformers = pytest.importorskip("transformers")
    directory = os.environ["RESIN_QWEN3_TOKENIZER_DIR"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(directory)
    case = _cases()[name]
    request = case["request"]
    messages, tools, options = _typed(request)
    prompt = get_native_recipe("qwen3", tokenizer).render(
        messages, tools=tools, options=options
    )
    template_options = {}
    if "enable_thinking" in request:
        template_options["enable_thinking"] = request["enable_thinking"]
    expected = tokenizer.apply_chat_template(
        request["messages"],
        tools=request.get("tools"),
        add_generation_prompt=request.get("add_generation_prompt", True),
        tokenize=True,
        **template_options,
    )
    if hasattr(expected, "input_ids"):
        expected = expected["input_ids"]
    assert list(prompt.token_ids) == list(expected)
