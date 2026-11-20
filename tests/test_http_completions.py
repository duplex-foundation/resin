"""/v1/completions tokenizes raw prompts, and ignore_eos keeps generating past EOS."""

from __future__ import annotations

import json
import urllib.error

import pytest
from output_helpers import forward_output
from test_http_lifecycle import Runner, post, request_body, running_server
from test_http_stops import events, send
from test_http_token_counts import CountingRecipe
from test_recipe import _byte_tokenizer

from resin import BatchPhase
from resin.recipe import get_native_recipe

EOS = "<｜end▁of▁sentence｜>"
MARKERS = [
    "<｜begin▁of▁sentence｜>",
    EOS,
    "<｜User｜>",
    "<｜System｜>",
    "<｜Assistant｜>",
    "<think>",
    "</think>",
    "｜DSML｜",
]


def native_recipe():
    tokenizer = _byte_tokenizer()
    tokenizer.add_special_tokens(MARKERS)
    return get_native_recipe("deepseek_v41", tokenizer), tokenizer


class ScriptedRunner:
    """Emit one scripted token per step and remember each prefill prompt."""

    def __init__(self, script):
        self.script = script
        self.steps = {}
        self.prompts = []

    def forward(self, plan):
        tokens = []
        for row, sequence_id in enumerate(plan.sequence_ids):
            if plan.phase == BatchPhase.PREFILL:
                start, end = plan.token_indptr[row : row + 2]
                self.prompts.append(plan.token_ids[start:end].tolist())
            step = self.steps.get(int(sequence_id), 0)
            self.steps[int(sequence_id)] = step + 1
            tokens.append(self.script[min(step, len(self.script) - 1)])
        return forward_output(tokens)

    def release(self, sequence_ids):
        pass


def serve_scripted(text):
    recipe, tokenizer = native_recipe()
    eos = tokenizer.token_to_id(EOS)
    script = [tokenizer.encode(piece).ids[0] for piece in text if piece != "$"]
    script.insert(text.index("$"), eos)
    runner = ScriptedRunner(script)
    return running_server(runner, recipe), runner, tokenizer


def complete(base, body):
    with send(base, "/v1/completions", {"model": "test", **body}) as response:
        return json.load(response)


def test_completion_tokenizes_the_prompt_without_a_chat_template():
    server, runner, tokenizer = serve_scripted("ab$cd")
    with server as (_, _, _, errors, base, _):
        body = complete(base, {"prompt": "hello there", "max_tokens": 8})
        assert runner.prompts == [tokenizer.encode("hello there").ids]
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "ab"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == len(runner.prompts[0])
        assert errors == []


def test_completion_accepts_token_id_prompts():
    server, runner, tokenizer = serve_scripted("ab$cd")
    ids = tokenizer.encode("hey").ids
    with server as (_, _, _, errors, base, _):
        for prompt in (ids, [ids]):
            body = complete(base, {"prompt": prompt, "max_tokens": 8})
            assert body["usage"]["prompt_tokens"] == len(ids)
        assert runner.prompts == [ids, ids]
        assert errors == []


@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_runs_to_the_length_limit_and_hides_eos_text(stream):
    server, _, _ = serve_scripted("ab$cd")
    with server as (_, _, _, errors, base, _):
        body = {"prompt": "hi", "max_tokens": 5, "ignore_eos": True, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
            with send(base, "/v1/completions", {"model": "test", **body}) as response:
                chunks = events(response)
            text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
            reason = next(
                c["choices"][0]["finish_reason"]
                for c in chunks
                if c["choices"] and c["choices"][0]["finish_reason"]
            )
            tokens = chunks[-1]["usage"]["completion_tokens"]
        else:
            result = complete(base, body)
            text = result["choices"][0]["text"]
            reason = result["choices"][0]["finish_reason"]
            tokens = result["usage"]["completion_tokens"]
        assert (text, reason, tokens) == ("abcd", "length", 5)
        assert errors == []


def test_eos_stops_generation_by_default():
    server, _, _ = serve_scripted("ab$cd")
    with server as (_, _, _, errors, base, _):
        body = complete(base, {"prompt": "hi", "max_tokens": 5, "ignore_eos": False})
        assert body["choices"][0]["text"] == "ab"
        # The EOS token itself counts as a completion token.
        assert body["usage"]["completion_tokens"] == 3
        assert errors == []


def test_chat_rejects_ignore_eos_because_response_grammars_end_at_eos():
    server, _, _ = serve_scripted("ab$cd")
    with server as (_, _, _, errors, base, _):
        chat = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(urllib.error.HTTPError) as failure:
            send(base, "/v1/chat/completions", chat | {"ignore_eos": True})
        assert failure.value.code == 400
        with send(base, "/v1/chat/completions", chat | {"ignore_eos": False}) as r:
            assert r.status == 200
        assert errors == []


def test_unknown_token_ids_are_rejected_before_the_engine_sees_them():
    server, runner, _ = serve_scripted("ab$cd")
    with server as (_, engine, _, errors, base, _):
        with pytest.raises(urllib.error.HTTPError) as failure:
            complete(base, {"prompt": [10**6], "max_tokens": 2})
        assert "outside the tokenizer vocabulary" in failure.value.read().decode()
        assert runner.prompts == []
        assert not engine.has_work()
        assert errors == []


def test_raw_prompts_require_a_native_recipe():
    with running_server(Runner(), CountingRecipe()) as (_, _, _, errors, base, _):
        with pytest.raises(urllib.error.HTTPError) as failure:
            complete(base, {"prompt": "hi", "max_tokens": 2})
        assert "raw prompts require a native recipe" in failure.value.read().decode()
        with post(base, request_body()) as response:
            assert response.status == 200
        assert errors == []
