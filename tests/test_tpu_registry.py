"""The TPU preset builds a workload-shaped runner and renders with the checkpoint template."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from deploy.models import (  # noqa: E402
    COMPILE_CACHE_ENV,
    TPU_COMPILE_CACHE_DIR,
    TPU_WORKLOADS,
    create_recipe,
    create_runner,
    export_tpu_compile_cache_env,
    get_model,
    require_tpu_chip,
)

QWEN3_TOKENIZER_DIR = Path(
    os.environ.get(
        "RESIN_QWEN3_TOKENIZER_DIR", "/home/raskin/vllm-repro/models/Qwen3-32B"
    )
).expanduser()

_QWEN3_CHAT_TURNS = {
    "system-user": (
        ("system", "You are concise."),
        ("user", "Hello"),
    ),
    "multi-turn": (
        ("user", "What is 2+2?"),
        ("assistant", "4"),
        ("user", "And times 3?"),
    ),
}


def test_workloads_are_the_published_server_shapes():
    short = TPU_WORKLOADS["1k/1k"]
    long = TPU_WORKLOADS["1k/8k"]
    prefill = TPU_WORKLOADS["8k/1k"]
    assert (short.prompt, short.output, short.max_model_len, short.decode_buckets) == (
        1024,
        1024,
        2048,
        (320,),
    )
    assert (long.prompt, long.output, long.decode_buckets) == (
        1024,
        8192,
        (96, 160, 192, 224, 288, 320),
    )
    assert (prefill.prompt, prefill.output, prefill.max_model_len) == (8192, 1024, 9216)
    assert long.max_model_len == prefill.max_model_len == 9216
    spec = get_model("qwen3-32b")
    assert spec.backend == "tpu"
    assert spec.repo_id == "Qwen/Qwen3-32B"
    assert spec.revision == "9216db5781bf21249d130ec9da846c4624c16137"
    assert spec.context_length == 9216
    assert spec.max_batch_size == 320
    assert spec.prefill_chunk_size == 4096
    assert spec.gpu_memory_utilization == 0.98
    assert spec.text_only is True


def test_unpinned_tpu_host_refuses_before_jax_starts(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("TPU_VISIBLE_CHIPS", raising=False)
    monkeypatch.delenv("TPU_CHIPS_PER_HOST_BOUNDS", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "dev-tpu-tpu7x8b-0")
    with pytest.raises(RuntimeError, match="chip 0"):
        require_tpu_chip()


def test_pinned_chip_still_needs_the_queue(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "0")
    monkeypatch.setenv("TPU_CHIPS_PER_HOST_BOUNDS", "1,1,1")
    monkeypatch.delenv("TPU_HOST_BOUNDS", raising=False)
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.delenv("RESIN_TPU_QUEUED", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "dev-tpu-tpu7x8b-0")
    with pytest.raises(RuntimeError, match="chip 0"):
        require_tpu_chip()
    monkeypatch.setenv("TPU_HOST_BOUNDS", "1,1,1")
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    require_tpu_chip()
    with pytest.raises(RuntimeError, match="chip queue"):
        require_tpu_chip(queued=True)


def test_cpu_platform_skips_the_chip_pin(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr("socket.gethostname", lambda: "dev-tpu-tpu7x8b-0")
    require_tpu_chip(queued=True)


def test_tpu_factory_uses_the_workload_shape(monkeypatch, tmp_path):
    calls = {}
    mesh = object()
    model = object()
    params = object()

    def load_model(checkpoint, got_mesh):
        calls["load"] = (checkpoint, got_mesh)
        return model, params

    package = ModuleType("resin_models_tpu")
    plug = ModuleType("resin_models_tpu.qwen3")
    cast(Any, plug).load_model = load_model
    monkeypatch.setitem(sys.modules, "resin_models_tpu", package)
    monkeypatch.setitem(sys.modules, "resin_models_tpu.qwen3", plug)

    class Runner:
        def __init__(self, got_model, got_params, **kwargs):
            calls["runner"] = (got_model, got_params, kwargs)

    monkeypatch.setattr("resin.backend.tpu.mesh.make_mesh", lambda: mesh)
    monkeypatch.setattr("resin.backend.tpu.runner.TpuRunner", Runner)
    spec = get_model("qwen3-32b")
    checkpoint = tmp_path / "qwen3"
    create_runner(
        spec,
        checkpoint,
        device="tpu",
        context_length=9216,
        max_batch_size=320,
        prefill_chunk_size=4096,
        draft_tokens=0,
        workload="1k/8k",
    )
    assert calls["load"] == (checkpoint, mesh)
    got_model, got_params, kwargs = calls["runner"]
    assert got_model is model and got_params is params
    assert kwargs["mesh"] is mesh
    assert kwargs["max_seq_len"] == 9216
    assert kwargs["max_num_seqs"] == 320
    assert kwargs["max_batch_tokens"] == 4096
    assert kwargs["memory_utilization"] == 0.98
    assert kwargs["buckets"].decode_seqs == (96, 160, 192, 224, 288, 320)
    assert kwargs["buckets"].prefill_tokens[-1] == 4096
    create_runner(
        spec,
        checkpoint,
        device="tpu",
        context_length=2048,
        max_batch_size=320,
        prefill_chunk_size=4096,
        draft_tokens=0,
        workload="1k/1k",
    )
    assert calls["runner"][2]["max_seq_len"] == 2048
    assert calls["runner"][2]["buckets"].decode_seqs == (320,)
    with pytest.raises(ValueError, match="2048"):
        create_runner(
            spec,
            checkpoint,
            device="tpu",
            context_length=9216,
            max_batch_size=320,
            prefill_chunk_size=4096,
            draft_tokens=0,
            workload="1k/1k",
        )
    with pytest.raises(ValueError, match="draft"):
        create_runner(
            spec,
            checkpoint,
            device="tpu",
            context_length=9216,
            max_batch_size=320,
            prefill_chunk_size=4096,
            draft_tokens=1,
            workload="1k/8k",
        )
    with pytest.raises(ValueError, match="one process"):
        create_runner(
            spec,
            checkpoint,
            device="tpu",
            groups=object(),
            context_length=9216,
            max_batch_size=320,
            prefill_chunk_size=4096,
            draft_tokens=0,
            workload="1k/8k",
        )


def test_qwen3_recipe_uses_the_checkpoint_template(monkeypatch, tmp_path):
    from resin.recipe import Message, RenderOptions, Role, Text, Thinking

    seen = {}

    class Tokenizer:
        eos_token_id = 151645

        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            seen["kwargs"] = kwargs
            return [9, 8]

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "AB"[: len(list(ids))]

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: Tokenizer(),
    )
    recipe = create_recipe(get_model("qwen3-32b"), tmp_path)
    assert recipe.stop_token_ids == [151645]
    prompt = recipe.render([Message(Role.USER, [Text("Hello")])])
    assert list(prompt.token_ids) == [9, 8]
    assert seen["messages"] == [{"role": "user", "content": "Hello"}]
    assert seen["kwargs"]["enable_thinking"] is False
    assert seen["kwargs"]["add_generation_prompt"] is True
    recipe.render(
        [Message(Role.USER, [Text("Hello")])],
        options=RenderOptions(reasoning_effort="high"),
    )
    assert seen["kwargs"]["enable_thinking"] is True
    parser = recipe.create_parser(prompt)
    assert parser.feed([1])[0].delta == "A"
    assert parser.feed([1])[0].delta == "B"
    assert parser.finish("stop") == []
    with pytest.raises(ValueError, match="text only"):
        recipe.render([Message(Role.USER, [Thinking("secret")])])
    with pytest.raises(ValueError, match="tools"):
        recipe.render([Message(Role.USER, [Text("Hello")])], tools=(object(),))


@pytest.mark.skipif(
    not (QWEN3_TOKENIZER_DIR / "tokenizer.json").is_file(),
    reason="Qwen3-32B tokenizer is not at RESIN_QWEN3_TOKENIZER_DIR",
)
@pytest.mark.parametrize("name", _QWEN3_CHAT_TURNS)
@pytest.mark.parametrize("thinking", [True, False])
def test_qwen3_chat_recipe_matches_the_checkpoint_template(name, thinking):
    from transformers import AutoTokenizer

    from resin.recipe import Message, RenderOptions, Role, Text

    roles = {
        "system": Role.SYSTEM,
        "user": Role.USER,
        "assistant": Role.ASSISTANT,
    }
    turns = _QWEN3_CHAT_TURNS[name]
    messages = [Message(roles[role], [Text(text)]) for role, text in turns]
    rows = [{"role": role, "content": text} for role, text in turns]
    recipe = create_recipe(get_model("qwen3-32b"), QWEN3_TOKENIZER_DIR)
    prompt = recipe.render(messages, options=RenderOptions(enable_thinking=thinking))
    reference = AutoTokenizer.from_pretrained(
        QWEN3_TOKENIZER_DIR, local_files_only=True
    )
    expected = reference.apply_chat_template(
        rows,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    assert list(prompt.token_ids) == list(expected["input_ids"])


def test_compile_cache_env_is_exported_and_overridable(monkeypatch):
    for name in COMPILE_CACHE_ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    export_tpu_compile_cache_env()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == TPU_COMPILE_CACHE_DIR
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "-1"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/elsewhere")
    export_tpu_compile_cache_env()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "/elsewhere"


def test_bench_exports_the_compile_cache_before_importing_jax(monkeypatch):
    from scripts import bench

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    monkeypatch.setattr(bench, "require_tpu_chip", lambda **kwargs: None)
    monkeypatch.setattr(bench, "export_tpu_compile_cache_env", stop)
    with pytest.raises(Stop):
        bench._rank(0, 0, None, "qwen3-32b", None, [], None, {})
