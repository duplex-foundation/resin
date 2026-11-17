"""The TPU serve wrapper takes the chip queue and the server uses the registry."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from scripts import tpu_serve  # noqa: E402


def test_wrapper_execs_the_chip_queue(tmp_path):
    recorded = tmp_path / "argv"
    queue = tmp_path / "tpu-queue.sh"
    queue.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$@" > "$RESIN_TPU_ARGV"\n',
        encoding="utf-8",
    )
    queue.chmod(0o755)
    script = Path(tpu_serve.__file__).with_name("tpu_serve.sh")
    done = subprocess.run(
        [str(script), "--model", "qwen3-32b", "--workload", "8k/1k"],
        check=False,
        env={
            **os.environ,
            "RESIN_TPU_QUEUE": str(queue),
            "RESIN_TPU_ARGV": str(recorded),
        },
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    lines = recorded.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["env", "RESIN_TPU_QUEUED=1", "/opt/pearl/tpu/bin/python"]
    assert lines[3].endswith("tpu_serve.py")
    assert lines[4:] == ["--model", "qwen3-32b", "--workload", "8k/1k"]


def test_wrapper_requires_the_queue_path():
    script = Path(tpu_serve.__file__).with_name("tpu_serve.sh")
    done = subprocess.run(
        [str(script), "--help"],
        check=False,
        env={
            key: value for key, value in os.environ.items() if key != "RESIN_TPU_QUEUE"
        },
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1
    assert "RESIN_TPU_QUEUE" in done.stderr


COMPILE_CACHE_ENV = (
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
)


@pytest.fixture(autouse=True)
def _restore_compile_cache_env(monkeypatch):
    for name in COMPILE_CACHE_ENV:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_serve_exports_the_jax_compile_cache(monkeypatch):
    tpu_serve.export_compile_cache_env()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == tpu_serve.COMPILE_CACHE_DIR
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] == "0"
    assert os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] == "-1"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/elsewhere")
    tpu_serve.export_compile_cache_env()
    assert os.environ["JAX_COMPILATION_CACHE_DIR"] == "/elsewhere"


def test_serve_builds_the_workload_and_keeps_prefix_cache(
    monkeypatch, tmp_path, capsys
):
    seen = {}

    def create_runner(*args, **kwargs):
        seen["runner"] = (args, kwargs)
        return type("Runner", (), {"benchmark_metadata": {"warmup_seconds": 1.5}})()

    def create_recipe(*args, **kwargs):
        seen["recipe"] = args
        return object()

    class Engine:
        closed = False

        def __init__(self, runner, **kwargs):
            seen["engine"] = kwargs

        def close(self):
            self.closed = True

    class Server:
        def __init__(self, engine, **kwargs):
            seen["server"] = kwargs

        def serve(self, *, host, port):
            seen["serve"] = (host, port)

    monkeypatch.setattr(tpu_serve, "create_runner", create_runner)
    monkeypatch.setattr(tpu_serve, "create_recipe", create_recipe)
    resin = cast(Any, ModuleType("resin"))
    resin.Engine = Engine
    resin.SchedulerConfig = lambda **kwargs: kwargs
    server = cast(Any, ModuleType("resin.server"))
    server.HttpServer = Server
    monkeypatch.setitem(sys.modules, "resin", resin)
    monkeypatch.setitem(sys.modules, "resin.server", server)
    checkpoint = tmp_path / "weights"
    tpu_serve.main(
        [
            "--model",
            "qwen3-32b",
            "--checkpoint",
            str(checkpoint),
            "--workload",
            "8k/1k",
            "--no-prefix-cache",
            "--port",
            "9",
        ]
    )
    kwargs = seen["runner"][1]
    assert kwargs["workload"] == "8k/1k"
    assert kwargs["context_length"] == 9216
    assert kwargs["max_batch_size"] == 320
    assert kwargs["prefill_chunk_size"] == 4096
    assert kwargs["draft_tokens"] == 0
    assert seen["engine"]["prefix_cache"] is False
    assert seen["engine"]["context_length"] == 9216
    assert seen["engine"]["max_running_requests"] == 320
    assert seen["serve"] == ("127.0.0.1", 9)
    assert "warmup 1.5s" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        tpu_serve.main(["--model", "qwen3-32b", "--checkpoint", str(checkpoint)])
