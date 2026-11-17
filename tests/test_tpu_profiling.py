import json

import jax.numpy as jnp

from resin.backend.tpu.profiling import (
    StepLog,
    TraceWindow,
    step_log_from_env,
    trace_window_from_env,
)


def test_hooks_are_off_without_their_environment(monkeypatch):
    monkeypatch.delenv("RESIN_TPU_STEP_LOG", raising=False)
    monkeypatch.delenv("RESIN_TPU_TRACE_DIR", raising=False)
    assert step_log_from_env() is None
    assert trace_window_from_env() is None


def test_step_log_appends_json_lines(tmp_path):
    path = tmp_path / "steps.jsonl"
    log = StepLog(str(path))
    log.write({"step": 0, "phase": "decode"})
    log.write({"step": 1, "phase": "prefill"})
    log.flush()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["step"] for record in records] == [0, 1]


def test_trace_window_runs_for_the_requested_steps(tmp_path):
    window = TraceWindow(str(tmp_path))
    drain = jnp.zeros(4)
    window.before_step(drain)
    assert not window.active

    (tmp_path / "trigger").write_text("steps=2\nlabel=probe\n")
    window.before_step(drain)
    assert window.active
    assert not (tmp_path / "trigger").exists()
    window.before_step(drain)
    assert window.active
    window.before_step(drain)
    assert not window.active
    assert (tmp_path / "probe.done").exists()
    assert list((tmp_path / "probe").rglob("*.xplane.pb"))
