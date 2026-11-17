"""Opt-in step timing and bounded device traces for a running TPU server.

Both hooks are off unless their environment variable is set, and each costs
one attribute check per step when off.

``RESIN_TPU_STEP_LOG=path`` appends one JSON line per step: phase, bucket,
real rows and tokens, KV tokens read, and host timings in milliseconds
(``plan`` builds the padded inputs, ``put`` transfers them, ``dispatch``
returns from the asynchronous launch, ``wait`` blocks reading the sampled
tokens). ``t`` fields are ``time.perf_counter()`` seconds, so gaps between
steps are host or scheduler time.

``RESIN_TPU_TRACE_DIR=dir`` arms a JAX device trace. Write
``dir/trigger`` containing ``steps=N`` (default 64) and optionally
``label=name``; the next step starts ``jax.profiler.start_trace`` into
``dir/<label>`` and stops it after N steps, after the device drains. The
trigger file is removed when the trace starts and ``dir/<label>.done``
appears when it stops.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import jax

_FLUSH_EVERY = 256
_FLUSH_SECONDS = 2.0


class StepLog:
    def __init__(self, path: str) -> None:
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._pending: list[str] = []
        self._flushed = time.monotonic()

    def write(self, record: dict[str, Any]) -> None:
        self._pending.append(json.dumps(record, separators=(",", ":")))
        if (
            len(self._pending) >= _FLUSH_EVERY
            or time.monotonic() - self._flushed > _FLUSH_SECONDS
        ):
            self.flush()

    def flush(self) -> None:
        self._flushed = time.monotonic()
        if self._pending:
            self._file.write("\n".join(self._pending) + "\n")
            self._file.flush()
            self._pending.clear()


class TraceWindow:
    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._trigger = self._dir / "trigger"
        self._remaining = 0
        self._label = ""

    @property
    def active(self) -> bool:
        return self._remaining > 0

    def before_step(self, drain: Any) -> None:
        """Start or stop the trace; ``drain`` is a pytree to wait on before stopping."""

        if self._remaining:
            self._remaining -= 1
            if not self._remaining:
                jax.block_until_ready(drain)
                jax.profiler.stop_trace()
                (self._dir / f"{self._label}.done").write_text(f"{time.time()}\n")
            return
        if not self._trigger.exists():
            return
        options = dict(
            line.split("=", 1)
            for line in self._trigger.read_text().split()
            if "=" in line
        )
        self._trigger.unlink()
        self._label = options.get("label") or time.strftime("trace-%H%M%S")
        self._remaining = max(1, int(options.get("steps", 64)))
        jax.profiler.start_trace(str(self._dir / self._label))


def step_log_from_env() -> StepLog | None:
    path = os.environ.get("RESIN_TPU_STEP_LOG")
    return StepLog(path) if path else None


def trace_window_from_env() -> TraceWindow | None:
    directory = os.environ.get("RESIN_TPU_TRACE_DIR")
    return TraceWindow(directory) if directory else None


__all__ = ["StepLog", "TraceWindow", "step_log_from_env", "trace_window_from_env"]
