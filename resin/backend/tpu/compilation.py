"""A process-wide count of XLA compilations, for asserting none after warmup.

JAX records the compile event around every executable it builds or loads from
the persistent cache, including recompiles for a new input sharding that need
no retrace. Executables reused from memory do not count. Eager operations on
new shapes do. The listener sees every thread, so two runners compiling at
once in one process would count each other's work.
"""

from __future__ import annotations

import threading
from collections import Counter

from jax import monitoring

COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"


class CompilationCounter:
    """Cumulative compilations observed since the counter was created."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names: list[str] = []
        monitoring.register_event_duration_secs_listener(self._observe)

    def _observe(self, event: str, duration_secs: float, **kwargs: str | int) -> None:
        del duration_secs
        if event != COMPILE_EVENT:
            return
        name = str(kwargs.get("fun_name", "?"))
        with self._lock:
            self._names.append(name)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._names)

    def by_function(self, since: int = 0) -> dict[str, int]:
        """Compilations per module name after the first ``since``."""

        with self._lock:
            return dict(Counter(self._names[since:]))

    def close(self) -> None:
        monitoring.unregister_event_duration_listener(self._observe)


__all__ = ["COMPILE_EVENT", "CompilationCounter"]
