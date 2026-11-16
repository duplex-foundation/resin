"""Device mesh with only the axes a model uses: ``tp`` until a plug-in needs more."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

TP_AXIS = "tp"


def make_mesh(
    tp: int | None = None, devices: Sequence[jax.Device] | None = None
) -> Mesh:
    """Build a one-axis tensor-parallel mesh over the first ``tp`` devices.

    On one Ironwood chip the two dies are two devices, so ``tp=2`` covers the chip.
    """

    devices = list(jax.devices() if devices is None else devices)
    tp = len(devices) if tp is None else tp
    if not 0 < tp <= len(devices):
        raise ValueError(f"tp={tp} needs between 1 and {len(devices)} devices")
    return Mesh(np.asarray(devices[:tp]), (TP_AXIS,))


def tp_size(mesh: Mesh) -> int:
    return int(mesh.shape.get(TP_AXIS, 1))


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


__all__ = ["TP_AXIS", "make_mesh", "replicated", "tp_size"]
