"""Engine-addressed KV pages in the ragged paged attention v3 layout.

Each cache group owns one array per layer, shaped
``[(parent_capacity + 1) * blocks_per_slab, page_tokens, rows, packing,
round_up(head_dim, 128)]``.
K and V heads interleave on the combined head axis (``2 * head + {0: K, 1: V}``),
packed ``packing = 32 // bits`` per 32-bit word. Row ``slab * blocks_per_slab +
slot`` is the Engine block ``(slab, slot)``; slab zero is the never-written null
page. Arrays are separate per layer because a Pallas call aliasing one slice
of a stacked array would copy that layer's whole cache every step.

The arrays shard on the combined head axis over ``tp``. Each die holds complete
K/V pairs for its heads, padded to the packing like a single-die v3 cache.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from resin import CacheGroupLayout, CacheLayout, CacheRetention
from resin.backend.capacity import CacheCapacityPlan, plan_cache_capacity
from resin.backend.resources import CachePoolBinding
from resin.backend.tpu.mesh import TP_AXIS, tp_size

if TYPE_CHECKING:
    from resin import ResourceBinding

KV_PARTITION = P(None, None, TP_AXIS, None, None)
LANES = 128

Caches = tuple[tuple[jax.Array, ...], ...]


MIN_HEADROOM_BYTES = 3584 * 1024**2
"""Free HBM across the mesh that a default budget leaves for activations and
compiled executables, which are allocated after the cache."""


def _align(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


@dataclass(frozen=True, slots=True)
class KvCacheSpec:
    """One attention cache group: its layers and per-token K/V geometry.

    ``sliding_window`` is the read window in tokens, including the current
    token; the Engine then keeps only the pages that window can still read.
    """

    layers: int
    kv_heads: int
    head_dim: int
    page_tokens: int = 256
    dtype: Any = jnp.float8_e4m3fn
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        if min(self.layers, self.kv_heads, self.head_dim, self.page_tokens) <= 0:
            raise ValueError("cache geometry must be positive")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if self.page_tokens % self.packing:
            raise ValueError(
                f"page_tokens must be a multiple of packing {self.packing}"
            )

    @property
    def packing(self) -> int:
        bits = jnp.dtype(self.dtype).itemsize * 8
        if 32 % bits:
            raise ValueError(f"cache dtype {self.dtype} does not pack into 32 bits")
        return 32 // bits

    def rows_per_shard(self, tp: int) -> int:
        if self.kv_heads % tp:
            raise ValueError(f"{self.kv_heads} KV heads do not shard over tp={tp}")
        return _align(2 * self.kv_heads // tp, self.packing) // self.packing

    def page_shape(self, tp: int) -> tuple[int, int, int, int]:
        """Global ``[page_tokens, rows, packing, round_up(head_dim, 128)]`` of
        one layer's page."""

        return (
            self.page_tokens,
            tp * self.rows_per_shard(tp),
            self.packing,
            _align(self.head_dim, LANES),
        )

    def slab_bytes(self, tp: int) -> int:
        """One Engine slab: one page for every layer of the group."""

        page = math.prod(self.page_shape(tp)) * jnp.dtype(self.dtype).itemsize
        return self.layers * page

    def pages_per_seq(self, max_seq_len: int) -> int:
        """Page-table width; the spare column holds a decode lease reserved
        past the last readable page."""

        return math.ceil(max_seq_len / self.page_tokens) + 1


def _group_layout(spec: KvCacheSpec, chunk_tokens: int) -> CacheGroupLayout:
    if spec.sliding_window is None:
        return CacheGroupLayout(spec.page_tokens)
    # A chunk's earliest query reads back sliding_window - 1 tokens, and the
    # chunk's own writes must stay resident while it runs.
    return CacheGroupLayout(
        spec.page_tokens,
        retention=CacheRetention.WINDOW,
        window_tokens=spec.sliding_window + chunk_tokens - 1,
        read_window_tokens=spec.sliding_window,
    )


def cache_layout(
    specs: Sequence[KvCacheSpec], parent_capacity: int, *, chunk_tokens: int
) -> CacheLayout:
    """One Engine cache group per spec; ``chunk_tokens`` is the largest
    number of tokens one step writes per sequence."""

    if not specs:
        raise ValueError("a TPU model must declare at least one cache group")
    return CacheLayout(
        parent_capacity=parent_capacity,
        groups=[_group_layout(spec, chunk_tokens) for spec in specs],
    )


def plan_capacity(
    specs: Sequence[KvCacheSpec],
    mesh: Mesh,
    *,
    memory_utilization: float,
    workspace_bytes: int | None = None,
) -> CacheCapacityPlan:
    """Size the slab pool from the mesh's free HBM after weights are resident.

    ``workspace_bytes`` is extra HBM held back for activations and compiled
    temporaries across the whole mesh, on top of the ``1 - memory_utilization``
    share the budget already protects. ``None`` holds back only what is needed
    to keep :data:`MIN_HEADROOM_BYTES` free in total, which is nothing at the
    default 0.98 utilization. Each group's arrays span every slab, so a slab
    costs the sum of the groups' slab bytes.
    """

    stats = [device.memory_stats() for device in mesh.devices.flat]
    if any(not stat or "bytes_limit" not in stat for stat in stats):
        raise ValueError(
            "this platform reports no device memory; pass cache_slab_capacity"
        )
    total = sum(int(stat["bytes_limit"]) for stat in stats)
    used = sum(int(stat.get("bytes_in_use", 0)) for stat in stats)
    tp = tp_size(mesh)
    if workspace_bytes is None:
        protected = math.ceil(total * (1.0 - memory_utilization))
        workspace_bytes = max(0, MIN_HEADROOM_BYTES - protected)
    return plan_cache_capacity(
        total,
        total - used,
        memory_utilization,
        sum(spec.slab_bytes(tp) for spec in specs),
        overhead_bytes=workspace_bytes,
    )


class CacheArena:
    """Zero-initialized, sharded per-layer page arrays, replaced by each step.

    The runner donates :attr:`caches` to the compiled step and stores the
    returned arrays, so XLA updates pages in place.
    """

    def __init__(
        self, specs: Sequence[KvCacheSpec], layout: CacheLayout, mesh: Mesh
    ) -> None:
        if len(specs) != len(layout.groups):
            raise ValueError("one KvCacheSpec is required per cache group")
        self.binding = CachePoolBinding(layout, resource="TPU KV cache")
        sharding = NamedSharding(mesh, KV_PARTITION)
        tp = tp_size(mesh)
        caches = []
        for spec, group in zip(specs, layout.groups, strict=True):
            pages = (layout.parent_capacity + 1) * group.blocks_per_slab
            shape = (pages, *spec.page_shape(tp))
            zeros = jax.jit(
                lambda shape=shape, dtype=spec.dtype: jnp.zeros(shape, dtype),
                out_shardings=sharding,
            )
            caches.append(tuple(zeros() for _ in range(spec.layers)))
        self.caches: Caches = tuple(caches)
        jax.block_until_ready(self.caches)

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for group in self.caches for array in group)

    def bind_resources(self, resources: ResourceBinding) -> None:
        self.binding.bind_resources(resources)


__all__ = [
    "KV_PARTITION",
    "MIN_HEADROOM_BYTES",
    "CacheArena",
    "Caches",
    "KvCacheSpec",
    "cache_layout",
    "plan_capacity",
]
