"""Layer-facing paged attention: K/V quantization, TP sharding and kernel choice."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from resin.backend.tpu import reference
from resin.backend.tpu.arena import KV_PARTITION
from resin.backend.tpu.mesh import TP_AXIS
from resin.backend.tpu.metadata import AttentionMetadata

logger = logging.getLogger(__name__)

AttentionImpl = Callable[..., tuple[jax.Array, jax.Array]]
HEADS = P(None, TP_AXIS, None)


def quantize_kv(x: jax.Array, dtype, scale: float | None) -> jax.Array:
    """Cast new K or V to the cache dtype, saturating 8-bit floats instead of overflowing to NaN."""

    dtype = jnp.dtype(dtype)
    if dtype.itemsize != 1:
        return x.astype(dtype)
    bound = float(jnp.finfo(dtype).max)
    scaled = x.astype(jnp.float32) / (1.0 if scale is None else scale)
    return jnp.clip(scaled, -bound, bound).astype(dtype)


@functools.cache
def default_impl() -> AttentionImpl:
    """The Pallas kernel on TPU, otherwise the pure-JAX reference."""

    if jax.default_backend() == "tpu":
        try:
            from resin.backend.tpu.kernels.ragged_paged_attention import (
                ragged_paged_attention,
            )
        except ImportError:
            logger.warning(
                "ragged paged attention kernel is unavailable; using the slow "
                "pure-JAX reference on TPU"
            )
        else:
            return ragged_paged_attention
    return reference.ragged_paged_attention


def paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    metadata: AttentionMetadata,
    *,
    mesh: Mesh,
    sm_scale: float,
    group: int = 0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    impl: AttentionImpl | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Attend for one layer and return ``(output, updated kv_cache)``.

    ``queries`` is ``[tokens, q_heads, head_dim]`` and ``keys``/``values`` are
    ``[tokens, kv_heads, head_dim]``, all sharded on heads over ``tp``. The
    kernel runs once per die on local shapes.
    """

    kernel = functools.partial(
        impl or default_impl(),
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    sharded = jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(HEADS, HEADS, HEADS, KV_PARTITION, P(), P(), P(), P()),
        out_specs=(HEADS, KV_PARTITION),
        check_vma=False,
    )
    return sharded(
        queries,
        quantize_kv(keys, kv_cache.dtype, k_scale),
        quantize_kv(values, kv_cache.dtype, v_scale),
        kv_cache,
        metadata.kv_lens,
        metadata.page_indices[group],
        metadata.cu_q_lens,
        metadata.distribution,
    )


__all__ = ["AttentionImpl", "default_impl", "paged_attention", "quantize_kv"]
