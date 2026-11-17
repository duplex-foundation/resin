"""Layer-facing paged attention: K/V quantization, TP sharding and kernel choice."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from resin.backend.tpu import reference
from resin.backend.tpu.arena import KV_PARTITION
from resin.backend.tpu.mesh import TP_AXIS, tp_size
from resin.backend.tpu.metadata import AttentionMetadata

logger = logging.getLogger(__name__)

AttentionImpl = Callable[..., tuple[jax.Array, jax.Array]]
HEADS = P(None, TP_AXIS, None)
Q_PACKED_PARTITION = P(TP_AXIS, None, None, None, None)
KV_MERGED_PARTITION = P(None, TP_AXIS, None, None)


def quantize_kv(x: jax.Array, dtype, scale: float | None) -> jax.Array:
    """Cast new K or V to the cache dtype, saturating 8-bit floats instead of overflowing to NaN."""

    dtype = jnp.dtype(dtype)
    if dtype.itemsize != 1:
        return x.astype(dtype)
    bound = float(jnp.finfo(dtype).max)
    scaled = x.astype(jnp.float32) / (1.0 if scale is None else scale)
    return jnp.clip(scaled, -bound, bound).astype(dtype)


_KERNEL_MODULE = "resin.backend.tpu.kernels.ragged_paged_attention"


@functools.cache
def default_impl() -> AttentionImpl:
    """The Pallas kernel on TPU, otherwise the pure-JAX reference."""

    if jax.default_backend() == "tpu":
        try:
            from resin.backend.tpu.kernels.ragged_paged_attention import (
                ragged_paged_attention,
            )
        except ModuleNotFoundError as error:
            # Only an absent kernel module falls back; a broken import inside
            # it must fail rather than serve the reference at serving shapes.
            if not f"{_KERNEL_MODULE}.".startswith(f"{error.name}."):
                raise
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
    values: jax.Array | None,
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
    prepared: bool = False,
    return_packed: bool = False,
    actual_num_q_heads: int | None = None,
    actual_head_dim: int | None = None,
    impl: AttentionImpl | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Attend for one layer and return ``(output, updated kv_cache)``.

    ``queries`` is ``[tokens, q_heads, head_dim]`` and ``keys``/``values`` are
    ``[tokens, kv_heads, head_dim]``, all sharded on heads over ``tp`` (or,
    when ``prepared=True``, pre-packed ``q_packed`` and ``kv_merged`` with
    ``values=None``). The kernel runs once per die on local shapes. Pass the
    ``group`` whose cache ``kv_cache`` is and that group's
    ``KvCacheSpec.sliding_window``, so the page table and the Engine's
    retention match this layer. K/V scales apply only to 8-bit caches.
    """

    if return_packed and not prepared:
        raise ValueError("return_packed=True requires prepared=True")
    if jnp.dtype(kv_cache.dtype).itemsize != 1:
        k_scale = v_scale = None
    chosen_impl = impl or default_impl()

    if prepared:
        if values is not None:
            raise ValueError("values must be None when prepared=True")
        tp = tp_size(mesh)
        if actual_num_q_heads is not None and actual_num_q_heads % tp != 0:
            raise ValueError(f"{actual_num_q_heads=} must be divisible by tp={tp}")
        local_q_heads = None if actual_num_q_heads is None else actual_num_q_heads // tp
        out_partition = Q_PACKED_PARTITION if return_packed else HEADS
        if chosen_impl is reference.ragged_paged_attention:
            from resin.backend.tpu.kernels.ragged_paged_attention import (
                pack_outputs,
                unpack_prepared_inputs,
            )

            def _ref_prepared(
                q_packed: jax.Array,
                kv_merged: jax.Array,
                cache_local: jax.Array,
                kv_lens: Any,
                pages: Any,
                cu_q_lens: Any,
                dist: Any,
            ) -> tuple[jax.Array, jax.Array]:
                q_u, k_u, v_u = unpack_prepared_inputs(
                    q_packed,
                    kv_merged,
                    actual_num_q_heads=local_q_heads,
                    actual_num_kv_heads=q_packed.shape[0],
                    actual_head_dim=actual_head_dim,
                )
                ref_out, ref_cache = reference.ragged_paged_attention(
                    q_u,
                    k_u,
                    v_u,
                    cache_local,
                    kv_lens,
                    pages,
                    cu_q_lens,
                    dist,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    soft_cap=soft_cap,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
                if return_packed:
                    return pack_outputs(ref_out, q_packed.shape[0]), ref_cache
                return ref_out, ref_cache

            return jax.shard_map(
                _ref_prepared,
                mesh=mesh,
                in_specs=(
                    Q_PACKED_PARTITION,
                    KV_MERGED_PARTITION,
                    KV_PARTITION,
                    P(),
                    P(),
                    P(),
                    P(),
                ),
                out_specs=(out_partition, KV_PARTITION),
                check_vma=False,
            )(
                queries,
                keys,
                kv_cache,
                metadata.kv_lens,
                metadata.page_indices[group],
                metadata.cu_q_lens,
                metadata.distribution,
            )

        extra_kw: dict[str, Any] = {"return_packed": True} if return_packed else {}

        def _kernel_prepared(
            q_packed: jax.Array,
            kv_merged: jax.Array,
            cache_local: jax.Array,
            kv_lens: Any,
            pages: Any,
            cu_q_lens: Any,
            dist: Any,
        ) -> tuple[jax.Array, jax.Array]:
            return chosen_impl(
                q_packed,
                kv_merged,
                None,
                cache_local,
                kv_lens,
                pages,
                cu_q_lens,
                dist,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                k_scale=k_scale,
                v_scale=v_scale,
                prepared=True,
                actual_num_q_heads=local_q_heads,
                actual_head_dim=actual_head_dim,
                **extra_kw,
            )

        return jax.shard_map(
            _kernel_prepared,
            mesh=mesh,
            in_specs=(
                Q_PACKED_PARTITION,
                KV_MERGED_PARTITION,
                KV_PARTITION,
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=(out_partition, KV_PARTITION),
            check_vma=False,
        )(
            queries,
            keys,
            kv_cache,
            metadata.kv_lens,
            metadata.page_indices[group],
            metadata.cu_q_lens,
            metadata.distribution,
        )

    if values is None:
        raise ValueError("values is required when prepared=False")
    kernel = functools.partial(
        chosen_impl,
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


__all__ = [
    "KV_MERGED_PARTITION",
    "Q_PACKED_PARTITION",
    "AttentionImpl",
    "default_impl",
    "paged_attention",
    "quantize_kv",
]
