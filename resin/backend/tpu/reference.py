"""Pure-JAX ragged paged attention with the RPA v3 signature and cache layout.

This is the CPU fallback and the parity oracle for the Pallas kernel. It is
jit-traceable with static shapes and gathers every row's full page table, so
its cost scales with the table width rather than the live history. Use the
kernel on TPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Append this step's K/V to their pages, then attend causally over them.

    ``keys`` and ``values`` are already in the cache dtype. Tokens at or past
    ``cu_q_lens[distribution[2]]`` belong to no row: they write nothing and
    their output rows are zero. Masking precedes arithmetic, so stale or NaN
    page tails cannot reach valid outputs.
    """

    tokens, q_heads, head_dim = queries.shape
    kv_heads = keys.shape[1]
    pages, page_tokens, rows, packing, padded_dim = kv_cache.shape
    seqs = kv_lens.shape[0]
    width = page_indices.shape[0] // seqs
    group_size = q_heads // kv_heads

    token = jnp.arange(tokens)
    valid = token < cu_q_lens[distribution[2]]
    row = jnp.clip(jnp.searchsorted(cu_q_lens, token, side="right") - 1, 0, seqs - 1)
    start = cu_q_lens[row]
    position = kv_lens[row] - (cu_q_lens[row + 1] - start) + token - start

    combined = jnp.stack((keys, values), axis=2).reshape(tokens, 2 * kv_heads, head_dim)
    combined = jnp.pad(
        combined,
        ((0, 0), (0, rows * packing - 2 * kv_heads), (0, padded_dim - head_dim)),
    ).reshape(tokens, rows, packing, padded_dim)
    page = page_indices[row * width + position // page_tokens]
    page = jnp.where(valid, page, pages)
    kv_cache = kv_cache.at[page, position % page_tokens].set(
        combined.astype(kv_cache.dtype), mode="drop"
    )

    history = kv_cache[page_indices.reshape(seqs, width)].reshape(
        seqs, width * page_tokens, rows * packing, padded_dim
    )[:, :, : 2 * kv_heads, :head_dim]
    length = width * page_tokens
    key_index = jnp.arange(length)
    stored = key_index[None, :] < kv_lens[:, None]
    history = jnp.where(stored[:, :, None, None], history.astype(jnp.float32), 0.0)
    cached_keys = history[:, :, 0::2] * (1.0 if k_scale is None else k_scale)
    cached_values = history[:, :, 1::2] * (1.0 if v_scale is None else v_scale)

    query = queries.astype(jnp.float32).reshape(tokens, kv_heads, group_size, head_dim)
    scores = jnp.einsum("tkgd,tlkd->tkgl", query, cached_keys[row]) * sm_scale
    if soft_cap is not None:
        scores = soft_cap * jnp.tanh(scores / soft_cap)
    visible = (key_index[None, :] <= position[:, None]) & stored[row] & valid[:, None]
    if sliding_window is not None:
        visible &= key_index[None, :] > position[:, None] - sliding_window
    scores = jnp.where(visible[:, None, None, :], scores, -jnp.inf)
    weights = jax.nn.softmax(scores, axis=-1)
    weights = jnp.where(valid[:, None, None, None], weights, 0.0)
    output = jnp.einsum("tkgl,tlkd->tkgd", weights, cached_values[row])
    return output.reshape(tokens, q_heads, head_dim).astype(queries.dtype), kv_cache


__all__ = ["ragged_paged_attention"]
