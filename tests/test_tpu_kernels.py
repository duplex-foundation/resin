"""Contract, numerical-parity, sharding and compilation-stability tests for TPU kernels."""

from __future__ import annotations

import functools
import os
from typing import Any, Literal

import numpy as np
import pytest

if "TPU_VISIBLE_CHIPS" not in os.environ:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from resin.backend.tpu import reference  # noqa: E402
from resin.backend.tpu.arena import KV_PARTITION, KvCacheSpec  # noqa: E402
from resin.backend.tpu.attention import default_impl, paged_attention  # noqa: E402
from resin.backend.tpu.compilation import CompilationCounter  # noqa: E402
from resin.backend.tpu.kernels.fp8_matmul import (  # noqa: E402
    _TPU7X_FP8_TUNED_BLOCKS,
    Fp8MatmulTileSizes,
    fp8_matmul,
    fp8_matmul_prequantized,
    get_tuned_block_sizes,
    quantize_fp8_weight,
    sharded_fp8_matmul,
    xla_fp8_matmul,
)
from resin.backend.tpu.kernels.norm_rope import (  # noqa: E402
    apply_rope,
    build_rope_cache,
    fused_add_rms_norm,
    rms_norm,
    rms_norm_fp8_quant,
)
from resin.backend.tpu.kernels.ragged_paged_attention import (  # noqa: E402
    RpaCase,
    get_default_block_sizes,
    get_kv_cache_shape,
    prepare_inputs,
    prepare_outputs,
    quantize_kv,
    ragged_paged_attention,
)
from resin.backend.tpu.mesh import make_mesh  # noqa: E402
from resin.backend.tpu.metadata import AttentionMetadata  # noqa: E402


def _reference_rpa(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    kv_cache: np.ndarray,
    kv_lens: np.ndarray,
    page_indices: np.ndarray,
    cu_q_lens: np.ndarray,
    distribution: np.ndarray,
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy reference for ragged paged attention v3."""

    q = np.asarray(queries, dtype=np.float32)
    _, q_heads, head_dim = q.shape
    kv_heads = keys.shape[1]
    group_size = q_heads // kv_heads
    cache = np.array(kv_cache, copy=True)
    _, page_size, rows, packing, padded_dim = cache.shape
    max_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_seqs

    cache_dtype = kv_cache.dtype
    if keys.dtype != cache_dtype:
        bound = float(jnp.finfo(cache_dtype).max)
        eff_k = 1.0 if k_scale is None else k_scale
        eff_v = 1.0 if v_scale is None else v_scale
        k_stored = np.asarray(
            jnp.clip(jnp.asarray(keys, jnp.float32) / eff_k, -bound, bound).astype(
                cache_dtype
            )
        )
        v_stored = np.asarray(
            jnp.clip(jnp.asarray(values, jnp.float32) / eff_v, -bound, bound).astype(
                cache_dtype
            )
        )
    else:
        k_stored = np.asarray(keys)
        v_stored = np.asarray(values)

    out = np.zeros_like(q)
    num_seqs = int(distribution[2])

    for seq in range(num_seqs):
        q_start = int(cu_q_lens[seq])
        q_end = int(cu_q_lens[seq + 1])
        q_len = q_end - q_start
        kv_len = int(kv_lens[seq])
        prefix_len = kv_len - q_len
        table = page_indices[seq * pages_per_seq : (seq + 1) * pages_per_seq]

        if update_kv_cache:
            for t in range(q_len):
                pos = prefix_len + t
                page = int(table[pos // page_size])
                offset = pos % page_size
                slot = np.zeros((rows * packing, padded_dim), dtype=cache_dtype)
                for h in range(kv_heads):
                    slot[2 * h, :head_dim] = k_stored[q_start + t, h]
                    slot[2 * h + 1, :head_dim] = v_stored[q_start + t, h]
                cache[page, offset] = slot.reshape(rows, packing, padded_dim)

        hist_k = np.zeros((kv_len, kv_heads, head_dim), dtype=np.float32)
        hist_v = np.zeros((kv_len, kv_heads, head_dim), dtype=np.float32)
        for pos in range(kv_len):
            page = int(table[pos // page_size])
            offset = pos % page_size
            unpacked = cache[page, offset].reshape(rows * packing, padded_dim)
            for h in range(kv_heads):
                hist_k[pos, h] = unpacked[2 * h, :head_dim].astype(np.float32)
                hist_v[pos, h] = unpacked[2 * h + 1, :head_dim].astype(np.float32)

        q_seq = q[q_start:q_end].copy()
        if q_scale is not None:
            q_bound = float(jnp.finfo(cache_dtype).max)
            q_seq = np.asarray(
                jnp.clip(jnp.asarray(q_seq) / q_scale, -q_bound, q_bound)
                .astype(cache_dtype)
                .astype(jnp.float32)
            )

        k_rep = np.repeat(hist_k, group_size, axis=1)
        v_rep = np.repeat(hist_v, group_size, axis=1)
        scores = np.einsum("qhd,khd->hqk", q_seq, k_rep) * sm_scale
        if k_scale is not None:
            scores *= k_scale
        if q_scale is not None:
            scores *= q_scale
        if soft_cap is not None:
            scores = soft_cap * np.tanh(scores / soft_cap)

        if use_causal_mask:
            q_pos = prefix_len + np.arange(q_len)[:, None]
            k_pos = np.arange(kv_len)[None, :]
            mask = q_pos >= k_pos
            if sliding_window is not None:
                mask &= q_pos - k_pos < sliding_window
            scores = np.where(mask[None, :, :], scores, -1e30)

        scores_max = np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores - scores_max)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        seq_out = np.einsum("hqk,khd->qhd", probs, v_rep)
        if v_scale is not None:
            seq_out *= v_scale
        out[q_start:q_end] = seq_out

    return np.asarray(jnp.asarray(out).astype(queries.dtype)), cache


class TestRaggedPagedAttention:
    def test_kv_cache_shape_interleaves_kv_and_pads_head_dim(self) -> None:
        assert get_kv_cache_shape(37, 256, 4, 128, jnp.float8_e4m3fn) == (
            37,
            256,
            2,
            4,
            128,
        )
        assert get_kv_cache_shape(16, 128, 4, 64, jnp.bfloat16) == (
            16,
            128,
            4,
            2,
            128,
        )

    @pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_prefill_continuation_and_decode_across_page_boundaries(
        self, kv_dtype: jnp.dtype, head_dim: int
    ) -> None:
        rng = np.random.default_rng(42)
        page_size = 16
        total_pages = 20
        max_seqs = 4
        pages_per_seq = 4
        max_tokens = 48
        q_heads, kv_heads = 8, 2
        sm_scale = head_dim**-0.5
        k_scale = 0.5 if kv_dtype == jnp.float8_e4m3fn else None
        v_scale = 0.25 if kv_dtype == jnp.float8_e4m3fn else None

        cache_shape = get_kv_cache_shape(
            total_pages, page_size, kv_heads, head_dim, kv_dtype
        )
        kv_cache_np = np.zeros(cache_shape, dtype=kv_dtype)

        # Prefill 2 active sequences of lengths 19 (crosses page boundary 16) and 7,
        # padded to bucket (max_tokens=48, max_seqs=4). Put garbage in padded tokens.
        q_np = rng.standard_normal((max_tokens, q_heads, head_dim)).astype(np.float32)
        k_np = rng.standard_normal((max_tokens, kv_heads, head_dim)).astype(np.float32)
        v_np = rng.standard_normal((max_tokens, kv_heads, head_dim)).astype(np.float32)
        q_jax = jnp.asarray(q_np, dtype=jnp.bfloat16)
        k_jax = quantize_kv(jnp.asarray(k_np, dtype=jnp.bfloat16), kv_dtype, k_scale)
        v_jax = quantize_kv(jnp.asarray(v_np, dtype=jnp.bfloat16), kv_dtype, v_scale)

        kv_lens = np.array([19, 7, 0, 0], dtype=np.int32)
        cu_q_lens = np.array([0, 19, 26, 26, 26], dtype=np.int32)
        # Prefill uses [0, 0, n] so variable-length rows run RpaCase.MIXED.
        distribution = np.array([0, 0, 2], dtype=np.int32)
        page_table = np.zeros((max_seqs, pages_per_seq), dtype=np.int32)
        page_table[0, :2] = [3, 11]
        page_table[1, :1] = [7]
        flat_pages = page_table.reshape(-1)

        out_jax, cache_after_prefill = ragged_paged_attention(
            q_jax,
            k_jax,
            v_jax,
            jnp.asarray(kv_cache_np),
            jnp.asarray(kv_lens),
            jnp.asarray(flat_pages),
            jnp.asarray(cu_q_lens),
            jnp.asarray(distribution),
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        ref_out, ref_cache = _reference_rpa(
            np.asarray(q_jax),
            np.asarray(k_jax),
            np.asarray(v_jax),
            kv_cache_np,
            kv_lens,
            flat_pages,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        atol = 3e-2 if kv_dtype == jnp.float8_e4m3fn else 2e-2
        np.testing.assert_allclose(
            np.asarray(out_jax[:26], dtype=np.float32),
            np.asarray(ref_out[:26], dtype=np.float32),
            rtol=2e-2,
            atol=atol,
        )
        # Null page 0 must remain untouched.
        np.testing.assert_array_equal(
            np.asarray(cache_after_prefill[0], dtype=np.float32), 0.0
        )

        # Decode step: sequence 0 advances 19 -> 20, sequence 1 advances 7 -> 8,
        # plus sequence 1 continues later across boundary 16 -> 17.
        q_dec = jnp.asarray(
            rng.standard_normal((8, q_heads, head_dim)), dtype=jnp.bfloat16
        )
        k_dec = jnp.asarray(
            rng.standard_normal((8, kv_heads, head_dim)), dtype=jnp.bfloat16
        )
        v_dec = jnp.asarray(
            rng.standard_normal((8, kv_heads, head_dim)), dtype=jnp.bfloat16
        )
        kv_lens_dec = np.array([20, 8, 0, 0], dtype=np.int32)
        cu_q_dec = np.array([0, 1, 2, 2, 2], dtype=np.int32)
        dist_dec = np.array([2, 2, 2], dtype=np.int32)

        # Pass unquantized bf16 keys/values directly to verify internal quantize_kv.
        out_dec, _ = ragged_paged_attention(
            q_dec,
            k_dec,
            v_dec,
            cache_after_prefill,
            jnp.asarray(kv_lens_dec),
            jnp.asarray(flat_pages),
            jnp.asarray(cu_q_dec),
            jnp.asarray(dist_dec),
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        ref_dec, _ = _reference_rpa(
            np.asarray(q_dec),
            np.asarray(k_dec),
            np.asarray(v_dec),
            ref_cache,
            kv_lens_dec,
            flat_pages,
            cu_q_dec,
            dist_dec,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        np.testing.assert_allclose(
            np.asarray(out_dec[:2], dtype=np.float32),
            np.asarray(ref_dec[:2], dtype=np.float32),
            rtol=2e-2,
            atol=atol,
        )

    def test_nan_page_tails_sliding_window_soft_cap_and_idle_batch(self) -> None:
        rng = np.random.default_rng(7)
        page_size = 8
        total_pages = 8
        q_heads, kv_heads, head_dim = 4, 2, 128
        sm_scale = head_dim**-0.5

        cache_shape = get_kv_cache_shape(
            total_pages, page_size, kv_heads, head_dim, jnp.bfloat16
        )
        # Fill entire arena with NaNs first; write 11 tokens across pages 2 and 5
        # (update_sz > page_size) with sliding_window=4 so both pre-window and
        # post-kv_len NaN regions exist in the Pallas kernel's fetched block.
        dirty_cache = np.full(cache_shape, np.nan, dtype=np.float32)
        dirty_cache[0] = 0.0
        kv_cache = jnp.asarray(dirty_cache, dtype=jnp.bfloat16)

        q = jnp.asarray(rng.standard_normal((16, q_heads, head_dim)), jnp.bfloat16)
        k = jnp.asarray(rng.standard_normal((16, kv_heads, head_dim)), jnp.bfloat16)
        v = jnp.asarray(rng.standard_normal((16, kv_heads, head_dim)), jnp.bfloat16)
        kv_lens = jnp.array([11, 0], dtype=jnp.int32)
        pages = jnp.array([2, 5, 0, 0], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 11, 11], dtype=jnp.int32)
        dist = jnp.array([0, 0, 1], dtype=jnp.int32)

        out, updated = ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            pages,
            cu_q_lens,
            dist,
            sm_scale=sm_scale,
            sliding_window=4,
            soft_cap=30.0,
            d_block_sizes=(1, 8, 1, 8),
            m_block_sizes=(4, 16, 4, 8),
            interpret=True,
        )
        ref_out, _ = _reference_rpa(
            np.asarray(q),
            np.asarray(k),
            np.asarray(v),
            np.asarray(kv_cache),
            np.asarray(kv_lens),
            np.asarray(pages),
            np.asarray(cu_q_lens),
            np.asarray(dist),
            sm_scale=sm_scale,
            sliding_window=4,
            soft_cap=30.0,
        )
        assert not np.any(np.isnan(np.asarray(out[:11], dtype=np.float32)))
        np.testing.assert_allclose(
            np.asarray(out[:11], dtype=np.float32),
            np.asarray(ref_out[:11], dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

        # Idle batch (distribution = [0, 0, 0]) must not touch the cache or fail.
        idle_out, idle_cache = ragged_paged_attention(
            q,
            k,
            v,
            updated,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.zeros((4,), dtype=jnp.int32),
            jnp.zeros((3,), dtype=jnp.int32),
            jnp.zeros((3,), dtype=jnp.int32),
            sm_scale=sm_scale,
            d_block_sizes=(1, 8, 1, 8),
            m_block_sizes=(4, 16, 4, 8),
            interpret=True,
        )
        assert idle_out.shape == q.shape
        np.testing.assert_array_equal(
            np.asarray(idle_cache[2, :8], dtype=np.float32),
            np.asarray(updated[2, :8], dtype=np.float32),
        )

    def test_no_recompilation_when_request_lengths_and_pages_change(self) -> None:
        @functools.partial(jax.jit, donate_argnums=(3,))
        def step(q, k, v, cache, kv_lens, pages, cu_q_lens, dist):
            return ragged_paged_attention(
                q,
                k,
                v,
                cache,
                kv_lens,
                pages,
                cu_q_lens,
                dist,
                sm_scale=128**-0.5,
                d_block_sizes=(1, 16, 1, 16),
                m_block_sizes=(4, 16, 4, 16),
                interpret=True,
            )

        cache_shape = get_kv_cache_shape(16, 16, 2, 128, jnp.float8_e4m3fn)
        cache = jnp.zeros(cache_shape, dtype=jnp.float8_e4m3fn)
        q = jnp.ones((8, 4, 128), dtype=jnp.bfloat16)
        k = jnp.ones((8, 2, 128), dtype=jnp.float8_e4m3fn)
        v = jnp.ones((8, 2, 128), dtype=jnp.float8_e4m3fn)

        counter = CompilationCounter()
        try:
            warm = None
            for seq_count, lens in [
                (1, [5, 0, 0, 0]),
                (3, [12, 19, 3, 0]),
                (2, [31, 8, 0, 0]),
            ]:
                pages = np.zeros((4, 2), dtype=np.int32)
                pages[0] = [1, 2]
                pages[1] = [3, 4]
                pages[2] = [5, 6]
                cu = np.array(
                    [0] + [min(i, seq_count) for i in range(1, 5)], dtype=np.int32
                )
                out, cache = step(
                    q,
                    k,
                    v,
                    cache,
                    jnp.asarray(lens, dtype=jnp.int32),
                    jnp.asarray(pages.reshape(-1)),
                    jnp.asarray(cu),
                    jnp.array([seq_count, seq_count, seq_count], dtype=jnp.int32),
                )
                out.block_until_ready()
                if warm is None:
                    warm = counter.count
                    assert warm > 0
                else:
                    assert counter.count == warm
        finally:
            counter.close()

    def test_shard_map_across_mesh_matches_single_device_oracle(self) -> None:
        devices = jax.devices()
        tp = min(len(devices), 2)
        mesh = Mesh(np.array(devices[:tp]), ("tp",))

        rng = np.random.default_rng(99)
        total_pages, page_size = 12, 16
        q_heads, kv_heads, head_dim = 8, 4, 128
        local_shape = get_kv_cache_shape(
            total_pages, page_size, kv_heads // tp, head_dim, jnp.float8_e4m3fn
        )
        global_shape = (
            local_shape[0],
            local_shape[1],
            local_shape[2] * tp,
            local_shape[3],
            local_shape[4],
        )
        cache_np = np.zeros(global_shape, dtype=jnp.float8_e4m3fn)
        q_np = rng.standard_normal((16, q_heads, head_dim)).astype(np.float32)
        k_np = rng.standard_normal((16, kv_heads, head_dim)).astype(np.float32)
        v_np = rng.standard_normal((16, kv_heads, head_dim)).astype(np.float32)

        q = jax.device_put(
            jnp.asarray(q_np, jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(None, "tp", None)),
        )
        k = jax.device_put(
            jnp.asarray(k_np, jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(None, "tp", None)),
        )
        v = jax.device_put(
            jnp.asarray(v_np, jnp.bfloat16),
            jax.sharding.NamedSharding(mesh, P(None, "tp", None)),
        )
        cache = jax.device_put(
            jnp.asarray(cache_np),
            jax.sharding.NamedSharding(mesh, P(None, None, "tp", None, None)),
        )
        kv_lens = jnp.array([9, 5], dtype=jnp.int32)
        pages = jnp.array([1, 0, 2, 0], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 9, 14], dtype=jnp.int32)
        dist = jnp.array([0, 0, 2], dtype=jnp.int32)

        sharded_fn = jax.jit(
            jax.shard_map(
                functools.partial(
                    ragged_paged_attention,
                    sm_scale=head_dim**-0.5,
                    k_scale=0.5,
                    v_scale=0.5,
                ),
                mesh=mesh,
                in_specs=(
                    P(None, "tp", None),
                    P(None, "tp", None),
                    P(None, "tp", None),
                    P(None, None, "tp", None, None),
                    P(),
                    P(),
                    P(),
                    P(),
                ),
                out_specs=(P(None, "tp", None), P(None, None, "tp", None, None)),
                check_vma=False,
            ),
            donate_argnums=(3,),
        )
        out, _ = sharded_fn(q, k, v, cache, kv_lens, pages, cu_q_lens, dist)
        ref_out, _ = _reference_rpa(
            np.asarray(jnp.asarray(q_np, jnp.bfloat16)),
            np.asarray(jnp.asarray(k_np, jnp.bfloat16)),
            np.asarray(jnp.asarray(v_np, jnp.bfloat16)),
            cache_np,
            np.asarray(kv_lens),
            np.asarray(pages),
            np.asarray(cu_q_lens),
            np.asarray(dist),
            sm_scale=head_dim**-0.5,
            k_scale=0.5,
            v_scale=0.5,
        )
        np.testing.assert_allclose(
            np.asarray(out[:14], dtype=np.float32),
            np.asarray(ref_out[:14], dtype=np.float32),
            rtol=2e-2,
            atol=3e-2,
        )

    def test_pallas_interpret_prefill_and_decode_fp8_kv(self) -> None:
        shape = get_kv_cache_shape(4, 8, 2, 128, jnp.float8_e4m3fn)
        cache = jnp.zeros(shape, jnp.float8_e4m3fn)
        rng = np.random.default_rng(17)
        q = jnp.asarray(rng.standard_normal((8, 4, 128)), jnp.bfloat16)
        k = jnp.asarray(rng.standard_normal((8, 2, 128)), jnp.bfloat16)
        v = jnp.asarray(rng.standard_normal((8, 2, 128)), jnp.bfloat16)
        pages = jnp.array([2, 1, 0, 0], jnp.int32)

        out_pallas, cache_pallas = ragged_paged_attention(
            q,
            k,
            v,
            cache,
            jnp.array([6, 0], jnp.int32),
            pages,
            jnp.array([0, 6, 6], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=128**-0.5,
            k_scale=0.5,
            v_scale=0.25,
            d_block_sizes=(1, 8, 1, 8),
            m_block_sizes=(4, 8, 2, 8),
            interpret=True,
        )
        ref_out, ref_cache = _reference_rpa(
            np.asarray(q),
            np.asarray(k),
            np.asarray(v),
            np.asarray(cache),
            np.array([6, 0], np.int32),
            np.asarray(pages),
            np.array([0, 6, 6], np.int32),
            np.array([0, 0, 1], np.int32),
            sm_scale=128**-0.5,
            k_scale=0.5,
            v_scale=0.25,
        )
        np.testing.assert_allclose(
            np.asarray(out_pallas[:6], np.float32),
            np.asarray(ref_out[:6], np.float32),
            rtol=2e-2,
            atol=3e-2,
        )

        # Decode crossing page 2 -> page 1 (pos 7 -> pos 8 on page_size=8).
        for next_len in (7, 8, 9):
            qd = jnp.asarray(rng.standard_normal((2, 4, 128)), jnp.bfloat16)
            kd = jnp.asarray(rng.standard_normal((2, 2, 128)), jnp.bfloat16)
            vd = jnp.asarray(rng.standard_normal((2, 2, 128)), jnp.bfloat16)
            out_pallas, cache_pallas = ragged_paged_attention(
                qd,
                kd,
                vd,
                cache_pallas,
                jnp.array([next_len, 0], jnp.int32),
                pages,
                jnp.array([0, 1, 1], jnp.int32),
                jnp.array([1, 1, 1], jnp.int32),
                sm_scale=128**-0.5,
                k_scale=0.5,
                v_scale=0.25,
                d_block_sizes=(1, 8, 1, 8),
                m_block_sizes=(2, 8, 2, 8),
                interpret=True,
            )
            ref_out, ref_cache = _reference_rpa(
                np.asarray(qd),
                np.asarray(kd),
                np.asarray(vd),
                ref_cache,
                np.array([next_len, 0], np.int32),
                np.asarray(pages),
                np.array([0, 1, 1], np.int32),
                np.array([1, 1, 1], np.int32),
                sm_scale=128**-0.5,
                k_scale=0.5,
                v_scale=0.25,
            )
            np.testing.assert_allclose(
                np.asarray(out_pallas[:1], np.float32),
                np.asarray(ref_out[:1], np.float32),
                rtol=2e-2,
                atol=3e-2,
            )

    @pytest.mark.parametrize("pages_per_seq", [1, 6, 9, 37])
    @pytest.mark.parametrize("page_size", [16, 128, 256])
    def test_default_block_sizes_are_always_page_aligned_and_divisible(
        self, pages_per_seq: int, page_size: int
    ) -> None:
        for case in (RpaCase.DECODE, RpaCase.PREFILL, RpaCase.MIXED):
            blocks = get_default_block_sizes(
                jnp.bfloat16,
                jnp.float8_e4m3fn,
                actual_num_q_heads=32,
                actual_num_kv_heads=4,
                head_dim=128,
                page_size=page_size,
                max_num_tokens=4096,
                max_num_seqs=320,
                pages_per_seq=pages_per_seq,
                case=case,
                tpu_version=7,
            )
            assert blocks["bq_sz"] % blocks["bq_csz"] == 0
            assert blocks["bkv_sz"] % blocks["bkv_csz"] == 0
            assert blocks["bkv_sz"] % page_size == 0
            assert blocks["bkv_csz"] % page_size == 0


class TestRpaDecodeBlockSizes:
    def test_rpa_block_sizes_and_vmem_estimate_match_scratch_buffers(self) -> None:
        from resin.backend.tpu.kernels import (
            RpaBlockSizes,
            get_vmem_estimate_bytes,
        )

        blocks = RpaBlockSizes(1, 2048, 1, 2048)
        assert tuple(blocks) == (1, 2048, 1, 2048)
        assert blocks.as_dict() == {
            "bq_sz": 1,
            "bkv_sz": 2048,
            "bq_csz": 1,
            "bkv_csz": 2048,
        }

        # Qwen3-32B TP2 FP8 KV: stride 2 -> 3 (bank conflict), 6 MiB bkv + 80 KiB q/l/m/acc.
        assert (
            get_vmem_estimate_bytes(
                actual_num_kv_heads=4,
                actual_num_q_heads_per_kv_head=8,
                actual_head_dim=128,
                bq_sz=1,
                bkv_sz=2048,
                q_dtype=jnp.bfloat16,
                kv_dtype=jnp.float8_e4m3fn,
            )
            == 6_373_376
        )
        # Qwen3-32B DP2 BF16 KV: stride 8 -> 9 (bank conflict), 18 MiB bkv + 160 KiB q/l/m/acc.
        assert (
            get_vmem_estimate_bytes(
                actual_num_kv_heads=8,
                actual_num_q_heads_per_kv_head=8,
                actual_head_dim=128,
                bq_sz=1,
                bkv_sz=2048,
                q_dtype=jnp.bfloat16,
                kv_dtype=jnp.bfloat16,
            )
            == 19_038_208
        )
        # Unaligned q_per_kv=3 -> 4, hd=64 -> 128, kv_heads=2 fp8 -> stride 1 (no bank conflict).
        assert (
            get_vmem_estimate_bytes(
                actual_num_kv_heads=2,
                actual_num_q_heads_per_kv_head=3,
                actual_head_dim=64,
                bq_sz=1,
                bkv_sz=256,
                q_dtype=jnp.bfloat16,
                kv_dtype=jnp.float8_e4m3fn,
            )
            == 282_624
        )

    def test_decode_block_sizes_divisibility_vmem_spare_page_and_mixed_invariance(
        self,
    ) -> None:
        from resin.backend.tpu.kernels import (
            RpaBlockSizes,
            get_rpa_block_sizes,
            get_vmem_estimate_bytes,
        )
        from resin.backend.tpu.kernels.ragged_paged_attention import (
            static_validate_inputs,
        )

        vmem_cap = int(0.75 * 64 * 1024 * 1024)
        dtypes = (jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.bfloat16, jnp.float32)
        head_dims = (8, 16, 64, 80, 128, 240, 256)
        head_pairs = ((32, 4), (64, 8), (16, 2), (16, 4), (32, 8), (8, 2), (12, 4))

        for page_size in (16, 64, 128, 256):
            for pages_per_seq in (1, 6, 9, 37):
                table_kv = pages_per_seq * page_size
                for kv_dtype in dtypes:
                    for head_dim in head_dims:
                        for q_heads, kv_heads in head_pairs:
                            q = jax.ShapeDtypeStruct(
                                (8, q_heads, head_dim), jnp.bfloat16
                            )
                            k = jax.ShapeDtypeStruct((8, kv_heads, head_dim), kv_dtype)
                            cache = jax.ShapeDtypeStruct(
                                get_kv_cache_shape(
                                    max(pages_per_seq, 8),
                                    page_size,
                                    kv_heads,
                                    head_dim,
                                    kv_dtype,
                                ),
                                kv_dtype,
                            )
                            kv_lens = jax.ShapeDtypeStruct((4,), jnp.int32)
                            page_indices = jax.ShapeDtypeStruct(
                                (4 * pages_per_seq,), jnp.int32
                            )
                            cu_q_lens = jax.ShapeDtypeStruct((5,), jnp.int32)
                            dist = jax.ShapeDtypeStruct((3,), jnp.int32)

                            for sliding_window in (None, 128, 1024):
                                blocks = get_rpa_block_sizes(
                                    jnp.bfloat16,
                                    kv_dtype,
                                    actual_num_q_heads=q_heads,
                                    actual_num_kv_heads=kv_heads,
                                    head_dim=head_dim,
                                    page_size=page_size,
                                    max_num_tokens=320,
                                    max_num_seqs=320,
                                    pages_per_seq=pages_per_seq,
                                    case=RpaCase.DECODE,
                                    tpu_version=7,
                                    sliding_window=sliding_window,
                                )
                                assert isinstance(blocks, RpaBlockSizes)
                                assert (blocks.bq_sz, blocks.bq_csz) == (1, 1)
                                assert blocks.bkv_sz <= table_kv
                                assert blocks.bkv_sz % blocks.bkv_csz == 0
                                assert blocks.bkv_sz % page_size == 0
                                assert blocks.bkv_csz % page_size == 0
                                static_validate_inputs(
                                    q,
                                    k,
                                    k,
                                    cache,
                                    kv_lens,
                                    page_indices,
                                    cu_q_lens,
                                    dist,
                                    sliding_window=sliding_window,
                                    d_block_sizes=tuple(blocks),
                                )
                                vmem = get_vmem_estimate_bytes(
                                    actual_num_kv_heads=kv_heads,
                                    actual_num_q_heads_per_kv_head=q_heads // kv_heads,
                                    actual_head_dim=head_dim,
                                    bq_sz=blocks.bq_sz,
                                    bkv_sz=blocks.bkv_sz,
                                    q_dtype=jnp.bfloat16,
                                    kv_dtype=kv_dtype,
                                )
                                assert vmem <= vmem_cap

        # Check non-v7 TPU versions and CPU interpret (-1) via VMEM heuristic.
        vmem_by_ver = {4: 16, 5: 64, 6: 128, 8: 128, -1: 64}
        for ver, vmem_mib in vmem_by_ver.items():
            ver_blocks = get_rpa_block_sizes(
                jnp.bfloat16,
                jnp.float8_e4m3fn,
                actual_num_q_heads=32,
                actual_num_kv_heads=4,
                head_dim=128,
                page_size=256,
                max_num_tokens=320,
                max_num_seqs=320,
                pages_per_seq=37,
                case=RpaCase.DECODE,
                tpu_version=ver,
            )
            assert ver_blocks.bkv_sz <= (512 if ver == 4 else 4096)
            assert get_vmem_estimate_bytes(
                4, 8, 128, 1, ver_blocks.bkv_sz, jnp.bfloat16, jnp.float8_e4m3fn
            ) <= int(0.375 * vmem_mib * 1024 * 1024)

        decode_v7 = functools.partial(
            get_rpa_block_sizes,
            jnp.bfloat16,
            max_num_tokens=320,
            max_num_seqs=320,
            case=RpaCase.DECODE,
            tpu_version=7,
        )
        # Spare-page regression checks on Qwen3-32B TP2 (1k/1k -> 9 pages, 9,216 -> 37 pages).
        short_srv = decode_v7(
            jnp.float8_e4m3fn,
            actual_num_q_heads=32,
            actual_num_kv_heads=4,
            head_dim=128,
            page_size=256,
            pages_per_seq=9,
        )
        assert short_srv.bkv_sz != 2304 and short_srv.bkv_sz <= 2048
        assert short_srv.bkv_csz > 512

        long_srv = decode_v7(
            jnp.float8_e4m3fn,
            actual_num_q_heads=32,
            actual_num_kv_heads=4,
            head_dim=128,
            page_size=256,
            pages_per_seq=37,
        )
        assert long_srv.bkv_sz != 9472 and long_srv.bkv_sz <= 4096

        # Sliding-window layers shrink bkv_sz to the window horizon; wide BF16 shapes step down.
        assert (
            decode_v7(
                jnp.bfloat16,
                actual_num_q_heads=32,
                actual_num_kv_heads=4,
                head_dim=64,
                page_size=128,
                pages_per_seq=37,
                sliding_window=128,
            ).bkv_sz
            == 256
        )
        assert (
            decode_v7(
                jnp.bfloat16,
                actual_num_q_heads=64,
                actual_num_kv_heads=8,
                head_dim=128,
                page_size=256,
                pages_per_seq=37,
            ).bkv_sz
            == 2048
        )
        assert (
            decode_v7(
                jnp.bfloat16,
                actual_num_q_heads=32,
                actual_num_kv_heads=8,
                head_dim=256,
                page_size=256,
                pages_per_seq=37,
            ).bkv_sz
            == 1024
        )

        # RpaCase.PREFILL remains identical to get_default_block_sizes; RpaCase.MIXED uses tuned v7 table.
        expected_default = {
            (9, 256): RpaBlockSizes(256, 1024, 128, 512),
            (37, 256): RpaBlockSizes(256, 2048, 128, 512),
            (9, 128): RpaBlockSizes(256, 512, 128, 512),
        }
        for (pps, ps), want in expected_default.items():
            for sw, vmem_lim in ((None, None), (128, 8 * 1024 * 1024)):
                got_prefill = get_rpa_block_sizes(
                    jnp.bfloat16,
                    jnp.float8_e4m3fn,
                    actual_num_q_heads=32,
                    actual_num_kv_heads=4,
                    head_dim=128,
                    page_size=ps,
                    max_num_tokens=4096,
                    max_num_seqs=320,
                    pages_per_seq=pps,
                    case=RpaCase.PREFILL,
                    tpu_version=7,
                    sliding_window=sw,
                    vmem_limit_bytes=vmem_lim,
                )
                assert got_prefill == want
                for case in (RpaCase.MIXED, RpaCase.PREFILL):
                    assert (
                        get_default_block_sizes(
                            jnp.bfloat16,
                            jnp.float8_e4m3fn,
                            actual_num_q_heads=32,
                            actual_num_kv_heads=4,
                            head_dim=128,
                            page_size=ps,
                            max_num_tokens=4096,
                            max_num_seqs=320,
                            pages_per_seq=pps,
                            case=case,
                            tpu_version=7,
                            sliding_window=sw,
                            vmem_limit_bytes=vmem_lim,
                        )
                        == want.as_dict()
                    )

    @pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
    @pytest.mark.parametrize("sliding_window", [None, 12])
    def test_pallas_interpret_elided_loop_matches_split_loop_and_reference(
        self, kv_dtype: jnp.dtype, sliding_window: int | None
    ) -> None:
        rng = np.random.default_rng(1234)
        page_size = 8
        total_pages = 20
        max_seqs = 4
        q_heads, kv_heads, head_dim = 8, 2, 64
        sm_scale = head_dim**-0.5
        k_scale = 0.5 if kv_dtype == jnp.float8_e4m3fn else None
        v_scale = 0.25 if kv_dtype == jnp.float8_e4m3fn else None

        cache_shape = get_kv_cache_shape(
            total_pages, page_size, kv_heads, head_dim, kv_dtype
        )
        # Poison the entire arena with NaNs outside null page 0, then populate
        # only the active prefix tokens [6, 15, 16, 28] before decode [7, 16, 17, 29].
        dirty = np.full(cache_shape, np.nan, dtype=np.float32)
        dirty[0] = 0.0
        kv_lens_arr = np.array([7, 16, 17, 29], dtype=np.int32)
        page_table = np.array(
            [
                [1, 0, 0, 0, 0],
                [2, 3, 0, 0, 0],
                [4, 5, 6, 0, 0],
                [7, 8, 9, 10, 0],
            ],
            dtype=np.int32,
        )
        rows = cache_shape[2]
        packing = cache_shape[3]
        padded_dim = cache_shape[4]
        for seq_idx, kv_len in enumerate(kv_lens_arr):
            prefix = int(kv_len) - 1
            for pos in range(prefix):
                p = int(page_table[seq_idx, pos // page_size])
                off = pos % page_size
                slot = np.zeros((rows * packing, padded_dim), dtype=np.float32)
                for h in range(kv_heads):
                    slot[2 * h, :head_dim] = rng.standard_normal(head_dim) * 0.5
                    slot[2 * h + 1, :head_dim] = rng.standard_normal(head_dim) * 0.5
                dirty[p, off] = slot.reshape(rows, packing, padded_dim)

        init_cache = quantize_kv(jnp.asarray(dirty, jnp.float32), kv_dtype)
        # Re-poison unused pages and unused tail slots of owned pages with NaN in kv_dtype.
        nan_tile = np.asarray(
            jnp.full((rows, packing, padded_dim), jnp.nan, dtype=kv_dtype)
        )
        init_cache_np = np.array(init_cache, copy=True)
        used_pages = {0}
        for seq_idx, kv_len in enumerate(kv_lens_arr):
            used_slots = int(kv_len)
            for p_col in range((used_slots + page_size - 1) // page_size):
                p_id = int(page_table[seq_idx, p_col])
                used_pages.add(p_id)
                valid_in_page = min(page_size, used_slots - p_col * page_size)
                if valid_in_page < page_size:
                    init_cache_np[p_id, valid_in_page:] = nan_tile
        for p_id in range(total_pages):
            if p_id not in used_pages:
                init_cache_np[p_id, :] = nan_tile

        q_f32 = jnp.asarray(
            rng.standard_normal((max_seqs, q_heads, head_dim)), jnp.float32
        )
        q_bf16 = q_f32.astype(jnp.bfloat16)
        k_bf16 = jnp.asarray(
            rng.standard_normal((max_seqs, kv_heads, head_dim)), jnp.bfloat16
        )
        v_bf16 = jnp.asarray(
            rng.standard_normal((max_seqs, kv_heads, head_dim)), jnp.bfloat16
        )
        flat_pages = jnp.asarray(page_table.reshape(-1))
        kv_lens = jnp.asarray(kv_lens_arr)
        cu_q_lens = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
        dist = jnp.array([4, 4, 4], dtype=jnp.int32)

        init_cache_jax = jnp.asarray(init_cache_np)
        run_interp = functools.partial(
            ragged_paged_attention,
            keys=k_bf16,
            values=v_bf16,
            kv_cache=init_cache_jax,
            kv_lens=kv_lens,
            page_indices=flat_pages,
            cu_q_lens=cu_q_lens,
            distribution=dist,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            k_scale=k_scale,
            v_scale=v_scale,
            m_block_sizes=(4, 16, 4, 8),
            interpret=True,
        )

        # Verify HLO elides the inner while loop when bkv_csz == bkv_sz vs bkv_csz < bkv_sz.
        if sliding_window is None and kv_dtype == jnp.bfloat16:
            lower_for = lambda d_blocks: ragged_paged_attention.lower(  # noqa: E731
                q_bf16,
                k_bf16,
                v_bf16,
                init_cache_jax,
                kv_lens,
                flat_pages,
                cu_q_lens,
                dist,
                sm_scale=sm_scale,
                d_block_sizes=d_blocks,
                m_block_sizes=(4, 16, 4, 8),
                interpret=True,
            ).as_text()
            assert lower_for((1, 16, 1, 16)).count("stablehlo.while") < lower_for(
                (1, 16, 1, 8)
            ).count("stablehlo.while")

        # Verify _blocks_for forwards sliding_window and vmem_limit_bytes when d_block_sizes=None.
        if sliding_window == 12 and kv_dtype == jnp.bfloat16:
            jaxpr_default = str(
                jax.make_jaxpr(
                    lambda q_in: run_interp(q_in, vmem_limit_bytes=64 * 1024)
                )(q_bf16)
            )
            # 0.375 * 64 KiB = 24,576 B caps bkv_sz from 16 down to page_size=8.
            assert "name=RPAd-p_8-bq_1_1-bkv_8_8-sw_12" in jaxpr_default

        # Float32 pre-cast check: (1, 16, 1, 16) and (1, 16, 1, 8) agree within 1e-5.
        out_f32_elided, _ = run_interp(q_f32, d_block_sizes=(1, 16, 1, 16))
        out_f32_split, _ = run_interp(q_f32, d_block_sizes=(1, 16, 1, 8))
        assert not np.any(np.isnan(np.asarray(out_f32_elided)))
        assert not np.any(np.isnan(np.asarray(out_f32_split)))
        np.testing.assert_allclose(
            np.asarray(out_f32_elided),
            np.asarray(out_f32_split),
            rtol=0.0,
            atol=1e-5,
        )

        # BF16 queries parity against _reference_rpa.
        out_elided, _ = run_interp(q_bf16, d_block_sizes=(1, 16, 1, 16))
        out_split, _ = run_interp(q_bf16, d_block_sizes=(1, 16, 1, 8))
        ref_out, _ = _reference_rpa(
            np.asarray(q_bf16),
            np.asarray(k_bf16),
            np.asarray(v_bf16),
            init_cache_np,
            kv_lens_arr,
            page_table.reshape(-1),
            np.asarray(cu_q_lens),
            np.asarray(dist),
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        for cand in (out_elided, out_split):
            arr = np.asarray(cand, dtype=np.float32)
            assert not np.any(np.isnan(arr))
            np.testing.assert_allclose(
                arr,
                np.asarray(ref_out, dtype=np.float32),
                rtol=2e-2,
                atol=3e-2,
            )

    def test_rpa_v3_case_vmem_limit_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        from jax._src import tpu_info
        from jax.experimental import pallas as pl

        rpa_mod = importlib.import_module(
            "resin.backend.tpu.kernels.ragged_paged_attention"
        )
        v7x_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion("7x"), 1)
        assert v7x_info.vmem_capacity_bytes == 64 * 1024 * 1024

        captured_calls: list[tuple[str, int | None]] = []
        orig_pallas_call = pl.pallas_call

        def _spy_pallas_call(*args: Any, **kwargs: Any) -> Any:
            cp = kwargs.get("compiler_params")
            captured_calls.append(
                (kwargs.get("name", ""), getattr(cp, "vmem_limit_bytes", None))
            )
            return orig_pallas_call(*args, **kwargs)

        monkeypatch.setattr(rpa_mod, "get_tpu_version", lambda: 7)
        monkeypatch.setattr(rpa_mod.pltpu, "get_tpu_info", lambda: v7x_info)
        monkeypatch.setattr(pl, "pallas_call", _spy_pallas_call)

        cache = jax.ShapeDtypeStruct(
            get_kv_cache_shape(64, 256, 2, 128, jnp.float8_e4m3fn), jnp.float8_e4m3fn
        )
        q = jax.ShapeDtypeStruct((8, 16, 128), jnp.bfloat16)
        k = jax.ShapeDtypeStruct((8, 2, 128), jnp.float8_e4m3fn)
        v = jax.ShapeDtypeStruct((8, 2, 128), jnp.float8_e4m3fn)
        kv_lens = jax.ShapeDtypeStruct((8,), jnp.int32)
        page_indices = jax.ShapeDtypeStruct((8 * 36,), jnp.int32)
        cu_q_lens = jax.ShapeDtypeStruct((9,), jnp.int32)
        dist = jax.ShapeDtypeStruct((3,), jnp.int32)

        _ = jax.make_jaxpr(
            lambda q_i, k_i, v_i, c_i, kl_i, pi_i, cq_i, d_i: (
                rpa_mod.ragged_paged_attention(
                    q_i, k_i, v_i, c_i, kl_i, pi_i, cq_i, d_i, sm_scale=128**-0.5
                )
            )
        )(q, k, v, cache, kv_lens, page_indices, cu_q_lens, dist)

        rpad_calls = [
            (name, vmem) for name, vmem in captured_calls if name.startswith("RPAd-")
        ]
        rpam_calls = [
            (name, vmem) for name, vmem in captured_calls if name.startswith("RPAm-")
        ]
        assert rpad_calls and rpam_calls
        assert rpad_calls[0][0] == "RPAd-p_256-bq_1_1-bkv_4096_4096"
        for _name, vmem in rpad_calls + rpam_calls:
            assert vmem is not None
            assert 16 * 1024 * 1024 <= vmem <= 24 * 1024 * 1024


def _qwix_reference_fp8_w8a8(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Independent Qwix symmetric absmax FP8 W8A8 reference (x: [T, K], w: [N, K])."""
    tiny = float(np.sqrt(np.finfo(np.float32).tiny))
    qmax = 448.0

    w_f32 = w.astype(np.float32)
    w_absmax = np.max(np.abs(w_f32), axis=-1, keepdims=True)
    w_scale = np.where(w_absmax / qmax < tiny, 1.0, w_absmax / qmax)
    w_q = jnp.asarray(np.clip(w_f32 / w_scale, -qmax, qmax)).astype(jnp.float8_e4m3fn)

    x_f32 = x.astype(np.float32)
    x_absmax = np.max(np.abs(x_f32), axis=-1, keepdims=True)
    x_scale = np.where(x_absmax / qmax < tiny, 1.0, x_absmax / qmax)
    x_q = jnp.asarray(np.clip(x_f32 / x_scale, -qmax, qmax)).astype(jnp.float8_e4m3fn)

    dot = np.asarray(x_q, np.float32) @ np.asarray(w_q, np.float32).T
    return np.asarray(jnp.asarray(dot * x_scale * w_scale.T, dtype=jnp.bfloat16))


class TestFp8Matmul:
    def test_matches_qwix_contract_and_handles_zero_padded_rows(self) -> None:
        rng = np.random.default_rng(101)
        x_np = rng.standard_normal((20, 256)).astype(np.float32)
        # Simulate padded shape-bucket rows and boundary magnitudes.
        x_np[16:] = 0.0
        x_np[0, 0] = 448.0
        w_np = rng.standard_normal((300, 256)).astype(np.float32)
        w_np[5] = 0.0

        x = jnp.asarray(x_np, jnp.bfloat16)
        w = jnp.asarray(w_np, jnp.bfloat16)
        fw = quantize_fp8_weight(w)
        assert fw.qvalue.shape == (300, 256)
        assert fw.qvalue.dtype == jnp.float8_e4m3fn
        assert fw.scale.shape == (300,)
        assert fw.scale.dtype == jnp.float32

        expected = _qwix_reference_fp8_w8a8(np.asarray(x), np.asarray(w))
        y_xla = xla_fp8_matmul(x, fw.qvalue, fw.scale)
        y_pallas_cached_xq = fp8_matmul(
            x,
            fw.qvalue,
            fw.scale,
            backend="pallas",
            tile_sizes=Fp8MatmulTileSizes(24, 128, 256),
            interpret=True,
        )
        y_pallas_accum_k = fp8_matmul(
            x,
            fw.qvalue,
            fw.scale,
            backend="pallas",
            tile_sizes=Fp8MatmulTileSizes(24, 128, 128),
            interpret=True,
        )

        for candidate in (y_xla, y_pallas_cached_xq, y_pallas_accum_k):
            arr = np.asarray(candidate, np.float32)
            assert not np.any(np.isnan(arr))
            np.testing.assert_allclose(arr[16:], 0.0, atol=0.0)
            np.testing.assert_allclose(
                arr, np.asarray(expected, np.float32), rtol=5e-2, atol=0.55
            )

        for pallas_out in (y_pallas_cached_xq, y_pallas_accum_k):
            np.testing.assert_allclose(
                np.asarray(pallas_out, np.float32),
                np.asarray(y_xla, np.float32),
                rtol=1e-3,
                atol=1e-2,
            )

        exact_f32 = np.asarray(x, np.float32) @ np.asarray(w, np.float32).T
        cosine = float(
            np.sum(np.asarray(y_xla[:16], np.float32) * exact_f32[:16])
            / (
                np.linalg.norm(np.asarray(y_xla[:16], np.float32))
                * np.linalg.norm(exact_f32[:16])
            )
        )
        assert cosine > 0.999

    def test_qwen3_32b_tp2_buckets_all_hit_zero_pad_tpu7x_tiles(self) -> None:
        from deploy.models import TPU_WORKLOADS, _tpu_buckets
        from resin.backend.tpu.kernels.fp8_matmul import (
            _LARGE_PREFILL_SLACK_BYTES,
            _TPU7X_GATE_UP_TUNED_BLOCKS,
            _estimate_vmem_limit_bytes,
            _get_tuned_gate_up_block_sizes,
            _raw_vmem_estimate_bytes,
            _should_save_x_q,
        )

        tp2_shapes = (
            (4096, 5120),
            (512, 5120),
            (5120, 4096),
            (5120, 5120),
            (5120, 12800),
            (12800, 5120),
            (25600, 5120),
        )
        tp2_shape_set = set(tp2_shapes)

        vmem_cap = 48 * 1024 * 1024
        for (batch, n_out, n_in), tile in _TPU7X_FP8_TUNED_BLOCKS.items():
            assert (
                get_tuned_block_sizes(
                    batch, n_out, n_in, tpu_version=7, quantize_activation=False
                )
                == tile
            )
            for x_dtype, quant in ((jnp.float8_e4m3fn, False), (jnp.bfloat16, True)):
                sel = get_tuned_block_sizes(
                    batch, n_out, n_in, tpu_version=7, quantize_activation=quant
                )
                assert batch % sel.batch_block_size == 0
                assert n_out % sel.out_block_size == 0
                assert n_in % sel.in_block_size == 0
                assert sel.batch_block_size % 8 == 0
                assert sel.out_block_size % 128 == 0
                assert sel.in_block_size % 128 == 0
                assert batch == sel.batch_block_size or sel.batch_block_size % 128 == 0

                nb = batch // sel.batch_block_size
                no = n_out // sel.out_block_size
                ni = n_in // sel.in_block_size
                save_acc = ni > 1

                save_xq = _should_save_x_q(
                    quantize_activation=quant,
                    n_batch=nb,
                    n_out=no,
                    n_in=ni,
                    batch_block_size=sel.batch_block_size,
                    out_block_size=sel.out_block_size,
                    in_block_size=sel.in_block_size,
                    x_dtype=x_dtype,
                    x_q_dtype=jnp.float8_e4m3fn,
                    w_q_dtype=jnp.float8_e4m3fn,
                    out_dtype=jnp.bfloat16,
                    save_acc=save_acc,
                    tpu_version=7,
                    w_buffer_count=sel.w_buffer_count,
                )

                def _vmem(*, headroom: bool) -> int:
                    return _raw_vmem_estimate_bytes(
                        n_batch=nb,
                        n_out=no,
                        n_in=ni,
                        batch_block_size=sel.batch_block_size,
                        out_block_size=sel.out_block_size,
                        in_block_size=sel.in_block_size,
                        x_dtype=x_dtype,
                        x_q_dtype=jnp.float8_e4m3fn,
                        w_q_dtype=jnp.float8_e4m3fn,
                        out_dtype=jnp.bfloat16,
                        save_acc=save_acc,
                        save_x_q=save_xq,
                        w_buffer_count=sel.w_buffer_count,
                        include_headroom=headroom,
                    )

                raw_no_headroom = _vmem(headroom=False)
                assert raw_no_headroom + _LARGE_PREFILL_SLACK_BYTES <= vmem_cap
                if batch >= 2048:
                    vmem_limit = _estimate_vmem_limit_bytes(
                        n_batch=nb,
                        n_out=no,
                        n_in=ni,
                        batch_block_size=sel.batch_block_size,
                        out_block_size=sel.out_block_size,
                        in_block_size=sel.in_block_size,
                        x_dtype=x_dtype,
                        x_q_dtype=jnp.float8_e4m3fn,
                        w_q_dtype=jnp.float8_e4m3fn,
                        out_dtype=jnp.bfloat16,
                        save_acc=save_acc,
                        save_x_q=save_xq,
                        tpu_version=7,
                        w_buffer_count=sel.w_buffer_count,
                    )
                    assert vmem_limit - raw_no_headroom >= _LARGE_PREFILL_SLACK_BYTES
                    if quant and (n_out, n_in) == (5120, 8192):
                        assert not save_xq

                if (n_out, n_in) in tp2_shape_set:
                    if batch <= 320:
                        assert _vmem(headroom=True) <= vmem_cap
                        assert sel.batch_block_size == batch
                        assert sel.out_block_size >= 512
                        assert sel.in_block_size == n_in
                    elif batch >= 2048:
                        if quant and (n_out, n_in) == (5120, 12800):
                            assert sel == Fp8MatmulTileSizes(1024, 2560, 2560)
                            # At out_block=2560 (n_out=2), 2-buffered VMEM + 12.5 MiB
                            # x_q_scratch is 55 MiB > 48 MiB, so save_x_q stays False;
                            # at out_block=1280 (n_out=4, 38.8 MiB + 4 MiB <= 48 MiB), save_x_q is True.
                            assert not save_xq
                            assert _should_save_x_q(
                                quantize_activation=True,
                                n_batch=batch // 1024,
                                n_out=4,
                                n_in=5,
                                batch_block_size=1024,
                                out_block_size=1280,
                                in_block_size=2560,
                                x_dtype=jnp.bfloat16,
                                x_q_dtype=jnp.float8_e4m3fn,
                                w_q_dtype=jnp.float8_e4m3fn,
                                out_dtype=jnp.bfloat16,
                                save_acc=True,
                                tpu_version=7,
                            )

        for (batch, n_ffn, n_in), tile in _TPU7X_GATE_UP_TUNED_BLOCKS.items():
            assert (
                _get_tuned_gate_up_block_sizes(batch, n_ffn, n_in, tpu_version=7)
                == tile
            )
            assert batch % tile.batch_block_size == 0
            assert n_ffn % tile.out_block_size == 0
            assert n_in % tile.in_block_size == 0
            assert tile.batch_block_size % 128 == 0
            assert tile.out_block_size % 128 == 0
            assert tile.in_block_size % 128 == 0
            n_in_tiles = n_in // tile.in_block_size
            x_q_bytes = tile.batch_block_size * tile.in_block_size
            gate_up_w_bytes = 2 * tile.out_block_size * tile.in_block_size
            gated_out_bytes = tile.batch_block_size * tile.out_block_size * 2
            acc_scratch_bytes = (
                2 * tile.batch_block_size * tile.out_block_size * 4
                if n_in_tiles > 1
                else 0
            )
            gate_up_vmem_bytes = (
                2 * (x_q_bytes + gate_up_w_bytes + gated_out_bytes) + acc_scratch_bytes
            )
            assert gate_up_vmem_bytes <= vmem_cap
        assert _get_tuned_gate_up_block_sizes(
            320, 12800, 5120, tpu_version=7
        ) == Fp8MatmulTileSizes(320, 1280, 5120)
        assert _get_tuned_gate_up_block_sizes(
            4096, 25600, 5120, tpu_version=7
        ) == Fp8MatmulTileSizes(512, 1280, 5120)

        buckets = {
            bucket.tokens
            for workload in TPU_WORKLOADS.values()
            for bucket in _tpu_buckets(workload, workload.max_batch_tokens).shapes()
        }
        for batch in buckets:
            for n_out, n_in in tp2_shapes:
                assert (batch, n_out, n_in) in _TPU7X_FP8_TUNED_BLOCKS

        for bs in (8, 16, 32, 64, 96, 128, 160, 192):
            assert get_tuned_block_sizes(
                bs, 5120, 4096, tpu_version=7, quantize_activation=False
            ) == Fp8MatmulTileSizes(bs, 640, 4096, 3)
        for bs in (224, 256, 288, 320):
            assert get_tuned_block_sizes(
                bs, 5120, 4096, tpu_version=7, quantize_activation=False
            ) == Fp8MatmulTileSizes(bs, 1280, 4096, 3)
        for bs in (288, 320):
            assert get_tuned_block_sizes(
                bs, 5120, 5120, tpu_version=7, quantize_activation=False
            ) == Fp8MatmulTileSizes(bs, 1280, 5120)
            assert get_tuned_block_sizes(
                bs, 5120, 12800, tpu_version=7, quantize_activation=False
            ) == Fp8MatmulTileSizes(bs, 512, 12800, 3)
            assert get_tuned_block_sizes(
                bs, 5120, 5120, tpu_version=7, quantize_activation=True
            ) == Fp8MatmulTileSizes(bs, 1280, 5120)
            assert get_tuned_block_sizes(
                bs, 5120, 12800, tpu_version=7, quantize_activation=True
            ) == Fp8MatmulTileSizes(bs, 512, 12800)
        assert get_tuned_block_sizes(
            512, 12800, 5120, tpu_version=7, quantize_activation=True
        ) == Fp8MatmulTileSizes(256, 2560, 5120)

        # Confirm tp=1 decode entries remain at their base values.
        assert get_tuned_block_sizes(
            320, 5120, 8192, tpu_version=7
        ) == Fp8MatmulTileSizes(320, 2560, 4096)
        assert get_tuned_block_sizes(
            320, 5120, 25600, tpu_version=7
        ) == Fp8MatmulTileSizes(320, 2560, 2560)
        assert get_tuned_block_sizes(
            320, 10240, 5120, tpu_version=7
        ) == Fp8MatmulTileSizes(320, 2560, 2560)
        assert get_tuned_block_sizes(
            320, 51200, 5120, tpu_version=7
        ) == Fp8MatmulTileSizes(320, 2560, 2560)

    def test_raw_vmem_estimate_counts_scratch_buffers_once(self) -> None:
        from resin.backend.tpu.kernels.fp8_matmul import _raw_vmem_estimate_bytes

        n_in = 5
        batch_block, out_block, in_block = 1024, 1280, 2560

        def _estimate(*, save_acc: bool, save_x_q: bool) -> int:
            return _raw_vmem_estimate_bytes(
                n_batch=4,
                n_out=4,
                n_in=n_in,
                batch_block_size=batch_block,
                out_block_size=out_block,
                in_block_size=in_block,
                x_dtype=jnp.bfloat16,
                x_q_dtype=jnp.float8_e4m3fn,
                w_q_dtype=jnp.float8_e4m3fn,
                out_dtype=jnp.bfloat16,
                save_acc=save_acc,
                save_x_q=save_x_q,
                include_headroom=False,
            )

        base = _estimate(save_acc=False, save_x_q=False)
        with_scratch = _estimate(save_acc=True, save_x_q=True)
        acc_scratch_bytes = batch_block * out_block * 4
        x_q_scratch_bytes = n_in * batch_block * in_block
        assert with_scratch - base == acc_scratch_bytes + x_q_scratch_bytes

    def test_large_prefill_vmem_limit_stays_under_32mib(self) -> None:
        from resin.backend.tpu.kernels.fp8_matmul import _estimate_vmem_limit_bytes

        for in_block in (4096, 5120):
            vmem = _estimate_vmem_limit_bytes(
                n_batch=4,
                n_out=4,
                n_in=1,
                batch_block_size=1024,
                out_block_size=1280,
                in_block_size=in_block,
                x_dtype=jnp.float8_e4m3fn,
                x_q_dtype=jnp.float8_e4m3fn,
                w_q_dtype=jnp.float8_e4m3fn,
                out_dtype=jnp.bfloat16,
                save_acc=False,
                save_x_q=False,
                tpu_version=7,
            )
            assert vmem < 32 * 1024 * 1024

    def test_pallas_save_x_q_16mib_cap_and_vmem_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        import jax.experimental.pallas as pl

        fp8_mod = importlib.import_module("resin.backend.tpu.kernels.fp8_matmul")
        captured_scratch: list[tuple[object, ...]] = []
        orig_pallas_call = pl.pallas_call

        def _spy_pallas_call(*args: Any, **kwargs: Any) -> Any:
            grid_spec = kwargs.get("grid_spec")
            if grid_spec is not None and hasattr(grid_spec, "scratch_shapes"):
                captured_scratch.append(tuple(grid_spec.scratch_shapes))
            return orig_pallas_call(*args, **kwargs)

        monkeypatch.setattr(fp8_mod, "get_tpu_version", lambda: 7)
        monkeypatch.setattr(pl, "pallas_call", _spy_pallas_call)

        rng = np.random.default_rng(4096)
        x = jnp.asarray(rng.standard_normal((2048, 10240)) * 0.2, jnp.bfloat16)
        weight = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 10240)) * 0.05, jnp.bfloat16)
        )
        tile_sizes = Fp8MatmulTileSizes(1024, 128, 2560)

        # Below 2048 padded tokens the 4 MiB cap still applies: 5 MiB stays uncached.
        fp8_mod.pallas_fp8_matmul(
            x[:512],
            weight.qvalue,
            weight.scale,
            tile_sizes=Fp8MatmulTileSizes(512, 128, 2560),
            interpret=True,
        )
        assert len(captured_scratch) == 1
        assert captured_scratch[0][1] is None
        captured_scratch.clear()

        # 10 MiB x_q_scratch at 2048 padded tokens (> 4 MiB old cap, <= 16 MiB cap): allocates scratch and matches XLA.
        y_cached = fp8_mod.pallas_fp8_matmul(
            x,
            weight.qvalue,
            weight.scale,
            tile_sizes=tile_sizes,
            interpret=True,
        )
        y_xla = xla_fp8_matmul(x, weight.qvalue, weight.scale)
        assert len(captured_scratch) == 1
        assert captured_scratch[0][1] is not None
        np.testing.assert_allclose(
            np.asarray(y_cached, np.float32),
            np.asarray(y_xla, np.float32),
            rtol=1e-2,
            atol=3e-2,
        )

        # Lowering VMEM cap below 2-buffered VMEM + x_q_scratch disables save_x_q cleanly.
        monkeypatch.setattr(
            fp8_mod, "_DEVICE_VMEM_LIMIT", {6: 96 * 1024 * 1024, 7: 8 * 1024 * 1024}
        )
        captured_scratch.clear()
        y_uncached = fp8_mod.pallas_fp8_matmul(
            x,
            weight.qvalue,
            weight.scale,
            tile_sizes=tile_sizes,
            interpret=True,
        )
        assert len(captured_scratch) == 1
        assert captured_scratch[0][1] is None
        np.testing.assert_allclose(
            np.asarray(y_uncached, np.float32),
            np.asarray(y_cached, np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.parametrize("tokens", [2048, 4096])
    def test_prefill_down_and_gate_up_tpu7x_mlir_export(
        self, monkeypatch: pytest.MonkeyPatch, tokens: int
    ) -> None:
        import importlib

        from jax import export
        from jax._src import tpu_info
        from jax._src.pallas.mosaic import lowering as mosaic_lowering
        from jax._src.pallas.mosaic import pallas_call_registration

        fp8_mod = importlib.import_module("resin.backend.tpu.kernels.fp8_matmul")
        v7x_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion("7x"), 1)
        monkeypatch.setattr(fp8_mod, "get_tpu_version", lambda: 7)
        monkeypatch.setattr(tpu_info, "get_tpu_info", lambda: v7x_info)
        monkeypatch.setattr(mosaic_lowering.tpu_info, "get_tpu_info", lambda: v7x_info)
        monkeypatch.setattr(
            pallas_call_registration.tpu_info, "get_tpu_info", lambda: v7x_info
        )

        down_exported = export.export(
            jax.jit(fp8_mod.pallas_fp8_matmul),
            platforms=["tpu"],
        )(
            jax.ShapeDtypeStruct((tokens, 12800), jnp.bfloat16),
            jax.ShapeDtypeStruct((5120, 12800), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((5120,), jnp.float32),
        )
        down_mlir = down_exported.mlir_module()
        assert "tpu_custom_call" in down_mlir
        assert "fp8_w8a8_matmul_1024_2560_2560" in down_mlir

        gate_up_exported = export.export(
            jax.jit(fp8_mod.pallas_fp8_gate_up_swiglu),
            platforms=["tpu"],
        )(
            jax.ShapeDtypeStruct((tokens, 5120), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, tokens), jnp.float32),
            jax.ShapeDtypeStruct((25600, 5120), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((25600,), jnp.float32),
        )
        gate_up_mlir = gate_up_exported.mlir_module()
        assert "tpu_custom_call" in gate_up_mlir
        assert "fp8_gate_up_swiglu_1024_1280_5120" in gate_up_mlir

    @pytest.mark.parametrize(("batch", "live_rows"), [(8, 1), (8, 4), (16, 9)], ids=str)
    @pytest.mark.parametrize(
        ("n_out", "n_in", "out_block", "in_block"),
        [(512, 5120, 512, 2560), (256, 512, 128, 512)],
        ids=["accumulate-k", "cache-x-q"],
    )
    def test_small_decode_buckets_8_and_16_match_xla_in_pallas_interpret(
        self,
        batch: int,
        live_rows: int,
        n_out: int,
        n_in: int,
        out_block: int,
        in_block: int,
    ) -> None:
        rng = np.random.default_rng(batch * 1000 + live_rows)
        x_np = rng.standard_normal((batch, n_in)).astype(np.float32)
        x_np[live_rows:] = 0.0
        w_np = rng.standard_normal((n_out, n_in)).astype(np.float32) * 0.1
        x = jnp.asarray(x_np, jnp.bfloat16)
        fw = quantize_fp8_weight(jnp.asarray(w_np, jnp.bfloat16))

        expected = np.asarray(xla_fp8_matmul(x, fw.qvalue, fw.scale), np.float32)
        actual = np.asarray(
            fp8_matmul(
                x,
                fw.qvalue,
                fw.scale,
                backend="pallas",
                tile_sizes=Fp8MatmulTileSizes(batch, out_block, in_block),
                interpret=True,
            ),
            np.float32,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-2)
        np.testing.assert_allclose(actual[live_rows:], 0.0, atol=0.0)

    def test_tp2_column_and_row_parallel_match_single_device_qwix(self) -> None:
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("needs 2 host devices")
        mesh = Mesh(np.array(devices[:2]), ("tp",))

        rng = np.random.default_rng(202)
        x = jnp.asarray(rng.standard_normal((16, 256)), jnp.bfloat16)
        # Put asymmetric peak magnitudes in the second half of K so row-parallel
        # shards diverge unless lax.pmax synchronizes x_absmax across tp.
        x = x.at[:, 128:].multiply(10.0)
        w = jnp.asarray(rng.standard_normal((256, 256)), jnp.bfloat16)
        fw = quantize_fp8_weight(w)
        single = xla_fp8_matmul(x, fw.qvalue, fw.scale)

        from resin.backend.tpu.kernels.fp8_matmul import quantize_fp8_act

        xq_full, xs_full = quantize_fp8_act(x, dtype=fw.qvalue.dtype)
        bf16_two_shard_oracle = (
            fp8_matmul_prequantized(
                xq_full[:, :128], xs_full, fw.qvalue[:, :128], fw.scale
            )
            + fp8_matmul_prequantized(
                xq_full[:, 128:], xs_full, fw.qvalue[:, 128:], fw.scale
            )
        ).astype(jnp.bfloat16)

        w_xavier = jnp.asarray(
            rng.standard_normal((256, 256)).astype(np.float32) * 0.02, jnp.bfloat16
        )
        fw_xavier = quantize_fp8_weight(w_xavier)
        single_xavier = xla_fp8_matmul(x, fw_xavier.qvalue, fw_xavier.scale)

        cases: tuple[tuple[Literal["xla", "pallas"], bool], ...] = (
            ("xla", False),
            ("pallas", True),
        )
        for backend, interpret in cases:
            col = sharded_fp8_matmul(
                jax.device_put(x, jax.sharding.NamedSharding(mesh, P(None, None))),
                jax.device_put(
                    fw.qvalue, jax.sharding.NamedSharding(mesh, P("tp", None))
                ),
                jax.device_put(fw.scale, jax.sharding.NamedSharding(mesh, P("tp"))),
                P("tp", None),
                mesh=mesh,
                backend=backend,
                interpret=interpret,
            )
            row_f32 = sharded_fp8_matmul(
                jax.device_put(x, jax.sharding.NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(
                    fw.qvalue, jax.sharding.NamedSharding(mesh, P(None, "tp"))
                ),
                jax.device_put(fw.scale, jax.sharding.NamedSharding(mesh, P())),
                P(None, "tp"),
                mesh=mesh,
                reduce_dtype=jnp.float32,
                backend=backend,
                interpret=interpret,
            )
            row_bf16 = sharded_fp8_matmul(
                jax.device_put(x, jax.sharding.NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(
                    fw.qvalue, jax.sharding.NamedSharding(mesh, P(None, "tp"))
                ),
                jax.device_put(fw.scale, jax.sharding.NamedSharding(mesh, P())),
                P(None, "tp"),
                mesh=mesh,
                backend=backend,
                interpret=interpret,
            )
            row_xavier = sharded_fp8_matmul(
                jax.device_put(x, jax.sharding.NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(
                    fw_xavier.qvalue, jax.sharding.NamedSharding(mesh, P(None, "tp"))
                ),
                jax.device_put(fw_xavier.scale, jax.sharding.NamedSharding(mesh, P())),
                P(None, "tp"),
                mesh=mesh,
                backend=backend,
                interpret=interpret,
            )
            np.testing.assert_allclose(
                np.asarray(col, np.float32),
                np.asarray(single, np.float32),
                rtol=1e-2,
                atol=2e-2,
            )
            np.testing.assert_allclose(
                np.asarray(row_f32, np.float32),
                np.asarray(single, np.float32),
                rtol=1e-2,
                atol=2e-2,
            )
            np.testing.assert_allclose(
                np.asarray(row_bf16, np.float32),
                np.asarray(bf16_two_shard_oracle, np.float32),
                rtol=1e-2,
                atol=2e-2,
            )
            np.testing.assert_allclose(
                np.asarray(row_xavier, np.float32),
                np.asarray(single_xavier, np.float32),
                rtol=1e-2,
                atol=2e-2,
            )


class TestNormAndRope:
    def test_rms_norm_fused_add_and_fp8_quant_preserve_rounding(self) -> None:
        rng = np.random.default_rng(303)
        x = jnp.asarray(rng.standard_normal((12, 128)), jnp.bfloat16)
        res = jnp.asarray(rng.standard_normal((12, 128)), jnp.bfloat16)
        weight = jnp.asarray(rng.uniform(0.5, 1.5, size=(128,)), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((64, 128)), jnp.bfloat16)
        fw = quantize_fp8_weight(w)

        normed, next_res = fused_add_rms_norm(x, res, weight, eps=1e-6)
        expected_res = (x + res).astype(jnp.bfloat16)
        ref_f32 = np.asarray(expected_res, np.float32)
        ref_norm = (
            ref_f32
            / np.sqrt(np.mean(ref_f32**2, axis=-1, keepdims=True) + 1e-6)
            * np.asarray(weight, np.float32)
        )
        np.testing.assert_allclose(
            np.asarray(normed, np.float32), ref_norm, rtol=1e-2, atol=1e-2
        )

        # Fused Pallas rms_norm_fp8_quant + fp8_matmul_prequantized matches unfused exactly.
        x_q, x_scale = rms_norm_fp8_quant(next_res, weight, eps=1e-6, interpret=True)
        fused_out = fp8_matmul_prequantized(x_q, x_scale, fw.qvalue, fw.scale)
        unfused_out = xla_fp8_matmul(
            rms_norm(next_res, weight, eps=1e-6), fw.qvalue, fw.scale
        )
        np.testing.assert_array_equal(np.asarray(fused_out), np.asarray(unfused_out))

    def test_apply_rope_cache_matches_on_the_fly_without_recompiling_on_positions(
        self,
    ) -> None:
        rng = np.random.default_rng(404)
        x = jnp.asarray(rng.standard_normal((8, 4, 128)), jnp.bfloat16)
        cache = build_rope_cache(512, 128, rope_theta=1000000.0)

        counter = CompilationCounter()
        try:
            roped_fn = jax.jit(
                lambda inp, pos: apply_rope(
                    inp, pos, 128, rope_theta=1000000.0, cos_sin_cache=cache
                )
            )
            pos1 = jnp.array([0, 1, 2, 3, 10, 11, 12, 0], jnp.int32)
            pos2 = jnp.array([100, 205, 310, 411, 5, 6, 7, 8], jnp.int32)

            out1 = roped_fn(x, pos1)
            out1.block_until_ready()
            warm = counter.count

            out2 = roped_fn(x, pos2)
            out2.block_until_ready()
            assert counter.count == warm

            direct = apply_rope(x, pos2, 128, rope_theta=1000000.0)
            np.testing.assert_allclose(
                np.asarray(out2, np.float32),
                np.asarray(direct, np.float32),
                rtol=1e-2,
                atol=1e-2,
            )
        finally:
            counter.close()

    @pytest.mark.parametrize("batch", [1, 7, 8, 33, 320])
    @pytest.mark.parametrize("k_dim", [256, 5120])
    def test_pallas_rms_norm_fp8_quant_1d_weight(self, batch: int, k_dim: int) -> None:
        from resin.backend.tpu.kernels.norm_rope import (
            pallas_qkv_norm_rope_pack,
            pallas_rms_norm_fp8_quant,
        )

        rng = np.random.default_rng(1100 + batch * 17 + k_dim)
        x = jnp.asarray(rng.standard_normal((batch, k_dim)) * 0.4, jnp.bfloat16)
        res = jnp.asarray(rng.standard_normal((batch, k_dim)) * 0.4, jnp.bfloat16)
        w_1d = jnp.asarray(rng.uniform(0.5, 1.5, size=(k_dim,)), jnp.bfloat16)

        jaxpr_plain = jax.make_jaxpr(
            lambda xi, wi: pallas_rms_norm_fp8_quant(
                xi, wi, eps=1e-6, transposed_scale=True, interpret=True
            )
        )(x, w_1d)
        assert not any(
            e.primitive.name == "reshape" and e.invars[0].aval.shape == (k_dim,)
            for e in jaxpr_plain.jaxpr.eqns
        )

        jaxpr_fused = jax.make_jaxpr(
            lambda xi, ri, wi: pallas_rms_norm_fp8_quant(
                xi, wi, eps=1e-6, residual=ri, transposed_scale=True, interpret=True
            )
        )(x, res, w_1d)
        assert not any(
            e.primitive.name == "reshape" and e.invars[0].aval.shape == (k_dim,)
            for e in jaxpr_fused.jaxpr.eqns
        )

        xq, xs = pallas_rms_norm_fp8_quant(
            x, w_1d, eps=1e-6, transposed_scale=True, interpret=True
        )
        xq_2d, xs_2d = pallas_rms_norm_fp8_quant(
            x, w_1d.reshape(1, k_dim), eps=1e-6, transposed_scale=True, interpret=True
        )
        np.testing.assert_array_equal(np.asarray(xq), np.asarray(xq_2d))
        np.testing.assert_array_equal(np.asarray(xs), np.asarray(xs_2d))

        fxq, fxs, fnr = pallas_rms_norm_fp8_quant(
            x, w_1d, eps=1e-6, residual=res, transposed_scale=True, interpret=True
        )
        fxq_2d, fxs_2d, fnr_2d = pallas_rms_norm_fp8_quant(
            x,
            w_1d.reshape(1, k_dim),
            eps=1e-6,
            residual=res,
            transposed_scale=True,
            interpret=True,
        )
        np.testing.assert_array_equal(np.asarray(fnr), np.asarray(fnr_2d))
        np.testing.assert_array_equal(np.asarray(fxq), np.asarray(fxq_2d))
        np.testing.assert_array_equal(np.asarray(fxs), np.asarray(fxs_2d))

        if k_dim == 5120 and batch in (1, 8):
            cos_sin = build_rope_cache(64, 128)
            pos = jnp.arange(batch, dtype=jnp.int32)
            qw = jnp.ones((128,), dtype=jnp.bfloat16)
            kw = jnp.ones((128,), dtype=jnp.bfloat16)
            jaxpr_qkv = jax.make_jaxpr(
                lambda qkv_i, qwi, kwi: pallas_qkv_norm_rope_pack(
                    qkv_i,
                    pos,
                    cos_sin,
                    qwi,
                    kwi,
                    jnp.float8_e4m3fn,
                    num_q_heads=32,
                    num_kv_heads=4,
                    interpret=True,
                )
            )(x, qw, kw)
            assert not any(
                e.primitive.name == "reshape" and e.invars[0].aval.shape == (128,)
                for e in jaxpr_qkv.jaxpr.eqns
            )


class TestPrefillKernels:
    def test_fused_add_rms_norm_fp8_quant_matches_unfused_bit_for_bit(self) -> None:
        from resin.backend.tpu.kernels.norm_rope import fused_add_rms_norm_fp8_quant

        rng = np.random.default_rng(505)
        x = jnp.asarray(rng.standard_normal((20, 256)), jnp.bfloat16)
        res = jnp.asarray(rng.standard_normal((20, 256)), jnp.bfloat16)
        x = x.at[16:].set(0.0)
        res = res.at[16:].set(0.0)
        weight = jnp.asarray(rng.uniform(0.5, 1.5, size=(256,)), jnp.bfloat16)

        xq, xs, next_res = fused_add_rms_norm_fp8_quant(
            x, res, weight, eps=1e-6, interpret=True
        )
        expected_res = (x + res).astype(jnp.bfloat16)
        exp_q, exp_s = rms_norm_fp8_quant(
            expected_res, weight, eps=1e-6, interpret=True
        )
        np.testing.assert_array_equal(np.asarray(next_res), np.asarray(expected_res))
        np.testing.assert_array_equal(np.asarray(xq), np.asarray(exp_q))
        np.testing.assert_array_equal(np.asarray(xs), np.asarray(exp_s))

        # CPU pure-JAX fallback also matches on non-128 widths.
        x_small = jnp.asarray(rng.standard_normal((12, 32)), jnp.bfloat16)
        r_small = jnp.asarray(rng.standard_normal((12, 32)), jnp.bfloat16)
        w_small = jnp.asarray(rng.uniform(0.5, 1.5, size=(32,)), jnp.bfloat16)
        xq_s, xs_s, nr_s = fused_add_rms_norm_fp8_quant(x_small, r_small, w_small)
        exp_nr_s = (x_small + r_small).astype(jnp.bfloat16)
        exp_q_s, exp_s_s = rms_norm_fp8_quant(exp_nr_s, w_small)
        np.testing.assert_array_equal(np.asarray(nr_s), np.asarray(exp_nr_s))
        np.testing.assert_array_equal(np.asarray(xq_s), np.asarray(exp_q_s))
        np.testing.assert_array_equal(np.asarray(xs_s), np.asarray(exp_s_s))

    @pytest.mark.parametrize("kv_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
    @pytest.mark.parametrize("pack_mode", ["reshape", "bitcast_u32"])
    def test_pallas_qk_and_qkv_norm_rope_pack_match_prepare_inputs(
        self, kv_dtype: jnp.dtype, pack_mode: Literal["reshape", "bitcast_u32"]
    ) -> None:
        from resin.backend.tpu.kernels import (
            pallas_qk_norm_rope_pack,
            pallas_qkv_norm_rope_pack,
            prepare_inputs,
        )

        rng = np.random.default_rng(606)
        tokens, q_heads, kv_heads, head_dim = 20, 8, 2, 128
        q_np = rng.standard_normal((tokens, q_heads, head_dim)).astype(np.float32)
        k_np = rng.standard_normal((tokens, kv_heads, head_dim)).astype(np.float32)
        v_np = rng.standard_normal((tokens, kv_heads, head_dim)).astype(np.float32)
        # Zero-padded tail tokens at indices 15..19.
        q_np[15:] = 0.0
        k_np[15:] = 0.0
        v_np[15:] = 0.0

        q = jnp.asarray(q_np, jnp.bfloat16)
        k = jnp.asarray(k_np, jnp.bfloat16)
        v = jnp.asarray(v_np, jnp.bfloat16)
        positions = jnp.array(
            [0, 3, 17, 128, 511, 42, 43, 44, 1, 2, 99, 100, 101, 102, 7] + [0] * 5,
            dtype=jnp.int32,
        )
        rope_cache = build_rope_cache(1024, head_dim, rope_theta=1000000.0)
        assert rope_cache[0].shape == (1024, 128)
        assert rope_cache[1].shape == (1024, 128)

        q_norm_w = jnp.asarray(rng.uniform(0.5, 1.5, size=(head_dim,)), jnp.bfloat16)
        k_norm_w = jnp.asarray(rng.uniform(0.5, 1.5, size=(head_dim,)), jnp.bfloat16)
        k_scale = 0.5 if kv_dtype == jnp.float8_e4m3fn else None
        v_scale = 0.25 if kv_dtype == jnp.float8_e4m3fn else None

        qp, kvm = pallas_qk_norm_rope_pack(
            q,
            k,
            v,
            positions,
            rope_cache,
            q_norm_w,
            k_norm_w,
            kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            eps=1e-6,
            pack_mode=pack_mode,
            interpret=True,
        )
        qkv = jnp.concatenate(
            [q.reshape(tokens, -1), k.reshape(tokens, -1), v.reshape(tokens, -1)],
            axis=-1,
        )
        qp_fused, kvm_fused = pallas_qkv_norm_rope_pack(
            qkv,
            positions,
            rope_cache,
            q_norm_w,
            k_norm_w,
            kv_dtype,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            k_scale=k_scale,
            v_scale=v_scale,
            eps=1e-6,
            pack_mode=pack_mode,
            interpret=True,
        )

        q_ref = apply_rope(
            rms_norm(q, q_norm_w, 1e-6), positions, head_dim, cos_sin_cache=rope_cache
        )
        k_ref = quantize_kv(
            apply_rope(
                rms_norm(k, k_norm_w, 1e-6),
                positions,
                head_dim,
                cos_sin_cache=rope_cache,
            ),
            kv_dtype,
            k_scale,
        )
        v_ref = quantize_kv(v, kv_dtype, v_scale)
        exp_qp, exp_kvm = prepare_inputs(q_ref, k_ref, v_ref)

        assert not np.any(np.isnan(np.asarray(qp, np.float32)))
        assert not np.any(np.isnan(np.asarray(kvm, np.float32)))
        np.testing.assert_array_equal(np.asarray(qp), np.asarray(qp_fused))
        np.testing.assert_array_equal(np.asarray(kvm), np.asarray(kvm_fused))
        np.testing.assert_allclose(
            np.asarray(qp, np.float32),
            np.asarray(exp_qp, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(kvm, np.float32),
            np.asarray(exp_kvm, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )

        # Qwen3-32B tp=2 head geometry (32 Q, 4 KV) at tb=512 with non-512-multiple tail.
        t_big, q_big, kv_big = 525, 32, 4
        q_big_np = rng.standard_normal((t_big, q_big, head_dim)).astype(np.float32)
        k_big_np = rng.standard_normal((t_big, kv_big, head_dim)).astype(np.float32)
        v_big_np = rng.standard_normal((t_big, kv_big, head_dim)).astype(np.float32)
        q_big_np[516:] = 0.0
        k_big_np[516:] = 0.0
        v_big_np[516:] = 0.0
        q_b = jnp.asarray(q_big_np, jnp.bfloat16)
        k_b = jnp.asarray(k_big_np, jnp.bfloat16)
        v_b = jnp.asarray(v_big_np, jnp.bfloat16)
        pos_b = jnp.arange(t_big, dtype=jnp.int32)
        qkv_b = jnp.concatenate(
            [q_b.reshape(t_big, -1), k_b.reshape(t_big, -1), v_b.reshape(t_big, -1)],
            axis=-1,
        )
        qp_512, kvm_512 = pallas_qkv_norm_rope_pack(
            qkv_b,
            pos_b,
            rope_cache,
            q_norm_w,
            k_norm_w,
            kv_dtype,
            num_q_heads=q_big,
            num_kv_heads=kv_big,
            k_scale=k_scale,
            v_scale=v_scale,
            eps=1e-6,
            pack_mode=pack_mode,
            interpret=True,
        )
        qp_512_qk, kvm_512_qk = pallas_qk_norm_rope_pack(
            q_b,
            k_b,
            v_b,
            pos_b,
            rope_cache,
            q_norm_w,
            k_norm_w,
            kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            eps=1e-6,
            pack_mode=pack_mode,
            interpret=True,
        )
        np.testing.assert_array_equal(np.asarray(qp_512), np.asarray(qp_512_qk))
        np.testing.assert_array_equal(np.asarray(kvm_512), np.asarray(kvm_512_qk))
        q_ref_b = apply_rope(
            rms_norm(q_b, q_norm_w, 1e-6), pos_b, head_dim, cos_sin_cache=rope_cache
        )
        k_ref_b = quantize_kv(
            apply_rope(
                rms_norm(k_b, k_norm_w, 1e-6),
                pos_b,
                head_dim,
                cos_sin_cache=rope_cache,
            ),
            kv_dtype,
            k_scale,
        )
        v_ref_b = quantize_kv(v_b, kv_dtype, v_scale)
        exp_qp_b, exp_kvm_b = prepare_inputs(q_ref_b, k_ref_b, v_ref_b)
        np.testing.assert_allclose(
            np.asarray(qp_512, np.float32),
            np.asarray(exp_qp_b, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
        kv_rtol = 0.125 if kv_dtype == jnp.float8_e4m3fn else 1e-2
        np.testing.assert_allclose(
            np.asarray(kvm_512, np.float32),
            np.asarray(exp_kvm_b, np.float32),
            rtol=kv_rtol,
            atol=2e-2,
        )
        assert (
            float(
                np.mean(
                    np.abs(
                        np.asarray(kvm_512, np.float32)
                        - np.asarray(exp_kvm_b, np.float32)
                    )
                )
            )
            < 1e-3
        )

    def test_prequantized_pallas_fp8_matmul_gate_up_swiglu_and_sequence_parallel(
        self,
    ) -> None:
        from resin.backend.tpu.kernels import (
            pallas_fp8_gate_up_swiglu,
            quantize_fp8_act,
            sharded_fp8_gate_up_swiglu,
            sharded_fp8_matmul_all_gather,
            sharded_fp8_matmul_reduce_scatter,
        )

        rng = np.random.default_rng(707)
        x_np = rng.standard_normal((32, 256)).astype(np.float32) * 0.5
        x_np[24:] = 0.0
        x = jnp.asarray(x_np, jnp.bfloat16)
        xq, xs = quantize_fp8_act(x)

        w_col = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 256)) * 0.02, jnp.bfloat16)
        )
        y_pal = fp8_matmul(
            xq,
            w_col.qvalue,
            w_col.scale,
            x_scale=xs,
            backend="pallas",
            tile_sizes=Fp8MatmulTileSizes(32, 128, 128),
            interpret=True,
        )
        y_ref = xla_fp8_matmul(x, w_col.qvalue, w_col.scale)
        assert not np.any(np.isnan(np.asarray(y_pal, np.float32)))
        np.testing.assert_allclose(np.asarray(y_pal, np.float32)[24:], 0.0, atol=0.0)
        np.testing.assert_allclose(
            np.asarray(y_pal, np.float32),
            np.asarray(y_ref, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )

        w_gate = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 256)) * 0.02, jnp.bfloat16)
        )
        w_up_np = rng.standard_normal((256, 256)).astype(np.float32) * 0.02
        # Skew the second TP shard's FFN magnitudes by 10x so dropping lax.pmax on
        # local_abs_max leaves shard 0's scale 10x too small and clips shard 1.
        w_up_np[128:] *= 10.0
        w_up = quantize_fp8_weight(jnp.asarray(w_up_np, jnp.bfloat16))
        w_gu_q = jnp.concatenate([w_gate.qvalue, w_up.qvalue], axis=0)
        w_gu_s = jnp.concatenate([w_gate.scale, w_up.scale], axis=0)
        gated_pal, local_abs = pallas_fp8_gate_up_swiglu(
            xq,
            xs,
            w_gu_q,
            w_gu_s,
            tile_sizes=Fp8MatmulTileSizes(32, 128, 128),
            interpret=True,
        )
        ref_gated = (
            jax.nn.silu(xla_fp8_matmul(x, w_gate.qvalue, w_gate.scale))
            * xla_fp8_matmul(x, w_up.qvalue, w_up.scale)
        ).astype(jnp.bfloat16)
        assert not np.any(np.isnan(np.asarray(gated_pal, np.float32)))
        np.testing.assert_allclose(
            np.asarray(gated_pal, np.float32)[24:], 0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(gated_pal, np.float32),
            np.asarray(ref_gated, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )

        if len(jax.devices()) >= 2:
            mesh = Mesh(np.array(jax.devices()[:2]), ("tp",))
            xq_sh = jax.device_put(xq, NamedSharding(mesh, P("tp", None)))
            xs_sh = jax.device_put(xs, NamedSharding(mesh, P("tp", None)))
            w_col_q = jax.device_put(w_col.qvalue, NamedSharding(mesh, P("tp", None)))
            w_col_s = jax.device_put(w_col.scale, NamedSharding(mesh, P("tp")))
            w_gu_sh_q = jax.device_put(
                jnp.concatenate(
                    [
                        w_gate.qvalue[:128],
                        w_up.qvalue[:128],
                        w_gate.qvalue[128:],
                        w_up.qvalue[128:],
                    ],
                    axis=0,
                ),
                NamedSharding(mesh, P("tp", None)),
            )
            w_gu_sh_s = jax.device_put(
                jnp.concatenate(
                    [
                        w_gate.scale[:128],
                        w_up.scale[:128],
                        w_gate.scale[128:],
                        w_up.scale[128:],
                    ],
                    axis=0,
                ),
                NamedSharding(mesh, P("tp")),
            )
            w_down = quantize_fp8_weight(
                jnp.asarray(rng.standard_normal((256, 256)) * 0.02, jnp.bfloat16)
            )
            w_down_q = jax.device_put(w_down.qvalue, NamedSharding(mesh, P(None, "tp")))
            w_down_s = jax.device_put(w_down.scale, NamedSharding(mesh, P()))
            ref_down = xla_fp8_matmul(ref_gated, w_down.qvalue, w_down.scale)

            for backend, interp in (
                ("xla", False),
                ("pallas", True),
            ):
                ag_out = sharded_fp8_matmul_all_gather(
                    xq_sh,
                    xs_sh,
                    w_col_q,
                    w_col_s,
                    mesh=mesh,
                    backend=backend,
                    interpret=interp,
                )
                np.testing.assert_allclose(
                    np.asarray(ag_out, np.float32),
                    np.asarray(y_ref, np.float32),
                    rtol=2e-2,
                    atol=3e-2,
                )
                g_sh, abs_sh = sharded_fp8_gate_up_swiglu(
                    xq_sh,
                    xs_sh,
                    w_gu_sh_q,
                    w_gu_sh_s,
                    mesh=mesh,
                    backend=backend,
                    interpret=interp,
                )
                if backend == "pallas":
                    assert abs_sh is not None
                    full_abs = np.max(np.abs(np.asarray(g_sh, np.float32)), axis=-1)
                    shard_arrays = [
                        np.asarray(shard.data, np.float32)
                        for shard in abs_sh.addressable_shards
                    ]
                    np.testing.assert_array_equal(shard_arrays[0], shard_arrays[1])
                    for s_arr in shard_arrays:
                        np.testing.assert_allclose(
                            s_arr, full_abs, rtol=1e-2, atol=1e-2
                        )
                rs_out = sharded_fp8_matmul_reduce_scatter(
                    g_sh,
                    w_down_q,
                    w_down_s,
                    mesh=mesh,
                    x_abs_max=abs_sh,
                    backend=backend,
                    interpret=interp,
                )
                assert isinstance(rs_out.sharding, NamedSharding)
                assert rs_out.sharding.spec == P("tp", None)
                np.testing.assert_allclose(
                    np.asarray(rs_out, np.float32),
                    np.asarray(ref_down, np.float32),
                    rtol=2e-2,
                    atol=3e-2,
                )

    def test_rpa_prepared_seam_lax_cond_empty_cases_and_mixed_tuned_blocks(
        self,
    ) -> None:
        from resin.backend.tpu.attention import (
            KV_MERGED_PARTITION,
            Q_PACKED_PARTITION,
        )
        from resin.backend.tpu.kernels import (
            RpaBlockSizes,
            get_rpa_block_sizes,
            get_vmem_estimate_bytes,
            pallas_qk_norm_rope_pack,
        )
        from resin.backend.tpu.kernels.ragged_paged_attention import (
            _TPU7X_RPA_MIXED_TUNED_BLOCKS,
            static_validate_inputs,
        )

        # 1. Validate every _TPU7X_RPA_MIXED_TUNED_BLOCKS entry matches the swept
        # Qwen3-32B tp=2 (32, 4, 128), page_size=256 domain and <= 28.8 MiB (0.45 * 64 MiB) cap.
        tpu7x_vmem_bytes = 64 * 1024 * 1024
        vmem_cap = int(0.45 * tpu7x_vmem_bytes)
        assert len(_TPU7X_RPA_MIXED_TUNED_BLOCKS) > 0
        for key, blocks in _TPU7X_RPA_MIXED_TUNED_BLOCKS.items():
            page_size, kv_name, q_heads, kv_heads, aligned_hd, q_bucket, kv_bucket = key
            assert (page_size, q_heads, kv_heads, aligned_hd) == (256, 32, 4, 128)
            assert isinstance(blocks, RpaBlockSizes)
            assert blocks.bq_sz % blocks.bq_csz == 0
            assert blocks.bkv_sz % blocks.bkv_csz == 0
            assert blocks.bkv_sz % page_size == 0
            assert blocks.bkv_csz % page_size == 0
            assert blocks.bkv_sz <= kv_bucket
            vmem = get_vmem_estimate_bytes(
                actual_num_kv_heads=kv_heads,
                actual_num_q_heads_per_kv_head=q_heads // kv_heads,
                actual_head_dim=aligned_hd,
                bq_sz=blocks.bq_sz,
                bkv_sz=blocks.bkv_sz,
                q_dtype=jnp.bfloat16,
                kv_dtype=jnp.dtype(kv_name),
            )
            assert vmem <= vmem_cap

        # Pin the swept Qwen3-32B tp=2 selections and runtime 64 MiB VMEM clamping.
        pinned_qwen3_tp2 = {
            (jnp.float8_e4m3fn, 9, 1024): RpaBlockSizes(128, 1024, 64, 256),
            (jnp.bfloat16, 9, 1024): RpaBlockSizes(128, 1024, 64, 256),
            (jnp.float8_e4m3fn, 9, 4096): RpaBlockSizes(256, 512, 64, 256),
            (jnp.bfloat16, 9, 4096): RpaBlockSizes(256, 512, 64, 256),
            (jnp.float8_e4m3fn, 37, 4096): RpaBlockSizes(256, 2048, 128, 1024),
            (jnp.bfloat16, 37, 4096): RpaBlockSizes(128, 2048, 128, 1024),
        }
        for (kv_dt, pps, max_tok), want_blocks in pinned_qwen3_tp2.items():
            for vlim in (None, tpu7x_vmem_bytes):
                got_pinned = get_rpa_block_sizes(
                    jnp.bfloat16,
                    kv_dt,
                    actual_num_q_heads=32,
                    actual_num_kv_heads=4,
                    head_dim=128,
                    page_size=256,
                    max_num_tokens=max_tok,
                    max_num_seqs=320,
                    pages_per_seq=pps,
                    case=RpaCase.MIXED,
                    tpu_version=7,
                    vmem_limit_bytes=vlim,
                )
                assert got_pinned == want_blocks
                assert (
                    get_vmem_estimate_bytes(
                        4,
                        8,
                        128,
                        got_pinned.bq_sz,
                        got_pinned.bkv_sz,
                        jnp.bfloat16,
                        kv_dt,
                    )
                    <= vmem_cap
                )

        # Unswept shapes (tp=1, head_dim=256, page_size=128) must fall back to get_default_block_sizes.
        for unswept_q, unswept_kv, unswept_hd, unswept_ps, unswept_pps in (
            (64, 8, 128, 256, 9),
            (64, 8, 128, 256, 37),
            (16, 4, 256, 256, 37),
            (32, 8, 256, 256, 37),
            (32, 4, 128, 128, 9),
        ):
            got_unswept = get_rpa_block_sizes(
                jnp.bfloat16,
                jnp.float8_e4m3fn,
                actual_num_q_heads=unswept_q,
                actual_num_kv_heads=unswept_kv,
                head_dim=unswept_hd,
                page_size=unswept_ps,
                max_num_tokens=4096,
                max_num_seqs=320,
                pages_per_seq=unswept_pps,
                case=RpaCase.MIXED,
                tpu_version=7,
                vmem_limit_bytes=tpu7x_vmem_bytes,
            )
            want_default = RpaBlockSizes(
                **get_default_block_sizes(
                    jnp.bfloat16,
                    jnp.float8_e4m3fn,
                    actual_num_q_heads=unswept_q,
                    actual_num_kv_heads=unswept_kv,
                    head_dim=unswept_hd,
                    page_size=unswept_ps,
                    max_num_tokens=4096,
                    max_num_seqs=320,
                    pages_per_seq=unswept_pps,
                    case=RpaCase.MIXED,
                    tpu_version=7,
                    vmem_limit_bytes=tpu7x_vmem_bytes,
                )
            )
            assert got_unswept == want_default

        for pps in (1, 6, 9, 37):
            for ps in (128, 256):
                got = get_rpa_block_sizes(
                    jnp.bfloat16,
                    jnp.float8_e4m3fn,
                    actual_num_q_heads=32,
                    actual_num_kv_heads=4,
                    head_dim=128,
                    page_size=ps,
                    max_num_tokens=4096,
                    max_num_seqs=320,
                    pages_per_seq=pps,
                    case=RpaCase.MIXED,
                    tpu_version=7,
                    vmem_limit_bytes=tpu7x_vmem_bytes,
                )
                assert got.bkv_sz <= pps * ps
                assert got.bkv_sz % got.bkv_csz == 0
                assert got.bkv_csz % ps == 0
                assert got.bq_sz % got.bq_csz == 0
                static_validate_inputs(
                    jax.ShapeDtypeStruct((4096, 32, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((4096, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((4096, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct(
                        get_kv_cache_shape(max(pps, 8), ps, 4, 128, jnp.float8_e4m3fn),
                        jnp.float8_e4m3fn,
                    ),
                    jax.ShapeDtypeStruct((320,), jnp.int32),
                    jax.ShapeDtypeStruct((320 * pps,), jnp.int32),
                    jax.ShapeDtypeStruct((321,), jnp.int32),
                    jax.ShapeDtypeStruct((3,), jnp.int32),
                    m_block_sizes=tuple(got),
                )

        # 2. Verify prepared=True and lax.cond empty-case skipping across
        # distribution=[0, 0, 2], [2, 2, 2], and [0, 0, 0].
        rng = np.random.default_rng(808)
        tp = min(len(jax.devices()), 2)
        mesh = Mesh(np.array(jax.devices()[:tp]), ("tp",))
        page_size, total_pages = 8, 12
        q_heads, kv_heads, head_dim = 8, 4, 128
        spec = KvCacheSpec(
            layers=1,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_tokens=page_size,
            dtype=jnp.float8_e4m3fn,
        )
        cache_sharding = NamedSharding(mesh, KV_PARTITION)
        head_sharding = NamedSharding(mesh, P(None, "tp", None))
        rep_sharding = NamedSharding(mesh, P())

        dirty_np = np.full(
            (total_pages, *spec.page_shape(tp)), np.nan, dtype=np.float32
        )
        dirty_np[0] = 0.0
        cache = jax.device_put(jnp.asarray(dirty_np, jnp.float8_e4m3fn), cache_sharding)
        ref_cache = jax.device_put(
            jnp.asarray(dirty_np, jnp.float8_e4m3fn), cache_sharding
        )

        q = jax.device_put(
            jnp.asarray(rng.standard_normal((16, q_heads, head_dim)), jnp.bfloat16),
            head_sharding,
        )
        k = jax.device_put(
            jnp.asarray(rng.standard_normal((16, kv_heads, head_dim)), jnp.bfloat16),
            head_sharding,
        )
        v = jax.device_put(
            jnp.asarray(rng.standard_normal((16, kv_heads, head_dim)), jnp.bfloat16),
            head_sharding,
        )
        pos = jax.device_put(jnp.arange(16, dtype=jnp.int32), rep_sharding)
        rc = build_rope_cache(64, head_dim)
        rc_rep = (
            jax.device_put(rc[0], rep_sharding),
            jax.device_put(rc[1], rep_sharding),
        )
        qw = jax.device_put(jnp.ones((head_dim,), jnp.bfloat16), rep_sharding)
        kw = jax.device_put(jnp.ones((head_dim,), jnp.bfloat16), rep_sharding)

        qp, kvm = jax.shard_map(
            lambda ql, kl, vl, p, c, s, qnw, knw: pallas_qk_norm_rope_pack(
                ql,
                kl,
                vl,
                p,
                (c, s),
                qnw,
                knw,
                jnp.float8_e4m3fn,
                k_scale=0.5,
                v_scale=0.25,
                interpret=True,
            ),
            mesh=mesh,
            in_specs=(
                P(None, "tp", None),
                P(None, "tp", None),
                P(None, "tp", None),
                P(),
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=(Q_PACKED_PARTITION, KV_MERGED_PARTITION),
            check_vma=False,
        )(q, k, v, pos, rc_rep[0], rc_rep[1], qw, kw)

        q_roped = apply_rope(rms_norm(q, qw, 1e-6), pos, head_dim, cos_sin_cache=rc_rep)
        k_roped = apply_rope(rms_norm(k, kw, 1e-6), pos, head_dim, cos_sin_cache=rc_rep)
        q_roped = jax.device_put(q_roped, head_sharding)
        k_roped = jax.device_put(k_roped, head_sharding)

        interp_rpa = functools.partial(
            ragged_paged_attention,
            d_block_sizes=(1, 8, 1, 8),
            m_block_sizes=(4, 8, 4, 8),
            interpret=True,
        )
        pages = jnp.array([1, 2, 3, 4], jnp.int32)

        # Prefill: distribution=[0, 0, 2] skips RPAd and runs RPAm.
        meta_pre = AttentionMetadata(
            jax.device_put(jnp.array([9, 5], jnp.int32), rep_sharding),
            (jax.device_put(pages, rep_sharding),),
            jax.device_put(jnp.array([0, 9, 14], jnp.int32), rep_sharding),
            jax.device_put(jnp.array([0, 0, 2], jnp.int32), rep_sharding),
        )
        out_prep, cache = paged_attention(
            qp,
            kvm,
            None,
            cache,
            meta_pre,
            mesh=mesh,
            sm_scale=head_dim**-0.5,
            k_scale=0.5,
            v_scale=0.25,
            prepared=True,
            actual_num_q_heads=q_heads,
            actual_head_dim=head_dim,
            impl=interp_rpa,
        )
        out_unprep, ref_cache = paged_attention(
            q_roped,
            k_roped,
            v,
            ref_cache,
            meta_pre,
            mesh=mesh,
            sm_scale=head_dim**-0.5,
            k_scale=0.5,
            v_scale=0.25,
            impl=reference.ragged_paged_attention,
        )
        assert not np.any(np.isnan(np.asarray(out_prep[:14], np.float32)))
        np.testing.assert_allclose(
            np.asarray(out_prep[:14], np.float32),
            np.asarray(out_unprep[:14], np.float32),
            rtol=2e-2,
            atol=3e-2,
        )

        # Decode: distribution=[2, 2, 2] runs RPAd and skips RPAm.
        meta_dec = AttentionMetadata(
            jax.device_put(jnp.array([10, 6], jnp.int32), rep_sharding),
            (jax.device_put(pages, rep_sharding),),
            jax.device_put(jnp.array([0, 1, 2], jnp.int32), rep_sharding),
            jax.device_put(jnp.array([2, 2, 2], jnp.int32), rep_sharding),
        )
        out_dec, cache = paged_attention(
            qp[:, :2],
            kvm[:2],
            None,
            cache,
            meta_dec,
            mesh=mesh,
            sm_scale=head_dim**-0.5,
            k_scale=0.5,
            v_scale=0.25,
            prepared=True,
            actual_num_q_heads=q_heads,
            actual_head_dim=head_dim,
            impl=interp_rpa,
        )
        out_dec_ref, ref_cache = paged_attention(
            q_roped[:2],
            k_roped[:2],
            v[:2],
            ref_cache,
            meta_dec,
            mesh=mesh,
            sm_scale=head_dim**-0.5,
            k_scale=0.5,
            v_scale=0.25,
            impl=reference.ragged_paged_attention,
        )
        np.testing.assert_allclose(
            np.asarray(out_dec[:2], np.float32),
            np.asarray(out_dec_ref[:2], np.float32),
            rtol=2e-2,
            atol=3e-2,
        )

        # Idle: distribution=[0, 0, 0] skips both RPAd and RPAm without touching cache.
        before_page = np.asarray(cache[1], np.float32).copy()
        meta_idle = AttentionMetadata(
            jax.device_put(jnp.zeros((2,), jnp.int32), rep_sharding),
            (jax.device_put(jnp.zeros((4,), jnp.int32), rep_sharding),),
            jax.device_put(jnp.zeros((3,), jnp.int32), rep_sharding),
            jax.device_put(jnp.zeros((3,), jnp.int32), rep_sharding),
        )
        _, idle_cache = paged_attention(
            qp[:, :2],
            kvm[:2],
            None,
            cache,
            meta_idle,
            mesh=mesh,
            sm_scale=head_dim**-0.5,
            k_scale=0.5,
            v_scale=0.25,
            prepared=True,
            actual_num_q_heads=q_heads,
            actual_head_dim=head_dim,
            impl=interp_rpa,
        )
        np.testing.assert_array_equal(
            np.asarray(idle_cache[1], np.float32), before_page
        )

    def test_prefill_tb512_unpack_norm_and_vmem_guards(self) -> None:
        import unittest.mock as mock

        from resin.backend.tpu.kernels import (
            fused_add_rms_norm_fp8_quant,
            pallas_qkv_norm_rope_pack,
            pallas_unpack_attn_fp8_quant,
            prepare_inputs,
            quantize_fp8_act,
        )
        from resin.backend.tpu.kernels.norm_rope import _select_qk_token_block

        rng = np.random.default_rng(4512)
        iter_eqns = TestDecodeMatmulPath._iter_jaxpr_eqns
        assert_no_pad = TestDecodeMatmulPath._assert_no_pad_eqns

        def _pallas_call_info(fn, *sds):
            call = next(
                e
                for e in iter_eqns(jax.make_jaxpr(fn)(*sds), enter_pallas=False)
                if e.primitive.name == "pallas_call"
            )
            return call.params["name"], call.params["compiler_params"].vmem_limit_bytes

        # Qwen3-32B tp=2 geometry: 32 Q heads, 4 KV heads, out_width=4096 at T=512/768 (zero pad) and T=525 (tail).
        q_heads, kv_heads, head_dim = 32, 4, 128
        for t_len, zero_pad_expected in ((512, True), (768, True), (525, False)):
            q_np = rng.standard_normal((t_len, q_heads, head_dim)).astype(np.float32)
            q_np[-9:] = 0.0
            q_t = jnp.asarray(q_np, jnp.bfloat16)
            dummy_kv = jnp.zeros((t_len, kv_heads, head_dim), jnp.bfloat16)
            out_packed, _ = prepare_inputs(q_t, dummy_kv, dummy_kv)
            ref_bf16 = q_t.reshape(t_len, q_heads * head_dim)
            ref_abs = jnp.max(jnp.abs(ref_bf16.astype(jnp.float32)), axis=-1)
            ref_q, ref_s = quantize_fp8_act(ref_bf16)

            unpack_modes: tuple[Literal["reshape", "bitcast_u32"], ...] = (
                "reshape",
                "bitcast_u32",
            )
            for mode in unpack_modes:
                if zero_pad_expected:
                    u_jaxpr = jax.make_jaxpr(
                        lambda op, _m=mode: pallas_unpack_attn_fp8_quant(
                            op, 8, head_dim, unpack_mode=_m, interpret=True
                        )
                    )(out_packed)
                    assert_no_pad(u_jaxpr)
                att_q, att_s = pallas_unpack_attn_fp8_quant(
                    out_packed, 8, head_dim, unpack_mode=mode, interpret=True
                )
                assert att_s.shape == (1, t_len)
                np.testing.assert_array_equal(np.asarray(att_q), np.asarray(ref_q))
                np.testing.assert_array_equal(np.asarray(att_s.T), np.asarray(ref_s))

                bf16_out, abs_out = pallas_unpack_attn_fp8_quant(
                    out_packed,
                    8,
                    head_dim,
                    return_bf16_and_absmax=True,
                    unpack_mode=mode,
                    interpret=True,
                )
                assert abs_out.shape == (1, t_len)
                np.testing.assert_array_equal(
                    np.asarray(bf16_out), np.asarray(ref_bf16)
                )
                np.testing.assert_array_equal(
                    np.asarray(abs_out[0]), np.asarray(ref_abs)
                )

        m32 = 32 * 1024 * 1024
        m48 = 48 * 1024 * 1024
        with mock.patch(
            "resin.backend.tpu.kernels.norm_rope.get_tpu_version", return_value=7
        ):
            assert _select_qk_token_block(4096, total_heads=40) == 512
            assert _select_qk_token_block(4096, total_heads=80) == 128

            assert _pallas_call_info(
                lambda xi, ri, wi: fused_add_rms_norm_fp8_quant(xi, ri, wi, 1e-6),
                jax.ShapeDtypeStruct((320, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((320, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
            ) == ("fused_add_rms_norm_fp8_quant_256_8192", m32)
            assert _pallas_call_info(
                lambda xi, ri, wi: fused_add_rms_norm_fp8_quant(xi, ri, wi, 1e-6),
                jax.ShapeDtypeStruct((512, 10240), jnp.bfloat16),
                jax.ShapeDtypeStruct((512, 10240), jnp.bfloat16),
                jax.ShapeDtypeStruct((10240,), jnp.bfloat16),
            ) == ("fused_add_rms_norm_fp8_quant_128_10240", m32)

            sp_jaxpr = jax.make_jaxpr(
                lambda xi, ri, wi: fused_add_rms_norm_fp8_quant(xi, ri, wi, 1e-6)
            )(
                jax.ShapeDtypeStruct((2048, 5120), jnp.bfloat16),
                jax.ShapeDtypeStruct((2048, 5120), jnp.bfloat16),
                jax.ShapeDtypeStruct((5120,), jnp.bfloat16),
            )
            assert_no_pad(sp_jaxpr)
            assert all(
                e.primitive.name != "transpose"
                for e in iter_eqns(sp_jaxpr, enter_pallas=False)
            )
            assert _pallas_call_info(
                lambda xi, ri, wi: fused_add_rms_norm_fp8_quant(xi, ri, wi, 1e-6),
                jax.ShapeDtypeStruct((2048, 5120), jnp.bfloat16),
                jax.ShapeDtypeStruct((2048, 5120), jnp.bfloat16),
                jax.ShapeDtypeStruct((5120,), jnp.bfloat16),
            ) == ("fused_add_rms_norm_fp8_quant_256_5120", m32)
            assert _pallas_call_info(
                lambda ql, p, c, s, qw, kw: pallas_qkv_norm_rope_pack(
                    ql,
                    p,
                    (c, s),
                    qw,
                    kw,
                    jnp.float8_e4m3fn,
                    num_q_heads=32,
                    num_kv_heads=4,
                ),
                jax.ShapeDtypeStruct((4096, 5120), jnp.bfloat16),
                jax.ShapeDtypeStruct((4096,), jnp.int32),
                jax.ShapeDtypeStruct((4096, 128), jnp.float32),
                jax.ShapeDtypeStruct((4096, 128), jnp.float32),
                jax.ShapeDtypeStruct((128,), jnp.bfloat16),
                jax.ShapeDtypeStruct((128,), jnp.bfloat16),
            ) == ("qkv_norm_rope_pack_512_32_4", m32)
            assert _pallas_call_info(
                lambda op: pallas_unpack_attn_fp8_quant(op, 8, 128),
                jax.ShapeDtypeStruct((8, 1024, 4, 2, 128), jnp.bfloat16),
            ) == ("unpack_attn_fp8_quant_256_8192", m32)
            assert _pallas_call_info(
                lambda op: pallas_unpack_attn_fp8_quant(
                    op, 8, 128, return_bf16_and_absmax=True
                ),
                jax.ShapeDtypeStruct((4, 1024, 4, 2, 128), jnp.bfloat16),
            ) == ("unpack_attn_absmax_512_4096", m48)

        with mock.patch(
            "resin.backend.tpu.kernels.norm_rope.get_tpu_version", return_value=6
        ):
            assert _select_qk_token_block(4096, total_heads=40) == 128
            assert _pallas_call_info(
                lambda xi, ri, wi: fused_add_rms_norm_fp8_quant(xi, ri, wi, 1e-6),
                jax.ShapeDtypeStruct((2048, 5120), jnp.bfloat16),
                jax.ShapeDtypeStruct((2048, 5120), jnp.bfloat16),
                jax.ShapeDtypeStruct((5120,), jnp.bfloat16),
            ) == ("fused_add_rms_norm_fp8_quant_256_5120", m32)
            assert _pallas_call_info(
                lambda op: pallas_unpack_attn_fp8_quant(op, 8, 128),
                jax.ShapeDtypeStruct((4, 1024, 4, 2, 128), jnp.bfloat16),
            ) == ("unpack_attn_fp8_quant_256_4096", m32)


class TestDecodeMatmulPath:
    @staticmethod
    def _iter_jaxpr_eqns(jaxpr_obj, *, enter_pallas: bool = True):
        jaxpr = jaxpr_obj.jaxpr if hasattr(jaxpr_obj, "jaxpr") else jaxpr_obj
        for eqn in jaxpr.eqns:
            yield eqn
            if not enter_pallas and eqn.primitive.name == "pallas_call":
                continue
            for val in eqn.params.values():
                if hasattr(val, "jaxpr") or hasattr(val, "eqns"):
                    yield from TestDecodeMatmulPath._iter_jaxpr_eqns(
                        val, enter_pallas=enter_pallas
                    )
                elif isinstance(val, (tuple, list)):
                    for item in val:
                        if hasattr(item, "jaxpr") or hasattr(item, "eqns"):
                            yield from TestDecodeMatmulPath._iter_jaxpr_eqns(
                                item, enter_pallas=enter_pallas
                            )

    @classmethod
    def _assert_no_pad_eqns(cls, jaxpr_obj) -> None:
        assert all(
            eqn.primitive.name != "pad" and eqn.params.get("name") != "_pad"
            for eqn in cls._iter_jaxpr_eqns(jaxpr_obj)
        )

    @pytest.mark.parametrize(
        ("batch", "live_rows"),
        [(8, 1), (16, 9), (96, 71), (320, 289)],
        ids=str,
    )
    def test_decode_gate_up_swiglu_zero_pad_rows_and_tile_absmax(
        self, batch: int, live_rows: int
    ) -> None:
        from resin.backend.tpu.kernels import (
            pallas_fp8_gate_up_swiglu,
            quantize_fp8_act,
        )
        from resin.backend.tpu.kernels.fp8_matmul import quantize_fp8_with_absmax

        rng = np.random.default_rng(9000 + batch)
        x_np = rng.standard_normal((batch, 256)).astype(np.float32) * 0.5
        x_np[live_rows:] = 0.0
        x = jnp.asarray(x_np, jnp.bfloat16)
        xq, xs = quantize_fp8_act(x)

        w_gate = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 256)) * 0.03, jnp.bfloat16)
        )
        w_up = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 256)) * 0.03, jnp.bfloat16)
        )
        w_gu_q = jnp.concatenate([w_gate.qvalue, w_up.qvalue], axis=0)
        w_gu_s = jnp.concatenate([w_gate.scale, w_up.scale], axis=0)

        jaxpr = jax.make_jaxpr(
            lambda xq_i, xs_i: pallas_fp8_gate_up_swiglu(
                xq_i, xs_i, w_gu_q, w_gu_s, interpret=True
            )
        )(xq, jnp.transpose(xs))
        self._assert_no_pad_eqns(jaxpr)
        assert all(
            eqn.primitive.name not in ("concatenate", "slice", "dynamic_slice")
            for eqn in self._iter_jaxpr_eqns(jaxpr, enter_pallas=False)
        )

        gated, tile_absmax = pallas_fp8_gate_up_swiglu(
            xq, jnp.transpose(xs), w_gu_q, w_gu_s, interpret=True
        )
        ref_gated = (
            jax.nn.silu(xla_fp8_matmul(x, w_gate.qvalue, w_gate.scale))
            * xla_fp8_matmul(x, w_up.qvalue, w_up.scale)
        ).astype(jnp.bfloat16)

        gated_f32 = np.asarray(gated, np.float32)
        assert not np.any(np.isnan(gated_f32))
        np.testing.assert_allclose(gated_f32[live_rows:], 0.0, atol=0.0)
        np.testing.assert_allclose(
            gated_f32, np.asarray(ref_gated, np.float32), rtol=2e-2, atol=5e-2
        )
        expected_absmax = np.max(np.abs(gated_f32), axis=-1)
        np.testing.assert_allclose(
            np.max(np.asarray(tile_absmax, np.float32), axis=0),
            expected_absmax,
            rtol=1e-2,
            atol=2e-3,
        )

        reduced_row = jnp.max(tile_absmax, axis=0, keepdims=True)
        q_jaxpr = jax.make_jaxpr(
            lambda g_i, a_i: quantize_fp8_with_absmax(
                g_i, a_i, transposed_scale=True, interpret=True
            )
        )(gated, reduced_row)
        self._assert_no_pad_eqns(q_jaxpr)
        assert all(
            eqn.primitive.name != "transpose"
            for eqn in self._iter_jaxpr_eqns(q_jaxpr, enter_pallas=False)
        )
        gq_pallas, gs_pallas = quantize_fp8_with_absmax(
            gated, reduced_row, transposed_scale=True, interpret=True
        )
        gq_xla, gs_xla = quantize_fp8_with_absmax(
            gated, reduced_row, transposed_scale=True, use_pallas=False
        )
        assert gs_pallas.shape == (1, batch)
        np.testing.assert_array_equal(np.asarray(gq_pallas), np.asarray(gq_xla))
        np.testing.assert_array_equal(np.asarray(gs_pallas), np.asarray(gs_xla))
        np.testing.assert_allclose(
            np.asarray(gq_pallas, np.float32)[live_rows:], 0.0, atol=0.0
        )

        exact_abs_row = jnp.max(
            jnp.abs(gated.astype(jnp.float32)), axis=-1, keepdims=True
        ).T
        exact_gq, exact_gs = quantize_fp8_with_absmax(
            gated, exact_abs_row, transposed_scale=True, interpret=True
        )
        ref_gq, ref_gs = quantize_fp8_act(gated)
        np.testing.assert_array_equal(np.asarray(exact_gq), np.asarray(ref_gq))
        np.testing.assert_array_equal(np.asarray(exact_gs.T), np.asarray(ref_gs))

    @pytest.mark.parametrize("batch", [1, 5, 8, 16, 96, 192, 313, 320])
    def test_pallas_fp8_gate_up_swiglu_quant_interpret(self, batch: int) -> None:
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call
        from jax.experimental.pallas import tpu as pltpu

        from resin.backend.tpu.kernels import (
            fp8_matmul_prequantized,
            pallas_fp8_gate_up_swiglu,
            pallas_fp8_gate_up_swiglu_quant,
            quantize_fp8_act,
            quantize_fp8_weight,
            sharded_fp8_gate_up_swiglu,
        )
        from resin.backend.tpu.kernels.fp8_matmul import quantize_fp8_with_absmax

        rng = np.random.default_rng(4200 + batch)
        ffn, k_dim = (300, 192) if batch in (5, 313) else (1280, 256)
        x = jnp.asarray(rng.standard_normal((batch, k_dim)) * 0.4, jnp.bfloat16)
        xq, xs = quantize_fp8_act(x)
        xs_row = jnp.transpose(xs)
        w_gate = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((ffn, k_dim)) * 0.04, jnp.bfloat16)
        )
        w_up = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((ffn, k_dim)) * 0.04, jnp.bfloat16)
        )
        w_gu_q = jnp.concatenate([w_gate.qvalue, w_up.qvalue], axis=0)
        w_gu_s = jnp.concatenate([w_gate.scale, w_up.scale], axis=0)

        gq, gs = pallas_fp8_gate_up_swiglu_quant(
            xq,
            xs_row,
            w_gu_q,
            w_gu_s,
            transposed_scale=True,
            interpret=pltpu.InterpretParams(detect_races=True),
        )
        assert interpret_pallas_call.races is not None
        assert not interpret_pallas_call.races.races_found

        assert gq.shape == (batch, ffn)
        assert gq.dtype == jnp.float8_e4m3fn
        assert gs.shape == (1, batch)
        deq_got = np.asarray(
            gq.astype(jnp.float32) * jnp.transpose(gs), dtype=np.float32
        )

        unfused_gated, unfused_abs = pallas_fp8_gate_up_swiglu(
            xq, xs_row, w_gu_q, w_gu_s, interpret=True
        )
        unfused_q, unfused_s = quantize_fp8_with_absmax(
            unfused_gated,
            jnp.max(unfused_abs, axis=0, keepdims=True),
            dtype=w_gu_q.dtype,
            transposed_scale=True,
            interpret=True,
        )
        np.testing.assert_array_equal(np.asarray(gq), np.asarray(unfused_q))
        np.testing.assert_allclose(
            np.asarray(gs), np.asarray(unfused_s), rtol=1e-6, atol=0.0
        )

        gate_f32 = fp8_matmul_prequantized(
            xq, xs, w_gate.qvalue, w_gate.scale, out_dtype=jnp.float32
        )
        up_f32 = fp8_matmul_prequantized(
            xq, xs, w_up.qvalue, w_up.scale, out_dtype=jnp.float32
        )
        ref_swiglu = (jax.nn.silu(gate_f32) * up_f32).astype(jnp.bfloat16)
        ref_q, ref_s = quantize_fp8_act(ref_swiglu)
        deq_ref = np.asarray(ref_q.astype(jnp.float32) * ref_s, dtype=np.float32)

        np.testing.assert_allclose(deq_got, deq_ref, rtol=2e-2, atol=4e-2)
        cos = float(
            np.dot(deq_got.ravel(), deq_ref.ravel())
            / (np.linalg.norm(deq_got.ravel()) * np.linalg.norm(deq_ref.ravel()))
        )
        assert cos >= 0.9995

        if batch in (1, 8):
            zero_xq = jnp.zeros_like(xq)
            zero_xs = jnp.ones((1, batch), dtype=jnp.float32)
            zq, zs = pallas_fp8_gate_up_swiglu_quant(
                zero_xq, zero_xs, w_gu_q, w_gu_s, interpret=True
            )
            assert not np.any(np.isnan(np.asarray(zq, np.float32)))
            np.testing.assert_array_equal(
                np.asarray(zs), np.ones((1, batch), dtype=np.float32)
            )
            np.testing.assert_array_equal(
                np.asarray(zq, np.float32), np.zeros((batch, ffn), dtype=np.float32)
            )

        if len(jax.devices()) >= 2 and ffn % 256 == 0 and k_dim == 256:
            mesh = Mesh(np.array(jax.devices()[:2]), ("tp",))
            half = ffn // 2
            w_sh_q = jax.device_put(
                jnp.concatenate(
                    [
                        w_gate.qvalue[:half],
                        w_up.qvalue[:half],
                        w_gate.qvalue[half:],
                        w_up.qvalue[half:],
                    ],
                    axis=0,
                ),
                NamedSharding(mesh, P("tp", None)),
            )
            w_sh_s = jax.device_put(
                jnp.concatenate(
                    [
                        w_gate.scale[:half],
                        w_up.scale[:half],
                        w_gate.scale[half:],
                        w_up.scale[half:],
                    ],
                    axis=0,
                ),
                NamedSharding(mesh, P("tp")),
            )
            xq_rep = jax.device_put(xq, NamedSharding(mesh, P(None, None)))
            xs_rep = jax.device_put(xs_row, NamedSharding(mesh, P(None, None)))
            sh_gq, sh_gs = sharded_fp8_gate_up_swiglu(
                xq_rep,
                xs_rep,
                w_sh_q,
                w_sh_s,
                mesh=mesh,
                sequence_parallel=False,
                sync_tp_scale=False,
                quantize_output=True,
                backend="pallas",
                interpret=True,
            )
            assert sh_gs is not None and sh_gs.shape == (2, batch)
            sh_deq = np.concatenate(
                [
                    np.asarray(
                        sh_gq[:, :half].astype(jnp.float32) * sh_gs[0:1].T,
                        dtype=np.float32,
                    ),
                    np.asarray(
                        sh_gq[:, half:].astype(jnp.float32) * sh_gs[1:2].T,
                        dtype=np.float32,
                    ),
                ],
                axis=-1,
            )
            ref_f32 = np.asarray(jax.nn.silu(gate_f32) * up_f32, dtype=np.float32)
            sh_cos = float(
                np.dot(sh_deq.ravel(), ref_f32.ravel())
                / (np.linalg.norm(sh_deq.ravel()) * np.linalg.norm(ref_f32.ravel()))
            )
            assert sh_cos >= 0.999

    def test_quantize_fp8_with_absmax_k_tiling_and_xla_fallback(self) -> None:
        from resin.backend.tpu.kernels import quantize_fp8_act
        from resin.backend.tpu.kernels.fp8_matmul import quantize_fp8_with_absmax

        rng = np.random.default_rng(320)
        wide_x = jnp.asarray(
            rng.standard_normal((320, 12800)).astype(np.float32) * 0.25,
            jnp.bfloat16,
        )
        wide_abs = jnp.max(
            jnp.abs(wide_x.astype(jnp.float32)), axis=-1, keepdims=True
        ).T
        wq_pallas, ws_pallas = quantize_fp8_with_absmax(
            wide_x,
            wide_abs,
            jnp.float8_e4m3fn,
            transposed_scale=True,
            use_pallas=True,
            interpret=True,
        )
        wq_ref, ws_ref = quantize_fp8_act(wide_x)
        np.testing.assert_array_equal(np.asarray(wq_pallas), np.asarray(wq_ref))
        np.testing.assert_array_equal(np.asarray(ws_pallas.T), np.asarray(ws_ref))

        for fallback_shape in ((384, 256), (16, 192)):
            fb_x = jnp.asarray(
                rng.standard_normal(fallback_shape).astype(np.float32) * 0.25,
                jnp.bfloat16,
            )
            fb_abs = jnp.max(
                jnp.abs(fb_x.astype(jnp.float32)), axis=-1, keepdims=True
            ).T
            with pytest.raises(ValueError, match="unsupported"):
                quantize_fp8_with_absmax(
                    fb_x, fb_abs, jnp.float8_e4m3fn, use_pallas=True, interpret=True
                )
            fq_auto, fs_auto = quantize_fp8_with_absmax(
                fb_x,
                fb_abs,
                jnp.float8_e4m3fn,
                transposed_scale=True,
                interpret=True,
            )
            fq_x, fs_x = quantize_fp8_with_absmax(
                fb_x,
                fb_abs,
                jnp.float8_e4m3fn,
                transposed_scale=True,
                use_pallas=False,
            )
            np.testing.assert_array_equal(np.asarray(fq_auto), np.asarray(fq_x))
            np.testing.assert_array_equal(np.asarray(fs_auto), np.asarray(fs_x))

    @pytest.mark.parametrize("batch", [160, 192, 320])
    def test_zero_pad_norms_qkv_pack_and_unpack_attn_fp8_quant(
        self, batch: int
    ) -> None:
        from resin.backend.tpu.kernels import (
            fused_add_rms_norm_fp8_quant,
            pack_outputs,
            pallas_qkv_norm_rope_pack,
            pallas_unpack_attn_fp8_quant,
            prepare_inputs,
            prepare_outputs,
            quantize_fp8_act,
        )

        rng = np.random.default_rng(4000 + batch)
        x_np = rng.standard_normal((batch, 256)).astype(np.float32) * 0.5
        res_np = rng.standard_normal((batch, 256)).astype(np.float32) * 0.5
        x_np[-7:] = 0.0
        res_np[-7:] = 0.0
        x = jnp.asarray(x_np, jnp.bfloat16)
        res = jnp.asarray(res_np, jnp.bfloat16)
        nw = jnp.asarray(rng.uniform(0.5, 1.5, size=(256,)), jnp.bfloat16)

        norm_jaxpr = jax.make_jaxpr(
            lambda xi: rms_norm_fp8_quant(
                xi, nw, 1e-6, transposed_scale=True, interpret=True
            )
        )(x)
        self._assert_no_pad_eqns(norm_jaxpr)
        assert all(
            eqn.primitive.name != "transpose"
            for eqn in self._iter_jaxpr_eqns(norm_jaxpr, enter_pallas=False)
        )

        add_norm_jaxpr = jax.make_jaxpr(
            lambda xi, ri: fused_add_rms_norm_fp8_quant(
                xi, ri, nw, 1e-6, transposed_scale=True, interpret=True
            )
        )(x, res)
        self._assert_no_pad_eqns(add_norm_jaxpr)
        assert all(
            eqn.primitive.name != "transpose"
            for eqn in self._iter_jaxpr_eqns(add_norm_jaxpr, enter_pallas=False)
        )

        xq, xs_t, next_res = fused_add_rms_norm_fp8_quant(
            x, res, nw, 1e-6, transposed_scale=True, interpret=True
        )
        assert xs_t.shape == (1, batch)
        ref_normed, ref_res = fused_add_rms_norm(x, res, nw, 1e-6)
        ref_xq, ref_xs = quantize_fp8_act(ref_normed)
        np.testing.assert_array_equal(np.asarray(next_res), np.asarray(ref_res))
        np.testing.assert_array_equal(np.asarray(xq), np.asarray(ref_xq))
        np.testing.assert_array_equal(np.asarray(xs_t.T), np.asarray(ref_xs))

        # QKV norm+RoPE+pack: zero pad primitives at 160, 192, 320.
        q_heads, kv_heads, head_dim = 8, 2, 128
        q_np = rng.standard_normal((batch, q_heads, head_dim)).astype(np.float32)
        k_np = rng.standard_normal((batch, kv_heads, head_dim)).astype(np.float32)
        v_np = rng.standard_normal((batch, kv_heads, head_dim)).astype(np.float32)
        q_np[-7:] = 0.0
        k_np[-7:] = 0.0
        v_np[-7:] = 0.0
        q_in = jnp.asarray(q_np, jnp.bfloat16)
        k_in = jnp.asarray(k_np, jnp.bfloat16)
        v_in = jnp.asarray(v_np, jnp.bfloat16)
        qkv = jnp.concatenate(
            [q_in.reshape(batch, -1), k_in.reshape(batch, -1), v_in.reshape(batch, -1)],
            axis=-1,
        )
        pos = jnp.arange(batch, dtype=jnp.int32)
        rc = build_rope_cache(512, head_dim)
        qnw = jnp.asarray(rng.uniform(0.5, 1.5, size=(head_dim,)), jnp.bfloat16)
        knw = jnp.asarray(rng.uniform(0.5, 1.5, size=(head_dim,)), jnp.bfloat16)

        qkv_jaxpr = jax.make_jaxpr(
            lambda qkv_i, p_i: pallas_qkv_norm_rope_pack(
                qkv_i,
                p_i,
                rc,
                qnw,
                knw,
                jnp.float8_e4m3fn,
                num_q_heads=q_heads,
                num_kv_heads=kv_heads,
                interpret=True,
            )
        )(qkv, pos)
        self._assert_no_pad_eqns(qkv_jaxpr)

        qp, kvm = pallas_qkv_norm_rope_pack(
            qkv,
            pos,
            rc,
            qnw,
            knw,
            jnp.float8_e4m3fn,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            interpret=True,
        )
        q_ref = apply_rope(rms_norm(q_in, qnw, 1e-6), pos, head_dim, cos_sin_cache=rc)
        k_ref = quantize_kv(
            apply_rope(rms_norm(k_in, knw, 1e-6), pos, head_dim, cos_sin_cache=rc),
            jnp.float8_e4m3fn,
        )
        v_ref = quantize_kv(v_in, jnp.float8_e4m3fn)
        exp_qp, exp_kvm = prepare_inputs(q_ref, k_ref, v_ref)
        np.testing.assert_allclose(
            np.asarray(qp, np.float32),
            np.asarray(exp_qp, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(kvm, np.float32),
            np.asarray(exp_kvm, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )

        # Unpack attention + FP8 quant across reshape & bitcast_u32 with zero pad primitives,
        # anchored directly to the original 3D heads q_ref.
        out_packed = pack_outputs(q_ref, kv_heads)
        np.testing.assert_array_equal(np.asarray(out_packed), np.asarray(exp_qp))
        np.testing.assert_array_equal(
            np.asarray(prepare_outputs(out_packed, q_heads // kv_heads, head_dim)),
            np.asarray(q_ref),
        )
        ref_att_q, ref_att_s = quantize_fp8_act(
            q_ref.reshape(batch, q_heads * head_dim)
        )

        unpack_modes: tuple[Literal["reshape", "bitcast_u32"], ...] = (
            "reshape",
            "bitcast_u32",
        )
        for mode in unpack_modes:
            unpack_jaxpr = jax.make_jaxpr(
                lambda op, _m=mode: pallas_unpack_attn_fp8_quant(
                    op,
                    q_heads // kv_heads,
                    head_dim,
                    unpack_mode=_m,
                    interpret=True,
                )
            )(out_packed)
            self._assert_no_pad_eqns(unpack_jaxpr)
            att_q, att_s = pallas_unpack_attn_fp8_quant(
                out_packed,
                q_heads // kv_heads,
                head_dim,
                unpack_mode=mode,
                interpret=True,
            )
            assert att_s.shape == (1, batch)
            np.testing.assert_array_equal(np.asarray(att_q), np.asarray(ref_att_q))
            np.testing.assert_array_equal(np.asarray(att_s.T), np.asarray(ref_att_s))

    def test_per_die_scaling_zero_pmax_and_prequantized_gate_up_down(self) -> None:
        from resin.backend.tpu.kernels import (
            MatmulBackend,
            quantize_fp8_act,
            sharded_fp8_gate_up_swiglu,
        )

        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("needs 2 host devices")
        mesh = Mesh(np.array(devices[:2]), ("tp",))

        rng = np.random.default_rng(7788)
        x_np = rng.standard_normal((32, 256)).astype(np.float32) * 0.25
        # Make die 1's input shard 16x larger so per-die scaling strictly beats global pmax.
        x_np[:, 128:] *= 16.0
        x_np[26:] = 0.0
        x = jnp.asarray(x_np, jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((256, 256)) * 0.05, jnp.bfloat16)
        fw = quantize_fp8_weight(w)

        exact_f32 = np.asarray(x, np.float32) @ np.asarray(w, np.float32).T
        x_sh = jax.device_put(x, NamedSharding(mesh, P(None, "tp")))
        wq_sh = jax.device_put(fw.qvalue, NamedSharding(mesh, P(None, "tp")))
        ws_rep = jax.device_put(fw.scale, NamedSharding(mesh, P()))

        row_jaxpr = jax.make_jaxpr(
            lambda xi, wqi, wsi: sharded_fp8_matmul(
                xi,
                wqi,
                wsi,
                P(None, "tp"),
                mesh=mesh,
                sync_tp_scale=False,
                backend="pallas",
                interpret=True,
            )
        )(x_sh, wq_sh, ws_rep)
        assert "pmax" not in str(row_jaxpr)

        y_synced = sharded_fp8_matmul(
            x_sh,
            wq_sh,
            ws_rep,
            P(None, "tp"),
            mesh=mesh,
            sync_tp_scale=True,
            reduce_dtype=jnp.float32,
            backend="pallas",
            interpret=True,
        )
        y_local = sharded_fp8_matmul(
            x_sh,
            wq_sh,
            ws_rep,
            P(None, "tp"),
            mesh=mesh,
            sync_tp_scale=False,
            reduce_dtype=jnp.float32,
            backend="pallas",
            interpret=True,
        )
        from unittest import mock

        from jax._src import tpu_info

        v7x_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion("7x"), 1)
        with (
            mock.patch.object(tpu_info, "get_tpu_info", return_value=v7x_info),
            mock.patch(
                "resin.backend.tpu.kernels.fp8_matmul.get_tpu_version", return_value=7
            ),
        ):
            y_local_b3 = sharded_fp8_matmul(
                x_sh,
                wq_sh,
                ws_rep,
                P(None, "tp"),
                mesh=mesh,
                sync_tp_scale=False,
                reduce_dtype=jnp.float32,
                tile_sizes=Fp8MatmulTileSizes(32, 128, 128, 3),
                backend="pallas",
                interpret=True,
            )
        y_local_xla = sharded_fp8_matmul(
            x_sh,
            wq_sh,
            ws_rep,
            P(None, "tp"),
            mesh=mesh,
            sync_tp_scale=False,
            reduce_dtype=jnp.float32,
            backend="xla",
        )
        err_synced = float(np.linalg.norm(np.asarray(y_synced, np.float32) - exact_f32))
        err_local = float(np.linalg.norm(np.asarray(y_local, np.float32) - exact_f32))
        assert err_local <= err_synced
        np.testing.assert_allclose(np.asarray(y_local, np.float32)[26:], 0.0, atol=0.0)
        np.testing.assert_array_equal(np.asarray(y_local_b3), np.asarray(y_local))
        np.testing.assert_allclose(
            np.asarray(y_local_xla, np.float32),
            np.asarray(y_local, np.float32),
            rtol=1e-3,
            atol=1e-2,
        )

        # Verify sharded_fp8_gate_up_swiglu(..., sync_tp_scale=False) emits zero pmax.
        xq_rep, xs_rep = quantize_fp8_act(x)
        xq_rep = jax.device_put(xq_rep, NamedSharding(mesh, P(None, None)))
        xs_rep = jax.device_put(
            jnp.transpose(xs_rep), NamedSharding(mesh, P(None, None))
        )
        w_gu = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((512, 256)) * 0.03, jnp.bfloat16)
        )
        w_gu_q = jax.device_put(w_gu.qvalue, NamedSharding(mesh, P("tp", None)))
        w_gu_s = jax.device_put(w_gu.scale, NamedSharding(mesh, P("tp")))
        gu_jaxpr = jax.make_jaxpr(
            lambda xqi, xsi, wqi, wsi: sharded_fp8_gate_up_swiglu(
                xqi,
                xsi,
                wqi,
                wsi,
                mesh=mesh,
                sequence_parallel=False,
                sync_tp_scale=False,
                backend="pallas",
                interpret=True,
            )
        )(xq_rep, xs_rep, w_gu_q, w_gu_s)
        assert "pmax" not in str(gu_jaxpr)

        # Verify per-die gate_up absmax reduces across all FFN tiles (n_out > 1)
        # and quantize_output=True feeds pre-quantized FP8 directly into down_proj.
        ffn_local = 2560  # 2 * 1280 -> n_out = 2 tiles per die
        w_gu_multi_np = (
            rng.standard_normal((2 * 2 * ffn_local, 256)).astype(np.float32) * 0.02
        )
        # Amplify tile 1 (rows 1280..2559 of gate and up on each die) so tile 0 alone under-reports absmax.
        for die in range(2):
            base_row = die * (2 * ffn_local)
            w_gu_multi_np[base_row + 1280 : base_row + 2560] *= 10.0
            w_gu_multi_np[base_row + ffn_local + 1280 : base_row + 2 * ffn_local] *= (
                10.0
            )
        w_gu_multi = quantize_fp8_weight(jnp.asarray(w_gu_multi_np, jnp.bfloat16))
        w_gu_m_q = jax.device_put(w_gu_multi.qvalue, NamedSharding(mesh, P("tp", None)))
        w_gu_m_s = jax.device_put(w_gu_multi.scale, NamedSharding(mesh, P("tp")))
        g_multi, g_abs_per_die = sharded_fp8_gate_up_swiglu(
            xq_rep,
            xs_rep,
            w_gu_m_q,
            w_gu_m_s,
            mesh=mesh,
            sequence_parallel=False,
            sync_tp_scale=False,
            backend="pallas",
            interpret=True,
        )
        assert g_abs_per_die is not None
        assert g_abs_per_die.shape == (2, 32)
        g_multi_f32 = np.asarray(g_multi, np.float32)
        expected_per_die = np.stack(
            [
                np.max(np.abs(g_multi_f32[:, :ffn_local]), axis=-1),
                np.max(np.abs(g_multi_f32[:, ffn_local:]), axis=-1),
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            np.asarray(g_abs_per_die, np.float32),
            expected_per_die,
            rtol=1e-2,
            atol=2e-3,
        )

        w_down_m = quantize_fp8_weight(
            jnp.asarray(
                rng.standard_normal((256, 2 * ffn_local)).astype(np.float32) * 0.02,
                jnp.bfloat16,
            )
        )
        w_d_q = jax.device_put(w_down_m.qvalue, NamedSharding(mesh, P(None, "tp")))
        w_d_s = jax.device_put(w_down_m.scale, NamedSharding(mesh, P()))
        for sync in (False, True):
            mlp_jaxpr = jax.make_jaxpr(
                lambda xqi, xsi, _sync=sync: sharded_fp8_gate_up_swiglu(
                    xqi,
                    xsi,
                    w_gu_m_q,
                    w_gu_m_s,
                    mesh=mesh,
                    sequence_parallel=False,
                    sync_tp_scale=_sync,
                    quantize_output=True,
                    backend="pallas",
                    interpret=True,
                )
            )(xq_rep, xs_rep)
            self._assert_no_pad_eqns(mlp_jaxpr)
            if not sync:
                assert "pmax" not in str(mlp_jaxpr)

            backend_cases: tuple[tuple[MatmulBackend, bool], ...] = (
                ("pallas", True),
                ("xla", False),
            )
            for be, interp in backend_cases:
                g_bf16, g_abs = sharded_fp8_gate_up_swiglu(
                    xq_rep,
                    xs_rep,
                    w_gu_m_q,
                    w_gu_m_s,
                    mesh=mesh,
                    sequence_parallel=False,
                    sync_tp_scale=sync,
                    quantize_output=False,
                    backend=be,
                    interpret=interp,
                )
                ref_down = sharded_fp8_matmul(
                    g_bf16,
                    w_d_q,
                    w_d_s,
                    P(None, "tp"),
                    mesh=mesh,
                    x_abs_max=g_abs,
                    sync_tp_scale=sync,
                    backend=be,
                    interpret=interp,
                )
                gq, gs = sharded_fp8_gate_up_swiglu(
                    xq_rep,
                    xs_rep,
                    w_gu_m_q,
                    w_gu_m_s,
                    mesh=mesh,
                    sequence_parallel=False,
                    sync_tp_scale=sync,
                    quantize_output=True,
                    backend=be,
                    interpret=interp,
                )
                assert gq.dtype == jnp.float8_e4m3fn
                assert gs is not None
                assert gs.shape == ((1, 32) if sync else (2, 32))
                got_down = sharded_fp8_matmul(
                    gq,
                    w_d_q,
                    w_d_s,
                    P(None, "tp"),
                    mesh=mesh,
                    x_scale=gs,
                    sync_tp_scale=sync,
                    backend=be,
                    interpret=interp,
                )
                np.testing.assert_allclose(
                    np.asarray(got_down, np.float32)[26:], 0.0, atol=0.0
                )
                np.testing.assert_allclose(
                    np.asarray(got_down, np.float32),
                    np.asarray(ref_down, np.float32),
                    rtol=1e-2,
                    atol=2e-2,
                )

    def test_async_row_all_reduce_race_free_zero_psum_and_vmem_budget(self) -> None:
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call
        from jax.experimental.pallas import tpu as pltpu

        from resin.backend.tpu.kernels import (
            estimate_async_ar_vmem_bytes,
            get_async_ar_tile_sizes,
            quantize_fp8_act,
            sharded_fp8_row_matmul_async_ar,
        )

        # VMEM audit across all Qwen3-32B tp=2 decode buckets (including <8 padded) on o_proj and down_proj.
        vmem_budget_bytes = 40 * 1024 * 1024
        decode_buckets = (1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320)
        for bs in decode_buckets:
            for n_out_dim, k_local in ((5120, 4096), (5120, 12800)):
                tiles = get_async_ar_tile_sizes(bs, n_out_dim, k_local)
                n_out = n_out_dim // tiles.out_block_size
                assert n_out >= 4
                for fuse_nq in (False, True):
                    vmem_bytes = estimate_async_ar_vmem_bytes(
                        tiles.batch_block_size,
                        tiles.out_block_size,
                        tiles.in_block_size,
                        n_out,
                        min(tiles.w_buffer_count, n_out),
                        fuse_norm_quant=fuse_nq,
                    )
                    assert vmem_bytes <= vmem_budget_bytes

        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("requires at least 2 devices for tp=2 async all-reduce")

        mesh = make_mesh(2, devices[:2])
        row_sharding = NamedSharding(mesh, P(None, "tp"))
        rep_sharding = NamedSharding(mesh, P())
        rng = np.random.default_rng(2026)

        def _quant_per_die(x_bf16_sh: jax.Array) -> tuple[jax.Array, jax.Array]:
            n_rows = x_bf16_sh.shape[0]

            def _local_q(xl: jax.Array) -> tuple[jax.Array, jax.Array]:
                ql, sl = quantize_fp8_act(xl, tp_axis="tp", sync_tp_scale=False)
                return ql, jnp.transpose(sl.reshape(n_rows, 1))

            return jax.shard_map(
                _local_q,
                mesh=mesh,
                in_specs=P(None, "tp"),
                out_specs=(P(None, "tp"), P("tp", None)),
                check_vma=False,
            )(x_bf16_sh)

        for dma_mode in ("on_wait", "eager"):
            interp_params = pltpu.InterpretParams(
                detect_races=True, dma_execution_mode=dma_mode
            )
            for batch, live, n_dim, k_global, tiles in (
                (24, 21, 512, 512, Fp8MatmulTileSizes(24, 128, 256, 2)),
                (13, 13, 640, 512, Fp8MatmulTileSizes(16, 128, 256, 3)),
                (8, 8, 256, 256, Fp8MatmulTileSizes(8, 128, 128, 2)),
            ):
                x_np = rng.standard_normal((batch, k_global)).astype(np.float32) * 0.4
                x_np[live:] = 0.0
                x_bf16 = jax.device_put(jnp.asarray(x_np, jnp.bfloat16), row_sharding)
                w = quantize_fp8_weight(
                    jnp.asarray(
                        rng.standard_normal((n_dim, k_global)) * 0.03, jnp.bfloat16
                    )
                )
                wq = jax.device_put(w.qvalue, row_sharding)
                ws = jax.device_put(w.scale, rep_sharding)
                xq, xs = _quant_per_die(x_bf16)

                @jax.jit
                def ref_fn(xqi: jax.Array, xsi: jax.Array) -> jax.Array:
                    return sharded_fp8_matmul(
                        xqi,
                        wq,
                        ws,
                        P(None, "tp"),
                        mesh=mesh,
                        x_scale=xsi,
                        sync_tp_scale=False,
                        tile_sizes=tiles,
                        backend="pallas",
                        interpret=True,
                    )

                @jax.jit
                def async_fn(xqi: jax.Array, xsi: jax.Array) -> jax.Array:
                    return sharded_fp8_row_matmul_async_ar(
                        xqi,
                        wq,
                        ws,
                        mesh=mesh,
                        x_scale=xsi,
                        sync_tp_scale=False,
                        tile_sizes=tiles,
                        backend="pallas",
                        interpret=interp_params,
                    )

                async_jaxpr = jax.make_jaxpr(async_fn)(xq, xs)
                assert "psum" not in str(async_jaxpr)

                y_ref = jax.block_until_ready(ref_fn(xq, xs))
                y_got = jax.block_until_ready(async_fn(xq, xs))
                assert interpret_pallas_call.races is not None
                assert not interpret_pallas_call.races.races_found

                shard_0 = np.asarray(y_got.addressable_shards[0].data)
                shard_1 = np.asarray(y_got.addressable_shards[1].data)
                np.testing.assert_array_equal(shard_0, shard_1)
                np.testing.assert_array_equal(shard_0, np.asarray(y_ref))
                if live < batch:
                    np.testing.assert_array_equal(
                        np.asarray(shard_0[live:], np.float32), 0.0
                    )

        # Back-to-back o_proj -> down_proj chain in a single jit across two consecutive calls.
        w_o = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 512)) * 0.03, jnp.bfloat16)
        )
        w_d = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 1024)) * 0.03, jnp.bfloat16)
        )
        wo_q = jax.device_put(w_o.qvalue, row_sharding)
        wo_s = jax.device_put(w_o.scale, rep_sharding)
        wd_q = jax.device_put(w_d.qvalue, row_sharding)
        wd_s = jax.device_put(w_d.scale, rep_sharding)
        x_o = jax.device_put(
            jnp.asarray(rng.standard_normal((16, 512)) * 0.3, jnp.bfloat16),
            row_sharding,
        )
        x_d = jax.device_put(
            jnp.asarray(rng.standard_normal((16, 1024)) * 0.3, jnp.bfloat16),
            row_sharding,
        )
        chain_ip = pltpu.InterpretParams(detect_races=True, dma_execution_mode="eager")

        @jax.jit
        def _two_proj_async(xo: jax.Array, xd: jax.Array) -> jax.Array:
            yo = sharded_fp8_row_matmul_async_ar(
                xo,
                wo_q,
                wo_s,
                mesh=mesh,
                sync_tp_scale=False,
                backend="pallas",
                interpret=chain_ip,
            )
            yd = sharded_fp8_row_matmul_async_ar(
                xd,
                wd_q,
                wd_s,
                mesh=mesh,
                sync_tp_scale=False,
                backend="pallas",
                interpret=chain_ip,
            )
            return yo + yd

        y1 = jax.block_until_ready(_two_proj_async(x_o, x_d))
        y2 = jax.block_until_ready(_two_proj_async(x_o, x_d))
        assert interpret_pallas_call.races is not None
        assert not interpret_pallas_call.races.races_found
        np.testing.assert_array_equal(np.asarray(y1), np.asarray(y2))
        y_ref_chain = sharded_fp8_matmul(
            x_o,
            wo_q,
            wo_s,
            P(None, "tp"),
            mesh=mesh,
            sync_tp_scale=False,
            backend="pallas",
            interpret=True,
        ) + sharded_fp8_matmul(
            x_d,
            wd_q,
            wd_s,
            P(None, "tp"),
            mesh=mesh,
            sync_tp_scale=False,
            backend="pallas",
            interpret=True,
        )
        np.testing.assert_array_equal(np.asarray(y1), np.asarray(y_ref_chain))

    @pytest.mark.parametrize("batch", [1, 8, 16, 64, 192, 224, 313, 320])
    @pytest.mark.parametrize("n_dim,k_global", [(256, 256), (512, 256), (384, 256)])
    def test_async_ar_fused_add_rms_norm_fp8_quant_interpret(
        self, batch: int, n_dim: int, k_global: int
    ) -> None:
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call
        from jax.experimental.pallas import tpu as pltpu

        from resin.backend.tpu.kernels import (
            estimate_async_ar_vmem_bytes,
            fp8_matmul_prequantized,
            fused_add_rms_norm_fp8_quant,
            get_async_ar_tile_sizes,
            quantize_fp8_act,
            quantize_fp8_weight,
            sharded_fp8_row_matmul_async_ar,
        )
        from resin.backend.tpu.kernels._rpa_v3_pallas import align_to

        if batch in (64, 192, 224, 320) and (n_dim, k_global) != (512, 256):
            return

        if batch == 320 and (n_dim, k_global) == (512, 256):
            for out_dim, k_loc in ((5120, 4096), (5120, 12800)):
                t_qwen = get_async_ar_tile_sizes(320, out_dim, k_loc)
                n_out_q = out_dim // t_qwen.out_block_size
                v_bytes = estimate_async_ar_vmem_bytes(
                    t_qwen.batch_block_size,
                    t_qwen.out_block_size,
                    t_qwen.in_block_size,
                    n_out_q,
                    min(t_qwen.w_buffer_count, n_out_q),
                    fuse_norm_quant=True,
                )
                assert v_bytes <= 40 * 1024 * 1024

        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("requires at least 2 devices for tp=2 async all-reduce")

        mesh = make_mesh(2, devices[:2])
        row_sharding = NamedSharding(mesh, P(None, "tp"))
        rep_sharding = NamedSharding(mesh, P())
        rng = np.random.default_rng(7100 + batch * 13 + n_dim)

        x_bf16 = jax.device_put(
            jnp.asarray(rng.standard_normal((batch, k_global)) * 0.4, jnp.bfloat16),
            row_sharding,
        )
        res_bf16 = jax.device_put(
            jnp.asarray(rng.standard_normal((batch, n_dim)) * 0.4, jnp.bfloat16),
            rep_sharding,
        )
        norm_w = jax.device_put(
            jnp.asarray(rng.uniform(0.5, 1.5, size=(n_dim,)), jnp.bfloat16),
            rep_sharding,
        )
        w = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((n_dim, k_global)) * 0.03, jnp.bfloat16)
        )
        wq = jax.device_put(w.qvalue, row_sharding)
        ws = jax.device_put(w.scale, rep_sharding)

        def _quant_per_die(x_sh: jax.Array) -> tuple[jax.Array, jax.Array]:
            def _lq(xl: jax.Array) -> tuple[jax.Array, jax.Array]:
                ql, sl = quantize_fp8_act(xl, tp_axis="tp", sync_tp_scale=False)
                return ql, jnp.transpose(sl.reshape(batch, 1))

            return jax.shard_map(
                _lq,
                mesh=mesh,
                in_specs=P(None, "tp"),
                out_specs=(P(None, "tp"), P("tp", None)),
                check_vma=False,
            )(x_sh)

        xq, xs = _quant_per_die(x_bf16)
        tiles = Fp8MatmulTileSizes(align_to(max(8, batch), 8), 128, 128, 2)
        interp_params = pltpu.InterpretParams(detect_races=True)

        xq_got, xs_got, res_got = sharded_fp8_row_matmul_async_ar(
            xq,
            wq,
            ws,
            mesh=mesh,
            x_scale=xs,
            sync_tp_scale=False,
            residual=res_bf16,
            norm_weight=norm_w,
            eps=1e-6,
            quantize_output_dtype=jnp.float8_e4m3fn,
            transposed_scale=True,
            tile_sizes=tiles,
            backend="pallas",
            interpret=interp_params,
        )
        assert interpret_pallas_call.races is not None
        assert not interpret_pallas_call.races.races_found

        delta_ref = sharded_fp8_row_matmul_async_ar(
            xq,
            wq,
            ws,
            mesh=mesh,
            x_scale=xs,
            sync_tp_scale=False,
            tile_sizes=tiles,
            backend="pallas",
            interpret=True,
        )
        xq_ref, xs_ref, res_ref = jax.shard_map(
            lambda dl, rl, nw: fused_add_rms_norm_fp8_quant(
                dl, rl, nw, 1e-6, transposed_scale=True, interpret=True
            ),
            mesh=mesh,
            in_specs=(P(None, None), P(None, None), P()),
            out_specs=(P(None, None), P(None, None), P(None, None)),
            check_vma=False,
        )(delta_ref, res_bf16, norm_w)

        np.testing.assert_array_equal(
            np.asarray(res_got.addressable_shards[0].data),
            np.asarray(res_got.addressable_shards[1].data),
        )
        np.testing.assert_array_equal(
            np.asarray(xq_got.addressable_shards[0].data),
            np.asarray(xq_got.addressable_shards[1].data),
        )

        k_half = k_global // 2
        d0 = fp8_matmul_prequantized(
            xq[:, :k_half], xs[0].reshape(batch, 1), w.qvalue[:, :k_half], w.scale
        )
        d1 = fp8_matmul_prequantized(
            xq[:, k_half:], xs[1].reshape(batch, 1), w.qvalue[:, k_half:], w.scale
        )
        exact_next_res = (
            d0.astype(jnp.float32)
            + d1.astype(jnp.float32)
            + res_bf16.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        np.testing.assert_allclose(
            np.asarray(res_got, np.float32),
            np.asarray(exact_next_res, np.float32),
            rtol=1e-2,
            atol=2e-3,
        )
        np.testing.assert_array_equal(np.asarray(res_got), np.asarray(res_ref))
        np.testing.assert_array_equal(np.asarray(xs_got), np.asarray(xs_ref))
        np.testing.assert_array_equal(np.asarray(xq_got), np.asarray(xq_ref))

        if batch == 8 and (n_dim, k_global) == (256, 256):
            fb_q, fb_s, fb_r = sharded_fp8_row_matmul_async_ar(
                xq,
                wq,
                ws,
                mesh=mesh,
                x_scale=xs,
                sync_tp_scale=False,
                residual=res_bf16,
                norm_weight=norm_w,
                eps=1e-6,
                quantize_output_dtype=jnp.float8_e4m3fn,
                transposed_scale=True,
                backend="xla",
            )
            np.testing.assert_allclose(
                np.asarray(fb_r, np.float32),
                np.asarray(res_got, np.float32),
                rtol=1e-2,
                atol=1e-2,
            )
            assert fb_q.shape == (batch, n_dim)
            assert fb_s.shape == (1, batch)

    def test_async_row_all_reduce_vmem_limit_tile_validation_and_fallbacks(
        self,
    ) -> None:
        from unittest import mock

        from resin.backend.tpu.kernels import (
            MatmulBackend,
            estimate_async_ar_vmem_bytes,
            get_async_ar_tile_sizes,
            pallas_fp8_row_matmul_async_ar,
            quantize_fp8_act,
            sharded_fp8_row_matmul_async_ar,
        )
        from resin.backend.tpu.kernels.collective_fp8_matmul import (
            _fp8_row_matmul_async_ar_kernel,
        )

        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("requires at least 2 devices for tp=2 async all-reduce")

        mesh_tp2 = make_mesh(2, devices[:2])
        mesh_tp1 = make_mesh(1, devices[:1])
        rng = np.random.default_rng(2027)

        # 1. Dynamic vmem_limit_bytes sizing (estimate + 4 MiB slack, [8 MiB, 48 MiB] clamp).
        captured_limits: list[int | None] = []

        def _spy_pallas_call(
            *args: object,
            out_shape: jax.ShapeDtypeStruct,
            compiler_params: object = None,
            **kwargs: object,
        ):
            captured_limits.append(getattr(compiler_params, "vmem_limit_bytes", None))
            return lambda *k_args: jnp.zeros(out_shape.shape, dtype=out_shape.dtype)

        with (
            mock.patch(
                "resin.backend.tpu.kernels.collective_fp8_matmul.get_tpu_version",
                return_value=7,
            ),
            mock.patch(
                "resin.backend.tpu.kernels.collective_fp8_matmul.pl.pallas_call",
                side_effect=_spy_pallas_call,
            ),
            mock.patch(
                "resin.backend.tpu.kernels.collective_fp8_matmul.pltpu.with_memory_space_constraint",
                side_effect=lambda arr, _ms: arr,
            ),
        ):
            for bs, n_dim, k_local, tiles_override, exact_clamp in (
                (8, 256, 128, None, 8 * 1024 * 1024),  # tiny -> 8 MiB floor
                (160, 5120, 4096, None, None),  # o_proj B=160 (~19.07 MiB)
                (320, 5120, 4096, None, None),  # o_proj B=320 (~24.64 MiB)
                (160, 5120, 12800, None, None),  # down_proj B=160 (~29.56 MiB)
                (320, 5120, 12800, None, None),  # down_proj B=320 (~36.04 MiB)
                (
                    320,
                    4096,
                    9216,
                    Fp8MatmulTileSizes(320, 1024, 9216, 4),
                    48 * 1024 * 1024,
                ),  # 46.33 MiB + 4 MiB -> clamped to 48 MiB
            ):
                captured_limits.clear()
                eff = (
                    get_async_ar_tile_sizes(bs, n_dim, k_local)
                    if tiles_override is None
                    else tiles_override
                )
                n_out = n_dim // eff.out_block_size
                est = estimate_async_ar_vmem_bytes(
                    eff.batch_block_size,
                    eff.out_block_size,
                    eff.in_block_size,
                    n_out,
                    min(eff.w_buffer_count, n_out),
                )
                expected_limit = min(
                    max(est + 4 * 1024 * 1024, 8 * 1024 * 1024), 48 * 1024 * 1024
                )

                @jax.jit
                def _trace_vmem() -> jax.Array:
                    xq = jnp.zeros((bs, 2 * k_local), jnp.float8_e4m3fn)
                    xs = jnp.ones((2, bs), jnp.float32)
                    wq = jnp.zeros((n_dim, 2 * k_local), jnp.float8_e4m3fn)
                    ws = jnp.ones((n_dim,), jnp.float32)
                    return jax.shard_map(
                        lambda xql, xsl, wql, wsl: pallas_fp8_row_matmul_async_ar(
                            xql, xsl, wql, wsl, tile_sizes=tiles_override
                        ),
                        mesh=mesh_tp2,
                        in_specs=(P(None, "tp"), P("tp", None), P(None, "tp"), P()),
                        out_specs=P(None, None),
                        check_vma=False,
                    )(xq, xs, wq, ws)

                _ = _trace_vmem.lower()
                assert captured_limits == [expected_limit]
                if exact_clamp is not None:
                    assert captured_limits == [exact_clamp]
                if tiles_override is None and n_dim == 5120:
                    assert expected_limit < 40 * 1024 * 1024

        # 2. Explicit tile_sizes VMEM > 48 MiB and geometry validation.
        strict_backends: tuple[MatmulBackend, ...] = ("auto", "pallas")
        for bad_tiles in (
            Fp8MatmulTileSizes(320, 2048, 12800, 2),  # > 48 MiB VMEM
            Fp8MatmulTileSizes(8, 100, 128, 2),  # unaligned out_block
            Fp8MatmulTileSizes(0, 128, 128, 2),  # non-positive batch_block
            Fp8MatmulTileSizes(8, 128, 128, 1),  # w_buffer_count < 2
            Fp8MatmulTileSizes(8, 256, 128, 2),  # n_out == 1 for N=256
            Fp8MatmulTileSizes(8, 128, 128, 2),  # multi-K tile for local_k=256
        ):
            n_bad = 4096 if bad_tiles.out_block_size == 2048 else 256
            k_local_bad = (
                12800
                if bad_tiles.in_block_size == 12800
                else (256 if bad_tiles == Fp8MatmulTileSizes(8, 128, 128, 2) else 128)
            )
            bs_bad = max(8, bad_tiles.batch_block_size)
            xq_b = jnp.zeros((bs_bad, 2 * k_local_bad), jnp.float8_e4m3fn)
            xs_b = jnp.ones((2, bs_bad), jnp.float32)
            wq_b = jnp.zeros((n_bad, 2 * k_local_bad), jnp.float8_e4m3fn)
            ws_b = jnp.ones((n_bad,), jnp.float32)
            for be in strict_backends:
                with pytest.raises(ValueError):
                    jax.jit(
                        lambda: sharded_fp8_row_matmul_async_ar(
                            xq_b,
                            wq_b,
                            ws_b,
                            mesh=mesh_tp2,
                            x_scale=xs_b,
                            sync_tp_scale=False,
                            tile_sizes=bad_tiles,
                            backend=be,
                            interpret=True,
                        )
                    ).lower()

        # 3. Consistent tp != 2 and unsupported-platform fallbacks (with and without tile_sizes).
        x_tp = jnp.asarray(rng.standard_normal((8, 256)) * 0.3, jnp.bfloat16)
        w_tp = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 256)) * 0.03, jnp.bfloat16)
        )
        ref_tp = xla_fp8_matmul(x_tp, w_tp.qvalue, w_tp.scale)

        for tiles_arg in (None, Fp8MatmulTileSizes(8, 128, 128, 2)):
            # tp=1 + backend="auto" falls back cleanly whether or not tile_sizes is passed.
            y_tp1_auto = sharded_fp8_row_matmul_async_ar(
                jax.device_put(x_tp, NamedSharding(mesh_tp1, P(None, "tp"))),
                jax.device_put(w_tp.qvalue, NamedSharding(mesh_tp1, P(None, "tp"))),
                jax.device_put(w_tp.scale, NamedSharding(mesh_tp1, P())),
                mesh=mesh_tp1,
                tile_sizes=tiles_arg,
                backend="auto",
                interpret=True,
            )
            np.testing.assert_allclose(
                np.asarray(y_tp1_auto, np.float32),
                np.asarray(ref_tp, np.float32),
                rtol=1e-2,
                atol=2e-2,
            )

            # tp=1 + backend="pallas" raises ValueError whether or not tile_sizes is passed.
            with pytest.raises(ValueError, match="tp == 2"):
                sharded_fp8_row_matmul_async_ar(
                    jax.device_put(x_tp, NamedSharding(mesh_tp1, P(None, "tp"))),
                    jax.device_put(w_tp.qvalue, NamedSharding(mesh_tp1, P(None, "tp"))),
                    jax.device_put(w_tp.scale, NamedSharding(mesh_tp1, P())),
                    mesh=mesh_tp1,
                    tile_sizes=tiles_arg,
                    backend="pallas",
                    interpret=True,
                )

            # Unsupported platform (tpu_version=0, interpret=False) on tp=2:
            # backend="auto" falls back cleanly; backend="pallas" raises ValueError.
            with mock.patch(
                "resin.backend.tpu.kernels.collective_fp8_matmul.get_tpu_version",
                return_value=0,
            ):
                y_cpu_auto = sharded_fp8_row_matmul_async_ar(
                    jax.device_put(x_tp, NamedSharding(mesh_tp2, P(None, "tp"))),
                    jax.device_put(w_tp.qvalue, NamedSharding(mesh_tp2, P(None, "tp"))),
                    jax.device_put(w_tp.scale, NamedSharding(mesh_tp2, P())),
                    mesh=mesh_tp2,
                    tile_sizes=tiles_arg,
                    backend="auto",
                    interpret=False,
                )
                np.testing.assert_allclose(
                    np.asarray(y_cpu_auto, np.float32),
                    np.asarray(ref_tp, np.float32),
                    rtol=1e-2,
                    atol=2e-2,
                )
                with pytest.raises(ValueError, match="TPU7x"):
                    sharded_fp8_row_matmul_async_ar(
                        jax.device_put(x_tp, NamedSharding(mesh_tp2, P(None, "tp"))),
                        jax.device_put(
                            w_tp.qvalue, NamedSharding(mesh_tp2, P(None, "tp"))
                        ),
                        jax.device_put(w_tp.scale, NamedSharding(mesh_tp2, P())),
                        mesh=mesh_tp2,
                        tile_sizes=tiles_arg,
                        backend="pallas",
                        interpret=False,
                    )

        # n_out == 1 (orig_n = 128, tile_sizes=None): "auto" falls back, "pallas" raises.
        w_n1 = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((128, 256)) * 0.03, jnp.bfloat16)
        )
        y_n1_auto = sharded_fp8_row_matmul_async_ar(
            jax.device_put(x_tp, NamedSharding(mesh_tp2, P(None, "tp"))),
            jax.device_put(w_n1.qvalue, NamedSharding(mesh_tp2, P(None, "tp"))),
            jax.device_put(w_n1.scale, NamedSharding(mesh_tp2, P())),
            mesh=mesh_tp2,
            backend="auto",
            interpret=True,
        )
        np.testing.assert_allclose(
            np.asarray(y_n1_auto, np.float32),
            np.asarray(xla_fp8_matmul(x_tp, w_n1.qvalue, w_n1.scale), np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
        with pytest.raises(ValueError, match="at least 2 output tiles"):
            sharded_fp8_row_matmul_async_ar(
                jax.device_put(x_tp, NamedSharding(mesh_tp2, P(None, "tp"))),
                jax.device_put(w_n1.qvalue, NamedSharding(mesh_tp2, P(None, "tp"))),
                jax.device_put(w_n1.scale, NamedSharding(mesh_tp2, P())),
                mesh=mesh_tp2,
                backend="pallas",
                interpret=True,
            )

        # 4. Direct pallas_fp8_row_matmul_async_ar and kernel body assert axis_size == 2.
        xq_1, xs_1 = quantize_fp8_act(x_tp[:, :128])
        with pytest.raises(ValueError, match="mapped axis"):
            pallas_fp8_row_matmul_async_ar(
                xq_1, xs_1, w_tp.qvalue[:, :128], w_tp.scale, interpret=True
            )
        with pytest.raises(ValueError, match="axis_size"):
            jax.shard_map(
                lambda xql, xsl, wql, wsl: pallas_fp8_row_matmul_async_ar(
                    xql, xsl, wql, wsl, interpret=True
                ),
                mesh=mesh_tp1,
                in_specs=(P(None, "tp"), P(None, None), P(None, "tp"), P()),
                out_specs=P(None, None),
                check_vma=False,
            )(
                jax.device_put(xq_1, NamedSharding(mesh_tp1, P(None, "tp"))),
                jax.device_put(xs_1, NamedSharding(mesh_tp1, P(None, None))),
                jax.device_put(
                    w_tp.qvalue[:, :128], NamedSharding(mesh_tp1, P(None, "tp"))
                ),
                jax.device_put(w_tp.scale, NamedSharding(mesh_tp1, P())),
            )

        def _bad_kernel(dummy: jax.Array) -> jax.Array:
            _fp8_row_matmul_async_ar_kernel(
                *(None,) * 13,
                out_block=128,
                n_out=2,
                n_w_slots=2,
                tp_axis="tp",
                eff_reduce=jnp.dtype(jnp.bfloat16),
                target_out_dtype=jnp.dtype(jnp.bfloat16),
            )
            return dummy

        with pytest.raises(ValueError, match="axis_size"):
            jax.shard_map(
                _bad_kernel,
                mesh=mesh_tp1,
                in_specs=P(None),
                out_specs=P(None),
                check_vma=False,
            )(
                jax.device_put(
                    jnp.zeros((1,), jnp.float32), NamedSharding(mesh_tp1, P(None))
                )
            )


@pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="requires TPU Mosaic hardware"
)
class TestTpuHardwareMosaic:
    def test_qwen3_tp2_mosaic_rpa_v3_fp8_kv_parity_and_zero_recompile(self) -> None:
        assert default_impl() is ragged_paged_attention
        mesh = make_mesh(min(len(jax.devices()), 2))
        tp = int(mesh.shape["tp"])
        spec = KvCacheSpec(
            layers=1,
            kv_heads=8,
            head_dim=128,
            page_tokens=256,
            dtype=jnp.float8_e4m3fn,
        )
        cache_sharding = NamedSharding(mesh, KV_PARTITION)
        head_sharding = NamedSharding(mesh, P(None, "tp", None))
        rep_sharding = NamedSharding(mesh, P())

        rng = np.random.default_rng(777)
        # Fill arena with NaNs outside null page 0 to prove Mosaic masks stale tails.
        init_cache = np.full((8, *spec.page_shape(tp)), np.nan, dtype=np.float32)
        init_cache[0] = 0.0
        cache = jax.device_put(
            jnp.asarray(init_cache, jnp.float8_e4m3fn), cache_sharding
        )
        ref_cache = jax.device_put(
            jnp.asarray(init_cache, jnp.float8_e4m3fn), cache_sharding
        )

        @functools.partial(
            jax.jit,
            donate_argnums=(3,),
            out_shardings=(head_sharding, cache_sharding),
        )
        def mosaic_step(q, k, v, kv_cache, meta):
            return paged_attention(
                q,
                k,
                v,
                kv_cache,
                meta,
                mesh=mesh,
                sm_scale=128**-0.5,
                k_scale=0.5,
                v_scale=0.5,
            )

        def ref_step(q, k, v, kv_cache, meta):
            return paged_attention(
                q,
                k,
                v,
                kv_cache,
                meta,
                mesh=mesh,
                sm_scale=128**-0.5,
                k_scale=0.5,
                v_scale=0.5,
                impl=reference.ragged_paged_attention,
            )

        # Prefill bucket (tokens=512, seqs=4, pages_per_seq=4):
        # Seq 0 has 260 tokens (crosses page 256 boundary across pages 3 and 5);
        # Seq 1 has 45 tokens on page 2.
        q_pre = jax.device_put(
            jnp.asarray(rng.standard_normal((512, 64, 128)), jnp.bfloat16),
            head_sharding,
        )
        k_pre = jax.device_put(
            jnp.asarray(rng.standard_normal((512, 8, 128)), jnp.bfloat16),
            head_sharding,
        )
        v_pre = jax.device_put(
            jnp.asarray(rng.standard_normal((512, 8, 128)), jnp.bfloat16),
            head_sharding,
        )
        pages = np.zeros((4, 4), dtype=np.int32)
        pages[0, :2] = [3, 5]
        pages[1, :1] = [2]
        meta_pre = AttentionMetadata(
            jax.device_put(jnp.array([260, 45, 0, 0], jnp.int32), rep_sharding),
            (jax.device_put(jnp.asarray(pages.reshape(-1)), rep_sharding),),
            jax.device_put(jnp.array([0, 260, 305, 305, 305], jnp.int32), rep_sharding),
            jax.device_put(jnp.array([0, 0, 2], jnp.int32), rep_sharding),
        )

        out_pre, cache = mosaic_step(q_pre, k_pre, v_pre, cache, meta_pre)
        ref_out_pre, ref_cache = ref_step(q_pre, k_pre, v_pre, ref_cache, meta_pre)
        out_pre.block_until_ready()
        assert isinstance(cache.sharding, NamedSharding)
        assert cache.sharding.spec == KV_PARTITION
        np.testing.assert_allclose(
            np.asarray(out_pre[:305], np.float32),
            np.asarray(ref_out_pre[:305], np.float32),
            rtol=3e-2,
            atol=5e-2,
        )

        # Decode bucket (tokens=8, seqs=4): warm once, then run unseen lengths/pages
        # with CompilationCounter asserting zero recompilations.
        q_dec = jax.device_put(
            jnp.asarray(rng.standard_normal((8, 64, 128)), jnp.bfloat16), head_sharding
        )
        k_dec = jax.device_put(
            jnp.asarray(rng.standard_normal((8, 8, 128)), jnp.bfloat16), head_sharding
        )
        v_dec = jax.device_put(
            jnp.asarray(rng.standard_normal((8, 8, 128)), jnp.bfloat16), head_sharding
        )

        counter = CompilationCounter()
        try:
            meta_dec1 = AttentionMetadata(
                jax.device_put(jnp.array([261, 46, 0, 0], jnp.int32), rep_sharding),
                (jax.device_put(jnp.asarray(pages.reshape(-1)), rep_sharding),),
                jax.device_put(jnp.array([0, 1, 2, 2, 2], jnp.int32), rep_sharding),
                jax.device_put(jnp.array([2, 2, 2], jnp.int32), rep_sharding),
            )
            out_d1, cache = mosaic_step(q_dec, k_dec, v_dec, cache, meta_dec1)
            out_d1.block_until_ready()
            ref_d1, ref_cache = ref_step(q_dec, k_dec, v_dec, ref_cache, meta_dec1)
            warm = counter.count

            # Unseen kv_lens and active row count (1 row instead of 2) in same bucket.
            meta_dec2 = AttentionMetadata(
                jax.device_put(jnp.array([262, 0, 0, 0], jnp.int32), rep_sharding),
                (jax.device_put(jnp.asarray(pages.reshape(-1)), rep_sharding),),
                jax.device_put(jnp.array([0, 1, 1, 1, 1], jnp.int32), rep_sharding),
                jax.device_put(jnp.array([1, 1, 1], jnp.int32), rep_sharding),
            )
            out_d2, cache = mosaic_step(q_dec, k_dec, v_dec, cache, meta_dec2)
            out_d2.block_until_ready()
            assert counter.count == warm

            ref_d2, _ = ref_step(q_dec, k_dec, v_dec, ref_cache, meta_dec2)
            np.testing.assert_allclose(
                np.asarray(out_d1[:2], np.float32),
                np.asarray(ref_d1[:2], np.float32),
                rtol=3e-2,
                atol=5e-2,
            )
            np.testing.assert_allclose(
                np.asarray(out_d2[:1], np.float32),
                np.asarray(ref_d2[:1], np.float32),
                rtol=3e-2,
                atol=5e-2,
            )
        finally:
            counter.close()

    @pytest.mark.parametrize("kv_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
    def test_qwen3_tp2_mosaic_rpa_v3_tuned_decode_blocks_and_zero_recompile(
        self, kv_dtype: jnp.dtype
    ) -> None:
        mesh = make_mesh(min(len(jax.devices()), 2))
        tp = int(mesh.shape["tp"])
        spec = KvCacheSpec(
            layers=1,
            kv_heads=8,
            head_dim=128,
            page_tokens=256,
            dtype=kv_dtype,
        )
        cache_sharding = NamedSharding(mesh, KV_PARTITION)
        head_sharding = NamedSharding(mesh, P(None, "tp", None))
        rep_sharding = NamedSharding(mesh, P())
        is_fp8 = kv_dtype == jnp.float8_e4m3fn
        k_scale = 0.5 if is_fp8 else None
        v_scale = 0.25 if is_fp8 else None
        rtol = 3e-2 if is_fp8 else 1e-2
        atol = 5e-2 if is_fp8 else 2e-2

        rng = np.random.default_rng(2026)

        @functools.partial(
            jax.jit,
            donate_argnums=(3,),
            out_shardings=(head_sharding, cache_sharding),
        )
        def mosaic_decode(q, k, v, kv_cache, meta):
            return paged_attention(
                q,
                k,
                v,
                kv_cache,
                meta,
                mesh=mesh,
                sm_scale=128**-0.5,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        def ref_decode(q, k, v, kv_cache, meta):
            return paged_attention(
                q,
                k,
                v,
                kv_cache,
                meta,
                mesh=mesh,
                sm_scale=128**-0.5,
                k_scale=k_scale,
                v_scale=v_scale,
                impl=reference.ragged_paged_attention,
            )

        for pages_per_seq, total_pages, step_kv_lens in (
            (9, 24, ([255, 1024], [256, 1536], [257, 1537], [2048, 0])),
            (37, 48, ([4096, 1024], [4097, 0])),
        ):
            global_cache_shape = (total_pages, *spec.page_shape(tp))
            init_np = rng.standard_normal(global_cache_shape).astype(np.float32) * 0.25
            init_cache = quantize_kv(jnp.asarray(init_np, jnp.float32), kv_dtype)
            nan_page = np.asarray(
                jnp.full(global_cache_shape[1:], jnp.nan, dtype=kv_dtype)
            )
            init_cache_np = np.array(init_cache, copy=True)
            init_cache_np[0] = 0
            # Point unused table entries and inactive rows at NaN-filled unowned pages.
            pages = np.full((4, pages_per_seq), total_pages - 1, dtype=np.int32)
            if pages_per_seq == 9:
                pages[0, :8] = np.arange(1, 9)
                pages[1, :8] = np.arange(9, 17)
                # Page 15 (seq 1's 7th page) has NaNs in slots 1..255 during kv_len=1537.
                init_cache_np[15, 1:] = nan_page[1:]
                init_cache_np[17:] = nan_page
            else:
                pages[0, :17] = np.arange(1, 18)
                pages[1, :5] = np.arange(18, 23)
                init_cache_np[17, 1:] = nan_page[1:]
                init_cache_np[22:] = nan_page

            cache = jax.device_put(jnp.asarray(init_cache_np), cache_sharding)
            ref_cache = jax.device_put(jnp.asarray(init_cache_np), cache_sharding)

            step_inputs = []
            for kv_pair in step_kv_lens:
                active = sum(1 for x in kv_pair if x > 0)
                q_d = jax.device_put(
                    jnp.asarray(rng.standard_normal((8, 64, 128)), jnp.bfloat16),
                    head_sharding,
                )
                k_d = jax.device_put(
                    jnp.asarray(rng.standard_normal((8, 8, 128)), jnp.bfloat16),
                    head_sharding,
                )
                v_d = jax.device_put(
                    jnp.asarray(rng.standard_normal((8, 8, 128)), jnp.bfloat16),
                    head_sharding,
                )
                cu = np.array(
                    [0] + [min(i, active) for i in range(1, 5)], dtype=np.int32
                )
                meta = AttentionMetadata(
                    jax.device_put(
                        np.array([kv_pair[0], kv_pair[1], 0, 0], dtype=np.int32),
                        rep_sharding,
                    ),
                    (jax.device_put(pages.reshape(-1), rep_sharding),),
                    jax.device_put(cu, rep_sharding),
                    jax.device_put(
                        np.array([active, active, active], dtype=np.int32),
                        rep_sharding,
                    ),
                )
                step_inputs.append((active, q_d, k_d, v_d, meta))

            mosaic_outs = []
            counter = CompilationCounter()
            try:
                warm = None
                for active, q_d, k_d, v_d, meta in step_inputs:
                    out_d, cache = mosaic_decode(q_d, k_d, v_d, cache, meta)
                    out_d.block_until_ready()
                    if warm is None:
                        warm = counter.count
                    else:
                        assert counter.count == warm
                    mosaic_outs.append((active, np.asarray(out_d[:active], np.float32)))
            finally:
                counter.close()

            for (active, out_arr), (_, q_d, k_d, v_d, meta) in zip(
                mosaic_outs, step_inputs, strict=True
            ):
                ref_d, ref_cache = ref_decode(q_d, k_d, v_d, ref_cache, meta)
                assert not np.any(np.isnan(out_arr))
                np.testing.assert_allclose(
                    out_arr,
                    np.asarray(ref_d[:active], np.float32),
                    rtol=rtol,
                    atol=atol,
                )

    def test_qwen3_tp2_mosaic_fp8_w8a8_matmul_parity(self) -> None:
        mesh = make_mesh(min(len(jax.devices()), 2))
        rng = np.random.default_rng(888)
        x_np = rng.standard_normal((32, 5120)).astype(np.float32) * 0.5
        x_np[24:] = 0.0  # zero-padded bucket tail
        w_col_np = rng.standard_normal((4096, 5120)).astype(np.float32) * 0.02
        w_row_np = rng.standard_normal((5120, 4096)).astype(np.float32) * 0.02

        x = jnp.asarray(x_np, jnp.bfloat16)
        fw_col = quantize_fp8_weight(jnp.asarray(w_col_np, jnp.bfloat16))
        fw_row = quantize_fp8_weight(jnp.asarray(w_row_np, jnp.bfloat16))

        col_pallas = jax.jit(
            lambda inp, wq, ws: sharded_fp8_matmul(
                inp, wq, ws, P("tp", None), mesh=mesh, backend="pallas"
            )
        )(
            jax.device_put(x, NamedSharding(mesh, P(None, None))),
            jax.device_put(fw_col.qvalue, NamedSharding(mesh, P("tp", None))),
            jax.device_put(fw_col.scale, NamedSharding(mesh, P("tp"))),
        )
        col_xla = xla_fp8_matmul(x, fw_col.qvalue, fw_col.scale)
        assert not np.any(np.isnan(np.asarray(col_pallas, np.float32)))
        np.testing.assert_allclose(
            np.asarray(col_pallas, np.float32),
            np.asarray(col_xla, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )

        row_pallas = jax.jit(
            lambda inp, wq, ws: sharded_fp8_matmul(
                inp, wq, ws, P(None, "tp"), mesh=mesh, backend="pallas"
            )
        )(
            jax.device_put(col_pallas, NamedSharding(mesh, P(None, "tp"))),
            jax.device_put(fw_row.qvalue, NamedSharding(mesh, P(None, "tp"))),
            jax.device_put(fw_row.scale, NamedSharding(mesh, P())),
        )
        row_xla = xla_fp8_matmul(col_pallas, fw_row.qvalue, fw_row.scale)
        assert not np.any(np.isnan(np.asarray(row_pallas, np.float32)))
        np.testing.assert_allclose(
            np.asarray(row_pallas, np.float32),
            np.asarray(row_xla, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )

        for small, live in ((8, 1), (16, 9), (320, 289)):
            x_small_np = rng.standard_normal((small, 5120)).astype(np.float32) * 0.5
            x_small_np[live:] = 0.0
            x_small = jnp.asarray(x_small_np, jnp.bfloat16)
            small_pallas = jax.jit(
                lambda inp, wq, ws: sharded_fp8_matmul(
                    inp, wq, ws, P("tp", None), mesh=mesh, backend="pallas"
                )
            )(
                jax.device_put(x_small, NamedSharding(mesh, P(None, None))),
                jax.device_put(fw_col.qvalue, NamedSharding(mesh, P("tp", None))),
                jax.device_put(fw_col.scale, NamedSharding(mesh, P("tp"))),
            )
            small_xla = xla_fp8_matmul(x_small, fw_col.qvalue, fw_col.scale)
            assert not np.any(np.isnan(np.asarray(small_pallas, np.float32)))
            np.testing.assert_allclose(
                np.asarray(small_pallas, np.float32)[live:], 0.0, atol=0.0
            )
            np.testing.assert_allclose(
                np.asarray(small_pallas, np.float32),
                np.asarray(small_xla, np.float32),
                rtol=2e-2,
                atol=5e-2,
            )
            small_row = jax.jit(
                lambda inp, wq, ws: sharded_fp8_matmul(
                    inp,
                    wq,
                    ws,
                    P(None, "tp"),
                    mesh=mesh,
                    sync_tp_scale=False,
                    backend="pallas",
                )
            )(
                jax.device_put(small_pallas, NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(fw_row.qvalue, NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(fw_row.scale, NamedSharding(mesh, P())),
            )
            small_row_xla = jax.jit(
                lambda inp, wq, ws: sharded_fp8_matmul(
                    inp,
                    wq,
                    ws,
                    P(None, "tp"),
                    mesh=mesh,
                    sync_tp_scale=False,
                    backend="xla",
                )
            )(
                jax.device_put(small_pallas, NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(fw_row.qvalue, NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(fw_row.scale, NamedSharding(mesh, P())),
            )
            assert not np.any(np.isnan(np.asarray(small_row, np.float32)))
            np.testing.assert_allclose(
                np.asarray(small_row, np.float32)[live:], 0.0, atol=0.0
            )
            np.testing.assert_allclose(
                np.asarray(small_row, np.float32),
                np.asarray(small_row_xla, np.float32),
                rtol=2e-2,
                atol=5e-2,
            )

        # Also exercise Pallas rms_norm_fp8_quant and Qwen3-32B MLP gate_up shape (256, 5120) -> 25600/die.
        norm_w = jnp.ones((5120,), dtype=jnp.bfloat16)
        xq_mosaic, xs_mosaic = jax.jit(rms_norm_fp8_quant)(x, norm_w)
        fused_col = fp8_matmul_prequantized(
            xq_mosaic, xs_mosaic, fw_col.qvalue, fw_col.scale
        )
        unfused_col = xla_fp8_matmul(rms_norm(x, norm_w), fw_col.qvalue, fw_col.scale)
        np.testing.assert_allclose(
            np.asarray(fused_col, np.float32),
            np.asarray(unfused_col, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )

        x_mlp = jnp.asarray(
            rng.standard_normal((256, 5120)).astype(np.float32) * 0.5, jnp.bfloat16
        )
        fw_mlp = quantize_fp8_weight(
            jnp.asarray(
                rng.standard_normal((25600, 5120)).astype(np.float32) * 0.02,
                jnp.bfloat16,
            )
        )
        mlp_pallas = jax.jit(
            lambda inp, wq, ws: fp8_matmul(inp, wq, ws, backend="pallas")
        )(x_mlp, fw_mlp.qvalue, fw_mlp.scale)
        mlp_xla = xla_fp8_matmul(x_mlp, fw_mlp.qvalue, fw_mlp.scale)
        np.testing.assert_allclose(
            np.asarray(mlp_pallas, np.float32),
            np.asarray(mlp_xla, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )

    def test_qwen3_tp2_mosaic_prefill_fused_kernels_parity(self) -> None:
        from resin.backend.tpu.attention import (
            KV_MERGED_PARTITION,
            Q_PACKED_PARTITION,
        )
        from resin.backend.tpu.kernels import (
            fused_add_rms_norm_fp8_quant,
            pallas_qkv_norm_rope_pack,
            pallas_unpack_attn_fp8_quant,
            prepare_inputs,
            quantize_fp8_act,
            sharded_fp8_gate_up_swiglu,
            sharded_fp8_matmul_all_gather,
            sharded_fp8_matmul_reduce_scatter,
        )

        mesh = make_mesh(min(len(jax.devices()), 2))
        tp = int(mesh.shape["tp"])
        rng = np.random.default_rng(2027)
        tokens = 1024

        # 1. Fused residual + RMSNorm + FP8 quant on [tokens // tp, 5120]
        x = jax.device_put(
            jnp.asarray(rng.standard_normal((tokens, 5120)) * 0.5, jnp.bfloat16),
            NamedSharding(mesh, P("tp", None)),
        )
        res = jax.device_put(
            jnp.asarray(rng.standard_normal((tokens, 5120)) * 0.5, jnp.bfloat16),
            NamedSharding(mesh, P("tp", None)),
        )
        nw = jax.device_put(jnp.ones((5120,), jnp.bfloat16), NamedSharding(mesh, P()))

        xq_sh, xs_sh, next_res = jax.jit(
            jax.shard_map(
                lambda xl, rl, w: fused_add_rms_norm_fp8_quant(xl, rl, w, 1e-6),
                mesh=mesh,
                in_specs=(P("tp", None), P("tp", None), P()),
                out_specs=(P("tp", None), P("tp", None), P("tp", None)),
                check_vma=False,
            )
        )(x, res, nw)
        exp_res = (np.asarray(x, np.float32) + np.asarray(res, np.float32)).astype(
            jnp.bfloat16
        )
        np.testing.assert_array_equal(np.asarray(next_res), exp_res)

        # 2. Fused QKV all_gather + pallas_qkv_norm_rope_pack + prepared RPA
        w_qkv = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((10240, 5120)) * 0.02, jnp.bfloat16)
        )
        qkv_out = jax.jit(
            lambda xq_i, xs_i, wq, ws: sharded_fp8_matmul_all_gather(
                xq_i, xs_i, wq, ws, mesh=mesh
            )
        )(
            xq_sh,
            xs_sh,
            jax.device_put(w_qkv.qvalue, NamedSharding(mesh, P("tp", None))),
            jax.device_put(w_qkv.scale, NamedSharding(mesh, P("tp"))),
        )

        rc = build_rope_cache(2048, 128)
        rc_rep = (
            jax.device_put(rc[0], NamedSharding(mesh, P())),
            jax.device_put(rc[1], NamedSharding(mesh, P())),
        )
        pos = jax.device_put(
            jnp.arange(tokens, dtype=jnp.int32), NamedSharding(mesh, P())
        )
        qnw = jax.device_put(jnp.ones((128,), jnp.bfloat16), NamedSharding(mesh, P()))
        knw = jax.device_put(jnp.ones((128,), jnp.bfloat16), NamedSharding(mesh, P()))

        qp, kvm = jax.jit(
            jax.shard_map(
                lambda qkv_l, p, c, s, qw, kw: pallas_qkv_norm_rope_pack(
                    qkv_l,
                    p,
                    (c, s),
                    qw,
                    kw,
                    jnp.float8_e4m3fn,
                    num_q_heads=64 // tp,
                    num_kv_heads=8 // tp,
                ),
                mesh=mesh,
                in_specs=(P(None, "tp"), P(), P(), P(), P(), P()),
                out_specs=(Q_PACKED_PARTITION, KV_MERGED_PARTITION),
                check_vma=False,
            )
        )(qkv_out, pos, rc_rep[0], rc_rep[1], qnw, knw)

        # Direct element-for-element parity of qp/kvm against prepare_inputs(q_ref, k_ref, v_ref).
        q_heads_local = 64 // tp
        kv_heads_local = 8 // tp
        qkv_host = np.asarray(qkv_out, np.float32)
        q_chunks, k_chunks, v_chunks = [], [], []
        local_w = (q_heads_local + 2 * kv_heads_local) * 128
        for s_idx in range(tp):
            sl = jnp.asarray(
                qkv_host[:, s_idx * local_w : (s_idx + 1) * local_w], jnp.bfloat16
            )
            qs, ks, vs = jnp.split(
                sl,
                [q_heads_local * 128, (q_heads_local + kv_heads_local) * 128],
                axis=-1,
            )
            q_chunks.append(qs.reshape(tokens, q_heads_local, 128))
            k_chunks.append(ks.reshape(tokens, kv_heads_local, 128))
            v_chunks.append(vs.reshape(tokens, kv_heads_local, 128))
        q_full = jnp.concatenate(q_chunks, axis=1)
        k_full = jnp.concatenate(k_chunks, axis=1)
        v_full = jnp.concatenate(v_chunks, axis=1)
        pos_host = jnp.arange(tokens, dtype=jnp.int32)
        q_ref_full = apply_rope(
            rms_norm(q_full, jnp.ones((128,), jnp.bfloat16), 1e-6),
            pos_host,
            128,
            cos_sin_cache=rc,
        )
        k_ref_full = quantize_kv(
            apply_rope(
                rms_norm(k_full, jnp.ones((128,), jnp.bfloat16), 1e-6),
                pos_host,
                128,
                cos_sin_cache=rc,
            ),
            jnp.float8_e4m3fn,
        )
        v_ref_full = quantize_kv(v_full, jnp.float8_e4m3fn)
        exp_qp_full, exp_kvm_full = prepare_inputs(q_ref_full, k_ref_full, v_ref_full)
        np.testing.assert_allclose(
            np.asarray(qp, np.float32),
            np.asarray(exp_qp_full, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(kvm, np.float32),
            np.asarray(exp_kvm_full, np.float32),
            rtol=0.125,
            atol=2e-2,
        )
        assert (
            float(
                np.mean(
                    np.abs(
                        np.asarray(kvm, np.float32)
                        - np.asarray(exp_kvm_full, np.float32)
                    )
                )
            )
            < 1e-3
        )

        spec = KvCacheSpec(
            layers=1, kv_heads=8, head_dim=128, page_tokens=256, dtype=jnp.float8_e4m3fn
        )
        cache_sharding = NamedSharding(mesh, KV_PARTITION)
        cache = jax.device_put(
            jnp.zeros((8, *spec.page_shape(tp)), jnp.float8_e4m3fn), cache_sharding
        )
        ref_cache = jax.device_put(
            jnp.zeros((8, *spec.page_shape(tp)), jnp.float8_e4m3fn), cache_sharding
        )
        pages = np.zeros((4, 4), dtype=np.int32)
        pages[0, :2] = [1, 2]
        pages[1, :1] = [3]
        rep = NamedSharding(mesh, P())
        meta = AttentionMetadata(
            jax.device_put(jnp.array([300, 120, 0, 0], jnp.int32), rep),
            (jax.device_put(jnp.asarray(pages.reshape(-1)), rep),),
            jax.device_put(jnp.array([0, 300, 420, 420, 420], jnp.int32), rep),
            jax.device_put(jnp.array([0, 0, 2], jnp.int32), rep),
        )
        att_mosaic, _ = jax.jit(
            lambda q_i, kv_i, c_i, m_i: paged_attention(
                q_i,
                kv_i,
                None,
                c_i,
                m_i,
                mesh=mesh,
                sm_scale=128**-0.5,
                prepared=True,
                actual_num_q_heads=64,
                actual_head_dim=128,
            )
        )(qp, kvm, cache, meta)
        att_ref, _ = paged_attention(
            qp,
            kvm,
            None,
            ref_cache,
            meta,
            mesh=mesh,
            sm_scale=128**-0.5,
            prepared=True,
            actual_num_q_heads=64,
            actual_head_dim=128,
            impl=reference.ragged_paged_attention,
        )
        assert not np.any(np.isnan(np.asarray(att_mosaic[:420], np.float32)))
        np.testing.assert_allclose(
            np.asarray(att_mosaic[:420], np.float32),
            np.asarray(att_ref[:420], np.float32),
            rtol=3e-2,
            atol=5e-2,
        )

        # 2b. Packed attention output + pallas_unpack_attn_fp8_quant (tb=512).
        cache_packed = jax.device_put(
            jnp.zeros((8, *spec.page_shape(tp)), jnp.float8_e4m3fn), cache_sharding
        )
        att_packed, _ = jax.jit(
            lambda q_i, kv_i, c_i, m_i: paged_attention(
                q_i,
                kv_i,
                None,
                c_i,
                m_i,
                mesh=mesh,
                sm_scale=128**-0.5,
                prepared=True,
                return_packed=True,
                actual_num_q_heads=64,
                actual_head_dim=128,
            )
        )(qp, kvm, cache_packed, meta)
        att_q_sh, att_s_sh = jax.jit(
            jax.shard_map(
                lambda pk: pallas_unpack_attn_fp8_quant(pk, 8, 128),
                mesh=mesh,
                in_specs=Q_PACKED_PARTITION,
                out_specs=(P(None, "tp"), P("tp", None)),
                check_vma=False,
            )
        )(att_packed)
        att_bf16_sh, att_abs_sh = jax.jit(
            jax.shard_map(
                lambda pk: pallas_unpack_attn_fp8_quant(
                    pk, 8, 128, return_bf16_and_absmax=True
                ),
                mesh=mesh,
                in_specs=Q_PACKED_PARTITION,
                out_specs=(P(None, "tp"), P("tp", None)),
                check_vma=False,
            )
        )(att_packed)
        att_mosaic_2d = jnp.asarray(att_mosaic).reshape(tokens, 8192)
        np.testing.assert_array_equal(
            np.asarray(att_bf16_sh[:420]), np.asarray(att_mosaic_2d[:420])
        )
        for s_idx in range(tp):
            local_2d = att_mosaic_2d[
                :420, s_idx * (8192 // tp) : (s_idx + 1) * (8192 // tp)
            ]
            exp_q_l, exp_s_l = quantize_fp8_act(local_2d)
            got_q_l = np.asarray(att_q_sh)[
                :420, s_idx * (8192 // tp) : (s_idx + 1) * (8192 // tp)
            ]
            got_s_l = np.asarray(att_s_sh)[s_idx, :420]
            np.testing.assert_allclose(
                got_q_l.astype(np.float32),
                np.asarray(exp_q_l, np.float32),
                rtol=0.125,
                atol=5e-2,
            )
            assert (
                float(
                    np.mean(
                        np.abs(
                            got_q_l.astype(np.float32) - np.asarray(exp_q_l, np.float32)
                        )
                    )
                )
                < 1e-3
            )
            np.testing.assert_allclose(
                got_s_l, np.asarray(exp_s_l[:, 0]), rtol=1e-4, atol=1e-5
            )
            np.testing.assert_allclose(
                np.asarray(att_abs_sh)[s_idx, :420],
                np.max(np.abs(np.asarray(local_2d, np.float32)), axis=-1),
                rtol=1e-5,
                atol=1e-5,
            )

        # 3. Fused gate_up_swiglu + down_proj reduce_scatter
        w_gu = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((51200, 5120)) * 0.02, jnp.bfloat16)
        )
        w_down = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((5120, 25600)) * 0.02, jnp.bfloat16)
        )
        w_gu_q = jax.device_put(w_gu.qvalue, NamedSharding(mesh, P("tp", None)))
        w_gu_s = jax.device_put(w_gu.scale, NamedSharding(mesh, P("tp")))
        w_d_q = jax.device_put(w_down.qvalue, NamedSharding(mesh, P(None, "tp")))
        w_d_s = jax.device_put(w_down.scale, NamedSharding(mesh, P()))

        def _gu_pallas(xq_i, xs_i):
            return sharded_fp8_gate_up_swiglu(
                xq_i, xs_i, w_gu_q, w_gu_s, mesh=mesh, backend="pallas"
            )

        g_pallas, g_abs = jax.jit(_gu_pallas)(xq_sh, xs_sh)
        g_xla = jax.jit(
            lambda xq_i, xs_i: sharded_fp8_gate_up_swiglu(
                xq_i, xs_i, w_gu_q, w_gu_s, mesh=mesh, backend="xla"
            )[0]
        )(xq_sh, xs_sh)
        assert not np.any(np.isnan(np.asarray(g_pallas, np.float32)))
        np.testing.assert_allclose(
            np.asarray(g_pallas, np.float32),
            np.asarray(g_xla, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )
        assert g_abs is not None
        np.testing.assert_allclose(
            np.asarray(g_abs, np.float32),
            np.max(np.abs(np.asarray(g_pallas, np.float32)), axis=-1),
            rtol=1e-5,
            atol=1e-5,
        )

        # Compare down_proj reduce_scatter on the same input g_pallas, plus chained cosine similarity.
        down_pallas = jax.jit(
            lambda g_i, a_i: sharded_fp8_matmul_reduce_scatter(
                g_i, w_d_q, w_d_s, mesh=mesh, x_abs_max=a_i, backend="pallas"
            )
        )(g_pallas, g_abs)
        down_xla_same_in = jax.jit(
            lambda g_i: sharded_fp8_matmul_reduce_scatter(
                g_i, w_d_q, w_d_s, mesh=mesh, backend="xla"
            )
        )(g_pallas)
        assert not np.any(np.isnan(np.asarray(down_pallas, np.float32)))
        np.testing.assert_allclose(
            np.asarray(down_pallas, np.float32),
            np.asarray(down_xla_same_in, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )

        down_xla_chained = jax.jit(
            lambda g_i: sharded_fp8_matmul_reduce_scatter(
                g_i, w_d_q, w_d_s, mesh=mesh, backend="xla"
            )
        )(g_xla)
        p_flat = np.asarray(down_pallas, np.float32).reshape(-1)
        x_flat = np.asarray(down_xla_chained, np.float32).reshape(-1)
        cosine = float(
            np.dot(p_flat, x_flat) / (np.linalg.norm(p_flat) * np.linalg.norm(x_flat))
        )
        assert cosine > 0.999

    def test_qwen3_tp2_mosaic_prefill_4096_w8a8_tiles_parity(self) -> None:
        from resin.backend.tpu.kernels import (
            Fp8Weight,
            fused_add_rms_norm_fp8_quant,
            quantize_fp8_act,
            sharded_fp8_gate_up_swiglu,
            sharded_fp8_matmul_all_gather,
            sharded_fp8_matmul_reduce_scatter,
        )

        mesh = make_mesh(min(len(jax.devices()), 2))
        rng = np.random.default_rng(4096)
        tokens = 4096

        def _sharded_fp8_weight(
            shape: tuple[int, int], *, col_parallel: bool
        ) -> Fp8Weight:
            weight = quantize_fp8_weight(
                jnp.asarray(rng.standard_normal(shape) * 0.02, jnp.bfloat16)
            )
            q_spec = P("tp", None) if col_parallel else P(None, "tp")
            s_spec = P("tp") if col_parallel else P()
            return Fp8Weight(
                qvalue=jax.device_put(weight.qvalue, NamedSharding(mesh, q_spec)),
                scale=jax.device_put(weight.scale, NamedSharding(mesh, s_spec)),
            )

        def _assert_close(actual: jax.Array, expected: jax.Array) -> None:
            np.testing.assert_allclose(
                np.asarray(actual, np.float32),
                np.asarray(expected, np.float32),
                rtol=2e-2,
                atol=5e-2,
            )

        x = jax.device_put(
            jnp.asarray(rng.standard_normal((tokens, 5120)) * 0.5, jnp.bfloat16),
            NamedSharding(mesh, P("tp", None)),
        )
        residual = jax.device_put(
            jnp.asarray(rng.standard_normal((tokens, 5120)) * 0.5, jnp.bfloat16),
            NamedSharding(mesh, P("tp", None)),
        )
        norm_weight = jax.device_put(
            jnp.ones((5120,), jnp.bfloat16), NamedSharding(mesh, P())
        )
        x_q, x_scale, _ = jax.jit(
            jax.shard_map(
                lambda xl, rl, w: fused_add_rms_norm_fp8_quant(xl, rl, w, 1e-6),
                mesh=mesh,
                in_specs=(P("tp", None), P("tp", None), P()),
                out_specs=(P("tp", None), P("tp", None), P("tp", None)),
                check_vma=False,
            )
        )(x, residual, norm_weight)

        def _jit_pallas_and_xla(call_fn: Any, *args: jax.Array) -> tuple[Any, Any]:
            return (
                jax.jit(functools.partial(call_fn, backend="pallas"))(*args),
                jax.jit(functools.partial(call_fn, backend="xla"))(*args),
            )

        w_qkv = _sharded_fp8_weight((10240, 5120), col_parallel=True)
        qkv_pallas, qkv_xla = _jit_pallas_and_xla(
            lambda xq_i, xs_i, *, backend: sharded_fp8_matmul_all_gather(
                xq_i, xs_i, w_qkv.qvalue, w_qkv.scale, mesh=mesh, backend=backend
            ),
            x_q,
            x_scale,
        )
        _assert_close(qkv_pallas, qkv_xla)

        # o_proj reduce_scatter: test both unquantized bf16 and pre-quantized fp8 activations.
        att_bf16 = jax.device_put(
            jnp.asarray(rng.standard_normal((tokens, 8192)) * 0.5, jnp.bfloat16),
            NamedSharding(mesh, P(None, "tp")),
        )
        w_o = _sharded_fp8_weight((5120, 8192), col_parallel=False)
        o_pallas_bf16, o_xla_bf16 = _jit_pallas_and_xla(
            lambda a_i, *, backend: sharded_fp8_matmul_reduce_scatter(
                a_i, w_o.qvalue, w_o.scale, mesh=mesh, backend=backend
            ),
            att_bf16,
        )
        _assert_close(o_pallas_bf16, o_xla_bf16)

        def _quant_per_die(a_shard: jax.Array) -> tuple[jax.Array, jax.Array]:
            aq_s, as_s = quantize_fp8_act(a_shard)
            return aq_s, as_s.reshape(1, tokens)

        att_q, att_scale = jax.jit(
            jax.shard_map(
                _quant_per_die,
                mesh=mesh,
                in_specs=P(None, "tp"),
                out_specs=(P(None, "tp"), P("tp", None)),
            )
        )(att_bf16)
        o_pallas_fp8, o_xla_fp8 = _jit_pallas_and_xla(
            lambda aq, a_s, *, backend: sharded_fp8_matmul_reduce_scatter(
                aq,
                w_o.qvalue,
                w_o.scale,
                mesh=mesh,
                x_scale=a_s,
                sync_tp_scale=False,
                backend=backend,
            ),
            att_q,
            att_scale,
        )
        _assert_close(o_pallas_fp8, o_xla_fp8)

        # gate_up_swiglu + down_proj (both default (1024, 2560, 2560) and save_x_q=True (1024, 1280, 2560)) at T=4096.
        w_gate_up = _sharded_fp8_weight((51200, 5120), col_parallel=True)
        w_down = _sharded_fp8_weight((5120, 25600), col_parallel=False)

        (gated_pallas, gated_absmax), (gated_xla, _) = _jit_pallas_and_xla(
            lambda xq_i, xs_i, *, backend: sharded_fp8_gate_up_swiglu(
                xq_i,
                xs_i,
                w_gate_up.qvalue,
                w_gate_up.scale,
                mesh=mesh,
                sync_tp_scale=False,
                backend=backend,
            ),
            x_q,
            x_scale,
        )
        _assert_close(gated_pallas, gated_xla)

        assert gated_absmax is not None
        down_pallas = jax.jit(
            lambda g_i, a_i: sharded_fp8_matmul_reduce_scatter(
                g_i,
                w_down.qvalue,
                w_down.scale,
                mesh=mesh,
                x_abs_max=a_i,
                sync_tp_scale=False,
                backend="pallas",
            )
        )(gated_pallas, gated_absmax)
        down_xla = jax.jit(
            lambda g_i: sharded_fp8_matmul_reduce_scatter(
                g_i,
                w_down.qvalue,
                w_down.scale,
                mesh=mesh,
                sync_tp_scale=False,
                backend="xla",
            )
        )(gated_pallas)
        _assert_close(down_pallas, down_xla)

        # Also exercise save_x_q=True + redundant-x-DMA skip on (1024, 1280, 2560) (n_out=4, n_in=5).
        down_savexq = jax.jit(
            jax.shard_map(
                lambda g_s, a_s, wq_s, ws_s: fp8_matmul(
                    g_s,
                    wq_s,
                    ws_s,
                    x_abs_max=a_s,
                    tp_axis=None,
                    scatter_axis="tp",
                    sync_tp_scale=False,
                    backend="pallas",
                    tile_sizes=Fp8MatmulTileSizes(1024, 1280, 2560),
                ),
                mesh=mesh,
                in_specs=(P(None, "tp"), P("tp", None), P(None, "tp"), P()),
                out_specs=P("tp", None),
                check_vma=False,
            )
        )(gated_pallas, gated_absmax, w_down.qvalue, w_down.scale)
        _assert_close(down_savexq, down_pallas)

    def test_qwen3_tp2_mosaic_decode_fused_kernels_parity(self) -> None:
        from resin.backend.tpu.attention import (
            KV_MERGED_PARTITION,
            Q_PACKED_PARTITION,
        )
        from resin.backend.tpu.kernels import (
            fused_add_rms_norm_fp8_quant,
            pallas_qkv_norm_rope_pack,
            pallas_unpack_attn_fp8_quant,
            sharded_fp8_gate_up_swiglu,
        )

        mesh = make_mesh(min(len(jax.devices()), 2))
        tp = int(mesh.shape["tp"])
        rng = np.random.default_rng(3200)
        tokens, live = 320, 289
        rep = NamedSharding(mesh, P())
        rep_2d = NamedSharding(mesh, P(None, None))

        x_np = rng.standard_normal((tokens, 5120)).astype(np.float32) * 0.5
        res_np = rng.standard_normal((tokens, 5120)).astype(np.float32) * 0.5
        x_np[live:] = 0.0
        res_np[live:] = 0.0
        x = jax.device_put(jnp.asarray(x_np, jnp.bfloat16), rep_2d)
        res = jax.device_put(jnp.asarray(res_np, jnp.bfloat16), rep_2d)
        nw = jax.device_put(jnp.ones((5120,), jnp.bfloat16), rep)

        xq, xs_t, _ = jax.jit(
            jax.shard_map(
                lambda xl, rl, w: fused_add_rms_norm_fp8_quant(
                    xl, rl, w, 1e-6, transposed_scale=True
                ),
                mesh=mesh,
                in_specs=(P(None, None), P(None, None), P()),
                out_specs=(P(None, None), P(None, None), P(None, None)),
                check_vma=False,
            )
        )(x, res, nw)

        w_qkv = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((10240, 5120)) * 0.02, jnp.bfloat16)
        )
        qkv_out = jax.jit(
            lambda xqi, xsi, wq, ws: sharded_fp8_matmul(
                xqi, wq, ws, P("tp", None), mesh=mesh, x_scale=xsi, backend="pallas"
            )
        )(
            xq,
            xs_t,
            jax.device_put(w_qkv.qvalue, NamedSharding(mesh, P("tp", None))),
            jax.device_put(w_qkv.scale, NamedSharding(mesh, P("tp"))),
        )

        rc = build_rope_cache(2048, 128)
        rc_rep = (jax.device_put(rc[0], rep), jax.device_put(rc[1], rep))
        pos_np = np.zeros((tokens,), dtype=np.int32)
        pos_np[:live] = 512
        pos = jax.device_put(jnp.asarray(pos_np), rep)
        qnw = jax.device_put(jnp.ones((128,), jnp.bfloat16), rep)
        knw = jax.device_put(jnp.ones((128,), jnp.bfloat16), rep)

        qp, kvm = jax.jit(
            jax.shard_map(
                lambda ql, p, c, s, qw, kw: pallas_qkv_norm_rope_pack(
                    ql,
                    p,
                    (c, s),
                    qw,
                    kw,
                    jnp.float8_e4m3fn,
                    num_q_heads=64 // tp,
                    num_kv_heads=8 // tp,
                ),
                mesh=mesh,
                in_specs=(P(None, "tp"), P(), P(), P(), P(), P()),
                out_specs=(Q_PACKED_PARTITION, KV_MERGED_PARTITION),
                check_vma=False,
            )
        )(qkv_out, pos, rc_rep[0], rc_rep[1], qnw, knw)

        spec = KvCacheSpec(
            layers=1, kv_heads=8, head_dim=128, page_tokens=256, dtype=jnp.float8_e4m3fn
        )
        cache_sh = NamedSharding(mesh, KV_PARTITION)
        total_pages = 2 * tokens + 2
        cache = jax.device_put(
            jnp.zeros((total_pages, *spec.page_shape(tp)), jnp.float8_e4m3fn), cache_sh
        )
        ref_cache = jax.device_put(
            jnp.zeros((total_pages, *spec.page_shape(tp)), jnp.float8_e4m3fn), cache_sh
        )
        pages = np.zeros((tokens, 4), dtype=np.int32)
        for i in range(live):
            pages[i, :2] = [2 * i + 1, 2 * i + 2]
        kv_lens_np = np.zeros((tokens,), dtype=np.int32)
        kv_lens_np[:live] = 260
        cu_np = np.array([min(i, live) for i in range(tokens + 1)], dtype=np.int32)
        meta = AttentionMetadata(
            jax.device_put(jnp.asarray(kv_lens_np), rep),
            (jax.device_put(jnp.asarray(pages.reshape(-1)), rep),),
            jax.device_put(jnp.asarray(cu_np), rep),
            jax.device_put(jnp.array([live, live, live], jnp.int32), rep),
        )

        w_o = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((5120, 8192)) * 0.02, jnp.bfloat16)
        )
        wo_q = jax.device_put(w_o.qvalue, NamedSharding(mesh, P(None, "tp")))
        wo_s = jax.device_put(w_o.scale, rep)

        def _fused_attn_o(qp_i, kvm_i, c_i, m_i):
            att_p, c_next = paged_attention(
                qp_i,
                kvm_i,
                None,
                c_i,
                m_i,
                mesh=mesh,
                sm_scale=128**-0.5,
                prepared=True,
                return_packed=True,
                actual_num_q_heads=64,
                actual_head_dim=128,
            )
            aq, as_t = jax.shard_map(
                lambda pl_i: pallas_unpack_attn_fp8_quant(pl_i, 8, 128),
                mesh=mesh,
                in_specs=Q_PACKED_PARTITION,
                out_specs=(P(None, "tp"), P("tp", None)),
                check_vma=False,
            )(att_p)
            o_out = sharded_fp8_matmul(
                aq,
                wo_q,
                wo_s,
                P(None, "tp"),
                mesh=mesh,
                x_scale=as_t,
                sync_tp_scale=False,
                backend="pallas",
            )
            return o_out, c_next

        o_mosaic, _ = jax.jit(_fused_attn_o)(qp, kvm, cache, meta)
        att_ref, _ = paged_attention(
            qp,
            kvm,
            None,
            ref_cache,
            meta,
            mesh=mesh,
            sm_scale=128**-0.5,
            prepared=True,
            actual_num_q_heads=64,
            actual_head_dim=128,
            impl=reference.ragged_paged_attention,
        )
        o_ref = sharded_fp8_matmul(
            att_ref.reshape(tokens, 8192),
            wo_q,
            wo_s,
            P(None, "tp"),
            mesh=mesh,
            sync_tp_scale=False,
            backend="xla",
        )
        assert not np.any(np.isnan(np.asarray(o_mosaic, np.float32)))
        np.testing.assert_allclose(
            np.asarray(o_mosaic, np.float32)[live:], 0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(o_mosaic, np.float32)[:live],
            np.asarray(o_ref, np.float32)[:live],
            rtol=3e-2,
            atol=5e-2,
        )

        # Decode gate_up_swiglu (320, 1280, 5120) + down_proj (320, 512, 12800) with zero-padded tail rows
        w_gu = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((51200, 5120)) * 0.02, jnp.bfloat16)
        )
        w_down = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((5120, 25600)) * 0.02, jnp.bfloat16)
        )
        w_gu_q = jax.device_put(w_gu.qvalue, NamedSharding(mesh, P("tp", None)))
        w_gu_s = jax.device_put(w_gu.scale, NamedSharding(mesh, P("tp")))
        w_d_q = jax.device_put(w_down.qvalue, NamedSharding(mesh, P(None, "tp")))
        w_d_s = jax.device_put(w_down.scale, rep)

        g_dec_pallas, g_dec_abs = jax.jit(
            lambda xqi, xsi: sharded_fp8_gate_up_swiglu(
                xqi,
                xsi,
                w_gu_q,
                w_gu_s,
                mesh=mesh,
                sequence_parallel=False,
                sync_tp_scale=False,
                backend="pallas",
            )
        )(xq, xs_t)
        g_dec_xla = jax.jit(
            lambda xqi, xsi: sharded_fp8_gate_up_swiglu(
                xqi,
                xsi,
                w_gu_q,
                w_gu_s,
                mesh=mesh,
                sequence_parallel=False,
                sync_tp_scale=False,
                backend="xla",
            )[0]
        )(xq, xs_t)
        assert not np.any(np.isnan(np.asarray(g_dec_pallas, np.float32)))
        np.testing.assert_allclose(
            np.asarray(g_dec_pallas, np.float32)[live:], 0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(g_dec_pallas, np.float32),
            np.asarray(g_dec_xla, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )
        assert g_dec_abs is not None

        down_dec_pallas = jax.jit(
            lambda g_i, a_i: sharded_fp8_matmul(
                g_i,
                w_d_q,
                w_d_s,
                P(None, "tp"),
                mesh=mesh,
                x_abs_max=a_i,
                sync_tp_scale=False,
                backend="pallas",
            )
        )(g_dec_pallas, g_dec_abs)
        down_dec_xla_same = jax.jit(
            lambda g_i: sharded_fp8_matmul(
                g_i,
                w_d_q,
                w_d_s,
                P(None, "tp"),
                mesh=mesh,
                sync_tp_scale=False,
                backend="xla",
            )
        )(g_dec_pallas)
        assert not np.any(np.isnan(np.asarray(down_dec_pallas, np.float32)))
        np.testing.assert_allclose(
            np.asarray(down_dec_pallas, np.float32)[live:], 0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(down_dec_pallas, np.float32),
            np.asarray(down_dec_xla_same, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )

        g_dec_q, g_dec_scale = jax.jit(
            lambda xqi, xsi: sharded_fp8_gate_up_swiglu(
                xqi,
                xsi,
                w_gu_q,
                w_gu_s,
                mesh=mesh,
                sequence_parallel=False,
                sync_tp_scale=False,
                quantize_output=True,
                backend="pallas",
            )
        )(xq, xs_t)
        assert g_dec_q.dtype == jnp.float8_e4m3fn
        assert g_dec_scale is not None
        down_dec_prequant = jax.jit(
            lambda gq_i, gs_i: sharded_fp8_matmul(
                gq_i,
                w_d_q,
                w_d_s,
                P(None, "tp"),
                mesh=mesh,
                x_scale=gs_i,
                sync_tp_scale=False,
                backend="pallas",
            )
        )(g_dec_q, g_dec_scale)
        assert not np.any(np.isnan(np.asarray(down_dec_prequant, np.float32)))
        np.testing.assert_allclose(
            np.asarray(down_dec_prequant, np.float32)[live:], 0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(down_dec_prequant, np.float32),
            np.asarray(down_dec_xla_same, np.float32),
            rtol=2e-2,
            atol=5e-2,
        )

        down_dec_xla_chained = jax.jit(
            lambda g_i: sharded_fp8_matmul(
                g_i,
                w_d_q,
                w_d_s,
                P(None, "tp"),
                mesh=mesh,
                sync_tp_scale=False,
                backend="xla",
            )
        )(g_dec_xla)
        p_dec_flat = np.asarray(down_dec_pallas, np.float32)[:live].reshape(-1)
        x_dec_flat = np.asarray(down_dec_xla_chained, np.float32)[:live].reshape(-1)
        dec_cos = float(
            np.dot(p_dec_flat, x_dec_flat)
            / (np.linalg.norm(p_dec_flat) * np.linalg.norm(x_dec_flat))
        )
        assert dec_cos > 0.999

        gq_dec_pallas, gs_dec_pallas = jax.jit(
            lambda xqi, xsi: sharded_fp8_gate_up_swiglu(
                xqi,
                xsi,
                w_gu_q,
                w_gu_s,
                mesh=mesh,
                sequence_parallel=False,
                sync_tp_scale=False,
                quantize_output=True,
                backend="pallas",
            )
        )(xq, xs_t)
        assert gq_dec_pallas.dtype == jnp.float8_e4m3fn
        assert gs_dec_pallas is not None
        down_preq_pallas = jax.jit(
            lambda gq_i, gs_i: sharded_fp8_matmul(
                gq_i,
                w_d_q,
                w_d_s,
                P(None, "tp"),
                mesh=mesh,
                x_scale=gs_i,
                sync_tp_scale=False,
                backend="pallas",
            )
        )(gq_dec_pallas, gs_dec_pallas)
        np.testing.assert_allclose(
            np.asarray(down_preq_pallas, np.float32)[live:], 0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(down_preq_pallas, np.float32),
            np.asarray(down_dec_pallas, np.float32),
            rtol=1e-2,
            atol=2e-2,
        )

    def test_qwen3_tp2_mosaic_async_row_all_reduce_parity_and_zero_recompile(
        self,
    ) -> None:
        from resin.backend.tpu.kernels import (
            fused_add_rms_norm_fp8_quant,
            quantize_fp8_act,
            sharded_fp8_gate_up_swiglu,
            sharded_fp8_row_matmul_async_ar,
        )

        if len(jax.devices()) < 2:
            pytest.skip("requires 2 TPU cores for tp=2 async all-reduce")

        hidden_size = 5120
        q_width = 8192
        ffn_width = 25600
        rms_eps = 1e-6

        mesh = make_mesh(2)
        rep = NamedSharding(mesh, P())
        rep_2d = NamedSharding(mesh, P(None, None))
        row_sh = NamedSharding(mesh, P(None, "tp"))
        col_sh = NamedSharding(mesh, P("tp", None))
        sc_col_sh = NamedSharding(mesh, P("tp"))
        sc_die_sh = NamedSharding(mesh, P("tp", None))
        rng = np.random.default_rng(7702)

        w_o = quantize_fp8_weight(
            jnp.asarray(
                rng.standard_normal((hidden_size, q_width)) * 0.02, jnp.bfloat16
            )
        )
        w_gu = quantize_fp8_weight(
            jnp.asarray(
                rng.standard_normal((2 * ffn_width, hidden_size)) * 0.02, jnp.bfloat16
            )
        )
        w_d = quantize_fp8_weight(
            jnp.asarray(
                rng.standard_normal((hidden_size, ffn_width)) * 0.02, jnp.bfloat16
            )
        )
        wo_q = jax.device_put(w_o.qvalue, row_sh)
        wo_s = jax.device_put(w_o.scale, rep)
        wgu_q = jax.device_put(w_gu.qvalue, col_sh)
        wgu_s = jax.device_put(w_gu.scale, sc_col_sh)
        wd_q = jax.device_put(w_d.qvalue, row_sh)
        wd_s = jax.device_put(w_d.scale, rep)
        nw1 = jax.device_put(jnp.ones((hidden_size,), jnp.bfloat16), rep)
        nw2 = jax.device_put(jnp.ones((hidden_size,), jnp.bfloat16), rep)

        def _add_norm_quant(
            dl: jax.Array, rl: jax.Array, w: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                lambda d, r, nw: fused_add_rms_norm_fp8_quant(
                    d, r, nw, rms_eps, transposed_scale=True
                ),
                mesh=mesh,
                in_specs=(P(None, None), P(None, None), P()),
                out_specs=(P(None, None), P(None, None), P(None, None)),
                check_vma=False,
            )(dl, rl, w)

        def _quant_per_die(x_bf16_sh: jax.Array) -> tuple[jax.Array, jax.Array]:
            n_rows = x_bf16_sh.shape[0]

            def _local_q(xl: jax.Array) -> tuple[jax.Array, jax.Array]:
                ql, sl = quantize_fp8_act(xl, tp_axis="tp", sync_tp_scale=False)
                return ql, jnp.transpose(sl.reshape(n_rows, 1))

            return jax.shard_map(
                _local_q,
                mesh=mesh,
                in_specs=P(None, "tp"),
                out_specs=(P(None, "tp"), P("tp", None)),
                check_vma=False,
            )(x_bf16_sh)

        def _row_proj(
            xq_in: jax.Array,
            xs_in: jax.Array | None,
            wq_in: jax.Array,
            ws_in: jax.Array,
            use_async: bool,
        ) -> jax.Array:
            if use_async:
                return sharded_fp8_row_matmul_async_ar(
                    xq_in,
                    wq_in,
                    ws_in,
                    mesh=mesh,
                    x_scale=xs_in,
                    sync_tp_scale=False,
                    backend="pallas",
                )
            return sharded_fp8_matmul(
                xq_in,
                wq_in,
                ws_in,
                P(None, "tp"),
                mesh=mesh,
                x_scale=xs_in,
                sync_tp_scale=False,
                backend="pallas",
            )

        def _two_step_body(
            res: jax.Array, aq: jax.Array, as_t: jax.Array, *, use_async: bool
        ) -> jax.Array:
            for _ in range(2):
                o_out = _row_proj(aq, as_t, wo_q, wo_s, use_async)
                mlp_q, mlp_s, res = _add_norm_quant(o_out, res, nw1)
                gq, gs = sharded_fp8_gate_up_swiglu(
                    mlp_q,
                    mlp_s,
                    wgu_q,
                    wgu_s,
                    mesh=mesh,
                    sequence_parallel=False,
                    sync_tp_scale=False,
                    quantize_output=True,
                    backend="pallas",
                )
                d_out = _row_proj(gq, gs, wd_q, wd_s, use_async)
                _, _, res = _add_norm_quant(d_out, res, nw2)
            return res

        _o_async = jax.jit(lambda x, s: _row_proj(x, s, wo_q, wo_s, True))
        _o_stock_psum = jax.jit(lambda x, s: _row_proj(x, s, wo_q, wo_s, False))
        _async_two_step = jax.jit(
            functools.partial(_two_step_body, use_async=True), donate_argnums=(0,)
        )
        _psum_two_step = jax.jit(
            functools.partial(_two_step_body, use_async=False),
            compiler_options={
                "xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes": "67108864"
            },
        )

        for bs in (8, 16, 160, 320):
            live = max(1, bs - 3)
            res_np = rng.standard_normal((bs, hidden_size)).astype(np.float32) * 0.4
            att_np = rng.standard_normal((bs, q_width)).astype(np.float32) * 0.4
            res_np[live:] = 0.0
            att_np[live:] = 0.0
            res0 = jax.device_put(jnp.asarray(res_np, jnp.bfloat16), rep_2d)
            att_bf16 = jax.device_put(jnp.asarray(att_np, jnp.bfloat16), row_sh)
            aq, as_t = _quant_per_die(att_bf16)
            as_t = jax.device_put(as_t, sc_die_sh)

            np.testing.assert_allclose(
                np.asarray(_o_async(aq, as_t), np.float32),
                np.asarray(_o_stock_psum(aq, as_t), np.float32),
                rtol=1e-3,
                atol=5e-2,
            )

            hlo_txt = _async_two_step.lower(res0, aq, as_t).as_text()
            assert "stablehlo.all_reduce" not in hlo_txt

            y_ref = jax.block_until_ready(_psum_two_step(res0, aq, as_t))
            counter = CompilationCounter()
            try:
                y_got = jax.block_until_ready(_async_two_step(res0.copy(), aq, as_t))
                shard_0 = np.asarray(y_got.addressable_shards[0].data)
                shard_1 = np.asarray(y_got.addressable_shards[1].data)
                warm = counter.count
                y_got2 = jax.block_until_ready(_async_two_step(y_got, aq, as_t))
                assert counter.count == warm
                assert not np.any(np.isnan(np.asarray(y_got2, np.float32)))
            finally:
                counter.close()

            np.testing.assert_array_equal(shard_0, shard_1)
            np.testing.assert_array_equal(np.asarray(shard_0[live:], np.float32), 0.0)
            np.testing.assert_allclose(
                np.asarray(shard_0, np.float32),
                np.asarray(y_ref, np.float32),
                rtol=1e-2,
                atol=2e-2,
            )

    def test_qwen3_tp2_mosaic_async_ar_fused_norm_quant_bit_exact_buckets(
        self,
    ) -> None:
        from resin.backend.tpu.kernels import (
            fused_add_rms_norm_fp8_quant,
            quantize_fp8_act,
            sharded_fp8_gate_up_swiglu,
            sharded_fp8_row_matmul_async_ar,
        )

        if len(jax.devices()) < 2:
            pytest.skip("requires 2 TPU cores for tp=2 async all-reduce")

        hidden_size = 5120
        q_width = 8192
        ffn_width = 25600
        rms_eps = 1e-6

        mesh = make_mesh(2)
        rep = NamedSharding(mesh, P())
        rep_2d = NamedSharding(mesh, P(None, None))
        row_sh = NamedSharding(mesh, P(None, "tp"))
        col_sh = NamedSharding(mesh, P("tp", None))
        sc_col_sh = NamedSharding(mesh, P("tp"))
        sc_die_sh = NamedSharding(mesh, P("tp", None))
        rng = np.random.default_rng(8803)

        w_o = quantize_fp8_weight(
            jnp.asarray(
                rng.standard_normal((hidden_size, q_width)) * 0.02, jnp.bfloat16
            )
        )
        w_gu = quantize_fp8_weight(
            jnp.asarray(
                rng.standard_normal((2 * ffn_width, hidden_size)) * 0.02, jnp.bfloat16
            )
        )
        w_d = quantize_fp8_weight(
            jnp.asarray(
                rng.standard_normal((hidden_size, ffn_width)) * 0.02, jnp.bfloat16
            )
        )
        wo_q = jax.device_put(w_o.qvalue, row_sh)
        wo_s = jax.device_put(w_o.scale, rep)
        wgu_q = jax.device_put(w_gu.qvalue, col_sh)
        wgu_s = jax.device_put(w_gu.scale, sc_col_sh)
        wd_q = jax.device_put(w_d.qvalue, row_sh)
        wd_s = jax.device_put(w_d.scale, rep)
        nw1 = jax.device_put(
            jnp.asarray(rng.uniform(0.5, 1.5, (hidden_size,)), jnp.bfloat16), rep
        )
        nw2 = jax.device_put(
            jnp.asarray(rng.uniform(0.5, 1.5, (hidden_size,)), jnp.bfloat16), rep
        )

        def _add_norm_quant(
            dl: jax.Array, rl: jax.Array, w: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                lambda d, r, nw: fused_add_rms_norm_fp8_quant(
                    d,
                    r,
                    nw,
                    rms_eps,
                    dtype=jnp.float8_e4m3fn,
                    transposed_scale=True,
                ),
                mesh=mesh,
                in_specs=(P(None, None), P(None, None), P()),
                out_specs=(P(None, None), P(None, None), P(None, None)),
                check_vma=False,
            )(dl, rl, w)

        @jax.jit
        def _quant_per_die(x_bf16_sh: jax.Array) -> tuple[jax.Array, jax.Array]:
            def _q(xs: jax.Array) -> tuple[jax.Array, jax.Array]:
                q, s = quantize_fp8_act(xs, dtype=jnp.float8_e4m3fn)
                return q, jnp.transpose(s)

            return jax.shard_map(
                _q,
                mesh=mesh,
                in_specs=(P(None, "tp"),),
                out_specs=(P(None, "tp"), P("tp", None)),
                check_vma=False,
            )(x_bf16_sh)

        weights_arg = (wo_q, wo_s, nw1, wgu_q, wgu_s, wd_q, wd_s, nw2)

        @functools.partial(jax.jit, static_argnames=("fuse",))
        def _step_one(
            w_tup, res_in: jax.Array, aq: jax.Array, as_t: jax.Array, *, fuse: bool
        ):
            o_q, o_s, w1, gu_q, gu_s, d_q, d_s, w2 = w_tup
            if fuse:
                mq, ms, x = sharded_fp8_row_matmul_async_ar(
                    aq,
                    o_q,
                    o_s,
                    mesh=mesh,
                    tp_axis="tp",
                    x_scale=as_t,
                    out_dtype=jnp.bfloat16,
                    residual=res_in,
                    norm_weight=w1,
                    eps=rms_eps,
                    quantize_output_dtype=jnp.float8_e4m3fn,
                    transposed_scale=True,
                    collective_id=0,
                )
            else:
                do = sharded_fp8_row_matmul_async_ar(
                    aq,
                    o_q,
                    o_s,
                    mesh=mesh,
                    tp_axis="tp",
                    x_scale=as_t,
                    out_dtype=jnp.bfloat16,
                    collective_id=0,
                )
                mq, ms, x = _add_norm_quant(do, res_in, w1)
            dq, ds = sharded_fp8_gate_up_swiglu(
                mq,
                ms,
                gu_q,
                gu_s,
                mesh=mesh,
                sequence_parallel=False,
                sync_tp_scale=False,
                quantize_output=True,
                fuse_quant_epilogue=True,
                out_dtype=jnp.bfloat16,
            )
            if fuse:
                hq, hs, x = sharded_fp8_row_matmul_async_ar(
                    dq,
                    d_q,
                    d_s,
                    mesh=mesh,
                    tp_axis="tp",
                    x_scale=ds,
                    out_dtype=jnp.bfloat16,
                    residual=x,
                    norm_weight=w2,
                    eps=rms_eps,
                    quantize_output_dtype=jnp.float8_e4m3fn,
                    transposed_scale=True,
                    collective_id=1,
                )
            else:
                dd = sharded_fp8_row_matmul_async_ar(
                    dq,
                    d_q,
                    d_s,
                    mesh=mesh,
                    tp_axis="tp",
                    x_scale=ds,
                    out_dtype=jnp.bfloat16,
                    collective_id=1,
                )
                hq, hs, x = _add_norm_quant(dd, x, w2)
            return mq, ms, hq, hs, x

        for b in (1, 8, 16, 32, 64, 96, 128):
            res0 = jax.device_put(
                jnp.asarray(rng.standard_normal((b, hidden_size)) * 0.5, jnp.bfloat16),
                rep_2d,
            )
            a_bf16 = jax.device_put(
                jnp.asarray(rng.standard_normal((b, q_width)) * 0.25, jnp.bfloat16),
                row_sh,
            )
            aq, as_t = _quant_per_die(a_bf16)
            as_t = jax.device_put(as_t, sc_die_sh)

            hlo_txt = _step_one.lower(weights_arg, res0, aq, as_t, fuse=True).as_text()
            if b <= 96:
                assert "_add_norm_q" in hlo_txt
            else:
                assert "_add_norm_q" not in hlo_txt

            xf, xu = res0, res0
            got = ref = None
            for _ in range(4):
                got = _step_one(weights_arg, xf, aq, as_t, fuse=True)
                ref = _step_one(weights_arg, xu, aq, as_t, fuse=False)
                xf = got[-1]
                xu = ref[-1]
            got = jax.block_until_ready(got)
            ref = jax.block_until_ready(ref)
            for g_arr, r_arr in zip(got, ref, strict=True):
                np.testing.assert_array_equal(np.asarray(g_arr), np.asarray(r_arr))

    @pytest.mark.parametrize("kv_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
    def test_qwen3_tp2_mosaic_batched_decode_parity_and_zero_recompile(
        self, kv_dtype: jnp.dtype
    ) -> None:
        mesh = make_mesh(min(len(jax.devices()), 2))
        tp = int(mesh.shape["tp"])
        spec = KvCacheSpec(
            layers=1,
            kv_heads=8,
            head_dim=128,
            page_tokens=256,
            dtype=kv_dtype,
        )
        cache_sharding = NamedSharding(mesh, KV_PARTITION)
        head_sharding = NamedSharding(mesh, P(None, "tp", None))
        q_packed_sharding = NamedSharding(mesh, P("tp", None, None, None, None))
        kv_merged_sharding = NamedSharding(mesh, P(None, "tp", None, None))
        rep_sharding = NamedSharding(mesh, P())
        is_fp8 = kv_dtype == jnp.float8_e4m3fn
        k_scale = 0.5 if is_fp8 else None
        v_scale = 0.25 if is_fp8 else None
        rtol = 3e-2 if is_fp8 else 1e-2
        atol = 5e-2 if is_fp8 else 2e-2

        max_seqs = 160
        pages_per_seq = 18  # up to 4608 tokens/seq, crossing bkv_sz=1280 and page=256
        total_pages = max_seqs * 3 + 36
        global_cache_shape = (total_pages, *spec.page_shape(tp))
        rng = np.random.default_rng(2027)
        init_np = rng.standard_normal(global_cache_shape).astype(np.float32) * 0.25
        init_cache = np.array(
            quantize_kv(jnp.asarray(init_np, jnp.float32), kv_dtype), copy=True
        )
        nan_page = np.asarray(jnp.full(global_cache_shape[1:], jnp.nan, dtype=kv_dtype))
        init_cache[0] = 0
        init_cache[-1] = nan_page

        pages = np.full((max_seqs, pages_per_seq), total_pages - 1, dtype=np.int32)
        next_page = 1
        cur_kv_lens = np.zeros(max_seqs, dtype=np.int32)
        # First 2 sequences get long histories crossing bkv_sz=1280 & page=256;
        # remaining 158 get 1..3 pages with ragged offsets. Poison from
        # base_kv_len - 1 onward so every decode step overwrites the slot
        # immediately adjacent to a trailing NaN inside the same page.
        for s in range(max_seqs):
            if s == 0:
                kv_l = 4096
            elif s == 1:
                kv_l = 1280
            else:
                kv_l = 254 + ((s * 37) % 515)
            cur_kv_lens[s] = kv_l
            used_p = (kv_l + 8 + 255) // 256
            pages[s, :used_p] = np.arange(next_page, next_page + used_p, dtype=np.int32)
            tail_page = next_page + (kv_l // 256)
            init_cache[tail_page, kv_l % 256 :] = nan_page[kv_l % 256 :]
            for p_extra in range(tail_page + 1, next_page + used_p):
                init_cache[p_extra] = nan_page
            next_page += used_p

        cache = jax.device_put(jnp.asarray(init_cache), cache_sharding)
        ref_cache = jax.device_put(jnp.asarray(init_cache), cache_sharding)

        @functools.partial(
            jax.jit,
            donate_argnums=(2,),
            out_shardings=(head_sharding, cache_sharding),
        )
        def mosaic_batched_decode(q_p, kv_m, kv_c, meta):
            att_p, kv_out = paged_attention(
                q_p,
                kv_m,
                None,
                kv_c,
                meta,
                mesh=mesh,
                sm_scale=128**-0.5,
                k_scale=k_scale,
                v_scale=v_scale,
                prepared=True,
                return_packed=True,
                actual_num_q_heads=64,
                actual_head_dim=128,
            )
            out_3d = jax.shard_map(
                lambda x: prepare_outputs(x, 8, 128),
                mesh=mesh,
                in_specs=P("tp", None, None, None, None),
                out_specs=P(None, "tp", None),
            )(att_p)
            return out_3d, kv_out

        def ref_batched_decode(q_p, kv_m, kv_c, meta):
            return paged_attention(
                q_p,
                kv_m,
                None,
                kv_c,
                meta,
                mesh=mesh,
                sm_scale=128**-0.5,
                k_scale=k_scale,
                v_scale=v_scale,
                prepared=True,
                return_packed=True,
                actual_num_q_heads=64,
                actual_head_dim=128,
                impl=reference.ragged_paged_attention,
            )

        flat_pages = jax.device_put(pages.reshape(-1), rep_sharding)
        step_records = []
        counter = CompilationCounter()
        try:
            warm = None
            for b_live in (1, 8, 9, 32, 154, 160):
                q_raw = jnp.asarray(
                    rng.standard_normal((max_seqs, 64, 128)), jnp.bfloat16
                )
                k_raw = quantize_kv(
                    jnp.asarray(rng.standard_normal((max_seqs, 8, 128)), jnp.bfloat16),
                    kv_dtype,
                    k_scale,
                )
                v_raw = quantize_kv(
                    jnp.asarray(rng.standard_normal((max_seqs, 8, 128)), jnp.bfloat16),
                    kv_dtype,
                    v_scale,
                )
                q_p, kv_m = prepare_inputs(q_raw, k_raw, v_raw)
                q_p = jax.device_put(q_p, q_packed_sharding)
                kv_m = jax.device_put(kv_m, kv_merged_sharding)

                cur_kv_lens[:b_live] += 1
                step_kv = np.zeros(max_seqs, dtype=np.int32)
                step_kv[:b_live] = cur_kv_lens[:b_live]
                cu_q = np.array(
                    [min(i, b_live) for i in range(max_seqs + 1)], dtype=np.int32
                )
                meta = AttentionMetadata(
                    jax.device_put(step_kv, rep_sharding),
                    (flat_pages,),
                    jax.device_put(cu_q, rep_sharding),
                    jax.device_put(
                        np.array([b_live, b_live, b_live], dtype=np.int32),
                        rep_sharding,
                    ),
                )
                out_p, cache = mosaic_batched_decode(q_p, kv_m, cache, meta)
                out_p.block_until_ready()
                if warm is None:
                    warm = counter.count
                else:
                    assert counter.count == warm
                step_records.append((b_live, step_kv.copy(), q_p, kv_m, meta, out_p))
        finally:
            counter.close()

        for b_live, step_kv, q_p, kv_m, meta, out_unpacked in step_records:
            ref_p, ref_cache = ref_batched_decode(q_p, kv_m, ref_cache, meta)
            ref_unpacked = prepare_outputs(ref_p, 8, 128)
            out_f32 = np.asarray(out_unpacked[:b_live], np.float32)
            assert not np.any(np.isnan(out_f32))
            np.testing.assert_allclose(
                out_f32,
                np.asarray(ref_unpacked[:b_live], np.float32),
                rtol=rtol,
                atol=atol,
            )
        # Verify final KV cache slots written by Mosaic match the reference.
        cache_host = np.asarray(cache, np.float32)
        ref_cache_host = np.asarray(ref_cache, np.float32)
        for s in (0, 1, 8, 31, 153, 159):
            pos = int(cur_kv_lens[s]) - 1
            pg = int(pages[s, pos // 256])
            off = pos % 256
            np.testing.assert_array_equal(cache_host[pg, off], ref_cache_host[pg, off])

    @pytest.mark.parametrize("kv_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
    def test_qwen3_tp2_mosaic_batched_mixed_parity_and_zero_recompile(
        self, kv_dtype: jnp.dtype
    ) -> None:
        mesh = make_mesh(min(len(jax.devices()), 2))
        tp = int(mesh.shape["tp"])
        spec = KvCacheSpec(
            layers=1,
            kv_heads=8,
            head_dim=128,
            page_tokens=256,
            dtype=kv_dtype,
        )
        cache_sharding = NamedSharding(mesh, KV_PARTITION)
        head_sharding = NamedSharding(mesh, P(None, "tp", None))
        q_packed_sharding = NamedSharding(mesh, P("tp", None, None, None, None))
        kv_merged_sharding = NamedSharding(mesh, P(None, "tp", None, None))
        rep_sharding = NamedSharding(mesh, P())
        is_fp8 = kv_dtype == jnp.float8_e4m3fn
        k_scale = 0.5 if is_fp8 else None
        v_scale = 0.25 if is_fp8 else None
        rtol = 3e-2 if is_fp8 else 1e-2
        atol = 5e-2 if is_fp8 else 2e-2

        max_tokens = 1024
        max_seqs = 16
        pages_per_seq = 8  # up to 2048 tokens/seq, crossing bkv_sz=512 and page=256
        total_pages = 40
        global_cache_shape = (total_pages, *spec.page_shape(tp))
        rng = np.random.default_rng(2028)
        nan_page = np.asarray(jnp.full(global_cache_shape[1:], jnp.nan, dtype=kv_dtype))
        init_cache = np.repeat(nan_page[None], total_pages, axis=0)
        init_cache[0] = 0

        pages = np.full((max_seqs, pages_per_seq), total_pages - 1, dtype=np.int32)
        next_page = 1
        for s in range(8):
            pages[s, :4] = np.arange(next_page, next_page + 4, dtype=np.int32)
            next_page += 4

        cache = jax.device_put(jnp.asarray(init_cache), cache_sharding)
        ref_cache = jax.device_put(jnp.asarray(init_cache), cache_sharding)

        @functools.partial(
            jax.jit,
            donate_argnums=(2,),
            out_shardings=(head_sharding, cache_sharding),
        )
        def mosaic_mixed(q_p, kv_m, kv_c, meta):
            att_p, kv_out = paged_attention(
                q_p,
                kv_m,
                None,
                kv_c,
                meta,
                mesh=mesh,
                sm_scale=128**-0.5,
                k_scale=k_scale,
                v_scale=v_scale,
                prepared=True,
                return_packed=True,
                actual_num_q_heads=64,
                actual_head_dim=128,
            )
            out_3d = jax.shard_map(
                lambda x: prepare_outputs(x, 8, 128),
                mesh=mesh,
                in_specs=P("tp", None, None, None, None),
                out_specs=P(None, "tp", None),
            )(att_p)
            return out_3d, kv_out

        def ref_mixed(q_p, kv_m, kv_c, meta):
            return paged_attention(
                q_p,
                kv_m,
                None,
                kv_c,
                meta,
                mesh=mesh,
                sm_scale=128**-0.5,
                k_scale=k_scale,
                v_scale=v_scale,
                prepared=True,
                return_packed=True,
                actual_num_q_heads=64,
                actual_head_dim=128,
                impl=reference.ragged_paged_attention,
            )

        flat_pages = jax.device_put(pages.reshape(-1), rep_sharding)
        # Step 1: initial prefill of 4 ragged sequences [300, 180, 260, 75] = 815 tokens.
        # Step 2: continuation prefill on seqs 0,1 (+530, +390 -> kv_lens [830, 570], crossing
        #         both page=256 and bkv_sz=512) while seqs 2,3 decode (+1 each, placed in 0..2).
        schedules = [
            ([0, 1, 2, 3], [300, 180, 260, 75], [300, 180, 260, 75], [0, 0, 4]),
            ([2, 3, 0, 1], [1, 1, 530, 390], [261, 76, 830, 570], [2, 2, 4]),
        ]
        prepared_steps = []
        for seq_order, q_lens_step, kv_lens_step, dist_step in schedules:
            q_raw = jnp.asarray(
                rng.standard_normal((max_tokens, 64, 128)) * 0.2, jnp.bfloat16
            )
            k_raw = quantize_kv(
                jnp.asarray(
                    rng.standard_normal((max_tokens, 8, 128)) * 0.2, jnp.bfloat16
                ),
                kv_dtype,
                k_scale,
            )
            v_raw = quantize_kv(
                jnp.asarray(
                    rng.standard_normal((max_tokens, 8, 128)) * 0.2, jnp.bfloat16
                ),
                kv_dtype,
                v_scale,
            )
            q_p, kv_m = prepare_inputs(q_raw, k_raw, v_raw)
            q_p = jax.device_put(q_p, q_packed_sharding)
            kv_m = jax.device_put(kv_m, kv_merged_sharding)

            step_pages = np.full(
                (max_seqs, pages_per_seq), total_pages - 1, dtype=np.int32
            )
            step_kv = np.zeros(max_seqs, dtype=np.int32)
            cu_q = np.zeros(max_seqs + 1, dtype=np.int32)
            cur = 0
            for idx, (orig_s, ql, kvl) in enumerate(
                zip(seq_order, q_lens_step, kv_lens_step, strict=True)
            ):
                step_pages[idx] = pages[orig_s]
                step_kv[idx] = kvl
                cur += ql
                cu_q[idx + 1] = cur
            cu_q[len(seq_order) :] = cur

            meta = AttentionMetadata(
                jax.device_put(step_kv, rep_sharding),
                (jax.device_put(step_pages.reshape(-1), rep_sharding),),
                jax.device_put(cu_q, rep_sharding),
                jax.device_put(np.array(dist_step, dtype=np.int32), rep_sharding),
            )
            jax.block_until_ready((q_p, kv_m, meta))
            prepared_steps.append((cur, q_p, kv_m, meta))

        step_records = []
        counter = CompilationCounter()
        try:
            warm = None
            for cur, q_p, kv_m, meta in prepared_steps:
                out_unpacked, cache = mosaic_mixed(q_p, kv_m, cache, meta)
                out_unpacked.block_until_ready()
                if warm is None:
                    warm = counter.count
                else:
                    assert counter.count == warm
                step_records.append((cur, q_p, kv_m, meta, out_unpacked))
        finally:
            del flat_pages
            counter.close()

        for cur, q_p, kv_m, meta, out_unpacked in step_records:
            ref_p, ref_cache = ref_mixed(q_p, kv_m, ref_cache, meta)
            ref_unpacked = prepare_outputs(ref_p, 8, 128)
            out_f32 = np.asarray(out_unpacked[:cur], np.float32)
            assert not np.any(np.isnan(out_f32))
            np.testing.assert_allclose(
                out_f32,
                np.asarray(ref_unpacked[:cur], np.float32),
                rtol=rtol,
                atol=atol,
            )

        cache_host = np.asarray(cache, np.float32)
        ref_cache_host = np.asarray(ref_cache, np.float32)
        for s, final_len in ((0, 830), (1, 570), (2, 261), (3, 76)):
            for pos in (0, 255, 256, final_len - 1):
                if pos < final_len:
                    pg = int(pages[s, pos // 256])
                    off = pos % 256
                    np.testing.assert_array_equal(
                        cache_host[pg, off], ref_cache_host[pg, off]
                    )


class TestBatchedDecodeAttention:
    def test_batched_decode_block_sizes_and_bucket_dispatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        rpa_mod = importlib.import_module(
            "resin.backend.tpu.kernels.ragged_paged_attention"
        )
        monkeypatch.setattr(rpa_mod, "get_tpu_version", lambda: 7)
        monkeypatch.setattr(
            rpa_mod.pltpu,
            "get_tpu_info",
            lambda: type("I", (), {"smem_capacity_bytes": 1024 * 1024})(),
        )
        monkeypatch.setattr(
            rpa_mod,
            "_effective_vmem_limit",
            lambda _v, _tv, vmem: 64 * 1024 * 1024 if vmem is None else vmem,
        )

        # Verify bucket dispatch and lowered TPU7x kernel block selection on
        # Qwen3-32B tp=2 per-die shapes (32 Q heads, 4 KV heads, head_dim=128).
        for seq_bucket, kv_dtype, sw, expected_kernel in (
            (8, jnp.float8_e4m3fn, None, "RPAd-p_256-bq_1_1-bkv_4096_4096"),
            (16, jnp.float8_e4m3fn, None, "RPAd-snh-p256-b8-q1-k1280"),
            (160, jnp.float8_e4m3fn, None, "RPAd-snh-p256-b8-q1-k1280"),
            (16, jnp.bfloat16, None, "RPAd-snh-p256-b8-q1-k512"),
            (16, jnp.float8_e4m3fn, 512, "RPAd-snh-p256-b8-q1-k768-sw512"),
        ):
            cache = jnp.zeros(get_kv_cache_shape(4, 256, 4, 128, kv_dtype), kv_dtype)
            q = jnp.ones((seq_bucket, 32, 128), jnp.bfloat16)
            k = jnp.ones((seq_bucket, 4, 128), kv_dtype)
            jaxpr_str = str(
                jax.make_jaxpr(
                    lambda qi, ki, ci, _sw=sw: rpa_mod.ragged_paged_attention(
                        qi,
                        ki,
                        ki,
                        ci,
                        jnp.ones((seq_bucket,), jnp.int32),
                        jnp.zeros((seq_bucket * 37,), jnp.int32),
                        jnp.arange(seq_bucket + 1, dtype=jnp.int32),
                        jnp.array([1, 1, 1], jnp.int32),
                        sliding_window=_sw,
                        interpret=True,
                    )
                )(q, k, cache)
            )
            assert expected_kernel in jaxpr_str
            assert ("rpa_metadata_schedule" in jaxpr_str) is (seq_bucket >= 16)

        with pytest.raises(ValueError, match="bq_sz == bq_csz == 1"):
            rpa_mod.ragged_paged_attention(
                jnp.ones((4, 8, 128), jnp.bfloat16),
                jnp.ones((4, 2, 128), jnp.bfloat16),
                jnp.ones((4, 2, 128), jnp.bfloat16),
                jnp.zeros(get_kv_cache_shape(4, 8, 2, 128, jnp.bfloat16), jnp.bfloat16),
                jnp.ones((4,), jnp.int32),
                jnp.zeros((8,), jnp.int32),
                jnp.arange(5, dtype=jnp.int32),
                jnp.array([1, 1, 1], jnp.int32),
                d_block_sizes=(2, 8, 2, 8),
                use_batched_decode=True,
                interpret=True,
            )

    def test_batched_decode_vmem_estimate_and_bf16_softmax_variants(self) -> None:
        from resin.backend.tpu.kernels.ragged_paged_attention import (
            _batched_decode_vmem_estimate_bytes,
            _get_batched_decode_block_sizes,
        )

        vmem_64m = 64 * 1024 * 1024

        # Qwen3-32B tp=2 FP8 (32 Q, 4 KV, hd=128, bkv=1280, bsz=8): 2*kv + qk_f32
        # dominates softmax temporaries; bf16 online scratch halves (m, l, acc).
        est_4a = _batched_decode_vmem_estimate_bytes(
            4,
            8,
            128,
            8,
            1280,
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            bf16_qk_softmax=True,
            bf16_online_scratch=False,
        )
        est_4b = _batched_decode_vmem_estimate_bytes(
            4,
            8,
            128,
            8,
            1280,
            jnp.bfloat16,
            jnp.float8_e4m3fn,
            bf16_qk_softmax=True,
            bf16_online_scratch=True,
        )
        assert est_4a == 59_424_768
        assert est_4a - est_4b == (2 * 32 * 128 + 32 * 128) * 2

        # High-GQA shapes step down below 64 MiB and bf16 softmax unlocks larger bkv_sz.
        for q_h, kv_h in ((64, 1), (128, 2)):
            bkv_bf16, bsz_bf16 = _get_batched_decode_block_sizes(
                jnp.bfloat16,
                jnp.float8_e4m3fn,
                q_h,
                kv_h,
                128,
                256,
                37,
                tpu_version=7,
                vmem_limit_bytes=vmem_64m,
                bf16_qk_softmax=True,
            )
            bkv_f32, bsz_f32 = _get_batched_decode_block_sizes(
                jnp.bfloat16,
                jnp.float8_e4m3fn,
                q_h,
                kv_h,
                128,
                256,
                37,
                tpu_version=7,
                vmem_limit_bytes=vmem_64m,
                bf16_qk_softmax=False,
            )
            assert bkv_bf16 > bkv_f32
            assert (
                _batched_decode_vmem_estimate_bytes(
                    kv_h,
                    q_h // kv_h,
                    128,
                    bsz_f32,
                    bkv_f32,
                    jnp.bfloat16,
                    jnp.float8_e4m3fn,
                    bf16_qk_softmax=False,
                )
                < vmem_64m
            )

        # Kernel naming and Pallas interpret numerical parity across f32, 4A, and 4B.
        rng = np.random.default_rng(2026)
        page_size, pages_per_seq, num_seqs, total_pages = 8, 5, 10, 56
        q_heads, kv_heads, head_dim = 16, 2, 128
        sm_scale = head_dim**-0.5
        kv_lens_np = np.array([1, 7, 8, 9, 16, 17, 24, 31, 32, 39], dtype=np.int32)
        pages = np.full((num_seqs, pages_per_seq), total_pages - 1, dtype=np.int32)
        next_page = 1
        for s, kv_l in enumerate(kv_lens_np):
            used = (int(kv_l) + page_size - 1) // page_size
            pages[s, :used] = np.arange(next_page, next_page + used, dtype=np.int32)
            next_page += used
        cu_q_np = np.arange(num_seqs + 1, dtype=np.int32)
        dist_np = np.array([num_seqs, num_seqs, num_seqs], dtype=np.int32)

        q = jnp.asarray(
            rng.standard_normal((num_seqs, q_heads, head_dim)), jnp.bfloat16
        )
        k = jnp.asarray(
            rng.standard_normal((num_seqs, kv_heads, head_dim)), jnp.bfloat16
        )
        v = jnp.asarray(
            rng.standard_normal((num_seqs, kv_heads, head_dim)), jnp.bfloat16
        )

        for kv_dtype in (jnp.float8_e4m3fn, jnp.bfloat16):
            fp8 = kv_dtype == jnp.float8_e4m3fn
            k_scale = 0.5 if fp8 else None
            v_scale = 0.25 if fp8 else None
            cache_shape = get_kv_cache_shape(
                total_pages, page_size, kv_heads, head_dim, kv_dtype
            )
            clean_cache = np.array(
                quantize_kv(
                    jnp.asarray(rng.standard_normal(cache_shape) * 0.25, jnp.float32),
                    kv_dtype,
                ),
                copy=True,
            )
            poisoned_cache = clean_cache.copy()
            nan_val = np.asarray(jnp.full((), jnp.nan, dtype=kv_dtype))
            poisoned_cache[-1] = nan_val
            for s, kv_l in enumerate(kv_lens_np):
                rem = int(kv_l) % page_size
                if rem != 0:
                    used = (int(kv_l) + page_size - 1) // page_size
                    poisoned_cache[pages[s, used - 1], rem:] = nan_val

            ref_out, _ = _reference_rpa(
                np.asarray(q),
                np.asarray(k),
                np.asarray(v),
                clean_cache,
                kv_lens_np,
                pages.reshape(-1),
                cu_q_np,
                dist_np,
                sm_scale=sm_scale,
                sliding_window=20,
                soft_cap=30.0,
                k_scale=k_scale,
                v_scale=v_scale,
            )

            for bf16_qkp, bf16_sc, sfx, atol in (
                (False, False, "-f32p", 3e-2 if fp8 else 2e-2),
                (True, False, "", 3e-2 if fp8 else 2e-2),
                (True, True, "-bf16s", 4e-2 if fp8 else 3e-2),
            ):

                def _call(
                    qi: jax.Array,
                    ki: jax.Array,
                    vi: jax.Array,
                    ci: jax.Array,
                    _qkp: bool = bf16_qkp,
                    _sc: bool = bf16_sc,
                ) -> tuple[jax.Array, jax.Array]:
                    return ragged_paged_attention(
                        qi,
                        ki,
                        vi,
                        ci,
                        jnp.asarray(kv_lens_np),
                        jnp.asarray(pages.reshape(-1)),
                        jnp.asarray(cu_q_np),
                        jnp.asarray(dist_np),
                        sm_scale=sm_scale,
                        sliding_window=20,
                        soft_cap=30.0,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        d_block_sizes=(1, 16, 1, 16),
                        use_batched_decode=True,
                        bf16_qk_softmax=_qkp,
                        bf16_online_scratch=_sc,
                        disable_bounds_checks=False,
                        disable_semaphore_checks=False,
                        interpret=True,
                    )

                jaxpr_str = str(
                    jax.make_jaxpr(_call)(q, k, v, jnp.asarray(poisoned_cache))
                )
                assert f"RPAd-snh-p8-b8-q1-k16{sfx}-sw20" in jaxpr_str
                out, _ = _call(q, k, v, jnp.asarray(poisoned_cache))
                out_f32 = np.asarray(out, np.float32)
                assert not np.any(np.isnan(out_f32))
                np.testing.assert_allclose(
                    out_f32,
                    np.asarray(ref_out, np.float32),
                    rtol=2e-2,
                    atol=atol,
                )

    @pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
    def test_pallas_interpret_batched_decode_ragged_batches_and_page_boundaries(
        self, kv_dtype: jnp.dtype
    ) -> None:
        rng = np.random.default_rng(4242)
        fp8 = kv_dtype == jnp.float8_e4m3fn
        k_scale = 0.5 if fp8 else None
        v_scale = 0.25 if fp8 else None
        atol = 3e-2 if fp8 else 2e-2
        page_size = 8
        pages_per_seq = 4
        max_seqs = 160
        total_pages = max_seqs * pages_per_seq + 4
        q_heads, kv_heads, head_dim = 8, 2, 128
        sm_scale = head_dim**-0.5

        cache_shape = get_kv_cache_shape(
            total_pages, page_size, kv_heads, head_dim, kv_dtype
        )
        raw_cache = rng.standard_normal(cache_shape).astype(np.float32) * 0.25
        clean_cache = np.array(quantize_kv(jnp.asarray(raw_cache), kv_dtype), copy=True)
        clean_cache[0] = 0
        poisoned_cache = clean_cache.copy()
        nan_val = np.asarray(jnp.full((), jnp.nan, dtype=kv_dtype))
        poisoned_cache[-1] = nan_val

        pages = np.full((max_seqs, pages_per_seq), total_pages - 1, dtype=np.int32)
        full_kv_lens = np.zeros(max_seqs, dtype=np.int32)
        next_page = 1
        for s in range(max_seqs):
            kv_l = int(1 + (s * 7 + 5) % 29)
            full_kv_lens[s] = kv_l
            used_p = (kv_l + page_size - 1) // page_size
            pages[s, :used_p] = np.arange(next_page, next_page + used_p, dtype=np.int32)
            rem = kv_l % page_size
            if rem != 0:
                poisoned_cache[next_page + used_p - 1, rem:] = nan_val
            next_page += used_p

        q = jnp.asarray(
            rng.standard_normal((max_seqs, q_heads, head_dim)), jnp.bfloat16
        )
        k = jnp.asarray(
            rng.standard_normal((max_seqs, kv_heads, head_dim)), jnp.bfloat16
        )
        v = jnp.asarray(
            rng.standard_normal((max_seqs, kv_heads, head_dim)), jnp.bfloat16
        )
        flat_pages_jax = jnp.asarray(pages.reshape(-1))
        cache_jax = jnp.asarray(poisoned_cache)

        batch_cases = (1, 8, 9, 32, 154, 160, 0)
        step_inputs = []
        for b_live in batch_cases:
            kv_lens = np.zeros(max_seqs, dtype=np.int32)
            kv_lens[:b_live] = full_kv_lens[:b_live]
            cu_q = np.array(
                [min(i, b_live) for i in range(max_seqs + 1)], dtype=np.int32
            )
            dist = np.array([b_live, b_live, b_live], dtype=np.int32)
            step_inputs.append(
                (
                    b_live,
                    kv_lens,
                    cu_q,
                    dist,
                    jnp.asarray(kv_lens),
                    jnp.asarray(cu_q),
                    jnp.asarray(dist),
                )
            )

        outputs = []
        counter = CompilationCounter()
        try:
            warm = None
            for (
                b_live,
                kv_lens,
                cu_q,
                dist,
                kv_lens_jax,
                cu_q_jax,
                dist_jax,
            ) in step_inputs:
                out, upd_cache = ragged_paged_attention(
                    q,
                    k,
                    v,
                    cache_jax,
                    kv_lens_jax,
                    flat_pages_jax,
                    cu_q_jax,
                    dist_jax,
                    sm_scale=sm_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    d_block_sizes=(1, 8, 1, 8),
                    use_batched_decode=True,
                    disable_bounds_checks=False,
                    disable_semaphore_checks=False,
                    interpret=True,
                )
                out.block_until_ready()
                if warm is None:
                    warm = counter.count
                else:
                    assert counter.count == warm
                outputs.append((b_live, kv_lens, cu_q, dist, out, upd_cache))
        finally:
            counter.close()

        for b_live, kv_lens, cu_q, dist, out, upd_cache in outputs:
            if b_live == 0:
                np.testing.assert_array_equal(np.asarray(upd_cache[0], np.float32), 0.0)
                continue

            ref_out, ref_cache = _reference_rpa(
                np.asarray(q),
                np.asarray(k),
                np.asarray(v),
                clean_cache,
                kv_lens,
                pages.reshape(-1),
                cu_q,
                dist,
                sm_scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            out_f32 = np.asarray(out[:b_live], np.float32)
            assert not np.any(np.isnan(out_f32))
            np.testing.assert_allclose(
                out_f32,
                np.asarray(ref_out[:b_live], np.float32),
                rtol=2e-2,
                atol=atol,
            )
            for s in (0, b_live // 2, b_live - 1):
                pos = int(kv_lens[s]) - 1
                pg = int(pages[s, pos // page_size])
                off = pos % page_size
                np.testing.assert_array_equal(
                    np.asarray(upd_cache[pg, off], np.float32),
                    np.asarray(ref_cache[pg, off], np.float32),
                )

    def test_pallas_interpret_batched_decode_prepared_mixed_sliding_window_and_scales(
        self,
    ) -> None:
        rng = np.random.default_rng(99)
        page_size, pages_per_seq, max_seqs, total_pages = 8, 4, 6, 20
        q_heads, kv_heads, head_dim = 6, 2, 64
        sm_scale = head_dim**-0.5
        cache_shape = get_kv_cache_shape(
            total_pages, page_size, kv_heads, head_dim, jnp.float8_e4m3fn
        )
        raw_cache = rng.standard_normal(cache_shape).astype(np.float32) * 0.3
        clean_cache = np.array(
            quantize_kv(jnp.asarray(raw_cache), jnp.float8_e4m3fn), copy=True
        )
        clean_cache[0] = 0
        poisoned_cache = clean_cache.copy()
        nan_fp8 = np.asarray(jnp.full((), jnp.nan, dtype=jnp.float8_e4m3fn))
        # Seq 0 has kv_len=13, sliding_window=3 (attends 10..12): poison both
        # pre-window slots (0..9) and post-kv_len slots (13..15) with NaNs.
        poisoned_cache[1, :] = nan_fp8
        poisoned_cache[2, :2] = nan_fp8
        poisoned_cache[2, 5:] = nan_fp8

        pages = np.zeros((max_seqs, pages_per_seq), dtype=np.int32)
        pages[0, :2] = [1, 2]
        pages[1, :1] = [3]
        pages[2, :2] = [4, 5]
        kv_lens = np.array([13, 5, 10, 0, 0, 0], dtype=np.int32)
        cu_q = np.array([0, 1, 2, 6, 6, 6, 6], dtype=np.int32)
        dist = np.array([2, 2, 3], dtype=np.int32)

        q_in = jnp.asarray(rng.standard_normal((8, q_heads, head_dim)), jnp.bfloat16)
        k_in = quantize_kv(
            jnp.asarray(rng.standard_normal((8, kv_heads, head_dim)), jnp.bfloat16),
            jnp.float8_e4m3fn,
            0.5,
        )
        v_in = quantize_kv(
            jnp.asarray(rng.standard_normal((8, kv_heads, head_dim)), jnp.bfloat16),
            jnp.float8_e4m3fn,
            0.25,
        )
        q_pack, kv_merg = prepare_inputs(q_in, k_in, v_in)

        out_packed, _ = ragged_paged_attention(
            q_pack,
            kv_merg,
            None,
            jnp.asarray(poisoned_cache),
            jnp.asarray(kv_lens),
            jnp.asarray(pages.reshape(-1)),
            jnp.asarray(cu_q),
            jnp.asarray(dist),
            sm_scale=sm_scale,
            sliding_window=3,
            soft_cap=20.0,
            q_scale=0.5,
            k_scale=0.5,
            v_scale=0.25,
            prepared=True,
            return_packed=True,
            actual_num_q_heads=q_heads,
            actual_head_dim=head_dim,
            d_block_sizes=(1, 8, 1, 8),
            m_block_sizes=(4, 8, 2, 8),
            use_batched_decode=True,
            disable_bounds_checks=False,
            disable_semaphore_checks=False,
            interpret=True,
        )
        out_unpacked = prepare_outputs(out_packed, q_heads // kv_heads, head_dim)
        ref_out, _ = _reference_rpa(
            np.asarray(q_in),
            np.asarray(k_in),
            np.asarray(v_in),
            clean_cache,
            kv_lens,
            pages.reshape(-1),
            cu_q,
            dist,
            sm_scale=sm_scale,
            sliding_window=3,
            soft_cap=20.0,
            q_scale=0.5,
            k_scale=0.5,
            v_scale=0.25,
        )
        out_f32 = np.asarray(out_unpacked[:6], np.float32)
        assert not np.any(np.isnan(out_f32))
        np.testing.assert_allclose(
            out_f32,
            np.asarray(ref_out[:6], np.float32),
            rtol=2e-2,
            atol=3e-2,
        )

    def test_pallas_interpret_batched_decode_multi_smem_chunk_and_exact_capacity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        rpa_mod = importlib.import_module(
            "resin.backend.tpu.kernels.ragged_paged_attention"
        )
        # Force max_steps_ub = 128 so 256x4/8 = 128 hits exact HBM schedule
        # capacity, and 265 sequences (133 steps) straddle chunk 0 -> chunk 1.
        monkeypatch.setattr(rpa_mod, "_DEFAULT_SMEM_BYTES", 40 * 1024)

        rng = np.random.default_rng(77)
        page_size, pages_per_seq = 8, 4
        q_heads, kv_heads, head_dim = 4, 2, 128
        sm_scale = head_dim**-0.5

        for max_seqs, b_live in ((256, 256), (272, 265)):
            total_pages = max_seqs * pages_per_seq + 2
            cache_np = (
                rng.standard_normal(
                    get_kv_cache_shape(
                        total_pages, page_size, kv_heads, head_dim, jnp.bfloat16
                    )
                ).astype(np.float32)
                * 0.2
            )
            cache_np[0] = 0.0
            cache_jax = jnp.asarray(cache_np, jnp.bfloat16)

            kv_lens = np.zeros(max_seqs, dtype=np.int32)
            pages = np.zeros((max_seqs, pages_per_seq), dtype=np.int32)
            nxt = 1
            for s in range(b_live):
                # 4 blocks (29..32 tokens) per sequence; on the 272-seq case give
                # sequence 255 3 blocks so sequence 256 straddles step 127 -> 128.
                kv_l = 22 if (max_seqs == 272 and s == 255) else 30
                kv_lens[s] = kv_l
                used_p = (kv_l + page_size - 1) // page_size
                pages[s, :used_p] = np.arange(nxt, nxt + used_p, dtype=np.int32)
                nxt += used_p

            q = jnp.asarray(
                rng.standard_normal((max_seqs, q_heads, head_dim)), jnp.bfloat16
            )
            k = jnp.asarray(
                rng.standard_normal((max_seqs, kv_heads, head_dim)), jnp.bfloat16
            )
            v = jnp.asarray(
                rng.standard_normal((max_seqs, kv_heads, head_dim)), jnp.bfloat16
            )
            cu_q = np.array(
                [min(i, b_live) for i in range(max_seqs + 1)], dtype=np.int32
            )
            dist = np.array([b_live, b_live, b_live], dtype=np.int32)

            out, _ = rpa_mod.ragged_paged_attention(
                q,
                k,
                v,
                cache_jax,
                jnp.asarray(kv_lens),
                jnp.asarray(pages.reshape(-1)),
                jnp.asarray(cu_q),
                jnp.asarray(dist),
                sm_scale=sm_scale,
                d_block_sizes=(1, 8, 1, 8),
                use_batched_decode=True,
                disable_bounds_checks=False,
                disable_semaphore_checks=False,
                interpret=True,
            )
            ref_out, _ = _reference_rpa(
                np.asarray(q),
                np.asarray(k),
                np.asarray(v),
                np.asarray(cache_jax),
                kv_lens,
                pages.reshape(-1),
                cu_q,
                dist,
                sm_scale=sm_scale,
            )
            out_f32 = np.asarray(out[:b_live], np.float32)
            assert not np.any(np.isnan(out_f32))
            np.testing.assert_allclose(
                out_f32,
                np.asarray(ref_out[:b_live], np.float32),
                rtol=2e-2,
                atol=2e-2,
            )


class TestBatchedMixedAttention:
    def test_batched_mixed_block_sizes_and_bucket_dispatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        rpa_mod = importlib.import_module(
            "resin.backend.tpu.kernels.ragged_paged_attention"
        )
        monkeypatch.setattr(rpa_mod, "get_tpu_version", lambda: 7)
        monkeypatch.setattr(
            rpa_mod.pltpu,
            "get_tpu_info",
            lambda: type("I", (), {"smem_capacity_bytes": 1024 * 1024})(),
        )
        monkeypatch.setattr(
            rpa_mod,
            "_effective_vmem_limit",
            lambda _v, _tv, vmem: 64 * 1024 * 1024 if vmem is None else vmem,
        )

        for tok_bucket, seq_bucket, kv_dtype, sw, expected_mixed in (
            (8, 8, jnp.float8_e4m3fn, None, "RPAm-p_256-"),
            (96, 96, jnp.float8_e4m3fn, None, "RPAm-p_256-"),
            (128, 128, jnp.float8_e4m3fn, None, "RPAm-snh-p256-b2-q128_8-k512"),
            (256, 16, jnp.bfloat16, None, "RPAm-snh-p256-b2-q256_8-k512"),
            (4096, 320, jnp.float8_e4m3fn, None, "RPAm-snh-p256-b2-q256_8-k512"),
            (
                4096,
                320,
                jnp.float8_e4m3fn,
                512,
                "RPAm-snh-p256-b2-q256_8-k512-sw512",
            ),
        ):
            cache = jnp.zeros(get_kv_cache_shape(4, 256, 4, 128, kv_dtype), kv_dtype)
            q = jnp.ones((tok_bucket, 32, 128), jnp.bfloat16)
            k = jnp.ones((tok_bucket, 4, 128), kv_dtype)
            jaxpr_str = str(
                jax.make_jaxpr(
                    lambda qi, ki, ci, _sw=sw: rpa_mod.ragged_paged_attention(
                        qi,
                        ki,
                        ki,
                        ci,
                        jnp.ones((seq_bucket,), jnp.int32),
                        jnp.zeros((seq_bucket * 37,), jnp.int32),
                        jnp.arange(seq_bucket + 1, dtype=jnp.int32),
                        jnp.array([0, 0, 1], jnp.int32),
                        sliding_window=_sw,
                        interpret=True,
                    )
                )(q, k, cache)
            )
            assert expected_mixed in jaxpr_str
            assert ("rpa_mixed_schedule" in jaxpr_str) is (tok_bucket >= 128)

        with pytest.raises(ValueError, match="bkv_csz == bkv_sz"):
            rpa_mod.ragged_paged_attention(
                jnp.ones((16, 8, 128), jnp.bfloat16),
                jnp.ones((16, 2, 128), jnp.bfloat16),
                jnp.ones((16, 2, 128), jnp.bfloat16),
                jnp.zeros(get_kv_cache_shape(4, 8, 2, 128, jnp.bfloat16), jnp.bfloat16),
                jnp.ones((4,), jnp.int32),
                jnp.zeros((8,), jnp.int32),
                jnp.arange(5, dtype=jnp.int32),
                jnp.array([0, 0, 1], jnp.int32),
                m_block_sizes=(8, 16, 4, 8),
                use_batched_mixed=True,
                interpret=True,
            )

    @pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
    def test_pallas_interpret_batched_mixed_continuation_ragged_and_page_boundaries(
        self, kv_dtype: jnp.dtype
    ) -> None:
        rng = np.random.default_rng(5150)
        fp8 = kv_dtype == jnp.float8_e4m3fn
        k_scale = 0.5 if fp8 else None
        v_scale = 0.25 if fp8 else None
        atol = 3e-2 if fp8 else 2e-2

        page_size = 8
        pages_per_seq = 6
        max_seqs = 8
        max_tokens = 40
        total_pages = max_seqs * pages_per_seq + 4
        q_heads, kv_heads, head_dim = 8, 2, 128
        sm_scale = head_dim**-0.5

        cache_shape = get_kv_cache_shape(
            total_pages, page_size, kv_heads, head_dim, kv_dtype
        )
        nan_val = np.asarray(jnp.full((), jnp.nan, dtype=kv_dtype))
        poisoned_cache = np.full(cache_shape, nan_val, dtype=kv_dtype)
        poisoned_cache[0] = 0
        clean_cache = np.zeros(cache_shape, dtype=kv_dtype)

        seq_pages = np.full((max_seqs, pages_per_seq), total_pages - 1, dtype=np.int32)
        nxt = 1
        for s in range(max_seqs):
            seq_pages[s, :5] = np.arange(nxt, nxt + 5, dtype=np.int32)
            nxt += 5

        cache_jax = jnp.asarray(poisoned_cache)
        ref_cache_np = clean_cache.copy()

        # Exercise:
        # 1) Mixed [1, 1, 3]: seq 2 decodes (+1 -> 1) while seqs 0,1 prefill [11, 6]
        # 2) Continuation prefill [0, 0, 2] on seqs 0,1 (+18 -> 29 across 4 pages; +13 -> 19 across 3 pages)
        # 3) Mixed [2, 2, 3]: seqs 0,1 decode (+1 -> 30, 20) while seq 3 prefills [19]
        # 4) Idle [0, 0, 0] batch
        plans = [
            ([2, 0, 1], [1, 11, 6], [1, 11, 6], [1, 1, 3]),
            ([0, 1], [18, 13], [29, 19], [0, 0, 2]),
            ([0, 1, 3], [1, 1, 19], [30, 20, 19], [2, 2, 3]),
            ([], [], [], [0, 0, 0]),
        ]
        step_tensors = []
        for seq_order, q_lens_step, kv_lens_step, dist_step in plans:
            q = jnp.asarray(
                rng.standard_normal((max_tokens, q_heads, head_dim)) * 0.3,
                jnp.bfloat16,
            )
            k = jnp.asarray(
                rng.standard_normal((max_tokens, kv_heads, head_dim)) * 0.3,
                jnp.bfloat16,
            )
            v = jnp.asarray(
                rng.standard_normal((max_tokens, kv_heads, head_dim)) * 0.3,
                jnp.bfloat16,
            )
            step_pages = np.full(
                (max_seqs, pages_per_seq), total_pages - 1, dtype=np.int32
            )
            kv_lens = np.zeros(max_seqs, dtype=np.int32)
            cu_q = np.zeros(max_seqs + 1, dtype=np.int32)
            cur = 0
            for i, (orig_s, ql, kvl) in enumerate(
                zip(seq_order, q_lens_step, kv_lens_step, strict=True)
            ):
                step_pages[i] = seq_pages[orig_s]
                kv_lens[i] = kvl
                cur += ql
                cu_q[i + 1] = cur
            cu_q[len(seq_order) :] = cur
            dist = np.array(dist_step, dtype=np.int32)
            step_tensors.append(
                (
                    cur,
                    q,
                    k,
                    v,
                    step_pages,
                    kv_lens,
                    cu_q,
                    dist,
                    jnp.asarray(step_pages.reshape(-1)),
                    jnp.asarray(kv_lens),
                    jnp.asarray(cu_q),
                    jnp.asarray(dist),
                )
            )

        outputs = []
        counter = CompilationCounter()
        try:
            warm = None
            for (
                cur,
                q,
                k,
                v,
                step_pages,
                kv_lens,
                cu_q,
                dist,
                pages_jax,
                kv_lens_jax,
                cu_q_jax,
                dist_jax,
            ) in step_tensors:
                out, cache_jax = ragged_paged_attention(
                    q,
                    k,
                    v,
                    cache_jax,
                    kv_lens_jax,
                    pages_jax,
                    cu_q_jax,
                    dist_jax,
                    sm_scale=sm_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    d_block_sizes=(1, 8, 1, 8),
                    m_block_sizes=(8, 16, 4, 16),
                    use_batched_decode=True,
                    use_batched_mixed=True,
                    disable_bounds_checks=False,
                    disable_semaphore_checks=False,
                    interpret=True,
                )
                out.block_until_ready()
                if warm is None:
                    warm = counter.count
                else:
                    assert counter.count == warm
                outputs.append(
                    (cur, q, k, v, step_pages, kv_lens, cu_q, dist, out, cache_jax)
                )
        finally:
            counter.close()

        for cur, q, k, v, step_pages, kv_lens, cu_q, dist, out, c_step in outputs:
            if cur == 0:
                np.testing.assert_array_equal(np.asarray(c_step[0], np.float32), 0.0)
                continue

            ref_out, ref_cache_np = _reference_rpa(
                np.asarray(q),
                np.asarray(k),
                np.asarray(v),
                ref_cache_np,
                kv_lens,
                step_pages.reshape(-1),
                cu_q,
                dist,
                sm_scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            out_f32 = np.asarray(out[:cur], np.float32)
            assert not np.any(np.isnan(out_f32))
            np.testing.assert_allclose(
                out_f32,
                np.asarray(ref_out[:cur], np.float32),
                rtol=2e-2,
                atol=atol,
            )

        for s, final_len in ((0, 30), (1, 20), (2, 1), (3, 19)):
            for pos in range(final_len):
                pg = int(seq_pages[s, pos // page_size])
                off = pos % page_size
                np.testing.assert_array_equal(
                    np.asarray(cache_jax[pg, off], np.float32),
                    np.asarray(ref_cache_np[pg, off], np.float32),
                )

    def test_pallas_interpret_batched_mixed_options_and_multi_smem_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib

        rpa_mod = importlib.import_module(
            "resin.backend.tpu.kernels.ragged_paged_attention"
        )
        monkeypatch.setattr(rpa_mod, "_DEFAULT_SMEM_BYTES", 40 * 1024)

        rng = np.random.default_rng(6060)
        page_size, pages_per_seq = 8, 4
        q_heads, kv_heads, head_dim = 6, 2, 64
        sm_scale = head_dim**-0.5

        # 1. Prepared + return_packed + sliding_window + soft_cap + q/k/v scales + bf16_qk_softmax
        total_pages = 16
        cache_shape = get_kv_cache_shape(
            total_pages, page_size, kv_heads, head_dim, jnp.float8_e4m3fn
        )
        raw_cache = rng.standard_normal(cache_shape).astype(np.float32) * 0.25
        clean_cache = np.array(
            quantize_kv(jnp.asarray(raw_cache), jnp.float8_e4m3fn), copy=True
        )
        clean_cache[0] = 0
        poisoned_cache = clean_cache.copy()
        nan_fp8 = np.asarray(jnp.full((), jnp.nan, dtype=jnp.float8_e4m3fn))
        # Seq 0 has kv_cache_len=11, q_len=7 (kv_len=18), sliding_window=5:
        # earliest query pos=11 attends 7..11, so slots 0..6 are pre-window NaNs
        # and slots 18..23 on page 3 are post-kv_len NaNs.
        poisoned_cache[1, :7] = nan_fp8
        poisoned_cache[2, 3:] = nan_fp8
        poisoned_cache[3, :] = nan_fp8

        pages = np.zeros((4, pages_per_seq), dtype=np.int32)
        pages[0, :3] = [1, 2, 3]
        pages[1, :2] = [4, 5]
        poisoned_cache[4, :] = nan_fp8
        poisoned_cache[5, :] = nan_fp8
        kv_lens = np.array([18, 9, 0, 0], dtype=np.int32)
        cu_q = np.array([0, 7, 16, 16, 16], dtype=np.int32)
        dist = np.array([0, 0, 2], dtype=np.int32)

        q_in = jnp.asarray(rng.standard_normal((16, q_heads, head_dim)), jnp.bfloat16)
        k_in = quantize_kv(
            jnp.asarray(rng.standard_normal((16, kv_heads, head_dim)), jnp.bfloat16),
            jnp.float8_e4m3fn,
            0.5,
        )
        v_in = quantize_kv(
            jnp.asarray(rng.standard_normal((16, kv_heads, head_dim)), jnp.bfloat16),
            jnp.float8_e4m3fn,
            0.25,
        )
        q_pack, kv_merg = prepare_inputs(q_in, k_in, v_in)

        for bf16_p in (False, True):
            out_packed, _ = rpa_mod.ragged_paged_attention(
                q_pack,
                kv_merg,
                None,
                jnp.asarray(poisoned_cache),
                jnp.asarray(kv_lens),
                jnp.asarray(pages.reshape(-1)),
                jnp.asarray(cu_q),
                jnp.asarray(dist),
                sm_scale=sm_scale,
                sliding_window=5,
                soft_cap=20.0,
                q_scale=0.5,
                k_scale=0.5,
                v_scale=0.25,
                prepared=True,
                return_packed=True,
                actual_num_q_heads=q_heads,
                actual_head_dim=head_dim,
                d_block_sizes=(1, 8, 1, 8),
                m_block_sizes=(4, 8, 2, 8),
                use_batched_mixed=True,
                bf16_qk_softmax=bf16_p,
                disable_bounds_checks=False,
                disable_semaphore_checks=False,
                interpret=True,
            )
            out_unpacked = prepare_outputs(out_packed, q_heads // kv_heads, head_dim)
            ref_out, _ = _reference_rpa(
                np.asarray(q_in),
                np.asarray(k_in),
                np.asarray(v_in),
                clean_cache,
                kv_lens,
                pages.reshape(-1),
                cu_q,
                dist,
                sm_scale=sm_scale,
                sliding_window=5,
                soft_cap=20.0,
                q_scale=0.5,
                k_scale=0.5,
                v_scale=0.25,
            )
            out_f32 = np.asarray(out_unpacked[:16], np.float32)
            assert not np.any(np.isnan(out_f32))
            np.testing.assert_allclose(
                out_f32,
                np.asarray(ref_out[:16], np.float32),
                rtol=2e-2,
                atol=3e-2,
            )

        # 2. Multi-SMEM-chunk schedule straddling max_steps_ub=128 (> 256 tasks across b=2)
        max_seqs = 36
        max_tokens = max_seqs * 16
        big_pages_total = max_seqs * pages_per_seq + 2
        big_cache = jnp.zeros(
            get_kv_cache_shape(
                big_pages_total, page_size, kv_heads, head_dim, jnp.bfloat16
            ),
            jnp.bfloat16,
        )
        big_kv_lens = np.full(max_seqs, 30, dtype=np.int32)
        big_pages = np.zeros((max_seqs, pages_per_seq), dtype=np.int32)
        for s in range(max_seqs):
            big_pages[s] = np.arange(1 + 4 * s, 5 + 4 * s, dtype=np.int32)
        # Each seq has q_len=16 (4 q-blocks of bq_sz=4), kv_len=30 (4 k-blocks of bkv_sz=8):
        # causal tasks per seq = 3 + 3 + 4 + 4 = 14 -> 36 * 14 = 504 tasks = 252 steps > 128!
        big_cu_q = np.arange(0, (max_seqs + 1) * 16, 16, dtype=np.int32)
        big_dist = np.array([0, 0, max_seqs], dtype=np.int32)
        big_q = jnp.asarray(
            rng.standard_normal((max_tokens, q_heads, head_dim)) * 0.2, jnp.bfloat16
        )
        big_k = jnp.asarray(
            rng.standard_normal((max_tokens, kv_heads, head_dim)) * 0.2, jnp.bfloat16
        )
        big_v = jnp.asarray(
            rng.standard_normal((max_tokens, kv_heads, head_dim)) * 0.2, jnp.bfloat16
        )
        # Pre-populate cached prefix (14 tokens per seq)
        init_big_cache = rng.standard_normal(big_cache.shape).astype(np.float32) * 0.2
        init_big_cache[0] = 0.0
        big_cache_jax = jnp.asarray(init_big_cache, jnp.bfloat16)

        out_big, _ = rpa_mod.ragged_paged_attention(
            big_q,
            big_k,
            big_v,
            big_cache_jax,
            jnp.asarray(big_kv_lens),
            jnp.asarray(big_pages.reshape(-1)),
            jnp.asarray(big_cu_q),
            jnp.asarray(big_dist),
            sm_scale=sm_scale,
            d_block_sizes=(1, 8, 1, 8),
            m_block_sizes=(4, 8, 2, 8),
            use_batched_mixed=True,
            disable_bounds_checks=False,
            disable_semaphore_checks=False,
            interpret=True,
        )
        ref_big, _ = _reference_rpa(
            np.asarray(big_q),
            np.asarray(big_k),
            np.asarray(big_v),
            np.asarray(big_cache_jax),
            big_kv_lens,
            big_pages.reshape(-1),
            big_cu_q,
            big_dist,
            sm_scale=sm_scale,
        )
        out_big_f32 = np.asarray(out_big, np.float32)
        assert not np.any(np.isnan(out_big_f32))
        np.testing.assert_allclose(
            out_big_f32,
            np.asarray(ref_big, np.float32),
            rtol=2e-2,
            atol=2e-2,
        )


def test_pallas_fp8_qkv_norm_rope_pack_interpret() -> None:
    from jax.experimental.pallas import tpu as pltpu

    from resin.backend.tpu.kernels.fp8_matmul import sharded_fp8_matmul
    from resin.backend.tpu.kernels.norm_rope import (
        _half_swap_perm,
        _slice_rope_cache_128,
        pallas_fp8_qkv_norm_rope_pack,
        pallas_qkv_norm_rope_pack,
        sharded_fp8_qkv_norm_rope_pack,
    )

    rng = np.random.default_rng(20261124)
    k_dim = 5120
    num_q_heads_total = 64
    num_kv_heads_total = 8
    head_dim = 128
    max_seq = 512
    eps = 1e-6

    w_qkv = (
        rng.standard_normal(
            ((num_q_heads_total + 2 * num_kv_heads_total) * head_dim, k_dim),
            dtype=np.float32,
        )
        * 0.15
    )
    w_q = jnp.asarray(w_qkv, dtype=jnp.float8_e4m3fn)
    w_scale = jnp.asarray(
        np.abs(
            rng.standard_normal(
                ((num_q_heads_total + 2 * num_kv_heads_total) * head_dim,),
                dtype=np.float32,
            )
        )
        * 0.03
        + 1e-3
    )
    q_norm_w = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(head_dim,)).astype(np.float32), dtype=jnp.bfloat16
    )
    k_norm_w = jnp.asarray(
        rng.uniform(0.5, 1.5, size=(head_dim,)).astype(np.float32), dtype=jnp.bfloat16
    )
    angles = rng.uniform(-1.0, 1.0, size=(max_seq, head_dim // 2)).astype(np.float32)
    cos_half = np.cos(angles)
    sin_half = np.sin(angles)
    cos_full = jnp.asarray(np.concatenate([cos_half, cos_half], axis=-1), jnp.float32)
    sin_signed = jnp.asarray(
        np.concatenate([-sin_half, sin_half], axis=-1), jnp.float32
    )
    rope_cache = (cos_full, sin_signed)

    # Local interpret check across aligned and unaligned decode buckets for both
    # TP=2 (group_size == 2 * num_kv_heads) and TP=1 (group_size == num_kv_heads).
    for local_q_heads, local_kv_heads in ((32, 4), (64, 8)):
        local_out_dim = (local_q_heads + 2 * local_kv_heads) * head_dim
        w_q_local = w_q[:local_out_dim]
        w_s_local = w_scale[:local_out_dim]
        for batch in (1, 7, 8, 32, 320):
            x_q = jnp.asarray(
                rng.standard_normal((batch, k_dim), dtype=np.float32) * 0.5,
                dtype=jnp.float8_e4m3fn,
            )
            x_scale = jnp.asarray(
                np.abs(rng.standard_normal((1, batch), dtype=np.float32)) * 0.04 + 1e-3,
                dtype=jnp.float32,
            )
            positions = jnp.asarray(
                rng.integers(0, max_seq, size=(batch,), dtype=np.int32)
            )
            q_fused, kv_fused = pallas_fp8_qkv_norm_rope_pack(
                x_q,
                x_scale,
                w_q_local,
                w_s_local,
                positions,
                rope_cache,
                q_norm_w,
                k_norm_w,
                jnp.float8_e4m3fn,
                num_q_heads=local_q_heads,
                num_kv_heads=local_kv_heads,
                eps=eps,
                interpret=pltpu.InterpretParams(detect_races=True),
            )
            from resin.backend.tpu.kernels.fp8_matmul import pallas_fp8_matmul

            qkv_ref = pallas_fp8_matmul(
                x_q,
                w_q_local,
                w_s_local,
                x_scale=x_scale,
                out_dtype=jnp.bfloat16,
                interpret=True,
            )
            q_ref, kv_ref = pallas_qkv_norm_rope_pack(
                qkv_ref,
                positions,
                rope_cache,
                q_norm_w,
                k_norm_w,
                jnp.float8_e4m3fn,
                num_q_heads=local_q_heads,
                num_kv_heads=local_kv_heads,
                eps=eps,
                interpret=True,
            )
            np.testing.assert_array_equal(np.asarray(q_fused), np.asarray(q_ref))
            np.testing.assert_array_equal(np.asarray(kv_fused), np.asarray(kv_ref))

    # Sharded TP=2 check with hoisted cos_sin_pos + perm_128 + non-uniform fallback.
    devs = jax.devices()
    if len(devs) >= 2:
        mesh = jax.sharding.Mesh(np.array(devs[:2]), ("tp",))
        rep_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        col_spec = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("tp", None)
        )
        col_1d_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("tp"))
        w_q_sharded = jax.device_put(w_q, col_spec)
        w_s_sharded = jax.device_put(w_scale, col_1d_spec)
        q_norm_rep = jax.device_put(q_norm_w, rep_spec)
        k_norm_rep = jax.device_put(k_norm_w, rep_spec)
        rope_rep = (
            jax.device_put(cos_full, rep_spec),
            jax.device_put(sin_signed, rep_spec),
        )
        for batch in (1, 8, 32, 320):
            x_q_rep = jax.device_put(
                jnp.asarray(
                    rng.standard_normal((batch, k_dim), dtype=np.float32) * 0.5,
                    dtype=jnp.float8_e4m3fn,
                ),
                rep_spec,
            )
            x_s_rep = jax.device_put(
                jnp.asarray(
                    np.abs(rng.standard_normal((1, batch), dtype=np.float32)) * 0.04
                    + 1e-3,
                    dtype=np.float32,
                ),
                rep_spec,
            )
            pos_rep = jax.device_put(
                jnp.asarray(rng.integers(0, max_seq, size=(batch,), dtype=np.int32)),
                rep_spec,
            )
            hoisted_cs = _slice_rope_cache_128(pos_rep, rope_rep)
            hoisted_perm = _half_swap_perm(128, dtype=jnp.bfloat16)

            q_sharded, kv_sharded = sharded_fp8_qkv_norm_rope_pack(
                x_q_rep,
                x_s_rep,
                w_q_sharded,
                w_s_sharded,
                pos_rep,
                rope_rep,
                q_norm_rep,
                k_norm_rep,
                jnp.float8_e4m3fn,
                mesh=mesh,
                num_q_heads=num_q_heads_total,
                num_kv_heads=num_kv_heads_total,
                eps=eps,
                cos_sin_pos=hoisted_cs,
                perm_128=hoisted_perm,
                interpret=True,
            )
            qkv_sharded_ref = sharded_fp8_matmul(
                x_q_rep,
                w_q_sharded,
                w_s_sharded,
                jax.sharding.PartitionSpec("tp", None),
                mesh=mesh,
                x_scale=x_s_rep,
                out_dtype=jnp.bfloat16,
                interpret=True,
            )
            q_sharded_ref, kv_sharded_ref = jax.shard_map(
                lambda qkv_l, p_l, c_l, s_l, qw_l, kw_l: pallas_qkv_norm_rope_pack(
                    qkv_l,
                    p_l,
                    (c_l, s_l),
                    qw_l,
                    kw_l,
                    jnp.float8_e4m3fn,
                    num_q_heads=num_q_heads_total // 2,
                    num_kv_heads=num_kv_heads_total // 2,
                    eps=eps,
                    interpret=True,
                ),
                mesh=mesh,
                in_specs=(
                    jax.sharding.PartitionSpec(None, "tp"),
                    jax.sharding.PartitionSpec(),
                    jax.sharding.PartitionSpec(),
                    jax.sharding.PartitionSpec(),
                    jax.sharding.PartitionSpec(),
                    jax.sharding.PartitionSpec(),
                ),
                out_specs=(
                    jax.sharding.PartitionSpec("tp", None, None, None, None),
                    jax.sharding.PartitionSpec(None, "tp", None, None),
                ),
                check_vma=False,
            )(
                qkv_sharded_ref,
                pos_rep,
                rope_rep[0],
                rope_rep[1],
                q_norm_rep,
                k_norm_rep,
            )
            np.testing.assert_array_equal(
                np.asarray(q_sharded), np.asarray(q_sharded_ref)
            )
            np.testing.assert_array_equal(
                np.asarray(kv_sharded), np.asarray(kv_sharded_ref)
            )

        # Fallback path when group_size is not equal to local_kv_heads or 2*local_kv_heads
        # (e.g. num_q_heads_total=16, num_kv_heads_total=4 on tp=2 -> local_q=8, local_kv=2, group_size=4 == 2*local_kv;
        #  for non-matching ratio use num_q_heads_total=24, num_kv_heads_total=4 on tp=2 -> local_q=12, local_kv=2, group_size=6).
        w_fb = jax.device_put(w_q[: (24 + 2 * 4) * head_dim], col_spec)
        w_s_fb = jax.device_put(w_scale[: (24 + 2 * 4) * head_dim], col_1d_spec)
        x_q_fb = jax.device_put(
            jnp.asarray(
                rng.standard_normal((8, k_dim), dtype=np.float32) * 0.5,
                dtype=jnp.float8_e4m3fn,
            ),
            rep_spec,
        )
        x_s_fb = jax.device_put(
            jnp.asarray(np.ones((1, 8), dtype=np.float32) * 0.02),
            rep_spec,
        )
        pos_fb = jax.device_put(jnp.arange(8, dtype=jnp.int32), rep_spec)
        q_fb, kv_fb = sharded_fp8_qkv_norm_rope_pack(
            x_q_fb,
            x_s_fb,
            w_fb,
            w_s_fb,
            pos_fb,
            rope_rep,
            q_norm_rep,
            k_norm_rep,
            jnp.float8_e4m3fn,
            mesh=mesh,
            num_q_heads=24,
            num_kv_heads=4,
            eps=eps,
            interpret=True,
        )
        assert q_fb.shape == (4, 8, 3, 2, 128)
        assert kv_fb.shape == (8, 2, 4, 128)
