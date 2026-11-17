"""Contract, numerical-parity, sharding and compilation-stability tests for TPU kernels."""

from __future__ import annotations

import functools
import os
from typing import Literal

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
            _estimate_vmem_limit_bytes,
            _raw_vmem_estimate_bytes,
        )

        vmem_cap = 48 * 1024 * 1024
        for (batch, n_out, n_in), tile in _TPU7X_FP8_TUNED_BLOCKS.items():
            assert get_tuned_block_sizes(batch, n_out, n_in, tpu_version=7) == tile
            assert batch % tile.batch_block_size == 0
            assert n_out % tile.out_block_size == 0
            assert n_in % tile.in_block_size == 0
            assert tile.batch_block_size % 8 == 0
            assert tile.out_block_size % 128 == 0
            assert tile.in_block_size % 128 == 0
            assert batch == tile.batch_block_size or tile.batch_block_size % 128 == 0

            nb = batch // tile.batch_block_size
            no = n_out // tile.out_block_size
            ni = n_in // tile.in_block_size
            save_acc = ni > 1
            for x_dtype, quant in ((jnp.float8_e4m3fn, False), (jnp.bfloat16, True)):
                save_xq = (
                    quant
                    and no > 1
                    and (
                        ni * tile.batch_block_size * tile.in_block_size
                        <= 4 * 1024 * 1024
                    )
                )
                est = _estimate_vmem_limit_bytes(
                    n_batch=nb,
                    n_out=no,
                    n_in=ni,
                    batch_block_size=tile.batch_block_size,
                    out_block_size=tile.out_block_size,
                    in_block_size=tile.in_block_size,
                    x_dtype=x_dtype,
                    x_q_dtype=jnp.float8_e4m3fn,
                    w_q_dtype=jnp.float8_e4m3fn,
                    out_dtype=jnp.bfloat16,
                    save_acc=save_acc,
                    save_x_q=save_xq,
                    tpu_version=7,
                )
                assert est <= vmem_cap
                if batch <= 320:
                    raw = _raw_vmem_estimate_bytes(
                        n_batch=nb,
                        n_out=no,
                        n_in=ni,
                        batch_block_size=tile.batch_block_size,
                        out_block_size=tile.out_block_size,
                        in_block_size=tile.in_block_size,
                        x_dtype=x_dtype,
                        x_q_dtype=jnp.float8_e4m3fn,
                        w_q_dtype=jnp.float8_e4m3fn,
                        out_dtype=jnp.bfloat16,
                        save_acc=save_acc,
                        save_x_q=save_xq,
                    )
                    assert raw <= vmem_cap
                    assert tile.batch_block_size == batch
                    assert tile.out_block_size % 256 == 0

        buckets = {
            bucket.tokens
            for workload in TPU_WORKLOADS.values()
            for bucket in _tpu_buckets(workload, workload.max_batch_tokens).shapes()
        }
        tp2_shapes = (
            (4096, 5120),
            (512, 5120),
            (5120, 4096),
            (5120, 5120),
            (5120, 12800),
            (12800, 5120),
            (25600, 5120),
        )
        for batch in buckets:
            for n_out, n_in in tp2_shapes:
                assert (batch, n_out, n_in) in _TPU7X_FP8_TUNED_BLOCKS
                if batch <= 320:
                    tile = _TPU7X_FP8_TUNED_BLOCKS[(batch, n_out, n_in)]
                    assert tile.in_block_size == n_in

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
        w_up = quantize_fp8_weight(
            jnp.asarray(rng.standard_normal((256, 256)) * 0.02, jnp.bfloat16)
        )
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
                ("gspmd_xla", False),
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

        # 1. Validate every _TPU7X_RPA_MIXED_TUNED_BLOCKS entry for divisibility,
        # page alignment, and <= 48 MiB VMEM cap.
        vmem_cap = 48 * 1024 * 1024
        assert len(_TPU7X_RPA_MIXED_TUNED_BLOCKS) > 0
        for key, blocks in _TPU7X_RPA_MIXED_TUNED_BLOCKS.items():
            page_size, kv_name, q_heads, kv_heads, aligned_hd, q_bucket, kv_bucket = key
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


class TestDecodeMatmulPath:
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
        assert all(eqn.primitive.name != "pad" for eqn in jaxpr.jaxpr.eqns)

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
        assert all(
            eqn.primitive.name not in ("pad", "transpose")
            for eqn in norm_jaxpr.jaxpr.eqns
        )

        add_norm_jaxpr = jax.make_jaxpr(
            lambda xi, ri: fused_add_rms_norm_fp8_quant(
                xi, ri, nw, 1e-6, transposed_scale=True, interpret=True
            )
        )(x, res)
        assert all(
            eqn.primitive.name not in ("pad", "transpose")
            for eqn in add_norm_jaxpr.jaxpr.eqns
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
        assert all(eqn.primitive.name != "pad" for eqn in qkv_jaxpr.jaxpr.eqns)

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

        # Unpack attention + FP8 quant across reshape & bitcast_u32 with zero pad primitives.
        out_packed = pack_outputs(q_ref, kv_heads)
        ref_unpacked = prepare_outputs(
            out_packed, q_heads // kv_heads, head_dim
        ).reshape(batch, q_heads * head_dim)
        ref_att_q, ref_att_s = quantize_fp8_act(ref_unpacked)

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
            assert all(eqn.primitive.name != "pad" for eqn in unpack_jaxpr.jaxpr.eqns)
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

    def test_per_die_scaling_zero_pmax_lower_l2_error_and_overlap_modes(self) -> None:
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
        np.testing.assert_allclose(
            np.asarray(y_local_xla, np.float32),
            np.asarray(y_local, np.float32),
            rtol=1e-3,
            atol=1e-2,
        )

        overlap_cases: tuple[tuple[MatmulBackend, int, bool, bool], ...] = (
            ("pallas", 2, False, True),
            ("xla", 2, False, False),
            ("gspmd_xla", 1, True, False),
        )
        for backend, n_split, sync, interp in overlap_cases:
            y_cand = sharded_fp8_matmul(
                x_sh,
                wq_sh,
                ws_rep,
                P(None, "tp"),
                mesh=mesh,
                sync_tp_scale=sync,
                n_split=n_split,
                backend=backend,
                interpret=interp,
            )
            arr = np.asarray(y_cand, np.float32)
            assert not np.any(np.isnan(arr))
            np.testing.assert_allclose(arr[26:], 0.0, atol=0.0)
            ref_target = y_synced if sync else y_local
            np.testing.assert_allclose(
                arr, np.asarray(ref_target, np.float32), rtol=2e-2, atol=5e-2
            )

        # Also verify sharded_fp8_gate_up_swiglu(..., sync_tp_scale=False) emits zero pmax.
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
            sharded_fp8_gate_up_swiglu,
            sharded_fp8_matmul_all_gather,
            sharded_fp8_matmul_reduce_scatter,
        )

        mesh = make_mesh(min(len(jax.devices()), 2))
        tp = int(mesh.shape["tp"])
        rng = np.random.default_rng(2027)
        tokens = 512

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

        def _mlp_pallas(xq_i, xs_i):
            g_out, g_abs = sharded_fp8_gate_up_swiglu(
                xq_i, xs_i, w_gu_q, w_gu_s, mesh=mesh, backend="pallas"
            )
            return sharded_fp8_matmul_reduce_scatter(
                g_out,
                w_d_q,
                w_d_s,
                mesh=mesh,
                x_abs_max=g_abs,
                backend="pallas",
            )

        out_pallas = jax.jit(_mlp_pallas)(xq_sh, xs_sh)
        out_xla = jax.jit(
            lambda xq_i, xs_i: sharded_fp8_matmul_reduce_scatter(
                sharded_fp8_gate_up_swiglu(
                    xq_i, xs_i, w_gu_q, w_gu_s, mesh=mesh, backend="xla"
                )[0],
                w_d_q,
                w_d_s,
                mesh=mesh,
                backend="xla",
            )
        )(xq_sh, xs_sh)
        assert not np.any(np.isnan(np.asarray(out_pallas, np.float32)))
        np.testing.assert_allclose(
            np.asarray(out_pallas, np.float32),
            np.asarray(out_xla, np.float32),
            rtol=3e-2,
            atol=5e-2,
        )

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

        xq, xs_t, next_res = jax.jit(
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
        cache = jax.device_put(
            jnp.zeros((16, *spec.page_shape(tp)), jnp.float8_e4m3fn), cache_sh
        )
        ref_cache = jax.device_put(
            jnp.zeros((16, *spec.page_shape(tp)), jnp.float8_e4m3fn), cache_sh
        )
        pages = np.zeros((tokens, 4), dtype=np.int32)
        for i in range(live):
            pages[i, :2] = [(i % 7) + 1, ((i + 3) % 7) + 8]
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

        def _dec_mlp_pallas(xqi, xsi):
            g_out, g_abs = sharded_fp8_gate_up_swiglu(
                xqi,
                xsi,
                w_gu_q,
                w_gu_s,
                mesh=mesh,
                sequence_parallel=False,
                sync_tp_scale=False,
                backend="pallas",
            )
            return sharded_fp8_matmul(
                g_out,
                w_d_q,
                w_d_s,
                P(None, "tp"),
                mesh=mesh,
                x_abs_max=g_abs,
                sync_tp_scale=False,
                backend="pallas",
            )

        mlp_mosaic = jax.jit(_dec_mlp_pallas)(xq, xs_t)
        mlp_ref = jax.jit(
            lambda xqi, xsi: sharded_fp8_matmul(
                sharded_fp8_gate_up_swiglu(
                    xqi,
                    xsi,
                    w_gu_q,
                    w_gu_s,
                    mesh=mesh,
                    sequence_parallel=False,
                    sync_tp_scale=False,
                    backend="xla",
                )[0],
                w_d_q,
                w_d_s,
                P(None, "tp"),
                mesh=mesh,
                sync_tp_scale=False,
                backend="xla",
            )
        )(xq, xs_t)
        assert not np.any(np.isnan(np.asarray(mlp_mosaic, np.float32)))
        np.testing.assert_allclose(
            np.asarray(mlp_mosaic, np.float32)[live:], 0.0, atol=0.0
        )
        np.testing.assert_allclose(
            np.asarray(mlp_mosaic, np.float32),
            np.asarray(mlp_ref, np.float32),
            rtol=3e-2,
            atol=5e-2,
        )
