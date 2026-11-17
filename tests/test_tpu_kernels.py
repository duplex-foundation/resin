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
        buckets = (8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320)
        buckets += (512, 1024, 2048, 4096)
        shapes = (
            (4096, 5120),
            (512, 5120),
            (5120, 4096),
            (5120, 5120),
            (5120, 12800),
            (12800, 5120),
            (25600, 5120),
        )
        for batch in buckets:
            for n_out, n_in in shapes:
                tile = _TPU7X_FP8_TUNED_BLOCKS[(batch, n_out, n_in)]
                assert get_tuned_block_sizes(batch, n_out, n_in, tpu_version=7) == tile
                assert batch % tile.batch_block_size == 0
                assert n_out % tile.out_block_size == 0
                assert n_in % tile.in_block_size == 0
                assert tile.batch_block_size % 8 == 0
                assert tile.out_block_size % 128 == 0
                assert tile.in_block_size % 128 == 0
                assert (
                    batch == tile.batch_block_size or tile.batch_block_size % 128 == 0
                )

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
            pytest.skip("needs 2 host devices")  # ty: ignore[too-many-positional-arguments]
        mesh = Mesh(np.array(devices[:2]), ("tp",))

        rng = np.random.default_rng(202)
        x = jnp.asarray(rng.standard_normal((16, 256)), jnp.bfloat16)
        # Put asymmetric peak magnitudes in the second half of K so row-parallel
        # shards diverge unless lax.pmax synchronizes x_absmax across tp.
        x = x.at[:, 128:].multiply(10.0)
        w = jnp.asarray(rng.standard_normal((256, 256)), jnp.bfloat16)
        fw = quantize_fp8_weight(w)
        single = xla_fp8_matmul(x, fw.qvalue, fw.scale)

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
            row = sharded_fp8_matmul(
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
            np.testing.assert_allclose(
                np.asarray(col, np.float32),
                np.asarray(single, np.float32),
                rtol=1e-2,
                atol=2e-2,
            )
            np.testing.assert_allclose(
                np.asarray(row, np.float32),
                np.asarray(single, np.float32),
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

        for small in (8, 16):
            x_small = jnp.asarray(x_np[:small], jnp.bfloat16)
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
                np.asarray(small_pallas, np.float32),
                np.asarray(small_xla, np.float32),
                rtol=2e-2,
                atol=5e-2,
            )
            small_row = jax.jit(
                lambda inp, wq, ws: sharded_fp8_matmul(
                    inp, wq, ws, P(None, "tp"), mesh=mesh, backend="pallas"
                )
            )(
                jax.device_put(small_pallas, NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(fw_row.qvalue, NamedSharding(mesh, P(None, "tp"))),
                jax.device_put(fw_row.scale, NamedSharding(mesh, P())),
            )
            small_row_xla = xla_fp8_matmul(small_pallas, fw_row.qvalue, fw_row.scale)
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
