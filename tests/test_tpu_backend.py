from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402

from resin import (  # noqa: E402
    BatchPhase,
    BatchPlan,
    CacheGroupLayout,
    CacheGroupPlan,
    CacheLayout,
)
from resin.backend.plans import PagedCachePlan  # noqa: E402
from resin.backend.tpu.arena import CacheArena, KvCacheSpec, cache_layout  # noqa: E402
from resin.backend.tpu.attention import paged_attention, quantize_kv  # noqa: E402
from resin.backend.tpu.buckets import Bucket, ShapeBuckets  # noqa: E402
from resin.backend.tpu.compilation import CompilationCounter  # noqa: E402
from resin.backend.tpu.mesh import make_mesh  # noqa: E402
from resin.backend.tpu.metadata import AttentionMetadata, build_inputs  # noqa: E402
from resin.backend.tpu.reference import ragged_paged_attention  # noqa: E402
from resin.backend.tpu.sampling import SamplingInputs, sample  # noqa: E402


def _cpu_devices(count: int) -> list[jax.Device]:
    devices = jax.devices("cpu")
    if len(devices) < count:
        pytest.skip(f"needs {count} host devices; set XLA_FLAGS before importing jax")  # ty: ignore[too-many-positional-arguments]
    return devices[:count]


class TestShapeBuckets:
    def test_powers_of_two_cover_the_limits(self) -> None:
        buckets = ShapeBuckets.powers_of_two(max_num_seqs=12, max_batch_tokens=300)
        assert buckets.decode_seqs == (8, 12)
        assert buckets.prefill_tokens == (128, 256, 300)
        assert buckets.prefill_seqs == (1, 2, 4, 8, 12)

    def test_select_pads_each_dimension_to_the_next_bucket(self) -> None:
        buckets = ShapeBuckets((2, 4), (8, 16), (1, 2, 4))
        assert buckets.select(BatchPhase.DECODE, 3, 3) == Bucket(4, 4)
        assert buckets.select(BatchPhase.PREFILL, 9, 2) == Bucket(16, 2)
        with pytest.raises(ValueError):
            buckets.select(BatchPhase.PREFILL, 17, 1)
        with pytest.raises(ValueError):
            buckets.select(BatchPhase.DECODE, 5, 5)

    def test_shapes_skip_prefill_buckets_with_more_rows_than_tokens(self) -> None:
        buckets = ShapeBuckets((2,), (2, 4), (1, 4))
        assert list(buckets.shapes()) == [
            Bucket(2, 2),
            Bucket(2, 1),
            Bucket(4, 1),
            Bucket(4, 4),
        ]


def _paged_plan() -> PagedCachePlan:
    plan = BatchPlan(
        sequence_ids=[10, 11],
        context_lengths=[0, 2],
        token_ids=[1, 2, 3],
        token_indptr=[0, 2, 3],
        sampling_mask=[True, True],
        cache_groups=[
            CacheGroupPlan(
                group_index=0,
                block_indptr=[0, 1, 3],
                block_slab_ids=[1, 2, 3],
                block_slotnums=[0, 0, 0],
                block_table_start_indices=[0, 0],
                append_slab_ids=[1, 1, 3],
                append_slotnums=[0, 0, 0],
                append_block_offsets=[0, 1, 0],
                blocks_per_slab=1,
            )
        ],
    )
    layout = CacheLayout(parent_capacity=3, groups=[CacheGroupLayout(2)])
    return PagedCachePlan.from_plan(plan, layout)


class TestMetadata:
    def test_build_inputs_pads_to_the_bucket(self) -> None:
        inputs = build_inputs(_paged_plan(), Bucket(8, 4), (3,))
        assert inputs.input_ids.tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
        assert inputs.positions[:3].tolist() == [0, 1, 2]
        meta = inputs.attention
        assert meta.kv_lens.tolist() == [2, 3, 0, 0]
        assert meta.cu_q_lens.tolist() == [0, 2, 3, 3, 3]
        assert meta.distribution.tolist() == [0, 0, 2]
        assert inputs.logits_indices.tolist() == [1, 2, 0, 0]

    def test_page_table_uses_packed_block_ids_at_absolute_columns(self) -> None:
        table = build_inputs(_paged_plan(), Bucket(8, 4), (3,)).attention.page_indices[
            0
        ]
        assert table.reshape(4, 3).tolist() == [
            [1, 0, 0],
            [2, 3, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

    def test_page_table_rejects_blocks_beyond_its_width(self) -> None:
        with pytest.raises(ValueError, match="beyond"):
            build_inputs(_paged_plan(), Bucket(8, 4), (1,))


class TestArena:
    def test_page_geometry_matches_rpa_v3(self) -> None:
        spec = KvCacheSpec(layers=64, kv_heads=8, head_dim=128)
        assert spec.packing == 4
        assert spec.rows_per_shard(2) == 2
        assert spec.page_shape(2) == (256, 4, 4, 128)
        assert spec.slab_bytes(2) == 32 * 1024**2

    def test_arena_allocates_null_page_plus_slabs_sharded_on_tp(self) -> None:
        mesh = make_mesh(2, _cpu_devices(2))
        spec = KvCacheSpec(layers=2, kv_heads=2, head_dim=8, page_tokens=4)
        layout = cache_layout([spec], parent_capacity=3)
        arena = CacheArena([spec], layout, mesh)
        ((layer, _),) = arena.caches
        bps = layout.groups[0].blocks_per_slab
        assert layer.shape == ((3 + 1) * bps, 4, 2, 4, 128)
        assert isinstance(layer.sharding, NamedSharding)
        assert layer.sharding.spec[2] == "tp"
        assert arena.nbytes == 2 * layer.nbytes


def _random_kv(seed: int, tokens: int, q_heads: int, kv_heads: int, dim: int):
    key_q, key_k, key_v = jax.random.split(jax.random.key(seed), 3)
    return (
        jax.random.normal(key_q, (tokens, q_heads, dim), jnp.float32),
        jax.random.normal(key_k, (tokens, kv_heads, dim), jnp.float32),
        jax.random.normal(key_v, (tokens, kv_heads, dim), jnp.float32),
    )


def _dense_attention(q, k, v, *, window: int | None = None) -> np.ndarray:
    q, k, v = (np.asarray(x, np.float64) for x in (q, k, v))
    tokens, q_heads, dim = q.shape
    k = np.repeat(k, q_heads // k.shape[1], axis=1)
    v = np.repeat(v, q_heads // v.shape[1], axis=1)
    scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(dim)
    index = np.arange(tokens)
    visible = index[None, :] <= index[:, None]
    if window is not None:
        visible &= index[None, :] > index[:, None] - window
    scores = np.where(visible, scores, -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", weights, v)


def _cache(spec: KvCacheSpec, pages: int, fill: float = 0.0) -> jax.Array:
    return jnp.full((pages, *spec.page_shape(1)), fill, spec.dtype)


class TestReferenceAttention:
    # Two sequences of 5 and 3 prompt tokens on 4-token pages, padded to a
    # 12-token, 4-row step. Row 0 owns pages 3 and 1; row 1 owns page 6.
    lengths = (5, 3)
    tables = np.array([[3, 1, 0], [6, 0, 0], [0, 0, 0], [0, 0, 0]], np.int32)
    f32_spec = KvCacheSpec(
        layers=1, kv_heads=2, head_dim=8, page_tokens=4, dtype=jnp.float32
    )

    def _prefill(self, spec, cache, q, k, v, **kwargs):
        cu = np.array([0, 5, 8, 8, 8], np.int32)
        return ragged_paged_attention(
            q,
            quantize_kv(k, spec.dtype, kwargs.get("k_scale")),
            quantize_kv(v, spec.dtype, kwargs.get("v_scale")),
            cache,
            jnp.array([5, 3, 0, 0], jnp.int32),
            jnp.asarray(self.tables.reshape(-1)),
            jnp.asarray(cu),
            jnp.array([0, 0, 2], jnp.int32),
            sm_scale=spec.head_dim**-0.5,
            **kwargs,
        )

    @pytest.mark.parametrize("window", [None, 2])
    def test_prefill_matches_dense_causal_attention(self, window) -> None:
        spec = self.f32_spec
        q, k, v = _random_kv(0, 12, 4, 2, 8)
        out, _ = self._prefill(spec, _cache(spec, 8), q, k, v, sliding_window=window)
        np.testing.assert_allclose(
            out[:5], _dense_attention(q[:5], k[:5], v[:5], window=window), atol=1e-5
        )
        np.testing.assert_allclose(
            out[5:8], _dense_attention(q[5:8], k[5:8], v[5:8], window=window), atol=1e-5
        )
        assert not np.any(np.asarray(out[8:]))

    def test_stale_page_tails_and_padding_never_reach_outputs(self) -> None:
        spec = self.f32_spec
        q, k, v = _random_kv(1, 12, 4, 2, 8)
        clean, _ = self._prefill(spec, _cache(spec, 8), q, k, v)
        poisoned, cache = self._prefill(spec, _cache(spec, 8, np.nan), q, k, v)
        np.testing.assert_array_equal(clean, poisoned)
        # Padding tokens write nothing: page 0 and unowned pages stay NaN.
        assert np.all(np.isnan(np.asarray(cache)[[0, 2, 4, 5, 7]]))

    def test_decode_reads_history_written_by_prefill(self) -> None:
        spec = self.f32_spec
        q, k, v = _random_kv(2, 12, 4, 2, 8)
        _, cache = self._prefill(spec, _cache(spec, 8), q, k, v)
        q2, k2, v2 = _random_kv(3, 4, 4, 2, 8)
        out, _ = ragged_paged_attention(
            q2,
            k2,
            v2,
            cache,
            jnp.array([6, 4, 0, 0], jnp.int32),
            jnp.asarray(self.tables.reshape(-1)),
            jnp.array([0, 1, 2, 2, 2], jnp.int32),
            jnp.array([2, 2, 2], jnp.int32),
            sm_scale=8**-0.5,
        )
        row0 = _dense_attention(
            jnp.concatenate([q[:5], q2[:1]]),
            jnp.concatenate([k[:5], k2[:1]]),
            jnp.concatenate([v[:5], v2[:1]]),
        )
        row1 = _dense_attention(
            jnp.concatenate([q[5:8], q2[1:2]]),
            jnp.concatenate([k[5:8], k2[1:2]]),
            jnp.concatenate([v[5:8], v2[1:2]]),
        )
        np.testing.assert_allclose(out[0], row0[-1], atol=1e-5)
        np.testing.assert_allclose(out[1], row1[-1], atol=1e-5)

    def test_fp8_cache_stores_scaled_saturated_values(self) -> None:
        spec = KvCacheSpec(layers=1, kv_heads=2, head_dim=8, page_tokens=4)
        q, k, v = _random_kv(4, 12, 4, 2, 8)
        k = k.at[0, 0, 0].set(1e6)
        out, cache = self._prefill(
            spec, _cache(spec, 8), q, k, v, k_scale=0.5, v_scale=0.25
        )
        assert cache.dtype == jnp.float8_e4m3fn
        assert float(cache[3, 0, 0, 0, 0].astype(jnp.float32)) == 448.0
        assert np.all(np.isfinite(np.asarray(out)))
        k = k.at[0, 0, 0].set(0.0)
        out, _ = self._prefill(
            spec, _cache(spec, 8), q, k, v, k_scale=0.5, v_scale=0.25
        )
        np.testing.assert_allclose(
            out[1:5], _dense_attention(q[:5], k[:5], v[:5])[1:], atol=0.15
        )


class TestShardedAttention:
    def test_tp2_matches_tp1(self) -> None:
        spec = KvCacheSpec(
            layers=1, kv_heads=2, head_dim=8, page_tokens=4, dtype=jnp.bfloat16
        )
        q, k, v = _random_kv(5, 8, 4, 2, 8)
        metadata = AttentionMetadata(
            jnp.array([5, 3], jnp.int32),
            (jnp.array([3, 1, 6, 0], jnp.int32),),
            jnp.array([0, 5, 8], jnp.int32),
            jnp.array([0, 0, 2], jnp.int32),
        )
        results = []
        for tp in (1, 2):
            mesh = make_mesh(tp, _cpu_devices(tp))
            cache = jnp.zeros((8, *spec.page_shape(tp)), spec.dtype)
            results.append(
                jax.jit(
                    lambda q, k, v, cache, meta, mesh=mesh: paged_attention(
                        q, k, v, cache, meta, mesh=mesh, sm_scale=8**-0.5
                    )
                )(q, k, v, cache, metadata)
            )
        np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-6)


def _policy(rows: int, *, temperature=0.0, top_k=64, top_p=1.0, seed=0, offset=0):
    return SamplingInputs(
        jnp.full(rows, temperature, jnp.float32),
        jnp.full(rows, top_k, jnp.int32),
        jnp.full(rows, top_p, jnp.float32),
        jnp.tile(jnp.array([[0, seed]], jnp.uint32), (rows, 1)),
        jnp.full(rows, offset, jnp.uint32),
    )


class TestSampling:
    logits = jax.random.normal(jax.random.key(0), (4, 64), jnp.float32)

    def test_greedy_is_argmax(self) -> None:
        tokens = sample(self.logits, _policy(4), stochastic=False)
        np.testing.assert_array_equal(tokens, np.argmax(self.logits, -1))

    def test_zero_temperature_rows_stay_greedy_in_stochastic_steps(self) -> None:
        tokens = sample(self.logits, _policy(4), stochastic=True)
        np.testing.assert_array_equal(tokens, np.argmax(self.logits, -1))

    def test_top_k_one_is_greedy(self) -> None:
        tokens = sample(
            self.logits, _policy(4, temperature=2.0, top_k=1), stochastic=True
        )
        np.testing.assert_array_equal(tokens, np.argmax(self.logits, -1))

    def test_draws_are_a_function_of_seed_and_offset(self) -> None:
        def draw(seed, offset):
            policy = _policy(4, temperature=1.0, seed=seed, offset=offset)
            return np.asarray(sample(self.logits, policy, stochastic=True))

        np.testing.assert_array_equal(draw(1, 3), draw(1, 3))
        draws = {tuple(draw(1, offset)) for offset in range(16)}
        assert len(draws) > 1

    def test_top_k_and_top_p_bound_the_support(self) -> None:
        ranked = np.argsort(-np.asarray(self.logits), axis=-1)
        for offset in range(32):
            policy = _policy(4, temperature=1.0, top_k=3, offset=offset)
            tokens = np.asarray(sample(self.logits, policy, stochastic=True))
            assert all(tokens[row] in ranked[row, :3] for row in range(4))
        peaked = self.logits.at[:, 5].set(100.0)
        policy = _policy(4, temperature=1.0, top_p=0.5, offset=7)
        assert np.all(np.asarray(sample(peaked, policy, stochastic=True)) == 5)

    def test_compilation_counter_sees_sharding_only_recompiles(self) -> None:
        devices = _cpu_devices(2)
        counter = CompilationCounter()
        try:
            double = jax.jit(lambda x: x * 2)
            values = np.arange(8.0, dtype=np.float32)
            local = jax.device_put(values, devices[0])
            spec = jax.sharding.PartitionSpec("tp")
            sharded = jax.device_put(values, NamedSharding(make_mesh(2, devices), spec))
            double(local).block_until_ready()
            warm = counter.count
            double(local).block_until_ready()
            assert counter.count == warm
            double(sharded).block_until_ready()
            assert counter.count == warm + 1
        finally:
            counter.close()

    def test_sampling_values_are_not_compilation_keys(self) -> None:
        counter = CompilationCounter()
        try:
            step = jax.jit(
                lambda logits, policy: sample(logits, policy, stochastic=True)
            )
            step(self.logits, _policy(4, temperature=1.0)).block_until_ready()
            warm = counter.count
            assert warm > 0
            step(
                self.logits, _policy(4, temperature=0.3, top_k=5, top_p=0.2, seed=9)
            ).block_until_ready()
            assert counter.count == warm
            step(self.logits[:2], _policy(2, temperature=1.0)).block_until_ready()
            assert counter.count > warm
        finally:
            counter.close()
