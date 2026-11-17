from __future__ import annotations

import os
from contextlib import closing
from dataclasses import dataclass

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from resin import CacheRetention, Engine, SamplingParams, SchedulerConfig  # noqa: E402
from resin.backend.tpu.arena import KvCacheSpec  # noqa: E402
from resin.backend.tpu.attention import paged_attention, quantize_kv  # noqa: E402
from resin.backend.tpu.buckets import Bucket, ShapeBuckets  # noqa: E402
from resin.backend.tpu.mesh import make_mesh  # noqa: E402
from resin.backend.tpu.metadata import ModelInputs, empty_inputs  # noqa: E402
from resin.backend.tpu.runner import TpuRunner  # noqa: E402

VOCAB = 64
HIDDEN = 32
LAYERS = 2
Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
FFN = 64
MAX_SEQ_LEN = 32


def _rope(x: jax.Array, positions: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    frequencies = 1.0 / (10000.0 ** (jnp.arange(half) / half))
    angles = positions[:, None].astype(jnp.float32) * frequencies[None, :]
    cos, sin = jnp.cos(angles)[:, None, :], jnp.sin(angles)[:, None, :]
    first, second = x[..., :half], x[..., half:]
    return jnp.concatenate([first * cos - second * sin, first * sin + second * cos], -1)


@dataclass(eq=False)
class ToyModel:
    """A tiny GQA transformer that implements the TpuModel protocol.

    With ``window`` set, layer 1 moves to a second cache group on pages twice
    as large, attending over that sliding window.
    """

    mesh: jax.sharding.Mesh
    kv_dtype: object = jnp.float32
    page_tokens: int = 4
    vocab_size: int = VOCAB
    window: int | None = None

    @property
    def cache_specs(self) -> tuple[KvCacheSpec, ...]:
        if self.window is None:
            return (
                KvCacheSpec(
                    LAYERS, KV_HEADS, HEAD_DIM, self.page_tokens, self.kv_dtype
                ),
            )
        return (
            KvCacheSpec(1, KV_HEADS, HEAD_DIM, self.page_tokens, self.kv_dtype),
            KvCacheSpec(
                1,
                KV_HEADS,
                HEAD_DIM,
                2 * self.page_tokens,
                self.kv_dtype,
                sliding_window=self.window,
            ),
        )

    def _window(self, layer: int) -> int | None:
        return self.window if layer == 1 else None

    def init(self, seed: int = 0) -> dict[str, jax.Array]:
        shapes = {
            "embed": ((VOCAB, HIDDEN), P()),
            "head": ((HIDDEN, VOCAB), P(None, "tp")),
            "wq": ((LAYERS, HIDDEN, Q_HEADS * HEAD_DIM), P(None, None, "tp")),
            "wk": ((LAYERS, HIDDEN, KV_HEADS * HEAD_DIM), P(None, None, "tp")),
            "wv": ((LAYERS, HIDDEN, KV_HEADS * HEAD_DIM), P(None, None, "tp")),
            "wo": ((LAYERS, Q_HEADS * HEAD_DIM, HIDDEN), P(None, "tp", None)),
            "w1": ((LAYERS, HIDDEN, FFN), P(None, None, "tp")),
            "w2": ((LAYERS, FFN, HIDDEN), P(None, "tp", None)),
        }
        keys = jax.random.split(jax.random.key(seed), len(shapes))
        params = {}
        for key, (name, (shape, spec)) in zip(keys, shapes.items(), strict=True):
            scale = 1.0 if name == "embed" else shape[-2] ** -0.5
            value = jax.random.normal(key, shape, jnp.float32) * scale
            params[name] = jax.device_put(value, NamedSharding(self.mesh, spec))
        return params

    def _layers(self, params, input_ids, positions, attend):
        x = params["embed"][input_ids]
        caches = []
        for layer in range(LAYERS):
            tokens = x.shape[0]
            q = (x @ params["wq"][layer]).reshape(tokens, Q_HEADS, HEAD_DIM)
            k = (x @ params["wk"][layer]).reshape(tokens, KV_HEADS, HEAD_DIM)
            v = (x @ params["wv"][layer]).reshape(tokens, KV_HEADS, HEAD_DIM)
            out, cache = attend(layer, _rope(q, positions), _rope(k, positions), v)
            caches.append(cache)
            x = x + out.reshape(tokens, -1) @ params["wo"][layer]
            x = x + jax.nn.gelu(x @ params["w1"][layer]) @ params["w2"][layer]
        return x, tuple(caches)

    def hidden_states(self, params, caches, inputs: ModelInputs):
        grouped = self.window is not None

        def attend(layer, q, k, v):
            group = layer if grouped else 0
            return paged_attention(
                q,
                k,
                v,
                caches[group][0 if grouped else layer],
                inputs.attention,
                mesh=self.mesh,
                sm_scale=HEAD_DIM**-0.5,
                group=group,
                sliding_window=self._window(layer),
            )

        hidden, layers = self._layers(
            params, inputs.input_ids, inputs.positions, attend
        )
        if grouped:
            return hidden, tuple((cache,) for cache in layers)
        return hidden, (layers,)

    def logits(self, params, hidden):
        return hidden @ params["head"]

    def _dense(self, params, ids):
        causal = jnp.tril(jnp.ones((MAX_SEQ_LEN, MAX_SEQ_LEN), bool))

        def attend(layer, q, k, v):
            # The cache stores K/V in its dtype, so round them the same way.
            k = quantize_kv(k, self.kv_dtype, None).astype(jnp.float32)
            v = quantize_kv(v, self.kv_dtype, None).astype(jnp.float32)
            k = jnp.repeat(k, Q_HEADS // KV_HEADS, axis=1)
            v = jnp.repeat(v, Q_HEADS // KV_HEADS, axis=1)
            scores = jnp.einsum("thd,shd->hts", q, k) * HEAD_DIM**-0.5
            visible = causal
            window = self._window(layer)
            if window is not None:
                visible &= ~jnp.tril(causal, -window)
            weights = jax.nn.softmax(jnp.where(visible, scores, -jnp.inf), -1)
            return jnp.einsum("hts,shd->thd", weights, v), None

        hidden, _ = self._layers(params, ids, jnp.arange(MAX_SEQ_LEN), attend)
        return self.logits(params, hidden)

    def dense_logits(self, params, token_ids: list[int]) -> np.ndarray:
        """Full causal forward without a cache: the parity oracle.

        It runs at one padded length so the oracle compiles once; floods of
        eager new-shape ops evict the runner's cached traces.
        """

        ids = np.zeros(MAX_SEQ_LEN, np.int32)
        ids[: len(token_ids)] = token_ids
        return np.asarray(_DENSE(self, params, ids))[: len(token_ids)]

    def dense_greedy(self, params, prompt: list[int], count: int) -> list[int]:
        tokens = list(prompt)
        for _ in range(count):
            tokens.append(int(np.argmax(self.dense_logits(params, tokens)[-1])))
        return tokens[len(prompt) :]


_DENSE = jax.jit(ToyModel._dense, static_argnums=0)
BUCKETS = ShapeBuckets(decode_seqs=(2, 4), prefill_tokens=(8, 16, 32))


def _runner(
    tp: int = 1, *, prefill_chunk_size=None, cache_slab_capacity=48, **model_options
):
    devices = jax.devices("cpu")
    if len(devices) < tp:
        pytest.skip(f"needs {tp} host devices; set XLA_FLAGS before importing jax")  # ty: ignore[too-many-positional-arguments]
    mesh = make_mesh(tp, devices)
    model = ToyModel(mesh, **model_options)
    params = model.init()
    runner = TpuRunner(
        model,
        params,
        mesh=mesh,
        max_seq_len=MAX_SEQ_LEN,
        max_num_seqs=4,
        max_batch_tokens=32,
        prefill_chunk_size=prefill_chunk_size,
        buckets=BUCKETS,
        cache_slab_capacity=cache_slab_capacity,
    )
    return runner, model, params


def _engine(runner: TpuRunner) -> Engine:
    return Engine(
        runner,
        max_running_requests=4,
        context_length=MAX_SEQ_LEN,
        scheduler_config=SchedulerConfig(max_batch_tokens=32),
        enable_metrics=False,
    )


def _generate(
    engine: Engine, prompts, *, overlap: bool = True, policies=None, **sampling
) -> list[list[int]]:
    sampling.setdefault("temperature", 0.0)
    policies = policies or [{}] * len(prompts)
    leases = [
        engine.submit(prompt, sampling_params=SamplingParams(**{**sampling, **policy}))
        for prompt, policy in zip(prompts, policies, strict=True)
    ]
    outputs: dict[int, list[int]] = {int(lease): [] for lease in leases}
    while engine.has_work():
        for event in engine.step(allow_overlap=overlap):
            outputs[event.request_id].append(event.token_id)
    return [outputs[int(lease)] for lease in leases]


PROMPTS = [[3, 1, 4, 1, 5, 9, 2], [6, 5], [3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]]


@pytest.fixture(scope="module")
def tp1():
    runner, model, params = _runner(1)
    with closing(_engine(runner)) as engine:
        yield runner, model, params, engine


class TestTpuRunner:
    @pytest.mark.parametrize("overlap", [True, False])
    def test_greedy_generation_matches_dense_forward(self, tp1, overlap) -> None:
        runner, model, params, engine = tp1
        continued = engine.overlap_stats[1]
        outputs = _generate(engine, PROMPTS, overlap=overlap, max_output_tokens=9)
        assert runner.compilations_after_warmup == 0
        assert runner.metrics_snapshot().compilations == 0
        assert outputs == [model.dense_greedy(params, prompt, 9) for prompt in PROMPTS]
        assert (engine.overlap_stats[1] > continued) == overlap

    def test_chunked_prefill_and_tp2_match_dense_forward(self) -> None:
        runner, model, params = _runner(2, prefill_chunk_size=4)
        with closing(_engine(runner)) as engine:
            outputs = _generate(engine, PROMPTS, max_output_tokens=6)
        assert runner.compilations_after_warmup == 0
        assert outputs == [model.dense_greedy(params, prompt, 6) for prompt in PROMPTS]
        inputs = runner._put(empty_inputs(Bucket(2, 2), runner.pages_per_seq))
        logits, _, runner._arena.caches = runner._forward(
            params, runner._arena.caches, inputs
        )
        assert logits.sharding.spec == P(None, "tp")

    def test_two_cache_groups_with_a_sliding_window_match_dense(self) -> None:
        runner, model, params = _runner(1, prefill_chunk_size=4, window=4)
        assert [group.retention for group in runner.cache_layout.groups] == [
            CacheRetention.HISTORY,
            CacheRetention.WINDOW,
        ]
        # Long enough that group 0's 4-token pages outgrow group 1's table.
        with closing(_engine(runner)) as engine:
            outputs = _generate(engine, PROMPTS, max_output_tokens=16)
        assert runner.compilations_after_warmup == 0
        assert outputs == [model.dense_greedy(params, prompt, 16) for prompt in PROMPTS]

    def test_steps_donate_the_cache_arena(self, tp1) -> None:
        runner, _, _, engine = tp1
        before = runner._arena.caches[0][0]
        _generate(engine, PROMPTS[:1], max_output_tokens=2)
        assert before.is_deleted()
        assert not runner._arena.caches[0][0].is_deleted()

    def test_steps_update_the_cache_in_place(self) -> None:
        # A large arena makes one layer's cache dwarf the step's own
        # temporaries, so a hidden copy of any layer shows up.
        runner, _, params = _runner(1, cache_slab_capacity=1024)
        layer = runner._arena.caches[0][0].nbytes
        for bucket in BUCKETS.shapes():
            inputs = runner._put(empty_inputs(bucket, runner.pages_per_seq))
            memory = (
                runner._forward.lower(params, runner._arena.caches, inputs)
                .compile()
                .memory_analysis()
            )
            assert memory.alias_size_in_bytes == runner._arena.nbytes
            assert memory.temp_size_in_bytes < layer

    def test_construction_leaves_jax_configuration_alone(self) -> None:
        names = (
            "jax_compilation_cache_dir",
            "jax_persistent_cache_min_compile_time_secs",
            "jax_persistent_cache_min_entry_size_bytes",
        )
        before = [getattr(jax.config, name) for name in names]
        _runner(1)
        assert [getattr(jax.config, name) for name in names] == before

    def test_logprobs_match_dense_log_softmax(self, tp1) -> None:
        runner, model, params, engine = tp1
        prompt = PROMPTS[0]
        lease = engine.submit(
            prompt,
            sampling_params=SamplingParams(
                temperature=0.0,
                max_output_tokens=3,
                logprobs=True,
                top_logprobs=2,
                prompt_logprobs=True,
            ),
        )
        events = []
        while engine.has_work():
            events.extend(engine.step())
        assert runner.compilations_after_warmup == 0
        tokens = [event.token_id for event in events]
        logits = model.dense_logits(params, prompt + tokens)
        expected = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        for index, event in enumerate(events):
            row = len(prompt) - 1 + index
            assert event.logprobs is not None
            assert event.logprobs.logprob == pytest.approx(
                expected[row, tokens[index]], abs=1e-4
            )
            assert len(event.logprobs.top_logprobs) == 2
        scores = lease.prompt_logprobs
        assert scores is not None and scores[0] is None
        for position in range(1, len(prompt)):
            score = scores[position]
            assert score is not None
            assert score.logprob == pytest.approx(
                expected[position - 1, prompt[position]], abs=1e-4
            )

    def test_greedy_and_seeded_rows_share_a_step(self, tp1) -> None:
        runner, model, params, engine = tp1

        def run(seed):
            return _generate(
                engine,
                PROMPTS[:2],
                max_output_tokens=6,
                policies=[{}, {"temperature": 1.5, "top_k": 8, "seed": seed}],
            )

        runs = [run(seed) for seed in (1, 2, 3, 1)]
        assert all(
            greedy == model.dense_greedy(params, PROMPTS[0], 6) for greedy, _ in runs
        )
        assert runs[0][1] == runs[3][1]
        assert len({tuple(sampled) for _, sampled in runs[:3]}) > 1
        assert runner.compilations_after_warmup == 0

    def test_fp8_kv_cache_matches_the_fp8_rounded_oracle(self) -> None:
        runner, model, params = _runner(1, kv_dtype=jnp.float8_e4m3fn)
        with closing(_engine(runner)) as engine:
            outputs = _generate(engine, PROMPTS, max_output_tokens=9)
        assert runner.compilations_after_warmup == 0
        assert outputs == [model.dense_greedy(params, prompt, 9) for prompt in PROMPTS]

    def test_invalid_requests_fail_before_dispatch_and_stay_retryable(
        self, tp1
    ) -> None:
        _, _, _, engine = tp1
        lease = engine.submit(
            [1, VOCAB + 1],
            sampling_params=SamplingParams(temperature=0.0, max_output_tokens=1),
        )
        with pytest.raises(ValueError, match="out of range"):
            engine.step()
        assert not engine.poisoned
        engine.cancel(lease)
        while engine.has_work():
            engine.step()

    def test_compilation_after_warmup_fails_the_step(self) -> None:
        runner, _, _ = _runner(1)
        compiled = jax.jit(lambda x: x * 3)
        original = runner._samplers[False]

        def recompiling_sampler(logits, policy):
            compiled(jnp.zeros(runner.compilations.count + 1)).block_until_ready()
            return original(logits, policy)

        runner._samplers[False] = recompiling_sampler
        engine = _engine(runner)
        engine.submit(
            [1, 2, 3],
            sampling_params=SamplingParams(temperature=0.0, max_output_tokens=1),
        )
        with pytest.raises(RuntimeError, match="compiled after warmup"):
            while engine.has_work():
                engine.step()
        assert runner.compilations_after_warmup > 0
        with pytest.raises(RuntimeError, match="poisoned"):
            engine.close()

    def test_largest_legal_request_matches_dense_forward(self, tp1) -> None:
        runner, model, params, engine = tp1
        outputs = _generate(engine, [[1] * 30], max_output_tokens=2)
        assert outputs == [model.dense_greedy(params, [1] * 30, 2)]
        assert runner.compilations_after_warmup == 0
