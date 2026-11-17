"""CPU parity of the Qwen3 TPU plug-in against Transformers on the tiny config.

Full-size BF16 parity (Qwen3-32B logits fixtures) is a separate follow-up; these
tests pin the layer graph, weight mapping, sharding and cache use with a
random-weight float32 model that ``transformers.Qwen3ForCausalLM`` can run.
"""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

import jax.numpy as jnp  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from resin import Engine, SamplingParams, SchedulerConfig  # noqa: E402
from resin.backend.tpu.buckets import ShapeBuckets  # noqa: E402
from resin.backend.tpu.kernels import Fp8Weight  # noqa: E402
from resin.backend.tpu.mesh import make_mesh  # noqa: E402
from resin.backend.tpu.runner import TpuRunner  # noqa: E402
from resin.models.qwen3.config import tiny_qwen3_config  # noqa: E402
from resin_models_tpu.qwen3 import load_model  # noqa: E402

MAX_SEQ_LEN = 64
PAGE_TOKENS = 4
BUCKETS = ShapeBuckets(decode_seqs=(2, 4), prefill_tokens=(8, 16, 32))
PROMPTS = [[3, 1, 4, 1, 5, 9, 2], [6, 5], [3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 7, 1]]


def _reference_checkpoint(directory: Path, *, tied: bool = False) -> torch.nn.Module:
    """Save a random-weight tiny Qwen3 and return the Transformers reference."""

    values = tiny_qwen3_config()
    values["tie_word_embeddings"] = tied
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config.from_dict(values)
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 1:
                parameter.copy_(1.0 + 0.25 * torch.randn_like(parameter))
            else:
                parameter.normal_(0.0, 0.5 / parameter.shape[-1] ** 0.5 * 4)
    model.save_pretrained(directory)
    return model


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[Path, torch.nn.Module]:
    directory = tmp_path_factory.mktemp("qwen3-tiny")
    return directory, _reference_checkpoint(directory)


def _reference_logits(model: torch.nn.Module, tokens: list[int]) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor([tokens])).logits[0].numpy()


def _reference_greedy(model: torch.nn.Module, prompt: list[int], count: int):
    tokens = list(prompt)
    for _ in range(count):
        tokens.append(int(np.argmax(_reference_logits(model, tokens)[-1])))
    return tokens[len(prompt) :]


def _runner(directory: Path, tp: int = 1, *, prefill_chunk_size=None, **options):
    devices = jax.devices("cpu")
    if len(devices) < tp:
        pytest.skip(f"needs {tp} host devices; set XLA_FLAGS before importing jax")  # ty: ignore[too-many-positional-arguments]
    mesh = make_mesh(tp, devices)
    options.setdefault("quantization", "none")
    options.setdefault("kv_dtype", jnp.float32)
    model, params = load_model(directory, mesh, page_tokens=PAGE_TOKENS, **options)
    runner = TpuRunner(
        model,
        params,
        mesh=mesh,
        max_seq_len=MAX_SEQ_LEN,
        max_num_seqs=4,
        max_batch_tokens=32,
        prefill_chunk_size=prefill_chunk_size,
        buckets=BUCKETS,
        cache_slab_capacity=64,
    )
    return runner, params


def _engine(runner: TpuRunner) -> Engine:
    return Engine(
        runner,
        max_running_requests=4,
        context_length=MAX_SEQ_LEN,
        scheduler_config=SchedulerConfig(max_batch_tokens=32),
        enable_metrics=False,
    )


def _generate(engine: Engine, prompts, **sampling) -> list[list[int]]:
    sampling.setdefault("temperature", 0.0)
    leases = [
        engine.submit(prompt, sampling_params=SamplingParams(**sampling))
        for prompt in prompts
    ]
    outputs: dict[int, list[int]] = {int(lease): [] for lease in leases}
    while engine.has_work():
        for event in engine.step():
            outputs[event.request_id].append(event.token_id)
    return [outputs[int(lease)] for lease in leases]


class TestGreedyParity:
    @pytest.mark.parametrize("tp", [1, 2])
    def test_generation_matches_transformers(self, checkpoint, tp) -> None:
        directory, reference = checkpoint
        # Chunks of 5 tokens straddle the 4-token pages and continue prefill.
        runner, _ = _runner(directory, tp, prefill_chunk_size=5)
        with closing(_engine(runner)) as engine:
            outputs = _generate(engine, PROMPTS, max_output_tokens=10)
        assert runner.compilations_after_warmup == 0
        assert outputs == [_reference_greedy(reference, p, 10) for p in PROMPTS]

    def test_logprobs_match_transformers(self, checkpoint) -> None:
        directory, reference = checkpoint
        runner, _ = _runner(directory)
        prompt = PROMPTS[0]
        with closing(_engine(runner)) as engine:
            lease = engine.submit(
                prompt,
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_output_tokens=4,
                    logprobs=True,
                    prompt_logprobs=True,
                ),
            )
            events = []
            while engine.has_work():
                events.extend(engine.step())
        tokens = [event.token_id for event in events]
        logits = _reference_logits(reference, prompt + tokens)
        expected = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        for index, event in enumerate(events):
            assert event.logprobs is not None
            assert event.logprobs.logprob == pytest.approx(
                expected[len(prompt) - 1 + index, tokens[index]], abs=1e-3
            )
        scores = lease.prompt_logprobs
        assert scores is not None
        for position in range(1, len(prompt)):
            score = scores[position]
            assert score is not None
            assert score.logprob == pytest.approx(
                expected[position - 1, prompt[position]], abs=1e-3
            )

    def test_tied_embeddings_match_transformers(self, tmp_path) -> None:
        reference = _reference_checkpoint(tmp_path, tied=True)
        runner, params = _runner(tmp_path)
        assert not (tmp_path / "lm_head.weight").exists()
        with closing(_engine(runner)) as engine:
            outputs = _generate(engine, PROMPTS[:2], max_output_tokens=6)
        assert outputs == [_reference_greedy(reference, p, 6) for p in PROMPTS[:2]]
        assert params.lm_head.shape == params.embed.shape


class TestFp8Weights:
    def test_linear_weights_are_quantized_and_sharded(self, checkpoint) -> None:
        directory, _ = checkpoint
        _, params = _runner(directory, 2, quantization="fp8")
        layer = params.layers[0]
        for name in ("q", "k", "v", "gate", "up"):
            weight = getattr(layer, name)
            assert isinstance(weight, Fp8Weight)
            assert weight.qvalue.dtype == jnp.float8_e4m3fn
            assert weight.qvalue.sharding.spec == P("tp", None)
            assert weight.scale.sharding.spec == P("tp")
        for name in ("o", "down"):
            weight = getattr(layer, name)
            assert weight.qvalue.sharding.spec == P(None, "tp")
            assert weight.scale.sharding.is_fully_replicated
        assert params.embed.sharding.is_fully_replicated
        assert params.lm_head.sharding.spec == P("tp", None)

    def test_logits_track_the_reference(self, checkpoint) -> None:
        directory, reference = checkpoint
        runner, _ = _runner(directory, 2, quantization="fp8")
        prompt = PROMPTS[0]
        with closing(_engine(runner)) as engine:
            engine.submit(
                prompt,
                sampling_params=SamplingParams(
                    temperature=0.0, max_output_tokens=1, logprobs=True
                ),
            )
            events = []
            while engine.has_work():
                events.extend(engine.step())
        assert runner.compilations_after_warmup == 0
        logits = _reference_logits(reference, prompt)[-1]
        expected = logits - np.log(np.exp(logits).sum())
        assert events[0].logprobs is not None
        assert events[0].token_id == int(np.argmax(logits))
        assert events[0].logprobs.logprob == pytest.approx(
            expected[events[0].token_id], abs=0.1
        )

    def test_default_cache_is_fp8_and_generates_without_recompiling(
        self, checkpoint
    ) -> None:
        directory, _ = checkpoint
        runner, _ = _runner(directory, quantization="fp8", kv_dtype=None)
        assert runner.benchmark_metadata["kv_cache_dtype"] == "float8_e4m3fn"
        with closing(_engine(runner)) as engine:
            outputs = _generate(engine, PROMPTS, max_output_tokens=4)
        assert [len(tokens) for tokens in outputs] == [4, 4, 4]
        assert runner.compilations_after_warmup == 0


class TestCheckpointValidation:
    def _copy(self, source: Path, target: Path, drop: str | None = None) -> Path:
        tensors = safetensors_torch.load_file(source / "model.safetensors")
        if drop is not None:
            del tensors[drop]
        safetensors_torch.save_file(tensors, target / "model.safetensors")
        (target / "config.json").write_text((source / "config.json").read_text())
        return target

    def test_missing_tensor_is_named(self, checkpoint, tmp_path) -> None:
        directory, _ = checkpoint
        name = "model.layers.1.self_attn.q_norm.weight"
        broken = self._copy(directory, tmp_path, drop=name)
        with pytest.raises(KeyError, match=name):
            load_model(broken, make_mesh(1, jax.devices("cpu")), quantization="none")

    def test_wrong_shape_is_rejected(self, checkpoint, tmp_path) -> None:
        directory, _ = checkpoint
        config = json.loads((directory / "config.json").read_text())
        config["num_attention_heads"] = 8
        broken = self._copy(directory, tmp_path)
        (broken / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="q_proj.*shape"):
            load_model(broken, make_mesh(1, jax.devices("cpu")), quantization="none")

    def test_unknown_quantization_is_rejected(self, checkpoint) -> None:
        directory, _ = checkpoint
        with pytest.raises(ValueError, match="quantization"):
            load_model(
                directory,
                make_mesh(1, jax.devices("cpu")),
                quantization="int4",  # ty: ignore[invalid-argument-type]
            )
