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
        pytest.skip(f"needs {tp} host devices; set XLA_FLAGS before importing jax")
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
        for name in ("qkv", "gate_up"):
            weight = getattr(layer, name)
            assert isinstance(weight, Fp8Weight)
            assert weight.qvalue.dtype == jnp.float8_e4m3fn
            assert weight.qvalue.sharding.spec == P("tp", None)
            assert weight.scale.sharding.spec == P("tp")
        assert layer.qkv.qvalue.shape == (128, 32)
        assert layer.gate_up.qvalue.shape == (128, 32)
        for name in ("o", "down"):
            weight = getattr(layer, name)
            assert weight.qvalue.sharding.spec == P(None, "tp")
            assert weight.scale.sharding.is_fully_replicated
        assert params.embed.sharding.is_fully_replicated
        assert params.lm_head.sharding.spec == P("tp", None)
        assert params.rope_cache is not None
        assert params.rope_cache[0].sharding.is_fully_replicated
        assert params.rope_cache[1].sharding.is_fully_replicated

    def test_tp2_fp8_matches_tp1_with_per_shard_packing_and_sequence_parallel(
        self, checkpoint, tmp_path
    ) -> None:
        directory, _ = checkpoint
        devices = jax.devices("cpu")
        if len(devices) < 2:
            pytest.skip("needs 2 host devices")

        def _prompt_and_next_logits(tp: int, *, sp_min_tokens: int = 512) -> np.ndarray:
            mesh = make_mesh(tp, devices)
            model, params = load_model(
                directory,
                mesh,
                page_tokens=PAGE_TOKENS,
                quantization="fp8",
                kv_dtype=jnp.float32,
            )
            model.sequence_parallel_min_tokens = sp_min_tokens
            runner = TpuRunner(
                model,
                params,
                mesh=mesh,
                max_seq_len=MAX_SEQ_LEN,
                max_num_seqs=4,
                max_batch_tokens=32,
                buckets=BUCKETS,
                cache_slab_capacity=64,
            )
            with closing(_engine(runner)) as engine:
                lease = engine.submit(
                    PROMPTS[2],
                    sampling_params=SamplingParams(
                        temperature=0.0,
                        max_output_tokens=2,
                        logprobs=True,
                        prompt_logprobs=True,
                    ),
                )
                events = []
                while engine.has_work():
                    events.extend(engine.step())
            assert runner.compilations_after_warmup == 0
            prompt_lps = [
                score.logprob
                for score in (lease.prompt_logprobs or [])
                if score is not None
            ]
            step_lps = [ev.logprobs.logprob for ev in events if ev.logprobs is not None]
            return np.array(prompt_lps + step_lps, dtype=np.float32)

        tp1_lps = _prompt_and_next_logits(1)
        tp2_rep_lps = _prompt_and_next_logits(2, sp_min_tokens=512)
        tp2_sp_lps = _prompt_and_next_logits(2, sp_min_tokens=8)
        np.testing.assert_allclose(tp2_rep_lps, tp1_lps, rtol=1e-2, atol=5e-2)
        np.testing.assert_allclose(tp2_sp_lps, tp1_lps, rtol=1e-2, atol=5e-2)

        # Exercise head_dim=128, bfloat16 fused QKV norm+RoPE+pack + prepared=True RPA on TP=2 vs TP=1.
        hd128_dir = tmp_path / "qwen3-hd128"
        hd128_dir.mkdir()
        values = tiny_qwen3_config()
        values.update(
            {
                "head_dim": 128,
                "hidden_size": 256,
                "intermediate_size": 256,
                "torch_dtype": "bfloat16",
                "dtype": "bfloat16",
            }
        )
        torch.manual_seed(11)
        hd_model = (
            transformers.Qwen3ForCausalLM(transformers.Qwen3Config.from_dict(values))
            .to(torch.bfloat16)
            .eval()
        )
        hd_model.save_pretrained(hd128_dir)

        outs = []
        for tp in (1, 2):
            mesh = make_mesh(tp, devices)
            m, p = load_model(
                hd128_dir,
                mesh,
                page_tokens=8,
                quantization="fp8",
                kv_dtype=jnp.float8_e4m3fn,
            )
            m.sequence_parallel_min_tokens = 8
            m._force_fused_qkv_pack_interpret = True
            import functools as _ft

            from resin.backend.tpu.kernels import ragged_paged_attention

            m.impl = _ft.partial(
                ragged_paged_attention,
                d_block_sizes=(1, 8, 1, 8),
                m_block_sizes=(4, 8, 4, 8),
                interpret=True,
            )
            r = TpuRunner(
                m,
                p,
                mesh=mesh,
                max_seq_len=32,
                max_num_seqs=2,
                max_batch_tokens=16,
                buckets=ShapeBuckets(decode_seqs=(2,), prefill_tokens=(8, 16)),
                cache_slab_capacity=32,
            )
            with closing(
                Engine(
                    r,
                    max_running_requests=2,
                    context_length=32,
                    scheduler_config=SchedulerConfig(max_batch_tokens=16),
                    enable_metrics=False,
                )
            ) as eng:
                lease = eng.submit(
                    PROMPTS[0],
                    sampling_params=SamplingParams(
                        temperature=0.0, max_output_tokens=2, logprobs=True
                    ),
                )
                evs = []
                while eng.has_work():
                    evs.extend(eng.step())
                del lease
            assert r.compilations_after_warmup == 0
            outs.append([ev.logprobs.logprob for ev in evs if ev.logprobs is not None])
        np.testing.assert_allclose(outs[1], outs[0], rtol=2e-2, atol=5e-2)

    def test_offline_tpu_v7x_mosaic_mlir_export(self) -> None:
        import unittest.mock as mock

        from jax import export
        from jax._src import tpu_info
        from jax._src.pallas.mosaic import lowering as mosaic_lowering
        from jax.sharding import NamedSharding

        from resin.backend.tpu.arena import KV_PARTITION, KvCacheSpec
        from resin.backend.tpu.kernels import ragged_paged_attention
        from resin.backend.tpu.metadata import AttentionMetadata, ModelInputs
        from resin.models.qwen3.config import Qwen3Config
        from resin_models_tpu.qwen3.model import (
            COLUMN,
            ROW,
            DecodeMatmulMode,
            LayerParams,
            Qwen3Model,
            Qwen3Params,
        )

        devices = jax.devices("cpu")
        if len(devices) < 2:
            pytest.skip("needs 2 host devices")

        cfg = Qwen3Config(
            vocab_size=1024,
            hidden_size=5120,
            intermediate_size=25600,
            num_hidden_layers=2,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            rope_theta=1e6,
            eos_token_id=2,
            bos_token_id=1,
            dtype="bfloat16",
        )
        v7x_info = tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion("7x"), 1)
        mesh = make_mesh(2, devices)
        rep = NamedSharding(mesh, P())
        col = NamedSharding(mesh, COLUMN)
        row = NamedSharding(mesh, ROW)
        sc_col = NamedSharding(mesh, P("tp"))
        cache_sh = NamedSharding(mesh, KV_PARTITION)

        def _fp8_sds(shape, qsh, ssh):
            return Fp8Weight(
                jax.ShapeDtypeStruct(shape, jnp.float8_e4m3fn, sharding=qsh),
                jax.ShapeDtypeStruct((shape[0],), jnp.float32, sharding=ssh),
            )

        lp = LayerParams(
            input_norm=jax.ShapeDtypeStruct((5120,), jnp.bfloat16, sharding=rep),
            q_norm=jax.ShapeDtypeStruct((128,), jnp.bfloat16, sharding=rep),
            k_norm=jax.ShapeDtypeStruct((128,), jnp.bfloat16, sharding=rep),
            post_norm=jax.ShapeDtypeStruct((5120,), jnp.bfloat16, sharding=rep),
            qkv=_fp8_sds((10240, 5120), col, sc_col),
            o=_fp8_sds((5120, 8192), row, rep),
            gate_up=_fp8_sds((51200, 5120), col, sc_col),
            down=_fp8_sds((5120, 25600), row, rep),
        )
        params = Qwen3Params(
            embed=jax.ShapeDtypeStruct((1024, 5120), jnp.bfloat16, sharding=rep),
            layers=(lp, lp),
            norm=jax.ShapeDtypeStruct((5120,), jnp.bfloat16, sharding=rep),
            lm_head=jax.ShapeDtypeStruct((1024, 5120), jnp.bfloat16, sharding=col),
            rope_cache=(
                jax.ShapeDtypeStruct((4096, 128), jnp.float32, sharding=rep),
                jax.ShapeDtypeStruct((4096, 128), jnp.float32, sharding=rep),
            ),
        )
        spec = KvCacheSpec(2, 8, 128, 256, jnp.float8_e4m3fn)
        caches = (
            (
                jax.ShapeDtypeStruct(
                    (64, *spec.page_shape(2)), jnp.float8_e4m3fn, sharding=cache_sh
                ),
                jax.ShapeDtypeStruct(
                    (64, *spec.page_shape(2)), jnp.float8_e4m3fn, sharding=cache_sh
                ),
            ),
        )
        with (
            mock.patch(
                "resin.backend.tpu.kernels.norm_rope.get_tpu_version", return_value=7
            ),
            mock.patch(
                "resin.backend.tpu.kernels.fp8_matmul.get_tpu_version", return_value=7
            ),
            mock.patch(
                "resin.backend.tpu.kernels.collective_fp8_matmul.get_tpu_version",
                return_value=7,
            ),
            mock.patch(
                "resin.backend.tpu.kernels.ragged_paged_attention.get_tpu_version",
                return_value=7,
            ),
            mock.patch(
                "resin.backend.tpu.kernels.ragged_paged_attention.pltpu.get_tpu_info",
                return_value=v7x_info,
            ),
            mock.patch.object(tpu_info, "get_tpu_info", return_value=v7x_info),
            mock.patch.object(
                mosaic_lowering.tpu_info, "get_tpu_info", return_value=v7x_info
            ),
            mock.patch(
                "jax._src.pallas.mosaic.pallas_call_registration.tpu_info.get_tpu_info",
                return_value=v7x_info,
            ),
            mock.patch("resin_models_tpu.qwen3.model.get_tpu_version", return_value=7),
        ):
            export_modes: tuple[DecodeMatmulMode, ...] = (
                "auto",
                "pallas",
                "xla",
            )
            for mode in export_modes:
                model = Qwen3Model(
                    cfg,
                    mesh,
                    jnp.float8_e4m3fn,
                    page_tokens=256,
                    decode_matmul_mode=mode,
                    impl=ragged_paged_attention,
                )
                if mode == "auto":
                    assert model._resolve_prefill_matmul_backend(512) == "xla"
                    assert model._resolve_prefill_matmul_backend(1024) == "xla"
                    assert model._resolve_prefill_matmul_backend(2048) == "pallas"
                    assert model._resolve_prefill_matmul_backend(4096) == "pallas"
                    assert model._resolve_prefill_matmul_backend(8192) == "pallas"

                tokens_to_check = (
                    (
                        (8, 8),
                        (160, 160),
                        (320, 320),
                        (384, 320),
                        (1024, 320),
                        (4096, 320),
                    )
                    if mode == "auto"
                    else (
                        ((8, 8), (16, 16), (96, 96), (160, 160), (320, 320))
                        if mode == "pallas"
                        else ((8, 8), (320, 320))
                    )
                )
                for tokens, seqs in tokens_to_check:
                    inp = ModelInputs(
                        input_ids=jax.ShapeDtypeStruct(
                            (tokens,), jnp.int32, sharding=rep
                        ),
                        positions=jax.ShapeDtypeStruct(
                            (tokens,), jnp.int32, sharding=rep
                        ),
                        logits_indices=jax.ShapeDtypeStruct(
                            (seqs,), jnp.int32, sharding=rep
                        ),
                        attention=AttentionMetadata(
                            jax.ShapeDtypeStruct((seqs,), jnp.int32, sharding=rep),
                            (
                                jax.ShapeDtypeStruct(
                                    (seqs * 9,), jnp.int32, sharding=rep
                                ),
                            ),
                            jax.ShapeDtypeStruct((seqs + 1,), jnp.int32, sharding=rep),
                            jax.ShapeDtypeStruct((3,), jnp.int32, sharding=rep),
                        ),
                    )
                    exported = export.export(
                        jax.jit(model.hidden_states, donate_argnums=(1,)),
                        platforms=["tpu"],
                    )(params, caches, inp)
                    mlir_txt = exported.mlir_module()
                    assert "tpu_custom_call" in mlir_txt
                    assert "func.func private @_run(" not in mlir_txt
                    if tokens <= 320:
                        assert f"quantize_fp8_with_absmax_{tokens}_12800" in mlir_txt
                    else:
                        assert "quantize_fp8_with_absmax_" not in mlir_txt
                    if mode == "xla":
                        assert "fp8_w8a8_matmul_" not in mlir_txt
                        assert "fp8_w8a8_async_ar_" not in mlir_txt
                        assert "stablehlo.all_reduce" in mlir_txt
                    elif mode in ("auto", "pallas") and tokens <= 320:
                        assert f"fp8_w8a8_async_ar_{tokens}_1280_4096" in mlir_txt
                        assert f"fp8_w8a8_matmul_{tokens}_1280_5120" in mlir_txt
                        assert f"fp8_w8a8_async_ar_{tokens}_512_12800_b3" in mlir_txt
                        assert "stablehlo.all_reduce" not in mlir_txt
                    elif mode == "auto" and tokens == 384:
                        assert "fp8_w8a8_async_ar_" not in mlir_txt
                        assert "stablehlo.all_reduce" in mlir_txt
                    elif mode == "auto" and tokens == 1024:
                        assert "fp8_gate_up_swiglu_512_" not in mlir_txt
                        assert "fp8_w8a8_matmul_" not in mlir_txt
                    elif mode == "auto" and tokens == 4096:
                        assert "fp8_gate_up_swiglu_512_1280_5120" in mlir_txt
                        assert "fp8_w8a8_matmul_256_5120_5120" in mlir_txt

    def test_decode_matmul_modes_tp2_vs_tp1_parity_and_zero_pmax(
        self, tmp_path
    ) -> None:
        import functools as _ft

        from resin.backend.tpu.kernels import ragged_paged_attention
        from resin_models_tpu.qwen3.model import DecodeMatmulMode

        devices = jax.devices("cpu")
        if len(devices) < 2:
            pytest.skip("needs 2 host devices")

        hd128_dir = tmp_path / "qwen3-decode-modes"
        hd128_dir.mkdir()
        values = tiny_qwen3_config()
        values.update(
            {
                "head_dim": 128,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "torch_dtype": "bfloat16",
                "dtype": "bfloat16",
            }
        )
        torch.manual_seed(29)
        hd_model = (
            transformers.Qwen3ForCausalLM(transformers.Qwen3Config.from_dict(values))
            .to(torch.bfloat16)
            .eval()
        )
        hd_model.save_pretrained(hd128_dir)

        def _run_mode(
            tp: int, mode: DecodeMatmulMode, *, async_ar: bool = True
        ) -> list[float]:
            mesh = make_mesh(tp, devices)
            m, p = load_model(
                hd128_dir,
                mesh,
                page_tokens=8,
                quantization="fp8",
                kv_dtype=jnp.float8_e4m3fn,
            )
            m.decode_matmul_mode = mode
            m.decode_async_all_reduce = async_ar
            m._force_fused_qkv_pack_interpret = True
            m._force_decode_pallas_interpret = mode in ("auto", "pallas")
            m.impl = _ft.partial(
                ragged_paged_attention,
                d_block_sizes=(1, 8, 1, 8),
                m_block_sizes=(4, 8, 4, 8),
                interpret=True,
            )
            r = TpuRunner(
                m,
                p,
                mesh=mesh,
                max_seq_len=32,
                max_num_seqs=2,
                max_batch_tokens=16,
                buckets=ShapeBuckets(decode_seqs=(8,), prefill_tokens=(16,)),
                cache_slab_capacity=32,
            )
            if tp == 2:
                from resin.backend.tpu.buckets import Bucket
                from resin.backend.tpu.metadata import empty_inputs

                dec_inp = r._put(empty_inputs(Bucket(8, 8), r.pages_per_seq))
                dec_jaxpr_txt = str(
                    jax.make_jaxpr(m.hidden_states)(p, r._arena.caches, dec_inp)
                )
                assert "pmax" not in dec_jaxpr_txt
                if mode in ("auto", "pallas") and async_ar:
                    assert "psum" not in dec_jaxpr_txt
            with closing(
                Engine(
                    r,
                    max_running_requests=2,
                    context_length=32,
                    scheduler_config=SchedulerConfig(max_batch_tokens=16),
                    enable_metrics=False,
                )
            ) as eng:
                lease = eng.submit(
                    PROMPTS[0],
                    sampling_params=SamplingParams(
                        temperature=0.0, max_output_tokens=3, logprobs=True
                    ),
                )
                evs = []
                while eng.has_work():
                    evs.extend(eng.step())
                del lease
            assert r.compilations_after_warmup == 0
            return [ev.logprobs.logprob for ev in evs if ev.logprobs is not None]

        tp1_ref = _run_mode(1, "pallas")
        tp2_psum_lps = _run_mode(2, "pallas", async_ar=False)
        tp2_async_lps = _run_mode(2, "pallas", async_ar=True)
        np.testing.assert_array_equal(tp2_async_lps, tp2_psum_lps)
        for lps in (tp2_async_lps, _run_mode(2, "xla")):
            np.testing.assert_allclose(lps, tp1_ref, rtol=2e-2, atol=5e-2)

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
