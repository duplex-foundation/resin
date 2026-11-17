"""Qwen3 layer graph for the TPU backend.

A dense decoder with grouped-query attention, per-head RMSNorm on Q and K
before RoPE, and a SwiGLU MLP. Attention and sampling come from
``resin.backend.tpu``; this module only wires layers, so it owns no request or
cache state. Linear weights are ``[out, in]`` as in the checkpoint, either
dense or FP8 with per-output-channel scales, with ``(q, k, v)`` packed per
``tp`` shard into ``qkv`` and ``(gate, up)`` packed per ``tp`` shard into
``gate_up``.

Tensor parallelism splits heads and the MLP width over ``tp``. ``qkv`` and
``gate_up`` are column-parallel (``P("tp", None)``); ``o`` and ``down`` are
row-parallel (``P(None, "tp")``). On FP8 prefill steps (``tokens >= 512`` on
``tp > 1``), the residual stream stays sequence-parallel (``P("tp", None)``),
reducing row-parallel outputs in ``bfloat16`` via ``psum_scatter`` and
gathering 1-byte FP8 normalized activations via ``all_gather``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from resin.backend.tpu.arena import Caches, KvCacheSpec
from resin.backend.tpu.attention import (
    HEADS,
    KV_MERGED_PARTITION,
    Q_PACKED_PARTITION,
    AttentionImpl,
    paged_attention,
)
from resin.backend.tpu.kernels import (
    Fp8Weight,
    MatmulBackend,
    apply_rope,
    fused_add_rms_norm,
    fused_add_rms_norm_fp8_quant,
    pallas_qkv_norm_rope_pack,
    rms_norm,
    rms_norm_fp8_quant,
    sharded_fp8_gate_up_swiglu,
    sharded_fp8_matmul,
    sharded_fp8_matmul_all_gather,
    sharded_fp8_matmul_reduce_scatter,
)
from resin.backend.tpu.kernels._rpa_v3_pallas import get_tpu_version
from resin.backend.tpu.mesh import TP_AXIS, tp_size
from resin.backend.tpu.metadata import ModelInputs
from resin.models.qwen3.config import Qwen3Config

COLUMN = P(TP_AXIS, None)
ROW = P(None, TP_AXIS)
TOKENS_SHARDED = P(TP_AXIS, None)
_PREFILL_MATMUL_MIN_TOKENS = 512

Linear = jax.Array | Fp8Weight


class LayerParams(NamedTuple):
    input_norm: jax.Array
    q_norm: jax.Array
    k_norm: jax.Array
    post_norm: jax.Array
    qkv: Linear
    o: Linear
    gate_up: Linear
    down: Linear


class Qwen3Params(NamedTuple):
    embed: jax.Array
    layers: tuple[LayerParams, ...]
    norm: jax.Array
    lm_head: jax.Array
    rope_cache: tuple[jax.Array, jax.Array] | None = None


def check_tensor_parallel(config: Qwen3Config, tp: int) -> None:
    """Reject geometry that cannot split over ``tp`` dies."""

    for label, size in (
        ("KV heads", config.num_key_value_heads),
        ("attention heads", config.num_attention_heads),
        ("intermediate_size", config.intermediate_size),
        ("vocab_size", config.vocab_size),
    ):
        if size % tp:
            raise ValueError(f"{label} ({size}) do not divide over tp={tp}")


def _per_shard_swiglu(gate_up: jax.Array, mesh: Mesh) -> jax.Array:
    def _local(gu_local: jax.Array) -> jax.Array:
        gate, up = jnp.split(gu_local, 2, axis=-1)
        return (jax.nn.silu(gate) * up).astype(gu_local.dtype)

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=ROW,
        out_specs=ROW,
        check_vma=False,
    )(gate_up)


@dataclass(eq=False)
class Qwen3Model:
    """Implements :class:`resin.backend.tpu.runner.TpuModel` for dense Qwen3."""

    config: Qwen3Config
    mesh: Mesh
    kv_dtype: Any
    page_tokens: int = 256
    impl: AttentionImpl | None = None
    prefill_matmul_backend: MatmulBackend = "auto"
    sequence_parallel_min_tokens: int = _PREFILL_MATMUL_MIN_TOKENS
    _force_fused_qkv_pack_interpret: bool = field(default=False, repr=False)
    vocab_size: int = field(init=False)

    def __post_init__(self) -> None:
        check_tensor_parallel(self.config, tp_size(self.mesh))
        self.vocab_size = self.config.vocab_size

    @property
    def cache_specs(self) -> Sequence[KvCacheSpec]:
        config = self.config
        return (
            KvCacheSpec(
                config.num_hidden_layers,
                config.num_key_value_heads,
                config.head_dim,
                self.page_tokens,
                self.kv_dtype,
            ),
        )

    def _linear(
        self,
        x: jax.Array,
        weight: Linear,
        spec: P,
        *,
        x_scale: jax.Array | None = None,
        x_abs_max: jax.Array | None = None,
        out_dtype: Any = None,
        backend: MatmulBackend = "auto",
    ) -> jax.Array:
        target_dtype = x.dtype if out_dtype is None else out_dtype
        if isinstance(weight, Fp8Weight):
            return sharded_fp8_matmul(
                x,
                weight.qvalue,
                weight.scale,
                spec,
                mesh=self.mesh,
                x_scale=x_scale,
                x_abs_max=x_abs_max,
                out_dtype=target_dtype,
                backend=backend,
            )
        return x @ weight.T

    def _attend_qkv(
        self,
        qkv_out: jax.Array,
        weights: LayerParams,
        kv_cache: jax.Array,
        inputs: ModelInputs,
        rope_cache: tuple[jax.Array, jax.Array] | None,
    ) -> tuple[jax.Array, jax.Array]:
        config = self.config
        eps = config.rms_norm_eps
        tokens = qkv_out.shape[0]
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        tp = tp_size(self.mesh)
        local_heads = heads // tp
        local_kv_heads = kv_heads // tp

        use_fused_pack = (
            head_dim == 128
            and qkv_out.dtype == jnp.bfloat16
            and rope_cache is not None
            and (get_tpu_version() >= 6 or self._force_fused_qkv_pack_interpret)
        )
        if use_fused_pack:
            assert rope_cache is not None
            cos_full, sin_signed = rope_cache

            def _pack_local(
                qkv_local: jax.Array,
                pos: Any,
                cos_t: jax.Array,
                sin_t: jax.Array,
                qw: jax.Array,
                kw: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return pallas_qkv_norm_rope_pack(
                    qkv_local,
                    pos,
                    (cos_t, sin_t),
                    qw,
                    kw,
                    kv_cache.dtype,
                    num_q_heads=local_heads,
                    num_kv_heads=local_kv_heads,
                    eps=eps,
                    interpret=self._force_fused_qkv_pack_interpret,
                )

            q_packed, kv_merged = jax.shard_map(
                _pack_local,
                mesh=self.mesh,
                in_specs=(ROW, P(), P(), P(), P(), P()),
                out_specs=(Q_PACKED_PARTITION, KV_MERGED_PARTITION),
                check_vma=False,
            )(
                qkv_out,
                inputs.positions,
                cos_full,
                sin_signed,
                weights.q_norm,
                weights.k_norm,
            )
            return paged_attention(
                q_packed,
                kv_merged,
                None,
                kv_cache,
                inputs.attention,
                mesh=self.mesh,
                sm_scale=head_dim**-0.5,
                prepared=True,
                actual_num_q_heads=heads,
                actual_head_dim=head_dim,
                impl=self.impl,
            )

        local_q_dim = local_heads * head_dim
        local_kv_dim = local_kv_heads * head_dim

        if rope_cache is not None:
            cos_table, sin_table = rope_cache

            def _split_norm_rope_cached(
                qkv_local: jax.Array,
                pos: Any,
                cos_t: jax.Array,
                sin_t: jax.Array,
                qw: jax.Array,
                kw: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                q_l = qkv_local[:, :local_q_dim].reshape(tokens, local_heads, head_dim)
                k_l = qkv_local[:, local_q_dim : local_q_dim + local_kv_dim].reshape(
                    tokens, local_kv_heads, head_dim
                )
                v_l = qkv_local[:, local_q_dim + local_kv_dim :].reshape(
                    tokens, local_kv_heads, head_dim
                )
                q_l = apply_rope(
                    rms_norm(q_l, qw, eps),
                    pos,
                    head_dim,
                    rope_theta=config.rope_theta,
                    cos_sin_cache=(cos_t, sin_t),
                )
                k_l = apply_rope(
                    rms_norm(k_l, kw, eps),
                    pos,
                    head_dim,
                    rope_theta=config.rope_theta,
                    cos_sin_cache=(cos_t, sin_t),
                )
                return q_l, k_l, v_l

            q, k, v = jax.shard_map(
                _split_norm_rope_cached,
                mesh=self.mesh,
                in_specs=(ROW, P(), P(), P(), P(), P()),
                out_specs=(HEADS, HEADS, HEADS),
                check_vma=False,
            )(
                qkv_out,
                inputs.positions,
                cos_table,
                sin_table,
                weights.q_norm,
                weights.k_norm,
            )
        else:

            def _split_norm_rope_uncached(
                qkv_local: jax.Array,
                pos: Any,
                qw: jax.Array,
                kw: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                q_l = qkv_local[:, :local_q_dim].reshape(tokens, local_heads, head_dim)
                k_l = qkv_local[:, local_q_dim : local_q_dim + local_kv_dim].reshape(
                    tokens, local_kv_heads, head_dim
                )
                v_l = qkv_local[:, local_q_dim + local_kv_dim :].reshape(
                    tokens, local_kv_heads, head_dim
                )
                q_l = apply_rope(
                    rms_norm(q_l, qw, eps), pos, head_dim, rope_theta=config.rope_theta
                )
                k_l = apply_rope(
                    rms_norm(k_l, kw, eps), pos, head_dim, rope_theta=config.rope_theta
                )
                return q_l, k_l, v_l

            q, k, v = jax.shard_map(
                _split_norm_rope_uncached,
                mesh=self.mesh,
                in_specs=(ROW, P(), P(), P()),
                out_specs=(HEADS, HEADS, HEADS),
                check_vma=False,
            )(qkv_out, inputs.positions, weights.q_norm, weights.k_norm)

        return paged_attention(
            q,
            k,
            v,
            kv_cache,
            inputs.attention,
            mesh=self.mesh,
            sm_scale=head_dim**-0.5,
            impl=self.impl,
        )

    def _resolve_prefill_matmul_backend(self, tokens: int) -> MatmulBackend:
        if self.prefill_matmul_backend == "auto" and get_tpu_version() >= 7:
            return "xla" if tokens <= 1024 else "pallas"
        return self.prefill_matmul_backend

    def _hidden_states_fp8_sequence_parallel(
        self, params: Qwen3Params, caches: Caches, inputs: ModelInputs
    ) -> tuple[jax.Array, Caches]:
        config = self.config
        eps = config.rms_norm_eps
        tokens = inputs.input_ids.shape[0]
        q_width = config.num_attention_heads * config.head_dim
        layers = params.layers
        backend = self._resolve_prefill_matmul_backend(tokens)

        x = jax.shard_map(
            lambda ids, emb: emb[ids],
            mesh=self.mesh,
            in_specs=(P(TP_AXIS), P()),
            out_specs=TOKENS_SHARDED,
        )(inputs.input_ids, params.embed)
        act_dtype = x.dtype

        def _sharded_norm_quant(
            x_in: jax.Array, norm_w: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return jax.shard_map(
                lambda xl, nw: rms_norm_fp8_quant(xl, nw, eps),
                mesh=self.mesh,
                in_specs=(TOKENS_SHARDED, P()),
                out_specs=(TOKENS_SHARDED, TOKENS_SHARDED),
                check_vma=False,
            )(x_in, norm_w)

        def _sharded_add_norm_quant(
            delta: jax.Array, res: jax.Array, norm_w: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                lambda dl, rl, nw: fused_add_rms_norm_fp8_quant(dl, rl, nw, eps),
                mesh=self.mesh,
                in_specs=(TOKENS_SHARDED, TOKENS_SHARDED, P()),
                out_specs=(TOKENS_SHARDED, TOKENS_SHARDED, TOKENS_SHARDED),
                check_vma=False,
            )(delta, res, norm_w)

        h_q, h_scale = _sharded_norm_quant(x, layers[0].input_norm)
        layer_caches = []
        num_layers = len(layers)

        for idx, (weights, kv_cache) in enumerate(zip(layers, caches[0], strict=True)):
            assert isinstance(weights.qkv, Fp8Weight)
            assert isinstance(weights.o, Fp8Weight)
            assert isinstance(weights.gate_up, Fp8Weight)
            assert isinstance(weights.down, Fp8Weight)

            qkv_out = sharded_fp8_matmul_all_gather(
                h_q,
                h_scale,
                weights.qkv.qvalue,
                weights.qkv.scale,
                mesh=self.mesh,
                out_dtype=act_dtype,
                backend=backend,
            )
            attended, kv_cache = self._attend_qkv(
                qkv_out, weights, kv_cache, inputs, params.rope_cache
            )
            layer_caches.append(kv_cache)

            o_out = sharded_fp8_matmul_reduce_scatter(
                attended.reshape(tokens, q_width),
                weights.o.qvalue,
                weights.o.scale,
                mesh=self.mesh,
                out_dtype=act_dtype,
                backend=backend,
            )
            mlp_q, mlp_scale, x = _sharded_add_norm_quant(o_out, x, weights.post_norm)

            gated, gated_absmax = sharded_fp8_gate_up_swiglu(
                mlp_q,
                mlp_scale,
                weights.gate_up.qvalue,
                weights.gate_up.scale,
                mesh=self.mesh,
                sequence_parallel=True,
                out_dtype=act_dtype,
                backend=backend,
            )
            down_out = sharded_fp8_matmul_reduce_scatter(
                gated,
                weights.down.qvalue,
                weights.down.scale,
                mesh=self.mesh,
                x_abs_max=gated_absmax,
                out_dtype=act_dtype,
                backend=backend,
            )

            if idx + 1 < num_layers:
                h_q, h_scale, x = _sharded_add_norm_quant(
                    down_out, x, layers[idx + 1].input_norm
                )
            else:
                hidden = jax.shard_map(
                    lambda dl, rl, nw: jax.lax.all_gather(
                        fused_add_rms_norm(dl, rl, nw, eps)[0],
                        axis_name=TP_AXIS,
                        axis=0,
                        tiled=True,
                    ),
                    mesh=self.mesh,
                    in_specs=(TOKENS_SHARDED, TOKENS_SHARDED, P()),
                    out_specs=P(None, None),
                    check_vma=False,
                )(down_out, x, params.norm)

        return hidden, (tuple(layer_caches),)

    def _hidden_states_fp8_replicated(
        self, params: Qwen3Params, caches: Caches, inputs: ModelInputs
    ) -> tuple[jax.Array, Caches]:
        config = self.config
        eps = config.rms_norm_eps
        tokens = inputs.input_ids.shape[0]
        q_width = config.num_attention_heads * config.head_dim
        layers = params.layers
        is_prefill = tokens >= _PREFILL_MATMUL_MIN_TOKENS
        backend = self._resolve_prefill_matmul_backend(tokens) if is_prefill else "auto"
        rep_2d = P(None, None)

        def _rep_norm_quant(
            x_in: jax.Array, norm_w: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return jax.shard_map(
                lambda xl, nw: rms_norm_fp8_quant(xl, nw, eps),
                mesh=self.mesh,
                in_specs=(rep_2d, P()),
                out_specs=(rep_2d, rep_2d),
                check_vma=False,
            )(x_in, norm_w)

        def _rep_add_norm_quant(
            delta: jax.Array, res: jax.Array, norm_w: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                lambda dl, rl, nw: fused_add_rms_norm_fp8_quant(dl, rl, nw, eps),
                mesh=self.mesh,
                in_specs=(rep_2d, rep_2d, P()),
                out_specs=(rep_2d, rep_2d, rep_2d),
                check_vma=False,
            )(delta, res, norm_w)

        x = params.embed[inputs.input_ids]
        act_dtype = x.dtype
        h_q, h_scale = _rep_norm_quant(x, layers[0].input_norm)
        layer_caches = []
        num_layers = len(layers)

        for idx, (weights, kv_cache) in enumerate(zip(layers, caches[0], strict=True)):
            assert isinstance(weights.qkv, Fp8Weight)
            assert isinstance(weights.o, Fp8Weight)
            assert isinstance(weights.gate_up, Fp8Weight)
            assert isinstance(weights.down, Fp8Weight)

            qkv_out = self._linear(
                h_q,
                weights.qkv,
                COLUMN,
                x_scale=h_scale,
                out_dtype=act_dtype,
                backend=backend,
            )
            attended, kv_cache = self._attend_qkv(
                qkv_out, weights, kv_cache, inputs, params.rope_cache
            )
            layer_caches.append(kv_cache)

            o_out = self._linear(
                attended.reshape(tokens, q_width),
                weights.o,
                ROW,
                out_dtype=act_dtype,
                backend=backend,
            )
            mlp_q, mlp_scale, x = _rep_add_norm_quant(o_out, x, weights.post_norm)

            if is_prefill:
                gated, gated_absmax = sharded_fp8_gate_up_swiglu(
                    mlp_q,
                    mlp_scale,
                    weights.gate_up.qvalue,
                    weights.gate_up.scale,
                    mesh=self.mesh,
                    sequence_parallel=False,
                    out_dtype=act_dtype,
                    backend=backend,
                )
            else:
                gate_up_out = self._linear(
                    mlp_q,
                    weights.gate_up,
                    COLUMN,
                    x_scale=mlp_scale,
                    out_dtype=act_dtype,
                    backend="auto",
                )
                gated = _per_shard_swiglu(gate_up_out, self.mesh)
                gated_absmax = None

            down_out = self._linear(
                gated,
                weights.down,
                ROW,
                x_abs_max=gated_absmax,
                out_dtype=act_dtype,
                backend=backend,
            )
            if idx + 1 < num_layers:
                h_q, h_scale, x = _rep_add_norm_quant(
                    down_out, x, layers[idx + 1].input_norm
                )
            else:
                normed, _ = fused_add_rms_norm(down_out, x, params.norm, eps)

        return normed, (tuple(layer_caches),)

    def hidden_states(
        self, params: Qwen3Params, caches: Caches, inputs: ModelInputs
    ) -> tuple[jax.Array, Caches]:
        config = self.config
        eps = config.rms_norm_eps
        tokens = inputs.input_ids.shape[0]
        tp = tp_size(self.mesh)
        is_fp8 = bool(params.layers) and isinstance(params.layers[0].qkv, Fp8Weight)

        if is_fp8:
            if (
                tp > 1
                and tokens >= self.sequence_parallel_min_tokens
                and tokens % tp == 0
            ):
                return self._hidden_states_fp8_sequence_parallel(params, caches, inputs)
            return self._hidden_states_fp8_replicated(params, caches, inputs)

        q_width = config.num_attention_heads * config.head_dim
        x = params.embed[inputs.input_ids]
        layer_caches = []
        for weights, kv_cache in zip(params.layers, caches[0], strict=True):
            h = rms_norm(x, weights.input_norm, eps)
            qkv_out = self._linear(h, weights.qkv, COLUMN)
            attended, kv_cache = self._attend_qkv(
                qkv_out, weights, kv_cache, inputs, params.rope_cache
            )
            layer_caches.append(kv_cache)
            x = x + self._linear(attended.reshape(tokens, q_width), weights.o, ROW)
            h = rms_norm(x, weights.post_norm, eps)
            gate_up_out = self._linear(h, weights.gate_up, COLUMN)
            gated = _per_shard_swiglu(gate_up_out, self.mesh)
            x = x + self._linear(gated, weights.down, ROW)
        return rms_norm(x, params.norm, eps), (tuple(layer_caches),)

    def logits(self, params: Qwen3Params, hidden: jax.Array) -> jax.Array:
        return hidden @ params.lm_head.T


__all__ = [
    "COLUMN",
    "ROW",
    "TOKENS_SHARDED",
    "LayerParams",
    "Linear",
    "Qwen3Model",
    "Qwen3Params",
    "check_tensor_parallel",
]
