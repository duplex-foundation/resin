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

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

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
    can_use_sp_async_rs,
    fp8_matmul,
    fused_add_rms_norm,
    fused_add_rms_norm_fp8_quant,
    local_fp8_row_matmul_async_rs,
    pallas_fp8_gate_up_swiglu,
    pallas_qkv_norm_rope_pack,
    pallas_unpack_attn_fp8_quant,
    rms_norm,
    rms_norm_fp8_quant,
    sharded_fp8_gate_up_swiglu,
    sharded_fp8_matmul,
    sharded_fp8_matmul_all_gather,
    sharded_fp8_matmul_reduce_scatter,
    sharded_fp8_row_matmul_async_ar,
    xla_fp8_gate_up_swiglu,
)
from resin.backend.tpu.kernels._rpa_v3_pallas import get_tpu_version
from resin.backend.tpu.mesh import TP_AXIS, tp_size
from resin.backend.tpu.metadata import ModelInputs
from resin.models.qwen3.config import Qwen3Config

COLUMN = P(TP_AXIS, None)
ROW = P(None, TP_AXIS)
TOKENS_SHARDED = P(TP_AXIS, None)
_PREFILL_MATMUL_MIN_TOKENS = 512
_DECODE_PALLAS_MAX_TOKENS = 320
_ASYNC_AR_MIN_HIDDEN_SIZE = 256


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


DecodeMatmulMode = Literal[
    "auto",
    "pallas",
    "xla",
]

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
    decode_matmul_mode: DecodeMatmulMode = "auto"
    decode_async_all_reduce: bool = True
    fuse_unpack_attn_quant: bool = True
    prefill_sp_async_rs: bool = field(
        default_factory=lambda: _env_bool("RESIN_TPU_PREFILL_SP_ASYNC_RS", True)
    )
    sequence_parallel_min_tokens: int = _PREFILL_MATMUL_MIN_TOKENS
    _force_fused_qkv_pack_interpret: bool = field(default=False, repr=False)
    _force_decode_pallas_interpret: bool = field(default=False, repr=False)
    sync_tp_scale: bool = False
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

    def _can_use_fused_qkv_pack(
        self,
        act_dtype: Any,
        rope_cache: tuple[jax.Array, jax.Array] | None,
    ) -> bool:
        return (
            self.config.head_dim == 128
            and jnp.dtype(act_dtype) == jnp.bfloat16
            and rope_cache is not None
            and (get_tpu_version() >= 6 or self._force_fused_qkv_pack_interpret)
        )

    def _linear(
        self,
        x: jax.Array,
        weight: Linear,
        spec: P,
        *,
        x_scale: jax.Array | None = None,
        x_abs_max: jax.Array | None = None,
        sync_tp_scale: bool = True,
        async_all_reduce: bool = False,
        out_dtype: Any = None,
        backend: MatmulBackend = "auto",
        interpret: bool = False,
    ) -> jax.Array:
        if not isinstance(weight, Fp8Weight):
            return x @ weight.T
        target_dtype = x.dtype if out_dtype is None else out_dtype
        if async_all_reduce and spec == ROW:
            return sharded_fp8_row_matmul_async_ar(
                x,
                weight.qvalue,
                weight.scale,
                mesh=self.mesh,
                x_scale=x_scale,
                x_abs_max=x_abs_max,
                sync_tp_scale=sync_tp_scale,
                out_dtype=target_dtype,
                backend=backend,
                interpret=interpret,
            )
        return sharded_fp8_matmul(
            x,
            weight.qvalue,
            weight.scale,
            spec,
            mesh=self.mesh,
            x_scale=x_scale,
            x_abs_max=x_abs_max,
            sync_tp_scale=sync_tp_scale,
            out_dtype=target_dtype,
            backend=backend,
            interpret=interpret,
        )

    def _attend_qkv(
        self,
        qkv_out: jax.Array,
        weights: LayerParams,
        kv_cache: jax.Array,
        inputs: ModelInputs,
        rope_cache: tuple[jax.Array, jax.Array] | None,
        *,
        return_packed: bool = False,
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

        use_fused_pack = self._can_use_fused_qkv_pack(qkv_out.dtype, rope_cache)
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
                return_packed=return_packed,
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
        group_size = config.num_attention_heads // config.num_key_value_heads
        tp = tp_size(self.mesh)
        layers = params.layers
        backend = self._resolve_prefill_matmul_backend(tokens)
        pallas_interp = self._force_fused_qkv_pack_interpret
        # Require 128-lane local contracting widths and a shard_map-backed matmul backend.
        per_die_scale = (
            not self.sync_tp_scale
            and backend != "gspmd_xla"
            and (q_width // tp) >= 128
            and (config.intermediate_size // tp) >= 128
        )
        sp_sync_tp_scale = not per_die_scale

        act_dtype = jnp.dtype(params.embed.dtype)
        use_fused_unpack = (
            self.fuse_unpack_attn_quant
            and (get_tpu_version() >= 7 or pallas_interp)
            and self._can_use_fused_qkv_pack(act_dtype, params.rope_cache)
            and (q_width // tp) % 128 == 0
        )
        use_sp_async_rs = (
            self.prefill_sp_async_rs
            and not sp_sync_tp_scale
            and use_fused_unpack
            and backend != "xla"
            and can_use_sp_async_rs(tokens, tp=tp, interpret=pallas_interp)
        )

        if use_sp_async_rs:
            num_layers = len(layers)
            first_qkv = layers[0].qkv
            assert isinstance(first_qkv, Fp8Weight)

            def _ag_col_matmul(
                xq_l: jax.Array,
                xs_l: jax.Array,
                w_q_l: jax.Array,
                w_s_l: jax.Array,
            ) -> jax.Array:
                xq_full = jax.lax.all_gather(
                    xq_l, axis_name=TP_AXIS, axis=0, tiled=True
                )
                xs_full = jax.lax.all_gather(
                    xs_l, axis_name=TP_AXIS, axis=0, tiled=True
                )
                return fp8_matmul(
                    xq_full,
                    w_q_l,
                    w_s_l,
                    x_scale=xs_full,
                    out_dtype=act_dtype,
                    backend=backend,
                    interpret=pallas_interp,
                )

            def _prologue_local(
                ids_l: Any,
                emb: jax.Array,
                norm_w: jax.Array,
                qkv_q_l: jax.Array,
                qkv_s_l: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                x_l = emb[ids_l]
                hq_l, hs_l = rms_norm_fp8_quant(
                    x_l, norm_w, eps, interpret=pallas_interp
                )
                return x_l, _ag_col_matmul(hq_l, hs_l, qkv_q_l, qkv_s_l)

            def _post_attn_mlp_local(
                packed_l: jax.Array,
                x_l: jax.Array,
                o_q_l: jax.Array,
                o_s_l: jax.Array,
                post_norm_w: jax.Array,
                gu_q_l: jax.Array,
                gu_s_l: jax.Array,
                down_q_l: jax.Array,
                down_s_l: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                att_q_l, att_s_l = pallas_unpack_attn_fp8_quant(
                    packed_l,
                    group_size,
                    config.head_dim,
                    dtype=o_q_l.dtype,
                    interpret=pallas_interp,
                )
                o_pair_l, o_peer_l = local_fp8_row_matmul_async_rs(
                    att_q_l,
                    o_q_l,
                    o_s_l,
                    x_scale_local=att_s_l,
                    tp_axis=TP_AXIS,
                    out_dtype=act_dtype,
                    collective_id=0,
                    return_partials_pair=True,
                    interpret=pallas_interp,
                )
                mlp_q_l, mlp_s_l, x_mid_l = fused_add_rms_norm_fp8_quant(
                    o_pair_l,
                    x_l,
                    post_norm_w,
                    eps,
                    extra_x=o_peer_l,
                    x_pair_index=1,
                    interpret=pallas_interp,
                )
                mlp_q_full = jax.lax.all_gather(
                    mlp_q_l, axis_name=TP_AXIS, axis=0, tiled=True
                )
                mlp_s_full = jax.lax.all_gather(
                    mlp_s_l, axis_name=TP_AXIS, axis=0, tiled=True
                )
                use_pallas_gu = backend == "pallas" or (
                    backend == "auto"
                    and (pallas_interp or get_tpu_version() >= 7)
                    and gu_q_l.shape[1] % 128 == 0
                    and (gu_q_l.shape[0] // 2) % 128 == 0
                )
                if use_pallas_gu:
                    gated_l, local_abs_max = pallas_fp8_gate_up_swiglu(
                        mlp_q_full,
                        mlp_s_full,
                        gu_q_l,
                        gu_s_l,
                        out_dtype=act_dtype,
                        interpret=pallas_interp,
                    )
                    gated_absmax_l: jax.Array | None = jnp.max(
                        local_abs_max, axis=0, keepdims=True
                    )
                else:
                    gated_l = xla_fp8_gate_up_swiglu(
                        mlp_q_full,
                        mlp_s_full,
                        gu_q_l,
                        gu_s_l,
                        out_dtype=act_dtype,
                    )
                    gated_absmax_l = None
                down_pair_l, down_peer_l = local_fp8_row_matmul_async_rs(
                    gated_l,
                    down_q_l,
                    down_s_l,
                    x_abs_max_local=gated_absmax_l,
                    tp_axis=TP_AXIS,
                    out_dtype=act_dtype,
                    collective_id=1,
                    return_partials_pair=True,
                    interpret=pallas_interp,
                )
                return down_pair_l, down_peer_l, x_mid_l

            def _mid_layer_local(
                packed_l: jax.Array,
                x_l: jax.Array,
                o_q_l: jax.Array,
                o_s_l: jax.Array,
                post_norm_w: jax.Array,
                gu_q_l: jax.Array,
                gu_s_l: jax.Array,
                down_q_l: jax.Array,
                down_s_l: jax.Array,
                next_norm_w: jax.Array,
                next_qkv_q_l: jax.Array,
                next_qkv_s_l: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                down_pair_l, down_peer_l, x_mid_l = _post_attn_mlp_local(
                    packed_l,
                    x_l,
                    o_q_l,
                    o_s_l,
                    post_norm_w,
                    gu_q_l,
                    gu_s_l,
                    down_q_l,
                    down_s_l,
                )
                hq_l, hs_l, x_next_l = fused_add_rms_norm_fp8_quant(
                    down_pair_l,
                    x_mid_l,
                    next_norm_w,
                    eps,
                    extra_x=down_peer_l,
                    x_pair_index=1,
                    interpret=pallas_interp,
                )
                return x_next_l, _ag_col_matmul(hq_l, hs_l, next_qkv_q_l, next_qkv_s_l)

            def _last_layer_local(
                packed_l: jax.Array,
                x_l: jax.Array,
                o_q_l: jax.Array,
                o_s_l: jax.Array,
                post_norm_w: jax.Array,
                gu_q_l: jax.Array,
                gu_s_l: jax.Array,
                down_q_l: jax.Array,
                down_s_l: jax.Array,
                final_norm_w: jax.Array,
            ) -> jax.Array:
                down_pair_l, down_peer_l, x_mid_l = _post_attn_mlp_local(
                    packed_l,
                    x_l,
                    o_q_l,
                    o_s_l,
                    post_norm_w,
                    gu_q_l,
                    gu_s_l,
                    down_q_l,
                    down_s_l,
                )
                down_sum_l = (
                    down_pair_l[1].astype(jnp.float32) + down_peer_l.astype(jnp.float32)
                ).astype(act_dtype)
                normed_l = fused_add_rms_norm(down_sum_l, x_mid_l, final_norm_w, eps)[0]
                return jax.lax.all_gather(
                    normed_l, axis_name=TP_AXIS, axis=0, tiled=True
                )

            x, qkv_out = jax.shard_map(
                _prologue_local,
                mesh=self.mesh,
                in_specs=(P(TP_AXIS), P(), P(), COLUMN, P(TP_AXIS)),
                out_specs=(TOKENS_SHARDED, ROW),
                check_vma=False,
            )(
                inputs.input_ids,
                params.embed,
                layers[0].input_norm,
                first_qkv.qvalue,
                first_qkv.scale,
            )
            layer_caches = []
            for idx, (weights, kv_cache) in enumerate(
                zip(layers, caches[0], strict=True)
            ):
                assert isinstance(weights.o, Fp8Weight)
                assert isinstance(weights.gate_up, Fp8Weight)
                assert isinstance(weights.down, Fp8Weight)

                att_packed, kv_cache = self._attend_qkv(
                    qkv_out,
                    weights,
                    kv_cache,
                    inputs,
                    params.rope_cache,
                    return_packed=True,
                )
                layer_caches.append(kv_cache)

                if idx + 1 < num_layers:
                    next_qkv = layers[idx + 1].qkv
                    assert isinstance(next_qkv, Fp8Weight)
                    x, qkv_out = jax.shard_map(
                        _mid_layer_local,
                        mesh=self.mesh,
                        in_specs=(
                            Q_PACKED_PARTITION,
                            TOKENS_SHARDED,
                            ROW,
                            P(),
                            P(),
                            COLUMN,
                            P(TP_AXIS),
                            ROW,
                            P(),
                            P(),
                            COLUMN,
                            P(TP_AXIS),
                        ),
                        out_specs=(TOKENS_SHARDED, ROW),
                        check_vma=False,
                    )(
                        att_packed,
                        x,
                        weights.o.qvalue,
                        weights.o.scale,
                        weights.post_norm,
                        weights.gate_up.qvalue,
                        weights.gate_up.scale,
                        weights.down.qvalue,
                        weights.down.scale,
                        layers[idx + 1].input_norm,
                        next_qkv.qvalue,
                        next_qkv.scale,
                    )
                else:
                    hidden = jax.shard_map(
                        _last_layer_local,
                        mesh=self.mesh,
                        in_specs=(
                            Q_PACKED_PARTITION,
                            TOKENS_SHARDED,
                            ROW,
                            P(),
                            P(),
                            COLUMN,
                            P(TP_AXIS),
                            ROW,
                            P(),
                            P(),
                        ),
                        out_specs=P(None, None),
                        check_vma=False,
                    )(
                        att_packed,
                        x,
                        weights.o.qvalue,
                        weights.o.scale,
                        weights.post_norm,
                        weights.gate_up.qvalue,
                        weights.gate_up.scale,
                        weights.down.qvalue,
                        weights.down.scale,
                        params.norm,
                    )

            return hidden, (tuple(layer_caches),)

        x = jax.shard_map(
            lambda ids, emb: emb[ids],
            mesh=self.mesh,
            in_specs=(P(TP_AXIS), P()),
            out_specs=TOKENS_SHARDED,
        )(inputs.input_ids, params.embed)

        def _sharded_norm_quant(
            x_in: jax.Array, norm_w: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return jax.shard_map(
                lambda xl, nw: rms_norm_fp8_quant(xl, nw, eps, interpret=pallas_interp),
                mesh=self.mesh,
                in_specs=(TOKENS_SHARDED, P()),
                out_specs=(TOKENS_SHARDED, TOKENS_SHARDED),
                check_vma=False,
            )(x_in, norm_w)

        def _sharded_add_norm_quant(
            delta: jax.Array, res: jax.Array, norm_w: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                lambda dl, rl, nw: fused_add_rms_norm_fp8_quant(
                    dl, rl, nw, eps, interpret=pallas_interp
                ),
                mesh=self.mesh,
                in_specs=(TOKENS_SHARDED, TOKENS_SHARDED, P()),
                out_specs=(TOKENS_SHARDED, TOKENS_SHARDED, TOKENS_SHARDED),
                check_vma=False,
            )(delta, res, norm_w)

        def _unpack_synced(packed_l: jax.Array) -> tuple[jax.Array, jax.Array]:
            bf16_l, absmax_l = pallas_unpack_attn_fp8_quant(
                packed_l,
                group_size,
                config.head_dim,
                return_bf16_and_absmax=True,
                interpret=pallas_interp,
            )
            return bf16_l, jax.lax.pmax(absmax_l, axis_name=TP_AXIS)

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
            if use_fused_unpack:
                att_packed, kv_cache = self._attend_qkv(
                    qkv_out,
                    weights,
                    kv_cache,
                    inputs,
                    params.rope_cache,
                    return_packed=True,
                )
                layer_caches.append(kv_cache)
                if sp_sync_tp_scale:
                    att_in, att_absmax = jax.shard_map(
                        _unpack_synced,
                        mesh=self.mesh,
                        in_specs=Q_PACKED_PARTITION,
                        out_specs=(ROW, P()),
                        check_vma=False,
                    )(att_packed)
                    att_scale = None
                else:
                    o_q_dtype = weights.o.qvalue.dtype
                    att_in, att_scale = jax.shard_map(
                        lambda packed_l, _dt=o_q_dtype: pallas_unpack_attn_fp8_quant(
                            packed_l,
                            group_size,
                            config.head_dim,
                            dtype=_dt,
                            interpret=pallas_interp,
                        ),
                        mesh=self.mesh,
                        in_specs=Q_PACKED_PARTITION,
                        out_specs=(ROW, P(TP_AXIS, None)),
                        check_vma=False,
                    )(att_packed)
                    att_absmax = None
                o_out = sharded_fp8_matmul_reduce_scatter(
                    att_in,
                    weights.o.qvalue,
                    weights.o.scale,
                    mesh=self.mesh,
                    x_scale=att_scale,
                    x_abs_max=att_absmax,
                    sync_tp_scale=sp_sync_tp_scale,
                    out_dtype=act_dtype,
                    backend=backend,
                )
            else:
                attended, kv_cache = self._attend_qkv(
                    qkv_out, weights, kv_cache, inputs, params.rope_cache
                )
                layer_caches.append(kv_cache)
                o_out = sharded_fp8_matmul_reduce_scatter(
                    attended.reshape(tokens, q_width),
                    weights.o.qvalue,
                    weights.o.scale,
                    mesh=self.mesh,
                    sync_tp_scale=sp_sync_tp_scale,
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
                sync_tp_scale=sp_sync_tp_scale,
                out_dtype=act_dtype,
                backend=backend,
            )
            down_out = sharded_fp8_matmul_reduce_scatter(
                gated,
                weights.down.qvalue,
                weights.down.scale,
                mesh=self.mesh,
                x_abs_max=gated_absmax,
                sync_tp_scale=sp_sync_tp_scale,
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
        group_size = config.num_attention_heads // config.num_key_value_heads
        tp = tp_size(self.mesh)
        layers = params.layers
        is_prefill = tokens >= _PREFILL_MATMUL_MIN_TOKENS
        prequantize_gated = tokens <= _DECODE_PALLAS_MAX_TOKENS
        dec_interp = (not is_prefill) and self._force_decode_pallas_interpret
        if is_prefill:
            proj_backend: MatmulBackend = self._resolve_prefill_matmul_backend(tokens)
            gu_backend: MatmulBackend = proj_backend
            row_sync_tp_scale = True
        else:
            # Pallas FP8 BlockSpecs require the local contracting dimension to be >= 128.
            per_die_scale = (q_width // tp) >= 128 and (
                config.intermediate_size // tp
            ) >= 128
            row_sync_tp_scale = not per_die_scale
            if self.decode_matmul_mode == "xla":
                proj_backend = "xla"
                gu_backend = "auto"
            else:
                proj_backend = (
                    "pallas"
                    if (dec_interp or self.decode_matmul_mode == "pallas")
                    else "auto"
                )
                gu_backend = proj_backend

        rep_2d = P(None, None)

        def _rep_norm_quant(
            x_in: jax.Array, norm_w: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return jax.shard_map(
                lambda xl, nw: rms_norm_fp8_quant(
                    xl, nw, eps, transposed_scale=True, interpret=dec_interp
                ),
                mesh=self.mesh,
                in_specs=(rep_2d, P()),
                out_specs=(rep_2d, rep_2d),
                check_vma=False,
            )(x_in, norm_w)

        def _rep_add_norm_quant(
            delta: jax.Array, res: jax.Array, norm_w: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jax.shard_map(
                lambda dl, rl, nw: fused_add_rms_norm_fp8_quant(
                    dl, rl, nw, eps, transposed_scale=True, interpret=dec_interp
                ),
                mesh=self.mesh,
                in_specs=(rep_2d, rep_2d, P()),
                out_specs=(rep_2d, rep_2d, rep_2d),
                check_vma=False,
            )(delta, res, norm_w)

        x = params.embed[inputs.input_ids]
        act_dtype = x.dtype
        if self.sync_tp_scale:
            row_sync_tp_scale = True
        use_async_row_ar = (
            self.decode_async_all_reduce
            and not is_prefill
            and tokens <= _DECODE_PALLAS_MAX_TOKENS
            and tp == 2
            and not row_sync_tp_scale
            and config.hidden_size >= _ASYNC_AR_MIN_HIDDEN_SIZE
            and proj_backend != "xla"
        )
        use_fused_unpack_quant = (
            self.fuse_unpack_attn_quant
            and not is_prefill
            and not row_sync_tp_scale
            and (get_tpu_version() >= 7 or self._force_fused_qkv_pack_interpret)
            and self._can_use_fused_qkv_pack(act_dtype, params.rope_cache)
            and (q_width // tp) % 128 == 0
        )

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
                backend=proj_backend,
                interpret=dec_interp,
            )
            if use_fused_unpack_quant:
                att_packed, kv_cache = self._attend_qkv(
                    qkv_out,
                    weights,
                    kv_cache,
                    inputs,
                    params.rope_cache,
                    return_packed=True,
                )
                layer_caches.append(kv_cache)
                o_q_dtype = weights.o.qvalue.dtype
                att_q, att_scale = jax.shard_map(
                    lambda packed_l, _dt=o_q_dtype: pallas_unpack_attn_fp8_quant(
                        packed_l,
                        group_size,
                        config.head_dim,
                        dtype=_dt,
                        interpret=self._force_fused_qkv_pack_interpret,
                    ),
                    mesh=self.mesh,
                    in_specs=Q_PACKED_PARTITION,
                    out_specs=(ROW, P(TP_AXIS, None)),
                    check_vma=False,
                )(att_packed)
                o_out = self._linear(
                    att_q,
                    weights.o,
                    ROW,
                    x_scale=att_scale,
                    sync_tp_scale=False,
                    async_all_reduce=use_async_row_ar,
                    out_dtype=act_dtype,
                    backend=proj_backend,
                    interpret=dec_interp,
                )
            else:
                attended, kv_cache = self._attend_qkv(
                    qkv_out, weights, kv_cache, inputs, params.rope_cache
                )
                layer_caches.append(kv_cache)
                o_out = self._linear(
                    attended.reshape(tokens, q_width),
                    weights.o,
                    ROW,
                    sync_tp_scale=row_sync_tp_scale,
                    async_all_reduce=use_async_row_ar,
                    out_dtype=act_dtype,
                    backend=proj_backend,
                    interpret=dec_interp,
                )

            mlp_q, mlp_scale, x = _rep_add_norm_quant(o_out, x, weights.post_norm)

            gated_in, gated_aux = sharded_fp8_gate_up_swiglu(
                mlp_q,
                mlp_scale,
                weights.gate_up.qvalue,
                weights.gate_up.scale,
                mesh=self.mesh,
                sequence_parallel=False,
                sync_tp_scale=row_sync_tp_scale,
                quantize_output=prequantize_gated,
                out_dtype=act_dtype,
                backend=gu_backend,
                interpret=dec_interp,
            )
            down_out = self._linear(
                gated_in,
                weights.down,
                ROW,
                x_scale=gated_aux if prequantize_gated else None,
                x_abs_max=None if prequantize_gated else gated_aux,
                sync_tp_scale=row_sync_tp_scale,
                async_all_reduce=use_async_row_ar,
                out_dtype=act_dtype,
                backend=proj_backend,
                interpret=dec_interp,
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
    "DecodeMatmulMode",
    "LayerParams",
    "Linear",
    "Qwen3Model",
    "Qwen3Params",
    "check_tensor_parallel",
]
