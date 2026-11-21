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
    fused_add_rms_norm,
    fused_add_rms_norm_fp8_quant,
    pallas_qkv_norm_rope_pack,
    pallas_unpack_attn_fp8_quant,
    rms_norm,
    rms_norm_fp8_quant,
    sharded_fp8_gate_up_swiglu,
    sharded_fp8_matmul,
    sharded_fp8_matmul_all_gather,
    sharded_fp8_matmul_reduce_scatter,
    sharded_fp8_qkv_norm_rope_pack,
    sharded_fp8_row_matmul_async_ar,
)
from resin.backend.tpu.kernels._rpa_v3_pallas import get_tpu_version
from resin.backend.tpu.kernels.norm_rope import _half_swap_perm, _slice_rope_cache_128
from resin.backend.tpu.mesh import TP_AXIS, tp_size
from resin.backend.tpu.metadata import ModelInputs
from resin.models.qwen3.config import Qwen3Config

COLUMN = P(TP_AXIS, None)
ROW = P(None, TP_AXIS)
TOKENS_SHARDED = P(TP_AXIS, None)
_PREFILL_MATMUL_MIN_TOKENS = 512
_DECODE_PALLAS_MAX_TOKENS = 320
_ASYNC_AR_MIN_HIDDEN_SIZE = 256

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
        cos_sin_pos: tuple[jax.Array, jax.Array] | None = None,
        perm_128: jax.Array | None = None,
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
            cos_pos, sin_pos = (
                _slice_rope_cache_128(inputs.positions, rope_cache)
                if cos_sin_pos is None
                else cos_sin_pos
            )
            eff_perm = (
                _half_swap_perm(128, dtype=jnp.bfloat16)
                if perm_128 is None
                else perm_128
            )

            def _pack_local(
                qkv_local: jax.Array,
                cos_p: jax.Array,
                sin_p: jax.Array,
                qw: jax.Array,
                kw: jax.Array,
                perm_l: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return pallas_qkv_norm_rope_pack(
                    qkv_local,
                    None,
                    None,
                    qw,
                    kw,
                    kv_cache.dtype,
                    num_q_heads=local_heads,
                    num_kv_heads=local_kv_heads,
                    eps=eps,
                    cos_sin_pos=(cos_p, sin_p),
                    perm_128=perm_l,
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
                cos_pos,
                sin_pos,
                weights.q_norm,
                weights.k_norm,
                eff_perm,
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

        x = jax.shard_map(
            lambda ids, emb: emb[ids],
            mesh=self.mesh,
            in_specs=(P(TP_AXIS), P()),
            out_specs=TOKENS_SHARDED,
        )(inputs.input_ids, params.embed)
        act_dtype = x.dtype
        use_fused_unpack = (
            self.fuse_unpack_attn_quant
            and (get_tpu_version() >= 7 or pallas_interp)
            and self._can_use_fused_qkv_pack(act_dtype, params.rope_cache)
            and (q_width // tp) % 128 == 0
        )

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

        def _row_proj_add_norm_quant(
            row_in: jax.Array,
            row_w: Fp8Weight,
            res: jax.Array,
            norm_w: jax.Array,
            q_out_dtype: Any,
            *,
            x_scale: jax.Array | None = None,
            x_abs_max: jax.Array | None = None,
            sync_scale: bool,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            if use_async_row_ar:
                return sharded_fp8_row_matmul_async_ar(
                    row_in,
                    row_w.qvalue,
                    row_w.scale,
                    mesh=self.mesh,
                    x_scale=x_scale,
                    x_abs_max=x_abs_max,
                    sync_tp_scale=sync_scale,
                    out_dtype=act_dtype,
                    residual=res,
                    norm_weight=norm_w,
                    eps=eps,
                    quantize_output_dtype=q_out_dtype,
                    transposed_scale=True,
                    backend=proj_backend,
                    interpret=dec_interp,
                )
            row_out = self._linear(
                row_in,
                row_w,
                ROW,
                x_scale=x_scale,
                x_abs_max=x_abs_max,
                sync_tp_scale=sync_scale,
                async_all_reduce=False,
                out_dtype=act_dtype,
                backend=proj_backend,
                interpret=dec_interp,
            )
            return _rep_add_norm_quant(row_out, res, norm_w)

        if params.rope_cache is not None and self._can_use_fused_qkv_pack(
            act_dtype, params.rope_cache
        ):
            hoisted_cos_sin: tuple[jax.Array, jax.Array] | None = _slice_rope_cache_128(
                inputs.positions, params.rope_cache
            )
            hoisted_perm: jax.Array | None = _half_swap_perm(128, dtype=jnp.bfloat16)
        else:
            hoisted_cos_sin = None
            hoisted_perm = None

        h_q, h_scale = _rep_norm_quant(x, layers[0].input_norm)
        layer_caches = []
        num_layers = len(layers)

        for idx, (weights, kv_cache) in enumerate(zip(layers, caches[0], strict=True)):
            assert isinstance(weights.qkv, Fp8Weight)
            assert isinstance(weights.o, Fp8Weight)
            assert isinstance(weights.gate_up, Fp8Weight)
            assert isinstance(weights.down, Fp8Weight)

            if use_fused_unpack_quant:
                q_packed, kv_merged = sharded_fp8_qkv_norm_rope_pack(
                    h_q,
                    h_scale,
                    weights.qkv.qvalue,
                    weights.qkv.scale,
                    inputs.positions,
                    params.rope_cache,
                    weights.q_norm,
                    weights.k_norm,
                    kv_cache.dtype,
                    mesh=self.mesh,
                    num_q_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    eps=eps,
                    cos_sin_pos=hoisted_cos_sin,
                    perm_128=hoisted_perm,
                    backend=proj_backend,
                    interpret=dec_interp,
                    pack_interpret=self._force_fused_qkv_pack_interpret,
                )
                att_packed, kv_cache = paged_attention(
                    q_packed,
                    kv_merged,
                    None,
                    kv_cache,
                    inputs.attention,
                    mesh=self.mesh,
                    sm_scale=config.head_dim**-0.5,
                    prepared=True,
                    return_packed=True,
                    actual_num_q_heads=config.num_attention_heads,
                    actual_head_dim=config.head_dim,
                    impl=self.impl,
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
                mlp_q, mlp_scale, x = _row_proj_add_norm_quant(
                    att_q,
                    weights.o,
                    x,
                    weights.post_norm,
                    weights.gate_up.qvalue.dtype,
                    x_scale=att_scale,
                    sync_scale=False,
                )
            else:
                qkv_out = self._linear(
                    h_q,
                    weights.qkv,
                    COLUMN,
                    x_scale=h_scale,
                    out_dtype=act_dtype,
                    backend=proj_backend,
                    interpret=dec_interp,
                )
                attended, kv_cache = self._attend_qkv(
                    qkv_out,
                    weights,
                    kv_cache,
                    inputs,
                    params.rope_cache,
                    cos_sin_pos=hoisted_cos_sin,
                    perm_128=hoisted_perm,
                )
                layer_caches.append(kv_cache)
                mlp_q, mlp_scale, x = _row_proj_add_norm_quant(
                    attended.reshape(tokens, q_width),
                    weights.o,
                    x,
                    weights.post_norm,
                    weights.gate_up.qvalue.dtype,
                    sync_scale=row_sync_tp_scale,
                )

            gated_in, gated_aux = sharded_fp8_gate_up_swiglu(
                mlp_q,
                mlp_scale,
                weights.gate_up.qvalue,
                weights.gate_up.scale,
                mesh=self.mesh,
                sequence_parallel=False,
                sync_tp_scale=row_sync_tp_scale,
                quantize_output=prequantize_gated,
                fuse_quant_epilogue=proj_backend != "xla",
                out_dtype=act_dtype,
                backend=gu_backend,
                interpret=dec_interp,
            )
            if idx + 1 < num_layers:
                next_layer = layers[idx + 1]
                assert isinstance(next_layer.qkv, Fp8Weight)
                h_q, h_scale, x = _row_proj_add_norm_quant(
                    gated_in,
                    weights.down,
                    x,
                    next_layer.input_norm,
                    next_layer.qkv.qvalue.dtype,
                    x_scale=gated_aux if prequantize_gated else None,
                    x_abs_max=None if prequantize_gated else gated_aux,
                    sync_scale=row_sync_tp_scale,
                )
            else:
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
