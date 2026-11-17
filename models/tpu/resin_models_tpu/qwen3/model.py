"""Qwen3 layer graph for the TPU backend.

A dense decoder with grouped-query attention, per-head RMSNorm on Q and K
before RoPE, and a SwiGLU MLP. Attention and sampling come from
``resin.backend.tpu``; this module only wires layers, so it owns no request or
cache state. Linear weights are ``[out, in]`` as in the checkpoint, either
dense or FP8 with per-output-channel scales.

Tensor parallelism splits heads and the MLP width over ``tp``. Q, K, V, gate
and up are column-parallel (``P("tp", None)``); O and down are row-parallel
(``P(None, "tp")``) and reduce across dies inside their matmul.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from resin.backend.tpu.arena import Caches, KvCacheSpec
from resin.backend.tpu.attention import AttentionImpl, paged_attention
from resin.backend.tpu.kernels import (
    Fp8Weight,
    apply_rope,
    rms_norm,
    sharded_fp8_matmul,
)
from resin.backend.tpu.mesh import TP_AXIS, tp_size
from resin.backend.tpu.metadata import ModelInputs
from resin.models.qwen3.config import Qwen3Config

COLUMN = P(TP_AXIS, None)
ROW = P(None, TP_AXIS)

Linear = jax.Array | Fp8Weight


class LayerParams(NamedTuple):
    input_norm: jax.Array
    q_norm: jax.Array
    k_norm: jax.Array
    post_norm: jax.Array
    q: Linear
    k: Linear
    v: Linear
    o: Linear
    gate: Linear
    up: Linear
    down: Linear


class Qwen3Params(NamedTuple):
    embed: jax.Array
    layers: tuple[LayerParams, ...]
    norm: jax.Array
    lm_head: jax.Array


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


@dataclass(eq=False)
class Qwen3Model:
    """Implements :class:`resin.backend.tpu.runner.TpuModel` for dense Qwen3."""

    config: Qwen3Config
    mesh: Mesh
    kv_dtype: Any
    page_tokens: int = 256
    impl: AttentionImpl | None = None
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

    def _linear(self, x: jax.Array, weight: Linear, spec: P) -> jax.Array:
        if isinstance(weight, Fp8Weight):
            return sharded_fp8_matmul(
                x, weight.qvalue, weight.scale, spec, mesh=self.mesh, out_dtype=x.dtype
            )
        return x @ weight.T

    def hidden_states(
        self, params: Qwen3Params, caches: Caches, inputs: ModelInputs
    ) -> tuple[jax.Array, Caches]:
        config = self.config
        eps = config.rms_norm_eps
        tokens = inputs.input_ids.shape[0]
        heads, kv_heads, head_dim = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        x = params.embed[inputs.input_ids]
        layer_caches = []
        for weights, kv_cache in zip(params.layers, caches[0], strict=True):
            h = rms_norm(x, weights.input_norm, eps)
            q = self._linear(h, weights.q, COLUMN).reshape(tokens, heads, head_dim)
            k = self._linear(h, weights.k, COLUMN).reshape(tokens, kv_heads, head_dim)
            v = self._linear(h, weights.v, COLUMN).reshape(tokens, kv_heads, head_dim)
            q = apply_rope(
                rms_norm(q, weights.q_norm, eps),
                inputs.positions,
                rope_theta=config.rope_theta,
            )
            k = apply_rope(
                rms_norm(k, weights.k_norm, eps),
                inputs.positions,
                rope_theta=config.rope_theta,
            )
            attended, kv_cache = paged_attention(
                q,
                k,
                v,
                kv_cache,
                inputs.attention,
                mesh=self.mesh,
                sm_scale=head_dim**-0.5,
                impl=self.impl,
            )
            layer_caches.append(kv_cache)
            x = x + self._linear(
                attended.reshape(tokens, heads * head_dim), weights.o, ROW
            )
            h = rms_norm(x, weights.post_norm, eps)
            gated = jax.nn.silu(self._linear(h, weights.gate, COLUMN)) * self._linear(
                h, weights.up, COLUMN
            )
            x = x + self._linear(gated, weights.down, ROW)
        return rms_norm(x, params.norm, eps), (tuple(layer_caches),)

    def logits(self, params: Qwen3Params, hidden: jax.Array) -> jax.Array:
        return hidden @ params.lm_head.T


__all__ = [
    "COLUMN",
    "ROW",
    "LayerParams",
    "Linear",
    "Qwen3Model",
    "Qwen3Params",
    "check_tensor_parallel",
]
