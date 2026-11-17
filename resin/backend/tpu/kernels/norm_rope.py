# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused RMSNorm, RoPE, and Pallas RMSNorm+FP8 quantization kernels for TPU.

Adapted from vllm-project/tpu-inference (Apache-2.0) at revision
d9c7e19da8c497ec63b9381bb0bc9b2a5e0d4bb2:
tpu_inference/layers/jax/{norm.py,rope.py,rope_interface.py}.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from resin.backend.tpu.kernels._rpa_v3_pallas import align_to, get_tpu_version
from resin.backend.tpu.kernels.fp8_matmul import quantize_fp8_act, safe_fp8_scale


def rms_norm(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
) -> jax.Array:
    """Apply RMSNorm over the trailing axis with FP32 variance accumulation."""
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(variance + eps)
    if weight is not None:
        normed = normed * weight.astype(jnp.float32)
    return normed.astype(x.dtype)


def fused_add_rms_norm(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """Fuse residual addition with RMSNorm, returning ``(normed, next_residual)``."""
    next_residual = (x + residual).astype(x.dtype)
    return rms_norm(next_residual, weight=weight, eps=eps), next_residual


def _rms_norm_fp8_pallas_kernel(
    x_ref,
    w_ref,
    x_q_ref,
    scale_t_ref,
    *,
    eps: float,
    bound: float,
) -> None:
    x_f32 = x_ref[...].astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = (x_f32 * jax.lax.rsqrt(var + eps) * w_ref[...].astype(jnp.float32)).astype(
        x_ref.dtype
    )
    normed_f32 = normed.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(normed_f32), axis=-1, keepdims=True)
    scale = safe_fp8_scale(absmax, bound)
    x_q_ref[...] = jnp.clip(normed_f32 / scale, -bound, bound).astype(x_q_ref.dtype)
    scale_t_ref[...] = jnp.transpose(scale)


def pallas_rms_norm_fp8_quant(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
    *,
    dtype: Any = jnp.float8_e4m3fn,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Single-pass Pallas VMEM kernel fusing RMSNorm with per-token FP8 quantization."""
    target = jnp.dtype(dtype)
    bound = float(jnp.finfo(target).max)
    leading = x.shape[:-1]
    k_dim = x.shape[-1]
    if k_dim % 128 != 0:
        raise ValueError(f"Pallas RMSNorm+FP8 requires K divisible by 128, got {k_dim}")

    x_2d = x.reshape(-1, k_dim)
    orig_batch = x_2d.shape[0]
    batch_block = (
        align_to(max(8, orig_batch), 8)
        if orig_batch < 128
        else min(align_to(orig_batch, 128), 512)
    )
    padded_batch = align_to(orig_batch, batch_block)
    if orig_batch < padded_batch:
        x_2d = jnp.pad(x_2d, ((0, padded_batch - orig_batch), (0, 0)))

    w_2d = (
        jnp.ones((1, k_dim), dtype=jnp.float32)
        if weight is None
        else weight.reshape(1, k_dim)
    )
    n_batch = padded_batch // batch_block

    kernel = pl.pallas_call(
        functools.partial(_rms_norm_fp8_pallas_kernel, eps=eps, bound=bound),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block, k_dim), lambda b: (b, 0)),
                pl.BlockSpec((1, k_dim), lambda _b: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((batch_block, k_dim), lambda b: (b, 0)),
                pl.BlockSpec((1, batch_block), lambda b: (0, b)),
            ],
            grid=(n_batch,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((padded_batch, k_dim), target),
            jax.ShapeDtypeStruct((1, padded_batch), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"rms_norm_fp8_quant_{batch_block}_{k_dim}",
    )
    x_q, scale_t = kernel(x_2d, w_2d)
    return (
        x_q[:orig_batch].reshape(*leading, k_dim),
        jnp.transpose(scale_t)[:orig_batch].reshape(*leading, 1),
    )


def rms_norm_fp8_quant(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
    *,
    dtype: Any = jnp.float8_e4m3fn,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fuse RMSNorm with dynamic per-token FP8 quantization, preserving BF16 rounding."""
    if x.shape[-1] % 128 == 0 and (interpret or get_tpu_version() >= 6):
        return pallas_rms_norm_fp8_quant(
            x, weight=weight, eps=eps, dtype=dtype, interpret=interpret
        )
    return quantize_fp8_act(rms_norm(x, weight=weight, eps=eps), dtype=dtype)


def _rope_inv_freq(actual_dim: int, rotary_dim: int, rope_theta: float) -> jax.Array:
    fractions = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / float(actual_dim)
    return 1.0 / (rope_theta**fractions)


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    rope_theta: float = 1000000.0,
) -> tuple[jax.Array, jax.Array]:
    """Precompute ``(cos, sin)`` tables in ``float32`` of shape ``[max_seq_len, head_dim // 2]``."""
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"Expected positive even head_dim, got {head_dim}")
    inv_freq = _rope_inv_freq(head_dim, head_dim, rope_theta)
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("t,f->tf", t, inv_freq, precision=jax.lax.Precision.HIGHEST)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    head_dim: int | None = None,
    *,
    rope_theta: float = 1000000.0,
    cos_sin_cache: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Apply split-half rotary positional embeddings to ``[tokens, heads, padded_head_dim]``."""
    if inputs.ndim != 3:
        raise ValueError(f"Expected 3D [tokens, heads, head_dim], got {inputs.shape=}")
    if positions.ndim != 1 or positions.shape[0] != inputs.shape[0]:
        raise ValueError(
            f"Expected 1D positions of length {inputs.shape[0]}, got {positions.shape=}"
        )

    padded_dim = inputs.shape[-1]
    actual_dim = padded_dim if head_dim is None else head_dim
    if actual_dim <= 0 or actual_dim % 2 != 0 or actual_dim > padded_dim:
        raise ValueError(f"Invalid RoPE dims: {actual_dim=}, {padded_dim=}")

    if cos_sin_cache is not None:
        cos_table, sin_table = cos_sin_cache
        cos = cos_table[positions][:, None, :].astype(jnp.float32)
        sin = sin_table[positions][:, None, :].astype(jnp.float32)
    else:
        inv_freq = _rope_inv_freq(actual_dim, actual_dim, rope_theta)
        freqs = jnp.einsum(
            "t,f->tf",
            positions.astype(jnp.float32),
            inv_freq,
            precision=jax.lax.Precision.HIGHEST,
        )
        cos = jnp.cos(freqs)[:, None, :]
        sin = jnp.sin(freqs)[:, None, :]

    half = actual_dim // 2
    first = inputs[..., :half].astype(jnp.float32)
    second = inputs[..., half:actual_dim].astype(jnp.float32)
    rotated = jnp.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    ).astype(inputs.dtype)

    if actual_dim < padded_dim:
        rotated = jnp.pad(rotated, ((0, 0), (0, 0), (0, padded_dim - actual_dim)))
    return rotated


__all__ = [
    "apply_rope",
    "build_rope_cache",
    "fused_add_rms_norm",
    "pallas_rms_norm_fp8_quant",
    "rms_norm",
    "rms_norm_fp8_quant",
]
