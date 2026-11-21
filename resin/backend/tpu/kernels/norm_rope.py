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
"""Fused RMSNorm, RoPE, and Pallas RMSNorm+FP8 / QK-RoPE-pack kernels for TPU.

Adapted from vllm-project/tpu-inference (Apache-2.0) at revision
d9c7e19da8c497ec63b9381bb0bc9b2a5e0d4bb2:
tpu_inference/layers/jax/{norm.py,rope.py,rope_interface.py}.
"""

from __future__ import annotations

import functools
from typing import Any, Literal, overload

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from resin.backend.tpu.kernels._rpa_v3_pallas import (
    align_to,
    get_dtype_packing,
    get_tpu_version,
)
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
    w_f32 = w_ref[...].astype(jnp.float32)[None, :]
    normed = (x_f32 * jax.lax.rsqrt(var + eps) * w_f32).astype(x_ref.dtype)
    normed_f32 = normed.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(normed_f32), axis=-1, keepdims=True)
    scale = safe_fp8_scale(absmax, bound)
    x_q_ref[...] = jnp.clip(normed_f32 / scale, -bound, bound).astype(x_q_ref.dtype)
    scale_t_ref[...] = jnp.transpose(scale)


def _fused_add_rms_norm_fp8_pallas_kernel(
    x_ref,
    res_ref,
    w_ref,
    x_q_ref,
    scale_t_ref,
    next_res_ref,
    *,
    eps: float,
    bound: float,
) -> None:
    next_res = (
        x_ref[...].astype(jnp.float32) + res_ref[...].astype(jnp.float32)
    ).astype(x_ref.dtype)
    next_res_ref[...] = next_res
    x_f32 = next_res.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    w_f32 = w_ref[...].astype(jnp.float32)[None, :]
    normed = (x_f32 * jax.lax.rsqrt(var + eps) * w_f32).astype(x_ref.dtype)
    normed_f32 = normed.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(normed_f32), axis=-1, keepdims=True)
    scale = safe_fp8_scale(absmax, bound)
    x_q_ref[...] = jnp.clip(normed_f32 / scale, -bound, bound).astype(x_q_ref.dtype)
    scale_t_ref[...] = jnp.transpose(scale)


def _pallas_vmem_limit_bytes(*, interpret: bool, need_48m: bool = False) -> int | None:
    if interpret:
        return None
    tpu_ver = get_tpu_version()
    if need_48m and tpu_ver >= 7:
        return 48 * 1024 * 1024
    return 32 * 1024 * 1024 if tpu_ver >= 6 else None


@overload
def pallas_rms_norm_fp8_quant(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
    *,
    residual: None = None,
    dtype: Any = jnp.float8_e4m3fn,
    transposed_scale: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]: ...


@overload
def pallas_rms_norm_fp8_quant(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
    *,
    residual: jax.Array,
    dtype: Any = jnp.float8_e4m3fn,
    transposed_scale: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]: ...


def pallas_rms_norm_fp8_quant(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
    *,
    residual: jax.Array | None = None,
    dtype: Any = jnp.float8_e4m3fn,
    transposed_scale: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Single-pass Pallas VMEM kernel fusing optional residual add, RMSNorm, and FP8 quant."""
    target = jnp.dtype(dtype)
    bound = float(jnp.finfo(target).max)
    leading = x.shape[:-1]
    k_dim = x.shape[-1]
    if k_dim % 128 != 0:
        raise ValueError(f"Pallas RMSNorm+FP8 requires K divisible by 128, got {k_dim}")
    if residual is not None and residual.shape != x.shape:
        raise ValueError(f"Expected residual shape {x.shape}, got {residual.shape}")

    x_2d = x if x.ndim == 2 else x.reshape(-1, k_dim)
    orig_batch = x_2d.shape[0]
    if residual is not None:
        max_block = 128 if k_dim > 8192 else (256 if k_dim >= 4096 else 512)
    else:
        max_block = 512 if k_dim <= 8192 else 256
    single_tile_cap = 320 if k_dim <= 6144 else min(256, max_block)
    batch_block = (
        align_to(max(8, orig_batch), 8)
        if orig_batch <= single_tile_cap
        else min(align_to(orig_batch, 128), max_block)
    )
    padded_batch = align_to(orig_batch, batch_block)
    if orig_batch < padded_batch:
        x_2d = jnp.pad(x_2d, ((0, padded_batch - orig_batch), (0, 0)))

    if weight is None:
        w_1d = jnp.ones((k_dim,), dtype=jnp.float32)
    elif weight.shape == (k_dim,):
        w_1d = weight
    elif weight.size == k_dim:
        w_1d = weight.reshape(k_dim)
    else:
        raise ValueError(f"Expected weight with {k_dim} elements, got {weight.shape=}")
    n_batch = padded_batch // batch_block
    b_idx = (lambda _b: 0) if n_batch == 1 else (lambda b: b)
    vmem_limit_bytes = _pallas_vmem_limit_bytes(interpret=interpret)

    if residual is None:
        kernel = pl.pallas_call(
            functools.partial(_rms_norm_fp8_pallas_kernel, eps=eps, bound=bound),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((batch_block, k_dim), lambda b: (b_idx(b), 0)),
                    pl.BlockSpec((k_dim,), lambda _b: (0,)),
                ],
                out_specs=[
                    pl.BlockSpec((batch_block, k_dim), lambda b: (b_idx(b), 0)),
                    pl.BlockSpec((1, batch_block), lambda b: (0, b_idx(b))),
                ],
                grid=(n_batch,),
            ),
            out_shape=[
                jax.ShapeDtypeStruct((padded_batch, k_dim), target),
                jax.ShapeDtypeStruct((1, padded_batch), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            interpret=pltpu.InterpretParams() if interpret else False,
            name=f"rms_norm_fp8_quant_{batch_block}_{k_dim}",
        )
        x_q, scale_t = kernel(x_2d, w_1d)
        if orig_batch < padded_batch:
            x_q = x_q[:orig_batch]
            scale_t = scale_t[:, :orig_batch]
        scale_out = scale_t if transposed_scale else scale_t.reshape(*leading, 1)
        return (
            x_q if x.ndim == 2 else x_q.reshape(*leading, k_dim),
            scale_out,
        )

    res_2d = residual.astype(x.dtype)
    if res_2d.ndim != 2:
        res_2d = res_2d.reshape(-1, k_dim)
    if orig_batch < padded_batch:
        res_2d = jnp.pad(res_2d, ((0, padded_batch - orig_batch), (0, 0)))

    fused_kernel = pl.pallas_call(
        functools.partial(_fused_add_rms_norm_fp8_pallas_kernel, eps=eps, bound=bound),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block, k_dim), lambda b: (b_idx(b), 0)),
                pl.BlockSpec((batch_block, k_dim), lambda b: (b_idx(b), 0)),
                pl.BlockSpec((k_dim,), lambda _b: (0,)),
            ],
            out_specs=[
                pl.BlockSpec((batch_block, k_dim), lambda b: (b_idx(b), 0)),
                pl.BlockSpec((1, batch_block), lambda b: (0, b_idx(b))),
                pl.BlockSpec((batch_block, k_dim), lambda b: (b_idx(b), 0)),
            ],
            grid=(n_batch,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((padded_batch, k_dim), target),
            jax.ShapeDtypeStruct((1, padded_batch), jnp.float32),
            jax.ShapeDtypeStruct((padded_batch, k_dim), x.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"fused_add_rms_norm_fp8_quant_{batch_block}_{k_dim}",
    )
    x_q, scale_t, next_res = fused_kernel(x_2d, res_2d, w_1d)
    if orig_batch < padded_batch:
        x_q = x_q[:orig_batch]
        scale_t = scale_t[:, :orig_batch]
        next_res = next_res[:orig_batch]
    scale_out = scale_t if transposed_scale else scale_t.reshape(*leading, 1)
    return (
        x_q if x.ndim == 2 else x_q.reshape(*leading, k_dim),
        scale_out,
        next_res if x.ndim == 2 else next_res.reshape(*leading, k_dim),
    )


def rms_norm_fp8_quant(
    x: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
    *,
    dtype: Any = jnp.float8_e4m3fn,
    transposed_scale: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fuse RMSNorm with dynamic per-token FP8 quantization, preserving BF16 rounding."""
    if x.shape[-1] % 128 == 0 and (interpret or get_tpu_version() >= 6):
        return pallas_rms_norm_fp8_quant(
            x,
            weight=weight,
            eps=eps,
            dtype=dtype,
            transposed_scale=transposed_scale,
            interpret=interpret,
        )
    x_q, x_scale = quantize_fp8_act(rms_norm(x, weight=weight, eps=eps), dtype=dtype)
    if transposed_scale:
        x_scale = jnp.transpose(x_scale.reshape(-1, 1))
    return x_q, x_scale


def fused_add_rms_norm_fp8_quant(
    x: jax.Array,
    residual: jax.Array,
    weight: jax.Array | None = None,
    eps: float = 1e-6,
    *,
    dtype: Any = jnp.float8_e4m3fn,
    transposed_scale: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fuse residual add, RMSNorm, and dynamic per-token FP8 quantization."""
    if x.shape[-1] % 128 == 0 and (interpret or get_tpu_version() >= 6):
        return pallas_rms_norm_fp8_quant(
            x,
            weight=weight,
            eps=eps,
            residual=residual,
            dtype=dtype,
            transposed_scale=transposed_scale,
            interpret=interpret,
        )
    normed, next_res = fused_add_rms_norm(x, residual, weight=weight, eps=eps)
    x_q, x_scale = quantize_fp8_act(normed, dtype=dtype)
    if transposed_scale:
        x_scale = jnp.transpose(x_scale.reshape(-1, 1))
    return x_q, x_scale, next_res


def _rope_inv_freq(actual_dim: int, rotary_dim: int, rope_theta: float) -> jax.Array:
    fractions = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / float(actual_dim)
    return 1.0 / (rope_theta**fractions)


def _half_swap_perm(head_dim: int = 128, dtype: Any = jnp.bfloat16) -> jax.Array:
    half = head_dim // 2
    eye = jnp.eye(half, dtype=dtype)
    zeros = jnp.zeros((half, half), dtype=dtype)
    return jnp.block([[zeros, eye], [eye, zeros]])


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    rope_theta: float = 1000000.0,
) -> tuple[jax.Array, jax.Array]:
    """Precompute full-width ``(cos_full, sin_signed)`` in ``float32`` of shape ``[max_seq_len, head_dim]``."""
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"Expected positive even head_dim, got {head_dim}")
    inv_freq = _rope_inv_freq(head_dim, head_dim, rope_theta)
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("t,f->tf", t, inv_freq, precision=jax.lax.Precision.HIGHEST)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return jnp.concatenate([cos, cos], axis=-1), jnp.concatenate([-sin, sin], axis=-1)


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

    half = actual_dim // 2
    if cos_sin_cache is not None:
        cos_table, sin_table = cos_sin_cache
        if cos_table.shape[-1] == actual_dim:
            if (
                actual_dim == 128
                and padded_dim == 128
                and inputs.dtype == jnp.bfloat16
                and jax.default_backend() == "tpu"
            ):
                cos_pos = cos_table[positions][:, None, :].astype(jnp.float32)
                sin_pos = sin_table[positions][:, None, :].astype(jnp.float32)
                perm = _half_swap_perm(128, dtype=jnp.bfloat16)
                swapped = jnp.matmul(
                    inputs.reshape(-1, 128),
                    perm,
                    preferred_element_type=jnp.float32,
                ).reshape(inputs.shape)
                return (
                    inputs.astype(jnp.float32) * cos_pos + swapped * sin_pos
                ).astype(inputs.dtype)
            cos = cos_table[positions][:, None, :half].astype(jnp.float32)
            sin = sin_table[positions][:, None, half:].astype(jnp.float32)
        elif cos_table.shape[-1] == half:
            cos = cos_table[positions][:, None, :].astype(jnp.float32)
            sin = sin_table[positions][:, None, :].astype(jnp.float32)
        else:
            raise ValueError(
                f"Expected RoPE cache width {actual_dim} or {half}, got {cos_table.shape[-1]}"
            )
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

    first = inputs[..., :half].astype(jnp.float32)
    second = inputs[..., half:actual_dim].astype(jnp.float32)
    rotated = jnp.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    ).astype(inputs.dtype)

    if actual_dim < padded_dim:
        rotated = jnp.pad(rotated, ((0, 0), (0, 0), (0, padded_dim - actual_dim)))
    return rotated


def _select_qk_token_block(
    orig_tokens: int,
    *,
    total_heads: int = 40,
    interpret: bool = False,
) -> int:
    if orig_tokens <= 320:
        return align_to(max(8, orig_tokens), 8)
    use_tb512 = total_heads <= 48 and (interpret or get_tpu_version() >= 7)
    candidates = (512, 256, 128, 64, 32, 16, 8) if use_tb512 else (128, 64, 32, 16, 8)
    for candidate in candidates:
        if orig_tokens % candidate == 0:
            return candidate
    return 512 if (use_tb512 and orig_tokens >= 512) else 128


def _saturate_cast_kv(
    x: jax.Array, kv_dtype: jnp.dtype, scale: float | None
) -> jax.Array:
    if kv_dtype.itemsize != 1:
        return x.astype(kv_dtype)
    bound = float(jnp.finfo(kv_dtype).max)
    eff_scale = 1.0 if scale is None else scale
    return jnp.clip(x.astype(jnp.float32) / eff_scale, -bound, bound).astype(kv_dtype)


def _norm_and_rope_group_128(
    tiles: list[jax.Array],
    norm_w_f32: jax.Array,
    cos_rep: jax.Array,
    sin_rep: jax.Array,
    perm: jax.Array,
    eps: float,
) -> list[jax.Array]:
    count = len(tiles)
    bsz = tiles[0].shape[0]
    stacked = jnp.concatenate(tiles, axis=0)
    stacked_f32 = stacked.astype(jnp.float32)
    var = jnp.mean(jnp.square(stacked_f32), axis=-1, keepdims=True)
    normed = (stacked_f32 * jax.lax.rsqrt(var + eps) * norm_w_f32).astype(
        tiles[0].dtype
    )
    swapped = jnp.matmul(normed, perm, preferred_element_type=jnp.float32)
    roped = (normed.astype(jnp.float32) * cos_rep + swapped * sin_rep).astype(
        tiles[0].dtype
    )
    return [roped[idx * bsz : (idx + 1) * bsz] for idx in range(count)]


def _process_and_store_qk_kv(
    q_tile: jax.Array,
    k_tile: jax.Array,
    v_tile: jax.Array,
    cos_b: jax.Array,
    sin_b: jax.Array,
    q_norm_w_ref,
    k_norm_w_ref,
    perm: jax.Array,
    q_out_ref,
    kv_out_ref,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    kv_dtype: jnp.dtype,
    k_scale: float | None,
    v_scale: float | None,
    eps: float,
    pack_mode: Literal["bitcast_u32", "reshape"],
) -> None:
    bsz = q_tile.shape[0]
    q_dtype = q_tile.dtype
    group_size = num_q_heads // num_kv_heads
    padded_group = align_to(group_size, 2)
    pairs_per_kv = padded_group // 2
    kv_packing = get_dtype_packing(kv_dtype)
    kv_rows = align_to(2 * num_kv_heads, kv_packing) // kv_packing

    qw_f32 = q_norm_w_ref[...].astype(jnp.float32)[None, :]
    kw_f32 = k_norm_w_ref[...].astype(jnp.float32)[None, :]
    zero_q = jnp.zeros((bsz, 128), dtype=q_dtype)
    q_cos_rep = jnp.tile(cos_b, (group_size, 1))
    q_sin_rep = jnp.tile(sin_b, (group_size, 1))

    if pack_mode == "reshape":
        kv_groups = []
        for h in range(num_kv_heads):
            q_group_in = [
                q_tile[:, (h * group_size + g) * 128 : (h * group_size + g + 1) * 128]
                for g in range(group_size)
            ]
            q_group_rot = _norm_and_rope_group_128(
                q_group_in, qw_f32, q_cos_rep, q_sin_rep, perm, eps
            )
            padded_list = [
                q_group_rot[g] if g < group_size else zero_q
                for g in range(padded_group)
            ]
            kv_groups.append(
                jnp.stack(padded_list, axis=1).reshape(bsz, pairs_per_kv, 2, 128)
            )
        q_out_ref[...] = jnp.stack(kv_groups, axis=0)
    else:
        q_u32_ref = q_out_ref.bitcast(jnp.uint32)
        for h in range(num_kv_heads):
            q_group_in = [
                q_tile[:, (h * group_size + g) * 128 : (h * group_size + g + 1) * 128]
                for g in range(group_size)
            ]
            q_group_rot = _norm_and_rope_group_128(
                q_group_in, qw_f32, q_cos_rep, q_sin_rep, perm, eps
            )
            head_u32_ref = q_u32_ref.at[h].reshape(bsz * pairs_per_kv, 128)
            for p in range(pairs_per_kv):
                g0, g1 = 2 * p, 2 * p + 1
                q_even = q_group_rot[g0] if g0 < group_size else zero_q
                q_odd = q_group_rot[g1] if g1 < group_size else zero_q
                u_even = pltpu.bitcast(
                    q_even.astype(jnp.float32), jnp.uint32
                ) >> jnp.uint32(16)
                u_odd = pltpu.bitcast(q_odd.astype(jnp.float32), jnp.uint32)
                head_u32_ref[pl.ds(p, bsz, pairs_per_kv)] = u_even | u_odd

    k_heads_in = [k_tile[:, h * 128 : (h + 1) * 128] for h in range(num_kv_heads)]
    v_heads_in = [v_tile[:, h * 128 : (h + 1) * 128] for h in range(num_kv_heads)]
    k_cos_rep = (
        q_cos_rep if num_kv_heads == group_size else jnp.tile(cos_b, (num_kv_heads, 1))
    )
    k_sin_rep = (
        q_sin_rep if num_kv_heads == group_size else jnp.tile(sin_b, (num_kv_heads, 1))
    )
    k_roped = _norm_and_rope_group_128(
        k_heads_in, kw_f32, k_cos_rep, k_sin_rep, perm, eps
    )
    k_quant = [_saturate_cast_kv(kh, kv_dtype, k_scale) for kh in k_roped]
    v_quant = [_saturate_cast_kv(vh, kv_dtype, v_scale) for vh in v_heads_in]

    zero_kv = jnp.zeros((bsz, 128), dtype=kv_dtype)
    interleaved_kv: list[jax.Array] = []
    for h in range(num_kv_heads):
        interleaved_kv.append(k_quant[h])
        interleaved_kv.append(v_quant[h])
    while len(interleaved_kv) < kv_rows * kv_packing:
        interleaved_kv.append(zero_kv)

    if pack_mode == "reshape":
        kv_out_ref[...] = jnp.stack(interleaved_kv, axis=1).reshape(
            bsz, kv_rows, kv_packing, 128
        )
        return

    kv_u32_ref = kv_out_ref.bitcast(jnp.uint32).reshape(bsz * kv_rows, 128)
    for r in range(kv_rows):
        if kv_packing == 2:
            u0 = pltpu.bitcast(
                interleaved_kv[2 * r].astype(jnp.float32), jnp.uint32
            ) >> jnp.uint32(16)
            u1 = pltpu.bitcast(
                interleaved_kv[2 * r + 1].astype(jnp.float32), jnp.uint32
            )
            row_u32 = u0 | u1
        elif kv_packing == 4:
            b0 = pltpu.bitcast(interleaved_kv[4 * r], jnp.uint8).astype(jnp.uint32)
            b1 = pltpu.bitcast(interleaved_kv[4 * r + 1], jnp.uint8).astype(
                jnp.uint32
            ) << jnp.uint32(8)
            b2 = pltpu.bitcast(interleaved_kv[4 * r + 2], jnp.uint8).astype(
                jnp.uint32
            ) << jnp.uint32(16)
            b3 = pltpu.bitcast(interleaved_kv[4 * r + 3], jnp.uint8).astype(
                jnp.uint32
            ) << jnp.uint32(24)
            row_u32 = b0 | b1 | b2 | b3
        elif kv_packing == 1:
            row_u32 = pltpu.bitcast(interleaved_kv[r].astype(jnp.float32), jnp.uint32)
        else:
            raise ValueError(f"Unsupported kv_packing={kv_packing}")
        kv_u32_ref[pl.ds(r, bsz, kv_rows)] = row_u32


def _qk_norm_rope_pack_pallas_kernel(
    q_ref,
    k_ref,
    v_ref,
    cos_ref,
    sin_ref,
    q_norm_w_ref,
    k_norm_w_ref,
    perm_ref,
    q_out_ref,
    kv_out_ref,
    **kwargs: Any,
) -> None:
    _process_and_store_qk_kv(
        q_ref[...],
        k_ref[...],
        v_ref[...],
        cos_ref[...],
        sin_ref[...],
        q_norm_w_ref,
        k_norm_w_ref,
        perm_ref[...],
        q_out_ref,
        kv_out_ref,
        **kwargs,
    )


def _qkv_norm_rope_pack_pallas_kernel(
    qkv_ref,
    cos_ref,
    sin_ref,
    q_norm_w_ref,
    k_norm_w_ref,
    perm_ref,
    q_out_ref,
    kv_out_ref,
    **kwargs: Any,
) -> None:
    qkv = qkv_ref[...]
    q_end = kwargs["num_q_heads"] * 128
    k_end = q_end + kwargs["num_kv_heads"] * 128
    _process_and_store_qk_kv(
        qkv[:, :q_end],
        qkv[:, q_end:k_end],
        qkv[:, k_end:],
        cos_ref[...],
        sin_ref[...],
        q_norm_w_ref,
        k_norm_w_ref,
        perm_ref[...],
        q_out_ref,
        kv_out_ref,
        **kwargs,
    )


def _run_qk_pack_pallas(
    tensor_inputs: tuple[jax.Array, ...],
    *,
    orig_t: int,
    num_q_heads: int,
    num_kv_heads: int,
    positions: jax.Array,
    cos_sin_cache: tuple[jax.Array, jax.Array],
    q_norm_w: jax.Array,
    k_norm_w: jax.Array,
    kv_dtype: Any,
    k_scale: float | None,
    v_scale: float | None,
    eps: float,
    pack_mode: Literal["auto", "bitcast_u32", "reshape"],
    interpret: bool,
    kernel_fn,
    scope_prefix: str,
) -> tuple[jax.Array, jax.Array]:
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(f"Invalid head counts: {num_q_heads=}, {num_kv_heads=}")
    if positions.shape != (orig_t,):
        raise ValueError(
            f"Expected positions shape ({orig_t},), got {positions.shape=}"
        )
    if q_norm_w.shape != (128,) or k_norm_w.shape != (128,):
        raise ValueError(
            f"Expected (128,) norm weights, got {q_norm_w.shape=}, {k_norm_w.shape=}"
        )

    cos_full, sin_signed = cos_sin_cache
    if cos_full.shape[-1] == 64 and sin_signed.shape[-1] == 64:
        cos_full = jnp.concatenate([cos_full, cos_full], axis=-1)
        sin_signed = jnp.concatenate([-sin_signed, sin_signed], axis=-1)
    if cos_full.shape[-1] != 128 or sin_signed.shape[-1] != 128:
        raise ValueError(
            f"Expected 128-wide RoPE cache, got {cos_full.shape=}, {sin_signed.shape=}"
        )

    cos_pos = cos_full[positions].astype(jnp.float32)
    sin_pos = sin_signed[positions].astype(jnp.float32)

    batch_block = _select_qk_token_block(
        orig_t,
        total_heads=num_q_heads + 2 * num_kv_heads,
        interpret=interpret,
    )
    padded_t = align_to(orig_t, batch_block)
    if orig_t < padded_t:
        pad = ((0, padded_t - orig_t), (0, 0))
        tensor_inputs = tuple(jnp.pad(arr, pad) for arr in tensor_inputs)
        cos_pos = jnp.pad(cos_pos, pad)
        sin_pos = jnp.pad(sin_pos, pad)

    target_kv_dtype = jnp.dtype(kv_dtype)
    if target_kv_dtype.itemsize != 1:
        k_scale = v_scale = None
    group_size = num_q_heads // num_kv_heads
    padded_group = align_to(group_size, 2)
    pairs_per_kv = padded_group // 2
    kv_packing = get_dtype_packing(target_kv_dtype)
    kv_rows = align_to(2 * num_kv_heads, kv_packing) // kv_packing

    resolved_pack: Literal["bitcast_u32", "reshape"] = (
        ("reshape" if interpret else "bitcast_u32")
        if pack_mode == "auto"
        else pack_mode
    )
    perm_128 = _half_swap_perm(128, dtype=jnp.bfloat16)
    n_grid = padded_t // batch_block
    b_idx = (lambda _b: 0) if n_grid == 1 else (lambda b: b)
    vmem_limit_bytes = _pallas_vmem_limit_bytes(
        interpret=interpret,
        need_48m=target_kv_dtype.itemsize > 1 and batch_block >= 512,
    )
    interpret_arg: Any = (
        (True if resolved_pack == "bitcast_u32" else pltpu.InterpretParams())
        if interpret
        else False
    )

    kernel = pl.pallas_call(
        functools.partial(
            kernel_fn,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            kv_dtype=target_kv_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            eps=eps,
            pack_mode=resolved_pack,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                *(
                    pl.BlockSpec((batch_block, arr.shape[1]), lambda b: (b_idx(b), 0))
                    for arr in tensor_inputs
                ),
                pl.BlockSpec((batch_block, 128), lambda b: (b_idx(b), 0)),
                pl.BlockSpec((batch_block, 128), lambda b: (b_idx(b), 0)),
                pl.BlockSpec((128,), lambda _b: (0,)),
                pl.BlockSpec((128,), lambda _b: (0,)),
                pl.BlockSpec((128, 128), lambda _b: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec(
                    (num_kv_heads, batch_block, pairs_per_kv, 2, 128),
                    lambda b: (0, b_idx(b), 0, 0, 0),
                ),
                pl.BlockSpec(
                    (batch_block, kv_rows, kv_packing, 128),
                    lambda b: (b_idx(b), 0, 0, 0),
                ),
            ],
            grid=(n_grid,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct(
                (num_kv_heads, padded_t, pairs_per_kv, 2, 128), jnp.bfloat16
            ),
            jax.ShapeDtypeStruct((padded_t, kv_rows, kv_packing, 128), target_kv_dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret_arg,
        name=f"{scope_prefix}_{batch_block}_{num_q_heads}_{num_kv_heads}",
    )

    q_packed, kv_merged = kernel(
        *tensor_inputs,
        cos_pos,
        sin_pos,
        q_norm_w,
        k_norm_w,
        perm_128,
    )
    if orig_t < padded_t:
        q_packed = q_packed[:, :orig_t]
        kv_merged = kv_merged[:orig_t]
    return q_packed, kv_merged


def pallas_qk_norm_rope_pack(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    positions: jax.Array,
    cos_sin_cache: tuple[jax.Array, jax.Array],
    q_norm_w: jax.Array,
    k_norm_w: jax.Array,
    kv_dtype: Any,
    *,
    k_scale: float | None = None,
    v_scale: float | None = None,
    eps: float = 1e-6,
    pack_mode: Literal["auto", "bitcast_u32", "reshape"] = "auto",
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused Pallas VMEM kernel for Q/K RMSNorm, 128-lane RoPE, KV quant, and RPA v3 packing."""
    orig_t = q.shape[0]
    q_2d = q.reshape(orig_t, -1)
    k_2d = k.reshape(orig_t, -1)
    v_2d = v.reshape(orig_t, -1)
    if q_2d.dtype != jnp.bfloat16:
        raise ValueError(
            f"pallas_qk_norm_rope_pack expects bfloat16 Q, got {q_2d.dtype}"
        )
    if q_2d.shape[1] % 128 != 0 or k_2d.shape[1] % 128 != 0 or k_2d.shape != v_2d.shape:
        raise ValueError(
            f"Invalid Q/K/V shapes for head_dim=128: {q.shape=}, {k.shape=}, {v.shape=}"
        )
    return _run_qk_pack_pallas(
        (q_2d, k_2d, v_2d),
        orig_t=orig_t,
        num_q_heads=q_2d.shape[1] // 128,
        num_kv_heads=k_2d.shape[1] // 128,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        q_norm_w=q_norm_w,
        k_norm_w=k_norm_w,
        kv_dtype=kv_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        eps=eps,
        pack_mode=pack_mode,
        interpret=interpret,
        kernel_fn=_qk_norm_rope_pack_pallas_kernel,
        scope_prefix="qk_norm_rope_pack",
    )


def pallas_qkv_norm_rope_pack(
    qkv: jax.Array,
    positions: jax.Array,
    cos_sin_cache: tuple[jax.Array, jax.Array],
    q_norm_w: jax.Array,
    k_norm_w: jax.Array,
    kv_dtype: Any,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    k_scale: float | None = None,
    v_scale: float | None = None,
    eps: float = 1e-6,
    pack_mode: Literal["auto", "bitcast_u32", "reshape"] = "auto",
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused Pallas VMEM kernel reading contiguous ``qkv`` ``[T, (H_q + 2 * H_kv) * 128]``."""
    orig_t = qkv.shape[0]
    expected_width = (num_q_heads + 2 * num_kv_heads) * 128
    qkv_2d = qkv.reshape(orig_t, -1)
    if qkv_2d.dtype != jnp.bfloat16:
        raise ValueError(
            f"pallas_qkv_norm_rope_pack expects bfloat16 QKV, got {qkv_2d.dtype}"
        )
    if qkv_2d.shape[1] != expected_width:
        raise ValueError(f"Expected QKV width {expected_width}, got {qkv_2d.shape[1]}")
    return _run_qk_pack_pallas(
        (qkv_2d,),
        orig_t=orig_t,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        q_norm_w=q_norm_w,
        k_norm_w=k_norm_w,
        kv_dtype=kv_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        eps=eps,
        pack_mode=pack_mode,
        interpret=interpret,
        kernel_fn=_qkv_norm_rope_pack_pallas_kernel,
        scope_prefix="qkv_norm_rope_pack",
    )


def _unpack_attn_fp8_pallas_kernel(
    in_ref,
    out_ref,
    scale_t_ref,
    *,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
    bound: float,
    return_bf16_and_absmax: bool,
    unpack_mode: Literal["bitcast_u32", "reshape"],
) -> None:
    h_kv, bsz, pairs_per_kv, q_packing, hd = in_ref.shape
    out_width = h_kv * actual_num_q_heads_per_kv_head * actual_head_dim
    if unpack_mode == "bitcast_u32":
        u32_ref = in_ref.bitcast(jnp.uint32)
        tiles: list[jax.Array] = []
        for h in range(h_kv):
            head_u32 = u32_ref.at[h].reshape(bsz * pairs_per_kv, hd)
            for p in range(pairs_per_kv):
                w_u32 = head_u32[pl.ds(p, bsz, pairs_per_kv)]
                g0, g1 = 2 * p, 2 * p + 1
                if g0 < actual_num_q_heads_per_kv_head:
                    even = pltpu.bitcast(w_u32 << jnp.uint32(16), jnp.float32)
                    tiles.append(even[:, :actual_head_dim])
                if g1 < actual_num_q_heads_per_kv_head:
                    odd = pltpu.bitcast(w_u32 & jnp.uint32(0xFFFF0000), jnp.float32)
                    tiles.append(odd[:, :actual_head_dim])
        flat_f32 = jnp.concatenate(tiles, axis=-1)
    else:
        unpacked = (
            in_ref[...]
            .swapaxes(0, 1)
            .reshape(bsz, h_kv, pairs_per_kv * q_packing, hd)[
                :, :, :actual_num_q_heads_per_kv_head, :actual_head_dim
            ]
            .reshape(bsz, out_width)
        )
        flat_f32 = unpacked.astype(jnp.float32)

    absmax = jnp.max(jnp.abs(flat_f32), axis=-1, keepdims=True)
    if return_bf16_and_absmax:
        out_ref[...] = flat_f32.astype(out_ref.dtype)
        scale_t_ref[...] = jnp.transpose(absmax)
        return
    scale = safe_fp8_scale(absmax, bound)
    out_ref[...] = jnp.clip(flat_f32 / scale, -bound, bound).astype(out_ref.dtype)
    scale_t_ref[...] = jnp.transpose(scale)


def pallas_unpack_attn_fp8_quant(
    out_packed: jax.Array,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
    *,
    dtype: Any = jnp.float8_e4m3fn,
    return_bf16_and_absmax: bool = False,
    unpack_mode: Literal["auto", "bitcast_u32", "reshape"] = "auto",
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Unpack 5D RPA v3 ``out_packed`` in VMEM to ``(attn_q, attn_scale)`` or ``(attn_bf16, attn_absmax)``."""
    if out_packed.ndim != 5:
        raise ValueError(
            f"Expected 5D packed attention output, got {out_packed.shape=}"
        )
    h_kv, orig_b, pairs_per_kv, q_packing, hd = out_packed.shape
    padded_group = pairs_per_kv * q_packing
    if not (0 < actual_num_q_heads_per_kv_head <= padded_group):
        raise ValueError(
            f"Invalid {actual_num_q_heads_per_kv_head=} for {padded_group=}"
        )
    if not (0 < actual_head_dim <= hd):
        raise ValueError(f"Invalid {actual_head_dim=} for {hd=}")

    out_width = h_kv * actual_num_q_heads_per_kv_head * actual_head_dim
    if out_width % 128 != 0:
        raise ValueError(
            f"pallas_unpack_attn_fp8_quant requires width divisible by 128, got {out_width}"
        )

    target = jnp.dtype(jnp.bfloat16) if return_bf16_and_absmax else jnp.dtype(dtype)
    bound = 0.0 if return_bf16_and_absmax else float(jnp.finfo(target).max)
    max_unpack_block = (
        512 if (out_width <= 5120 and (interpret or get_tpu_version() >= 7)) else 256
    )
    if orig_b <= 320:
        batch_block = align_to(max(8, orig_b), 8)
    elif max_unpack_block == 512 and (orig_b % 512 == 0 or orig_b % 256 != 0):
        batch_block = 512 if orig_b >= 512 else 256
    else:
        batch_block = 256
    padded_b = align_to(orig_b, batch_block)
    packed_in = (
        jnp.pad(out_packed, ((0, 0), (0, padded_b - orig_b), (0, 0), (0, 0), (0, 0)))
        if orig_b < padded_b
        else out_packed
    )

    can_bitcast = (
        out_packed.dtype == jnp.bfloat16
        and q_packing == 2
        and actual_head_dim == hd
        and hd % 128 == 0
    )
    if unpack_mode == "auto":
        resolved_mode: Literal["bitcast_u32", "reshape"] = (
            "bitcast_u32" if (not interpret and can_bitcast) else "reshape"
        )
    else:
        if unpack_mode == "bitcast_u32" and not can_bitcast:
            raise ValueError(
                f"bitcast_u32 unpack requires bfloat16 q_packing=2 actual_head_dim==hd%128==0, got {out_packed.dtype}, {out_packed.shape}, {actual_head_dim=}"
            )
        resolved_mode = unpack_mode

    n_grid = padded_b // batch_block
    b_idx = (lambda _b: 0) if n_grid == 1 else (lambda b: b)
    vmem_limit_bytes = _pallas_vmem_limit_bytes(
        interpret=interpret,
        need_48m=return_bf16_and_absmax and batch_block >= 512,
    )
    interpret_arg: Any = (
        (True if resolved_mode == "bitcast_u32" else pltpu.InterpretParams())
        if interpret
        else False
    )
    kernel_stem = (
        "unpack_attn_absmax" if return_bf16_and_absmax else "unpack_attn_fp8_quant"
    )

    kernel = pl.pallas_call(
        functools.partial(
            _unpack_attn_fp8_pallas_kernel,
            actual_num_q_heads_per_kv_head=actual_num_q_heads_per_kv_head,
            actual_head_dim=actual_head_dim,
            bound=bound,
            return_bf16_and_absmax=return_bf16_and_absmax,
            unpack_mode=resolved_mode,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (h_kv, batch_block, pairs_per_kv, q_packing, hd),
                    lambda b: (0, b_idx(b), 0, 0, 0),
                )
            ],
            out_specs=[
                pl.BlockSpec((batch_block, out_width), lambda b: (b_idx(b), 0)),
                pl.BlockSpec((1, batch_block), lambda b: (0, b_idx(b))),
            ],
            grid=(n_grid,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((padded_b, out_width), target),
            jax.ShapeDtypeStruct((1, padded_b), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret_arg,
        name=f"{kernel_stem}_{batch_block}_{out_width}",
    )

    attn_out, attn_scale = kernel(packed_in)
    if orig_b < padded_b:
        attn_out = attn_out[:orig_b]
        attn_scale = attn_scale[:, :orig_b]
    return attn_out, attn_scale


__all__ = [
    "apply_rope",
    "build_rope_cache",
    "fused_add_rms_norm",
    "fused_add_rms_norm_fp8_quant",
    "pallas_qk_norm_rope_pack",
    "pallas_qkv_norm_rope_pack",
    "pallas_rms_norm_fp8_quant",
    "pallas_unpack_attn_fp8_quant",
    "rms_norm",
    "rms_norm_fp8_quant",
]
