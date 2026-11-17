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
"""FP8 W8A8 quantization, fused SwiGLU, and Pallas/XLA matmuls for TPU.

Adapted from:
- vllm-project/tpu-inference (Apache-2.0) at revision
  d9c7e19da8c497ec63b9381bb0bc9b2a5e0d4bb2:
  tpu_inference/kernels/quantized_matmul/{kernel.py,tuned_block_sizes.py,util.py}.
- google/qwix (Apache-2.0) at revision
  250aeb1aafbd3d8f6098c42b61d007a04948fd63:
  qwix/_src/core/{numerics.py,qarray.py,dot_general.py} (numerical contract).
"""

from __future__ import annotations

import functools
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from resin.backend.tpu.kernels._rpa_v3_pallas import align_to, get_tpu_version
from resin.backend.tpu.mesh import TP_AXIS

_FP32_TINY_SQRT = float(jnp.sqrt(jnp.finfo(jnp.float32).tiny))
_DEVICE_VMEM_LIMIT = {6: 96 * 1024 * 1024, 7: 48 * 1024 * 1024}

MatmulBackend = Literal["auto", "pallas", "xla", "gspmd_xla"]


class Fp8Weight(NamedTuple):
    """Pre-quantized ``[N, K]`` FP8 weight and its ``f32[N]`` per-channel scale."""

    qvalue: jax.Array
    scale: jax.Array


class Fp8MatmulTileSizes(NamedTuple):
    """Tile sizes ``(batch_block_size, out_block_size, in_block_size)`` for Pallas W8A8."""

    batch_block_size: int
    out_block_size: int
    in_block_size: int


def _build_tpu7x_tuned_table() -> dict[tuple[int, int, int], Fp8MatmulTileSizes]:
    table: dict[tuple[int, int, int], Fp8MatmulTileSizes] = {
        (8, 4096, 5120): Fp8MatmulTileSizes(8, 1024, 5120),
        (8, 512, 5120): Fp8MatmulTileSizes(8, 512, 2560),
        (8, 5120, 4096): Fp8MatmulTileSizes(8, 2560, 4096),
        (8, 5120, 5120): Fp8MatmulTileSizes(8, 1024, 5120),
        (8, 5120, 8192): Fp8MatmulTileSizes(8, 1280, 8192),
        (8, 5120, 12800): Fp8MatmulTileSizes(8, 5120, 512),
        (8, 5120, 25600): Fp8MatmulTileSizes(8, 640, 25600),
        (8, 12800, 5120): Fp8MatmulTileSizes(8, 1280, 5120),
        (8, 25600, 5120): Fp8MatmulTileSizes(8, 1024, 5120),
        (16, 4096, 5120): Fp8MatmulTileSizes(16, 1024, 5120),
        (16, 512, 5120): Fp8MatmulTileSizes(16, 512, 2560),
        (16, 5120, 4096): Fp8MatmulTileSizes(16, 2560, 4096),
        (16, 5120, 5120): Fp8MatmulTileSizes(16, 1024, 5120),
        (16, 5120, 8192): Fp8MatmulTileSizes(16, 1280, 8192),
        (16, 5120, 12800): Fp8MatmulTileSizes(16, 5120, 512),
        (16, 5120, 25600): Fp8MatmulTileSizes(16, 640, 25600),
        (16, 12800, 5120): Fp8MatmulTileSizes(16, 1280, 5120),
        (16, 25600, 5120): Fp8MatmulTileSizes(16, 1024, 5120),
        (32, 4096, 5120): Fp8MatmulTileSizes(32, 2048, 2560),
        (32, 512, 5120): Fp8MatmulTileSizes(32, 512, 2560),
        (32, 5120, 4096): Fp8MatmulTileSizes(32, 5120, 2048),
        (32, 5120, 5120): Fp8MatmulTileSizes(32, 2560, 2560),
        (32, 5120, 8192): Fp8MatmulTileSizes(32, 2560, 4096),
        (32, 5120, 12800): Fp8MatmulTileSizes(32, 2560, 2560),
        (32, 5120, 25600): Fp8MatmulTileSizes(32, 640, 25600),
        (32, 12800, 5120): Fp8MatmulTileSizes(32, 3200, 2560),
        (32, 25600, 5120): Fp8MatmulTileSizes(32, 25600, 512),
        (64, 4096, 5120): Fp8MatmulTileSizes(64, 2048, 2560),
        (64, 512, 5120): Fp8MatmulTileSizes(64, 512, 2560),
        (64, 5120, 4096): Fp8MatmulTileSizes(64, 5120, 2048),
        (64, 5120, 5120): Fp8MatmulTileSizes(64, 5120, 1024),
        (64, 5120, 8192): Fp8MatmulTileSizes(64, 1280, 8192),
        (64, 5120, 12800): Fp8MatmulTileSizes(64, 5120, 2560),
        (64, 5120, 25600): Fp8MatmulTileSizes(64, 640, 25600),
        (64, 12800, 5120): Fp8MatmulTileSizes(64, 3200, 1280),
        (64, 25600, 5120): Fp8MatmulTileSizes(64, 5120, 1024),
        (128, 4096, 5120): Fp8MatmulTileSizes(128, 2048, 2560),
        (128, 512, 5120): Fp8MatmulTileSizes(128, 512, 2560),
        (128, 5120, 4096): Fp8MatmulTileSizes(128, 5120, 1024),
        (128, 5120, 5120): Fp8MatmulTileSizes(128, 5120, 2560),
        (128, 5120, 8192): Fp8MatmulTileSizes(128, 5120, 2048),
        (128, 5120, 12800): Fp8MatmulTileSizes(128, 1024, 2560),
        (128, 5120, 25600): Fp8MatmulTileSizes(128, 512, 25600),
        (128, 12800, 5120): Fp8MatmulTileSizes(128, 2560, 2560),
        (128, 25600, 5120): Fp8MatmulTileSizes(128, 1024, 2560),
        (256, 4096, 5120): Fp8MatmulTileSizes(256, 2048, 2560),
        (256, 512, 5120): Fp8MatmulTileSizes(256, 512, 2560),
        (256, 5120, 4096): Fp8MatmulTileSizes(128, 5120, 4096),
        (256, 5120, 5120): Fp8MatmulTileSizes(256, 2560, 2560),
        (256, 5120, 8192): Fp8MatmulTileSizes(256, 1280, 8192),
        (256, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 512),
        (256, 5120, 25600): Fp8MatmulTileSizes(256, 5120, 2560),
        (256, 12800, 5120): Fp8MatmulTileSizes(256, 2560, 2560),
        (256, 25600, 5120): Fp8MatmulTileSizes(256, 2560, 2560),
        (512, 4096, 5120): Fp8MatmulTileSizes(256, 4096, 2560),
        (512, 512, 5120): Fp8MatmulTileSizes(512, 512, 2560),
        (512, 5120, 4096): Fp8MatmulTileSizes(256, 5120, 4096),
        (512, 5120, 5120): Fp8MatmulTileSizes(256, 5120, 5120),
        (512, 5120, 8192): Fp8MatmulTileSizes(512, 1024, 8192),
        (512, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 2560),
        (512, 5120, 25600): Fp8MatmulTileSizes(512, 2560, 2560),
        (512, 12800, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (512, 25600, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (1024, 4096, 5120): Fp8MatmulTileSizes(256, 4096, 2560),
        (1024, 512, 5120): Fp8MatmulTileSizes(512, 512, 2560),
        (1024, 5120, 4096): Fp8MatmulTileSizes(256, 5120, 4096),
        (1024, 5120, 5120): Fp8MatmulTileSizes(256, 5120, 5120),
        (1024, 5120, 8192): Fp8MatmulTileSizes(1024, 2560, 2048),
        (1024, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 2560),
        (1024, 5120, 25600): Fp8MatmulTileSizes(1024, 2560, 2560),
        (1024, 12800, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (1024, 25600, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (2048, 4096, 5120): Fp8MatmulTileSizes(256, 4096, 2560),
        (2048, 512, 5120): Fp8MatmulTileSizes(512, 512, 2560),
        (2048, 5120, 4096): Fp8MatmulTileSizes(256, 5120, 4096),
        (2048, 5120, 5120): Fp8MatmulTileSizes(256, 5120, 5120),
        (2048, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 2560),
        (2048, 12800, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (2048, 25600, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (4096, 4096, 5120): Fp8MatmulTileSizes(256, 4096, 2560),
        (4096, 512, 5120): Fp8MatmulTileSizes(512, 512, 2560),
        (4096, 5120, 4096): Fp8MatmulTileSizes(256, 5120, 4096),
        (4096, 5120, 5120): Fp8MatmulTileSizes(256, 5120, 5120),
        (4096, 5120, 8192): Fp8MatmulTileSizes(1024, 2560, 2048),
        (4096, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 2560),
        (4096, 5120, 25600): Fp8MatmulTileSizes(1024, 2560, 2560),
        (4096, 12800, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (4096, 25600, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (8192, 5120, 4096): Fp8MatmulTileSizes(256, 5120, 4096),
        (8192, 5120, 5120): Fp8MatmulTileSizes(256, 5120, 5120),
        (8192, 5120, 8192): Fp8MatmulTileSizes(1024, 2560, 2048),
        (8192, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 2560),
        (8192, 5120, 25600): Fp8MatmulTileSizes(1024, 2560, 2560),
        (8192, 12800, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
        (8192, 25600, 5120): Fp8MatmulTileSizes(512, 2560, 5120),
    }
    # Qwen3-32B decode compilation buckets [96, 160, 192, 224, 288, 320]:
    # batch_block == bs keeps n_batch == 1 so x_abs_max [1, bs] needs no 128-lane pad.
    for bs in (96, 160, 192, 224, 288, 320):
        table[(bs, 4096, 5120)] = Fp8MatmulTileSizes(bs, 2048, 2560)
        table[(bs, 512, 5120)] = Fp8MatmulTileSizes(bs, 512, 2560)
        table[(bs, 5120, 4096)] = Fp8MatmulTileSizes(bs, 2560, 2048)
        table[(bs, 5120, 5120)] = Fp8MatmulTileSizes(bs, 2560, 2560)
        table[(bs, 5120, 8192)] = Fp8MatmulTileSizes(bs, 2560, 4096)
        table[(bs, 5120, 12800)] = Fp8MatmulTileSizes(bs, 2560, 1280)
        table[(bs, 5120, 25600)] = Fp8MatmulTileSizes(bs, 2560, 2560)
        table[(bs, 12800, 5120)] = Fp8MatmulTileSizes(bs, 2560, 2560)
        table[(bs, 25600, 5120)] = Fp8MatmulTileSizes(bs, 2560, 2560)
    # Qwen3-32B tp=1 fused QKV (10240, 5120), fused gate_up (51200, 5120), and 2048 row shapes:
    table[(2048, 5120, 8192)] = Fp8MatmulTileSizes(1024, 2560, 2048)
    table[(2048, 5120, 25600)] = Fp8MatmulTileSizes(1024, 2560, 2560)
    for bs in (8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320):
        table[(bs, 10240, 5120)] = Fp8MatmulTileSizes(
            bs, 2048 if bs <= 16 else 2560, 5120 if bs <= 16 else 2560
        )
        table[(bs, 51200, 5120)] = Fp8MatmulTileSizes(
            bs, 1024 if bs <= 16 else 2560, 5120 if bs <= 16 else 2560
        )
    for bs in (512, 1024, 2048, 4096, 8192):
        table[(bs, 10240, 5120)] = Fp8MatmulTileSizes(256, 2560, 5120)
        table[(bs, 51200, 5120)] = Fp8MatmulTileSizes(512, 2560, 5120)
    return table


_TPU7X_FP8_TUNED_BLOCKS = _build_tpu7x_tuned_table()


def _fp8_bound(dtype: Any) -> float:
    target = jnp.dtype(dtype)
    if not jnp.issubdtype(target, jnp.floating) or target.itemsize != 1:
        raise ValueError(f"Expected 8-bit floating dtype, got {target}")
    return float(jnp.finfo(target).max)


def safe_fp8_scale(absmax: jax.Array, bound: float) -> jax.Array:
    raw = absmax.astype(jnp.float32) / bound
    return jnp.where(raw < _FP32_TINY_SQRT, jnp.ones_like(raw), raw)


def _quantize_symmetric_fp8(
    x: jax.Array,
    *,
    axis: int,
    dtype: Any,
    tp_axis: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    target = jnp.dtype(dtype)
    bound = _fp8_bound(target)
    x_f32 = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(x_f32), axis=axis, keepdims=True)
    if tp_axis is not None:
        absmax = jax.lax.pmax(absmax, axis_name=tp_axis)
    scale = safe_fp8_scale(absmax, bound)
    qvalue = jnp.clip(x_f32 / scale, -bound, bound).astype(target)
    return qvalue, scale


def quantize_fp8_weight(
    w: jax.Array,
    *,
    dtype: Any = jnp.float8_e4m3fn,
    contracting_axis: int = -1,
) -> Fp8Weight:
    """Quantize a weight tensor per output channel using Qwix's symmetric absmax rule."""
    qvalue, scale = _quantize_symmetric_fp8(w, axis=contracting_axis, dtype=dtype)
    return Fp8Weight(qvalue=qvalue, scale=jnp.squeeze(scale, axis=contracting_axis))


def quantize_fp8_act(
    x: jax.Array,
    *,
    dtype: Any = jnp.float8_e4m3fn,
    tp_axis: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Dynamically quantize activations per token along the trailing ``K`` axis."""
    return _quantize_symmetric_fp8(x, axis=-1, dtype=dtype, tp_axis=tp_axis)


def _validate_matmul_shapes(x_k: int, w: jax.Array, w_scale: jax.Array) -> None:
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight [N, K], got {w.shape=}")
    _fp8_bound(w.dtype)
    if x_k != w.shape[1]:
        raise ValueError(f"Contracting dim mismatch: {x_k=} vs {w.shape[1]=}")
    if w_scale.shape != (w.shape[0],):
        raise ValueError(
            f"Expected w_scale shape ({w.shape[0]},), got {w_scale.shape=}"
        )


def get_tuned_block_sizes(
    n_batch: int,
    n_out: int,
    n_in: int,
    *,
    tpu_version: int | None = None,
) -> Fp8MatmulTileSizes:
    """Pick Pallas ``(batch_block, out_block, in_block)`` obeying TPU ``(8, 128)`` rules."""
    version = get_tpu_version() if tpu_version is None else tpu_version
    if version >= 7 and (tuned := _TPU7X_FP8_TUNED_BLOCKS.get((n_batch, n_out, n_in))):
        return tuned

    if n_batch < 128:
        batch_block = align_to(max(8, n_batch), 8)
    else:
        batch_block = min(align_to(n_batch, 128), 512)
    out_block = min(align_to(max(128, n_out), 128), 2048)
    in_block = min(align_to(max(128, n_in), 128), 2048)
    return Fp8MatmulTileSizes(batch_block, out_block, in_block)


def _estimate_vmem_limit_bytes(
    *,
    n_batch: int,
    n_out: int,
    n_in: int,
    batch_block_size: int,
    out_block_size: int,
    in_block_size: int,
    x_dtype: Any,
    x_q_dtype: Any,
    w_q_dtype: Any,
    out_dtype: Any,
    save_acc: bool,
    save_x_q: bool,
    tpu_version: int,
) -> int:
    upper = _DEVICE_VMEM_LIMIT.get(tpu_version, 48 * 1024 * 1024)
    x_bytes = batch_block_size * in_block_size * jnp.dtype(x_dtype).itemsize
    absmax_bytes = batch_block_size * 4
    w_bytes = out_block_size * in_block_size * jnp.dtype(w_q_dtype).itemsize
    w_scale_bytes = out_block_size * 4
    out_bytes = batch_block_size * out_block_size * jnp.dtype(out_dtype).itemsize

    in_out = 2 * (x_bytes + absmax_bytes + w_bytes + w_scale_bytes + out_bytes)
    if n_batch > 1 or n_in > 1:
        in_out += x_bytes
    if n_batch > 1:
        in_out += absmax_bytes
    if n_out > 1 or n_in > 1:
        in_out += w_bytes
    if n_out > 1:
        in_out += w_scale_bytes
    if n_batch > 1 or n_out > 1:
        in_out += out_bytes

    scratch = 0
    if save_acc:
        scratch += batch_block_size * out_block_size * 4
    if save_x_q:
        scratch += (
            n_in * batch_block_size * in_block_size * jnp.dtype(x_q_dtype).itemsize
        )
    return min(max(in_out + 2 * scratch, 4 * 1024 * 1024), upper)


def _unfold_bools(
    conditions: tuple[jax.Array | bool, ...],
    resolved: tuple[bool, ...],
    fn,
) -> None:
    if not conditions:
        fn(*resolved)
        return
    head, tail = conditions[0], conditions[1:]
    if isinstance(head, bool):
        _unfold_bools(tail, resolved + (head,), fn)
    else:
        jax.lax.cond(
            head,
            lambda: _unfold_bools(tail, resolved + (True,), fn),
            lambda: _unfold_bools(tail, resolved + (False,), fn),
        )


def _fp8_matmul_pallas_kernel(
    x_ref,
    w_q_ref,
    w_scale_ref,
    x_scale_or_absmax_ref,
    out_ref,
    acc_scratch,
    x_q_scratch,
    *,
    x_q_dtype: jnp.dtype,
    has_x_scale: bool,
    save_acc: bool,
    save_x_q: bool,
) -> None:
    out_idx = pl.program_id(1)
    in_idx = pl.program_id(2)
    n_in = pl.num_programs(2)

    quantize_activation = x_q_dtype != x_ref.dtype
    quant = (out_idx == 0) if save_x_q else quantize_activation
    is_first_step = (in_idx == 0) if save_acc else True
    is_last_step = (in_idx == (n_in - 1)) if save_acc else True
    bound = _fp8_bound(x_q_dtype)

    def body(do_quant: bool, first_step: bool, last_step: bool) -> None:
        if quantize_activation:
            x_scale_row = safe_fp8_scale(x_scale_or_absmax_ref[...], bound)
            x_scale = jnp.transpose(x_scale_row)
            if do_quant:
                inv_scale = jnp.transpose(1.0 / x_scale_row)
                x_q = jnp.clip(
                    x_ref[...].astype(jnp.float32) * inv_scale, -bound, bound
                ).astype(x_q_dtype)
                if save_x_q:
                    if n_in == 1:
                        x_q_scratch[...] = x_q
                    else:
                        x_q_scratch[in_idx] = x_q
            else:
                x_q = x_q_scratch[...] if n_in == 1 else x_q_scratch[in_idx]
            lhs = x_q
        else:
            x_scale = jnp.transpose(x_scale_or_absmax_ref[...]) if has_x_scale else None
            lhs = x_ref[...]

        acc = jax.lax.dot_general(
            lhs,
            w_q_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if not first_step:
            acc = acc + acc_scratch[...]

        if last_step:
            acc = acc * w_scale_ref[...]
            if x_scale is not None:
                acc = acc * x_scale
            out_ref[...] = acc.astype(out_ref.dtype)
        else:
            acc_scratch[...] = acc

    _unfold_bools((quant, is_first_step, is_last_step), (), body)


def fp8_matmul_prequantized(
    x_q: jax.Array,
    x_scale: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    tp_axis: str | None = None,
    scatter_axis: str | None = None,
) -> jax.Array:
    """Compute ``(x_q @ w.T) * x_scale * w_scale`` on pre-quantized FP8 activations."""
    _validate_matmul_shapes(x_q.shape[-1], w, w_scale)
    leading = x_q.shape[:-1]
    x_q_2d = x_q.reshape(-1, x_q.shape[-1])
    x_scale_2d = x_scale.reshape(-1, 1).astype(jnp.float32)

    dot = jax.lax.dot_general(
        x_q_2d,
        w,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    eff_reduce = jnp.dtype(out_dtype if reduce_dtype is None else reduce_dtype)
    if eff_reduce == jnp.float32 and scatter_axis is None:
        if tp_axis is not None:
            dot = jax.lax.psum(dot, axis_name=tp_axis)
        scaled = dot * x_scale_2d * w_scale.astype(jnp.float32)[None, :]
        return scaled.astype(out_dtype).reshape(*leading, w.shape[0])

    scaled = (dot * x_scale_2d * w_scale.astype(jnp.float32)[None, :]).astype(
        eff_reduce
    )
    if scatter_axis is not None:
        scaled = jax.lax.psum_scatter(
            scaled, axis_name=scatter_axis, scatter_dimension=0, tiled=True
        )
        return scaled.astype(out_dtype).reshape(-1, *leading[1:], w.shape[0])
    if tp_axis is not None:
        scaled = jax.lax.psum(scaled, axis_name=tp_axis)
    return scaled.astype(out_dtype).reshape(*leading, w.shape[0])


def xla_fp8_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    x_scale: jax.Array | None = None,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    tp_axis: str | None = None,
    scatter_axis: str | None = None,
) -> jax.Array:
    """Pure-JAX Qwix-equivalent FP8 W8A8 matmul using XLA native MXU ``dot_general``."""
    if x.dtype == w.dtype:
        eff_x_scale = (
            jnp.ones((*x.shape[:-1], 1), dtype=jnp.float32)
            if x_scale is None
            else x_scale
        )
        x_q = x
    elif x_abs_max is not None:
        bound = _fp8_bound(w.dtype)
        eff_x_scale = safe_fp8_scale(x_abs_max.reshape(*x.shape[:-1], 1), bound)
        x_q = jnp.clip(x.astype(jnp.float32) / eff_x_scale, -bound, bound).astype(
            w.dtype
        )
    else:
        x_q, eff_x_scale = quantize_fp8_act(x, dtype=w.dtype, tp_axis=tp_axis)
    return fp8_matmul_prequantized(
        x_q,
        eff_x_scale,
        w,
        w_scale,
        out_dtype=out_dtype,
        reduce_dtype=reduce_dtype,
        tp_axis=tp_axis,
        scatter_axis=scatter_axis,
    )


def pallas_fp8_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    x_scale: jax.Array | None = None,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    tp_axis: str | None = None,
    scatter_axis: str | None = None,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused Pallas W8A8 matmul supporting both BF16 and pre-quantized FP8 inputs."""
    _validate_matmul_shapes(x.shape[-1], w, w_scale)

    leading = x.shape[:-1]
    orig_k = x.shape[-1]
    orig_n = w.shape[0]
    x_2d = x.reshape(-1, orig_k)
    orig_batch = x_2d.shape[0]
    x_q_dtype = jnp.dtype(w.dtype)
    quantize_activation = x_q_dtype != x_2d.dtype

    if quantize_activation:
        if x_abs_max is not None:
            absmax_1d = x_abs_max.reshape(orig_batch).astype(jnp.float32)
        else:
            absmax_1d = jnp.max(jnp.abs(x_2d.astype(jnp.float32)), axis=-1)
            if tp_axis is not None:
                absmax_1d = jax.lax.pmax(absmax_1d, axis_name=tp_axis)
        x_scale_or_absmax = jnp.expand_dims(absmax_1d, axis=0)
        has_x_scale = True
    elif x_scale is not None:
        x_scale_or_absmax = x_scale.reshape(1, orig_batch).astype(jnp.float32)
        has_x_scale = True
    else:
        x_scale_or_absmax = jnp.ones((1, orig_batch), dtype=jnp.float32)
        has_x_scale = False

    tpu_version = get_tpu_version()
    if tile_sizes is None:
        tile_sizes = get_tuned_block_sizes(
            orig_batch, orig_n, orig_k, tpu_version=tpu_version
        )
        if (
            quantize_activation
            and tile_sizes.batch_block_size == 512
            and tile_sizes.out_block_size == 2560
            and tile_sizes.in_block_size == 5120
        ):
            tile_sizes = Fp8MatmulTileSizes(256, 2560, 5120)
    batch_block, out_block, in_block = (
        tile_sizes.batch_block_size,
        tile_sizes.out_block_size,
        tile_sizes.in_block_size,
    )

    padded_batch = align_to(orig_batch, batch_block)
    padded_n = align_to(orig_n, out_block)
    padded_k = align_to(orig_k, in_block)

    if orig_batch < padded_batch:
        pad_b = padded_batch - orig_batch
        x_2d = jnp.pad(x_2d, ((0, pad_b), (0, 0)))
        x_scale_or_absmax = jnp.pad(x_scale_or_absmax, ((0, 0), (0, pad_b)))
    if orig_n < padded_n:
        pad_n = padded_n - orig_n
        w = jnp.pad(w, ((0, pad_n), (0, 0)))
        w_scale = jnp.pad(w_scale, (0, pad_n))
    if orig_k < padded_k:
        pad_k = padded_k - orig_k
        x_2d = jnp.pad(x_2d, ((0, 0), (0, pad_k)))
        w = jnp.pad(w, ((0, 0), (0, pad_k)))

    w_scale_2d = jnp.expand_dims(w_scale.astype(jnp.float32), axis=0)
    n_batch = padded_batch // batch_block
    n_out = padded_n // out_block
    n_in = padded_k // in_block

    save_acc = n_in > 1
    save_x_q = (
        quantize_activation
        and n_out > 1
        and (n_in * batch_block * in_block * x_q_dtype.itemsize <= 4 * 1024 * 1024)
    )
    eff_reduce = jnp.dtype(out_dtype if reduce_dtype is None else reduce_dtype)
    kernel_out_dtype = (
        eff_reduce
        if (tp_axis is not None or scatter_axis is not None)
        else jnp.dtype(out_dtype)
    )
    vmem_limit_bytes = _estimate_vmem_limit_bytes(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=batch_block,
        out_block_size=out_block,
        in_block_size=in_block,
        x_dtype=x_2d.dtype,
        x_q_dtype=x_q_dtype,
        w_q_dtype=w.dtype,
        out_dtype=kernel_out_dtype,
        save_acc=save_acc,
        save_x_q=save_x_q,
        tpu_version=tpu_version,
    )

    b_idx = (lambda _b: 0) if n_batch == 1 else (lambda b: b)
    o_idx = (lambda _o: 0) if n_out == 1 else (lambda o: o)
    i_idx = (lambda _i: 0) if n_in == 1 else (lambda i: i)
    x_q_scratch_shape = (
        (batch_block, in_block) if n_in == 1 else (n_in, batch_block, in_block)
    )
    kernel = pl.pallas_call(
        functools.partial(
            _fp8_matmul_pallas_kernel,
            x_q_dtype=x_q_dtype,
            has_x_scale=has_x_scale,
            save_acc=save_acc,
            save_x_q=save_x_q,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (batch_block, in_block), lambda b, _o, i: (b_idx(b), i_idx(i))
                ),
                pl.BlockSpec(
                    (out_block, in_block), lambda _b, o, i: (o_idx(o), i_idx(i))
                ),
                pl.BlockSpec((1, out_block), lambda _b, o, _i: (0, o_idx(o))),
                pl.BlockSpec((1, batch_block), lambda b, _o, _i: (0, b_idx(b))),
            ],
            out_specs=pl.BlockSpec(
                (batch_block, out_block), lambda b, o, _i: (b_idx(b), o_idx(o))
            ),
            scratch_shapes=[
                pltpu.VMEM((batch_block, out_block), jnp.float32) if save_acc else None,
                pltpu.VMEM(x_q_scratch_shape, x_q_dtype) if save_x_q else None,
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_batch, padded_n), kernel_out_dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"fp8_w8a8_matmul_{batch_block}_{out_block}_{in_block}",
    )

    out = kernel(x_2d, w, w_scale_2d, x_scale_or_absmax)[:orig_batch, :orig_n]
    if scatter_axis is not None:
        out = jax.lax.psum_scatter(
            out, axis_name=scatter_axis, scatter_dimension=0, tiled=True
        ).astype(out_dtype)
        return out.reshape(-1, *leading[1:], orig_n)
    if tp_axis is not None:
        out = jax.lax.psum(out, axis_name=tp_axis).astype(out_dtype)
    return out.reshape(*leading, orig_n)


def _gate_up_swiglu_pallas_kernel(
    x_q_ref,
    w_gate_ref,
    w_up_ref,
    w_gate_scale_ref,
    w_up_scale_ref,
    x_scale_ref,
    gated_ref,
    absmax_ref,
    gate_acc_scratch,
    up_acc_scratch,
    *,
    save_acc: bool,
) -> None:
    in_idx = pl.program_id(2)
    n_in = pl.num_programs(2)
    is_first = (in_idx == 0) if save_acc else True
    is_last = (in_idx == (n_in - 1)) if save_acc else True

    def body(first_step: bool, last_step: bool) -> None:
        lhs = x_q_ref[...]
        g_acc = jax.lax.dot_general(
            lhs,
            w_gate_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        u_acc = jax.lax.dot_general(
            lhs,
            w_up_ref[...],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        if not first_step:
            g_acc = g_acc + gate_acc_scratch[...]
            u_acc = u_acc + up_acc_scratch[...]

        if last_step:
            x_scale = jnp.transpose(x_scale_ref[...])
            gate = (g_acc * w_gate_scale_ref[...] * x_scale).astype(gated_ref.dtype)
            up = (u_acc * w_up_scale_ref[...] * x_scale).astype(gated_ref.dtype)
            gated = (jax.nn.silu(gate) * up).astype(gated_ref.dtype)
            gated_ref[...] = gated
            tile_absmax = jnp.max(
                jnp.abs(gated.astype(jnp.float32)), axis=-1, keepdims=True
            )
            absmax_ref[...] = jnp.transpose(tile_absmax)
        else:
            gate_acc_scratch[...] = g_acc
            up_acc_scratch[...] = u_acc

    _unfold_bools((is_first, is_last), (), body)


def pallas_fp8_gate_up_swiglu(
    x_q: jax.Array,
    x_scale: jax.Array,
    w_gate_up: jax.Array,
    w_scale: jax.Array,
    *,
    out_dtype: Any = jnp.bfloat16,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused Pallas W8A8 ``gate_up`` + SwiGLU + tile ``absmax`` for pre-quantized FP8 ``x_q``."""
    _validate_matmul_shapes(x_q.shape[-1], w_gate_up, w_scale)
    if x_q.dtype != w_gate_up.dtype:
        raise ValueError(
            f"pallas_fp8_gate_up_swiglu expects pre-quantized {w_gate_up.dtype} x_q, got {x_q.dtype}"
        )
    if w_gate_up.shape[0] % 2 != 0:
        raise ValueError(f"Expected even gate_up rows, got {w_gate_up.shape[0]}")

    leading = x_q.shape[:-1]
    orig_k = x_q.shape[-1]
    orig_ffn = w_gate_up.shape[0] // 2
    x_2d = x_q.reshape(-1, orig_k)
    orig_batch = x_2d.shape[0]
    x_scale_row = x_scale.reshape(1, orig_batch).astype(jnp.float32)

    if tile_sizes is None:
        batch_block = (
            align_to(max(8, orig_batch), 8)
            if orig_batch < 128
            else min(align_to(orig_batch, 128), 512)
        )
        out_block = (
            1280
            if orig_ffn % 1280 == 0
            else min(align_to(max(128, orig_ffn), 128), 1280)
        )
        in_block = (
            orig_k
            if (orig_k <= 5120 and orig_k % 128 == 0)
            else min(align_to(max(128, orig_k), 128), 2560)
        )
    else:
        batch_block, out_block, in_block = (
            tile_sizes.batch_block_size,
            tile_sizes.out_block_size,
            tile_sizes.in_block_size,
        )

    padded_batch = align_to(orig_batch, batch_block)
    padded_ffn = align_to(orig_ffn, out_block)
    padded_k = align_to(orig_k, in_block)

    w_gate = w_gate_up[:orig_ffn]
    w_up = w_gate_up[orig_ffn:]
    ws_gate = w_scale[:orig_ffn].astype(jnp.float32)
    ws_up = w_scale[orig_ffn:].astype(jnp.float32)

    if orig_batch < padded_batch:
        pad_b = padded_batch - orig_batch
        x_2d = jnp.pad(x_2d, ((0, pad_b), (0, 0)))
        x_scale_row = jnp.pad(x_scale_row, ((0, 0), (0, pad_b)))
    if orig_ffn < padded_ffn:
        pad_n = padded_ffn - orig_ffn
        w_gate = jnp.pad(w_gate, ((0, pad_n), (0, 0)))
        w_up = jnp.pad(w_up, ((0, pad_n), (0, 0)))
        ws_gate = jnp.pad(ws_gate, (0, pad_n))
        ws_up = jnp.pad(ws_up, (0, pad_n))
    if orig_k < padded_k:
        pad_k = padded_k - orig_k
        x_2d = jnp.pad(x_2d, ((0, 0), (0, pad_k)))
        w_gate = jnp.pad(w_gate, ((0, 0), (0, pad_k)))
        w_up = jnp.pad(w_up, ((0, 0), (0, pad_k)))

    n_batch = padded_batch // batch_block
    n_out = padded_ffn // out_block
    n_in = padded_k // in_block
    save_acc = n_in > 1
    target_out_dtype = jnp.dtype(out_dtype)

    tpu_version = get_tpu_version()
    vmem_limit_bytes = (
        _DEVICE_VMEM_LIMIT.get(tpu_version, 48 * 1024 * 1024)
        if (tpu_version >= 6 and not interpret)
        else None
    )

    b_idx = (lambda _b: 0) if n_batch == 1 else (lambda b: b)
    o_idx = (lambda _o: 0) if n_out == 1 else (lambda o: o)
    i_idx = (lambda _i: 0) if n_in == 1 else (lambda i: i)
    kernel = pl.pallas_call(
        functools.partial(_gate_up_swiglu_pallas_kernel, save_acc=save_acc),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(
                    (batch_block, in_block), lambda b, _o, i: (b_idx(b), i_idx(i))
                ),
                pl.BlockSpec(
                    (out_block, in_block), lambda _b, o, i: (o_idx(o), i_idx(i))
                ),
                pl.BlockSpec(
                    (out_block, in_block), lambda _b, o, i: (o_idx(o), i_idx(i))
                ),
                pl.BlockSpec((1, out_block), lambda _b, o, _i: (0, o_idx(o))),
                pl.BlockSpec((1, out_block), lambda _b, o, _i: (0, o_idx(o))),
                pl.BlockSpec((1, batch_block), lambda b, _o, _i: (0, b_idx(b))),
            ],
            out_specs=[
                pl.BlockSpec(
                    (batch_block, out_block), lambda b, o, _i: (b_idx(b), o_idx(o))
                ),
                pl.BlockSpec(
                    (None, 1, batch_block), lambda b, o, _i: (o_idx(o), 0, b_idx(b))
                ),
            ],
            scratch_shapes=[
                pltpu.VMEM((batch_block, out_block), jnp.float32) if save_acc else None,
                pltpu.VMEM((batch_block, out_block), jnp.float32) if save_acc else None,
            ],
            grid=(n_batch, n_out, n_in),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((padded_batch, padded_ffn), target_out_dtype),
            jax.ShapeDtypeStruct((n_out, 1, padded_batch), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"fp8_gate_up_swiglu_{batch_block}_{out_block}_{in_block}",
    )

    gated_padded, absmax_3d = kernel(
        x_2d,
        w_gate,
        w_up,
        jnp.expand_dims(ws_gate, axis=0),
        jnp.expand_dims(ws_up, axis=0),
        x_scale_row,
    )
    gated = gated_padded[:orig_batch, :orig_ffn].reshape(*leading, orig_ffn)
    local_abs_max = absmax_3d[:, 0, :orig_batch]
    return gated, local_abs_max


def xla_fp8_gate_up_swiglu(
    x_q: jax.Array,
    x_scale: jax.Array,
    w_gate_up: jax.Array,
    w_scale: jax.Array,
    *,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """XLA native ``dot_general`` ``gate_up`` + SwiGLU on pre-quantized FP8 ``x_q``."""
    gate_up = fp8_matmul_prequantized(
        x_q, x_scale, w_gate_up, w_scale, out_dtype=out_dtype
    )
    gate, up = jnp.split(gate_up, 2, axis=-1)
    return (jax.nn.silu(gate) * up).astype(out_dtype)


def fp8_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    x_scale: jax.Array | None = None,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    tp_axis: str | None = None,
    scatter_axis: str | None = None,
    backend: MatmulBackend = "auto",
    tile_sizes: Fp8MatmulTileSizes | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Compute Qwix-equivalent FP8 W8A8 ``x @ w.T`` for ``x: [..., K]``, ``w: [N, K]``.

    On TPU7x+ (or when ``interpret=True`` or ``backend="pallas"``), runs the
    fused Pallas kernel; otherwise runs ``xla_fp8_matmul``.
    """
    if backend == "pallas" or (
        backend == "auto" and (interpret or get_tpu_version() >= 7)
    ):
        return pallas_fp8_matmul(
            x,
            w,
            w_scale,
            x_scale=x_scale,
            x_abs_max=x_abs_max,
            out_dtype=out_dtype,
            reduce_dtype=reduce_dtype,
            tp_axis=tp_axis,
            scatter_axis=scatter_axis,
            tile_sizes=tile_sizes,
            interpret=interpret,
        )
    return xla_fp8_matmul(
        x,
        w,
        w_scale,
        x_scale=x_scale,
        x_abs_max=x_abs_max,
        out_dtype=out_dtype,
        reduce_dtype=reduce_dtype,
        tp_axis=tp_axis,
        scatter_axis=scatter_axis,
    )


def sharded_fp8_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    weight_sharding: P | NamedSharding,
    *,
    mesh: Mesh,
    x_scale: jax.Array | None = None,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    backend: MatmulBackend = "auto",
    interpret: bool = False,
) -> jax.Array:
    """Run column-parallel (``P("tp", None)``) or row-parallel (``P(None, "tp")``) ``fp8_matmul``."""
    spec = (
        weight_sharding.spec
        if isinstance(weight_sharding, NamedSharding)
        else weight_sharding
    )
    out_axis = spec[0] if len(spec) > 0 else None
    in_axis = spec[1] if len(spec) > 1 else None
    prefix_none = (None,) * (x.ndim - 1)
    x_spec = P(*prefix_none, in_axis)
    scale_spec = P(out_axis)
    out_spec = P(*prefix_none, out_axis)

    if x_scale is not None:
        xs_spec = P(*((None,) * x_scale.ndim))

        def _local_with_scale(
            x_shard: jax.Array,
            xs_shard: jax.Array,
            w_shard: jax.Array,
            s_shard: jax.Array,
        ) -> jax.Array:
            return fp8_matmul(
                x_shard,
                w_shard,
                s_shard,
                x_scale=xs_shard,
                out_dtype=out_dtype,
                reduce_dtype=reduce_dtype,
                tp_axis=in_axis,
                backend=backend,
                interpret=interpret,
            )

        return jax.shard_map(
            _local_with_scale,
            mesh=mesh,
            in_specs=(x_spec, xs_spec, spec, scale_spec),
            out_specs=out_spec,
            check_vma=False,
        )(x, x_scale, w, w_scale)

    if x_abs_max is not None:
        xa_spec = P(*((None,) * x_abs_max.ndim))

        def _local_with_absmax(
            x_shard: jax.Array,
            xa_shard: jax.Array,
            w_shard: jax.Array,
            s_shard: jax.Array,
        ) -> jax.Array:
            return fp8_matmul(
                x_shard,
                w_shard,
                s_shard,
                x_abs_max=xa_shard,
                out_dtype=out_dtype,
                reduce_dtype=reduce_dtype,
                tp_axis=in_axis,
                backend=backend,
                interpret=interpret,
            )

        return jax.shard_map(
            _local_with_absmax,
            mesh=mesh,
            in_specs=(x_spec, xa_spec, spec, scale_spec),
            out_specs=out_spec,
            check_vma=False,
        )(x, x_abs_max, w, w_scale)

    def _local(x_shard: jax.Array, w_shard: jax.Array, s_shard: jax.Array) -> jax.Array:
        return fp8_matmul(
            x_shard,
            w_shard,
            s_shard,
            out_dtype=out_dtype,
            reduce_dtype=reduce_dtype,
            tp_axis=in_axis,
            backend=backend,
            interpret=interpret,
        )

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(x_spec, spec, scale_spec),
        out_specs=out_spec,
        check_vma=False,
    )(x, w, w_scale)


def sharded_fp8_matmul_all_gather(
    x_q_shard: jax.Array,
    x_scale_shard: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    mesh: Mesh,
    out_dtype: Any = jnp.bfloat16,
    backend: MatmulBackend = "auto",
    interpret: bool = False,
) -> jax.Array:
    """All-gather sequence-parallel FP8 ``x_q`` (``P("tp", None)``) and run column-parallel matmul."""
    if backend == "gspmd_xla":
        leading = x_q_shard.shape[:-1]
        x_q_rep = jax.lax.with_sharding_constraint(
            x_q_shard.reshape(-1, x_q_shard.shape[-1]),
            NamedSharding(mesh, P(None, None)),
        )
        dot = jax.lax.dot_general(
            x_q_rep,
            w,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        dot = jax.lax.with_sharding_constraint(
            dot, NamedSharding(mesh, P(None, TP_AXIS))
        )
        x_scale_rep = jax.lax.with_sharding_constraint(
            x_scale_shard.reshape(-1, 1).astype(jnp.float32),
            NamedSharding(mesh, P(None, None)),
        )
        scaled = (dot * x_scale_rep * w_scale.astype(jnp.float32)[None, :]).astype(
            out_dtype
        )
        return scaled.reshape(*leading, w.shape[0])

    def _local(
        xq_local: jax.Array,
        xs_local: jax.Array,
        w_local: jax.Array,
        ws_local: jax.Array,
    ) -> jax.Array:
        xq_full = jax.lax.all_gather(xq_local, axis_name=TP_AXIS, axis=0, tiled=True)
        xs_full = jax.lax.all_gather(xs_local, axis_name=TP_AXIS, axis=0, tiled=True)
        return fp8_matmul(
            xq_full,
            w_local,
            ws_local,
            x_scale=xs_full,
            out_dtype=out_dtype,
            backend=backend,
            interpret=interpret,
        )

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(TP_AXIS, None), P(TP_AXIS, None), P(TP_AXIS, None), P(TP_AXIS)),
        out_specs=P(None, TP_AXIS),
        check_vma=False,
    )(x_q_shard, x_scale_shard, w, w_scale)


def sharded_fp8_gate_up_swiglu(
    x_q: jax.Array,
    x_scale: jax.Array,
    w_gate_up: jax.Array,
    w_scale: jax.Array,
    *,
    mesh: Mesh,
    sequence_parallel: bool = True,
    out_dtype: Any = jnp.bfloat16,
    backend: MatmulBackend = "auto",
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Run column-parallel ``gate_up`` + SwiGLU, returning ``(gated, synced_absmax)``."""
    use_pallas = backend == "pallas" or (
        backend == "auto"
        and (interpret or get_tpu_version() >= 7)
        and w_gate_up.shape[1] % 128 == 0
        and (w_gate_up.shape[0] // (2 * int(mesh.shape[TP_AXIS]))) % 128 == 0
    )

    if backend == "gspmd_xla":
        leading = x_q.shape[:-1]
        if sequence_parallel:
            gate_up = sharded_fp8_matmul_all_gather(
                x_q,
                x_scale,
                w_gate_up,
                w_scale,
                mesh=mesh,
                out_dtype=out_dtype,
                backend="gspmd_xla",
            )
        else:
            dot = jax.lax.dot_general(
                x_q.reshape(-1, x_q.shape[-1]),
                w_gate_up,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            dot = jax.lax.with_sharding_constraint(
                dot, NamedSharding(mesh, P(None, TP_AXIS))
            )
            gate_up = (
                (
                    dot
                    * x_scale.reshape(-1, 1).astype(jnp.float32)
                    * w_scale.astype(jnp.float32)[None, :]
                )
                .astype(out_dtype)
                .reshape(*leading, w_gate_up.shape[0])
            )

        prefix_none = (None,) * len(leading)

        def _swiglu_local(gu_local: jax.Array) -> jax.Array:
            gate, up = jnp.split(gu_local, 2, axis=-1)
            return (jax.nn.silu(gate) * up).astype(out_dtype)

        gated = jax.shard_map(
            _swiglu_local,
            mesh=mesh,
            in_specs=P(*prefix_none, TP_AXIS),
            out_specs=P(*prefix_none, TP_AXIS),
            check_vma=False,
        )(gate_up)
        return gated, None

    x_spec = P(TP_AXIS, None) if sequence_parallel else P(None, None)

    if use_pallas:

        def _pallas_local(
            xq_local: jax.Array,
            xs_local: jax.Array,
            w_local: jax.Array,
            ws_local: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            if sequence_parallel:
                xq_full = jax.lax.all_gather(
                    xq_local, axis_name=TP_AXIS, axis=0, tiled=True
                )
                xs_full = jax.lax.all_gather(
                    xs_local, axis_name=TP_AXIS, axis=0, tiled=True
                )
            else:
                xq_full, xs_full = xq_local, xs_local
            gated_local, local_abs_max = pallas_fp8_gate_up_swiglu(
                xq_full,
                xs_full,
                w_local,
                ws_local,
                out_dtype=out_dtype,
                interpret=interpret,
            )
            synced_abs_max = jax.lax.pmax(
                jnp.max(local_abs_max, axis=0), axis_name=TP_AXIS
            )
            return gated_local, synced_abs_max

        return jax.shard_map(
            _pallas_local,
            mesh=mesh,
            in_specs=(x_spec, x_spec, P(TP_AXIS, None), P(TP_AXIS)),
            out_specs=(P(None, TP_AXIS), P()),
            check_vma=False,
        )(x_q, x_scale, w_gate_up, w_scale)

    def _xla_local(
        xq_local: jax.Array,
        xs_local: jax.Array,
        w_local: jax.Array,
        ws_local: jax.Array,
    ) -> jax.Array:
        if sequence_parallel:
            xq_full = jax.lax.all_gather(
                xq_local, axis_name=TP_AXIS, axis=0, tiled=True
            )
            xs_full = jax.lax.all_gather(
                xs_local, axis_name=TP_AXIS, axis=0, tiled=True
            )
        else:
            xq_full, xs_full = xq_local, xs_local
        return xla_fp8_gate_up_swiglu(
            xq_full, xs_full, w_local, ws_local, out_dtype=out_dtype
        )

    gated = jax.shard_map(
        _xla_local,
        mesh=mesh,
        in_specs=(x_spec, x_spec, P(TP_AXIS, None), P(TP_AXIS)),
        out_specs=P(None, TP_AXIS),
        check_vma=False,
    )(x_q, x_scale, w_gate_up, w_scale)
    return gated, None


def sharded_fp8_matmul_reduce_scatter(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    mesh: Mesh,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    backend: MatmulBackend = "auto",
    interpret: bool = False,
) -> jax.Array:
    """Run row-parallel (``P(None, "tp")``) ``fp8_matmul`` and ``psum_scatter`` to ``P("tp", None)``."""
    if backend == "gspmd_xla":
        # GSPMD native dot_general reduces partial sums in preferred_element_type (float32)
        # with is_spmd_generated="true" so post_spmd_conservative can overlap the collective.
        del reduce_dtype
        bound = _fp8_bound(w.dtype)
        if x_abs_max is not None:
            x_scale_full = safe_fp8_scale(x_abs_max.reshape(-1, 1), bound)

            def _quant_from_scale(
                x_local: jax.Array, s_full: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                xq = jnp.clip(
                    x_local.astype(jnp.float32) / s_full, -bound, bound
                ).astype(w.dtype)
                tp = jax.lax.psum(1, axis_name=TP_AXIS)
                idx = jax.lax.axis_index(TP_AXIS)
                chunk = s_full.shape[0] // tp
                s_shard = jax.lax.dynamic_slice_in_dim(
                    s_full, idx * chunk, chunk, axis=0
                )
                return xq, s_shard

            x_q, x_scale_shard = jax.shard_map(
                _quant_from_scale,
                mesh=mesh,
                in_specs=(P(None, TP_AXIS), P()),
                out_specs=(P(None, TP_AXIS), P(TP_AXIS, None)),
                check_vma=False,
            )(x, x_scale_full)
        else:

            def _quant_with_pmax(
                x_local: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                xq, s_full = quantize_fp8_act(x_local, dtype=w.dtype, tp_axis=TP_AXIS)
                tp = jax.lax.psum(1, axis_name=TP_AXIS)
                idx = jax.lax.axis_index(TP_AXIS)
                chunk = s_full.shape[0] // tp
                s_shard = jax.lax.dynamic_slice_in_dim(
                    s_full, idx * chunk, chunk, axis=0
                )
                return xq, s_shard

            x_q, x_scale_shard = jax.shard_map(
                _quant_with_pmax,
                mesh=mesh,
                in_specs=P(None, TP_AXIS),
                out_specs=(P(None, TP_AXIS), P(TP_AXIS, None)),
                check_vma=False,
            )(x)

        dot = jax.lax.dot_general(
            x_q.reshape(-1, x_q.shape[-1]),
            w,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        dot_shard = jax.lax.with_sharding_constraint(
            dot, NamedSharding(mesh, P(TP_AXIS, None))
        )
        scaled = (
            dot_shard
            * x_scale_shard.astype(jnp.float32)
            * w_scale.astype(jnp.float32)[None, :]
        ).astype(out_dtype)
        return jax.lax.with_sharding_constraint(
            scaled.reshape(*x.shape[:-1], w.shape[0]),
            NamedSharding(mesh, P(TP_AXIS, None)),
        )

    if x_abs_max is not None:

        def _local_pre_absmax(
            x_shard: jax.Array,
            xa_rep: jax.Array,
            w_shard: jax.Array,
            s_rep: jax.Array,
        ) -> jax.Array:
            return fp8_matmul(
                x_shard,
                w_shard,
                s_rep,
                x_abs_max=xa_rep,
                out_dtype=out_dtype,
                reduce_dtype=reduce_dtype,
                tp_axis=None,
                scatter_axis=TP_AXIS,
                backend=backend,
                interpret=interpret,
            )

        return jax.shard_map(
            _local_pre_absmax,
            mesh=mesh,
            in_specs=(P(None, TP_AXIS), P(), P(None, TP_AXIS), P()),
            out_specs=P(TP_AXIS, None),
            check_vma=False,
        )(x, x_abs_max, w, w_scale)

    def _local(
        x_shard: jax.Array,
        w_shard: jax.Array,
        s_rep: jax.Array,
    ) -> jax.Array:
        return fp8_matmul(
            x_shard,
            w_shard,
            s_rep,
            out_dtype=out_dtype,
            reduce_dtype=reduce_dtype,
            tp_axis=TP_AXIS,
            scatter_axis=TP_AXIS,
            backend=backend,
            interpret=interpret,
        )

    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(None, TP_AXIS), P(None, TP_AXIS), P()),
        out_specs=P(TP_AXIS, None),
        check_vma=False,
    )(x, w, w_scale)


__all__ = [
    "Fp8MatmulTileSizes",
    "Fp8Weight",
    "MatmulBackend",
    "fp8_matmul",
    "fp8_matmul_prequantized",
    "get_tuned_block_sizes",
    "pallas_fp8_gate_up_swiglu",
    "pallas_fp8_matmul",
    "quantize_fp8_act",
    "quantize_fp8_weight",
    "safe_fp8_scale",
    "sharded_fp8_gate_up_swiglu",
    "sharded_fp8_matmul",
    "sharded_fp8_matmul_all_gather",
    "sharded_fp8_matmul_reduce_scatter",
    "xla_fp8_gate_up_swiglu",
    "xla_fp8_matmul",
]
