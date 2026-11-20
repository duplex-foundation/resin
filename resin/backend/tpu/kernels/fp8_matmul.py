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
_MIN_VMEM_LIMIT_BYTES = 4 * 1024 * 1024
_DEFAULT_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
_LARGE_PREFILL_SLACK_BYTES = 4 * 1024 * 1024
# Padded token count at which the T>=2048 prefill tuning applies. Below it,
# decode and small-chunk behavior is unchanged.
_LARGE_PREFILL_MIN_TOKENS = 2048
_DEVICE_VMEM_LIMIT = {6: 96 * 1024 * 1024, 7: _DEFAULT_VMEM_LIMIT_BYTES}
_SAVE_X_Q_MAX_BYTES = 16 * 1024 * 1024
_SAVE_X_Q_SMALL_MAX_BYTES = 4 * 1024 * 1024
_QUANT_MAX_K_TILES = 20


def _device_vmem_limit_bytes(tpu_version: int) -> int:
    return _DEVICE_VMEM_LIMIT.get(tpu_version, _DEFAULT_VMEM_LIMIT_BYTES)


def _row_scale_vmem_bytes(batch_block: int) -> int:
    return 8 * align_to(batch_block, 128) * 4


def _clamp_vmem_limit_bytes(raw_bytes: int, tpu_version: int) -> int:
    upper = _device_vmem_limit_bytes(tpu_version)
    return min(max(raw_bytes, _MIN_VMEM_LIMIT_BYTES), upper)


MatmulBackend = Literal["auto", "pallas", "xla"]


class Fp8Weight(NamedTuple):
    """Pre-quantized ``[N, K]`` FP8 weight and its ``f32[N]`` per-channel scale."""

    qvalue: jax.Array
    scale: jax.Array


class Fp8MatmulTileSizes(NamedTuple):
    """Tile sizes and weight pipeline buffer count for Pallas W8A8."""

    batch_block_size: int
    out_block_size: int
    in_block_size: int
    w_buffer_count: int = 2


def _build_tpu7x_tuned_table() -> dict[tuple[int, int, int], Fp8MatmulTileSizes]:
    table: dict[tuple[int, int, int], Fp8MatmulTileSizes] = {
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
        (2048, 5120, 4096): Fp8MatmulTileSizes(1024, 1280, 4096),
        (2048, 5120, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
        (2048, 5120, 8192): Fp8MatmulTileSizes(1024, 2560, 2048),
        (2048, 5120, 12800): Fp8MatmulTileSizes(1024, 2560, 2560),
        (2048, 5120, 25600): Fp8MatmulTileSizes(1024, 2560, 2560),
        (2048, 12800, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
        (2048, 25600, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
        (4096, 4096, 5120): Fp8MatmulTileSizes(256, 4096, 2560),
        (4096, 512, 5120): Fp8MatmulTileSizes(512, 512, 2560),
        (4096, 5120, 4096): Fp8MatmulTileSizes(1024, 1280, 4096),
        (4096, 5120, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
        (4096, 5120, 8192): Fp8MatmulTileSizes(1024, 2560, 2048),
        (4096, 5120, 12800): Fp8MatmulTileSizes(1024, 2560, 2560),
        (4096, 5120, 25600): Fp8MatmulTileSizes(1024, 2560, 2560),
        (4096, 12800, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
        (4096, 25600, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
        (8192, 5120, 4096): Fp8MatmulTileSizes(1024, 1280, 4096),
        (8192, 5120, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
        (8192, 5120, 8192): Fp8MatmulTileSizes(1024, 2560, 2048),
        (8192, 5120, 12800): Fp8MatmulTileSizes(1024, 2560, 2560),
        (8192, 5120, 25600): Fp8MatmulTileSizes(1024, 2560, 2560),
        (8192, 12800, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
        (8192, 25600, 5120): Fp8MatmulTileSizes(1024, 1280, 5120),
    }
    # Qwen3-32B tp=2 decode buckets (bs <= 320, pre-quantized FP8 activation):
    # batch_block == bs keeps n_batch == 1 (zero 128-lane scale pad) and
    # in_block == K_local keeps n_in == 1 (zero acc_scratch).
    for bs in (8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320):
        table[(bs, 4096, 5120)] = Fp8MatmulTileSizes(bs, 1024, 5120)
        table[(bs, 512, 5120)] = Fp8MatmulTileSizes(bs, 512, 5120)
        table[(bs, 5120, 4096)] = Fp8MatmulTileSizes(bs, 1280, 4096)
        table[(bs, 5120, 5120)] = Fp8MatmulTileSizes(bs, 1280, 5120)
        table[(bs, 5120, 12800)] = Fp8MatmulTileSizes(bs, 512, 12800, 3)
        table[(bs, 12800, 5120)] = Fp8MatmulTileSizes(bs, 1280, 5120)
        table[(bs, 25600, 5120)] = Fp8MatmulTileSizes(bs, 1280, 5120)
    # Qwen3-32B tp=1 shapes remain at their base entries:
    table.update(
        {
            (8, 5120, 8192): Fp8MatmulTileSizes(8, 1280, 8192),
            (8, 5120, 25600): Fp8MatmulTileSizes(8, 640, 25600),
            (16, 5120, 8192): Fp8MatmulTileSizes(16, 1280, 8192),
            (16, 5120, 25600): Fp8MatmulTileSizes(16, 640, 25600),
            (32, 5120, 8192): Fp8MatmulTileSizes(32, 2560, 4096),
            (32, 5120, 25600): Fp8MatmulTileSizes(32, 640, 25600),
            (64, 5120, 8192): Fp8MatmulTileSizes(64, 1280, 8192),
            (64, 5120, 25600): Fp8MatmulTileSizes(64, 640, 25600),
            (128, 5120, 8192): Fp8MatmulTileSizes(128, 5120, 2048),
            (128, 5120, 25600): Fp8MatmulTileSizes(128, 512, 25600),
            (256, 5120, 8192): Fp8MatmulTileSizes(256, 1280, 8192),
            (256, 5120, 25600): Fp8MatmulTileSizes(256, 5120, 2560),
        }
    )
    for bs in (96, 160, 192, 224, 288, 320):
        table[(bs, 5120, 8192)] = Fp8MatmulTileSizes(bs, 2560, 4096)
        table[(bs, 5120, 25600)] = Fp8MatmulTileSizes(bs, 2560, 2560)
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


def _build_tpu7x_quant_tuned_table() -> dict[tuple[int, int, int], Fp8MatmulTileSizes]:
    """Overrides for ``quantize_activation=True`` (bf16 input) so pipelined VMEM <= 48 MiB."""
    table: dict[tuple[int, int, int], Fp8MatmulTileSizes] = {}
    for bs in (2048, 4096, 8192):
        for n_out in (5120, 12800, 25600):
            table[(bs, n_out, 5120)] = Fp8MatmulTileSizes(512, 1280, 5120)
        table[(bs, 51200, 5120)] = Fp8MatmulTileSizes(256, 2560, 5120)
    return table


_TPU7X_FP8_QUANT_TUNED_BLOCKS = _build_tpu7x_quant_tuned_table()

_TPU7X_GATE_UP_TUNED_BLOCKS: dict[tuple[int, int, int], Fp8MatmulTileSizes] = {
    (bs, 12800, 5120): Fp8MatmulTileSizes(1024, 1280, 5120) for bs in (2048, 4096, 8192)
}


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
    sync_tp_scale: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Dynamically quantize activations per token along the trailing ``K`` axis."""
    return _quantize_symmetric_fp8(
        x, axis=-1, dtype=dtype, tp_axis=tp_axis if sync_tp_scale else None
    )


def _quantize_fp8_with_absmax_pallas_kernel(
    x_ref,
    scale_row_ref,
    x_q_ref,
    *,
    bound: float,
) -> None:
    scale_col = jnp.transpose(scale_row_ref[...])
    x_q_ref[...] = jnp.clip(
        x_ref[...].astype(jnp.float32) / scale_col, -bound, bound
    ).astype(x_q_ref.dtype)


def quantize_fp8_with_absmax(
    x: jax.Array,
    abs_max: jax.Array,
    dtype: Any = jnp.float8_e4m3fn,
    *,
    transposed_scale: bool = False,
    use_pallas: bool | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Quantize ``x: [..., K]`` using pre-computed per-token ``abs_max`` via K-tiled VMEM Pallas."""
    target = jnp.dtype(dtype)
    bound = _fp8_bound(target)
    leading = x.shape[:-1]
    k_dim = x.shape[-1]
    x_2d = x if x.ndim == 2 else x.reshape(-1, k_dim)
    orig_batch = x_2d.shape[0]
    if abs_max.size != orig_batch:
        raise ValueError(
            f"Expected abs_max with {orig_batch} elements, got {abs_max.shape=}"
        )

    tpu_version = get_tpu_version()
    batch_block = align_to(max(8, orig_batch), 8) if orig_batch > 0 else 8
    can_pallas = (
        0 < orig_batch <= 320
        and k_dim >= 128
        and k_dim % 128 == 0
        and (interpret or tpu_version >= 7)
    )
    if use_pallas is True and not can_pallas:
        raise ValueError(
            f"Pallas quantize_fp8_with_absmax unsupported for shape={x.shape}, {tpu_version=}"
        )
    run_pallas = can_pallas if use_pallas is None else use_pallas
    if not run_pallas:
        scale_col = safe_fp8_scale(abs_max.reshape(orig_batch, 1), bound)
        q_2d = jnp.clip(x_2d.astype(jnp.float32) / scale_col, -bound, bound).astype(
            target
        )
        scale_t = jnp.transpose(scale_col)
    else:
        if abs_max.shape == (1, orig_batch) and abs_max.dtype == jnp.float32:
            absmax_row = abs_max
        else:
            absmax_row = abs_max.reshape(1, orig_batch).astype(jnp.float32)
        scale_t = safe_fp8_scale(absmax_row, bound)
        scale_in = scale_t
        if orig_batch < batch_block:
            pad_b = batch_block - orig_batch
            x_2d = jnp.pad(x_2d, ((0, pad_b), (0, 0)))
            scale_in = jnp.pad(scale_t, ((0, 0), (0, pad_b)), constant_values=1.0)

        k_tiles = k_dim // 128
        k_block = 128 * next(
            d
            for d in range(min(k_tiles, _QUANT_MAX_K_TILES), 0, -1)
            if k_tiles % d == 0
        )
        n_k = k_dim // k_block
        if tpu_version >= 7 and not interpret:
            buf_mult = 2 if n_k > 1 else 1
            tile_vmem = (
                buf_mult
                * batch_block
                * k_block
                * (x_2d.dtype.itemsize + target.itemsize)
                + _row_scale_vmem_bytes(batch_block)
                + batch_block * k_block * 4
            )
            vmem_limit_bytes: int | None = _clamp_vmem_limit_bytes(
                tile_vmem, tpu_version
            )
        else:
            vmem_limit_bytes = None
        k_idx = (lambda _k: 0) if n_k == 1 else (lambda k: k)
        q_2d = pl.pallas_call(
            functools.partial(_quantize_fp8_with_absmax_pallas_kernel, bound=bound),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((batch_block, k_block), lambda k: (0, k_idx(k))),
                    pl.BlockSpec((1, batch_block), lambda _k: (0, 0)),
                ],
                out_specs=pl.BlockSpec((batch_block, k_block), lambda k: (0, k_idx(k))),
                grid=(n_k,),
            ),
            out_shape=jax.ShapeDtypeStruct((batch_block, k_dim), target),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            interpret=pltpu.InterpretParams() if interpret else False,
            name=f"quantize_fp8_with_absmax_{batch_block}_{k_block}",
        )(x_2d, scale_in)
        if orig_batch < batch_block:
            q_2d = q_2d[:orig_batch]

    q_out = q_2d if x.ndim == 2 else q_2d.reshape(*leading, k_dim)
    if transposed_scale:
        return q_out, scale_t
    return q_out, jnp.transpose(scale_t).reshape(*leading, 1)


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
    quantize_activation: bool = False,
) -> Fp8MatmulTileSizes:
    """Pick Pallas ``(batch_block, out_block, in_block, w_buffer_count)`` obeying TPU ``(8, 128)`` rules."""
    version = get_tpu_version() if tpu_version is None else tpu_version
    if version >= 7 and (tuned := _TPU7X_FP8_TUNED_BLOCKS.get((n_batch, n_out, n_in))):
        if not quantize_activation:
            return tuned
        if n_batch >= _LARGE_PREFILL_MIN_TOKENS:
            return _TPU7X_FP8_QUANT_TUNED_BLOCKS.get((n_batch, n_out, n_in), tuned)
        if n_batch <= 320 and tuned.w_buffer_count > 2:
            return tuned._replace(w_buffer_count=2)
        if (
            tuned.batch_block_size == 512
            and tuned.out_block_size == 2560
            and tuned.in_block_size == 5120
        ):
            return Fp8MatmulTileSizes(256, 2560, 5120)
        return tuned

    if n_batch < 128:
        batch_block = align_to(max(8, n_batch), 8)
    else:
        batch_block = min(align_to(n_batch, 128), 512)
    out_block = min(align_to(max(128, n_out), 128), 2048)
    in_block = min(align_to(max(128, n_in), 128), 2048)
    return Fp8MatmulTileSizes(batch_block, out_block, in_block)


def _get_tuned_gate_up_block_sizes(
    n_batch: int,
    n_ffn: int,
    n_in: int,
    *,
    tpu_version: int | None = None,
) -> Fp8MatmulTileSizes:
    """Pick Pallas ``(batch_block, out_block, in_block)`` for fused gate+up SwiGLU."""
    version = get_tpu_version() if tpu_version is None else tpu_version
    if version >= 7 and (
        tuned := _TPU7X_GATE_UP_TUNED_BLOCKS.get((n_batch, n_ffn, n_in))
    ):
        return tuned
    batch_block = (
        align_to(max(8, n_batch), 8)
        if n_batch <= 320
        else min(align_to(n_batch, 128), 512)
    )
    out_block = 1280 if n_ffn % 1280 == 0 else min(align_to(max(128, n_ffn), 128), 1280)
    in_block = (
        n_in
        if (n_in <= 5120 and n_in % 128 == 0)
        else min(align_to(max(128, n_in), 128), 2560)
    )
    return Fp8MatmulTileSizes(batch_block, out_block, in_block)


def _raw_vmem_estimate_bytes(
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
    w_buffer_count: int = 2,
    include_headroom: bool = True,
) -> int:
    x_bytes = batch_block_size * in_block_size * jnp.dtype(x_dtype).itemsize
    absmax_bytes = batch_block_size * 4
    w_bytes = out_block_size * in_block_size * jnp.dtype(w_q_dtype).itemsize
    w_scale_bytes = out_block_size * 4
    out_bytes = batch_block_size * out_block_size * jnp.dtype(out_dtype).itemsize
    w_extra_buffers = max(0, w_buffer_count - 1)
    base_buffers = 2 if include_headroom else 1

    in_out = base_buffers * (
        x_bytes + absmax_bytes + w_bytes + w_scale_bytes + out_bytes
    )
    if n_batch > 1 or n_in > 1:
        in_out += x_bytes
    if n_batch > 1:
        in_out += absmax_bytes
    if n_out > 1 or n_in > 1:
        in_out += w_extra_buffers * w_bytes
    if n_out > 1:
        in_out += w_extra_buffers * w_scale_bytes
    if n_batch > 1 or n_out > 1:
        in_out += out_bytes

    scratch = 0
    if save_acc:
        scratch += batch_block_size * out_block_size * 4
    if save_x_q:
        scratch += (
            n_in * batch_block_size * in_block_size * jnp.dtype(x_q_dtype).itemsize
        )
    return in_out + (2 if include_headroom else 1) * scratch


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
    w_buffer_count: int = 2,
) -> int:
    large_prefill = n_batch * batch_block_size >= _LARGE_PREFILL_MIN_TOKENS
    raw = _raw_vmem_estimate_bytes(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        x_dtype=x_dtype,
        x_q_dtype=x_q_dtype,
        w_q_dtype=w_q_dtype,
        out_dtype=out_dtype,
        save_acc=save_acc,
        save_x_q=save_x_q,
        w_buffer_count=w_buffer_count,
        include_headroom=not large_prefill,
    )
    if large_prefill:
        raw += _LARGE_PREFILL_SLACK_BYTES
    return _clamp_vmem_limit_bytes(raw, tpu_version)


def _should_save_x_q(
    *,
    quantize_activation: bool,
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
    tpu_version: int,
    w_buffer_count: int = 2,
) -> bool:
    if not quantize_activation or n_out <= 1:
        return False
    q_dtype = jnp.dtype(x_q_dtype)
    x_q_scratch_bytes = n_in * batch_block_size * in_block_size * q_dtype.itemsize
    if n_batch * batch_block_size < _LARGE_PREFILL_MIN_TOKENS:
        return x_q_scratch_bytes <= _SAVE_X_Q_SMALL_MAX_BYTES
    if x_q_scratch_bytes > _SAVE_X_Q_MAX_BYTES:
        return False
    vmem_upper = _device_vmem_limit_bytes(tpu_version)
    raw_bytes = _raw_vmem_estimate_bytes(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=batch_block_size,
        out_block_size=out_block_size,
        in_block_size=in_block_size,
        x_dtype=x_dtype,
        x_q_dtype=q_dtype,
        w_q_dtype=w_q_dtype,
        out_dtype=out_dtype,
        save_acc=save_acc,
        save_x_q=True,
        w_buffer_count=w_buffer_count,
        include_headroom=False,
    )
    return raw_bytes <= vmem_upper


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
    scaled = dot * x_scale_2d * w_scale.astype(jnp.float32)[None, :]
    if eff_reduce != jnp.float32:
        scaled = scaled.astype(eff_reduce)
    if scatter_axis is not None:
        scaled = jax.lax.psum_scatter(
            scaled, axis_name=scatter_axis, scatter_dimension=0, tiled=True
        )
        return scaled.astype(out_dtype).reshape(-1, *leading[1:], w.shape[0])
    if tp_axis is not None:
        scaled = jax.lax.psum(scaled, axis_name=tp_axis)
    return scaled.astype(out_dtype).reshape(*leading, w.shape[0])


def _validate_x_scale_args(
    quantize_activation: bool,
    x_scale: jax.Array | None,
    x_abs_max: jax.Array | None,
) -> None:
    if x_scale is not None and x_abs_max is not None:
        raise ValueError("Pass at most one of x_scale or x_abs_max")
    if quantize_activation and x_scale is not None:
        raise ValueError("Unquantized activation x cannot pass pre-quantized x_scale")
    if not quantize_activation and x_abs_max is not None:
        raise ValueError(
            "Pre-quantized FP8 activation x requires x_scale, not x_abs_max"
        )


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
    sync_tp_scale: bool = True,
) -> jax.Array:
    """Pure-JAX Qwix-equivalent FP8 W8A8 matmul using XLA native MXU ``dot_general``."""
    _validate_x_scale_args(x.dtype != w.dtype, x_scale, x_abs_max)
    if x.dtype == w.dtype:
        eff_x_scale = (
            jnp.ones((*x.shape[:-1], 1), dtype=jnp.float32)
            if x_scale is None
            else x_scale
        )
        x_q = x
    elif x_abs_max is not None:
        x_q, eff_x_scale = quantize_fp8_with_absmax(
            x, x_abs_max, w.dtype, use_pallas=False
        )
    else:
        x_q, eff_x_scale = quantize_fp8_act(
            x, dtype=w.dtype, tp_axis=tp_axis, sync_tp_scale=sync_tp_scale
        )
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
    sync_tp_scale: bool = True,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused Pallas W8A8 matmul supporting both BF16 and pre-quantized FP8 inputs."""
    _validate_matmul_shapes(x.shape[-1], w, w_scale)

    leading = x.shape[:-1]
    orig_k = x.shape[-1]
    orig_n = w.shape[0]
    x_2d = x if x.ndim == 2 else x.reshape(-1, orig_k)
    orig_batch = x_2d.shape[0]
    x_q_dtype = jnp.dtype(w.dtype)
    quantize_activation = x_q_dtype != x_2d.dtype
    _validate_x_scale_args(quantize_activation, x_scale, x_abs_max)

    if quantize_activation:
        if x_abs_max is not None:
            if x_abs_max.shape == (1, orig_batch) and x_abs_max.dtype == jnp.float32:
                x_scale_or_absmax = x_abs_max
            else:
                x_scale_or_absmax = x_abs_max.reshape(1, orig_batch).astype(jnp.float32)
        else:
            absmax_1d = jnp.max(jnp.abs(x_2d.astype(jnp.float32)), axis=-1)
            if tp_axis is not None and sync_tp_scale:
                absmax_1d = jax.lax.pmax(absmax_1d, axis_name=tp_axis)
            x_scale_or_absmax = jnp.expand_dims(absmax_1d, axis=0)
        has_x_scale = True
    elif x_scale is not None:
        if x_scale.shape == (1, orig_batch) and x_scale.dtype == jnp.float32:
            x_scale_or_absmax = x_scale
        else:
            x_scale_or_absmax = x_scale.reshape(1, orig_batch).astype(jnp.float32)
        has_x_scale = True
    else:
        x_scale_or_absmax = jnp.ones((1, orig_batch), dtype=jnp.float32)
        has_x_scale = False

    eff_reduce = jnp.dtype(out_dtype if reduce_dtype is None else reduce_dtype)
    kernel_out_dtype = (
        eff_reduce
        if (tp_axis is not None or scatter_axis is not None)
        else jnp.dtype(out_dtype)
    )
    tpu_version = get_tpu_version()
    resolved_tiles = (
        get_tuned_block_sizes(
            orig_batch,
            orig_n,
            orig_k,
            tpu_version=tpu_version,
            quantize_activation=quantize_activation,
        )
        if tile_sizes is None
        else tile_sizes
    )
    batch_block, out_block, in_block, w_buffer_count = resolved_tiles
    if w_buffer_count < 1:
        raise ValueError(f"Expected w_buffer_count >= 1, got {w_buffer_count}")

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
    save_x_q = _should_save_x_q(
        quantize_activation=quantize_activation,
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
        tpu_version=tpu_version,
        w_buffer_count=w_buffer_count,
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
        w_buffer_count=w_buffer_count,
    )

    b_idx = (lambda _b: 0) if n_batch == 1 else (lambda b: b)
    o_idx = (lambda _o: 0) if n_out == 1 else (lambda o: o)
    i_idx = (lambda _i: 0) if n_in == 1 else (lambda i: i)
    # Pin o > 0 to the last tile loaded during o == 0 (n_in - 1) so Pallas skips redundant x DMAs.
    x_in_idx = (
        (lambda b, o, i: (b_idx(b), jnp.where(o == 0, i, n_in - 1)))
        if (save_x_q and n_in > 1 and padded_batch >= _LARGE_PREFILL_MIN_TOKENS)
        else (lambda b, _o, i: (b_idx(b), i_idx(i)))
    )
    x_q_scratch_shape = (
        (batch_block, in_block) if n_in == 1 else (n_in, batch_block, in_block)
    )
    w_pipe = (
        pl.Buffered(buffer_count=w_buffer_count)
        if w_buffer_count != 2 and (n_out > 1 or n_in > 1)
        else None
    )
    in_specs = [
        pl.BlockSpec((batch_block, in_block), x_in_idx),
        pl.BlockSpec(
            (out_block, in_block),
            lambda _b, o, i: (o_idx(o), i_idx(i)),
            pipeline_mode=w_pipe,
        ),
        pl.BlockSpec(
            (1, out_block),
            lambda _b, o, _i: (0, o_idx(o)),
            pipeline_mode=w_pipe if n_out > 1 else None,
        ),
        pl.BlockSpec((1, batch_block), lambda b, _o, _i: (0, b_idx(b))),
    ]
    out_specs = pl.BlockSpec(
        (batch_block, out_block), lambda b, o, _i: (b_idx(b), o_idx(o))
    )
    scratch_shapes = [
        pltpu.VMEM((batch_block, out_block), jnp.float32) if save_acc else None,
        pltpu.VMEM(x_q_scratch_shape, x_q_dtype) if save_x_q else None,
    ]
    grid = (n_batch, n_out, n_in)
    dim_semantics = ("parallel", "arbitrary", "arbitrary")
    kernel_body = functools.partial(
        _fp8_matmul_pallas_kernel,
        x_q_dtype=x_q_dtype,
        has_x_scale=has_x_scale,
        save_acc=save_acc,
        save_x_q=save_x_q,
    )
    kernel_name = (
        f"fp8_w8a8_matmul_{batch_block}_{out_block}_{in_block}"
        if w_pipe is None
        else f"fp8_w8a8_matmul_{batch_block}_{out_block}_{in_block}_b{w_buffer_count}"
    )

    if w_pipe is not None and tpu_version >= (7 if interpret else 6):

        def _pipelined_kernel(
            x_hbm, w_hbm, ws_hbm, xs_hbm, out_hbm, acc_scratch, x_q_scratch
        ) -> None:
            pltpu.emit_pipeline(
                kernel_body,
                grid=grid,
                in_specs=in_specs,
                out_specs=out_specs,
                dimension_semantics=dim_semantics,
            )(
                x_hbm,
                w_hbm,
                ws_hbm,
                xs_hbm,
                out_hbm,
                scratches=(acc_scratch, x_q_scratch),
            )

        kernel = pl.pallas_call(
            _pipelined_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=pl.ANY),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    pl.BlockSpec(memory_space=pl.ANY),
                ],
                out_specs=pl.BlockSpec(memory_space=pl.ANY),
                scratch_shapes=scratch_shapes,
            ),
            out_shape=jax.ShapeDtypeStruct((padded_batch, padded_n), kernel_out_dtype),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
            interpret=pltpu.InterpretParams() if interpret else False,
            name=kernel_name,
        )
    else:
        kernel = pl.pallas_call(
            kernel_body,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch_shapes,
                grid=grid,
            ),
            out_shape=jax.ShapeDtypeStruct((padded_batch, padded_n), kernel_out_dtype),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=dim_semantics,
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            interpret=pltpu.InterpretParams() if interpret else False,
            name=kernel_name,
        )

    if tpu_version >= 6 and not interpret:
        if scatter_axis is not None and padded_batch >= _LARGE_PREFILL_MIN_TOKENS:
            x_2d = pltpu.with_memory_space_constraint(x_2d, pltpu.HBM)
        w = pltpu.with_memory_space_constraint(w, pltpu.HBM)
        w_scale_2d = pltpu.with_memory_space_constraint(w_scale_2d, pltpu.HBM)

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
    x_2d = x_q if x_q.ndim == 2 else x_q.reshape(-1, orig_k)
    orig_batch = x_2d.shape[0]
    if x_scale.shape == (1, orig_batch) and x_scale.dtype == jnp.float32:
        x_scale_row = x_scale
    else:
        x_scale_row = x_scale.reshape(1, orig_batch).astype(jnp.float32)

    tpu_version = get_tpu_version()
    resolved_tiles = (
        _get_tuned_gate_up_block_sizes(
            orig_batch, orig_ffn, orig_k, tpu_version=tpu_version
        )
        if tile_sizes is None
        else tile_sizes
    )
    batch_block, out_block, in_block, _ = resolved_tiles

    padded_batch = align_to(orig_batch, batch_block)
    padded_ffn = align_to(orig_ffn, out_block)
    padded_k = align_to(orig_k, in_block)

    if orig_batch < padded_batch:
        pad_b = padded_batch - orig_batch
        x_2d = jnp.pad(x_2d, ((0, pad_b), (0, 0)))
        x_scale_row = jnp.pad(x_scale_row, ((0, 0), (0, pad_b)))
    if orig_k < padded_k:
        x_2d = jnp.pad(x_2d, ((0, 0), (0, padded_k - orig_k)))

    if orig_ffn == padded_ffn and orig_k == padded_k:
        w_in = w_gate_up
        ws_2d = jnp.expand_dims(w_scale.astype(jnp.float32), axis=0)
    else:
        pad_n = padded_ffn - orig_ffn
        pad_k = padded_k - orig_k
        w_gate = jnp.pad(w_gate_up[:orig_ffn], ((0, pad_n), (0, pad_k)))
        w_up = jnp.pad(w_gate_up[orig_ffn:], ((0, pad_n), (0, pad_k)))
        w_in = jnp.concatenate([w_gate, w_up], axis=0)
        ws_gate = jnp.pad(w_scale[:orig_ffn].astype(jnp.float32), (0, pad_n))
        ws_up = jnp.pad(w_scale[orig_ffn:].astype(jnp.float32), (0, pad_n))
        ws_2d = jnp.expand_dims(jnp.concatenate([ws_gate, ws_up], axis=0), axis=0)

    n_batch = padded_batch // batch_block
    n_out = padded_ffn // out_block
    n_in = padded_k // in_block
    save_acc = n_in > 1
    target_out_dtype = jnp.dtype(out_dtype)

    if tpu_version >= 6 and not interpret:
        if padded_batch >= _LARGE_PREFILL_MIN_TOKENS:
            vmem_limit_bytes: int | None = _device_vmem_limit_bytes(tpu_version)
        else:
            row_scale_bytes = _row_scale_vmem_bytes(batch_block)
            # w_in packs [gate; up] along axis 0, so its block specs are non-trivial (2x buffered).
            gate_up_vmem = (
                (2 if (n_batch > 1 or save_acc) else 1)
                * batch_block
                * in_block
                * x_2d.dtype.itemsize
                + 4 * out_block * (in_block * w_in.dtype.itemsize + 8 * 4)
                + (2 if n_batch > 1 else 1) * row_scale_bytes
                + (2 if (n_batch > 1 or n_out > 1) else 1)
                * (
                    batch_block * out_block * target_out_dtype.itemsize
                    + row_scale_bytes
                )
                + (6 if save_acc else 4) * batch_block * out_block * 4
            )
            vmem_limit_bytes = _clamp_vmem_limit_bytes(gate_up_vmem, tpu_version)
    else:
        vmem_limit_bytes = None

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
                    (out_block, in_block), lambda _b, o, i: (n_out + o_idx(o), i_idx(i))
                ),
                pl.BlockSpec((1, out_block), lambda _b, o, _i: (0, o_idx(o))),
                pl.BlockSpec((1, out_block), lambda _b, o, _i: (0, n_out + o_idx(o))),
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

    if tpu_version >= 6 and not interpret:
        w_in = pltpu.with_memory_space_constraint(w_in, pltpu.HBM)
        ws_2d = pltpu.with_memory_space_constraint(ws_2d, pltpu.HBM)

    gated_padded, absmax_3d = kernel(
        x_2d,
        w_in,
        w_in,
        ws_2d,
        ws_2d,
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
    sync_tp_scale: bool = True,
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
            sync_tp_scale=sync_tp_scale,
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
        sync_tp_scale=sync_tp_scale,
    )


def _scale_shard_spec(
    scale: jax.Array,
    *,
    mesh: Mesh,
    in_axis: str | None,
    sync_tp_scale: bool,
    n_rows: int,
) -> P:
    if in_axis is not None:
        tp = int(mesh.shape[in_axis])
        if not sync_tp_scale and scale.shape == (tp, n_rows):
            return P(in_axis, None)
        if sync_tp_scale and tp > 1 and scale.shape == (tp, n_rows):
            raise ValueError(
                f"Per-die scale shape {scale.shape} requires sync_tp_scale=False"
            )
    return P(*((None,) * scale.ndim))


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
    sync_tp_scale: bool = True,
    tile_sizes: Fp8MatmulTileSizes | None = None,
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
    n_rows = x.reshape(-1, x.shape[-1]).shape[0]
    prefix_none = (None,) * (x.ndim - 1)
    x_spec = P(*prefix_none, in_axis)
    scale_spec = P(out_axis)
    out_spec = P(*prefix_none, out_axis)

    def _run_local(
        x_shard: jax.Array,
        w_shard: jax.Array,
        s_shard: jax.Array,
        *,
        xs_shard: jax.Array | None = None,
        xa_shard: jax.Array | None = None,
    ) -> jax.Array:
        return fp8_matmul(
            x_shard,
            w_shard,
            s_shard,
            x_scale=xs_shard,
            x_abs_max=xa_shard,
            out_dtype=out_dtype,
            reduce_dtype=reduce_dtype,
            tp_axis=in_axis,
            sync_tp_scale=sync_tp_scale,
            backend=backend,
            tile_sizes=tile_sizes,
            interpret=interpret,
        )

    if x_scale is not None:
        xs_spec = _scale_shard_spec(
            x_scale,
            mesh=mesh,
            in_axis=in_axis,
            sync_tp_scale=sync_tp_scale,
            n_rows=n_rows,
        )

        def _local_with_scale(
            x_shard: jax.Array,
            xs_shard: jax.Array,
            w_shard: jax.Array,
            s_shard: jax.Array,
        ) -> jax.Array:
            return _run_local(x_shard, w_shard, s_shard, xs_shard=xs_shard)

        return jax.shard_map(
            _local_with_scale,
            mesh=mesh,
            in_specs=(x_spec, xs_spec, spec, scale_spec),
            out_specs=out_spec,
            check_vma=False,
        )(x, x_scale, w, w_scale)

    if x_abs_max is not None:
        xa_spec = _scale_shard_spec(
            x_abs_max,
            mesh=mesh,
            in_axis=in_axis,
            sync_tp_scale=sync_tp_scale,
            n_rows=n_rows,
        )

        def _local_with_absmax(
            x_shard: jax.Array,
            xa_shard: jax.Array,
            w_shard: jax.Array,
            s_shard: jax.Array,
        ) -> jax.Array:
            return _run_local(x_shard, w_shard, s_shard, xa_shard=xa_shard)

        return jax.shard_map(
            _local_with_absmax,
            mesh=mesh,
            in_specs=(x_spec, xa_spec, spec, scale_spec),
            out_specs=out_spec,
            check_vma=False,
        )(x, x_abs_max, w, w_scale)

    def _local(x_shard: jax.Array, w_shard: jax.Array, s_shard: jax.Array) -> jax.Array:
        return _run_local(x_shard, w_shard, s_shard)

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
    sync_tp_scale: bool = True,
    quantize_output: bool = False,
    out_dtype: Any = jnp.bfloat16,
    backend: MatmulBackend = "auto",
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Run column-parallel ``gate_up`` + SwiGLU, returning ``(gated, absmax_or_scale)``."""
    use_pallas = backend == "pallas" or (
        backend == "auto"
        and (interpret or get_tpu_version() >= 7)
        and w_gate_up.shape[1] % 128 == 0
        and (w_gate_up.shape[0] // (2 * int(mesh.shape[TP_AXIS]))) % 128 == 0
    )

    x_spec = P(TP_AXIS, None) if sequence_parallel else P(None, None)
    scale_out_spec = P(None, None) if sync_tp_scale else P(TP_AXIS, None)

    if use_pallas:
        aux_out_spec = (
            scale_out_spec
            if quantize_output
            else (P() if sync_tp_scale else P(TP_AXIS, None))
        )

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
            local_max_row = jnp.max(local_abs_max, axis=0, keepdims=True)
            reduced_row = (
                jax.lax.pmax(local_max_row, axis_name=TP_AXIS)
                if sync_tp_scale
                else local_max_row
            )
            if quantize_output:
                return quantize_fp8_with_absmax(
                    gated_local,
                    reduced_row,
                    dtype=w_local.dtype,
                    transposed_scale=True,
                    interpret=interpret,
                )
            return (
                gated_local,
                jnp.squeeze(reduced_row, axis=0) if sync_tp_scale else reduced_row,
            )

        return jax.shard_map(
            _pallas_local,
            mesh=mesh,
            in_specs=(x_spec, x_spec, P(TP_AXIS, None), P(TP_AXIS)),
            out_specs=(P(None, TP_AXIS), aux_out_spec),
            check_vma=False,
        )(x_q, x_scale, w_gate_up, w_scale)

    def _xla_local(
        xq_local: jax.Array,
        xs_local: jax.Array,
        w_local: jax.Array,
        ws_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array | None]:
        if sequence_parallel:
            xq_full = jax.lax.all_gather(
                xq_local, axis_name=TP_AXIS, axis=0, tiled=True
            )
            xs_full = jax.lax.all_gather(
                xs_local, axis_name=TP_AXIS, axis=0, tiled=True
            )
        else:
            xq_full, xs_full = xq_local, xs_local
        gated_local = xla_fp8_gate_up_swiglu(
            xq_full, xs_full, w_local, ws_local, out_dtype=out_dtype
        )
        if not quantize_output:
            return gated_local, None
        gated_q, scale_col = quantize_fp8_act(
            gated_local,
            dtype=w_local.dtype,
            tp_axis=TP_AXIS,
            sync_tp_scale=sync_tp_scale,
        )
        return gated_q, jnp.transpose(scale_col.reshape(-1, 1))

    return jax.shard_map(
        _xla_local,
        mesh=mesh,
        in_specs=(x_spec, x_spec, P(TP_AXIS, None), P(TP_AXIS)),
        out_specs=(P(None, TP_AXIS), scale_out_spec if quantize_output else None),
        check_vma=False,
    )(x_q, x_scale, w_gate_up, w_scale)


def sharded_fp8_matmul_reduce_scatter(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    mesh: Mesh,
    x_scale: jax.Array | None = None,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    sync_tp_scale: bool = True,
    backend: MatmulBackend = "auto",
    interpret: bool = False,
) -> jax.Array:
    """Run row-parallel (``P(None, "tp")``) ``fp8_matmul`` and ``psum_scatter`` to ``P("tp", None)``."""
    n_rows = x.reshape(-1, x.shape[-1]).shape[0]
    if x_scale is not None:
        xs_spec = _scale_shard_spec(
            x_scale,
            mesh=mesh,
            in_axis=TP_AXIS,
            sync_tp_scale=sync_tp_scale,
            n_rows=n_rows,
        )

        def _local_pre_scale(
            x_shard: jax.Array,
            xs_shard: jax.Array,
            w_shard: jax.Array,
            s_rep: jax.Array,
        ) -> jax.Array:
            return fp8_matmul(
                x_shard,
                w_shard,
                s_rep,
                x_scale=xs_shard,
                out_dtype=out_dtype,
                reduce_dtype=reduce_dtype,
                tp_axis=None,
                scatter_axis=TP_AXIS,
                sync_tp_scale=sync_tp_scale,
                backend=backend,
                interpret=interpret,
            )

        return jax.shard_map(
            _local_pre_scale,
            mesh=mesh,
            in_specs=(P(None, TP_AXIS), xs_spec, P(None, TP_AXIS), P()),
            out_specs=P(TP_AXIS, None),
            check_vma=False,
        )(x, x_scale, w, w_scale)

    if x_abs_max is not None:
        xa_spec = _scale_shard_spec(
            x_abs_max,
            mesh=mesh,
            in_axis=TP_AXIS,
            sync_tp_scale=sync_tp_scale,
            n_rows=n_rows,
        )

        def _local_pre_absmax(
            x_shard: jax.Array,
            xa_shard: jax.Array,
            w_shard: jax.Array,
            s_rep: jax.Array,
        ) -> jax.Array:
            return fp8_matmul(
                x_shard,
                w_shard,
                s_rep,
                x_abs_max=xa_shard,
                out_dtype=out_dtype,
                reduce_dtype=reduce_dtype,
                tp_axis=None,
                scatter_axis=TP_AXIS,
                sync_tp_scale=sync_tp_scale,
                backend=backend,
                interpret=interpret,
            )

        return jax.shard_map(
            _local_pre_absmax,
            mesh=mesh,
            in_specs=(P(None, TP_AXIS), xa_spec, P(None, TP_AXIS), P()),
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
            sync_tp_scale=sync_tp_scale,
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
    "quantize_fp8_with_absmax",
    "safe_fp8_scale",
    "sharded_fp8_gate_up_swiglu",
    "sharded_fp8_matmul",
    "sharded_fp8_matmul_all_gather",
    "sharded_fp8_matmul_reduce_scatter",
    "xla_fp8_gate_up_swiglu",
    "xla_fp8_matmul",
]
