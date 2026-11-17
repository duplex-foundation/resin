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
"""FP8 W8A8 quantization and Pallas/XLA matrix multiplication for TPU.

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

_FP32_TINY_SQRT = float(jnp.sqrt(jnp.finfo(jnp.float32).tiny))
_DEVICE_VMEM_LIMIT = {6: 96 * 1024 * 1024, 7: 48 * 1024 * 1024}


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
        (256, 4096, 5120): Fp8MatmulTileSizes(256, 2048, 2560),
        (256, 512, 5120): Fp8MatmulTileSizes(256, 512, 2560),
        (256, 5120, 4096): Fp8MatmulTileSizes(128, 5120, 4096),
        (256, 5120, 5120): Fp8MatmulTileSizes(256, 2560, 2560),
        (256, 5120, 8192): Fp8MatmulTileSizes(256, 1280, 8192),
        (256, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 512),
        (256, 5120, 25600): Fp8MatmulTileSizes(256, 5120, 2560),
        (256, 12800, 5120): Fp8MatmulTileSizes(256, 2560, 2560),
        (256, 25600, 5120): Fp8MatmulTileSizes(256, 2560, 2560),
        (512, 4096, 5120): Fp8MatmulTileSizes(512, 2048, 1280),
        (512, 512, 5120): Fp8MatmulTileSizes(512, 512, 2560),
        (512, 5120, 4096): Fp8MatmulTileSizes(512, 5120, 1024),
        (512, 5120, 5120): Fp8MatmulTileSizes(512, 5120, 1024),
        (512, 5120, 8192): Fp8MatmulTileSizes(512, 1024, 8192),
        (512, 5120, 12800): Fp8MatmulTileSizes(512, 5120, 512),
        (512, 5120, 25600): Fp8MatmulTileSizes(512, 2560, 2560),
        (512, 12800, 5120): Fp8MatmulTileSizes(512, 2560, 1024),
        (512, 25600, 5120): Fp8MatmulTileSizes(512, 2560, 1024),
        (2048, 4096, 5120): Fp8MatmulTileSizes(256, 4096, 2560),
        (2048, 512, 5120): Fp8MatmulTileSizes(512, 512, 2560),
        (2048, 5120, 4096): Fp8MatmulTileSizes(256, 5120, 4096),
        (2048, 5120, 5120): Fp8MatmulTileSizes(256, 5120, 5120),
        (2048, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 2560),
        (2048, 12800, 5120): Fp8MatmulTileSizes(512, 3200, 1024),
        (2048, 25600, 5120): Fp8MatmulTileSizes(512, 6400, 512),
        (4096, 4096, 5120): Fp8MatmulTileSizes(256, 4096, 2560),
        (4096, 512, 5120): Fp8MatmulTileSizes(512, 512, 2560),
        (4096, 5120, 4096): Fp8MatmulTileSizes(256, 5120, 4096),
        (4096, 5120, 5120): Fp8MatmulTileSizes(256, 5120, 5120),
        (4096, 5120, 8192): Fp8MatmulTileSizes(1024, 2560, 2048),
        (4096, 5120, 12800): Fp8MatmulTileSizes(256, 5120, 2560),
        (4096, 5120, 25600): Fp8MatmulTileSizes(1024, 2560, 2560),
        (4096, 12800, 5120): Fp8MatmulTileSizes(512, 3200, 1024),
        (4096, 25600, 5120): Fp8MatmulTileSizes(512, 6400, 512),
        (8192, 5120, 4096): Fp8MatmulTileSizes(256, 5120, 4096),
        (8192, 5120, 5120): Fp8MatmulTileSizes(256, 5120, 5120),
        (8192, 5120, 12800): Fp8MatmulTileSizes(1024, 5120, 512),
        (8192, 25600, 5120): Fp8MatmulTileSizes(512, 2560, 2560),
    }
    # Qwen3-32B decode compilation buckets [96, 160, 192, 224, 288, 320]:
    # batch_block == bs keeps n_batch == 1 so x_abs_max [1, bs] needs no 128-lane pad.
    for bs in (96, 160, 192, 224, 288, 320):
        table[(bs, 4096, 5120)] = Fp8MatmulTileSizes(bs, 2048, 2560)
        table[(bs, 512, 5120)] = Fp8MatmulTileSizes(bs, 512, 2560)
        table[(bs, 5120, 4096)] = Fp8MatmulTileSizes(bs, 2560, 2048)
        table[(bs, 5120, 5120)] = Fp8MatmulTileSizes(bs, 2560, 2560)
        table[(bs, 5120, 12800)] = Fp8MatmulTileSizes(bs, 2560, 1280)
        table[(bs, 12800, 5120)] = Fp8MatmulTileSizes(bs, 2560, 2560)
        table[(bs, 25600, 5120)] = Fp8MatmulTileSizes(bs, 2560, 2560)
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
        scratch += batch_block_size * in_block_size * jnp.dtype(x_q_dtype).itemsize
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
    x_abs_max_ref,
    out_ref,
    acc_scratch,
    x_q_scratch,
    *,
    x_q_dtype: jnp.dtype,
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
        x_scale = safe_fp8_scale(jnp.transpose(x_abs_max_ref[...]), bound)
        if quantize_activation:
            if do_quant:
                x_q = jnp.clip(
                    x_ref[...].astype(jnp.float32) / x_scale, -bound, bound
                ).astype(x_q_dtype)
                if save_x_q:
                    x_q_scratch[...] = x_q
            else:
                x_q = x_q_scratch[...]
            lhs = x_q
        else:
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
            if quantize_activation:
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
    tp_axis: str | None = None,
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
    if tp_axis is not None:
        dot = jax.lax.psum(dot, axis_name=tp_axis)
    scaled = dot * x_scale_2d * w_scale.astype(jnp.float32)[None, :]
    return scaled.astype(out_dtype).reshape(*leading, w.shape[0])


def xla_fp8_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    out_dtype: Any = jnp.bfloat16,
    tp_axis: str | None = None,
) -> jax.Array:
    """Pure-JAX Qwix-equivalent FP8 W8A8 matmul using XLA native MXU ``dot_general``."""
    x_q, x_scale = quantize_fp8_act(x, dtype=w.dtype, tp_axis=tp_axis)
    return fp8_matmul_prequantized(
        x_q, x_scale, w, w_scale, out_dtype=out_dtype, tp_axis=tp_axis
    )


def pallas_fp8_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    out_dtype: Any = jnp.bfloat16,
    tp_axis: str | None = None,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Fused Pallas W8A8 matmul with VMEM dynamic activation quantization."""
    _validate_matmul_shapes(x.shape[-1], w, w_scale)

    leading = x.shape[:-1]
    orig_k = x.shape[-1]
    orig_n = w.shape[0]
    x_2d = x.reshape(-1, orig_k)
    orig_batch = x_2d.shape[0]
    x_q_dtype = jnp.dtype(w.dtype)

    x_abs_max = jnp.max(jnp.abs(x_2d.astype(jnp.float32)), axis=-1)
    if tp_axis is not None:
        x_abs_max = jax.lax.pmax(x_abs_max, axis_name=tp_axis)
    x_abs_max = jnp.expand_dims(x_abs_max, axis=0)

    tpu_version = get_tpu_version()
    if tile_sizes is None:
        tile_sizes = get_tuned_block_sizes(
            orig_batch, orig_n, orig_k, tpu_version=tpu_version
        )
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
        x_abs_max = jnp.pad(x_abs_max, ((0, 0), (0, pad_b)))
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

    quantize_activation = x_q_dtype != x_2d.dtype
    save_acc = n_in > 1
    save_x_q = quantize_activation and n_in == 1 and n_out > 1
    kernel_out_dtype = jnp.float32 if tp_axis is not None else jnp.dtype(out_dtype)
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

    kernel = pl.pallas_call(
        functools.partial(
            _fp8_matmul_pallas_kernel,
            x_q_dtype=x_q_dtype,
            save_acc=save_acc,
            save_x_q=save_x_q,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block, in_block), lambda b, o, i: (b, i)),
                pl.BlockSpec((out_block, in_block), lambda b, o, i: (o, i)),
                pl.BlockSpec((1, out_block), lambda b, o, i: (0, o)),
                pl.BlockSpec((1, batch_block), lambda b, o, i: (0, b)),
            ],
            out_specs=pl.BlockSpec((batch_block, out_block), lambda b, o, i: (b, o)),
            scratch_shapes=[
                pltpu.VMEM((batch_block, out_block), jnp.float32) if save_acc else None,
                pltpu.VMEM((batch_block, in_block), x_q_dtype) if save_x_q else None,
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

    out = kernel(x_2d, w, w_scale_2d, x_abs_max)[:orig_batch, :orig_n]
    if tp_axis is not None:
        out = jax.lax.psum(out, axis_name=tp_axis).astype(out_dtype)
    return out.reshape(*leading, orig_n)


def fp8_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    out_dtype: Any = jnp.bfloat16,
    tp_axis: str | None = None,
    backend: Literal["auto", "pallas", "xla"] = "auto",
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
            out_dtype=out_dtype,
            tp_axis=tp_axis,
            tile_sizes=tile_sizes,
            interpret=interpret,
        )
    return xla_fp8_matmul(x, w, w_scale, out_dtype=out_dtype, tp_axis=tp_axis)


def sharded_fp8_matmul(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    weight_sharding: P | NamedSharding,
    *,
    mesh: Mesh,
    out_dtype: Any = jnp.bfloat16,
    backend: Literal["auto", "pallas", "xla"] = "auto",
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

    def _local(x_shard: jax.Array, w_shard: jax.Array, s_shard: jax.Array) -> jax.Array:
        return fp8_matmul(
            x_shard,
            w_shard,
            s_shard,
            out_dtype=out_dtype,
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


__all__ = [
    "Fp8MatmulTileSizes",
    "Fp8Weight",
    "fp8_matmul",
    "fp8_matmul_prequantized",
    "get_tuned_block_sizes",
    "pallas_fp8_matmul",
    "quantize_fp8_act",
    "quantize_fp8_weight",
    "safe_fp8_scale",
    "sharded_fp8_matmul",
    "xla_fp8_matmul",
]
