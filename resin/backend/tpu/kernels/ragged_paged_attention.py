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
"""Ragged Paged Attention v3 kernel for the Resin TPU backend.

Adapted from vllm-project/tpu-inference (Apache-2.0) at revision
d9c7e19da8c497ec63b9381bb0bc9b2a5e0d4bb2:
tpu_inference/kernels/ragged_paged_attention/v3/kernel.py.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from resin.backend.tpu import reference
from resin.backend.tpu.kernels._rpa_v3_pallas import (
    RpaCase,
    _ragged_paged_attention_kernel,
    align_to,
    cdiv,
    get_dtype_packing,
    get_tpu_version,
    has_bank_conflicts,
    next_power_of_2,
)

DEFAULT_INTERPRET_VMEM_BYTES = 64 * 1024 * 1024


def get_kv_cache_shape(
    total_num_pages: int,
    page_size: int,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    kv_dtype: Any,
) -> tuple[int, int, int, int, int]:
    """Return the TPU-tiled per-die KV cache shape for RPA v3."""
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


def quantize_kv(x: jax.Array, dtype: Any, scale: float | None = None) -> jax.Array:
    """Cast K or V to ``dtype``, saturating 8-bit floats instead of overflowing to NaN."""
    target = jnp.dtype(dtype)
    if x.dtype == target and scale is None:
        return x
    if target.itemsize != 1:
        return x.astype(target)
    bound = float(jnp.finfo(target).max)
    scaled = x.astype(jnp.float32) / (1.0 if scale is None else scale)
    return jnp.clip(scaled, -bound, bound).astype(target)


def merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)
    head_dim = align_to(actual_head_dim, 128)
    return jnp.pad(
        jnp.concat([k, v], axis=-1).reshape(
            max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
        ),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )


def prepare_inputs(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> tuple[jax.Array, jax.Array]:
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q_packed = (
        jnp.pad(
            q.reshape(
                max_num_tokens,
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            head_dim,
        )
        .swapaxes(0, 1)
    )
    return q_packed, merge_kv(k, v)


def prepare_outputs(
    out: jax.Array,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
) -> jax.Array:
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = out.shape
    actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
    return (
        out.swapaxes(0, 1)
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing * q_packing,
            head_dim,
        )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim]
        .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
    )


def static_validate_inputs(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
) -> None:
    if not (queries.ndim == keys.ndim == values.ndim == 3):
        raise ValueError(
            f"Expected 3D Q/K/V, got {queries.shape=}, {keys.shape=}, {values.shape=}"
        )
    if keys.shape != values.shape:
        raise ValueError(f"Expected {keys.shape=} == {values.shape=}")
    if queries.shape[0] != keys.shape[0] or queries.shape[2] != keys.shape[2]:
        raise ValueError(f"Mismatched Q/K shapes: {queries.shape=} vs {keys.shape=}")

    actual_num_q_heads = queries.shape[1]
    actual_head_dim = queries.shape[2]
    actual_num_kv_heads = keys.shape[1]
    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(
            f"{actual_num_q_heads=} must be divisible by {actual_num_kv_heads=}."
        )

    expected_kv_shape = get_kv_cache_shape(
        kv_cache.shape[0],
        kv_cache.shape[1],
        actual_num_kv_heads,
        actual_head_dim,
        kv_cache.dtype,
    )
    if kv_cache.shape != expected_kv_shape:
        raise ValueError(f"Expected {kv_cache.shape=} == {expected_kv_shape=}")
    if not (kv_cache.dtype == keys.dtype == values.dtype):
        raise ValueError(
            f"Expected matching KV dtypes: {kv_cache.dtype=}, {keys.dtype=}, {values.dtype=}"
        )
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected floating KV cache dtype, got {kv_cache.dtype=}")

    page_size = kv_cache.shape[1]
    kv_packing = kv_cache.shape[3]
    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == distribution.dtype
    ):
        raise ValueError("Metadata arrays must have int32 dtype.")
    if not (kv_lens.ndim == page_indices.ndim == cu_q_lens.ndim == 1):
        raise ValueError("kv_lens, page_indices and cu_q_lens must be 1D.")
    max_num_seqs = kv_lens.shape[0]
    if page_indices.shape[0] % max_num_seqs != 0:
        raise ValueError(
            f"{page_indices.shape[0]=} must be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_q_lens.shape=} == ({max_num_seqs + 1},).")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} == (3,).")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError("soft_cap must not be 0.0.")

    def _validate_blocks(blocks: tuple[int, int, int, int] | None, label: str) -> None:
        if blocks is None:
            return
        bq_sz, bkv_sz, bq_csz, bkv_csz = blocks
        if not (0 < bq_csz and bq_sz % bq_csz == 0):
            raise ValueError(f"{label}: invalid query blocks {bq_sz=}, {bq_csz=}")
        if not (0 < bkv_csz and bkv_sz % bkv_csz == 0):
            raise ValueError(f"{label}: invalid KV blocks {bkv_sz=}, {bkv_csz=}")
        if bkv_sz % page_size != 0 or bkv_csz % page_size != 0:
            raise ValueError(
                f"{label}: KV blocks {bkv_sz=}, {bkv_csz=} must be multiples of {page_size=}"
            )

    _validate_blocks(d_block_sizes, "decode")
    _validate_blocks(m_block_sizes, "mixed")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")


def get_default_block_sizes(
    q_dtype: Any,
    kv_dtype: Any,
    actual_num_q_heads: int,
    actual_num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_num_tokens: int,
    max_num_seqs: int,
    pages_per_seq: int,
    *,
    case: RpaCase = RpaCase.MIXED,
    tpu_version: int | None = None,
) -> dict[str, int]:
    """Select divisible, page-aligned ``(bq_sz, bkv_sz, bq_csz, bkv_csz)``."""
    del q_dtype, kv_dtype, head_dim, max_num_seqs
    version = get_tpu_version() if tpu_version is None else tpu_version
    group_pow2 = next_power_of_2(actual_num_q_heads // actual_num_kv_heads)
    max_q = next_power_of_2(max_num_tokens)
    kv_unit = math.lcm(128, page_size) if version >= 4 else page_size
    max_units = max(1, int(cdiv(pages_per_seq * page_size, kv_unit)))

    if case == RpaCase.DECODE:
        bkv_units = max(1, min(int(cdiv(2048, kv_unit)), max_units))
        bkv_c_units = max(1, min(int(cdiv(512, kv_unit)), bkv_units))
        while bkv_units % bkv_c_units != 0:
            bkv_c_units -= 1
        return {
            "bq_sz": 1,
            "bkv_sz": bkv_units * kv_unit,
            "bq_csz": 1,
            "bkv_csz": bkv_c_units * kv_unit,
        }

    is_v7_plus = version >= 7
    target_bq = 2048 if is_v7_plus else 1024
    target_bq_c = 1024 if is_v7_plus else 512
    bq_sz = max(1, min(target_bq // group_pow2, max(1, max_q // 2)))
    bq_csz = max(1, min(target_bq_c // group_pow2, bq_sz))
    while bq_sz % bq_csz != 0:
        bq_csz -= 1

    target_bkv_tokens = 2048 if is_v7_plus else 1024
    max_bkv_units = max(1, max_units // 2) if is_v7_plus else max_units
    bkv_units = max(1, min(int(cdiv(target_bkv_tokens, kv_unit)), max_bkv_units))
    bkv_c_units = max(1, min(int(cdiv(512, kv_unit)), bkv_units))
    while bkv_units % bkv_c_units != 0:
        bkv_c_units -= 1

    return {
        "bq_sz": bq_sz,
        "bkv_sz": bkv_units * kv_unit,
        "bq_csz": bq_csz,
        "bkv_csz": bkv_c_units * kv_unit,
    }


def _cpu_ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None,
    soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
) -> tuple[jax.Array, jax.Array]:
    if q_scale is not None:
        queries = quantize_kv(queries, kv_cache.dtype, q_scale).astype(queries.dtype)
        sm_scale = sm_scale * q_scale
    return reference.ragged_paged_attention(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        k_scale=k_scale,
        v_scale=v_scale,
    )


_DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "k_scale",
        "v_scale",
        "q_scale",
        "use_causal_mask",
        "update_kv_cache",
        "mask_value",
        "d_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "disable_bounds_checks",
        "disable_semaphore_checks",
        "interpret",
    ),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    q_scale: float | None = None,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    mask_value: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    disable_bounds_checks: bool = True,
    disable_semaphore_checks: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run Ragged Paged Attention v3 over per-die ``(queries, keys, values, kv_cache)``.

    Uses the Pallas Mosaic kernel on TPU (and on CPU when ``interpret=True``),
    and a jit-traceable pure-JAX fallback on CPU otherwise. Online softmax and
    cross-block accumulation run in ``float32``; outputs match ``queries.dtype``.
    """
    if keys.dtype != kv_cache.dtype:
        keys = quantize_kv(keys, kv_cache.dtype, k_scale)
    if values.dtype != kv_cache.dtype:
        values = quantize_kv(values, kv_cache.dtype, v_scale)

    if mask_value is None:
        mask_value = _DEFAULT_MASK_VALUE

    tpu_version = get_tpu_version()
    use_pallas = tpu_version >= 4 or interpret

    if use_pallas and vmem_limit_bytes is None:
        vmem_limit_bytes = (
            pltpu.get_tpu_info().vmem_capacity_bytes
            if tpu_version >= 4
            else DEFAULT_INTERPRET_VMEM_BYTES
        )

    static_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        d_block_sizes=d_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    if not use_pallas:
        if not use_causal_mask or not update_kv_cache:
            raise NotImplementedError(
                "Non-causal or KV-share RPA on CPU requires interpret=True."
            )
        return _cpu_ragged_paged_attention(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    actual_num_q_heads = queries.shape[1]
    actual_head_dim = queries.shape[2]
    actual_num_kv_heads = keys.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q, kv = prepare_inputs(queries, keys, values)
    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache.shape[1]
    num_kv_heads_x2_per_kv_packing = kv_cache.shape[2]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing

    init_sem_ids = jnp.zeros((3,), jnp.int32)
    init_bo_ids = jnp.full((4,), -1, jnp.int32)
    init_bkv_update_ids = jnp.full((6,), -1, jnp.int32)
    interpret_params = pltpu.InterpretParams() if interpret else False

    def run_rpa_case(
        cur_q: jax.Array,
        cur_cache: jax.Array,
        *,
        bq_sz: int,
        bkv_sz: int,
        bq_csz: int,
        bkv_csz: int,
        static_q_len: int | None,
        case: RpaCase,
    ) -> tuple[jax.Array, jax.Array]:
        bkv_stride = num_kv_heads_x2_per_kv_packing
        if has_bank_conflicts(bkv_stride):
            bkv_stride += 1

        bkv_double_buf = pltpu.VMEM(
            (2, bkv_sz, bkv_stride, *cur_cache.shape[3:]),
            cur_cache.dtype,
        )
        bq_double_buf = pltpu.VMEM(
            (2, actual_num_kv_heads, bq_sz, *cur_q.shape[2:]),
            cur_q.dtype,
        )
        l_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
            jnp.float32,
        )
        acc_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
            jnp.float32,
        )
        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            init_sem_ids,
            init_bo_ids,
            init_bkv_update_ids,
        )
        scope_name = (
            f"RPA{case.symbol}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
        )
        if sliding_window is not None:
            scope_name += f"-sw_{sliding_window}"

        pin_hbm = tpu_version >= 7 and not interpret
        kernel = pl.pallas_call(
            functools.partial(
                _ragged_paged_attention_kernel,
                use_causal_mask=use_causal_mask,
                update_kv_cache=update_kv_cache,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                bq_csz=bq_csz,
                bkv_csz=bkv_csz,
                case=case,
                tpu_version=tpu_version,
                interpret_mode=interpret,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 3,
                out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
                grid=(1,),
                scratch_shapes=[
                    bkv_double_buf,
                    bq_double_buf,
                    bq_double_buf,
                    pltpu.SemaphoreType.DMA((4, 2)),
                    l_scratch,
                    l_scratch,
                    acc_scratch,
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=disable_bounds_checks,
                disable_semaphore_checks=disable_semaphore_checks,
            ),
            out_shape=(
                [
                    pltpu.HBM(shape=cur_q.shape, dtype=cur_q.dtype),
                    pltpu.HBM(shape=cur_cache.shape, dtype=cur_cache.dtype),
                ]
                if pin_hbm
                else [
                    jax.ShapeDtypeStruct(shape=cur_q.shape, dtype=cur_q.dtype),
                    jax.ShapeDtypeStruct(shape=cur_cache.shape, dtype=cur_cache.dtype),
                ]
            ),
            input_output_aliases={7: 0, 9: 1},
            interpret=interpret_params,
            name=scope_name,
        )

        if pin_hbm:

            @jax.jit
            def _run(prefetches, q_in, kv_in, cache_in):
                return kernel(
                    *prefetches,
                    pltpu.with_memory_space_constraint(q_in, pltpu.HBM),
                    pltpu.with_memory_space_constraint(kv_in, pltpu.HBM),
                    pltpu.with_memory_space_constraint(cache_in, pltpu.HBM),
                )

            return _run(scalar_prefetches, cur_q, kv, cur_cache)
        return kernel(*scalar_prefetches, cur_q, kv, cur_cache)

    def _blocks_for(
        override: tuple[int, int, int, int] | None, case: RpaCase
    ) -> dict[str, int]:
        if override is not None:
            return {
                "bq_sz": override[0],
                "bkv_sz": override[1],
                "bq_csz": override[2],
                "bkv_csz": override[3],
            }
        return get_default_block_sizes(
            q.dtype,
            kv_cache.dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            head_dim,
            page_size,
            max_num_tokens,
            max_num_seqs,
            pages_per_seq,
            case=case,
            tpu_version=tpu_version,
        )

    q, kv_cache = run_rpa_case(
        q,
        kv_cache,
        **_blocks_for(d_block_sizes, RpaCase.DECODE),
        static_q_len=1,
        case=RpaCase.DECODE,
    )
    q, kv_cache = run_rpa_case(
        q,
        kv_cache,
        **_blocks_for(m_block_sizes, RpaCase.MIXED),
        static_q_len=None,
        case=RpaCase.MIXED,
    )
    return (
        prepare_outputs(q, actual_num_q_heads_per_kv_head, actual_head_dim),
        kv_cache,
    )


__all__ = [
    "RpaCase",
    "get_default_block_sizes",
    "get_kv_cache_shape",
    "quantize_kv",
    "ragged_paged_attention",
    "static_validate_inputs",
]
