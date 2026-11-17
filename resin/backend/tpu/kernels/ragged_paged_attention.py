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
from typing import Any, NamedTuple

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
_TPU7X_VMEM_BYTES = 64 * 1024 * 1024
_DEVICE_VMEM_BYTES = {
    4: 16 * 1024 * 1024,
    5: 64 * 1024 * 1024,
    6: 128 * 1024 * 1024,
    7: _TPU7X_VMEM_BYTES,
    8: 128 * 1024 * 1024,
}
_DECODE_VMEM_SCRATCH_FRACTION = 0.375
_MIXED_VMEM_SCRATCH_FRACTION = 0.45
_TUNED_VMEM_CAP_FRACTION = 0.75


class RpaBlockSizes(NamedTuple):
    """Fetch and compute block sizes ``(bq_sz, bkv_sz, bq_csz, bkv_csz)`` for RPA v3."""

    bq_sz: int
    bkv_sz: int
    bq_csz: int
    bkv_csz: int

    def as_dict(self) -> dict[str, int]:
        return self._asdict()


def get_vmem_estimate_bytes(
    actual_num_kv_heads: int,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
    bq_sz: int,
    bkv_sz: int,
    q_dtype: Any,
    kv_dtype: Any,
) -> int:
    """Estimate VMEM scratch bytes allocated by ``run_rpa_case`` for a given block shape."""
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_q_per_kv = align_to(actual_num_q_heads_per_kv_head, q_packing)
    bkv_stride = int(cdiv(actual_num_kv_heads * 2, kv_packing))
    if has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    head_dim = align_to(actual_head_dim, 128)
    q_itemsize = jnp.dtype(q_dtype).itemsize
    kv_itemsize = jnp.dtype(kv_dtype).itemsize

    bkv_bytes = 2 * bkv_sz * bkv_stride * kv_packing * head_dim * kv_itemsize
    bq_bo_bytes = 4 * actual_num_kv_heads * bq_sz * num_q_per_kv * head_dim * q_itemsize
    lm_bytes = 2 * actual_num_kv_heads * bq_sz * num_q_per_kv * 128 * 4
    acc_bytes = actual_num_kv_heads * bq_sz * num_q_per_kv * head_dim * 4
    return bkv_bytes + bq_bo_bytes + lm_bytes + acc_bytes


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


def unpack_prepared_inputs(
    q_packed: jax.Array,
    kv_merged: jax.Array,
    *,
    actual_num_q_heads: int | None = None,
    actual_num_kv_heads: int | None = None,
    actual_head_dim: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Unpack ``(q_packed, kv_merged)`` back to 3D ``(queries, keys, values)``."""
    kv_heads = q_packed.shape[0] if actual_num_kv_heads is None else actual_num_kv_heads
    tokens = q_packed.shape[1]
    padded_group = q_packed.shape[2] * q_packed.shape[3]
    q_heads = (
        kv_heads * padded_group if actual_num_q_heads is None else actual_num_q_heads
    )
    group_size = q_heads // kv_heads
    head_dim = q_packed.shape[4] if actual_head_dim is None else actual_head_dim

    q = (
        q_packed[:kv_heads]
        .swapaxes(0, 1)
        .reshape(tokens, kv_heads, padded_group, q_packed.shape[4])[
            :, :, :group_size, :head_dim
        ]
        .reshape(tokens, q_heads, head_dim)
    )
    kv_flat = kv_merged.reshape(
        tokens, kv_merged.shape[1] * kv_merged.shape[2], kv_merged.shape[3]
    )[:, : 2 * kv_heads, :head_dim]
    return q, kv_flat[:, 0::2, :], kv_flat[:, 1::2, :]


def static_validate_inputs(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array | None,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    prepared: bool = False,
    actual_num_q_heads: int | None = None,
    actual_head_dim: int | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
) -> None:
    if prepared:
        if values is not None:
            raise ValueError("Expected values=None when prepared=True.")
        if queries.ndim != 5 or keys.ndim != 4:
            raise ValueError(
                f"Expected 5D q_packed and 4D kv_merged when prepared=True, got {queries.shape=}, {keys.shape=}"
            )
        if queries.shape[1] != keys.shape[0]:
            raise ValueError(
                f"Mismatched token count in prepared inputs: {queries.shape=} vs {keys.shape=}"
            )
        actual_num_kv_heads = queries.shape[0]
        q_packing = get_dtype_packing(queries.dtype)
        if queries.shape[3] != q_packing:
            raise ValueError(
                f"Expected q_packed axis 3 == {q_packing}, got {queries.shape[3]}"
            )
        padded_group = queries.shape[2] * q_packing
        eff_q_heads = (
            actual_num_kv_heads * padded_group
            if actual_num_q_heads is None
            else actual_num_q_heads
        )
        eff_head_dim = queries.shape[4] if actual_head_dim is None else actual_head_dim
        if eff_q_heads % actual_num_kv_heads != 0:
            raise ValueError(
                f"{eff_q_heads=} must be divisible by {actual_num_kv_heads=}."
            )
        if align_to(eff_q_heads // actual_num_kv_heads, q_packing) != padded_group:
            raise ValueError(f"Incompatible {eff_q_heads=} and {queries.shape=}.")
        expected_kv_shape = get_kv_cache_shape(
            kv_cache.shape[0],
            kv_cache.shape[1],
            actual_num_kv_heads,
            eff_head_dim,
            kv_cache.dtype,
        )
        if kv_cache.shape != expected_kv_shape:
            raise ValueError(f"Expected {kv_cache.shape=} == {expected_kv_shape=}")
        if (
            keys.shape[1:] != kv_cache.shape[2:]
            or queries.shape[4] != kv_cache.shape[4]
        ):
            raise ValueError(
                f"Prepared shapes {queries.shape=}, {keys.shape=} do not match {kv_cache.shape=}"
            )
        if keys.dtype != kv_cache.dtype:
            raise ValueError(
                f"Expected matching KV dtypes: {kv_cache.dtype=}, {keys.dtype=}"
            )
    else:
        if values is None or not (queries.ndim == keys.ndim == values.ndim == 3):
            raise ValueError(
                f"Expected 3D Q/K/V, got {queries.shape=}, {keys.shape=}, {getattr(values, 'shape', None)=}"
            )
        if keys.shape != values.shape:
            raise ValueError(f"Expected {keys.shape=} == {values.shape=}")
        if queries.shape[0] != keys.shape[0] or queries.shape[2] != keys.shape[2]:
            raise ValueError(
                f"Mismatched Q/K shapes: {queries.shape=} vs {keys.shape=}"
            )

        eff_q_heads = queries.shape[1]
        eff_head_dim = queries.shape[2]
        actual_num_kv_heads = keys.shape[1]
        if eff_q_heads % actual_num_kv_heads != 0:
            raise ValueError(
                f"actual_num_q_heads={eff_q_heads} must be divisible by {actual_num_kv_heads=}."
            )

        expected_kv_shape = get_kv_cache_shape(
            kv_cache.shape[0],
            kv_cache.shape[1],
            actual_num_kv_heads,
            eff_head_dim,
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
    sliding_window: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> dict[str, int]:
    """Select divisible, page-aligned ``(bq_sz, bkv_sz, bq_csz, bkv_csz)``."""
    del max_num_seqs
    version = get_tpu_version() if tpu_version is None else tpu_version
    group_pow2 = next_power_of_2(actual_num_q_heads // actual_num_kv_heads)
    max_q = next_power_of_2(max_num_tokens)
    kv_unit = math.lcm(128, page_size) if version >= 4 else page_size
    max_units = max(1, int(cdiv(pages_per_seq * page_size, kv_unit)))

    if case == RpaCase.DECODE:
        kv_packing = get_dtype_packing(kv_dtype)
        aligned_hd = align_to(head_dim, 128)
        num_kv_heads_x2 = next_power_of_2(align_to(actual_num_kv_heads * 2, kv_packing))
        min_bkv_sz_to_peak = (
            (16 * 1024 * 1024 * kv_packing) // 4 // aligned_hd // num_kv_heads_x2
        )
        horizon_kv = _decode_horizon_kv(page_size, pages_per_seq, sliding_window)
        version_cap = 512 if version == 4 else (2048 if version < 7 else 4096)
        target_bkv = min(min_bkv_sz_to_peak, version_cap, horizon_kv)
        table_kv = pages_per_seq * page_size
        bkv_sz = min(_largest_pow2_multiple(target_bkv, page_size), table_kv)
        return _clamp_decode_vmem(
            RpaBlockSizes(1, bkv_sz, 1, bkv_sz),
            actual_num_kv_heads=actual_num_kv_heads,
            actual_num_q_heads_per_kv_head=actual_num_q_heads // actual_num_kv_heads,
            actual_head_dim=head_dim,
            min_unit=page_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            max_scratch=int(
                _DECODE_VMEM_SCRATCH_FRACTION
                * _effective_vmem_limit(version, tpu_version, vmem_limit_bytes)
            ),
        ).as_dict()

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


_DecodeTableKey = tuple[int, str, int, int, int, int]
_MixedTableKey = tuple[int, str, int, int, int, int, int]


def _largest_pow2_multiple(limit: int, unit: int) -> int:
    mult = max(1, limit // unit)
    return unit * (1 << (mult.bit_length() - 1))


def _decode_horizon_kv(
    page_size: int, pages_per_seq: int, sliding_window: int | None
) -> int:
    horizon_kv = (pages_per_seq - 1) * page_size if pages_per_seq > 1 else page_size
    if sliding_window is not None:
        horizon_kv = min(horizon_kv, align_to(sliding_window + page_size, page_size))
    return horizon_kv


def _effective_vmem_limit(
    version: int, tpu_version: int | None, vmem_limit_bytes: int | None
) -> int:
    if vmem_limit_bytes is not None:
        return vmem_limit_bytes
    if tpu_version is None and version >= 4:
        return pltpu.get_tpu_info().vmem_capacity_bytes
    return _DEVICE_VMEM_BYTES.get(version, DEFAULT_INTERPRET_VMEM_BYTES)


def _clamp_decode_vmem(
    blocks: RpaBlockSizes,
    *,
    actual_num_kv_heads: int,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
    min_unit: int,
    q_dtype: Any,
    kv_dtype: Any,
    max_scratch: int,
) -> RpaBlockSizes:
    bkv_sz = blocks.bkv_sz
    bkv_csz = blocks.bkv_csz
    while (
        bkv_sz >= 2 * min_unit
        and get_vmem_estimate_bytes(
            actual_num_kv_heads,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
            blocks.bq_sz,
            bkv_sz,
            q_dtype,
            kv_dtype,
        )
        > max_scratch
    ):
        bkv_sz //= 2
    return RpaBlockSizes(blocks.bq_sz, bkv_sz, blocks.bq_csz, min(bkv_csz, bkv_sz))


def _build_tpu7x_rpa_decode_tuned_table() -> dict[_DecodeTableKey, RpaBlockSizes]:
    # Seeded from vllm-project/tpu-inference TUNED_BLOCK_SIZES['TPU v7'] with
    # decode invariants (bq_sz=1, bq_csz=1, bkv_csz=bkv_sz). Entries whose
    # double-buffered KV tensor exceeds 24 MiB step down to 2048 or 1024.
    max_bkv_fp8 = {
        (32, 4, 128): 4096,
        (64, 8, 128): 4096,
        (16, 2, 128): 4096,
        (16, 2, 256): 4096,
        (16, 4, 256): 4096,
        (32, 4, 256): 4096,
        (32, 8, 256): 2048,
        (8, 2, 256): 4096,
    }
    max_bkv_bf16 = {
        (32, 4, 128): 4096,
        (64, 8, 128): 2048,
        (16, 2, 128): 4096,
        (16, 2, 256): 4096,
        (16, 4, 256): 2048,
        (32, 4, 256): 2048,
        (32, 8, 256): 1024,
        (8, 2, 256): 4096,
    }
    table: dict[_DecodeTableKey, RpaBlockSizes] = {}
    for page_size in (128, 256):
        for kv_name, shape_caps in (
            ("float8_e4m3fn", max_bkv_fp8),
            ("float8_e5m2", max_bkv_fp8),
            ("bfloat16", max_bkv_bf16),
        ):
            for (q_heads, kv_heads, aligned_hd), cap in shape_caps.items():
                for kv_bucket in (256, 512, 1024, 2048, 4096):
                    bkv = min(kv_bucket, cap)
                    table[
                        (page_size, kv_name, q_heads, kv_heads, aligned_hd, kv_bucket)
                    ] = RpaBlockSizes(1, bkv, 1, bkv)
    return table


_TPU7X_RPA_DECODE_TUNED_BLOCKS = _build_tpu7x_rpa_decode_tuned_table()


def _clamp_mixed_vmem(
    blocks: RpaBlockSizes,
    *,
    actual_num_kv_heads: int,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
    min_kv_unit: int,
    q_dtype: Any,
    kv_dtype: Any,
    max_scratch: int,
) -> RpaBlockSizes:
    bq_sz = blocks.bq_sz
    bkv_sz = blocks.bkv_sz
    bq_csz = min(blocks.bq_csz, bq_sz)
    bkv_csz = min(blocks.bkv_csz, bkv_sz)
    while bq_sz >= 2 * bq_csz and (
        get_vmem_estimate_bytes(
            actual_num_kv_heads,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
            bq_sz,
            bkv_sz,
            q_dtype,
            kv_dtype,
        )
        > max_scratch
    ):
        bq_sz //= 2
        bq_csz = min(bq_csz, bq_sz)
    while bkv_sz >= 2 * min_kv_unit and (
        get_vmem_estimate_bytes(
            actual_num_kv_heads,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
            bq_sz,
            bkv_sz,
            q_dtype,
            kv_dtype,
        )
        > max_scratch
    ):
        bkv_sz //= 2
        bkv_csz = min(bkv_csz, bkv_sz)
    while bq_sz >= 2 and (
        get_vmem_estimate_bytes(
            actual_num_kv_heads,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
            bq_sz,
            bkv_sz,
            q_dtype,
            kv_dtype,
        )
        > max_scratch
    ):
        bq_sz //= 2
        bq_csz = min(bq_csz, bq_sz)
    while bq_sz % bq_csz != 0:
        bq_csz -= 1
    while bkv_sz % bkv_csz != 0:
        bkv_csz -= min_kv_unit
    return RpaBlockSizes(bq_sz, bkv_sz, bq_csz, max(min_kv_unit, bkv_csz))


def _build_tpu7x_rpa_mixed_tuned_table() -> dict[_MixedTableKey, RpaBlockSizes]:
    # Restricted to shapes swept on TPU7x chip 0: Qwen3-32B tp=2 (32, 4, 128) at
    # page_size=256. Unswept shapes (tp=1, head_dim=256, page_size=128) fall back
    # to get_default_block_sizes until measured on hardware.
    page_size = 256
    q_heads, kv_heads, aligned_hd = 32, 4, 128
    group_pow2 = next_power_of_2(q_heads // kv_heads)
    target_bq = max(1, 2048 // group_pow2)
    target_bq_c = max(1, 1024 // group_pow2)
    table: dict[_MixedTableKey, RpaBlockSizes] = {}
    for kv_name in ("float8_e4m3fn", "float8_e5m2", "bfloat16"):
        q_dtype = jnp.bfloat16
        kv_dtype = jnp.dtype(kv_name)
        for q_bucket in (128, 256, 512, 1024, 2048, 4096, 8192):
            bq = max(1, min(target_bq, max(1, q_bucket // 2)))
            bq_c = max(1, min(target_bq_c, bq))
            while bq % bq_c != 0:
                bq_c -= 1
            for kv_bucket in (256, 512, 1024, 2048, 4096, 8192):
                cur_bq = bq
                cur_bq_c = bq_c
                if kv_bucket <= 2048:
                    if q_bucket <= 1024:
                        cur_bq = min(128, cur_bq)
                        cur_bq_c = min(64, cur_bq)
                        bkv = min(1024, kv_bucket)
                        bkv_c = min(256, bkv)
                    else:
                        cur_bq_c = min(64, cur_bq)
                        bkv = min(512, kv_bucket)
                        bkv_c = min(256, bkv)
                else:
                    bkv = min(2048, kv_bucket)
                    bkv_c = min(1024, bkv)
                while cur_bq % cur_bq_c != 0:
                    cur_bq_c -= 1
                bkv = max(page_size, (bkv // page_size) * page_size)
                bkv_c = max(page_size, (bkv_c // page_size) * page_size)
                while bkv % bkv_c != 0:
                    bkv_c -= page_size
                cand = _clamp_mixed_vmem(
                    RpaBlockSizes(cur_bq, bkv, cur_bq_c, bkv_c),
                    actual_num_kv_heads=kv_heads,
                    actual_num_q_heads_per_kv_head=q_heads // kv_heads,
                    actual_head_dim=aligned_hd,
                    min_kv_unit=page_size,
                    q_dtype=q_dtype,
                    kv_dtype=kv_dtype,
                    max_scratch=int(_MIXED_VMEM_SCRATCH_FRACTION * _TPU7X_VMEM_BYTES),
                )
                table[
                    (
                        page_size,
                        kv_name,
                        q_heads,
                        kv_heads,
                        aligned_hd,
                        q_bucket,
                        kv_bucket,
                    )
                ] = cand
    return table


_TPU7X_RPA_MIXED_TUNED_BLOCKS = _build_tpu7x_rpa_mixed_tuned_table()


def get_rpa_block_sizes(
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
    sliding_window: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> RpaBlockSizes:
    """Return tuned TPU7x decode/mixed block sizes, falling back to ``get_default_block_sizes``."""
    version = get_tpu_version() if tpu_version is None else tpu_version
    if case == RpaCase.DECODE and version == 7:
        table_kv = pages_per_seq * page_size
        horizon_kv = _decode_horizon_kv(page_size, pages_per_seq, sliding_window)
        kv_bucket = _largest_pow2_multiple(min(horizon_kv, 4096), page_size)
        key = (
            page_size,
            jnp.dtype(kv_dtype).name,
            actual_num_q_heads,
            actual_num_kv_heads,
            align_to(head_dim, 128),
            kv_bucket,
        )
        if (tuned := _TPU7X_RPA_DECODE_TUNED_BLOCKS.get(key)) is not None:
            clamped_bkv = min(tuned.bkv_sz, _largest_pow2_multiple(table_kv, page_size))
            bounded = RpaBlockSizes(1, clamped_bkv, 1, min(tuned.bkv_csz, clamped_bkv))
            eff_limit = _effective_vmem_limit(version, tpu_version, vmem_limit_bytes)
            cap_frac = (
                _DECODE_VMEM_SCRATCH_FRACTION
                if vmem_limit_bytes is not None
                else _TUNED_VMEM_CAP_FRACTION
            )
            return _clamp_decode_vmem(
                bounded,
                actual_num_kv_heads=actual_num_kv_heads,
                actual_num_q_heads_per_kv_head=actual_num_q_heads
                // actual_num_kv_heads,
                actual_head_dim=head_dim,
                min_unit=page_size,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                max_scratch=int(cap_frac * eff_limit),
            )
    elif case == RpaCase.MIXED and version == 7:
        # kv_bucket keys on the static arena horizon (pages_per_seq - 1) * page_size
        # rather than runtime prompt length, matching per-workload max_model_len.
        table_kv = pages_per_seq * page_size
        horizon_kv = _decode_horizon_kv(page_size, pages_per_seq, sliding_window)
        kv_bucket = _largest_pow2_multiple(min(horizon_kv, 8192), page_size)
        q_bucket = next_power_of_2(min(max(max_num_tokens, 128), 8192))
        mixed_key = (
            page_size,
            jnp.dtype(kv_dtype).name,
            actual_num_q_heads,
            actual_num_kv_heads,
            align_to(head_dim, 128),
            q_bucket,
            kv_bucket,
        )
        if (tuned := _TPU7X_RPA_MIXED_TUNED_BLOCKS.get(mixed_key)) is not None:
            max_q_pow2 = max(1, next_power_of_2(max_num_tokens))
            clamped_bq = max(1, min(tuned.bq_sz, max(1, max_q_pow2 // 2)))
            clamped_bq_c = max(1, min(tuned.bq_csz, clamped_bq))
            while clamped_bq % clamped_bq_c != 0:
                clamped_bq_c -= 1
            clamped_bkv = min(tuned.bkv_sz, _largest_pow2_multiple(table_kv, page_size))
            clamped_bkv_c = min(tuned.bkv_csz, clamped_bkv)
            while clamped_bkv % clamped_bkv_c != 0:
                clamped_bkv_c -= page_size
            eff_limit = _effective_vmem_limit(version, tpu_version, vmem_limit_bytes)
            return _clamp_mixed_vmem(
                RpaBlockSizes(
                    clamped_bq,
                    clamped_bkv,
                    clamped_bq_c,
                    max(page_size, clamped_bkv_c),
                ),
                actual_num_kv_heads=actual_num_kv_heads,
                actual_num_q_heads_per_kv_head=actual_num_q_heads
                // actual_num_kv_heads,
                actual_head_dim=head_dim,
                min_kv_unit=page_size,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                max_scratch=int(_MIXED_VMEM_SCRATCH_FRACTION * eff_limit),
            )

    return RpaBlockSizes(
        **get_default_block_sizes(
            q_dtype,
            kv_dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            head_dim,
            page_size,
            max_num_tokens,
            max_num_seqs,
            pages_per_seq,
            case=case,
            tpu_version=tpu_version,
            sliding_window=sliding_window,
            vmem_limit_bytes=vmem_limit_bytes,
        )
    )


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
        "prepared",
        "actual_num_q_heads",
        "actual_head_dim",
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
    values: jax.Array | None,
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
    prepared: bool = False,
    actual_num_q_heads: int | None = None,
    actual_head_dim: int | None = None,
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
    if not prepared:
        if values is None:
            raise ValueError("values is required when prepared=False.")
        if keys.dtype != kv_cache.dtype:
            keys = quantize_kv(keys, kv_cache.dtype, k_scale)
        if values.dtype != kv_cache.dtype:
            values = quantize_kv(values, kv_cache.dtype, v_scale)

    if mask_value is None:
        mask_value = _DEFAULT_MASK_VALUE

    tpu_version = get_tpu_version()
    use_pallas = tpu_version >= 4 or interpret

    if use_pallas and vmem_limit_bytes is None:
        vmem_limit_bytes = _effective_vmem_limit(tpu_version, None, None)

    static_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        prepared=prepared,
        actual_num_q_heads=actual_num_q_heads,
        actual_head_dim=actual_head_dim,
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
        if prepared:
            cpu_q, cpu_k, cpu_v = unpack_prepared_inputs(
                queries,
                keys,
                actual_num_q_heads=actual_num_q_heads,
                actual_num_kv_heads=queries.shape[0],
                actual_head_dim=actual_head_dim,
            )
        else:
            assert values is not None
            cpu_q, cpu_k, cpu_v = queries, keys, values
        return _cpu_ragged_paged_attention(
            cpu_q,
            cpu_k,
            cpu_v,
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

    if prepared:
        actual_num_kv_heads = queries.shape[0]
        eff_q_heads = (
            actual_num_kv_heads * queries.shape[2] * queries.shape[3]
            if actual_num_q_heads is None
            else actual_num_q_heads
        )
        eff_head_dim = queries.shape[4] if actual_head_dim is None else actual_head_dim
        actual_num_q_heads_per_kv_head = eff_q_heads // actual_num_kv_heads
        q, kv = queries, keys
    else:
        assert values is not None
        eff_q_heads = queries.shape[1]
        eff_head_dim = queries.shape[2]
        actual_num_kv_heads = keys.shape[1]
        actual_num_q_heads_per_kv_head = eff_q_heads // actual_num_kv_heads
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
            return RpaBlockSizes(*override).as_dict()
        return get_rpa_block_sizes(
            q.dtype,
            kv_cache.dtype,
            eff_q_heads,
            actual_num_kv_heads,
            eff_head_dim,
            page_size,
            max_num_tokens,
            max_num_seqs,
            pages_per_seq,
            case=case,
            tpu_version=tpu_version,
            sliding_window=sliding_window,
            vmem_limit_bytes=vmem_limit_bytes,
        ).as_dict()

    d_blocks = _blocks_for(d_block_sizes, RpaCase.DECODE)
    q, kv_cache = jax.lax.cond(
        distribution[0] > 0,
        lambda args: run_rpa_case(
            args[0],
            args[1],
            **d_blocks,
            static_q_len=1,
            case=RpaCase.DECODE,
        ),
        lambda args: args,
        (q, kv_cache),
    )
    m_blocks = _blocks_for(m_block_sizes, RpaCase.MIXED)
    q, kv_cache = jax.lax.cond(
        distribution[2] > distribution[1],
        lambda args: run_rpa_case(
            args[0],
            args[1],
            **m_blocks,
            static_q_len=None,
            case=RpaCase.MIXED,
        ),
        lambda args: args,
        (q, kv_cache),
    )
    return (
        prepare_outputs(q, actual_num_q_heads_per_kv_head, eff_head_dim),
        kv_cache,
    )


__all__ = [
    "RpaBlockSizes",
    "RpaCase",
    "get_default_block_sizes",
    "get_kv_cache_shape",
    "get_rpa_block_sizes",
    "get_vmem_estimate_bytes",
    "merge_kv",
    "prepare_inputs",
    "quantize_kv",
    "ragged_paged_attention",
    "static_validate_inputs",
    "unpack_prepared_inputs",
]
