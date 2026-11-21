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
"""Pallas Mosaic kernel bodies for Ragged Paged Attention v3 and batched RPA.

Adapted from vllm-project/tpu-inference (Apache-2.0):
- Stock RPA v3 at revision d9c7e19da8c497ec63b9381bb0bc9b2a5e0d4bb2:
  tpu_inference/kernels/ragged_paged_attention/v3/{kernel.py,util.py}.
- Batched decode and mixed RPA at revision da76a6a40bf4cc8fd32c0d07b471c0b4ae32754c:
  tpu_inference/kernels/experimental/batched_rpa/{bref_override,configs,flash_attention,kernel,pipeline,schedule,utils}.py.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def cdiv(a: int | jax.Array, b: int | jax.Array) -> int | jax.Array:
    return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def get_dtype_bitwidth(dtype) -> int:
    return jnp.dtype(dtype).itemsize * 8


def get_dtype_packing(dtype) -> int:
    return 32 // get_dtype_bitwidth(dtype)


def next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_tpu_version() -> int:
    """Return the active TPU generation number, or -1 when not on TPU."""
    if jax.default_backend() != "tpu":
        return -1
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    if kind[-1:] in ("p", "e", "i", "x", "t"):
        kind = kind[:-1]
    for idx in range(len(kind) - 1, -1, -1):
        if kind[idx].isdigit():
            return int(kind[idx])
    return -1


def has_bank_conflicts(stride: int, distance: int = 24, num_banks: int = 32) -> bool:
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False


_TPU_NUM_LANES = 128


def _broadcast_minor(src: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    if src.shape == shape:
        return src
    target_minor = align_to(shape[-1], src.shape[-1])
    return jnp.tile(src, (target_minor // src.shape[-1],))[..., : shape[-1]]


def _strided_load(
    ref: Any, start: int, sz: int, step: int, *, dtype: Any = None
) -> jax.Array:
    r, l_dim = ref.shape
    folds = l_dim // _TPU_NUM_LANES
    ref_flat = ref.reshape(r * folds, _TPU_NUM_LANES)
    start *= folds
    sz *= folds
    step *= folds
    vec = jnp.concat(
        [ref_flat[pl.ds(start + i, sz // step, step)] for i in range(folds)],
        axis=1,
    )
    return pltpu.bitcast(vec, dtype) if dtype is not None else vec


def _strided_store(ref: Any, start: int, sz: int, step: int, val: jax.Array) -> None:
    r, l_dim = ref.shape
    folds = l_dim // _TPU_NUM_LANES
    ref_flat = ref.reshape(r * folds, _TPU_NUM_LANES)
    start *= folds
    sz *= folds
    step *= folds
    for i in range(folds):
        ref_flat[pl.ds(start + i, sz // step, step)] = val[
            :, i * _TPU_NUM_LANES : (i + 1) * _TPU_NUM_LANES
        ]


class RpaCase(Enum):
    """Execution specialization over ranges of ``distribution = [i, j, k]``."""

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self) -> str:
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(self, distribution: jax.Array) -> tuple[int | jax.Array, jax.Array]:
        if self == RpaCase.DECODE:
            return 0, distribution[0]
        if self == RpaCase.PREFILL:
            return distribution[0], distribution[1]
        if self == RpaCase.MIXED:
            return distribution[1], distribution[2]
        raise ValueError(f"Unsupported RPA case: {self}")


def _ragged_paged_attention_kernel(*args, **kwargs) -> None:
    distribution_ref = args[3]
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(distribution_ref)

    @pl.loop(start_seq_idx, end_seq_idx)
    def _(seq_idx):
        _ragged_paged_attention_kernel_loop(seq_idx, *args, **kwargs)


def _ragged_paged_attention_kernel_loop(
    seq_idx,
    # Scalar prefetch (SMEM)
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    distribution_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    # Inputs (HBM)
    q_hbm_ref,
    kv_hbm_ref,
    kv_cache_hbm_ref,
    # Outputs (HBM)
    o_hbm_ref,
    updated_kv_cache_hbm_ref,
    # Scratch (VMEM + DMA semaphores)
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    use_causal_mask: bool = True,
    update_kv_cache: bool = True,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    static_q_len: int | None = None,
    bq_sz: int,
    bkv_sz: int,
    bq_csz: int,
    bkv_csz: int,
    case: RpaCase = RpaCase.MIXED,
    tpu_version: int = -1,
    interpret_mode: bool = False,
) -> None:
    if case == RpaCase.DECODE:
        use_causal_mask = False

    acc_dtype = acc_ref.dtype
    (
        actual_num_kv_heads,
        _max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    (
        _total_num_pages,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    bkv_stride = bkv_x2_ref.shape[2]
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    bkv_p = bkv_sz // page_size
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]
    kv_q_gap = kv_len - q_len
    cur_seq_start_bkv_idx = 0
    next_seq_start_bkv_idx = 0

    if sliding_window is not None:
        cur_seq_start_bkv_idx = jnp.maximum(kv_q_gap - sliding_window, 0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, end_seq_idx - 1)
        next_q_start = cu_q_lens_ref[next_seq_idx]
        next_q_end = cu_q_lens_ref[next_seq_idx + 1]
        next_q_len = next_q_end - next_q_start
        next_kv_len = kv_lens_ref[next_seq_idx]
        next_kv_q_gap = next_kv_len - next_q_len
        next_seq_start_bkv_idx = (
            jnp.maximum(next_kv_q_gap - sliding_window, 0) // bkv_sz
        )

    broadcast_minor = _broadcast_minor

    def mask_and(mask, new_mask):
        return new_mask if mask is None else jnp.logical_and(mask, new_mask)

    def flash_attention_step1_qk_softmax(
        q,
        k,
        v,
        l_slice_ref,
        m_slice_ref,
        *,
        processed_q_len,
        processed_kv_len,
        effective_kv_len,
    ):
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                q = jnp.clip(q, min=float(dtype_info.min), max=float(dtype_info.max))
            q = q.astype(k.dtype)

        int_ty = jnp.int32
        max_kv_len = pages_per_seq * page_size
        if (
            get_dtype_packing(q_dtype) != 1
            and tpu_version >= 6
            and max_kv_len <= jnp.iinfo(jnp.int16).max
        ):
            int_ty = jnp.int16
        processed_q_len_int = processed_q_len.astype(int_ty)
        processed_kv_len_int = processed_kv_len.astype(int_ty)
        effective_kv_len_int = effective_kv_len.astype(int_ty)
        v_span = processed_kv_len_int + lax.broadcasted_iota(int_ty, v.shape, 0)
        valid_kv = v_span < effective_kv_len_int
        if sliding_window is not None:
            valid_kv = jnp.logical_and(
                valid_kv, processed_q_len_int < v_span + sliding_window
            )
        # Mask K and V before matmul so NaN-filled stale or pre-window page
        # tails never participate in floating-point dot products.
        k = jnp.where(valid_kv, k, jnp.zeros_like(k))
        v = jnp.where(valid_kv, v, jnp.zeros_like(v))

        s = jnp.matmul(q, k.T, preferred_element_type=jnp.float32)

        s_scale = sm_scale
        if k_scale is not None:
            s_scale *= k_scale
        if q_scale is not None:
            s_scale *= q_scale
        s *= s_scale

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        q_span = processed_q_len_int + (
            lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
        ).astype(int_ty)
        k_span = processed_kv_len_int + lax.broadcasted_iota(int_ty, s.shape, 1)

        mask = k_span < effective_kv_len_int
        if use_causal_mask:
            mask = mask_and(mask, q_span >= k_span)
        if sliding_window is not None:
            mask = mask_and(mask, q_span < k_span + sliding_window)
        s = jnp.where(mask, s, mask_value)

        s_rowmax = jnp.max(s, axis=1, keepdims=True).astype(acc_dtype)
        m_prev = m_slice_ref[...]
        m_curr = jnp.maximum(m_prev, s_rowmax)
        m_slice_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True, dtype=acc_dtype)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = l_slice_ref[...]
        l_slice_ref[...] = exp_m_diff * l_prev + p_rowsum
        return p, v, exp_m_diff

    def flash_attention_step2_pv(p, v, exp_m_diff, o_slice_ref):
        pv = jnp.matmul(p, v, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale
        pv = pv.astype(acc_dtype)
        o_prev = o_slice_ref[...]
        o_slice_ref[...] = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv

    def _async_copy(src, dst, sem, wait: bool):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _update_kv_cache(seq_i, bkv_sem_idx, offset, update_sz, *, wait: bool = False):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        page_indices_offset = seq_i * pages_per_seq + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )

        def loop_body(i, states):
            rem_sz, cur_ignore = states
            sz = jnp.minimum(page_size - cur_ignore, rem_sz)
            dst_ref = (
                updated_kv_cache_hbm_ref.at[
                    page_indices_ref[page_indices_offset + i],
                    pl.ds(cur_ignore, sz),
                ]
                if interpret_mode
                else cache_hbm_ref.at[
                    pl.ds(
                        page_indices_ref[page_indices_offset + i] * page_size
                        + cur_ignore,
                        sz,
                    )
                ]
            )
            _async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * page_size + cur_ignore, sz)],
                dst_ref,
                sem,
                wait,
            )
            return rem_sz - sz, 0

        if not wait or interpret_mode:
            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),
                unroll=False,
            )
        else:
            dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def start_update_kv_cache(seq_i, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_i
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_i, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            upd_seq = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(upd_seq, bkv_sem_idx, offset, update_sz, wait=True)

    def _fetch_bkv(seq_i, bkv_idx, bkv_sem_idx, *, wait: bool = False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )
        seq_kv_len = kv_lens_ref[seq_i]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        seq_q_start = cu_q_lens_ref[seq_i]
        seq_q_end = cu_q_lens_ref[seq_i + 1]
        seq_q_len = seq_q_end - seq_q_start

        kv_left = seq_kv_len - kv_len_start
        kv_left_frm_cache = (
            jnp.maximum(kv_left - seq_q_len, 0) if update_kv_cache else kv_left
        )
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        page_indices_offset = seq_i * pages_per_seq + kv_p_start

        if not wait:
            wait_update_kv_cache(bkv_sem_idx)
            for i in range(bkv_p):
                sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                src_ref = (
                    kv_cache_hbm_ref.at[page_indices_ref[page_idx], pl.ds(0, sz)]
                    if interpret_mode
                    else cache_hbm_ref.at[
                        pl.ds(page_indices_ref[page_idx] * page_size, sz)
                    ]
                )
                _async_copy(
                    src_ref,
                    vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait=False,
                )
            new_kv_len_start = seq_q_end - kv_left_frm_new
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait=False,
            )
        else:
            dst = vmem_ref.at[pl.ds(0, bkv_sz_frm_cache + bkv_sz_frm_new)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _fetch_bq(seq_i, bq_idx, bq_sem_idx, *, wait: bool = False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_i] + bq_idx * bq_sz
        seq_q_end = cu_q_lens_ref[seq_i + 1]
        sz = jnp.minimum(bq_sz, seq_q_end - q_len_start)
        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_i, bo_idx, bo_sem_idx, *, wait: bool = False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_i] + bo_idx * bq_sz
        seq_q_end = cu_q_lens_ref[seq_i + 1]
        sz = jnp.minimum(bq_sz, seq_q_end - q_len_start)
        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_send_bo(seq_i, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_i
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_i, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    strided_load = _strided_load
    strided_store = _strided_store

    def load_bq(bq_sem_idx, kv_head_idx, start, sz):
        if interpret_mode:
            tile = bq_x2_ref[bq_sem_idx, kv_head_idx, pl.ds(start, sz)]
            return tile.reshape(sz * num_q_heads_per_kv_head, head_dim)
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
        )
        start *= num_q_heads_per_kv_head_per_packing
        sz *= num_q_heads_per_kv_head_per_packing
        return strided_load(q_ref, start, sz, 1, dtype=q_dtype)

    def load_bkv(bkv_sem_idx, kv_head_idx, start, sz):
        if interpret_mode:
            raw = bkv_x2_ref[
                bkv_sem_idx, pl.ds(start, sz), :num_kv_heads_x2_per_kv_packing
            ].reshape(sz, num_kv_heads_x2_per_kv_packing * kv_packing, head_dim)
            return raw[:, 2 * kv_head_idx, :], raw[:, 2 * kv_head_idx + 1, :]

        start *= bkv_stride
        sz *= bkv_stride
        step = bkv_stride
        kv_ref = (
            bkv_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx]
            .reshape(bkv_sz * step, head_dim)
        )
        if kv_packing == 1:
            start += kv_head_idx * 2
            k = strided_load(kv_ref, start, sz, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, sz, step, dtype=kv_dtype)
            return pltpu.bitcast(k, kv_dtype), pltpu.bitcast(v, kv_dtype)

        num_kv_per_load = kv_packing // 2
        offset = kv_head_idx // num_kv_per_load
        kv_idx_in_load = kv_head_idx % num_kv_per_load
        kv = strided_load(kv_ref, start + offset, sz, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k = kv >> (kv_idx_in_load * 2 * bitwidth)
        v = k >> bitwidth
        return (
            pltpu.bitcast(k.astype(repack_ty), kv_dtype),
            pltpu.bitcast(v.astype(repack_ty), kv_dtype),
        )

    def process(cur_static_q_len=None):
        if cur_static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, cur_static_q_len)
            num_bq = cdiv(cur_static_q_len, actual_bq_sz)

        actual_bq_csz = min(bq_csz, actual_bq_sz)

        def get_next_bq_ids(cur_seq, cur_bq, cur_sem):
            next_bq = cur_bq + 1
            is_last_bq = next_bq == num_bq
            return (
                lax.select(is_last_bq, cur_seq + 1, cur_seq),
                lax.select(is_last_bq, 0, next_bq),
                lax.select(cur_sem == 0, 1, 0),
            )

        def get_next_bkv_ids(cur_seq, cur_bq, cur_bkv, cur_sem, *, num_bkv):
            next_bkv = cur_bkv + 1
            is_last_bkv = next_bkv == num_bkv
            next_bq = lax.select(is_last_bkv, cur_bq + 1, cur_bq)
            is_last_bq = next_bq == num_bq
            next_bq = lax.select(is_last_bq, 0, next_bq)
            next_seq = lax.select(is_last_bq, cur_seq + 1, cur_seq)
            next_sem = lax.select(cur_sem == 0, 1, 0)

            next_bq_start_bkv = 0
            if sliding_window is not None:
                next_bq_start_bkv = (
                    jnp.maximum(
                        kv_q_gap + (cur_bq + 1) * actual_bq_sz - sliding_window, 0
                    )
                    // bkv_sz
                )
            next_bkv = lax.select(is_last_bkv, next_bq_start_bkv, next_bkv)
            next_bkv = lax.select(is_last_bq, next_seq_start_bkv_idx, next_bkv)
            return next_seq, next_bq, next_bkv, next_sem

        @pl.loop(0, num_bq, unroll=False)
        def compute_with_bq(bq_idx):
            l_ref[...] = jnp.full_like(l_ref, 0.0)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            start_bkv_idx = 0
            if sliding_window is not None:
                start_bkv_idx = (
                    jnp.maximum(processed_q_len - sliding_window, 0) // bkv_sz
                )
            effective_kv_len = (
                jnp.minimum(kv_len, processed_q_len + actual_bq_sz)
                if use_causal_mask
                else kv_len
            )
            end_bkv_idx = cdiv(effective_kv_len, bkv_sz)

            @pl.when(next_seq_idx < end_seq_idx)
            def _prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                _fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
            def compute_with_bkv(bkv_idx):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_bkv, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx
                )
                processed_kv_len = bkv_idx * bkv_sz

                @pl.when(next_seq_bkv < end_seq_idx)
                def _prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    _fetch_bkv(next_seq_bkv, next_bkv_idx, next_bkv_sem_idx)

                @pl.when(bkv_idx == start_bkv_idx)
                def _wait_cur_bq():
                    _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

                offset, update_sz = _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

                if update_kv_cache:

                    @pl.when(jnp.logical_and(update_sz > 0, bq_idx == num_bq - 1))
                    def _update_cur_bkv():
                        start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                def attention_loop(idx):
                    prev_lm_slice = None
                    prev_p = None
                    prev_v = None
                    prev_exp_m_diff = None
                    bkv_start = idx * bkv_csz

                    for bq_start in range(0, actual_bq_sz, actual_bq_csz):
                        for kv_head_idx in range(actual_num_kv_heads):
                            bk_c, bv_c = load_bkv(
                                bkv_sem_idx, kv_head_idx, bkv_start, bkv_csz
                            )
                            bq_c = load_bq(
                                bq_sem_idx, kv_head_idx, bq_start, actual_bq_csz
                            )
                            lm_slice = (
                                kv_head_idx,
                                pl.ds(
                                    bq_start * num_q_heads_per_kv_head,
                                    actual_bq_csz * num_q_heads_per_kv_head,
                                ),
                            )
                            cur_p, cur_v, cur_exp_m_diff = (
                                flash_attention_step1_qk_softmax(
                                    bq_c,
                                    bk_c,
                                    bv_c,
                                    l_ref.at[*lm_slice],
                                    m_ref.at[*lm_slice],
                                    processed_q_len=processed_q_len + bq_start,
                                    processed_kv_len=processed_kv_len + bkv_start,
                                    effective_kv_len=effective_kv_len,
                                )
                            )
                            if prev_lm_slice is not None:
                                flash_attention_step2_pv(
                                    prev_p,
                                    prev_v,
                                    prev_exp_m_diff,
                                    acc_ref.at[*prev_lm_slice],
                                )
                            prev_lm_slice = lm_slice
                            prev_p = cur_p
                            prev_v = cur_v
                            prev_exp_m_diff = cur_exp_m_diff

                    assert prev_lm_slice is not None
                    flash_attention_step2_pv(
                        prev_p, prev_v, prev_exp_m_diff, acc_ref.at[*prev_lm_slice]
                    )

                if bkv_csz == bkv_sz:
                    attention_loop(jnp.int32(0))
                else:
                    effective_bkv_sz = jnp.maximum(
                        jnp.minimum(effective_kv_len - bkv_idx * bkv_sz, bkv_sz), 0
                    )
                    num_loops = cdiv(effective_bkv_sz, bkv_csz)
                    pl.loop(0, num_loops, unroll=False)(attention_loop)

            acc = acc_ref[...]
            l_denom = broadcast_minor(l_ref[...], acc.shape)
            out = (
                acc * pltpu.reciprocal(l_denom, approx=True)
                if not interpret_mode
                and l_denom.dtype == jnp.float32
                and q_dtype != jnp.float32
                else lax.div(acc, l_denom)
            ).astype(q_dtype)

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            if interpret_mode:
                bo_x2_ref[bo_sem_idx] = out.reshape(bo_x2_ref.shape[1:])
            else:
                out_ref = (
                    bo_x2_ref.at[bo_sem_idx]
                    .bitcast(jnp.int32)
                    .reshape(
                        actual_num_kv_heads
                        * bq_sz
                        * num_q_heads_per_kv_head_per_packing,
                        head_dim,
                    )
                )
                out_packed = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
                strided_store(out_ref, 0, out_ref.shape[0], 1, out_packed)
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    @pl.when(seq_idx == start_seq_idx)
    def _prologue():
        _fetch_bq(seq_i=start_seq_idx, bq_idx=0, bq_sem_idx=0)
        _fetch_bkv(seq_i=start_seq_idx, bkv_idx=cur_seq_start_bkv_idx, bkv_sem_idx=0)

    @pl.when(jnp.logical_and(start_seq_idx <= seq_idx, seq_idx < end_seq_idx))
    def _pipeline():
        process(cur_static_q_len=static_q_len)

    @pl.when(seq_idx == end_seq_idx - 1)
    def _epilogue():
        for i in range(2):
            wait_send_bo(bo_sem_idx=i)
            wait_update_kv_cache(bkv_sem_idx=i)


# ---------------------------------------------------------------------------
# Batched RPA decode (triple-buffered manual DMA + SMEM task schedule)
# ---------------------------------------------------------------------------

_BATCHED_NUM_INPUT_SLOTS = 3
_BATCHED_NUM_OUTPUT_SLOTS = 2
_DMA_CHUNK_BYTES = _TPU_NUM_LANES * 4
_LOG2_E = math.log2(math.e)


def _unpack_kv_words(
    val: jax.Array, *, target_bitwidth: int, kv_dtype: Any
) -> list[tuple[jax.Array, jax.Array]]:
    curr_dtype = val.dtype
    curr_bitwidth = get_dtype_bitwidth(curr_dtype)
    next_bitwidth = curr_bitwidth // 2
    next_dtype = jnp.dtype(f"uint{next_bitwidth}")
    left = val.astype(next_dtype)
    val_u32 = pltpu.bitcast(val, jnp.uint32)
    right = pltpu.bitcast(val_u32 >> next_bitwidth, curr_dtype).astype(next_dtype)
    if next_bitwidth == target_bitwidth:
        return [(pltpu.bitcast(left, kv_dtype), pltpu.bitcast(right, kv_dtype))]
    return _unpack_kv_words(
        left, target_bitwidth=target_bitwidth, kv_dtype=kv_dtype
    ) + _unpack_kv_words(right, target_bitwidth=target_bitwidth, kv_dtype=kv_dtype)


@dataclass(frozen=True, eq=True, kw_only=True)
class _BatchedRpaConfigMixin:
    """Shared geometry and memory-sizing helpers for batched decode and mixed RPA."""

    batch_size: int
    bkv_sz: int
    page_size: int
    pages_per_seq: int
    max_num_seqs: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype_name: str
    kv_dtype_name: str
    sm_scale: float
    mask_value: float
    sliding_window: int | None = None
    soft_cap: float | None = None
    q_scale: float | None = None
    k_scale: float | None = None
    v_scale: float | None = None
    update_kv_cache: bool = True
    vmem_limit_bytes: int = 64 * 1024 * 1024
    smem_capacity_bytes: int = 1024 * 1024
    disable_bounds_checks: bool = True
    disable_semaphore_checks: bool = True
    interpret_mode: bool = False

    @property
    def bq_tokens_per_block(self) -> int:
        return 1

    @property
    def scratch_dtype(self) -> jnp.dtype:
        return jnp.float32

    @property
    def dtype_q(self) -> jnp.dtype:
        return jnp.dtype(self.q_dtype_name)

    @property
    def dtype_kv(self) -> jnp.dtype:
        return jnp.dtype(self.kv_dtype_name)

    @property
    def packing_q(self) -> int:
        return get_dtype_packing(self.dtype_q)

    @property
    def packing_kv(self) -> int:
        return get_dtype_packing(self.dtype_kv)

    @property
    def num_q_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def aligned_num_q_heads_per_kv_head(self) -> int:
        return align_to(self.num_q_heads_per_kv_head, self.packing_q)

    @property
    def aligned_head_dim(self) -> int:
        return align_to(self.head_dim, _TPU_NUM_LANES)

    @property
    def kv_hbm_stride(self) -> int:
        return align_to(self.num_kv_heads * 2, self.packing_kv) // self.packing_kv

    @property
    def bkv_stride(self) -> int:
        stride = self.kv_hbm_stride
        if has_bank_conflicts(stride):
            stride += 1
        return stride

    @property
    def bkv_p(self) -> int:
        return self.bkv_sz // self.page_size

    @property
    def num_page_indices(self) -> int:
        return self.max_num_seqs * self.pages_per_seq

    @property
    def kv_bytes_per_token(self) -> int:
        return (
            self.kv_hbm_stride
            * self.packing_kv
            * self.aligned_head_dim
            * self.dtype_kv.itemsize
        )

    @property
    def q_bytes_per_token(self) -> int:
        return (
            self.num_kv_heads
            * self.aligned_num_q_heads_per_kv_head
            * self.aligned_head_dim
            * self.dtype_q.itemsize
        )

    @property
    def q_vmem_shape(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.batch_size,
            self.num_kv_heads,
            self.bq_tokens_per_block,
            self.aligned_num_q_heads_per_kv_head // self.packing_q,
            self.packing_q,
            self.aligned_head_dim,
        )

    @property
    def kv_vmem_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.batch_size,
            self.bkv_sz,
            self.bkv_stride,
            self.packing_kv,
            self.aligned_head_dim,
        )

    @property
    def lm_scratch_shape(self) -> tuple[int, int, int]:
        return (
            self.num_kv_heads,
            self.bq_tokens_per_block * self.aligned_num_q_heads_per_kv_head,
            _TPU_NUM_LANES,
        )

    @property
    def acc_scratch_shape(self) -> tuple[int, int, int]:
        return (
            self.num_kv_heads,
            self.bq_tokens_per_block * self.aligned_num_q_heads_per_kv_head,
            self.aligned_head_dim,
        )


@dataclass(frozen=True, eq=True, kw_only=True)
class BatchedDecodeConfig(_BatchedRpaConfigMixin):
    """Static configuration for the triple-buffered batched decode RPA kernel."""

    bf16_qk_softmax: bool = True
    bf16_online_scratch: bool = False

    @property
    def use_bf16_qk_softmax(self) -> bool:
        return self.bf16_qk_softmax and self.dtype_q == jnp.bfloat16

    @property
    def scratch_dtype(self) -> jnp.dtype:
        if self.use_bf16_qk_softmax and self.bf16_online_scratch:
            return jnp.bfloat16
        return jnp.float32

    @property
    def max_steps_ub(self) -> int:
        fixed_words = (
            self.max_num_seqs + (self.max_num_seqs + 1) + self.num_page_indices + 3 + 1
        )
        available_bytes = (
            int((self.smem_capacity_bytes - 32 * 1024) * 0.40) - fixed_words * 4
        )
        # Per step per batch item: s_idx, k_idx, is_last_k, do_writeback (4),
        # dma_q (2), dma_kv_cache (3 * bkv_p), dma_kv_new (5); plus 4 wait words.
        bytes_per_step = (11 + 3 * self.bkv_p) * 4 * self.batch_size + 16
        raw_steps = max(_TPU_NUM_LANES, available_bytes // bytes_per_step)
        return max(_TPU_NUM_LANES, (raw_steps // _TPU_NUM_LANES) * _TPU_NUM_LANES)

    @property
    def schedule_hbm_steps(self) -> int:
        max_k_per_seq = max(
            1, int(cdiv(self.pages_per_seq * self.page_size, self.bkv_sz))
        )
        max_total_steps = max(
            1, int(cdiv(self.max_num_seqs * max_k_per_seq, self.batch_size))
        )
        return max(self.max_steps_ub, align_to(max_total_steps, self.max_steps_ub))

    @property
    def kernel_name(self) -> str:
        name = f"RPAd-snh-p{self.page_size}-b{self.batch_size}-q1-k{self.bkv_sz}"
        if not self.use_bf16_qk_softmax:
            name += "-f32p"
        elif self.bf16_online_scratch:
            name += "-bf16s"
        if self.sliding_window is not None:
            name += f"-sw{self.sliding_window}"
        return name


class _ScheduleSpecMixin:
    def scratch_shapes(self) -> Any:
        return jax.tree.map(lambda x: pltpu.SMEM(x.shape, x.dtype), self)

    def in_specs(self) -> Any:
        return jax.tree.map(
            lambda x: (
                pl.BlockSpec(memory_space=pltpu.SMEM)
                if x.size == 1
                else pl.BlockSpec(memory_space=pltpu.HBM)
            ),
            self,
        )

    def out_specs(self) -> Any:
        return jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), self)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _SmemArray:
    data: Any
    shape: tuple[int, ...] = field(metadata={"static": True})

    @classmethod
    def create_shape_dtype(cls, shape: tuple[int, ...]) -> _SmemArray:
        return cls(
            data=jax.ShapeDtypeStruct((math.prod(shape),), jnp.int32), shape=shape
        )

    def _pos(self, indices: Any) -> Any:
        if not isinstance(indices, tuple):
            indices = (indices,)
        strides = pl.strides_from_shape(self.shape)
        pos: Any = 0
        for stride, idx in zip(strides, indices, strict=True):
            pos = pos + stride * idx
        return pos

    def __getitem__(self, indices: Any) -> Any:
        return self.data[self._pos(indices)]

    def __setitem__(self, indices, value: Any) -> None:
        self.data[self._pos(indices)] = value


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BatchedDecodeSchedule(_ScheduleSpecMixin):
    """Flat SMEM/HBM schedule for batched decode RPA."""

    s_idx: _SmemArray  # [steps, B]
    k_idx: _SmemArray  # [steps, B]
    is_last_k: _SmemArray  # [steps, B]
    do_writeback: _SmemArray  # [steps, B]
    dma_q: _SmemArray  # [steps, B, 2] -> (q_src, q_sz)
    dma_kv_cache: _SmemArray  # [steps, B, bkv_p, 3] -> (p_slot, dst_vmem, sz)
    dma_kv_new: (
        _SmemArray  # [steps, B, 5] -> (wb_p_slot, wb_p_off, fetch_hbm, vmem_off, sz)
    )
    total_wait_kv_in: _SmemArray  # [steps]
    total_wait_kv_out: _SmemArray  # [steps]
    total_wait_q_in: _SmemArray  # [steps]
    total_wait_o_out: _SmemArray  # [steps]
    actual_steps: Any  # [1]

    @classmethod
    def create_shape_dtype(
        cls, cfgs: BatchedDecodeConfig, *, steps: int
    ) -> BatchedDecodeSchedule:
        idx_arr = _SmemArray.create_shape_dtype((steps, cfgs.batch_size))
        step_arr = _SmemArray.create_shape_dtype((steps,))
        return cls(
            s_idx=idx_arr,
            k_idx=idx_arr,
            is_last_k=idx_arr,
            do_writeback=idx_arr,
            dma_q=_SmemArray.create_shape_dtype((steps, cfgs.batch_size, 2)),
            dma_kv_cache=_SmemArray.create_shape_dtype(
                (steps, cfgs.batch_size, cfgs.bkv_p, 3)
            ),
            dma_kv_new=_SmemArray.create_shape_dtype((steps, cfgs.batch_size, 5)),
            total_wait_kv_in=step_arr,
            total_wait_kv_out=step_arr,
            total_wait_q_in=step_arr,
            total_wait_o_out=step_arr,
            actual_steps=jax.ShapeDtypeStruct((1,), jnp.int32),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _SchedCarry:
    hbm_offset: jax.Array | int
    count: jax.Array | int


def _batched_decode_schedule_kernel(
    kv_lens_ref: Any,
    cu_q_lens_ref: Any,
    distribution_ref: Any,
    schedule_hbm_ref: BatchedDecodeSchedule,
    schedule_ref: BatchedDecodeSchedule,
    dma_sem: Any,
    *,
    cfgs: BatchedDecodeConfig,
) -> None:
    schedule_ref.actual_steps[0] = 0

    def _mask_entry(step: Any, b_idx: Any) -> None:
        schedule_ref.s_idx[step, b_idx] = -1
        schedule_ref.k_idx[step, b_idx] = 0
        schedule_ref.is_last_k[step, b_idx] = 0
        schedule_ref.do_writeback[step, b_idx] = 0
        schedule_ref.dma_q[step, b_idx, 0] = 0
        schedule_ref.dma_q[step, b_idx, 1] = 0
        for i in range(cfgs.bkv_p):
            schedule_ref.dma_kv_cache[step, b_idx, i, 0] = 0
            schedule_ref.dma_kv_cache[step, b_idx, i, 1] = 0
            schedule_ref.dma_kv_cache[step, b_idx, i, 2] = 0
        for f in range(5):
            schedule_ref.dma_kv_new[step, b_idx, f] = 0
        schedule_ref.total_wait_kv_in[step] = 0
        schedule_ref.total_wait_kv_out[step] = 0
        schedule_ref.total_wait_q_in[step] = 0
        schedule_ref.total_wait_o_out[step] = 0

    def _flush(count: Any, hbm_offset: Any) -> _SchedCarry:
        last_task = align_to(count, cfgs.batch_size)

        @pl.loop(count, last_task)
        def _mask_tail(idx: Any) -> None:
            step, b_idx = divmod(idx, cfgs.batch_size)
            _mask_entry(step, b_idx)

        def _compute_step_waits(step: Any, _: None) -> None:
            kv_in_tokens: Any = 0
            kv_out_tokens: Any = 0
            q_in_tokens: Any = 0
            o_out_tokens: Any = 0
            for b in range(cfgs.batch_size):
                for i in range(cfgs.bkv_p):
                    kv_in_tokens = (
                        kv_in_tokens + schedule_ref.dma_kv_cache[step, b, i, 2]
                    )
                new_sz = schedule_ref.dma_kv_new[step, b, 4]
                kv_in_tokens = kv_in_tokens + new_sz
                do_wb = schedule_ref.do_writeback[step, b] == 1
                kv_out_tokens = kv_out_tokens + jnp.where(do_wb, new_sz, 0)
                q_sz = schedule_ref.dma_q[step, b, 1]
                q_in_tokens = q_in_tokens + q_sz
                is_last = schedule_ref.is_last_k[step, b] == 1
                o_out_tokens = o_out_tokens + jnp.where(is_last, q_sz, 0)

            schedule_ref.total_wait_kv_in[step] = (
                kv_in_tokens * cfgs.kv_bytes_per_token
            ) // _DMA_CHUNK_BYTES
            schedule_ref.total_wait_kv_out[step] = (
                kv_out_tokens * cfgs.kv_bytes_per_token
            ) // _DMA_CHUNK_BYTES
            schedule_ref.total_wait_q_in[step] = (
                q_in_tokens * cfgs.q_bytes_per_token
            ) // _DMA_CHUNK_BYTES
            schedule_ref.total_wait_o_out[step] = (
                o_out_tokens * cfgs.q_bytes_per_token
            ) // _DMA_CHUNK_BYTES

        lax.fori_loop(0, last_task // cfgs.batch_size, _compute_step_waits, None)

        hbm_offset_aligned = pl.multiple_of(hbm_offset, _TPU_NUM_LANES)
        flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
        flat_smem = jax.tree_util.tree_leaves(schedule_ref)
        copies = []
        for h, s in zip(flat_hbm, flat_smem, strict=True):
            if h.shape[0] > 1:
                elem_sz = s.shape[0] // cfgs.max_steps_ub
                write_sz = cfgs.max_steps_ub * elem_sz
                out_off = hbm_offset_aligned * elem_sz
                copies.append(
                    pltpu.make_async_copy(
                        s.at[pl.ds(0, write_sz)],
                        h.at[pl.ds(out_off, write_sz)],
                        dma_sem.at[0],
                    )
                )
        for cp in copies:
            cp.start()
        for cp in copies:
            cp.wait()
        return _SchedCarry(hbm_offset + cfgs.max_steps_ub, 0)

    def _seq_body(s_idx: Any, carry: _SchedCarry) -> _SchedCarry:
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        q_sz_task = jnp.clip(q_end - q_start, 0, 1)
        kv_len = kv_lens_ref[s_idx]
        kv_cache_len = (
            jnp.maximum(kv_len - q_sz_task, 0) if cfgs.update_kv_cache else kv_len
        )
        q_offset = jnp.maximum(kv_len - q_sz_task, 0)

        has_work = jnp.logical_and(q_sz_task > 0, kv_len > 0)
        end_k_idx = jnp.where(has_work, cdiv(kv_len, cfgs.bkv_sz), 0)
        start_k_idx: Any = 0
        if cfgs.sliding_window is not None:
            sw_start = jnp.maximum(0, q_offset - cfgs.sliding_window + 1)
            start_k_idx = jnp.where(has_work, sw_start // cfgs.bkv_sz, 0)

        def _k_body(k_idx: Any, k_carry: _SchedCarry) -> _SchedCarry:
            count = k_carry.count
            step, lane = divmod(count, cfgs.batch_size)
            schedule_ref.s_idx[step, lane] = s_idx
            schedule_ref.k_idx[step, lane] = k_idx
            schedule_ref.is_last_k[step, lane] = jnp.where(k_idx == end_k_idx - 1, 1, 0)
            schedule_ref.dma_q[step, lane, 0] = q_start
            schedule_ref.dma_q[step, lane, 1] = q_sz_task

            kv_len_start = k_idx * cfgs.bkv_sz
            kv_p_start = k_idx * cfgs.bkv_p
            kv_left = kv_len - kv_len_start
            kv_left_cache = (
                jnp.maximum(kv_cache_len - kv_len_start, 0)
                if cfgs.update_kv_cache
                else kv_left
            )
            p_base = s_idx * cfgs.pages_per_seq + kv_p_start

            for i in range(cfgs.bkv_p):
                dst_vmem = i * cfgs.page_size
                dma_sz = jnp.clip(kv_left_cache - dst_vmem, 0, cfgs.page_size)
                p_slot = jnp.minimum(p_base + i, cfgs.num_page_indices - 1)
                schedule_ref.dma_kv_cache[step, lane, i, 0] = p_slot
                schedule_ref.dma_kv_cache[step, lane, i, 1] = dst_vmem
                schedule_ref.dma_kv_cache[step, lane, i, 2] = dma_sz

            bkv_sz_cache = jnp.minimum(kv_left_cache, cfgs.bkv_sz)
            kv_left_new = kv_left - kv_left_cache
            new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_new)
            do_wb = jnp.where(new_sz > 0, 1, 0)
            schedule_ref.do_writeback[step, lane] = do_wb

            tok_idx = kv_len_start + bkv_sz_cache
            p_idx = jnp.minimum(tok_idx // cfgs.page_size, cfgs.pages_per_seq - 1)
            p_off = tok_idx % cfgs.page_size
            wb_p_slot = jnp.minimum(
                s_idx * cfgs.pages_per_seq + p_idx, cfgs.num_page_indices - 1
            )
            fetch_hbm = jnp.maximum(q_start + q_sz_task - kv_left_new, 0)
            schedule_ref.dma_kv_new[step, lane, 0] = wb_p_slot
            schedule_ref.dma_kv_new[step, lane, 1] = p_off
            schedule_ref.dma_kv_new[step, lane, 2] = fetch_hbm
            schedule_ref.dma_kv_new[step, lane, 3] = bkv_sz_cache
            schedule_ref.dma_kv_new[step, lane, 4] = new_sz

            next_count = count + 1
            return lax.cond(
                next_count < cfgs.max_steps_ub * cfgs.batch_size,
                lambda c: c,
                lambda c: _flush(c.count, c.hbm_offset),
                _SchedCarry(k_carry.hbm_offset, next_count),
            )

        return lax.fori_loop(start_k_idx, end_k_idx, _k_body, carry)

    end_seq = distribution_ref[0]
    final_carry = lax.fori_loop(0, end_seq, _seq_body, _SchedCarry(0, 0))
    total_steps = cdiv(final_carry.count, cfgs.batch_size) + final_carry.hbm_offset
    schedule_ref.actual_steps[0] = total_steps

    @pl.when(final_carry.count > 0)
    def _flush_remainder() -> None:
        _flush(final_carry.count, final_carry.hbm_offset)

    steps_cp = pltpu.make_async_copy(
        schedule_ref.actual_steps.at[pl.ds(0, 1)],
        schedule_hbm_ref.actual_steps.at[pl.ds(0, 1)],
        dma_sem.at[0],
    )
    steps_cp.start()
    steps_cp.wait()


def generate_batched_decode_schedule(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    cfgs: BatchedDecodeConfig,
) -> BatchedDecodeSchedule:
    """Build the SMEM/HBM work schedule for batched decode attention."""
    sched_smem_struct = BatchedDecodeSchedule.create_shape_dtype(
        cfgs, steps=cfgs.max_steps_ub
    )
    sched_hbm_struct = BatchedDecodeSchedule.create_shape_dtype(
        cfgs, steps=cfgs.schedule_hbm_steps
    )
    interpret_params = pltpu.InterpretParams() if cfgs.interpret_mode else False
    return pl.pallas_call(
        functools.partial(_batched_decode_schedule_kernel, cfgs=cfgs),
        out_shape=sched_hbm_struct,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[],
            out_specs=sched_hbm_struct.out_specs(),
            scratch_shapes=[
                sched_smem_struct.scratch_shapes(),
                pltpu.SemaphoreType.DMA((1,)),
            ],
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=cfgs.disable_bounds_checks,
            disable_semaphore_checks=cfgs.disable_semaphore_checks,
        ),
        interpret=interpret_params,
        name="rpa_metadata_schedule",
    )(kv_lens, cu_q_lens, distribution)


def _step_guard(
    step: int | jax.Array, total_steps: int | jax.Array
) -> tuple[jax.Array, jax.Array]:
    is_noop = jnp.logical_or(step < 0, step >= total_steps)
    block_idx = jnp.where(is_noop, 0, jnp.maximum(step, 0))
    return is_noop, block_idx


def _run_manual_triple_buffer(
    total_steps: int | jax.Array,
    compute_fn: Callable[..., None],
    *,
    copy_in_q: Callable[[int | jax.Array, int | jax.Array, int | jax.Array], None],
    wait_in_q: Callable[[int | jax.Array, int | jax.Array, int | jax.Array], None],
    copy_in_kv: Callable[[int | jax.Array, int | jax.Array, int | jax.Array], None],
    wait_in_kv: Callable[[int | jax.Array, int | jax.Array, int | jax.Array], None],
    copy_out_kv: Callable[[int | jax.Array, int | jax.Array, int | jax.Array], None],
    wait_out_kv: Callable[[int | jax.Array, int | jax.Array, int | jax.Array], None],
    copy_out_o: Callable[[int | jax.Array, int | jax.Array, int | jax.Array], None],
    wait_out_o: Callable[[int | jax.Array, int | jax.Array, int | jax.Array], None],
) -> None:
    in_slots = _BATCHED_NUM_INPUT_SLOTS
    out_slots = _BATCHED_NUM_OUTPUT_SLOTS

    @pl.when(total_steps > 0)
    def _active_pipeline() -> None:
        with jax.named_scope("ep_initialize"):
            copy_in_q(0, 0, total_steps)
            copy_in_kv(0, 0, total_steps)
            copy_in_q(1, 1, total_steps)
            copy_in_kv(1, 1, total_steps)

        @pl.loop(0, total_steps)
        def _steady(step: jax.Array) -> None:
            comp_slot = lax.rem(step, in_slots)
            comm_slot = lax.rem(step + 2, in_slots)
            out_slot = lax.rem(step, out_slots)

            with jax.named_scope("wait_in_q_kv"):
                wait_in_q(step, comp_slot, total_steps)
                wait_in_kv(step, comp_slot, total_steps)

            with jax.named_scope("wait_out_o"):
                wait_out_o(step - 2, out_slot, total_steps)

            with jax.named_scope("ep_run_kernel"):

                def start_next_fetch() -> None:
                    copy_in_q(step + 2, comm_slot, total_steps)
                    copy_in_kv(step + 2, comm_slot, total_steps)

                def start_kv_writeback() -> None:
                    copy_out_kv(step, comp_slot, total_steps)

                compute_fn(
                    step,
                    comp_slot,
                    out_slot,
                    start_next_fetch,
                    start_kv_writeback,
                )

                with jax.named_scope("wait_out_kv"):
                    wait_out_kv(step, comp_slot, total_steps)

                copy_out_o(step, out_slot, total_steps)

        with jax.named_scope("epilogue"):
            wait_out_o(
                total_steps - 2,
                lax.rem(total_steps, out_slots),
                total_steps,
            )
            wait_out_o(
                total_steps - 1,
                lax.rem(total_steps + 1, out_slots),
                total_steps,
            )


def _load_batched_q(
    q_vref: Any,
    *,
    batch_size: int,
    num_kv_heads: int,
    bq_sz: int,
    tq_per_tok: int,
    packing_q: int,
    dtype_q: Any,
    aligned_head_dim: int,
    interpret_mode: bool,
) -> jax.Array:
    total_tq = bq_sz * tq_per_tok
    if interpret_mode:
        return q_vref[...].reshape(
            batch_size,
            num_kv_heads,
            total_tq,
            aligned_head_dim,
        )
    q_p = tq_per_tok // packing_q
    q_ref = q_vref.bitcast(jnp.uint32).reshape(-1, aligned_head_dim)
    q_loaded = _strided_load(
        q_ref,
        0,
        batch_size * num_kv_heads * bq_sz * q_p,
        1,
        dtype=dtype_q,
    )
    return q_loaded.reshape(
        batch_size,
        num_kv_heads,
        total_tq,
        aligned_head_dim,
    )


def _load_batched_kv(
    kv_in_vref: Any,
    *,
    batch_size: int,
    num_kv_heads: int,
    bkv_sz: int,
    kv_hbm_stride: int,
    bkv_stride: int,
    packing_kv: int,
    dtype_kv: Any,
    aligned_head_dim: int,
    interpret_mode: bool,
) -> tuple[jax.Array, jax.Array]:
    k_b: list[jax.Array] = []
    v_b: list[jax.Array] = []
    if interpret_mode:
        for b_idx in range(batch_size):
            raw = kv_in_vref[b_idx, :, :kv_hbm_stride].reshape(
                bkv_sz,
                kv_hbm_stride * packing_kv,
                aligned_head_dim,
            )
            k_heads = raw[:, 0 : 2 * num_kv_heads : 2, :].swapaxes(0, 1)
            v_heads = raw[:, 1 : 2 * num_kv_heads : 2, :].swapaxes(0, 1)
            k_b.append(k_heads)
            v_b.append(v_heads)
    else:
        heads_per_load = max(1, packing_kv // 2)
        bitwidth = get_dtype_bitwidth(dtype_kv)
        for b_idx in range(batch_size):
            kv_u32_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32)
            kv_ref = pltpu.annotate(
                kv_u32_ref.reshape(-1, aligned_head_dim),
                no_hazard_no_deps=True,
            )
            ks: list[jax.Array] = []
            vs: list[jax.Array] = []
            for kv_head_start in range(0, num_kv_heads, heads_per_load):
                word_start = (kv_head_start * 2) // packing_kv
                if packing_kv == 1:
                    k_h = _strided_load(
                        kv_ref,
                        word_start,
                        bkv_sz * bkv_stride,
                        bkv_stride,
                        dtype=dtype_kv,
                    )
                    v_h = _strided_load(
                        kv_ref,
                        word_start + 1,
                        bkv_sz * bkv_stride,
                        bkv_stride,
                        dtype=dtype_kv,
                    )
                    pairs = [(k_h, v_h)]
                else:
                    kv_words = _strided_load(
                        kv_ref,
                        word_start,
                        bkv_sz * bkv_stride,
                        bkv_stride,
                    )
                    pairs = _unpack_kv_words(
                        kv_words,
                        target_bitwidth=bitwidth,
                        kv_dtype=dtype_kv,
                    )
                ks.append(jnp.stack([p_k for p_k, _ in pairs], axis=0))
                vs.append(jnp.stack([p_v for _, p_v in pairs], axis=0))
            k_cat = jnp.concat(ks, axis=0).reshape(-1, bkv_sz, aligned_head_dim)[
                :num_kv_heads
            ]
            v_cat = jnp.concat(vs, axis=0).reshape(-1, bkv_sz, aligned_head_dim)[
                :num_kv_heads
            ]
            k_b.append(k_cat)
            v_b.append(v_cat)

    return jnp.stack(k_b, axis=0), jnp.stack(v_b, axis=0)


def _batched_decode_rpa_body(
    step: jax.Array,
    q_vref: Any,
    kv_in_vref: Any,
    o_vref: Any,
    schedule_ref: BatchedDecodeSchedule,
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    acc_scratch_ref: Any,
    *,
    kv_lens_ref: Any,
    cfgs: BatchedDecodeConfig,
    start_next_fetch: Callable[[], None],
    start_kv_writeback: Callable[[], None],
) -> None:
    if not cfgs.interpret_mode:
        kv_in_vref = pltpu.annotate(kv_in_vref, no_hazard_no_deps=True)

    always_true = kv_lens_ref[0] >= 0

    @pl.when(always_true)
    @jax.named_scope("stitch_and_issue_writeback")
    def _kv_store_fence() -> None:
        start_kv_writeback()

    start_next_fetch()

    tq = cfgs.aligned_num_q_heads_per_kv_head
    tot_q = cfgs.num_kv_heads * tq
    kv_iota = lax.broadcasted_iota(jnp.int32, (tot_q, cfgs.bkv_sz), 1)
    kv_vpu_iota = (
        lax.broadcasted_iota(
            jnp.int32,
            (cfgs.num_kv_heads, cfgs.bkv_sz, cfgs.aligned_head_dim),
            1,
        )
        if cfgs.sliding_window is not None
        else None
    )

    masks: list[jax.Array] = []
    valid_kv_masks: list[jax.Array] = []
    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        safe_s = jnp.maximum(s_idx, 0)
        k_idx = schedule_ref.k_idx[step, b_idx]
        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        kv_len_val = jnp.where(is_valid, kv_lens_ref[safe_s], 0)
        # Keep column 0 unmasked on unused tail lanes (kv_limit = 1) using scalar
        # int32 select so Mosaic never emits an i1 vector select_n.
        kv_limit = jnp.where(is_valid, kv_len_val - k_id, 1)
        mask_b = kv_iota < kv_limit
        if cfgs.sliding_window is not None:
            assert kv_vpu_iota is not None
            sw_low = jnp.where(
                is_valid, kv_len_val - 1 - k_id - cfgs.sliding_window, -1
            )
            mask_b = jnp.logical_and(mask_b, kv_iota > sw_low)
            vpu_limit = jnp.where(is_valid, kv_len_val - k_id, 0)
            valid_kv_masks.append(
                jnp.logical_and(kv_vpu_iota < vpu_limit, kv_vpu_iota > sw_low)
            )
        masks.append(mask_b)

    q = _load_batched_q(
        q_vref,
        batch_size=cfgs.batch_size,
        num_kv_heads=cfgs.num_kv_heads,
        bq_sz=1,
        tq_per_tok=tq,
        packing_q=cfgs.packing_q,
        dtype_q=cfgs.dtype_q,
        aligned_head_dim=cfgs.aligned_head_dim,
        interpret_mode=cfgs.interpret_mode,
    )
    k, v = _load_batched_kv(
        kv_in_vref,
        batch_size=cfgs.batch_size,
        num_kv_heads=cfgs.num_kv_heads,
        bkv_sz=cfgs.bkv_sz,
        kv_hbm_stride=cfgs.kv_hbm_stride,
        bkv_stride=cfgs.bkv_stride,
        packing_kv=cfgs.packing_kv,
        dtype_kv=cfgs.dtype_kv,
        aligned_head_dim=cfgs.aligned_head_dim,
        interpret_mode=cfgs.interpret_mode,
    )

    if cfgs.sliding_window is not None:
        valid_kv = jnp.stack(valid_kv_masks, axis=0)
        k = jnp.where(valid_kv, k, jnp.zeros_like(k))
        v = jnp.where(valid_kv, v, jnp.zeros_like(v))

    if cfgs.q_scale is not None:
        q_scaled = q / cfgs.q_scale
        if jnp.issubdtype(k.dtype, jnp.floating):
            finfo = jnp.finfo(k.dtype)
            q_scaled = jnp.clip(q_scaled, min=float(finfo.min), max=float(finfo.max))
        q_dot = q_scaled.astype(k.dtype)
    else:
        q_dot = q

    qk = lax.dot_general(
        q_dot.reshape(-1, tq, cfgs.aligned_head_dim),
        k.reshape(-1, cfgs.bkv_sz, cfgs.aligned_head_dim),
        dimension_numbers=(([2], [2]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).reshape(cfgs.batch_size, tot_q, cfgs.bkv_sz)

    s_scale = cfgs.sm_scale
    if cfgs.q_scale is not None:
        s_scale *= cfgs.q_scale
    if cfgs.k_scale is not None:
        s_scale *= cfgs.k_scale

    use_bf16_qkp = cfgs.use_bf16_qk_softmax
    acc_dt = cfgs.scratch_dtype
    if cfgs.soft_cap is not None:
        qk = cfgs.soft_cap * jnp.tanh(qk * (s_scale / cfgs.soft_cap))
        if use_bf16_qkp:
            qk = (qk * _LOG2_E).astype(jnp.bfloat16)
    elif use_bf16_qkp:
        qk = (qk * (s_scale * _LOG2_E)).astype(jnp.bfloat16)
    else:
        qk = qk * s_scale

    if use_bf16_qkp:
        bf16_floor = -0.7 * float(jnp.finfo(jnp.bfloat16).max)
        mask_val = (
            max(cfgs.mask_value * _LOG2_E, bf16_floor)
            if math.isfinite(cfgs.mask_value)
            else bf16_floor
        )
        mask_fill: Any = jnp.asarray(mask_val, dtype=jnp.bfloat16)
        exp_fn = lax.exp2
    else:
        mask_fill = jnp.asarray(cfgs.mask_value, dtype=qk.dtype)
        exp_fn = jnp.exp

    # Finite mask_fill guarantees finite m_next even on all-masked tiles; on the
    # first tile after is_last_k == 1, m_prev == -inf gives alpha == 0.0 and
    # zeros carried-over finite l_prev / o_prev without extra vselect ops.
    skip_lo_reset = use_bf16_qkp

    qk = jnp.where(jnp.stack(masks, axis=0), qk, mask_fill)

    m_reset = jnp.asarray(-jnp.inf, dtype=acc_dt)
    zero_acc = jnp.asarray(0.0, dtype=acc_dt)
    m_prev = m_scratch_ref[...].reshape(tot_q, _TPU_NUM_LANES)
    l_prev = l_scratch_ref[...].reshape(tot_q, _TPU_NUM_LANES)
    o_prev = acc_scratch_ref[...].reshape(tot_q, cfgs.aligned_head_dim)

    m_curr = jnp.max(qk, axis=-1, keepdims=True)
    alpha_list: list[jax.Array] = []
    m_next_list: list[jax.Array] = []
    for b_idx in range(cfgs.batch_size):
        m_next_b = jnp.maximum(m_prev, m_curr[b_idx].astype(acc_dt))
        alpha_b = exp_fn(m_prev - m_next_b)
        alpha_list.append(alpha_b)
        m_next_list.append(m_next_b)
        m_prev = jnp.where(schedule_ref.is_last_k[step, b_idx] == 1, m_reset, m_next_b)
    m_scratch_ref[...] = m_prev.reshape(cfgs.lm_scratch_shape)

    m_next = jnp.stack(m_next_list, axis=0).astype(qk.dtype)
    p = exp_fn(qk - _broadcast_minor(m_next, qk.shape))
    p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=acc_dt)

    l_next_list: list[jax.Array] = []
    for b_idx in range(cfgs.batch_size):
        l_next_b = alpha_list[b_idx] * l_prev + p_rowsum[b_idx]
        l_next_list.append(l_next_b)
        l_prev = (
            l_next_b
            if skip_lo_reset
            else jnp.where(schedule_ref.is_last_k[step, b_idx] == 1, zero_acc, l_next_b)
        )
    l_scratch_ref[...] = l_prev.reshape(cfgs.lm_scratch_shape)

    pv = lax.dot_general(
        p.reshape(-1, tq, cfgs.bkv_sz),
        v.reshape(-1, cfgs.bkv_sz, cfgs.aligned_head_dim),
        dimension_numbers=(([2], [1]), ([0], [0])),
        preferred_element_type=jnp.float32,
    ).reshape(cfgs.batch_size, tot_q, cfgs.aligned_head_dim)
    if cfgs.v_scale is not None:
        pv = pv * cfgs.v_scale
    pv = pv.astype(acc_dt)

    o_next_list: list[jax.Array] = []
    for b_idx in range(cfgs.batch_size):
        alpha_full = _broadcast_minor(alpha_list[b_idx], o_prev.shape)
        o_next_b = alpha_full * o_prev + pv[b_idx]
        o_next_list.append(o_next_b)
        o_prev = (
            o_next_b
            if skip_lo_reset
            else jnp.where(schedule_ref.is_last_k[step, b_idx] == 1, zero_acc, o_next_b)
        )
    acc_scratch_ref[...] = o_prev.reshape(cfgs.acc_scratch_shape)

    for b_idx in range(cfgs.batch_size):
        batch_acc = o_next_list[b_idx]
        batch_l = _broadcast_minor(l_next_list[b_idx], batch_acc.shape)
        if (
            cfgs.interpret_mode
            or cfgs.dtype_q == jnp.float32
            or cfgs.dtype_q == batch_l.dtype == jnp.bfloat16
        ):
            res = lax.div(batch_acc, batch_l)
        else:
            res = batch_acc * pltpu.reciprocal(batch_l, approx=True)
        out_b = res.astype(cfgs.dtype_q)
        if cfgs.interpret_mode:
            o_vref[b_idx] = out_b.reshape(o_vref.shape[1:])
        else:
            out_ref = (
                o_vref.at[b_idx].bitcast(jnp.uint32).reshape(-1, cfgs.aligned_head_dim)
            )
            out_u32 = pltpu.bitcast(out_b, jnp.uint32).reshape(out_ref.shape)
            _strided_store(out_ref, 0, out_ref.shape[0], 1, out_u32)


def _init_batched_scratches(
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    acc_scratch_ref: Any,
    bq_x3_ref: Any,
    bkv_x3_ref: Any,
    *,
    interpret_mode: bool,
) -> None:
    m_scratch_ref[...] = jnp.full_like(m_scratch_ref, -jnp.inf)
    l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
    acc_scratch_ref[...] = jnp.zeros_like(acc_scratch_ref)
    if interpret_mode:
        bq_x3_ref[...] = jnp.zeros_like(bq_x3_ref)
        bkv_x3_ref[...] = jnp.zeros_like(bkv_x3_ref)
    else:
        q_flat = bq_x3_ref.bitcast(jnp.uint32).reshape(-1, _TPU_NUM_LANES)
        kv_flat = bkv_x3_ref.bitcast(jnp.uint32).reshape(-1, _TPU_NUM_LANES)
        q_flat[...] = jnp.zeros_like(q_flat)
        kv_flat[...] = jnp.zeros_like(kv_flat)


def _make_batched_q_o_dma(
    q_hbm_ref: Any,
    o_hbm_ref: Any,
    bq_x3_ref: Any,
    bo_x2_ref: Any,
    schedule_ref: Any,
    q_recv_sem: Any,
    o_send_sem: Any,
    *,
    batch_size: int,
    interpret_mode: bool,
) -> tuple[
    Callable[[Any, Any, Any], None],
    Callable[[Any, Any, Any], None],
    Callable[[Any, Any, Any], None],
    Callable[[Any, Any, Any], None],
]:
    def copy_in_q(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = q_recv_sem.at[slot]
        vmem_dst = bq_x3_ref.at[slot]
        for b in range(batch_size):
            q_sz = jnp.where(is_noop, 0, schedule_ref.dma_q[blk, b, 1])
            q_src = jnp.where(q_sz > 0, schedule_ref.dma_q[blk, b, 0], 0)
            pltpu.make_async_copy(
                q_hbm_ref.at[:, pl.ds(q_src, q_sz)],
                vmem_dst.at[b, :, pl.ds(0, q_sz)],
                sem,
            ).start()

    def wait_in_q(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = q_recv_sem.at[slot]
        if interpret_mode:
            for b in range(batch_size):
                q_sz = jnp.where(is_noop, 0, schedule_ref.dma_q[blk, b, 1])
                pltpu.make_async_copy(
                    bq_x3_ref.at[slot, b, :, pl.ds(0, q_sz)],
                    bq_x3_ref.at[slot, b, :, pl.ds(0, q_sz)],
                    sem,
                ).wait()
            return
        wait_lanes = jnp.where(is_noop, 0, schedule_ref.total_wait_q_in[blk])
        flat = bq_x3_ref.at[slot].bitcast(jnp.uint32).reshape((-1, _TPU_NUM_LANES))
        pltpu.make_async_copy(
            flat.at[pl.ds(0, wait_lanes), :],
            flat.at[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()

    def copy_out_o(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = o_send_sem.at[slot]
        vmem_src = bo_x2_ref.at[slot]
        for b in range(batch_size):
            is_last = schedule_ref.is_last_k[blk, b] == 1
            q_sz = schedule_ref.dma_q[blk, b, 1]
            sz = jnp.where(jnp.logical_or(is_noop, jnp.logical_not(is_last)), 0, q_sz)
            q_src = jnp.where(sz > 0, schedule_ref.dma_q[blk, b, 0], 0)
            pltpu.make_async_copy(
                vmem_src.at[b, :, pl.ds(0, sz)],
                o_hbm_ref.at[:, pl.ds(q_src, sz)],
                sem,
            ).start()

    def wait_out_o(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = o_send_sem.at[slot]
        if interpret_mode:
            vmem_src = bo_x2_ref.at[slot]
            for b in range(batch_size):
                is_last = schedule_ref.is_last_k[blk, b] == 1
                q_sz = schedule_ref.dma_q[blk, b, 1]
                sz = jnp.where(
                    jnp.logical_or(is_noop, jnp.logical_not(is_last)), 0, q_sz
                )
                pltpu.make_async_copy(
                    vmem_src.at[b, :, pl.ds(0, sz)],
                    vmem_src.at[b, :, pl.ds(0, sz)],
                    sem,
                ).wait()
            return
        wait_lanes = jnp.where(is_noop, 0, schedule_ref.total_wait_o_out[blk])
        flat = bo_x2_ref.at[slot].bitcast(jnp.uint32).reshape((-1, _TPU_NUM_LANES))
        pltpu.make_async_copy(
            flat.at[pl.ds(0, wait_lanes), :],
            flat.at[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()

    return copy_in_q, wait_in_q, copy_out_o, wait_out_o


def _run_batched_schedule_chunks(
    schedule_hbm_ref: Any,
    schedule_ref: Any,
    sched_dma_sem: Any,
    compute_fn: Callable[..., None],
    *,
    max_steps_ub: int,
    copy_in_q: Callable[[Any, Any, Any], None],
    wait_in_q: Callable[[Any, Any, Any], None],
    copy_in_kv: Callable[[Any, Any, Any], None],
    wait_in_kv: Callable[[Any, Any, Any], None],
    copy_out_kv: Callable[[Any, Any, Any], None],
    wait_out_kv: Callable[[Any, Any, Any], None],
    copy_out_o: Callable[[Any, Any, Any], None],
    wait_out_o: Callable[[Any, Any, Any], None],
) -> None:
    def execute_schedule_chunk(start_step: jax.Array, total_steps: jax.Array) -> None:
        flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
        flat_smem = jax.tree_util.tree_leaves(schedule_ref)
        copies = []
        for h, s in zip(flat_hbm, flat_smem, strict=True):
            if s.shape[0] > 1:
                elem_sz = s.shape[0] // max_steps_ub
                read_sz = jnp.minimum(align_to(elem_sz * total_steps, 1024), s.shape[0])
                src_off = pl.multiple_of(elem_sz * start_step, _TPU_NUM_LANES)
                cp = pltpu.make_async_copy(
                    h.at[pl.ds(src_off, read_sz)],
                    s.at[pl.ds(0, read_sz)],
                    sched_dma_sem.at[0],
                )
                cp.start()
                copies.append(cp)
        for cp in copies:
            cp.wait()

        _run_manual_triple_buffer(
            total_steps,
            compute_fn,
            copy_in_q=copy_in_q,
            wait_in_q=wait_in_q,
            copy_in_kv=copy_in_kv,
            wait_in_kv=wait_in_kv,
            copy_out_kv=copy_out_kv,
            wait_out_kv=wait_out_kv,
            copy_out_o=copy_out_o,
            wait_out_o=wait_out_o,
        )

    actual_steps = schedule_hbm_ref.actual_steps[0]
    num_chunks = cdiv(actual_steps, max_steps_ub)

    @pl.loop(0, num_chunks)
    def _chunk_loop(chunk_idx: jax.Array) -> None:
        start = chunk_idx * max_steps_ub
        execute_schedule_chunk(start, jnp.minimum(actual_steps - start, max_steps_ub))


def _invoke_batched_rpa_pallas(
    pipeline_fn: Callable[..., None],
    scalar_prefetches: tuple[jax.Array, ...],
    schedule_hbm: Any,
    sched_smem_struct: Any,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    *,
    cfgs: Any,
    tpu_version: int,
) -> tuple[jax.Array, jax.Array]:
    num_scalar_prefetch = len(scalar_prefetches)
    schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    input_output_aliases = {
        num_scalar_prefetch + schedule_leaves: 0,
        num_scalar_prefetch + schedule_leaves + 2: 1,
    }
    pin_hbm = tpu_version >= 7 and not cfgs.interpret_mode
    interpret_params = pltpu.InterpretParams() if cfgs.interpret_mode else False
    compiler_flags: dict[str, bool | int | float | str] = (
        {"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True}
        if not cfgs.interpret_mode
        else {}
    )

    kernel = pl.pallas_call(
        functools.partial(pipeline_fn, cfgs=cfgs),
        out_shape=(
            [
                pltpu.HBM(shape=q_hbm.shape, dtype=q_hbm.dtype),
                pltpu.HBM(shape=kv_cache_hbm.shape, dtype=kv_cache_hbm.dtype),
            ]
            if pin_hbm
            else [
                jax.ShapeDtypeStruct(shape=q_hbm.shape, dtype=q_hbm.dtype),
                jax.ShapeDtypeStruct(
                    shape=kv_cache_hbm.shape, dtype=kv_cache_hbm.dtype
                ),
            ]
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[
                schedule_hbm.in_specs(),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM(
                    (_BATCHED_NUM_INPUT_SLOTS, *cfgs.kv_vmem_shape),
                    cfgs.dtype_kv,
                ),
                pltpu.VMEM(
                    (_BATCHED_NUM_INPUT_SLOTS, *cfgs.q_vmem_shape),
                    cfgs.dtype_q,
                ),
                pltpu.VMEM(
                    (_BATCHED_NUM_OUTPUT_SLOTS, *cfgs.q_vmem_shape),
                    cfgs.dtype_q,
                ),
                sched_smem_struct.scratch_shapes(),
                pltpu.SemaphoreType.DMA((_BATCHED_NUM_INPUT_SLOTS,)),
                pltpu.SemaphoreType.DMA((_BATCHED_NUM_INPUT_SLOTS,)),
                pltpu.SemaphoreType.DMA((_BATCHED_NUM_INPUT_SLOTS,)),
                pltpu.SemaphoreType.DMA((_BATCHED_NUM_OUTPUT_SLOTS,)),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.VMEM(cfgs.lm_scratch_shape, cfgs.scratch_dtype),
                pltpu.VMEM(cfgs.lm_scratch_shape, cfgs.scratch_dtype),
                pltpu.VMEM(cfgs.acc_scratch_shape, cfgs.scratch_dtype),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=cfgs.disable_bounds_checks,
            disable_semaphore_checks=cfgs.disable_semaphore_checks,
            flags=compiler_flags,
        ),
        input_output_aliases=input_output_aliases,
        interpret=interpret_params,
        name=cfgs.kernel_name,
    )

    if pin_hbm:
        return kernel(
            *scalar_prefetches,
            schedule_hbm,
            pltpu.with_memory_space_constraint(q_hbm, pltpu.HBM),
            pltpu.with_memory_space_constraint(new_kv_hbm, pltpu.HBM),
            pltpu.with_memory_space_constraint(kv_cache_hbm, pltpu.HBM),
        )
    return kernel(
        *scalar_prefetches,
        schedule_hbm,
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
    )


def _batched_decode_rpa_pipeline(
    # Scalar prefetch (SMEM)
    kv_lens_ref: Any,
    page_indices_ref: Any,
    # Inputs (HBM + SMEM actual_steps)
    schedule_hbm_ref: BatchedDecodeSchedule,
    q_hbm_ref: Any,
    new_kv_hbm_ref: Any,
    kv_cache_hbm_ref: Any,
    # Outputs (HBM, aliased to q_hbm_ref and kv_cache_hbm_ref)
    o_hbm_ref: Any,
    updated_kv_cache_hbm_ref: Any,
    # Scratch (VMEM + SMEM + DMA semaphores)
    bkv_x3_ref: Any,
    bq_x3_ref: Any,
    bo_x2_ref: Any,
    schedule_ref: BatchedDecodeSchedule,
    q_recv_sem: Any,
    kv_recv_sem: Any,
    kv_send_sem: Any,
    o_send_sem: Any,
    sched_dma_sem: Any,
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    acc_scratch_ref: Any,
    *,
    cfgs: BatchedDecodeConfig,
) -> None:
    del updated_kv_cache_hbm_ref
    _init_batched_scratches(
        m_scratch_ref,
        l_scratch_ref,
        acc_scratch_ref,
        bq_x3_ref,
        bkv_x3_ref,
        interpret_mode=cfgs.interpret_mode,
    )
    copy_in_q, wait_in_q, copy_out_o, wait_out_o = _make_batched_q_o_dma(
        q_hbm_ref,
        o_hbm_ref,
        bq_x3_ref,
        bo_x2_ref,
        schedule_ref,
        q_recv_sem,
        o_send_sem,
        batch_size=cfgs.batch_size,
        interpret_mode=cfgs.interpret_mode,
    )

    def copy_in_kv(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = kv_recv_sem.at[slot]
        vmem_dst = bkv_x3_ref.at[slot, :, :, : cfgs.kv_hbm_stride]
        kv_hbm_flat = (
            None
            if cfgs.interpret_mode
            else kv_cache_hbm_ref.reshape(-1, *kv_cache_hbm_ref.shape[2:])
        )
        for b in range(cfgs.batch_size):
            for i in range(cfgs.bkv_p):
                sz = jnp.where(is_noop, 0, schedule_ref.dma_kv_cache[blk, b, i, 2])
                p_slot = jnp.where(sz > 0, schedule_ref.dma_kv_cache[blk, b, i, 0], 0)
                dst_off = jnp.where(sz > 0, schedule_ref.dma_kv_cache[blk, b, i, 1], 0)
                page_id = jnp.where(sz > 0, page_indices_ref[p_slot], 0)
                if cfgs.interpret_mode:
                    src = kv_cache_hbm_ref.at[page_id, pl.ds(0, sz)]
                else:
                    assert kv_hbm_flat is not None
                    src = kv_hbm_flat.at[pl.ds(page_id * cfgs.page_size, sz)]
                pltpu.make_async_copy(
                    src,
                    vmem_dst.at[b, pl.ds(dst_off, sz)],
                    sem,
                ).start()
            new_sz = jnp.where(is_noop, 0, schedule_ref.dma_kv_new[blk, b, 4])
            fetch_hbm = jnp.where(new_sz > 0, schedule_ref.dma_kv_new[blk, b, 2], 0)
            dst_vmem_off = jnp.where(new_sz > 0, schedule_ref.dma_kv_new[blk, b, 3], 0)
            pltpu.make_async_copy(
                new_kv_hbm_ref.at[pl.ds(fetch_hbm, new_sz)],
                vmem_dst.at[b, pl.ds(dst_vmem_off, new_sz)],
                sem,
            ).start()

    def wait_in_kv(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = kv_recv_sem.at[slot]
        if cfgs.interpret_mode:
            vmem_dst = bkv_x3_ref.at[slot, :, :, : cfgs.kv_hbm_stride]
            for b in range(cfgs.batch_size):
                lane_sz: Any = schedule_ref.dma_kv_new[blk, b, 4]
                for i in range(cfgs.bkv_p):
                    lane_sz = lane_sz + schedule_ref.dma_kv_cache[blk, b, i, 2]
                lane_sz = jnp.where(is_noop, 0, lane_sz)
                pltpu.make_async_copy(
                    vmem_dst.at[b, pl.ds(0, lane_sz)],
                    vmem_dst.at[b, pl.ds(0, lane_sz)],
                    sem,
                ).wait()
            return
        wait_lanes = jnp.where(is_noop, 0, schedule_ref.total_wait_kv_in[blk])
        flat = bkv_x3_ref.at[slot].bitcast(jnp.uint32).reshape((-1, _TPU_NUM_LANES))
        pltpu.make_async_copy(
            flat.at[pl.ds(0, wait_lanes), :],
            flat.at[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()

    def copy_out_kv(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = kv_send_sem.at[slot]
        vmem_src = bkv_x3_ref.at[slot, :, :, : cfgs.kv_hbm_stride]
        kv_hbm_flat = (
            None
            if cfgs.interpret_mode
            else kv_cache_hbm_ref.reshape(-1, *kv_cache_hbm_ref.shape[2:])
        )
        for b in range(cfgs.batch_size):
            do_wb = schedule_ref.do_writeback[blk, b] == 1
            new_sz = schedule_ref.dma_kv_new[blk, b, 4]
            sz = jnp.where(jnp.logical_or(is_noop, jnp.logical_not(do_wb)), 0, new_sz)
            wb_p_slot = jnp.where(sz > 0, schedule_ref.dma_kv_new[blk, b, 0], 0)
            wb_p_off = jnp.where(sz > 0, schedule_ref.dma_kv_new[blk, b, 1], 0)
            src_vmem_off = jnp.where(sz > 0, schedule_ref.dma_kv_new[blk, b, 3], 0)
            page_id = jnp.where(sz > 0, page_indices_ref[wb_p_slot], 0)
            if cfgs.interpret_mode:
                dst = kv_cache_hbm_ref.at[page_id, pl.ds(wb_p_off, sz)]
            else:
                assert kv_hbm_flat is not None
                dst = kv_hbm_flat.at[pl.ds(page_id * cfgs.page_size + wb_p_off, sz)]
            pltpu.make_async_copy(
                vmem_src.at[b, pl.ds(src_vmem_off, sz)],
                dst,
                sem,
            ).start()

    def wait_out_kv(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = kv_send_sem.at[slot]
        if cfgs.interpret_mode:
            vmem_src = bkv_x3_ref.at[slot, :, :, : cfgs.kv_hbm_stride]
            for b in range(cfgs.batch_size):
                do_wb = schedule_ref.do_writeback[blk, b] == 1
                new_sz = schedule_ref.dma_kv_new[blk, b, 4]
                sz = jnp.where(
                    jnp.logical_or(is_noop, jnp.logical_not(do_wb)), 0, new_sz
                )
                pltpu.make_async_copy(
                    vmem_src.at[b, pl.ds(0, sz)],
                    vmem_src.at[b, pl.ds(0, sz)],
                    sem,
                ).wait()
            return
        wait_lanes = jnp.where(is_noop, 0, schedule_ref.total_wait_kv_out[blk])
        flat = bkv_x3_ref.at[slot].bitcast(jnp.uint32).reshape((-1, _TPU_NUM_LANES))
        pltpu.make_async_copy(
            flat.at[pl.ds(0, wait_lanes), :],
            flat.at[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()

    def compute_fn(
        step: jax.Array,
        comp_slot: jax.Array,
        out_slot: jax.Array,
        start_next_fetch: Callable[[], None],
        start_kv_writeback: Callable[[], None],
    ) -> None:
        _batched_decode_rpa_body(
            step,
            bq_x3_ref.at[comp_slot],
            bkv_x3_ref.at[comp_slot],
            bo_x2_ref.at[out_slot],
            schedule_ref,
            m_scratch_ref,
            l_scratch_ref,
            acc_scratch_ref,
            kv_lens_ref=kv_lens_ref,
            cfgs=cfgs,
            start_next_fetch=start_next_fetch,
            start_kv_writeback=start_kv_writeback,
        )

    _run_batched_schedule_chunks(
        schedule_hbm_ref,
        schedule_ref,
        sched_dma_sem,
        compute_fn,
        max_steps_ub=cfgs.max_steps_ub,
        copy_in_q=copy_in_q,
        wait_in_q=wait_in_q,
        copy_in_kv=copy_in_kv,
        wait_in_kv=wait_in_kv,
        copy_out_kv=copy_out_kv,
        wait_out_kv=wait_out_kv,
        copy_out_o=copy_out_o,
        wait_out_o=wait_out_o,
    )


def run_batched_decode_rpa(
    kv_lens: jax.Array,
    page_indices: jax.Array,
    schedule_hbm: BatchedDecodeSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    *,
    cfgs: BatchedDecodeConfig,
    tpu_version: int,
) -> tuple[jax.Array, jax.Array]:
    """Execute the triple-buffered batched decode attention Pallas kernel."""
    return _invoke_batched_rpa_pallas(
        _batched_decode_rpa_pipeline,
        (kv_lens, page_indices),
        schedule_hbm,
        BatchedDecodeSchedule.create_shape_dtype(cfgs, steps=cfgs.max_steps_ub),
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
        cfgs=cfgs,
        tpu_version=tpu_version,
    )


# ---------------------------------------------------------------------------
# Batched RPA mixed / prefill (triple-buffered manual DMA + SMEM task schedule)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=True, kw_only=True)
class BatchedMixedConfig(_BatchedRpaConfigMixin):
    """Static configuration for the triple-buffered batched mixed/prefill RPA kernel."""

    bq_sz: int
    bq_csz: int
    max_num_tokens: int
    use_causal_mask: bool = True
    bf16_qk_softmax: bool = True

    @property
    def bq_tokens_per_block(self) -> int:
        return self.bq_sz

    @property
    def scratch_dtype(self) -> jnp.dtype:
        if self.bf16_qk_softmax and self.dtype_q == jnp.bfloat16:
            return jnp.bfloat16
        return jnp.float32

    @property
    def max_steps_ub(self) -> int:
        fixed_words = (
            self.max_num_seqs + (self.max_num_seqs + 1) + self.num_page_indices + 3 + 1
        )
        available_bytes = (
            int((self.smem_capacity_bytes - 32 * 1024) * 0.33) - fixed_words * 4
        )
        # Per step per batch item: s_idx, q_idx, k_idx, is_last_k (4), dma_q (2),
        # dma_kv_cache (3 * bkv_p), dma_kv_new_in (3), dma_kv_wb (4 * bkv_p);
        # plus 4 1D wait words per step.
        bytes_per_step = ((9 + 7 * self.bkv_p) * self.batch_size + 4) * 4
        raw_steps = max(_TPU_NUM_LANES, available_bytes // bytes_per_step)
        return max(_TPU_NUM_LANES, (raw_steps // _TPU_NUM_LANES) * _TPU_NUM_LANES)

    @property
    def schedule_hbm_steps(self) -> int:
        max_q_blocks = max(
            1,
            self.max_num_tokens // self.bq_sz
            + min(self.max_num_seqs, self.max_num_tokens),
        )
        max_k_per_q = max(
            1, int(cdiv(self.pages_per_seq * self.page_size, self.bkv_sz))
        )
        max_total_steps = max(1, int(cdiv(max_q_blocks * max_k_per_q, self.batch_size)))
        return max(self.max_steps_ub, align_to(max_total_steps, self.max_steps_ub))

    @property
    def kernel_name(self) -> str:
        name = (
            f"RPAm-snh-p{self.page_size}-b{self.batch_size}"
            f"-q{self.bq_sz}_{self.bq_csz}-k{self.bkv_sz}"
        )
        if not self.bf16_qk_softmax:
            name += "-f32p"
        if self.sliding_window is not None:
            name += f"-sw{self.sliding_window}"
        return name


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BatchedMixedSchedule(_ScheduleSpecMixin):
    """Flat SMEM/HBM schedule for batched mixed/prefill RPA."""

    s_idx: _SmemArray  # [steps, B]
    q_idx: _SmemArray  # [steps, B]
    k_idx: _SmemArray  # [steps, B]
    is_last_k: _SmemArray  # [steps, B]
    dma_q: _SmemArray  # [steps, B, 2] -> (q_src, q_sz)
    dma_kv_cache: _SmemArray  # [steps, B, bkv_p, 3] -> (p_slot, dst_vmem, sz)
    dma_kv_new_in: _SmemArray  # [steps, B, 3] -> (fetch_hbm, vmem_off, new_sz)
    dma_kv_wb: _SmemArray  # [steps, B, bkv_p, 4] -> (wb_p_slot, p_off, wb_vmem, wb_sz)
    total_wait_kv_in: _SmemArray  # [steps]
    total_wait_kv_out: _SmemArray  # [steps]
    total_wait_q_in: _SmemArray  # [steps]
    total_wait_o_out: _SmemArray  # [steps]
    actual_steps: Any  # [1]

    @classmethod
    def create_shape_dtype(
        cls, cfgs: BatchedMixedConfig, *, steps: int
    ) -> BatchedMixedSchedule:
        idx_arr = _SmemArray.create_shape_dtype((steps, cfgs.batch_size))
        step_arr = _SmemArray.create_shape_dtype((steps,))
        return cls(
            s_idx=idx_arr,
            q_idx=idx_arr,
            k_idx=idx_arr,
            is_last_k=idx_arr,
            dma_q=_SmemArray.create_shape_dtype((steps, cfgs.batch_size, 2)),
            dma_kv_cache=_SmemArray.create_shape_dtype(
                (steps, cfgs.batch_size, cfgs.bkv_p, 3)
            ),
            dma_kv_new_in=_SmemArray.create_shape_dtype((steps, cfgs.batch_size, 3)),
            dma_kv_wb=_SmemArray.create_shape_dtype(
                (steps, cfgs.batch_size, cfgs.bkv_p, 4)
            ),
            total_wait_kv_in=step_arr,
            total_wait_kv_out=step_arr,
            total_wait_q_in=step_arr,
            total_wait_o_out=step_arr,
            actual_steps=jax.ShapeDtypeStruct((1,), jnp.int32),
        )


def _batched_mixed_schedule_kernel(
    kv_lens_ref: Any,
    cu_q_lens_ref: Any,
    distribution_ref: Any,
    schedule_hbm_ref: BatchedMixedSchedule,
    schedule_ref: BatchedMixedSchedule,
    dma_sem: Any,
    *,
    cfgs: BatchedMixedConfig,
) -> None:
    schedule_ref.actual_steps[0] = 0

    def _mask_entry(step: Any, b_idx: Any) -> None:
        schedule_ref.s_idx[step, b_idx] = -1
        schedule_ref.q_idx[step, b_idx] = 0
        schedule_ref.k_idx[step, b_idx] = 0
        schedule_ref.is_last_k[step, b_idx] = 0
        schedule_ref.dma_q[step, b_idx, 0] = 0
        schedule_ref.dma_q[step, b_idx, 1] = 0
        for i in range(cfgs.bkv_p):
            schedule_ref.dma_kv_cache[step, b_idx, i, 0] = 0
            schedule_ref.dma_kv_cache[step, b_idx, i, 1] = 0
            schedule_ref.dma_kv_cache[step, b_idx, i, 2] = 0
            for f in range(4):
                schedule_ref.dma_kv_wb[step, b_idx, i, f] = 0
        for f in range(3):
            schedule_ref.dma_kv_new_in[step, b_idx, f] = 0
        schedule_ref.total_wait_kv_in[step] = 0
        schedule_ref.total_wait_kv_out[step] = 0
        schedule_ref.total_wait_q_in[step] = 0
        schedule_ref.total_wait_o_out[step] = 0

    def _flush(count: Any, hbm_offset: Any) -> _SchedCarry:
        last_task = align_to(count, cfgs.batch_size)

        @pl.loop(count, last_task)
        def _mask_tail(idx: Any) -> None:
            step, b_idx = divmod(idx, cfgs.batch_size)
            _mask_entry(step, b_idx)

        def _compute_step_waits(step: Any, _: None) -> None:
            kv_in_tokens: Any = 0
            kv_out_tokens: Any = 0
            q_in_tokens: Any = 0
            o_out_tokens: Any = 0
            for b in range(cfgs.batch_size):
                for i in range(cfgs.bkv_p):
                    kv_in_tokens = (
                        kv_in_tokens + schedule_ref.dma_kv_cache[step, b, i, 2]
                    )
                    kv_out_tokens = (
                        kv_out_tokens + schedule_ref.dma_kv_wb[step, b, i, 3]
                    )
                kv_in_tokens = kv_in_tokens + schedule_ref.dma_kv_new_in[step, b, 2]
                q_sz = schedule_ref.dma_q[step, b, 1]
                q_in_tokens = q_in_tokens + q_sz
                is_last = schedule_ref.is_last_k[step, b] == 1
                o_out_tokens = o_out_tokens + jnp.where(is_last, q_sz, 0)

            schedule_ref.total_wait_kv_in[step] = (
                kv_in_tokens * cfgs.kv_bytes_per_token
            ) // _DMA_CHUNK_BYTES
            schedule_ref.total_wait_kv_out[step] = (
                kv_out_tokens * cfgs.kv_bytes_per_token
            ) // _DMA_CHUNK_BYTES
            schedule_ref.total_wait_q_in[step] = (
                q_in_tokens * cfgs.q_bytes_per_token
            ) // _DMA_CHUNK_BYTES
            schedule_ref.total_wait_o_out[step] = (
                o_out_tokens * cfgs.q_bytes_per_token
            ) // _DMA_CHUNK_BYTES

        lax.fori_loop(0, last_task // cfgs.batch_size, _compute_step_waits, None)

        hbm_offset_aligned = pl.multiple_of(hbm_offset, _TPU_NUM_LANES)
        flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
        flat_smem = jax.tree_util.tree_leaves(schedule_ref)
        copies = []
        for h, s in zip(flat_hbm, flat_smem, strict=True):
            if h.shape[0] > 1:
                elem_sz = s.shape[0] // cfgs.max_steps_ub
                write_sz = cfgs.max_steps_ub * elem_sz
                out_off = hbm_offset_aligned * elem_sz
                copies.append(
                    pltpu.make_async_copy(
                        s.at[pl.ds(0, write_sz)],
                        h.at[pl.ds(out_off, write_sz)],
                        dma_sem.at[0],
                    )
                )
        for cp in copies:
            cp.start()
        for cp in copies:
            cp.wait()
        return _SchedCarry(hbm_offset + cfgs.max_steps_ub, 0)

    def _seq_body(s_idx: Any, carry: _SchedCarry) -> _SchedCarry:
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        q_len = jnp.maximum(q_end - q_start, 0)
        kv_len = kv_lens_ref[s_idx]
        q_offset = jnp.maximum(kv_len - q_len, 0)
        kv_cache_len = q_offset if cfgs.update_kv_cache else kv_len

        has_work = jnp.logical_and(q_len > 0, kv_len > 0)
        num_q = jnp.where(has_work, cdiv(q_len, cfgs.bq_sz), 0)
        num_k = jnp.where(has_work, cdiv(kv_len, cfgs.bkv_sz), 0)

        def _q_body(q_idx: Any, q_carry: _SchedCarry) -> _SchedCarry:
            q_src = q_start + q_idx * cfgs.bq_sz
            q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)
            start_k_idx: Any = 0
            if cfgs.sliding_window is not None:
                sw_start = jnp.maximum(
                    0, q_offset + q_idx * cfgs.bq_sz - cfgs.sliding_window + 1
                )
                start_k_idx = sw_start // cfgs.bkv_sz

            if cfgs.use_causal_mask:
                end_k_causal = (
                    q_offset + q_idx * cfgs.bq_sz + q_sz_task - 1
                ) // cfgs.bkv_sz + 1
                end_k_idx = jnp.minimum(num_k, end_k_causal)
            else:
                end_k_idx = num_k

            def _k_body(k_idx: Any, k_carry: _SchedCarry) -> _SchedCarry:
                count = k_carry.count
                step, lane = divmod(count, cfgs.batch_size)
                schedule_ref.s_idx[step, lane] = s_idx
                schedule_ref.q_idx[step, lane] = q_idx
                schedule_ref.k_idx[step, lane] = k_idx
                schedule_ref.is_last_k[step, lane] = jnp.where(
                    k_idx == end_k_idx - 1, 1, 0
                )
                schedule_ref.dma_q[step, lane, 0] = q_src
                schedule_ref.dma_q[step, lane, 1] = q_sz_task

                kv_len_start = k_idx * cfgs.bkv_sz
                kv_p_start = k_idx * cfgs.bkv_p
                kv_left = kv_len - kv_len_start
                kv_left_cache = (
                    jnp.maximum(kv_cache_len - kv_len_start, 0)
                    if cfgs.update_kv_cache
                    else kv_left
                )
                p_base = s_idx * cfgs.pages_per_seq + kv_p_start

                for i in range(cfgs.bkv_p):
                    dst_vmem = i * cfgs.page_size
                    dma_sz = jnp.clip(kv_left_cache - dst_vmem, 0, cfgs.page_size)
                    p_slot = jnp.minimum(p_base + i, cfgs.num_page_indices - 1)
                    schedule_ref.dma_kv_cache[step, lane, i, 0] = p_slot
                    schedule_ref.dma_kv_cache[step, lane, i, 1] = dst_vmem
                    schedule_ref.dma_kv_cache[step, lane, i, 2] = dma_sz

                bkv_sz_cache = jnp.minimum(kv_left_cache, cfgs.bkv_sz)
                kv_left_new = kv_left - kv_left_cache
                new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_new)
                fetch_hbm = jnp.maximum(q_end - kv_left_new, 0)
                schedule_ref.dma_kv_new_in[step, lane, 0] = fetch_hbm
                schedule_ref.dma_kv_new_in[step, lane, 1] = bkv_sz_cache
                schedule_ref.dma_kv_new_in[step, lane, 2] = new_sz

                q_wb = jnp.maximum(0, kv_len_start - kv_cache_len) // cfgs.bq_sz
                do_wb = jnp.logical_and(new_sz > 0, q_idx == q_wb)
                for i in range(cfgs.bkv_p):
                    slot_start = i * cfgs.page_size
                    slot_end = slot_start + cfgs.page_size
                    wb_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                    end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                    wb_sz = jnp.where(do_wb, jnp.maximum(0, end_in_slot - wb_vmem), 0)
                    tok_idx = kv_len_start + wb_vmem
                    p_idx = jnp.minimum(
                        tok_idx // cfgs.page_size, cfgs.pages_per_seq - 1
                    )
                    p_off = tok_idx % cfgs.page_size
                    wb_p_slot = jnp.minimum(
                        s_idx * cfgs.pages_per_seq + p_idx,
                        cfgs.num_page_indices - 1,
                    )
                    schedule_ref.dma_kv_wb[step, lane, i, 0] = wb_p_slot
                    schedule_ref.dma_kv_wb[step, lane, i, 1] = p_off
                    schedule_ref.dma_kv_wb[step, lane, i, 2] = wb_vmem
                    schedule_ref.dma_kv_wb[step, lane, i, 3] = wb_sz

                next_count = count + 1
                return lax.cond(
                    next_count < cfgs.max_steps_ub * cfgs.batch_size,
                    lambda c: c,
                    lambda c: _flush(c.count, c.hbm_offset),
                    _SchedCarry(k_carry.hbm_offset, next_count),
                )

            return lax.fori_loop(start_k_idx, end_k_idx, _k_body, q_carry)

        return lax.fori_loop(0, num_q, _q_body, carry)

    start_seq = distribution_ref[1]
    end_seq = distribution_ref[2]
    final_carry = lax.fori_loop(start_seq, end_seq, _seq_body, _SchedCarry(0, 0))
    total_steps = cdiv(final_carry.count, cfgs.batch_size) + final_carry.hbm_offset
    schedule_ref.actual_steps[0] = total_steps

    @pl.when(final_carry.count > 0)
    def _flush_remainder() -> None:
        _flush(final_carry.count, final_carry.hbm_offset)

    steps_cp = pltpu.make_async_copy(
        schedule_ref.actual_steps.at[pl.ds(0, 1)],
        schedule_hbm_ref.actual_steps.at[pl.ds(0, 1)],
        dma_sem.at[0],
    )
    steps_cp.start()
    steps_cp.wait()


def generate_batched_mixed_schedule(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    cfgs: BatchedMixedConfig,
) -> BatchedMixedSchedule:
    """Build the SMEM/HBM work schedule for batched mixed/prefill attention."""
    sched_smem_struct = BatchedMixedSchedule.create_shape_dtype(
        cfgs, steps=cfgs.max_steps_ub
    )
    sched_hbm_struct = BatchedMixedSchedule.create_shape_dtype(
        cfgs, steps=cfgs.schedule_hbm_steps
    )
    interpret_params = pltpu.InterpretParams() if cfgs.interpret_mode else False
    return pl.pallas_call(
        functools.partial(_batched_mixed_schedule_kernel, cfgs=cfgs),
        out_shape=sched_hbm_struct,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[],
            out_specs=sched_hbm_struct.out_specs(),
            scratch_shapes=[
                sched_smem_struct.scratch_shapes(),
                pltpu.SemaphoreType.DMA((1,)),
            ],
            grid=(1,),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=cfgs.disable_bounds_checks,
            disable_semaphore_checks=cfgs.disable_semaphore_checks,
        ),
        interpret=interpret_params,
        name="rpa_mixed_schedule",
    )(kv_lens, cu_q_lens, distribution)


def _batched_mixed_rpa_body(
    step: jax.Array,
    q_vref: Any,
    kv_in_vref: Any,
    o_vref: Any,
    schedule_ref: BatchedMixedSchedule,
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    acc_scratch_ref: Any,
    *,
    kv_lens_ref: Any,
    cu_q_lens_ref: Any,
    cfgs: BatchedMixedConfig,
    start_next_fetch: Callable[[], None],
    start_kv_writeback: Callable[[], None],
) -> None:
    if not cfgs.interpret_mode:
        kv_in_vref = pltpu.annotate(kv_in_vref, no_hazard_no_deps=True)

    always_true = kv_lens_ref[0] >= 0

    @pl.when(always_true)
    @jax.named_scope("stitch_and_issue_writeback")
    def _kv_store_fence() -> None:
        start_kv_writeback()

    start_next_fetch()

    tq_per_tok = cfgs.aligned_num_q_heads_per_kv_head
    tq_c = cfgs.bq_csz * tq_per_tok
    kv_vpu_iota = (
        lax.broadcasted_iota(
            jnp.int32,
            (cfgs.num_kv_heads, cfgs.bkv_sz, cfgs.aligned_head_dim),
            1,
        )
        if cfgs.sliding_window is not None
        else None
    )

    is_valids: list[jax.Array] = []
    kv_limits: list[jax.Array] = []
    causal_offs: list[jax.Array] = []
    is_lasts: list[jax.Array] = []
    valid_kv_masks: list[jax.Array] = []
    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        safe_s = jnp.maximum(s_idx, 0)
        q_idx = schedule_ref.q_idx[step, b_idx]
        k_idx = schedule_ref.k_idx[step, b_idx]
        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        kv_len_val = jnp.where(is_valid, kv_lens_ref[safe_s], 0)
        q_start_s = jnp.where(is_valid, cu_q_lens_ref[safe_s], 0)
        q_end_s = jnp.where(is_valid, cu_q_lens_ref[safe_s + 1], 0)
        q_offset = jnp.maximum(kv_len_val - (q_end_s - q_start_s), 0)
        q_base = jnp.where(is_valid, q_offset + q_idx * cfgs.bq_sz, 0)

        is_valids.append(is_valid)
        kv_limits.append(jnp.where(is_valid, kv_len_val - k_id, 1))
        causal_offs.append(k_id - q_base)
        is_lasts.append(schedule_ref.is_last_k[step, b_idx] == 1)

        if cfgs.sliding_window is not None:
            assert kv_vpu_iota is not None
            sw_low = jnp.where(is_valid, q_base - k_id - cfgs.sliding_window, -1)
            vpu_limit = jnp.where(is_valid, kv_len_val - k_id, 0)
            valid_kv_masks.append(
                jnp.logical_and(kv_vpu_iota < vpu_limit, kv_vpu_iota > sw_low)
            )

    q = _load_batched_q(
        q_vref,
        batch_size=cfgs.batch_size,
        num_kv_heads=cfgs.num_kv_heads,
        bq_sz=cfgs.bq_sz,
        tq_per_tok=tq_per_tok,
        packing_q=cfgs.packing_q,
        dtype_q=cfgs.dtype_q,
        aligned_head_dim=cfgs.aligned_head_dim,
        interpret_mode=cfgs.interpret_mode,
    )
    k, v = _load_batched_kv(
        kv_in_vref,
        batch_size=cfgs.batch_size,
        num_kv_heads=cfgs.num_kv_heads,
        bkv_sz=cfgs.bkv_sz,
        kv_hbm_stride=cfgs.kv_hbm_stride,
        bkv_stride=cfgs.bkv_stride,
        packing_kv=cfgs.packing_kv,
        dtype_kv=cfgs.dtype_kv,
        aligned_head_dim=cfgs.aligned_head_dim,
        interpret_mode=cfgs.interpret_mode,
    )

    if cfgs.sliding_window is not None:
        valid_kv = jnp.stack(valid_kv_masks, axis=0)
        k = jnp.where(valid_kv, k, jnp.zeros_like(k))
        v = jnp.where(valid_kv, v, jnp.zeros_like(v))

    kv_iota = lax.broadcasted_iota(jnp.int32, (cfgs.num_kv_heads, tq_c, cfgs.bkv_sz), 2)
    q_iota = (
        lax.broadcasted_iota(jnp.int32, (cfgs.num_kv_heads, tq_c, cfgs.bkv_sz), 1)
        // tq_per_tok
    )
    q_kv_diff = q_iota - kv_iota

    s_scale = cfgs.sm_scale
    if cfgs.q_scale is not None:
        s_scale *= cfgs.q_scale
    if cfgs.k_scale is not None:
        s_scale *= cfgs.k_scale
    acc_dt = cfgs.scratch_dtype
    mask_fill = jnp.asarray(cfgs.mask_value, dtype=acc_dt)
    m_reset = jnp.asarray(-jnp.inf, dtype=acc_dt)

    m_val = m_scratch_ref[...]
    l_val = l_scratch_ref[...]
    acc_val = acc_scratch_ref[...]

    def _run_pv_stage(
        p_chunk: jax.Array,
        alphas: list[jax.Array],
        row_start: int,
    ) -> jax.Array:
        pv = (
            lax.dot_general(
                p_chunk.reshape(-1, tq_c, cfgs.bkv_sz),
                v.reshape(-1, cfgs.bkv_sz, cfgs.aligned_head_dim),
                dimension_numbers=(([2], [1]), ([0], [0])),
                preferred_element_type=jnp.float32,
            )
            .astype(acc_dt)
            .reshape(cfgs.batch_size, cfgs.num_kv_heads, tq_c, cfgs.aligned_head_dim)
        )
        if cfgs.v_scale is not None:
            pv = pv * cfgs.v_scale

        o_prev_c = (
            acc_val
            if cfgs.bq_csz == cfgs.bq_sz
            else acc_val[:, row_start : row_start + tq_c, :]
        )
        o_next_list: list[jax.Array] = []
        for b_idx in range(cfgs.batch_size):
            alpha_full = _broadcast_minor(alphas[b_idx], o_prev_c.shape)
            o_next_b = alpha_full * o_prev_c + pv[b_idx]
            o_next_list.append(o_next_b)
            o_prev_c = o_next_b
        if cfgs.bq_csz == cfgs.bq_sz:
            acc_scratch_ref[...] = o_prev_c
        else:
            acc_scratch_ref[:, pl.ds(row_start, tq_c), :] = o_prev_c
        return jnp.stack(o_next_list, axis=0)

    l_new_chunks: list[jax.Array] = []
    acc_new_chunks: list[jax.Array] = []
    prev_p: jax.Array | None = None
    prev_alphas: list[jax.Array] | None = None
    prev_row_start = 0

    for bq_start in range(0, cfgs.bq_sz, cfgs.bq_csz):
        row_start = bq_start * tq_per_tok
        q_chunk = (
            q if cfgs.bq_csz == cfgs.bq_sz else q[:, :, row_start : row_start + tq_c, :]
        )
        if cfgs.q_scale is not None:
            q_scaled = q_chunk / cfgs.q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                finfo = jnp.finfo(k.dtype)
                q_scaled = jnp.clip(
                    q_scaled, min=float(finfo.min), max=float(finfo.max)
                )
            q_dot = q_scaled.astype(k.dtype)
        else:
            q_dot = q_chunk

        qk = lax.dot_general(
            q_dot.reshape(-1, tq_c, cfgs.aligned_head_dim),
            k.reshape(-1, cfgs.bkv_sz, cfgs.aligned_head_dim),
            dimension_numbers=(([2], [2]), ([0], [0])),
            preferred_element_type=jnp.float32,
        ).reshape(cfgs.batch_size, cfgs.num_kv_heads, tq_c, cfgs.bkv_sz)
        qk = (qk * s_scale).astype(acc_dt)
        if cfgs.soft_cap is not None:
            qk = (
                cfgs.soft_cap * jnp.tanh(qk.astype(jnp.float32) / cfgs.soft_cap)
            ).astype(acc_dt)

        masked_list: list[jax.Array] = []
        for b_idx in range(cfgs.batch_size):
            if cfgs.use_causal_mask:
                offset = causal_offs[b_idx] - bq_start
                mask_b = q_kv_diff >= offset
            else:
                mask_b = kv_iota < kv_limits[b_idx]
            if cfgs.sliding_window is not None:
                sw_off = jnp.where(
                    is_valids[b_idx],
                    cfgs.sliding_window + causal_offs[b_idx] - bq_start,
                    cfgs.bq_sz + 1,
                )
                mask_b = jnp.logical_and(mask_b, q_kv_diff < sw_off)
            masked_list.append(jnp.where(mask_b, qk[b_idx], mask_fill))
        qk_masked = jnp.stack(masked_list, axis=0)

        m_curr = jnp.max(qk_masked, axis=-1, keepdims=True)
        m_prev_c = (
            m_val
            if cfgs.bq_csz == cfgs.bq_sz
            else m_val[:, row_start : row_start + tq_c, :]
        )
        l_prev_c = (
            l_val
            if cfgs.bq_csz == cfgs.bq_sz
            else l_val[:, row_start : row_start + tq_c, :]
        )
        alpha_list: list[jax.Array] = []
        m_next_list: list[jax.Array] = []
        for b_idx in range(cfgs.batch_size):
            m_next_b = jnp.maximum(m_prev_c, m_curr[b_idx])
            alpha_b = jnp.exp(m_prev_c - m_next_b)
            alpha_list.append(alpha_b)
            m_next_list.append(m_next_b)
            m_prev_c = jnp.where(is_lasts[b_idx], m_reset, m_next_b)
        if cfgs.bq_csz == cfgs.bq_sz:
            m_scratch_ref[...] = m_prev_c
        else:
            m_scratch_ref[:, pl.ds(row_start, tq_c), :] = m_prev_c

        m_next_stack = jnp.stack(m_next_list, axis=0)
        p = jnp.exp(qk_masked - _broadcast_minor(m_next_stack, qk_masked.shape))
        p_rowsum = jnp.sum(p, axis=-1, keepdims=True, dtype=acc_dt)

        l_next_list: list[jax.Array] = []
        for b_idx in range(cfgs.batch_size):
            l_next_b = alpha_list[b_idx] * l_prev_c + p_rowsum[b_idx]
            l_next_list.append(l_next_b)
            l_prev_c = l_next_b
        if cfgs.bq_csz == cfgs.bq_sz:
            l_scratch_ref[...] = l_prev_c
        else:
            l_scratch_ref[:, pl.ds(row_start, tq_c), :] = l_prev_c
        l_new_chunks.append(jnp.stack(l_next_list, axis=0))

        if prev_p is not None:
            assert prev_alphas is not None
            acc_new_chunks.append(_run_pv_stage(prev_p, prev_alphas, prev_row_start))
        prev_p = p
        prev_alphas = alpha_list
        prev_row_start = row_start

    assert prev_p is not None and prev_alphas is not None
    acc_new_chunks.append(_run_pv_stage(prev_p, prev_alphas, prev_row_start))

    l_all = (
        l_new_chunks[0]
        if len(l_new_chunks) == 1
        else jnp.concatenate(l_new_chunks, axis=2)
    )
    acc_all = (
        acc_new_chunks[0]
        if len(acc_new_chunks) == 1
        else jnp.concatenate(acc_new_chunks, axis=2)
    )

    for b_idx in range(cfgs.batch_size):

        def _store_out(
            batch_acc: jax.Array, batch_l: jax.Array, _b: int = b_idx
        ) -> None:
            l_full = _broadcast_minor(batch_l, batch_acc.shape)
            if (
                cfgs.interpret_mode
                or cfgs.dtype_q == jnp.float32
                or cfgs.dtype_q == batch_l.dtype == jnp.bfloat16
            ):
                res = lax.div(batch_acc, l_full)
            else:
                res = batch_acc * pltpu.reciprocal(l_full, approx=True)
            out_b = res.astype(cfgs.dtype_q)
            if cfgs.interpret_mode:
                o_vref[_b] = out_b.reshape(o_vref.shape[1:])
            else:
                out_ref = (
                    o_vref.at[_b].bitcast(jnp.uint32).reshape(-1, cfgs.aligned_head_dim)
                )
                out_u32 = pltpu.bitcast(out_b, jnp.uint32).reshape(out_ref.shape)
                _strided_store(out_ref, 0, out_ref.shape[0], 1, out_u32)

        lax.cond(
            is_lasts[b_idx],
            _store_out,
            lambda _a, _l: None,
            acc_all[b_idx],
            l_all[b_idx],
        )


def _batched_mixed_rpa_pipeline(
    # Scalar prefetch (SMEM)
    kv_lens_ref: Any,
    cu_q_lens_ref: Any,
    page_indices_ref: Any,
    # Inputs (HBM + SMEM actual_steps)
    schedule_hbm_ref: BatchedMixedSchedule,
    q_hbm_ref: Any,
    new_kv_hbm_ref: Any,
    kv_cache_hbm_ref: Any,
    # Outputs (HBM, aliased to q_hbm_ref and kv_cache_hbm_ref)
    o_hbm_ref: Any,
    updated_kv_cache_hbm_ref: Any,
    # Scratch (VMEM + SMEM + DMA semaphores)
    bkv_x3_ref: Any,
    bq_x3_ref: Any,
    bo_x2_ref: Any,
    schedule_ref: BatchedMixedSchedule,
    q_recv_sem: Any,
    kv_recv_sem: Any,
    kv_send_sem: Any,
    o_send_sem: Any,
    sched_dma_sem: Any,
    m_scratch_ref: Any,
    l_scratch_ref: Any,
    acc_scratch_ref: Any,
    *,
    cfgs: BatchedMixedConfig,
) -> None:
    del updated_kv_cache_hbm_ref
    _init_batched_scratches(
        m_scratch_ref,
        l_scratch_ref,
        acc_scratch_ref,
        bq_x3_ref,
        bkv_x3_ref,
        interpret_mode=cfgs.interpret_mode,
    )
    copy_in_q, wait_in_q, copy_out_o, wait_out_o = _make_batched_q_o_dma(
        q_hbm_ref,
        o_hbm_ref,
        bq_x3_ref,
        bo_x2_ref,
        schedule_ref,
        q_recv_sem,
        o_send_sem,
        batch_size=cfgs.batch_size,
        interpret_mode=cfgs.interpret_mode,
    )

    def copy_in_kv(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = kv_recv_sem.at[slot]
        vmem_dst = bkv_x3_ref.at[slot, :, :, : cfgs.kv_hbm_stride]
        kv_hbm_flat = (
            None
            if cfgs.interpret_mode
            else kv_cache_hbm_ref.reshape(-1, *kv_cache_hbm_ref.shape[2:])
        )
        for b in range(cfgs.batch_size):
            for i in range(cfgs.bkv_p):
                sz = jnp.where(is_noop, 0, schedule_ref.dma_kv_cache[blk, b, i, 2])
                p_slot = jnp.where(sz > 0, schedule_ref.dma_kv_cache[blk, b, i, 0], 0)
                dst_off = jnp.where(sz > 0, schedule_ref.dma_kv_cache[blk, b, i, 1], 0)
                page_id = jnp.where(sz > 0, page_indices_ref[p_slot], 0)
                if cfgs.interpret_mode:
                    src = kv_cache_hbm_ref.at[page_id, pl.ds(0, sz)]
                else:
                    assert kv_hbm_flat is not None
                    src = kv_hbm_flat.at[pl.ds(page_id * cfgs.page_size, sz)]
                pltpu.make_async_copy(
                    src,
                    vmem_dst.at[b, pl.ds(dst_off, sz)],
                    sem,
                ).start()
            new_sz = jnp.where(is_noop, 0, schedule_ref.dma_kv_new_in[blk, b, 2])
            fetch_hbm = jnp.where(new_sz > 0, schedule_ref.dma_kv_new_in[blk, b, 0], 0)
            dst_vmem_off = jnp.where(
                new_sz > 0, schedule_ref.dma_kv_new_in[blk, b, 1], 0
            )
            pltpu.make_async_copy(
                new_kv_hbm_ref.at[pl.ds(fetch_hbm, new_sz)],
                vmem_dst.at[b, pl.ds(dst_vmem_off, new_sz)],
                sem,
            ).start()

    def wait_in_kv(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = kv_recv_sem.at[slot]
        if cfgs.interpret_mode:
            vmem_dst = bkv_x3_ref.at[slot, :, :, : cfgs.kv_hbm_stride]
            for b in range(cfgs.batch_size):
                lane_sz: Any = schedule_ref.dma_kv_new_in[blk, b, 2]
                for i in range(cfgs.bkv_p):
                    lane_sz = lane_sz + schedule_ref.dma_kv_cache[blk, b, i, 2]
                lane_sz = jnp.where(is_noop, 0, lane_sz)
                pltpu.make_async_copy(
                    vmem_dst.at[b, pl.ds(0, lane_sz)],
                    vmem_dst.at[b, pl.ds(0, lane_sz)],
                    sem,
                ).wait()
            return
        wait_lanes = jnp.where(is_noop, 0, schedule_ref.total_wait_kv_in[blk])
        flat = bkv_x3_ref.at[slot].bitcast(jnp.uint32).reshape((-1, _TPU_NUM_LANES))
        pltpu.make_async_copy(
            flat.at[pl.ds(0, wait_lanes), :],
            flat.at[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()

    def copy_out_kv(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = kv_send_sem.at[slot]
        vmem_src = bkv_x3_ref.at[slot, :, :, : cfgs.kv_hbm_stride]
        kv_hbm_flat = (
            None
            if cfgs.interpret_mode
            else kv_cache_hbm_ref.reshape(-1, *kv_cache_hbm_ref.shape[2:])
        )
        for b in range(cfgs.batch_size):
            for i in range(cfgs.bkv_p):
                sz = jnp.where(is_noop, 0, schedule_ref.dma_kv_wb[blk, b, i, 3])
                wb_p_slot = jnp.where(sz > 0, schedule_ref.dma_kv_wb[blk, b, i, 0], 0)
                wb_p_off = jnp.where(sz > 0, schedule_ref.dma_kv_wb[blk, b, i, 1], 0)
                src_vmem_off = jnp.where(
                    sz > 0, schedule_ref.dma_kv_wb[blk, b, i, 2], 0
                )
                page_id = jnp.where(sz > 0, page_indices_ref[wb_p_slot], 0)
                if cfgs.interpret_mode:
                    dst = kv_cache_hbm_ref.at[page_id, pl.ds(wb_p_off, sz)]
                else:
                    assert kv_hbm_flat is not None
                    dst = kv_hbm_flat.at[pl.ds(page_id * cfgs.page_size + wb_p_off, sz)]
                pltpu.make_async_copy(
                    vmem_src.at[b, pl.ds(src_vmem_off, sz)],
                    dst,
                    sem,
                ).start()

    def wait_out_kv(step: Any, slot: Any, total_steps: Any) -> None:
        is_noop, blk = _step_guard(step, total_steps)
        sem = kv_send_sem.at[slot]
        if cfgs.interpret_mode:
            vmem_src = bkv_x3_ref.at[slot, :, :, : cfgs.kv_hbm_stride]
            for b in range(cfgs.batch_size):
                wb_lane_sz: Any = 0
                for i in range(cfgs.bkv_p):
                    wb_lane_sz = wb_lane_sz + schedule_ref.dma_kv_wb[blk, b, i, 3]
                wb_lane_sz = jnp.where(is_noop, 0, wb_lane_sz)
                pltpu.make_async_copy(
                    vmem_src.at[b, pl.ds(0, wb_lane_sz)],
                    vmem_src.at[b, pl.ds(0, wb_lane_sz)],
                    sem,
                ).wait()
            return
        wait_lanes = jnp.where(is_noop, 0, schedule_ref.total_wait_kv_out[blk])
        flat = bkv_x3_ref.at[slot].bitcast(jnp.uint32).reshape((-1, _TPU_NUM_LANES))
        pltpu.make_async_copy(
            flat.at[pl.ds(0, wait_lanes), :],
            flat.at[pl.ds(0, wait_lanes), :],
            sem,
        ).wait()

    def compute_fn(
        step: jax.Array,
        comp_slot: jax.Array,
        out_slot: jax.Array,
        start_next_fetch: Callable[[], None],
        start_kv_writeback: Callable[[], None],
    ) -> None:
        _batched_mixed_rpa_body(
            step,
            bq_x3_ref.at[comp_slot],
            bkv_x3_ref.at[comp_slot],
            bo_x2_ref.at[out_slot],
            schedule_ref,
            m_scratch_ref,
            l_scratch_ref,
            acc_scratch_ref,
            kv_lens_ref=kv_lens_ref,
            cu_q_lens_ref=cu_q_lens_ref,
            cfgs=cfgs,
            start_next_fetch=start_next_fetch,
            start_kv_writeback=start_kv_writeback,
        )

    _run_batched_schedule_chunks(
        schedule_hbm_ref,
        schedule_ref,
        sched_dma_sem,
        compute_fn,
        max_steps_ub=cfgs.max_steps_ub,
        copy_in_q=copy_in_q,
        wait_in_q=wait_in_q,
        copy_in_kv=copy_in_kv,
        wait_in_kv=wait_in_kv,
        copy_out_kv=copy_out_kv,
        wait_out_kv=wait_out_kv,
        copy_out_o=copy_out_o,
        wait_out_o=wait_out_o,
    )


def run_batched_mixed_rpa(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    page_indices: jax.Array,
    schedule_hbm: BatchedMixedSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    *,
    cfgs: BatchedMixedConfig,
    tpu_version: int,
) -> tuple[jax.Array, jax.Array]:
    """Execute the triple-buffered batched mixed/prefill attention Pallas kernel."""
    return _invoke_batched_rpa_pallas(
        _batched_mixed_rpa_pipeline,
        (kv_lens, cu_q_lens, page_indices),
        schedule_hbm,
        BatchedMixedSchedule.create_shape_dtype(cfgs, steps=cfgs.max_steps_ub),
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
        cfgs=cfgs,
        tpu_version=tpu_version,
    )
