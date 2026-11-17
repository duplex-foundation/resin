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
"""Pallas Mosaic kernel body for Ragged Paged Attention v3.

Adapted from vllm-project/tpu-inference (Apache-2.0) at revision
d9c7e19da8c497ec63b9381bb0bc9b2a5e0d4bb2:
tpu_inference/kernels/ragged_paged_attention/v3/{kernel.py,util.py}.
"""

from __future__ import annotations

from enum import Enum

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

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])],
            axis=-1,
        )[..., : shape[-1]]

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

    def strided_load(ref, start, sz, step, *, dtype=None):
        r, l_dim = ref.shape
        folds = l_dim // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        vec = jnp.concat(
            [ref[pl.ds(start + i, sz // step, step)] for i in range(folds)],
            axis=1,
        )
        if dtype is not None:
            vec = pltpu.bitcast(vec, dtype)
        return vec

    def strided_store(ref, start, sz, step, val):
        r, l_dim = ref.shape
        folds = l_dim // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        for i in range(folds):
            ref[pl.ds(start + i, sz // step, step)] = val[:, i * 128 : (i + 1) * 128]

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
