"""Row-parallel TPU7x FP8 W8A8 matmul overlapped with cross-die async all-reduce."""

from __future__ import annotations

import functools
from typing import Any, overload

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from resin.backend.tpu.kernels._rpa_v3_pallas import align_to, get_tpu_version
from resin.backend.tpu.kernels.fp8_matmul import (
    _DEVICE_VMEM_LIMIT,
    Fp8MatmulTileSizes,
    MatmulBackend,
    _fp8_bound,
    _scale_shard_spec,
    _validate_matmul_shapes,
    _validate_x_scale_args,
    get_tuned_block_sizes,
    quantize_fp8_act,
    quantize_fp8_with_absmax,
    safe_fp8_scale,
    sharded_fp8_matmul,
)
from resin.backend.tpu.kernels.norm_rope import fused_add_rms_norm_fp8_quant
from resin.backend.tpu.mesh import TP_AXIS

_ASYNC_AR_MAX_VMEM_BYTES = 40 * 1024 * 1024
_ASYNC_AR_VMEM_SLACK_BYTES = 4 * 1024 * 1024
_ASYNC_AR_MIN_VMEM_LIMIT_BYTES = 8 * 1024 * 1024
_ASYNC_AR_HBM_RES_MIN_BATCH = 128
_ASYNC_AR_FUSED_NORM_MAX_BATCH = 96


def estimate_async_ar_vmem_bytes(
    batch_block: int,
    out_block: int,
    in_block: int,
    n_out: int,
    n_w_slots: int,
    *,
    x_itemsize: int = 1,
    w_itemsize: int = 1,
    reduce_itemsize: int = 2,
    out_itemsize: int = 2,
    fuse_norm_quant: bool = False,
    q_out_itemsize: int = 1,
    norm_w_itemsize: int = 2,
) -> int:
    """Return total VMEM bytes (inputs + output + scratch) for the async all-reduce kernel."""
    padded_n = n_out * out_block
    x_bytes = batch_block * in_block * x_itemsize
    xs_bytes = batch_block * 4
    out_bytes = batch_block * padded_n * out_itemsize
    w_buf_bytes = n_w_slots * out_block * in_block * w_itemsize
    ws_buf_bytes = n_w_slots * out_block * 4
    local_peer_bytes = 2 * n_out * batch_block * out_block * reduce_itemsize
    base = (
        x_bytes + xs_bytes + out_bytes + w_buf_bytes + ws_buf_bytes + local_peer_bytes
    )
    if not fuse_norm_quant:
        return base
    res_in_bytes = 0 if batch_block >= _ASYNC_AR_HBM_RES_MIN_BATCH else out_bytes
    xq_out_bytes = batch_block * padded_n * q_out_itemsize
    q_scale_bytes = batch_block * 4
    norm_w_bytes = padded_n * 8 * norm_w_itemsize
    return base + res_in_bytes + xq_out_bytes + q_scale_bytes + norm_w_bytes


_ASYNC_AR_TUNED_BATCH_BUCKETS = (8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320)


def get_async_ar_tile_sizes(
    orig_batch: int,
    orig_n: int,
    orig_k: int,
) -> Fp8MatmulTileSizes:
    """Select ``(batch_block, out_block, in_block, w_buffer_count)`` for ``tp=2`` async all-reduce."""
    batch_block = align_to(max(8, orig_batch), 8)
    lookup_batch = min(
        (b for b in _ASYNC_AR_TUNED_BATCH_BUCKETS if b >= batch_block),
        default=batch_block,
    )
    tuned = get_tuned_block_sizes(
        lookup_batch,
        orig_n,
        orig_k,
        tpu_version=max(get_tpu_version(), 7),
        quantize_activation=False,
    )
    in_block = align_to(max(128, orig_k), 128)
    out_block = tuned.out_block_size
    if orig_n >= 256 and out_block >= orig_n:
        out_block = min(out_block, align_to(max(128, orig_n // 4), 128))
    return Fp8MatmulTileSizes(batch_block, out_block, in_block, tuned.w_buffer_count)


def _validate_async_ar_tiles(
    orig_batch: int,
    orig_n: int,
    orig_k: int,
    tiles: Fp8MatmulTileSizes,
) -> tuple[int, int, int, int]:
    """Validate single-(batch, K) async all-reduce tiles and return ``(padded_batch, padded_n, padded_k, n_out)``."""
    if orig_batch <= 0:
        raise ValueError(f"Expected orig_batch > 0, got {orig_batch}")
    batch_block, out_block, in_block, w_buf = tiles
    if (
        batch_block <= 0
        or out_block <= 0
        or in_block <= 0
        or batch_block % 8 != 0
        or out_block % 128 != 0
        or in_block % 128 != 0
    ):
        raise ValueError(
            f"Invalid async all-reduce tile sizes {(batch_block, out_block, in_block)}"
        )
    if w_buf < 2:
        raise ValueError(f"w_buffer_count must be >= 2, got {w_buf}")

    padded_batch = align_to(orig_batch, batch_block)
    padded_n = align_to(orig_n, out_block)
    padded_k = align_to(orig_k, in_block)
    if padded_batch != batch_block or padded_k != in_block:
        raise ValueError(
            "Async all-reduce requires single (batch, K) tile; "
            f"got padded {(padded_batch, padded_k)} vs {(batch_block, in_block)}"
        )
    n_out = padded_n // out_block
    if n_out < 2:
        raise ValueError(
            "Async all-reduce requires at least 2 output tiles, "
            f"got {n_out=} for N={orig_n}, {out_block=}"
        )
    return padded_batch, padded_n, padded_k, n_out


def _pad_async_ar_inputs(
    x_2d: jax.Array,
    x_scale_row: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    padded_batch: int,
    padded_n: int,
    padded_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    orig_batch, orig_k = x_2d.shape
    orig_n = w.shape[0]
    if orig_batch < padded_batch:
        pad_b = padded_batch - orig_batch
        x_2d = jnp.pad(x_2d, ((0, pad_b), (0, 0)))
        x_scale_row = jnp.pad(x_scale_row, ((0, 0), (0, pad_b)))
    if orig_k < padded_k:
        x_2d = jnp.pad(x_2d, ((0, 0), (0, padded_k - orig_k)))

    if orig_n == padded_n and orig_k == padded_k:
        w_in = w
        ws_2d = jnp.expand_dims(w_scale.astype(jnp.float32), axis=0)
    else:
        w_in = jnp.pad(w, ((0, padded_n - orig_n), (0, padded_k - orig_k)))
        ws_2d = jnp.expand_dims(
            jnp.pad(w_scale.astype(jnp.float32), (0, padded_n - orig_n)), axis=0
        )
    return x_2d, x_scale_row, w_in, ws_2d


def _can_use_async_ar(
    *,
    tp: int,
    orig_batch: int,
    orig_n: int,
    local_k: int,
    w_dtype: Any,
    out_dtype: Any,
    reduce_dtype: Any | None,
    fuse_norm_quant: bool = False,
    q_out_dtype: Any | None = None,
) -> bool:
    if tp != 2 or orig_batch <= 0 or local_k < 128:
        return False
    tiles = get_async_ar_tile_sizes(orig_batch, orig_n, local_k)
    try:
        _, _, _, n_out = _validate_async_ar_tiles(orig_batch, orig_n, local_k, tiles)
    except ValueError:
        return False
    target_out = jnp.dtype(out_dtype)
    eff_reduce = jnp.dtype(target_out if reduce_dtype is None else reduce_dtype)
    w_itemsize = jnp.dtype(w_dtype).itemsize
    q_itemsize = jnp.dtype(w_dtype if q_out_dtype is None else q_out_dtype).itemsize
    vmem_bytes = estimate_async_ar_vmem_bytes(
        tiles.batch_block_size,
        tiles.out_block_size,
        tiles.in_block_size,
        n_out,
        min(tiles.w_buffer_count, n_out),
        x_itemsize=w_itemsize,
        w_itemsize=w_itemsize,
        reduce_itemsize=eff_reduce.itemsize,
        out_itemsize=target_out.itemsize,
        fuse_norm_quant=fuse_norm_quant,
        q_out_itemsize=q_itemsize,
    )
    return vmem_bytes <= _ASYNC_AR_MAX_VMEM_BYTES


def _run_async_ar_pipeline(
    x_ref: Any,
    w_hbm_ref: Any,
    ws_hbm_ref: Any,
    xs_ref: Any,
    w_buf_ref: Any,
    ws_buf_ref: Any,
    local_buf: Any,
    peer_buf: Any,
    w_sems: Any,
    ws_sems: Any,
    send_sems: Any,
    recv_sems: Any,
    *,
    out_block: int,
    n_out: int,
    n_w_slots: int,
    tp_axis: str,
    eff_reduce: jnp.dtype,
    target_out_dtype: jnp.dtype,
    store_reduced_tile: Any,
) -> None:
    tp_size = jax.lax.axis_size(tp_axis)
    if tp_size != 2:
        raise ValueError(
            f"async all-reduce kernel requires axis_size({tp_axis!r}) == 2, got {tp_size}"
        )
    peer_id = 1 - jax.lax.axis_index(tp_axis)
    xs_col = jnp.transpose(xs_ref[...])

    def _start_w_dma(tile_idx: Any, slot: Any) -> None:
        pltpu.make_async_copy(
            w_hbm_ref.at[pl.ds(tile_idx * out_block, out_block), :],
            w_buf_ref.at[slot],
            w_sems.at[slot],
        ).start()
        pltpu.make_async_copy(
            ws_hbm_ref.at[:, pl.ds(tile_idx * out_block, out_block)],
            ws_buf_ref.at[slot],
            ws_sems.at[slot],
        ).start()

    def _wait_w_and_send_tile(tile_idx: Any, slot: Any) -> None:
        pltpu.make_async_copy(
            w_hbm_ref.at[pl.ds(tile_idx * out_block, out_block), :],
            w_buf_ref.at[slot],
            w_sems.at[slot],
        ).wait()
        pltpu.make_async_copy(
            ws_hbm_ref.at[:, pl.ds(tile_idx * out_block, out_block)],
            ws_buf_ref.at[slot],
            ws_sems.at[slot],
        ).wait()
        acc = jax.lax.dot_general(
            x_ref[...],
            w_buf_ref[slot],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        scaled = (acc * ws_buf_ref[slot]) * xs_col
        local_buf[tile_idx] = (
            scaled.astype(eff_reduce) if eff_reduce != jnp.float32 else scaled
        )
        pltpu.make_async_remote_copy(
            src_ref=local_buf.at[tile_idx],
            dst_ref=peer_buf.at[tile_idx],
            send_sem=send_sems.at[tile_idx],
            recv_sem=recv_sems.at[tile_idx],
            device_id={tp_axis: peer_id},
        ).start()

    def _finish_tile(tile_idx: Any) -> None:
        rc = pltpu.make_async_remote_copy(
            src_ref=local_buf.at[tile_idx],
            dst_ref=peer_buf.at[tile_idx],
            send_sem=send_sems.at[tile_idx],
            recv_sem=recv_sems.at[tile_idx],
            device_id={tp_axis: peer_id},
        )
        rc.wait_recv()
        reduced = local_buf[tile_idx] + peer_buf[tile_idx]
        reduced_out = (
            reduced.astype(target_out_dtype)
            if target_out_dtype != eff_reduce
            else reduced
        )
        rc.wait_send()
        store_reduced_tile(tile_idx, reduced_out)

    for slot in range(n_w_slots):
        _start_w_dma(slot, slot)

    barrier_sem = pltpu.get_barrier_semaphore()
    pl.semaphore_signal(barrier_sem, 1, device_id={tp_axis: peer_id})
    pl.semaphore_wait(barrier_sem, 1)

    _wait_w_and_send_tile(0, 0)
    if n_out > n_w_slots:
        _start_w_dma(n_w_slots, 0)

    def _step(o: jax.Array, _: None) -> None:
        slot = jax.lax.rem(o, n_w_slots)
        _wait_w_and_send_tile(o, slot)
        if n_out > n_w_slots:
            next_w = o + n_w_slots

            @pl.when(next_w < n_out)
            def _prefetch_next() -> None:
                _start_w_dma(next_w, slot)

        _finish_tile(o - 1)
        return None

    jax.lax.fori_loop(1, n_out, _step, None)
    _finish_tile(n_out - 1)


def _fp8_row_matmul_async_ar_kernel(
    x_ref: Any,
    w_hbm_ref: Any,
    ws_hbm_ref: Any,
    xs_ref: Any,
    out_ref: Any,
    w_buf_ref: Any,
    ws_buf_ref: Any,
    local_buf: Any,
    peer_buf: Any,
    w_sems: Any,
    ws_sems: Any,
    send_sems: Any,
    recv_sems: Any,
    *,
    out_block: int,
    n_out: int,
    n_w_slots: int,
    tp_axis: str,
    eff_reduce: jnp.dtype,
    target_out_dtype: jnp.dtype,
) -> None:
    def _store_tile(tile_idx: Any, reduced_out: jax.Array) -> None:
        out_ref[:, pl.ds(tile_idx * out_block, out_block)] = reduced_out

    _run_async_ar_pipeline(
        x_ref,
        w_hbm_ref,
        ws_hbm_ref,
        xs_ref,
        w_buf_ref,
        ws_buf_ref,
        local_buf,
        peer_buf,
        w_sems,
        ws_sems,
        send_sems,
        recv_sems,
        out_block=out_block,
        n_out=n_out,
        n_w_slots=n_w_slots,
        tp_axis=tp_axis,
        eff_reduce=eff_reduce,
        target_out_dtype=target_out_dtype,
        store_reduced_tile=_store_tile,
    )


def _fp8_row_matmul_async_ar_fused_norm_kernel(
    x_ref: Any,
    w_hbm_ref: Any,
    ws_hbm_ref: Any,
    xs_ref: Any,
    res_ref: Any,
    norm_w_ref: Any,
    xq_out_ref: Any,
    scale_out_ref: Any,
    next_res_ref: Any,
    w_buf_ref: Any,
    ws_buf_ref: Any,
    local_buf: Any,
    peer_buf: Any,
    w_sems: Any,
    ws_sems: Any,
    send_sems: Any,
    recv_sems: Any,
    res_sem: Any,
    *,
    out_block: int,
    n_out: int,
    n_w_slots: int,
    orig_n: int,
    tp_axis: str,
    eff_reduce: jnp.dtype,
    target_out_dtype: jnp.dtype,
    stream_res_from_hbm: bool,
    eps: float,
    bound: float,
) -> None:
    padded_n = n_out * out_block
    target_q_dtype = xq_out_ref.dtype

    if stream_res_from_hbm:
        pltpu.make_async_copy(res_ref, next_res_ref, res_sem).start()

        def _store_tile(tile_idx: Any, reduced_out: jax.Array) -> None:
            if eff_reduce == target_out_dtype:
                peer_buf[tile_idx] = reduced_out
            else:
                peer_buf[tile_idx] = jax.lax.optimization_barrier(reduced_out).astype(
                    eff_reduce
                )

    else:

        def _store_tile(tile_idx: Any, reduced_out: jax.Array) -> None:
            next_res_ref[:, pl.ds(tile_idx * out_block, out_block)] = reduced_out

    _run_async_ar_pipeline(
        x_ref,
        w_hbm_ref,
        ws_hbm_ref,
        xs_ref,
        w_buf_ref,
        ws_buf_ref,
        local_buf,
        peer_buf,
        w_sems,
        ws_sems,
        send_sems,
        recv_sems,
        out_block=out_block,
        n_out=n_out,
        n_w_slots=n_w_slots,
        tp_axis=tp_axis,
        eff_reduce=eff_reduce,
        target_out_dtype=target_out_dtype,
        store_reduced_tile=_store_tile,
    )

    if stream_res_from_hbm:
        pltpu.make_async_copy(res_ref, next_res_ref, res_sem).wait()
        for t in range(n_out):
            sl = pl.ds(t * out_block, out_block)
            next_res_ref[:, sl] = (
                peer_buf[t].astype(jnp.float32)
                + next_res_ref[:, sl].astype(jnp.float32)
            ).astype(target_out_dtype)
        next_res = next_res_ref[...]
    else:
        next_res = (
            next_res_ref[...].astype(jnp.float32) + res_ref[...].astype(jnp.float32)
        ).astype(target_out_dtype)

    if orig_n < padded_n:
        valid_mask = (jnp.arange(padded_n, dtype=jnp.int32) < orig_n)[None, :]
        next_res = jnp.where(valid_mask, next_res, jnp.zeros_like(next_res))
    if not stream_res_from_hbm or orig_n < padded_n:
        next_res_ref[...] = next_res

    next_res_f32 = next_res.astype(jnp.float32)
    if orig_n == padded_n:
        mean_sq = jnp.mean(jnp.square(next_res_f32), axis=-1, keepdims=True)
    else:
        mean_sq = jnp.sum(jnp.square(next_res_f32), axis=-1, keepdims=True) / float(
            orig_n
        )
    inv_rms = jax.lax.rsqrt(mean_sq + eps)
    nw_f32 = norm_w_ref[...].astype(jnp.float32)[None, :]
    normed = (next_res_f32 * inv_rms * nw_f32).astype(target_out_dtype)
    normed_f32 = normed.astype(jnp.float32)
    if orig_n < padded_n:
        normed_f32 = jnp.where(valid_mask, normed_f32, 0.0)
    row_absmax = jnp.max(jnp.abs(normed_f32), axis=-1, keepdims=True)
    scale_col = safe_fp8_scale(row_absmax, bound)
    xq_out_ref[...] = jnp.clip(normed_f32 / scale_col, -bound, bound).astype(
        target_q_dtype
    )
    scale_out_ref[...] = jnp.transpose(scale_col)


@overload
def pallas_fp8_row_matmul_async_ar(
    x_q: jax.Array,
    x_scale: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    tp_axis: str = TP_AXIS,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    residual: None = None,
    norm_weight: None = None,
    eps: float = 1e-6,
    quantize_output_dtype: None = None,
    transposed_scale: bool = True,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    collective_id: int = 0,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array: ...


@overload
def pallas_fp8_row_matmul_async_ar(
    x_q: jax.Array,
    x_scale: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    tp_axis: str = TP_AXIS,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    residual: jax.Array,
    norm_weight: jax.Array,
    eps: float = 1e-6,
    quantize_output_dtype: Any,
    transposed_scale: bool = True,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    collective_id: int = 0,
    interpret: bool | pltpu.InterpretParams = False,
) -> tuple[jax.Array, jax.Array, jax.Array]: ...


def pallas_fp8_row_matmul_async_ar(
    x_q: jax.Array,
    x_scale: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    tp_axis: str = TP_AXIS,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    residual: jax.Array | None = None,
    norm_weight: jax.Array | None = None,
    eps: float = 1e-6,
    quantize_output_dtype: Any | None = None,
    transposed_scale: bool = True,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    collective_id: int = 0,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
    """Compute local FP8 W8A8 ``x_q @ w.T`` and cross-die ``psum`` via ``make_async_remote_copy``."""
    _validate_matmul_shapes(x_q.shape[-1], w, w_scale)
    if x_q.dtype != w.dtype:
        raise ValueError(
            f"pallas_fp8_row_matmul_async_ar expects pre-quantized {w.dtype} x_q, got {x_q.dtype}"
        )
    fuse_norm_quant = residual is not None
    if fuse_norm_quant:
        if norm_weight is None or quantize_output_dtype is None:
            raise ValueError(
                "pallas_fp8_row_matmul_async_ar with residual requires norm_weight and quantize_output_dtype"
            )
    elif norm_weight is not None or quantize_output_dtype is not None:
        raise ValueError(
            "norm_weight and quantize_output_dtype require residual to be provided"
        )
    is_interpret = bool(interpret)
    tpu_version = get_tpu_version()
    if not is_interpret and tpu_version < 7:
        raise ValueError(
            f"pallas_fp8_row_matmul_async_ar requires TPU7x+ or interpret=True, got {tpu_version=}"
        )
    try:
        tp_size = jax.lax.axis_size(tp_axis)
    except NameError as exc:
        raise ValueError(
            f"pallas_fp8_row_matmul_async_ar requires mapped axis {tp_axis!r} of size 2"
        ) from exc
    if tp_size != 2:
        raise ValueError(
            f"pallas_fp8_row_matmul_async_ar requires axis_size({tp_axis!r}) == 2, got {tp_size}"
        )

    leading = x_q.shape[:-1]
    orig_k = x_q.shape[-1]
    orig_n = w.shape[0]
    x_2d = x_q if x_q.ndim == 2 else x_q.reshape(-1, orig_k)
    orig_batch = x_2d.shape[0]
    if x_scale.shape == (1, orig_batch) and x_scale.dtype == jnp.float32:
        x_scale_row = x_scale
    else:
        x_scale_row = x_scale.reshape(1, orig_batch).astype(jnp.float32)

    eff_tiles = (
        get_async_ar_tile_sizes(orig_batch, orig_n, orig_k)
        if tile_sizes is None
        else tile_sizes
    )
    batch_block, out_block, in_block, w_buf = eff_tiles
    padded_batch, padded_n, padded_k, n_out = _validate_async_ar_tiles(
        orig_batch, orig_n, orig_k, eff_tiles
    )
    if fuse_norm_quant and (
        orig_batch > _ASYNC_AR_FUSED_NORM_MAX_BATCH
        or batch_block > _ASYNC_AR_FUSED_NORM_MAX_BATCH
    ):
        raise ValueError(
            "pallas_fp8_row_matmul_async_ar fused norm epilogue requires "
            f"orig_batch and batch_block_size <= {_ASYNC_AR_FUSED_NORM_MAX_BATCH}, "
            f"got orig_batch={orig_batch}, batch_block_size={batch_block}"
        )
    target_out_dtype = jnp.dtype(out_dtype)
    eff_reduce = jnp.dtype(target_out_dtype if reduce_dtype is None else reduce_dtype)
    n_w_slots = min(w_buf, n_out)
    w_itemsize = jnp.dtype(w.dtype).itemsize
    target_q_dtype = (
        jnp.dtype(quantize_output_dtype)
        if quantize_output_dtype is not None
        else jnp.dtype(w.dtype)
    )
    upper = _DEVICE_VMEM_LIMIT.get(max(tpu_version, 7), 48 * 1024 * 1024)
    norm_w_itemsize = (
        jnp.dtype(norm_weight.dtype).itemsize if norm_weight is not None else 2
    )
    vmem_bytes = estimate_async_ar_vmem_bytes(
        batch_block,
        out_block,
        in_block,
        n_out,
        n_w_slots,
        x_itemsize=w_itemsize,
        w_itemsize=w_itemsize,
        reduce_itemsize=eff_reduce.itemsize,
        out_itemsize=target_out_dtype.itemsize,
        fuse_norm_quant=fuse_norm_quant,
        q_out_itemsize=target_q_dtype.itemsize,
        norm_w_itemsize=norm_w_itemsize,
    )
    if vmem_bytes > upper:
        raise ValueError(
            f"Async all-reduce VMEM estimate {vmem_bytes} bytes exceeds limit {upper} bytes "
            f"for tiles={eff_tiles}, shape=({orig_batch}, {orig_n}, {orig_k})"
        )

    x_2d, x_scale_row, w_in, ws_2d = _pad_async_ar_inputs(
        x_2d,
        x_scale_row,
        w,
        w_scale,
        padded_batch=padded_batch,
        padded_n=padded_n,
        padded_k=padded_k,
    )

    interpret_arg: bool | pltpu.InterpretParams = (
        interpret
        if isinstance(interpret, pltpu.InterpretParams)
        else (pltpu.InterpretParams() if interpret else False)
    )
    vmem_limit_bytes = (
        min(
            max(
                vmem_bytes + _ASYNC_AR_VMEM_SLACK_BYTES,
                _ASYNC_AR_MIN_VMEM_LIMIT_BYTES,
            ),
            upper,
        )
        if not is_interpret
        else None
    )

    buf_sfx = f"_b{w_buf}" if w_buf != 2 else ""
    if fuse_norm_quant:
        assert residual is not None and norm_weight is not None
        expected_res_shape = (*leading, orig_n)
        if residual.shape != expected_res_shape:
            raise ValueError(
                f"Expected residual shape {expected_res_shape}, got {residual.shape}"
            )
        res_2d = (
            residual if residual.ndim == 2 else residual.reshape(orig_batch, orig_n)
        ).astype(target_out_dtype)
        if orig_batch < padded_batch or orig_n < padded_n:
            res_2d = jnp.pad(
                res_2d, ((0, padded_batch - orig_batch), (0, padded_n - orig_n))
            )
        nw_1d = (
            norm_weight
            if norm_weight.shape == (orig_n,)
            else norm_weight.reshape(orig_n)
        )
        if orig_n < padded_n:
            nw_1d = jnp.pad(nw_1d, (0, padded_n - orig_n))

        bound = _fp8_bound(target_q_dtype)
        stream_res_from_hbm = batch_block >= _ASYNC_AR_HBM_RES_MIN_BATCH
        res_in_spec = (
            pl.BlockSpec(memory_space=pltpu.HBM)
            if stream_res_from_hbm
            else pl.BlockSpec((batch_block, padded_n), lambda: (0, 0))
        )
        fused_kernel = pl.pallas_call(
            functools.partial(
                _fp8_row_matmul_async_ar_fused_norm_kernel,
                out_block=out_block,
                n_out=n_out,
                n_w_slots=n_w_slots,
                orig_n=orig_n,
                tp_axis=tp_axis,
                eff_reduce=eff_reduce,
                target_out_dtype=target_out_dtype,
                stream_res_from_hbm=stream_res_from_hbm,
                eps=eps,
                bound=bound,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((batch_block, in_block), lambda: (0, 0)),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                    pl.BlockSpec((1, batch_block), lambda: (0, 0)),
                    res_in_spec,
                    pl.BlockSpec((padded_n,), lambda: (0,)),
                ],
                out_specs=[
                    pl.BlockSpec((batch_block, padded_n), lambda: (0, 0)),
                    pl.BlockSpec((1, batch_block), lambda: (0, 0)),
                    pl.BlockSpec((batch_block, padded_n), lambda: (0, 0)),
                ],
                scratch_shapes=[
                    pltpu.VMEM((n_w_slots, out_block, in_block), w_in.dtype),
                    pltpu.VMEM((n_w_slots, 1, out_block), jnp.float32),
                    pltpu.VMEM((n_out, batch_block, out_block), eff_reduce),
                    pltpu.VMEM((n_out, batch_block, out_block), eff_reduce),
                    pltpu.SemaphoreType.DMA((n_w_slots,)),
                    pltpu.SemaphoreType.DMA((n_w_slots,)),
                    pltpu.SemaphoreType.DMA((n_out,)),
                    pltpu.SemaphoreType.DMA((n_out,)),
                    pltpu.SemaphoreType.DMA(()) if stream_res_from_hbm else None,
                ],
                grid=(),
            ),
            out_shape=[
                jax.ShapeDtypeStruct((padded_batch, padded_n), target_q_dtype),
                jax.ShapeDtypeStruct((1, padded_batch), jnp.float32),
                jax.ShapeDtypeStruct((padded_batch, padded_n), target_out_dtype),
            ],
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=vmem_limit_bytes,
                collective_id=collective_id,
            ),
            interpret=interpret_arg,
            name=f"fp8_w8a8_async_ar_{batch_block}_{out_block}_{in_block}{buf_sfx}_add_norm_q",
        )
        if not is_interpret:
            w_in = pltpu.with_memory_space_constraint(w_in, pltpu.HBM)
            ws_2d = pltpu.with_memory_space_constraint(ws_2d, pltpu.HBM)
            if stream_res_from_hbm:
                res_2d = pltpu.with_memory_space_constraint(res_2d, pltpu.HBM)
        xq_pad, scale_pad, next_res_pad = fused_kernel(
            x_2d, w_in, ws_2d, x_scale_row, res_2d, nw_1d
        )
        xq_out = xq_pad[:orig_batch, :orig_n].reshape(*leading, orig_n)
        next_res = next_res_pad[:orig_batch, :orig_n].reshape(*leading, orig_n)
        scale_row = scale_pad[:, :orig_batch]
        if transposed_scale:
            return xq_out, scale_row, next_res
        return xq_out, jnp.transpose(scale_row).reshape(*leading, 1), next_res

    kernel = pl.pallas_call(
        functools.partial(
            _fp8_row_matmul_async_ar_kernel,
            out_block=out_block,
            n_out=n_out,
            n_w_slots=n_w_slots,
            tp_axis=tp_axis,
            eff_reduce=eff_reduce,
            target_out_dtype=target_out_dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((batch_block, in_block), lambda: (0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((1, batch_block), lambda: (0, 0)),
            ],
            out_specs=pl.BlockSpec((batch_block, padded_n), lambda: (0, 0)),
            scratch_shapes=[
                pltpu.VMEM((n_w_slots, out_block, in_block), w_in.dtype),
                pltpu.VMEM((n_w_slots, 1, out_block), jnp.float32),
                pltpu.VMEM((n_out, batch_block, out_block), eff_reduce),
                pltpu.VMEM((n_out, batch_block, out_block), eff_reduce),
                pltpu.SemaphoreType.DMA((n_w_slots,)),
                pltpu.SemaphoreType.DMA((n_w_slots,)),
                pltpu.SemaphoreType.DMA((n_out,)),
                pltpu.SemaphoreType.DMA((n_out,)),
            ],
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_batch, padded_n), target_out_dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            collective_id=collective_id,
        ),
        interpret=interpret_arg,
        name=f"fp8_w8a8_async_ar_{batch_block}_{out_block}_{in_block}{buf_sfx}",
    )

    if not is_interpret:
        w_in = pltpu.with_memory_space_constraint(w_in, pltpu.HBM)
        ws_2d = pltpu.with_memory_space_constraint(ws_2d, pltpu.HBM)

    out_padded = kernel(x_2d, w_in, ws_2d, x_scale_row)
    return out_padded[:orig_batch, :orig_n].reshape(*leading, orig_n)


@overload
def sharded_fp8_row_matmul_async_ar(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    mesh: Mesh,
    tp_axis: str = TP_AXIS,
    x_scale: jax.Array | None = None,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    sync_tp_scale: bool = False,
    residual: None = None,
    norm_weight: None = None,
    eps: float = 1e-6,
    quantize_output_dtype: None = None,
    transposed_scale: bool = True,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    collective_id: int = 0,
    backend: MatmulBackend = "auto",
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array: ...


@overload
def sharded_fp8_row_matmul_async_ar(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    mesh: Mesh,
    tp_axis: str = TP_AXIS,
    x_scale: jax.Array | None = None,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    sync_tp_scale: bool = False,
    residual: jax.Array,
    norm_weight: jax.Array,
    eps: float = 1e-6,
    quantize_output_dtype: Any,
    transposed_scale: bool = True,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    collective_id: int = 0,
    backend: MatmulBackend = "auto",
    interpret: bool | pltpu.InterpretParams = False,
) -> tuple[jax.Array, jax.Array, jax.Array]: ...


def sharded_fp8_row_matmul_async_ar(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    mesh: Mesh,
    tp_axis: str = TP_AXIS,
    x_scale: jax.Array | None = None,
    x_abs_max: jax.Array | None = None,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    sync_tp_scale: bool = False,
    residual: jax.Array | None = None,
    norm_weight: jax.Array | None = None,
    eps: float = 1e-6,
    quantize_output_dtype: Any | None = None,
    transposed_scale: bool = True,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    collective_id: int = 0,
    backend: MatmulBackend = "auto",
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array | tuple[jax.Array, jax.Array, jax.Array]:
    """Run ``tp=2`` row-parallel (``P(None, tp_axis)``) FP8 matmul with overlapped async all-reduce."""
    quantize_activation = jnp.dtype(x.dtype) != jnp.dtype(w.dtype)
    _validate_x_scale_args(quantize_activation, x_scale, x_abs_max)
    fuse_norm_quant = residual is not None
    if fuse_norm_quant:
        if norm_weight is None or quantize_output_dtype is None:
            raise ValueError(
                "sharded_fp8_row_matmul_async_ar with residual requires norm_weight and quantize_output_dtype"
            )
    elif norm_weight is not None or quantize_output_dtype is not None:
        raise ValueError(
            "norm_weight and quantize_output_dtype require residual to be provided"
        )

    if backend not in ("auto", "pallas", "xla"):
        raise ValueError(f"Unsupported backend {backend!r}")

    tp = int(mesh.shape[tp_axis])
    is_interpret = bool(interpret)
    orig_batch = x.reshape(-1, x.shape[-1]).shape[0]
    orig_n = w.shape[0]
    local_k = w.shape[1] // max(tp, 1)

    is_supported_tpu = is_interpret or get_tpu_version() >= 7
    use_fused_epilogue = (
        fuse_norm_quant
        and orig_batch <= _ASYNC_AR_FUSED_NORM_MAX_BATCH
        and (
            tile_sizes is None
            or tile_sizes.batch_block_size <= _ASYNC_AR_FUSED_NORM_MAX_BATCH
        )
    )
    if backend == "xla":
        should_fallback = True
    elif backend == "auto":
        should_fallback = (
            not is_supported_tpu
            or tp != 2
            or (
                tile_sizes is None
                and not _can_use_async_ar(
                    tp=tp,
                    orig_batch=orig_batch,
                    orig_n=orig_n,
                    local_k=local_k,
                    w_dtype=w.dtype,
                    out_dtype=out_dtype,
                    reduce_dtype=reduce_dtype,
                    fuse_norm_quant=use_fused_epilogue,
                    q_out_dtype=quantize_output_dtype,
                )
            )
        )
    else:
        if not is_supported_tpu:
            raise ValueError(
                "sharded_fp8_row_matmul_async_ar(backend='pallas') requires TPU7x+ or interpret=True"
            )
        if tp != 2:
            raise ValueError(
                f"sharded_fp8_row_matmul_async_ar requires tp == 2 when backend='pallas', got tp={tp}"
            )
        should_fallback = False

    prefix_none = (None,) * (x.ndim - 1)
    rep_out_spec = P(*prefix_none, None)
    scale_out_spec = P(None, None) if transposed_scale else P(*prefix_none, None)

    if should_fallback:
        delta = sharded_fp8_matmul(
            x,
            w,
            w_scale,
            P(None, tp_axis),
            mesh=mesh,
            x_scale=x_scale,
            x_abs_max=x_abs_max,
            out_dtype=out_dtype,
            reduce_dtype=reduce_dtype,
            sync_tp_scale=sync_tp_scale,
            tile_sizes=tile_sizes,
            backend=backend,
            interpret=is_interpret,
        )
        if not fuse_norm_quant:
            return delta
        assert residual is not None and norm_weight is not None
        assert quantize_output_dtype is not None
        if tp == 1 or backend == "xla":
            return fused_add_rms_norm_fp8_quant(
                delta,
                residual,
                norm_weight,
                eps,
                dtype=quantize_output_dtype,
                transposed_scale=transposed_scale,
                interpret=is_interpret,
            )
        return jax.shard_map(
            lambda dl, rl, nw: fused_add_rms_norm_fp8_quant(
                dl,
                rl,
                nw,
                eps,
                dtype=quantize_output_dtype,
                transposed_scale=transposed_scale,
                interpret=is_interpret,
            ),
            mesh=mesh,
            in_specs=(rep_out_spec, rep_out_spec, P()),
            out_specs=(rep_out_spec, scale_out_spec, rep_out_spec),
            check_vma=False,
        )(delta, residual, norm_weight)

    x_spec = P(*prefix_none, tp_axis)
    w_spec = P(None, tp_axis)

    out_spec: Any = (
        (rep_out_spec, scale_out_spec, rep_out_spec)
        if fuse_norm_quant
        else rep_out_spec
    )
    extra_specs: tuple[Any, ...] = (rep_out_spec, P()) if fuse_norm_quant else ()
    if fuse_norm_quant:
        assert residual is not None and norm_weight is not None
        extra_args: tuple[jax.Array, ...] = (residual, norm_weight)
    else:
        extra_args = ()

    def _run_async_ar(
        xq_shard: jax.Array,
        xs_shard: jax.Array,
        w_shard: jax.Array,
        ws_rep: jax.Array,
        *fused_tail: jax.Array,
    ) -> Any:
        if fuse_norm_quant:
            res_rep, nw_rep = fused_tail
            assert quantize_output_dtype is not None
            if not use_fused_epilogue:
                delta_rep = pallas_fp8_row_matmul_async_ar(
                    xq_shard,
                    xs_shard,
                    w_shard,
                    ws_rep,
                    tp_axis=tp_axis,
                    out_dtype=out_dtype,
                    reduce_dtype=reduce_dtype,
                    tile_sizes=tile_sizes,
                    collective_id=collective_id,
                    interpret=interpret,
                )
                return fused_add_rms_norm_fp8_quant(
                    delta_rep,
                    res_rep,
                    nw_rep,
                    eps,
                    dtype=quantize_output_dtype,
                    transposed_scale=transposed_scale,
                    interpret=is_interpret,
                )
            return pallas_fp8_row_matmul_async_ar(
                xq_shard,
                xs_shard,
                w_shard,
                ws_rep,
                tp_axis=tp_axis,
                out_dtype=out_dtype,
                reduce_dtype=reduce_dtype,
                residual=res_rep,
                norm_weight=nw_rep,
                eps=eps,
                quantize_output_dtype=quantize_output_dtype,
                transposed_scale=transposed_scale,
                tile_sizes=tile_sizes,
                collective_id=collective_id,
                interpret=interpret,
            )
        return pallas_fp8_row_matmul_async_ar(
            xq_shard,
            xs_shard,
            w_shard,
            ws_rep,
            tp_axis=tp_axis,
            out_dtype=out_dtype,
            reduce_dtype=reduce_dtype,
            tile_sizes=tile_sizes,
            collective_id=collective_id,
            interpret=interpret,
        )

    if not quantize_activation:
        if x_scale is None:
            eff_scale = jnp.ones((1, orig_batch), dtype=jnp.float32)
            xs_spec = P(None, None)
        else:
            eff_scale = x_scale
            xs_spec = _scale_shard_spec(
                x_scale,
                mesh=mesh,
                in_axis=tp_axis,
                sync_tp_scale=sync_tp_scale,
                n_rows=orig_batch,
            )
        return jax.shard_map(
            _run_async_ar,
            mesh=mesh,
            in_specs=(x_spec, xs_spec, w_spec, P(), *extra_specs),
            out_specs=out_spec,
            check_vma=False,
        )(x, eff_scale, w, w_scale, *extra_args)

    if x_abs_max is not None:
        xa_spec = _scale_shard_spec(
            x_abs_max,
            mesh=mesh,
            in_axis=tp_axis,
            sync_tp_scale=sync_tp_scale,
            n_rows=orig_batch,
        )

        def _local_from_absmax(
            x_shard: jax.Array,
            xa_shard: jax.Array,
            w_shard: jax.Array,
            ws_rep: jax.Array,
            *fused_tail: jax.Array,
        ) -> Any:
            x_2d = x_shard.reshape(orig_batch, -1)
            xq_2d, xs_row = quantize_fp8_with_absmax(
                x_2d,
                xa_shard,
                dtype=w_shard.dtype,
                transposed_scale=True,
                interpret=is_interpret,
            )
            return _run_async_ar(
                xq_2d.reshape(x_shard.shape), xs_row, w_shard, ws_rep, *fused_tail
            )

        return jax.shard_map(
            _local_from_absmax,
            mesh=mesh,
            in_specs=(x_spec, xa_spec, w_spec, P(), *extra_specs),
            out_specs=out_spec,
            check_vma=False,
        )(x, x_abs_max, w, w_scale, *extra_args)

    def _local_quantize(
        x_shard: jax.Array,
        w_shard: jax.Array,
        ws_rep: jax.Array,
        *fused_tail: jax.Array,
    ) -> Any:
        xq_shard, xs_col = quantize_fp8_act(
            x_shard,
            dtype=w_shard.dtype,
            tp_axis=tp_axis,
            sync_tp_scale=sync_tp_scale,
        )
        return _run_async_ar(
            xq_shard,
            jnp.transpose(xs_col.reshape(orig_batch, 1)),
            w_shard,
            ws_rep,
            *fused_tail,
        )

    return jax.shard_map(
        _local_quantize,
        mesh=mesh,
        in_specs=(x_spec, w_spec, P(), *extra_specs),
        out_specs=out_spec,
        check_vma=False,
    )(x, w, w_scale, *extra_args)


__all__ = [
    "estimate_async_ar_vmem_bytes",
    "get_async_ar_tile_sizes",
    "pallas_fp8_row_matmul_async_ar",
    "sharded_fp8_row_matmul_async_ar",
]
