"""Row-parallel TPU7x FP8 W8A8 matmul overlapped with cross-die async all-reduce."""

from __future__ import annotations

import functools
from typing import Any

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
    _scale_shard_spec,
    _validate_matmul_shapes,
    _validate_x_scale_args,
    get_tuned_block_sizes,
    quantize_fp8_act,
    quantize_fp8_with_absmax,
    sharded_fp8_matmul,
)
from resin.backend.tpu.mesh import TP_AXIS

_ASYNC_AR_MAX_VMEM_BYTES = 40 * 1024 * 1024


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
) -> int:
    """Return total VMEM bytes (inputs + output + scratch) for the async all-reduce kernel."""
    padded_n = n_out * out_block
    x_bytes = batch_block * in_block * x_itemsize
    xs_bytes = batch_block * 4
    out_bytes = batch_block * padded_n * out_itemsize
    w_buf_bytes = n_w_slots * out_block * in_block * w_itemsize
    ws_buf_bytes = n_w_slots * out_block * 4
    local_peer_bytes = 2 * n_out * batch_block * out_block * reduce_itemsize
    return (
        x_bytes + xs_bytes + out_bytes + w_buf_bytes + ws_buf_bytes + local_peer_bytes
    )


def get_async_ar_tile_sizes(
    orig_batch: int,
    orig_n: int,
    orig_k: int,
) -> Fp8MatmulTileSizes:
    """Select ``(batch_block, out_block, in_block, w_buffer_count)`` for ``tp=2`` async all-reduce."""
    tuned = get_tuned_block_sizes(
        orig_batch,
        orig_n,
        orig_k,
        tpu_version=max(get_tpu_version(), 7),
        quantize_activation=False,
    )
    batch_block = align_to(max(8, orig_batch), 8)
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
    batch_block, out_block, in_block, w_buf = tiles
    if batch_block % 8 != 0 or out_block % 128 != 0 or in_block % 128 != 0:
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
) -> bool:
    if tp != 2 or local_k < 128:
        return False
    tiles = get_async_ar_tile_sizes(orig_batch, orig_n, local_k)
    try:
        _, _, _, n_out = _validate_async_ar_tiles(orig_batch, orig_n, local_k, tiles)
    except ValueError:
        return False
    target_out = jnp.dtype(out_dtype)
    eff_reduce = jnp.dtype(target_out if reduce_dtype is None else reduce_dtype)
    w_itemsize = jnp.dtype(w_dtype).itemsize
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
    )
    return vmem_bytes <= _ASYNC_AR_MAX_VMEM_BYTES


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
        out_ref[:, pl.ds(tile_idx * out_block, out_block)] = (
            reduced.astype(target_out_dtype)
            if target_out_dtype != eff_reduce
            else reduced
        )
        rc.wait_send()

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


def pallas_fp8_row_matmul_async_ar(
    x_q: jax.Array,
    x_scale: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    *,
    tp_axis: str = TP_AXIS,
    out_dtype: Any = jnp.bfloat16,
    reduce_dtype: Any | None = None,
    tile_sizes: Fp8MatmulTileSizes | None = None,
    collective_id: int = 0,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Compute local FP8 W8A8 ``x_q @ w.T`` and cross-die ``psum`` via ``make_async_remote_copy``."""
    _validate_matmul_shapes(x_q.shape[-1], w, w_scale)
    if x_q.dtype != w.dtype:
        raise ValueError(
            f"pallas_fp8_row_matmul_async_ar expects pre-quantized {w.dtype} x_q, got {x_q.dtype}"
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
    x_2d, x_scale_row, w_in, ws_2d = _pad_async_ar_inputs(
        x_2d,
        x_scale_row,
        w,
        w_scale,
        padded_batch=padded_batch,
        padded_n=padded_n,
        padded_k=padded_k,
    )

    target_out_dtype = jnp.dtype(out_dtype)
    eff_reduce = jnp.dtype(target_out_dtype if reduce_dtype is None else reduce_dtype)
    n_w_slots = min(w_buf, n_out)

    is_interpret = bool(interpret)
    interpret_arg: bool | pltpu.InterpretParams = (
        interpret
        if isinstance(interpret, pltpu.InterpretParams)
        else (pltpu.InterpretParams() if interpret else False)
    )
    tpu_version = get_tpu_version()
    vmem_limit_bytes = (
        _DEVICE_VMEM_LIMIT.get(tpu_version, 48 * 1024 * 1024)
        if (tpu_version >= 6 and not is_interpret)
        else None
    )

    buf_sfx = f"_b{w_buf}" if w_buf != 2 else ""
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

    if tpu_version >= 6 and not is_interpret:
        w_in = pltpu.with_memory_space_constraint(w_in, pltpu.HBM)
        ws_2d = pltpu.with_memory_space_constraint(ws_2d, pltpu.HBM)

    out_padded = kernel(x_2d, w_in, ws_2d, x_scale_row)
    return out_padded[:orig_batch, :orig_n].reshape(*leading, orig_n)


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
    tile_sizes: Fp8MatmulTileSizes | None = None,
    collective_id: int = 0,
    backend: MatmulBackend = "auto",
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Run ``tp=2`` row-parallel (``P(None, tp_axis)``) FP8 matmul with overlapped async all-reduce."""
    quantize_activation = jnp.dtype(x.dtype) != jnp.dtype(w.dtype)
    _validate_x_scale_args(quantize_activation, x_scale, x_abs_max)

    tp = int(mesh.shape[tp_axis])
    is_interpret = bool(interpret)
    orig_batch = x.reshape(-1, x.shape[-1]).shape[0]
    orig_n = w.shape[0]
    local_k = w.shape[1] // max(tp, 1)

    is_supported_tpu = is_interpret or get_tpu_version() >= 7
    should_fallback = backend == "xla" or (
        tile_sizes is None
        and (
            (backend == "auto" and not is_supported_tpu)
            or not _can_use_async_ar(
                tp=tp,
                orig_batch=orig_batch,
                orig_n=orig_n,
                local_k=local_k,
                w_dtype=w.dtype,
                out_dtype=out_dtype,
                reduce_dtype=reduce_dtype,
            )
        )
    )
    if should_fallback:
        return sharded_fp8_matmul(
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

    if tp != 2:
        raise ValueError(
            f"sharded_fp8_row_matmul_async_ar requires tp == 2 when backend='pallas', got tp={tp}"
        )

    prefix_none = (None,) * (x.ndim - 1)
    x_spec = P(*prefix_none, tp_axis)
    w_spec = P(None, tp_axis)
    out_spec = P(*prefix_none, None)

    def _run_async_ar(
        xq_shard: jax.Array,
        xs_shard: jax.Array,
        w_shard: jax.Array,
        ws_rep: jax.Array,
    ) -> jax.Array:
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
            in_specs=(x_spec, xs_spec, w_spec, P()),
            out_specs=out_spec,
            check_vma=False,
        )(x, eff_scale, w, w_scale)

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
        ) -> jax.Array:
            x_2d = x_shard.reshape(orig_batch, -1)
            xq_2d, xs_row = quantize_fp8_with_absmax(
                x_2d,
                xa_shard,
                dtype=w_shard.dtype,
                transposed_scale=True,
                interpret=is_interpret,
            )
            return _run_async_ar(xq_2d.reshape(x_shard.shape), xs_row, w_shard, ws_rep)

        return jax.shard_map(
            _local_from_absmax,
            mesh=mesh,
            in_specs=(x_spec, xa_spec, w_spec, P()),
            out_specs=out_spec,
            check_vma=False,
        )(x, x_abs_max, w, w_scale)

    def _local_quantize(
        x_shard: jax.Array,
        w_shard: jax.Array,
        ws_rep: jax.Array,
    ) -> jax.Array:
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
        )

    return jax.shard_map(
        _local_quantize,
        mesh=mesh,
        in_specs=(x_spec, w_spec, P()),
        out_specs=out_spec,
        check_vma=False,
    )(x, w, w_scale)


__all__ = [
    "estimate_async_ar_vmem_bytes",
    "get_async_ar_tile_sizes",
    "pallas_fp8_row_matmul_async_ar",
    "sharded_fp8_row_matmul_async_ar",
]
