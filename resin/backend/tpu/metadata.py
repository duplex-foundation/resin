"""Padded per-step model inputs built from a validated paged cache plan.

Every array's shape depends only on the step's bucket and the runner's fixed
page-table width. Lengths, positions, page IDs and the phase split are runtime
values, never compilation keys.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import numpy as np

from resin.backend.plans import CacheGroupView, PagedCachePlan
from resin.backend.tpu.buckets import Bucket

# Step inputs are host arrays until the runner transfers them.
StepArray = jax.Array | np.ndarray


class AttentionMetadata(NamedTuple):
    """Ragged paged attention v3 metadata, shared by every layer of a step.

    ``page_indices`` holds one flattened ``[seqs * pages_per_seq]`` table per
    cache group. Row ``i``'s page for logical block ``b`` is entry
    ``i * pages_per_seq + b``; unused entries address null page zero.
    """

    kv_lens: StepArray
    page_indices: tuple[StepArray, ...]
    cu_q_lens: StepArray
    distribution: StepArray


class ModelInputs(NamedTuple):
    """Token inputs, attention metadata, and the rows that produce logits."""

    input_ids: StepArray
    positions: StepArray
    attention: AttentionMetadata
    logits_indices: StepArray


def page_table(
    group: CacheGroupView, rows: int, seqs: int, pages_per_seq: int
) -> np.ndarray:
    """Place each row's leased blocks at their absolute logical columns."""

    counts = np.diff(group.block_indptr).astype(np.int64)
    row_ids = np.repeat(np.arange(rows), counts)
    starts = group.block_indptr[:-1].astype(np.int64)
    columns = group.block_table_start_indices[row_ids].astype(np.int64) + (
        np.arange(len(row_ids)) - starts[row_ids]
    )
    if len(columns) and int(np.max(columns)) >= pages_per_seq:
        raise ValueError(
            f"cache group {group.group_index} leases a block beyond the runner's "
            f"{pages_per_seq}-page table"
        )
    table = np.zeros((seqs, pages_per_seq), dtype=np.int32)
    table[row_ids, columns] = group.packed_block_ids.astype(np.int32)
    return table.reshape(-1)


def _padded(values: np.ndarray, size: int, *, fill: int = 0) -> np.ndarray:
    padded = np.full(size, fill, dtype=np.int32)
    padded[: len(values)] = values
    return padded


def build_inputs(
    plan: PagedCachePlan, bucket: Bucket, pages_per_seq: tuple[int, ...]
) -> ModelInputs:
    """Pad one plan to ``bucket`` as host arrays; the caller transfers them."""

    batch = plan.batch
    rows, tokens = batch.num_sequences, batch.num_tokens
    if rows > bucket.seqs or tokens > bucket.tokens:
        raise ValueError(f"{rows} rows and {tokens} tokens do not fit {bucket}")
    if len(pages_per_seq) != len(plan.groups):
        raise ValueError("one page-table width is required per cache group")

    input_ids = _padded(batch.token_ids, bucket.tokens)
    positions = _padded(batch.positions, bucket.tokens)
    kv_lens = _padded(batch.post_lengths, bucket.seqs)
    cu_q_lens = _padded(batch.token_indptr, bucket.seqs + 1, fill=tokens)
    logits_indices = _padded(batch.last_token_indices, bucket.seqs)
    # RPA v3 runs its [i, j) prefill range only with a static chunk size, so
    # variable-length prefill rows go to the mixed range [j, k). A mixed
    # prefill plan leads with its decode rows, which keep the decode range.
    decode_rows = batch.decode_rows
    distribution = np.array([decode_rows, decode_rows, rows], dtype=np.int32)
    tables = tuple(
        page_table(group, rows, bucket.seqs, width)
        for group, width in zip(plan.groups, pages_per_seq, strict=True)
    )
    return ModelInputs(
        input_ids,
        positions,
        AttentionMetadata(kv_lens, tables, cu_q_lens, distribution),
        logits_indices,
    )


def empty_inputs(bucket: Bucket, pages_per_seq: tuple[int, ...]) -> ModelInputs:
    """A step with no rows: compiles a bucket without touching any page."""

    return ModelInputs(
        np.zeros(bucket.tokens, dtype=np.int32),
        np.zeros(bucket.tokens, dtype=np.int32),
        AttentionMetadata(
            np.zeros(bucket.seqs, dtype=np.int32),
            tuple(
                np.zeros(bucket.seqs * width, dtype=np.int32) for width in pages_per_seq
            ),
            np.zeros(bucket.seqs + 1, dtype=np.int32),
            np.zeros(3, dtype=np.int32),
        ),
        np.zeros(bucket.seqs, dtype=np.int32),
    )


__all__ = [
    "AttentionMetadata",
    "ModelInputs",
    "StepArray",
    "build_inputs",
    "empty_inputs",
    "page_table",
]
