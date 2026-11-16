"""On-device token sampling with request values as runtime arrays.

Two regimes are compiled per row bucket: greedy, and stochastic with exact
temperature, top-k and top-p filtering. Temperature, top-k, top-p, seeds and
offsets never become compilation keys. Randomness is stateless per row: the
request seed and the absolute sampling offset select a Threefry key, as in
:func:`resin.backend.sampling.batch_sampling_inputs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from resin.backend.sampling import batch_sampling_inputs
from resin.backend.tpu.metadata import StepArray

if TYPE_CHECKING:
    from resin import BatchPlan


class SamplingInputs(NamedTuple):
    temperature: StepArray  # f32[seqs]; zero selects the maximum logit
    top_k: StepArray  # i32[seqs]; vocabulary size when unset
    top_p: StepArray  # f32[seqs]
    seeds: StepArray  # u32[seqs, 2]; Threefry key data
    offsets: StepArray  # u32[seqs]; absolute sampling position offset


def sampling_inputs(plan: BatchPlan, seqs: int, vocab_size: int) -> SamplingInputs:
    """Pad one plan's sampling policies to ``seqs`` rows as host arrays."""

    values = batch_sampling_inputs(plan, vocab_size)
    rows = plan.num_sequences
    inputs = empty_sampling_inputs(seqs, vocab_size)
    temperature, top_k, top_p, seeds, offsets = inputs
    temperature[:rows] = values["temperatures"]
    temperature[:rows] *= values["sampling_mask"]
    top_k[:rows] = values["top_ks"]
    top_p[:rows] = values["top_ps"]
    raw = np.array(values["seeds"], dtype=np.uint64)
    seeds[:rows, 0] = (raw >> np.uint64(32)).astype(np.uint32)
    seeds[:rows, 1] = (raw & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    offsets[:rows] = np.array(values["offsets"], dtype=np.uint64).astype(np.uint32)
    return inputs


def empty_sampling_inputs(seqs: int, vocab_size: int) -> SamplingInputs:
    """Greedy, unfiltered policies for ``seqs`` padding rows."""

    return SamplingInputs(
        np.zeros(seqs, np.float32),
        np.full(seqs, vocab_size, np.int32),
        np.ones(seqs, np.float32),
        np.zeros((seqs, 2), np.uint32),
        np.zeros(seqs, np.uint32),
    )


def is_stochastic(inputs: SamplingInputs) -> bool:
    return bool(np.any(np.asarray(inputs.temperature) > 0.0))


def sample(logits: jax.Array, inputs: SamplingInputs, *, stochastic: bool) -> jax.Array:
    """Select one token per row from ``logits[seqs, vocab]``."""

    greedy = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    if not stochastic:
        return greedy
    temperature = inputs.temperature[:, None]
    scaled = logits.astype(jnp.float32) / jnp.where(temperature > 0, temperature, 1.0)
    order = jnp.argsort(scaled, axis=-1, descending=True)
    ranked = jnp.take_along_axis(scaled, order, axis=-1)
    probabilities = jax.nn.softmax(ranked, axis=-1)
    exclusive = jnp.cumsum(probabilities, axis=-1) - probabilities
    rank = jnp.arange(logits.shape[-1])[None, :]
    # The nucleus comes from the full tempered distribution before top-k;
    # the highest-ranked token always survives.
    keep = (rank < inputs.top_k[:, None]) & (exclusive < inputs.top_p[:, None])
    keep |= rank == 0
    keys = jax.vmap(
        lambda data, offset: jax.random.fold_in(jax.random.wrap_key_data(data), offset)
    )(inputs.seeds, inputs.offsets)
    noise = jax.vmap(lambda key: jax.random.gumbel(key, ranked.shape[-1:]))(keys)
    choice = jnp.argmax(jnp.where(keep, ranked + noise, -jnp.inf), axis=-1)
    sampled = jnp.take_along_axis(order, choice[:, None], axis=-1)[:, 0]
    return jnp.where(inputs.temperature > 0, sampled.astype(jnp.int32), greedy)


__all__ = [
    "SamplingInputs",
    "empty_sampling_inputs",
    "is_stochastic",
    "sample",
    "sampling_inputs",
]
