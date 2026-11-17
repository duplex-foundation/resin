"""On-device token sampling with request values as runtime arrays.

Two regimes are compiled per row bucket: greedy, and stochastic with exact
temperature, top-k and top-p filtering. Temperature, top-k, top-p, seeds and
offsets never become compilation keys. Randomness is stateless per row: the
seed and absolute offset from
:func:`resin.backend.sampling.batch_sampling_inputs` select a Threefry key, so
draws differ from CUDA's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

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


def _draw(
    scaled: jax.Array, lse: jax.Array, inputs: SamplingInputs, keys, k: int
) -> tuple[jax.Array, jax.Array]:
    """Draw from the ``k`` highest tempered logits; also return their mass."""

    ranked, order = lax.top_k(scaled, k)
    # The nucleus comes from the full tempered distribution before top-k;
    # the highest-ranked token always survives.
    probabilities = jnp.exp(ranked - lse[:, None])
    exclusive = jnp.cumsum(probabilities, axis=-1) - probabilities
    rank = jnp.arange(k)[None, :]
    keep = (rank < inputs.top_k[:, None]) & (exclusive < inputs.top_p[:, None])
    keep |= rank == 0
    noise = jax.vmap(lambda key: jax.random.gumbel(key, (k,)))(keys)
    choice = jnp.argmax(jnp.where(keep, ranked + noise, -jnp.inf), axis=-1)
    sampled = jnp.take_along_axis(order, choice[:, None], axis=-1)[:, 0]
    return sampled.astype(jnp.int32), probabilities.sum(-1)


def sample(
    logits: jax.Array,
    inputs: SamplingInputs,
    *,
    stochastic: bool,
    candidates: int = 128,
) -> jax.Array:
    """Select one token per row from ``logits[seqs, vocab]``.

    Stochastic rows draw from their ``candidates`` highest logits. A row whose
    top-k and nucleus can reach past them draws from the full sorted
    vocabulary instead; each row's draw depends only on its own logits and
    policy, never on the rest of the batch.
    """

    greedy = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    if not stochastic:
        return greedy
    vocab = logits.shape[-1]
    temperature = inputs.temperature[:, None]
    scaled = logits.astype(jnp.float32) / jnp.where(temperature > 0, temperature, 1.0)
    lse = jax.nn.logsumexp(scaled, axis=-1)
    keys = jax.vmap(
        lambda data, offset: jax.random.fold_in(
            jax.random.wrap_key_data(data, impl="threefry2x32"), offset
        )
    )(inputs.seeds, inputs.offsets)
    sampled, mass = _draw(scaled, lse, inputs, keys, min(candidates, vocab))
    covered = (
        (inputs.top_k <= candidates) | (mass >= inputs.top_p) | (temperature[:, 0] <= 0)
    )
    sampled = lax.cond(
        jnp.all(covered),
        lambda: sampled,
        lambda: jnp.where(covered, sampled, _draw(scaled, lse, inputs, keys, vocab)[0]),
    )
    return jnp.where(inputs.temperature > 0, sampled, greedy)


__all__ = [
    "SamplingInputs",
    "empty_sampling_inputs",
    "is_stochastic",
    "sample",
    "sampling_inputs",
]
