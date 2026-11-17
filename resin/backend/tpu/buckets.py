"""Fixed padded step shapes, so XLA compiles each one once at startup."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from resin import BatchPhase


@dataclass(frozen=True, slots=True)
class Bucket:
    """One compiled step shape: padded token count by padded sequence count."""

    tokens: int
    seqs: int


def _powers_of_two(low: int, high: int) -> tuple[int, ...]:
    values = []
    value = low
    while value < high:
        values.append(value)
        value *= 2
    values.append(high)
    return tuple(values)


def _sorted_unique(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(sorted(set(values)))
    if not result or result[0] <= 0:
        raise ValueError(f"{name} buckets must be positive and nonempty")
    return result


def _smallest(values: tuple[int, ...], needed: int, name: str) -> int:
    for value in values:
        if value >= needed:
            return value
    raise ValueError(f"{needed} {name} exceed the largest bucket {values[-1]}")


@dataclass(frozen=True, slots=True)
class ShapeBuckets:
    """Decode steps pad rows; prefill steps pad tokens.

    Decode has one token per row, so a decode bucket's token count equals its
    row count. A prefill bucket holds ``min(tokens, decode_seqs[-1])`` rows:
    ragged attention loops over live rows only, so padded rows cost little,
    while every extra row count would compile the whole model again. Bucket
    boundaries are engine-internal tuning, not workload settings.
    """

    decode_seqs: tuple[int, ...]
    prefill_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decode_seqs", _sorted_unique(self.decode_seqs, "decode")
        )
        object.__setattr__(
            self, "prefill_tokens", _sorted_unique(self.prefill_tokens, "prefill token")
        )

    @classmethod
    def powers_of_two(
        cls,
        *,
        max_num_seqs: int,
        max_batch_tokens: int,
        min_decode_seqs: int = 8,
        min_prefill_tokens: int = 128,
    ) -> ShapeBuckets:
        return cls(
            decode_seqs=_powers_of_two(
                min(min_decode_seqs, max_num_seqs), max_num_seqs
            ),
            prefill_tokens=_powers_of_two(
                min(min_prefill_tokens, max_batch_tokens), max_batch_tokens
            ),
        )

    def _prefill(self, tokens: int) -> Bucket:
        return Bucket(tokens, min(tokens, self.decode_seqs[-1]))

    def select(self, phase: BatchPhase, tokens: int, seqs: int) -> Bucket:
        if phase == BatchPhase.DECODE:
            if tokens != seqs:
                raise ValueError("decode steps schedule exactly one token per row")
            rows = _smallest(self.decode_seqs, seqs, "decode rows")
            return Bucket(rows, rows)
        bucket = self._prefill(_smallest(self.prefill_tokens, tokens, "prefill tokens"))
        if seqs > bucket.seqs:
            raise ValueError(f"{seqs} prefill rows exceed {bucket}")
        return bucket

    def shapes(self) -> Iterator[Bucket]:
        """Every bucket a step can select, for startup compilation."""

        for rows in self.decode_seqs:
            yield Bucket(rows, rows)
        for tokens in self.prefill_tokens:
            yield self._prefill(tokens)


__all__ = ["Bucket", "ShapeBuckets"]
