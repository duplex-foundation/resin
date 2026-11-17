"""Model-agnostic TPU runner: Engine plans onto bucketed, donated device steps.

The runner owns no request state. Each step validates the plan, pads it to a
precompiled shape bucket, transfers the metadata, dispatches the model step
with the cache arena donated, and samples on device. JAX dispatch is
asynchronous, so ``enqueue`` returns before the device finishes, and a decode
continuation feeds the previous step's device tokens straight into the next.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh

from resin import BatchPhase, RunnerConfig, RunnerMetrics
from resin.backend.plans import PagedCachePlan
from resin.backend.sampling import format_output, prompt_scores, token_output
from resin.backend.topology import ParallelismCapabilities
from resin.backend.tpu.arena import (
    CacheArena,
    Caches,
    KvCacheSpec,
    cache_layout,
    plan_capacity,
)
from resin.backend.tpu.buckets import ShapeBuckets
from resin.backend.tpu.compilation import CompilationCounter
from resin.backend.tpu.mesh import replicated, tp_size
from resin.backend.tpu.metadata import ModelInputs, build_inputs, empty_inputs
from resin.backend.tpu.sampling import (
    empty_sampling_inputs,
    is_stochastic,
    sample,
    sampling_inputs,
)

if TYPE_CHECKING:
    from resin import BatchPlan, ForwardOutput, ResourceBinding

# Prompt scoring projects hidden rows in fixed chunks of this many rows.
_SCORE_ROWS = 128


class TpuModel(Protocol):
    """What a model plug-in provides: cache geometry, layer graph and LM head.

    ``hidden_states`` returns ``[tokens, hidden]`` activations for the padded
    step and the updated caches, calling
    :func:`resin.backend.tpu.attention.paged_attention` once per layer with
    ``caches[group][layer]``. ``logits`` projects ``[rows, hidden]`` to
    ``[rows, vocab]``. Both run inside one compiled step; parameters arrive as
    an argument so they are never baked into the executable.
    """

    vocab_size: int

    @property
    def cache_specs(self) -> Sequence[KvCacheSpec]: ...

    def hidden_states(
        self, params: Any, caches: Caches, inputs: ModelInputs
    ) -> tuple[jax.Array, Caches]: ...

    def logits(self, params: Any, hidden: jax.Array) -> jax.Array: ...


@dataclass(slots=True)
class _Step:
    """One dispatched step's device outputs, read only when harvested."""

    plan: BatchPlan
    tokens: jax.Array
    logits: jax.Array
    hidden: jax.Array


@dataclass(slots=True)
class _Completion:
    """An overlapped decode step; ``result`` waits only for this step."""

    _runner: TpuRunner
    _step: _Step
    _continues: bool
    _result: ForwardOutput | None = field(default=None, init=False)

    @property
    def feedback(self) -> jax.Array | None:
        # None drains the Engine, so a step past max_seq_len fails in forward
        # rather than in an overlapped enqueue.
        return self._step.tokens if self._continues else None

    def result(self) -> ForwardOutput:
        if self._result is None:
            self._result = self._runner._output(self._step)
        return self._result


class PostAcceptTpuError(RuntimeError):
    """Dispatched device work may have changed the cache; do not retry."""

    _resin_post_accept = True


class TpuRunner:
    """Run a :class:`TpuModel` on a ``tp`` mesh with Engine-owned KV pages.

    ``cache_slab_capacity`` caps the slab pool; it is required on platforms
    that report no device memory, such as the JAX CPU backend. Construction
    compiles every step shape, and any compilation after that fails the step,
    which is how the backend enforces its shape-stability rule. Configure
    JAX's persistent compile cache through its ``JAX_COMPILATION_CACHE_DIR``
    environment variables. The Engine's ``max_running_requests`` must not
    exceed ``max_num_seqs``, and its ``context_length`` must not exceed
    ``max_seq_len``.
    """

    parallelism_capabilities = ParallelismCapabilities()

    def __init__(
        self,
        model: TpuModel,
        params: Any,
        *,
        mesh: Mesh,
        max_seq_len: int,
        max_num_seqs: int,
        max_batch_tokens: int,
        prefill_chunk_size: int | None = None,
        buckets: ShapeBuckets | None = None,
        memory_utilization: float = 0.9,
        workspace_bytes: int = 8 * 1024**3,
        cache_slab_capacity: int | None = None,
    ) -> None:
        if min(max_seq_len, max_num_seqs, max_batch_tokens) <= 0:
            raise ValueError("runner limits must be positive")
        self.compilations = CompilationCounter()
        self.model = model
        self.params = params
        self.mesh = mesh
        self.vocab_size = model.vocab_size
        self.max_seq_len = max_seq_len
        self.buckets = buckets or ShapeBuckets.powers_of_two(
            max_num_seqs=max_num_seqs, max_batch_tokens=max_batch_tokens
        )
        if self.buckets.decode_seqs[-1] < max_num_seqs:
            raise ValueError("decode buckets must hold max_num_seqs rows")
        if self.buckets.prefill_tokens[-1] < max_batch_tokens:
            raise ValueError("prefill buckets must hold max_batch_tokens tokens")

        specs = tuple(model.cache_specs)
        capacity = cache_slab_capacity
        self.cache_capacity_plan = None
        if any(device.memory_stats() for device in mesh.devices.flat):
            self.cache_capacity_plan = plan_capacity(
                specs,
                mesh,
                memory_utilization=memory_utilization,
                workspace_bytes=workspace_bytes,
            )
            planned = self.cache_capacity_plan.parent_capacity
            capacity = planned if capacity is None else min(capacity, planned)
        if capacity is None:
            raise ValueError(
                "pass cache_slab_capacity on a platform without memory stats"
            )
        chunk_tokens = prefill_chunk_size or max_batch_tokens
        self.cache_layout = cache_layout(specs, capacity, chunk_tokens=chunk_tokens)
        self._arena = CacheArena(specs, self.cache_layout, mesh)
        self.pages_per_seq = tuple(spec.pages_per_seq(max_seq_len) for spec in specs)
        self.runner_config = RunnerConfig(
            max_batch_tokens=max_batch_tokens,
            prefill_chunk_size=chunk_tokens,
            continuation_prefill=True,
            cache_layout=self.cache_layout,
        )
        self.benchmark_metadata: dict[str, Any] = {
            "kv_cache_dtype": str(jnp.dtype(specs[0].dtype)),
            "page_tokens": [spec.page_tokens for spec in specs],
            "tp": tp_size(mesh),
            "buckets": {
                "decode_seqs": list(self.buckets.decode_seqs),
                "prefill_tokens": list(self.buckets.prefill_tokens),
            },
        }

        self._replicated = replicated(mesh)
        # Pinned cache output shardings keep the donated caches' input
        # signature stable from step to step, so no step recompiles for
        # layout. Logits and hidden keep the model's sharding: replicating a
        # vocab-sharded head's logits would all-gather [seqs, vocab] per step.
        cache_shardings = jax.tree.map(lambda array: array.sharding, self._arena.caches)
        forward_kwargs: dict[str, Any] = {
            "donate_argnums": (1,),
            "out_shardings": (None, None, cache_shardings),
        }
        if jax.default_backend() == "tpu":
            forward_kwargs["compiler_options"] = {
                "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative",
                "xla_tpu_reduce_scatter_collective_matmul_mode": "post_spmd_conservative",
                "xla_tpu_use_minor_sharding_for_major_trivial_input": "true",
            }
        self._forward = jax.jit(self._forward_step, **forward_kwargs)
        self._samplers = {
            stochastic: jax.jit(
                functools.partial(sample, stochastic=stochastic),
                out_shardings=self._replicated,
            )
            for stochastic in (False, True)
        }
        self._project = jax.jit(
            lambda params, hidden: model.logits(params, hidden).astype(jnp.float32),
            out_shardings=self._replicated,
        )
        self._step_compilations = 0
        self.warmup()

    def _forward_step(
        self, params: Any, caches: Caches, inputs: ModelInputs
    ) -> tuple[jax.Array, jax.Array, Caches]:
        hidden, caches = self.model.hidden_states(params, caches, inputs)
        logits = self.model.logits(params, hidden[inputs.logits_indices])
        return logits, hidden, caches

    def _put(self, tree):
        return jax.device_put(tree, self._replicated)

    def warmup(self) -> None:
        """Compile every bucket, both sampling regimes and prompt scoring."""

        start = time.perf_counter()
        hidden = None
        for bucket in self.buckets.shapes():
            inputs = self._put(empty_inputs(bucket, self.pages_per_seq))
            logits, hidden, self._arena.caches = self._forward(
                self.params, self._arena.caches, inputs
            )
            policy = self._put(empty_sampling_inputs(bucket.seqs, self.vocab_size))
            for sampler in self._samplers.values():
                sampler(logits, policy)
        assert hidden is not None
        rows = self._put(np.zeros((_SCORE_ROWS, hidden.shape[-1]), np.float32))
        jax.block_until_ready((self._project(self.params, rows), self._arena.caches))
        self.warmup_compilations = self.compilations.count
        self.benchmark_metadata["warmup_seconds"] = time.perf_counter() - start

    @property
    def compilations_after_warmup(self) -> int:
        """XLA compilations or cache loads inside steps since warmup."""

        return self._step_compilations

    def metrics_snapshot(self) -> RunnerMetrics:
        """Publish compilations after warmup. The read does not sync the device."""

        return RunnerMetrics(compilations=self.compilations_after_warmup)

    def bind_resources(self, resources: ResourceBinding) -> None:
        self._arena.bind_resources(resources)

    def _cache_plan(self, plan: BatchPlan) -> PagedCachePlan:
        if any(media is not None for media in plan.media):
            raise ValueError("the TPU runner does not accept media inputs")
        cache_plan = PagedCachePlan.from_plan(plan, self.cache_layout)
        if np.any(cache_plan.token_ids >= self.vocab_size):
            raise ValueError("token id is out of range for this model")
        if int(np.max(cache_plan.batch.post_lengths)) > self.max_seq_len:
            raise ValueError(
                f"sequence exceeds this runner's {self.max_seq_len}-token context"
            )
        return cache_plan

    def _launch(self, plan: BatchPlan, feedback: jax.Array | None = None) -> _Step:
        # Nothing before dispatch may touch device state: these failures must
        # stay retryable.
        cache_plan = self._cache_plan(plan)
        batch = cache_plan.batch
        bucket = self.buckets.select(plan.phase, batch.num_tokens, batch.num_sequences)
        host = build_inputs(cache_plan, bucket, self.pages_per_seq)
        policy = sampling_inputs(plan, bucket.seqs, self.vocab_size)
        stochastic = is_stochastic(policy)
        if feedback is not None:
            if feedback.shape != (bucket.tokens,):
                raise ValueError(
                    "decode feedback does not match the continuation bucket"
                )
            host = host._replace(input_ids=feedback)
        inputs, policy = self._put((host, policy))
        before = self.compilations.count
        try:
            logits, hidden, self._arena.caches = self._forward(
                self.params, self._arena.caches, inputs
            )
            tokens = self._samplers[stochastic](logits, policy)
        except Exception as error:
            raise PostAcceptTpuError(
                f"TPU step for {bucket} failed after dispatch"
            ) from error
        self._check_compilations(before, f"step for {bucket}")
        return _Step(plan, tokens, logits, hidden)

    def _check_compilations(self, before: int, label: str) -> None:
        compiled = self.compilations.count - before
        self._step_compilations += compiled
        if compiled:
            raise PostAcceptTpuError(
                f"TPU {label} compiled after warmup: "
                f"{self.compilations.by_function(before)}"
            )

    def _output(self, step: _Step) -> ForwardOutput:
        # Asynchronous device failures surface when outputs are read, after
        # the step consumed its donated caches.
        try:
            return self._harvest(step)
        except PostAcceptTpuError:
            raise
        except Exception as error:
            raise PostAcceptTpuError("TPU step failed while reading outputs") from error

    def _harvest(self, step: _Step) -> ForwardOutput:
        plan = step.plan
        rows = plan.num_sequences
        # Slice on the host: an eager device slice compiles for every new size.
        tokens = np.asarray(step.tokens)[:rows]
        if not plan.needs_logprobs and not plan.needs_prompt_logprobs:
            return token_output(tokens)
        logits = None
        if plan.needs_logprobs:
            logits = torch.from_numpy(np.asarray(step.logits)[:rows].astype(np.float32))
        prompt = None
        if plan.needs_prompt_logprobs:
            hidden = np.asarray(step.hidden)[: plan.num_tokens].astype(np.float32)
            prompt = prompt_scores(plan, torch.from_numpy(hidden), self._score_rows)
        return format_output(
            logits, tokens.astype(np.int64), plan, prompt_logprobs=prompt
        )

    def _score_rows(self, hidden: torch.Tensor) -> torch.Tensor:
        rows = hidden.shape[0]
        if rows > _SCORE_ROWS:
            raise ValueError(f"prompt scoring projects at most {_SCORE_ROWS} rows")
        padded = np.zeros((_SCORE_ROWS, hidden.shape[1]), dtype=np.float32)
        padded[:rows] = hidden.numpy()
        before = self.compilations.count
        logits = self._project(self.params, self._put(padded))
        self._check_compilations(before, "prompt scoring")
        return torch.from_numpy(np.array(logits)[:rows])

    def forward(self, plan: BatchPlan) -> ForwardOutput:
        return self._output(self._launch(plan))

    def enqueue(
        self, plan: BatchPlan, *, feedback: jax.Array | None = None
    ) -> _Completion | ForwardOutput:
        """Dispatch a decode step without waiting for the device.

        ``feedback`` holds the previous step's sampled tokens for the same rows
        in the same order; they replace the continuation plan's token IDs.
        """

        if plan.phase != BatchPhase.DECODE or plan.needs_logprobs:
            if feedback is not None:
                raise RuntimeError("a TPU continuation requires unscored decode")
            return self.forward(plan)
        step = self._launch(plan, feedback)
        continues = int(np.max(plan.context_lengths)) + 1 < self.max_seq_len
        return _Completion(self, step, continues)


__all__ = ["PostAcceptTpuError", "TpuModel", "TpuRunner"]
