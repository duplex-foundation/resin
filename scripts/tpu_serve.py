"""Serve a registered TPU preset through Resin's HTTP server.

Run it under the chip queue, which sources ``tpu-env.sh`` and holds chip 0:

    RESIN_TPU_QUEUE=/path/to/tpu-queue.sh scripts/tpu_serve.sh \\
        --model qwen3-32b --checkpoint PATH --workload 1k/8k

Prefix caching stays on. Pass ``--no-prefix-cache`` for the InferenceX runs,
which disable it. A step that compiles after warmup fails the run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.models import (
    MODEL_SPECS,
    TPU_WORKLOADS,
    create_recipe,
    create_runner,
    get_model,
    require_tpu_chip,
)


def main(argv=None) -> None:
    presets = [name for name, spec in MODEL_SPECS.items() if spec.backend == "tpu"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=presets, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workload", choices=tuple(TPU_WORKLOADS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--prefix-cache", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    require_tpu_chip(queued=True)

    from resin import Engine, SchedulerConfig
    from resin.server import HttpServer

    spec = get_model(args.model)
    workload = TPU_WORKLOADS[args.workload]
    started = time.perf_counter()
    runner = create_runner(
        spec,
        args.checkpoint.expanduser(),
        device="tpu",
        context_length=workload.max_model_len,
        max_batch_size=workload.max_num_seqs,
        prefill_chunk_size=spec.prefill_chunk_size,
        draft_tokens=0,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        workload=args.workload,
    )
    recipe = create_recipe(spec, args.checkpoint.expanduser(), runner=runner)
    engine = Engine(
        runner,
        max_running_requests=workload.max_num_seqs,
        context_length=workload.max_model_len,
        prefix_cache=args.prefix_cache,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=spec.prefill_chunk_size,
            prefill_chunk_size=spec.prefill_chunk_size,
        ),
    )
    warmup = getattr(runner, "benchmark_metadata", {}).get("warmup_seconds")
    ready = f"[serve] {spec.name} {args.workload} ready in {time.perf_counter() - started:.1f}s"
    if warmup is not None:
        ready += f" (warmup {warmup:.1f}s)"
    print(
        ready + f" at http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        HttpServer(
            engine,
            recipe=recipe,
            model=spec.repo_id,
            enable_anthropic=spec.enable_anthropic,
        ).serve(host=args.host, port=args.port)
    finally:
        if not engine.closed:
            engine.close()


if __name__ == "__main__":
    main()
