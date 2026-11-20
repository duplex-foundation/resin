# TPU model package boundaries

A TPU model family is a small plug-in over the model-agnostic backend in
`resin.backend.tpu`. Backend-neutral configuration and recipes live in
`resin.models.<family>` and `crates/resin-recipe`, shared with CUDA. The plug-in
in `resin_models_tpu.<family>` owns only what is specific to that model on TPU:

- the layer graph (`model.py`), which implements `resin.backend.tpu.runner.TpuModel`:
  `cache_specs`, `hidden_states`, `logits`;
- checkpoint mapping and weight placement (`weights.py`), exported as
  `load_model(checkpoint, mesh) -> (model, params)`;
- the partition spec of every weight, named next to the layer that uses it.

The backend owns the mesh, the KV arena, attention metadata, shape buckets,
sampling and the compile guard. A plug-in never allocates cache pages, never
reads request state and never branches on a request value, so a step compiles
once per bucket. Kernels (`ragged_paged_attention`, `fp8_matmul`, norms, RoPE)
are imported from `resin.backend.tpu.kernels`; a plug-in does not carry its own.

## Adding a family

1. Put geometry in `resin/models/<family>/config.py`, with a tiny CPU config.
2. Write `model.py` against `TpuModel`. Keep parameters as a NamedTuple pytree
   so they pass into the compiled step as an argument.
3. Write `weights.py`. Stream one tensor at a time to its final sharding, check
   every shape against the config, and name missing tensors.
4. Prove parity on the tiny config against the Transformers reference at tp=1
   and tp=2 with chunked prefill that crosses page boundaries, then add the
   full-size reference fixtures.
5. Register the preset in `deploy/models.py` with `backend="tpu"`.

## Qwen3

`load_model(checkpoint, mesh, *, quantization="fp8", kv_dtype=None, page_tokens=256)`.

- `quantization="fp8"` stores every projection as float8 e4m3 with a
  per-output-channel scale and runs W8A8 matmuls. Embeddings, norms and the
  LM head stay in the checkpoint dtype. `"none"` keeps dense weights; the CPU
  parity tests use it with a float32 cache.
- `kv_dtype=None` selects a float8 e4m3 cache.
- Q, K, V, gate and up are column-parallel (`P("tp", None)`). O and down are
  row-parallel (`P(None, "tp")`) and reduce over `tp` inside the matmul. The LM
  head shards its vocabulary rows, and the runner gathers the logits.
- Q and K are RMS-normalized per head before RoPE, which has no config flag in
  Transformers' `Qwen3Attention`. Query width is `heads * head_dim`, wider than
  `hidden_size`.
- Tied embeddings (0.6B, 1.7B, 4B) reuse the embedding rows as the LM head.
