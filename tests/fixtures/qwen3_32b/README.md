# Qwen3-32B BF16 reference fixtures

`scripts/generate_qwen3_reference.py` writes the transformers BF16 greedy-token
and logit oracle for `Qwen/Qwen3-32B` at revision
`9216db5781bf21249d130ec9da846c4624c16137`. Git does not track that output.
The files land in this directory unless you pass `--out`.

`logits/<case>.npz` holds `logits_bf16` (uint16 bfloat16 bits, one row per
step), `generated_ids`, and `prompt_ids`. `manifest.json` records the run.
`wikitext-2-windows.jsonl` holds four 2048-token WikiText-2 raw v1 windows.
`mmlu-1000.jsonl` is a copy of the step 2 gate sample, sha256
`aea69f647253862af0c427c27d158f052e749f868d2241fe672ba26e9c0b7476`.

From the repository root, on CPU, with torch capped at 56 threads:

```bash
JAX_PLATFORMS=cpu TPU_VISIBLE_CHIPS= \
  numactl --cpunodebind=1 --membind=1 \
  python scripts/generate_qwen3_reference.py \
    --model-dir /path/to/Qwen3-32B \
    --mmlu /path/to/mmlu-1000.jsonl
```

The script binds to NUMA node 1 if you did not, and it exits if JAX or
libtpu is already imported. `--mmlu` must be that pinned sample. The
generator copies it. It does not build a new one.

Point the test at the directory you generated:

```bash
RESIN_QWEN3_FIXTURES=tests/fixtures/qwen3_32b pytest tests/test_qwen3_reference.py
```

The data checks skip when `RESIN_QWEN3_FIXTURES` is unset or the directory
is missing.
