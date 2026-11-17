"""Qwen3-32B BF16 reference fixtures.

The oracle is the on-disk manifest, logit rows, WikiText-2 windows, and the
pinned MMLU sample. Greedy tokens are the argmax of the stored logits.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
MMLU_SHA256 = "aea69f647253862af0c427c27d158f052e749f868d2241fe672ba26e9c0b7476"
PAGE = 256
VOCAB = 151936


def _bf16_argmax(row: np.ndarray) -> int:
    floated = (row.astype(np.uint32) << 16).view(np.float32)
    return int(floated.argmax())


def _root() -> Path:
    raw = os.environ.get("RESIN_QWEN3_FIXTURES")
    if not raw or not Path(raw).is_dir():
        pytest.skip(
            "set RESIN_QWEN3_FIXTURES to the generated Qwen3-32B fixture directory"
        )
    return Path(raw)


def _manifest() -> dict:
    return json.loads((_root() / "manifest.json").read_text())


def test_generator_does_not_import_jax() -> None:
    source = (
        Path(__file__).parents[1] / "scripts" / "generate_qwen3_reference.py"
    ).read_text()
    assert "import jax" not in source
    assert "import jaxlib" not in source
    assert "import libtpu" not in source
    config = (Path(__file__).parents[1] / "resin/models/qwen3/config.py").read_text()
    assert f'QWEN3_32B_REVISION = "{REVISION}"' in config
    assert f'REVISION = "{REVISION}"' in source


def test_manifest_pins_the_bf16_checkpoint_and_page_size() -> None:
    manifest = _manifest()

    assert manifest["model"] == "Qwen/Qwen3-32B"
    assert manifest["revision"] == REVISION
    assert manifest["dtype"] == "bfloat16"
    assert manifest["attn_implementation"] == "eager"
    assert manifest["page_size"] == PAGE
    assert manifest["vocab_size"] == VOCAB
    assert manifest["greedy"] is True
    assert manifest["mmlu_sha256"] == MMLU_SHA256


def test_prompt_set_covers_prefill_decode_continuation_and_page_boundaries() -> None:
    cases = {case["id"]: case for case in _manifest()["cases"]}

    assert 10 <= len(cases) <= 14
    lengths = {case["prompt_len"] for case in cases.values()}
    assert {255, 256, 257, 511, 512, 513} <= lengths
    crossed = [
        case
        for case in cases.values()
        if case["prompt_len"] < PAGE
        and case["prompt_len"] + len(case["generated_ids"]) > PAGE
        and len(case["generated_ids"]) >= 36
    ]
    assert crossed
    continuation = cases["continue_page_255"]
    parent = cases[continuation["parent"]]
    prefix = continuation["prefix_new_tokens"]
    assert continuation["exercises"][:1] == ["continuation"]
    assert continuation["prompt_len"] == parent["prompt_len"] + prefix
    assert continuation["prompt_len"] > PAGE
    assert continuation["matches_parent"] is True
    assert (
        continuation["generated_ids"]
        == parent["generated_ids"][prefix : prefix + len(continuation["generated_ids"])]
    )
    assert continuation["prompt_ids"] == (
        parent["prompt_ids"] + parent["generated_ids"][:prefix]
    )


def test_greedy_tokens_are_the_logit_argmax() -> None:
    for case in _manifest()["cases"]:
        with np.load(_root() / case["logits_file"], allow_pickle=False) as data:
            logits = data["logits_bf16"]
            generated = data["generated_ids"]
            prompt = data["prompt_ids"]
        assert logits.dtype == np.uint16
        assert logits.shape == (len(case["generated_ids"]), VOCAB)
        assert generated.tolist() == case["generated_ids"]
        assert prompt.tolist() == case["prompt_ids"]
        assert len(case["generated_ids"]) >= 16
        if case.get("parent") is None:
            assert len(case["generated_ids"]) >= 36
        assert [_bf16_argmax(row) for row in logits] == case["generated_ids"]


def test_mmlu_sample_is_the_step2_gate_file() -> None:
    path = _root() / "mmlu-1000.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert digest == MMLU_SHA256
    assert len(rows) == 1000
    assert set(rows[0]) == {"question", "choices", "answer", "subject"}


def test_wikitext_windows_are_contiguous_2048_token_slices() -> None:
    rows = [
        json.loads(line)
        for line in (_root() / "wikitext-2-windows.jsonl").read_text().splitlines()
    ]
    manifest = _manifest()
    stream = rows[0]["token_ids"]

    assert len(rows) == 4
    for index, row in enumerate(rows):
        assert row["dataset"] == "Salesforce/wikitext"
        assert row["revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3"
        assert row["config"] == "wikitext-2-raw-v1"
        assert row["split"] == "test"
        assert row["window"] == index
        assert row["token_start"] == index * 2048
        assert row["token_len"] == 2048
        assert len(row["token_ids"]) == 2048
        assert row["text"]
    assert manifest["wikitext"]["window_tokens"] == 2048
    for case in manifest["cases"]:
        if case.get("token_start") == 0:
            assert case["prompt_ids"] == stream[: case["prompt_len"]]
