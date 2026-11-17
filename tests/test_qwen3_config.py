"""Dense Qwen3 checkpoint geometry.

The literals are the published ``Qwen/Qwen3-32B`` config at revision
``9216db5781bf21249d130ec9da846c4624c16137``. Query width is
``num_attention_heads * head_dim``, which is wider than ``hidden_size``.
"""

from __future__ import annotations

import pytest
from transformers import Qwen3Config as HFQwen3Config

from resin.models.qwen3.config import (
    QWEN3_32B_REPO,
    QWEN3_32B_REVISION,
    Qwen3Config,
    tiny_qwen3_config,
)


def _qwen3_32b() -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "max_position_embeddings": 40960,
        "model_type": "qwen3",
        "num_attention_heads": 64,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 1_000_000,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_sliding_window": False,
        "vocab_size": 151936,
    }


def test_qwen3_32b_query_width_exceeds_hidden_size() -> None:
    config = Qwen3Config.from_hugging_face(_qwen3_32b())

    assert config.vocab_size == 151936
    assert config.hidden_size == 5120
    assert config.intermediate_size == 25600
    assert config.num_hidden_layers == 64
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.max_position_embeddings == 40960
    assert config.rope_theta == 1_000_000
    assert config.rms_norm_eps == 1e-6
    assert config.eos_token_id == 151645
    assert config.bos_token_id == 151643
    assert config.dtype == "bfloat16"
    assert config.tie_word_embeddings is False
    assert QWEN3_32B_REPO == "Qwen/Qwen3-32B"
    assert QWEN3_32B_REVISION == "9216db5781bf21249d130ec9da846c4624c16137"


def test_tiny_config_is_a_two_layer_qwen3_with_wide_queries() -> None:
    raw = tiny_qwen3_config()
    config = Qwen3Config.from_hugging_face(raw)

    assert raw["num_hidden_layers"] == 2
    assert raw["hidden_size"] == 32
    assert raw["num_attention_heads"] == 4
    assert raw["head_dim"] == 16
    assert raw["vocab_size"] == 128
    assert raw["torch_dtype"] == "float32"
    assert config.num_hidden_layers == 2
    assert config.hidden_size == 32
    assert config.head_dim == 16
    assert config.dtype == "float32"
    assert config.num_attention_heads * config.head_dim == 64


def test_tiny_config_loads_in_transformers_as_full_attention() -> None:
    raw = tiny_qwen3_config()
    hf = HFQwen3Config.from_dict(raw)

    assert hf.model_type == "qwen3"
    assert (hf.hidden_size, hf.head_dim, hf.num_hidden_layers) == (32, 16, 2)
    assert hf.layer_types == ["full_attention", "full_attention"]
    assert hf.rope_parameters == {"rope_theta": 1_000_000, "rope_type": "default"}


def test_transformers_export_keeps_rope_and_dtype() -> None:
    exported = HFQwen3Config.from_dict(tiny_qwen3_config()).to_dict()

    config = Qwen3Config.from_hugging_face(exported)

    assert config.rope_theta == 1_000_000
    assert config.dtype == "float32"
    assert config.head_dim == 16


def test_non_default_rope_type_is_rejected() -> None:
    raw = tiny_qwen3_config()
    raw["rope_parameters"] = {
        "rope_type": "yarn",
        "rope_theta": 1_000_000,
        "factor": 2.0,
    }

    with pytest.raises(ValueError, match="rope scaling"):
        Qwen3Config.from_hugging_face(raw)


def test_tied_embeddings_parse() -> None:
    raw = _qwen3_32b()
    raw["tie_word_embeddings"] = True

    config = Qwen3Config.from_hugging_face(raw)

    assert config.tie_word_embeddings is True


def test_moe_checkpoint_is_rejected() -> None:
    raw = _qwen3_32b()
    raw["model_type"] = "qwen3_moe"

    with pytest.raises(ValueError, match="qwen3_moe"):
        Qwen3Config.from_hugging_face(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_sliding_window", True, "use_sliding_window"),
        ("rope_scaling", {"rope_type": "yarn", "factor": 2.0}, "rope_scaling"),
        ("attention_bias", True, "attention_bias"),
        ("hidden_act", "gelu", "hidden_act"),
        ("attention_dropout", 0.1, "attention_dropout"),
    ],
)
def test_unsupported_qwen3_options_are_rejected(
    field: str, value: object, message: str
) -> None:
    raw = _qwen3_32b()
    raw[field] = value

    with pytest.raises(ValueError, match=message):
        Qwen3Config.from_hugging_face(raw)


def test_kv_heads_must_divide_query_heads() -> None:
    raw = _qwen3_32b()
    raw["num_key_value_heads"] = 3

    with pytest.raises(ValueError, match="KV heads"):
        Qwen3Config.from_hugging_face(raw)


def test_head_dim_must_be_even() -> None:
    raw = tiny_qwen3_config()
    raw["head_dim"] = 15

    with pytest.raises(ValueError, match="even"):
        Qwen3Config.from_hugging_face(raw)


def test_hidden_size_must_be_positive() -> None:
    raw = tiny_qwen3_config()
    raw["hidden_size"] = 0

    with pytest.raises(ValueError, match="hidden_size"):
        Qwen3Config.from_hugging_face(raw)
