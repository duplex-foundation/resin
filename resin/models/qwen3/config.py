"""Dense Qwen3 checkpoint geometry.

Query and key width is ``num_attention_heads * head_dim``. On the pinned
Qwen3-32B checkpoint that width is 8192, wider than ``hidden_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

from resin.models.hf_config import (
    PositiveInt,
    invalid_dimensions,
    parse_hf_config,
    require_options,
)

# Published BF16 checkpoint.
QWEN3_32B_REPO = "Qwen/Qwen3-32B"
QWEN3_32B_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    vocab_size: PositiveInt
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    num_hidden_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    head_dim: PositiveInt
    max_position_embeddings: PositiveInt
    rms_norm_eps: float
    rope_theta: float
    eos_token_id: int
    bos_token_id: int | None
    dtype: str = "bfloat16"
    tie_word_embeddings: bool = False

    @classmethod
    def from_hugging_face(cls, values: dict[str, Any]) -> Self:
        if values.get("model_type") != "qwen3":
            raise ValueError(
                f"expected a Qwen3 checkpoint, got {values.get('model_type')!r}"
            )
        require_options(
            values,
            {
                "hidden_act": "silu",
                "attention_bias": False,
                "attention_dropout": 0.0,
                "use_sliding_window": False,
                "rope_scaling": None,
            },
            "Qwen3",
        )
        rope = values.get("rope_parameters") or {}
        if not isinstance(rope, dict):
            raise ValueError("Qwen3 rope_parameters must be an object")
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Qwen3 rope scaling is not supported")
        theta = rope.get("rope_theta", values.get("rope_theta"))
        if theta is None:
            raise KeyError("rope_theta")
        config = parse_hf_config(
            cls,
            values,
            dtype=str(values.get("dtype", values.get("torch_dtype", "bfloat16"))),
            rope_theta=float(theta),
        )
        invalid = list(invalid_dimensions(config))
        if invalid:
            raise ValueError(f"Qwen3 dimensions must be positive: {invalid}")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        if config.head_dim % 2:
            raise ValueError("head_dim must be even")
        if config.rms_norm_eps <= 0 or config.rope_theta <= 0:
            raise ValueError("Qwen3 norm epsilon and RoPE theta must be positive")
        return config


def tiny_qwen3_config() -> dict[str, Any]:
    """Two-layer Qwen3 geometry for a random-weight CPU check.

    Query width is 64 and ``hidden_size`` is 32, the same mismatch as Qwen3-32B.
    Weights are not included; the caller initializes them.
    """

    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
        "model_type": "qwen3",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": 1_000_000,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_sliding_window": False,
        "vocab_size": 128,
    }


__all__ = [
    "QWEN3_32B_REPO",
    "QWEN3_32B_REVISION",
    "Qwen3Config",
    "tiny_qwen3_config",
]
