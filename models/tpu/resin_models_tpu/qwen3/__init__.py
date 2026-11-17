"""Dense Qwen3 on the Resin TPU backend: layer graph, weight mapping and sharding."""

from resin_models_tpu.qwen3.model import LayerParams, Qwen3Model, Qwen3Params
from resin_models_tpu.qwen3.weights import load_model

__all__ = ["LayerParams", "Qwen3Model", "Qwen3Params", "load_model"]
