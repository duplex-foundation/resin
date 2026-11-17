"""Load a Hugging Face Qwen3 checkpoint onto a ``tp`` mesh.

Tensors stream one at a time from the memory-mapped safetensors files to their
final sharding, so host memory stays near one weight. With ``quantization="fp8"``
each linear weight is quantized on device after it is placed, per output
channel, so a row-parallel weight's scale uses the absmax over the full input
dimension. Column-parallel ``(q, k, v)`` and ``(gate, up)`` projections are
concatenated per ``tp`` shard after sharding so each die holds contiguous
``[q_local; k_local; v_local]`` and ``[gate_local; up_local]`` slices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from resin.backend.loading import read_config
from resin.backend.safetensors import SafetensorReader, indexed_tensor_files
from resin.backend.tpu.attention import AttentionImpl
from resin.backend.tpu.kernels import Fp8Weight, build_rope_cache, quantize_fp8_weight
from resin.backend.tpu.mesh import TP_AXIS
from resin.models.qwen3.config import Qwen3Config
from resin_models_tpu.qwen3.model import (
    COLUMN,
    ROW,
    LayerParams,
    Linear,
    Qwen3Model,
    Qwen3Params,
)

Quantization = Literal["fp8", "none"]


def _host_array(tensor: Any, dtype: Any) -> np.ndarray:
    """Convert a CPU torch tensor to NumPy in ``dtype``, avoiding a float32 copy for bfloat16."""

    import torch

    if tensor.dtype == torch.bfloat16:
        array = tensor.view(torch.int16).numpy().view(jnp.bfloat16)
    else:
        array = tensor.to(torch.float32).numpy()
    return array.astype(dtype, copy=False)


class _Loader:
    def __init__(
        self, root: Path, config: Qwen3Config, mesh: Mesh, quantization: Quantization
    ) -> None:
        self.files = indexed_tensor_files(root)
        self.reader = SafetensorReader()
        self.config = config
        self.mesh = mesh
        self.dtype = jnp.dtype(config.dtype)
        self.quantizers: dict[P, Callable[[jax.Array], Fp8Weight]] = {}
        self.fp8_packers: dict[int, Callable[..., Fp8Weight]] = {}
        self.dense_packers: dict[int, Callable[..., jax.Array]] = {}
        col_sharding = NamedSharding(mesh, COLUMN)
        scale_sharding = NamedSharding(mesh, P(TP_AXIS))

        if quantization == "fp8":
            for spec in (COLUMN, ROW):
                self.quantizers[spec] = jax.jit(
                    quantize_fp8_weight,
                    out_shardings=cast(Any, Fp8Weight)(
                        NamedSharding(mesh, spec), NamedSharding(mesh, P(spec[0]))
                    ),
                )
            for arity in (2, 3):

                def _make_fp8_packer(n: int) -> Callable[..., Fp8Weight]:
                    q_specs = (COLUMN,) * n
                    s_specs = (P(TP_AXIS),) * n

                    def _pack(*parts: Fp8Weight) -> Fp8Weight:
                        qvalue = jax.shard_map(
                            lambda *qs: jnp.concatenate(qs, axis=0),
                            mesh=mesh,
                            in_specs=q_specs,
                            out_specs=COLUMN,
                        )(*(p.qvalue for p in parts))
                        scale = jax.shard_map(
                            lambda *ss: jnp.concatenate(ss, axis=0),
                            mesh=mesh,
                            in_specs=s_specs,
                            out_specs=P(TP_AXIS),
                        )(*(p.scale for p in parts))
                        return Fp8Weight(qvalue=qvalue, scale=scale)

                    return jax.jit(
                        _pack,
                        out_shardings=cast(Any, Fp8Weight)(
                            col_sharding, scale_sharding
                        ),
                    )

                self.fp8_packers[arity] = _make_fp8_packer(arity)
        else:
            for arity in (2, 3):

                def _make_dense_packer(n: int) -> Callable[..., jax.Array]:
                    in_specs = (COLUMN,) * n

                    def _pack(*parts: jax.Array) -> jax.Array:
                        return jax.shard_map(
                            lambda *xs: jnp.concatenate(xs, axis=0),
                            mesh=mesh,
                            in_specs=in_specs,
                            out_specs=COLUMN,
                        )(*parts)

                    return jax.jit(_pack, out_shardings=col_sharding)

                self.dense_packers[arity] = _make_dense_packer(arity)

    def tensor(self, name: str, shape: tuple[int, ...], spec: P) -> jax.Array:
        if name not in self.files:
            raise KeyError(f"checkpoint is missing tensor {name}")
        tensor = self.reader.read(self.files[name], name)
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, expected {shape} for this config"
            )
        return jax.device_put(
            _host_array(tensor, self.dtype), NamedSharding(self.mesh, spec)
        )

    def linear(self, name: str, shape: tuple[int, ...], spec: P) -> Linear:
        weight = self.tensor(name, shape, spec)
        quantize = self.quantizers.get(spec)
        return weight if quantize is None else quantize(weight)

    def _pack_column(self, parts: Sequence[Linear]) -> Linear:
        arity = len(parts)
        if isinstance(parts[0], Fp8Weight):
            return self.fp8_packers[arity](*parts)
        return self.dense_packers[arity](*parts)

    def layer(self, index: int) -> LayerParams:
        config = self.config
        hidden, head_dim = config.hidden_size, config.head_dim
        query = config.num_attention_heads * head_dim
        kv = config.num_key_value_heads * head_dim
        ffn = config.intermediate_size
        prefix = f"model.layers.{index}."
        norm = (hidden,), P()
        q = self.linear(prefix + "self_attn.q_proj.weight", (query, hidden), COLUMN)
        k = self.linear(prefix + "self_attn.k_proj.weight", (kv, hidden), COLUMN)
        v = self.linear(prefix + "self_attn.v_proj.weight", (kv, hidden), COLUMN)
        gate = self.linear(prefix + "mlp.gate_proj.weight", (ffn, hidden), COLUMN)
        up = self.linear(prefix + "mlp.up_proj.weight", (ffn, hidden), COLUMN)
        return LayerParams(
            input_norm=self.tensor(prefix + "input_layernorm.weight", *norm),
            q_norm=self.tensor(prefix + "self_attn.q_norm.weight", (head_dim,), P()),
            k_norm=self.tensor(prefix + "self_attn.k_norm.weight", (head_dim,), P()),
            post_norm=self.tensor(prefix + "post_attention_layernorm.weight", *norm),
            qkv=self._pack_column((q, k, v)),
            o=self.linear(prefix + "self_attn.o_proj.weight", (hidden, query), ROW),
            gate_up=self._pack_column((gate, up)),
            down=self.linear(prefix + "mlp.down_proj.weight", (hidden, ffn), ROW),
        )

    def params(self) -> Qwen3Params:
        config = self.config
        embed_shape = (config.vocab_size, config.hidden_size)
        rep = NamedSharding(self.mesh, P())
        with self.reader:
            embed = self.tensor("model.embed_tokens.weight", embed_shape, P())
            layers = []
            for index in range(config.num_hidden_layers):
                layers.append(self.layer(index))
                jax.block_until_ready(layers[-1])
            if config.tie_word_embeddings:
                lm_head = jax.device_put(embed, NamedSharding(self.mesh, COLUMN))
            else:
                lm_head = self.tensor("lm_head.weight", embed_shape, COLUMN)
            norm = self.tensor("model.norm.weight", (config.hidden_size,), P())
        cos_full, sin_signed = build_rope_cache(
            config.max_position_embeddings, config.head_dim, config.rope_theta
        )
        rope_cache = (jax.device_put(cos_full, rep), jax.device_put(sin_signed, rep))
        return Qwen3Params(embed, tuple(layers), norm, lm_head, rope_cache)


def load_model(
    checkpoint: str | Path,
    mesh: Mesh,
    *,
    quantization: Quantization = "fp8",
    kv_dtype: Any = None,
    page_tokens: int = 256,
    impl: AttentionImpl | None = None,
) -> tuple[Qwen3Model, Qwen3Params]:
    """Build the model and place its weights on ``mesh``.

    ``quantization="fp8"`` stores linear weights as float8 e4m3 with
    per-channel scales and runs W8A8 matmuls; embeddings, norms and the LM head
    stay in the checkpoint dtype. ``"none"`` keeps every weight dense.
    ``kv_dtype`` defaults to float8 e4m3, the Qwen3-32B serving cache.
    """

    if quantization not in ("fp8", "none"):
        raise ValueError(f"quantization must be 'fp8' or 'none', not {quantization!r}")
    root = Path(checkpoint)
    config = Qwen3Config.from_hugging_face(read_config(root))
    model = Qwen3Model(
        config,
        mesh,
        jnp.float8_e4m3fn if kv_dtype is None else kv_dtype,
        page_tokens=page_tokens,
        impl=impl,
    )
    return model, _Loader(root, config, mesh, quantization).params()


__all__ = ["Quantization", "load_model"]
