"""Shared Pallas and XLA kernels for the Resin TPU backend."""

from resin.backend.tpu.kernels.fp8_matmul import (
    Fp8MatmulTileSizes,
    Fp8Weight,
    fp8_matmul,
    fp8_matmul_prequantized,
    get_tuned_block_sizes,
    pallas_fp8_matmul,
    quantize_fp8_act,
    quantize_fp8_weight,
    sharded_fp8_matmul,
    xla_fp8_matmul,
)
from resin.backend.tpu.kernels.norm_rope import (
    apply_rope,
    build_rope_cache,
    fused_add_rms_norm,
    pallas_rms_norm_fp8_quant,
    rms_norm,
    rms_norm_fp8_quant,
)
from resin.backend.tpu.kernels.ragged_paged_attention import (
    RpaBlockSizes,
    RpaCase,
    get_default_block_sizes,
    get_kv_cache_shape,
    get_rpa_block_sizes,
    get_vmem_estimate_bytes,
    quantize_kv,
    ragged_paged_attention,
)

__all__ = [
    "Fp8MatmulTileSizes",
    "Fp8Weight",
    "RpaBlockSizes",
    "RpaCase",
    "apply_rope",
    "build_rope_cache",
    "fp8_matmul",
    "fp8_matmul_prequantized",
    "fused_add_rms_norm",
    "get_default_block_sizes",
    "get_kv_cache_shape",
    "get_rpa_block_sizes",
    "get_tuned_block_sizes",
    "get_vmem_estimate_bytes",
    "pallas_fp8_matmul",
    "pallas_rms_norm_fp8_quant",
    "quantize_fp8_act",
    "quantize_fp8_weight",
    "quantize_kv",
    "ragged_paged_attention",
    "rms_norm",
    "rms_norm_fp8_quant",
    "sharded_fp8_matmul",
    "xla_fp8_matmul",
]
