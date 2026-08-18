"""Backward-compatible FP8 conversion helpers for Qwen checkpoints.

Historically these helpers live in this module. They now forward to the
canonical definitions in ``language.py``.
"""

from .language import (
    FP8_BLOCK_SIZE,
    MLX_MXFP8_QUANTIZATION,
    _dequantize_qwen_fp8_weight,
    convert_qwen_fp8_weights,
    make_quantization_config,
    quantize_qwen_fp8_weight,
)

__all__ = [
    "FP8_BLOCK_SIZE",
    "MLX_MXFP8_QUANTIZATION",
    "_dequantize_qwen_fp8_weight",
    "convert_qwen_fp8_weights",
    "make_quantization_config",
    "quantize_qwen_fp8_weight",
]
