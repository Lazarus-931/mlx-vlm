from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_vl.config import VisionConfig as Qwen3VLVisionConfig
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config


def _resolve_eos_token_id(eos_token_id, text_config):
    if eos_token_id is not None:
        return eos_token_id
    if isinstance(text_config, dict):
        return text_config.get("eos_token_id")
    return getattr(text_config, "eos_token_id", None)


@dataclass
class VisionConfig(Qwen3VLVisionConfig):
    model_type: str = "agnes_vision"

    def __post_init__(self):
        self.deepstack_visual_indexes = []


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    intermediate_size: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    eos_token_id: Optional[Union[int, List[int]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = 256
    parallel_ffn_intermediate_size: int = (
        0  # width of the per-layer parallel SwiGLU branch (0 disables it)
    )
    global_attention_interval: int = 4
    layer_types: Optional[List[str]] = None
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        }
    )

    def __post_init__(self):
        # Alias so machinery copied from qwen3_5 that reads the old field name works.
        self.full_attention_interval = self.global_attention_interval

        if self.layer_types is None:
            self.layer_types = [
                (
                    "agnes_global_attention"
                    if (i + 1) % self.global_attention_interval == 0
                    else "agnes_delta_attention"
                )
                for i in range(self.num_hidden_layers)
            ]

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")
            required_keys = {
                "mrope_section",
                "type",
                "rope_theta",
                "partial_rotary_factor",
            }
            if not all(key in self.rope_parameters for key in required_keys):
                raise ValueError(f"rope_parameters must contain keys {required_keys}")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 248056
    video_token_id: int = 248057
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    vocab_size: int = 248320
    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id
        self.eos_token_id = _resolve_eos_token_id(self.eos_token_id, self.text_config)

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["vision_config"] = _maybe_deserialize_config(
            VisionConfig, params.get("vision_config")
        )
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config"), require_all_fields=True
        )
        return cls(**_config_kwargs(cls, params))
