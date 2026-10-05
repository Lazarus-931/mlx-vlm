from ..base import install_auto_processor_patch
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .agnes import LanguageModel, Model, VisionModel
from .config import ModelConfig, TextConfig, VisionConfig

install_auto_processor_patch("agnes", Qwen3VLProcessor)
