from dataclasses import dataclass

from ..qwen3_dflash.config import DFlashConfig


@dataclass
class DSparkConfig(DFlashConfig):
    markov_rank: int = 0
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = False
    num_anchors: int = 1

    @classmethod
    def from_dict(cls, params: dict) -> "DSparkConfig":
        flat = dict(params)
        sub = flat.pop("dspark_config", None) or {}
        for k in (
            "markov_rank",
            "markov_head_type",
            "enable_confidence_head",
            "confidence_head_with_markov",
            "num_anchors",
        ):
            if k in sub:
                flat[k] = sub[k]
        return super().from_dict(flat)

    from_hf_dict = from_dict
