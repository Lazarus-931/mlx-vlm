from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..qwen3_dflash.dflash import DFlashDraftModel
from .config import DSparkConfig
from .markov import build_markov_head


def _confident_prefix_length(
    conf_logits: mx.array, block_size: int, threshold: float
) -> int:
    if threshold <= 0.0:
        return int(block_size)
    below = (mx.sigmoid(conf_logits[0]) < threshold).tolist()
    for i, is_below in enumerate(below):
        if is_below:
            return i
    return int(block_size)


class DSparkDraftModel(DFlashDraftModel):
    def __init__(self, config: DSparkConfig):
        super().__init__(config)
        self.markov_head = build_markov_head(config)
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", False)
        )
        self.confidence_head = None
        if getattr(config, "enable_confidence_head", False):
            in_dim = config.hidden_size + (
                int(config.markov_rank) if self.confidence_head_with_markov else 0
            )
            self.confidence_head = nn.Linear(in_dim, 1)

    def predict_confidence(
        self, hidden: mx.array, prev_token_ids: mx.array
    ) -> Optional[mx.array]:
        if self.confidence_head is None:
            return None
        feats = hidden
        if self.confidence_head_with_markov:
            prev_emb = self.markov_head.get_prev_embeddings(prev_token_ids).astype(
                hidden.dtype
            )
            feats = mx.concatenate([hidden, prev_emb], axis=-1)
        return self.confidence_head(feats).squeeze(-1)

    def _block_inputs(
        self, last_bonus, block_size: int, token_dtype: mx.Dtype
    ) -> mx.array:
        mask_id = int(self.config.mask_token_id)
        if isinstance(last_bonus, int):
            return mx.array(
                [[last_bonus] + [mask_id] * (block_size - 1)], dtype=token_dtype
            )
        masks = mx.full(
            (last_bonus.shape[0], block_size - 1), mask_id, dtype=token_dtype
        )
        return mx.concatenate([last_bonus[:, None].astype(token_dtype), masks], axis=1)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache: list,
        block_size: int,
        temperature: float = 0.0,
        confidence_threshold: float = 0.0,
        token_dtype: mx.Dtype = mx.int32,
    ):
        block = self._block_inputs(last_bonus, block_size, token_dtype)
        anchor = block[:, 0]
        proposal_hidden = self._hidden(block, hidden, cache)[:, :block_size]
        base_logits = self._logits(proposal_hidden)

        if self.markov_head is None:
            tokens = (
                mx.argmax(base_logits, axis=-1)
                if temperature <= 0.0
                else mx.random.categorical(base_logits * (1.0 / temperature), axis=-1)
            )
        else:
            tokens = self.markov_head.sample_block(
                base_logits, anchor, proposal_hidden, temperature
            )

        conf = None
        cut = block_size
        if self.confidence_head is not None:
            prev_ids = mx.concatenate([anchor[:, None], tokens[:, :-1]], axis=1)
            conf = self.predict_confidence(proposal_hidden, prev_ids)
            cut = _confident_prefix_length(conf, block_size, confidence_threshold)
        return tokens[:, :cut], conf, cut

    def sanitize(self, weights: dict) -> dict:
        return {
            (k[len("model.") :] if k.startswith("model.") else k): v
            for k, v in weights.items()
        }
