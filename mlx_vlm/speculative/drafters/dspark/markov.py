from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _sample(logits: mx.array, temperature: float) -> mx.array:
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits * (1.0 / temperature), axis=-1)


class VanillaMarkov(nn.Module):
    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def step_bias(self, token_ids: mx.array, hidden: Optional[mx.array]) -> mx.array:
        return self.markov_w2(self.get_prev_embeddings(token_ids))

    def sample_block(self, base_logits, first_prev_token_ids, hidden, temperature):
        L = base_logits.shape[1]
        tokens, prev = [], first_prev_token_ids
        for k in range(L):
            hk = None if hidden is None else hidden[:, k]
            prev = _sample(base_logits[:, k] + self.step_bias(prev, hk), temperature)
            tokens.append(prev)
        return mx.stack(tokens, axis=1)


class GatedMarkovHead(VanillaMarkov):
    def __init__(self, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size, markov_rank)
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def step_bias(self, token_ids: mx.array, hidden: Optional[mx.array]) -> mx.array:
        prev_emb = self.get_prev_embeddings(token_ids)
        gate = mx.sigmoid(self.gate_proj(mx.concatenate([hidden, prev_emb], axis=-1)))
        return self.markov_w2(gate * prev_emb)


class RNNHead(VanillaMarkov):
    def __init__(self, vocab_size: int, markov_rank: int, hidden_size: int):
        super().__init__(vocab_size, markov_rank)
        self.markov_rank = markov_rank
        self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def _rnn_step(self, state, prev_emb, hidden):
        proj = self.joint_proj(mx.concatenate([state, prev_emb, hidden], axis=-1))
        gate_raw, cand_raw, out_raw = mx.split(proj, 3, axis=-1)
        gate = mx.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * mx.tanh(cand_raw)
        return new_state, self.markov_w2(mx.tanh(out_raw))

    def sample_block(self, base_logits, first_prev_token_ids, hidden, temperature):
        B, L = base_logits.shape[:2]
        state = mx.zeros((B, self.markov_rank), dtype=base_logits.dtype)
        tokens, prev = [], first_prev_token_ids
        for k in range(L):
            state, bias = self._rnn_step(
                state, self.get_prev_embeddings(prev), hidden[:, k]
            )
            prev = _sample(base_logits[:, k] + bias, temperature)
            tokens.append(prev)
        return mx.stack(tokens, axis=1)


def build_markov_head(config) -> Optional[nn.Module]:
    rank = int(getattr(config, "markov_rank", 0))
    if rank <= 0:
        return None
    kind = str(getattr(config, "markov_head_type", "vanilla")).lower()
    if kind == "vanilla":
        return VanillaMarkov(config.vocab_size, rank)
    if kind == "gated":
        return GatedMarkovHead(config.vocab_size, rank, config.hidden_size)
    if kind == "rnn":
        return RNNHead(config.vocab_size, rank, config.hidden_size)
    raise ValueError(f"Unsupported markov_head_type: {kind!r}")
