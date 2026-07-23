"""In-house DWQ (distilled weight quantization).

Recovers low-bit quality by distilling a quantized student toward the
full-precision teacher: the packed integer weights are frozen and only the
continuous quantization ``scales``/``biases`` are optimized to match the
teacher's output distribution over a calibration set. Pure mlx.
"""

from typing import Callable, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten


def logits_forward(model: nn.Module) -> Optional[Callable[[mx.array], mx.array]]:
    """Return a callable mapping input ids to language-model logits, or None."""
    lm = getattr(model, "language_model", None)
    candidates = [lm, model]

    def make(mod):
        def call(ids):
            for attempt in (lambda: mod(ids), lambda: mod(input_ids=ids)):
                try:
                    out = attempt()
                except Exception:
                    continue
                logits = out.logits if hasattr(out, "logits") else out
                if isinstance(logits, mx.array) and logits.ndim == 3:
                    return logits
            raise RuntimeError("no logits")

        return call

    for mod in candidates:
        if mod is None:
            continue
        fn = make(mod)
        try:
            probe = mx.zeros((1, 2), dtype=mx.int32)
            mx.eval(fn(probe))
            return fn
        except Exception:
            continue
    return None


def capture_teacher(
    forward: Callable[[mx.array], mx.array],
    inputs: List[mx.array],
    temp: float = 1.0,
) -> List[mx.array]:
    """Cache the teacher's softened output distribution for each calibration input."""
    teacher = []
    for ids in inputs:
        logits = forward(ids).astype(mx.float32) / temp
        probs = mx.softmax(logits, axis=-1).astype(mx.float16)
        mx.eval(probs)
        teacher.append(probs)
    return teacher


def apply_dwq(
    model: nn.Module,
    forward: Callable[[mx.array], mx.array],
    inputs: List[mx.array],
    teacher: List[mx.array],
    steps: int = 200,
    lr: float = 3e-6,
    temp: float = 1.0,
) -> Dict[str, object]:
    """Distill the quantized model's scales/biases toward cached teacher outputs."""
    model.freeze()
    model.unfreeze(recurse=True, keys=["scales", "biases"])
    n_trainable = len(tree_flatten(model.trainable_parameters()))
    if n_trainable == 0 or steps <= 0:
        return {"trained_tensors": n_trainable, "final_loss": None, "first_loss": None}

    opt = optim.Adam(learning_rate=lr)

    def loss_fn(ids, tprob):
        logits = forward(ids).astype(mx.float32) / temp
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(mx.sum(tprob.astype(mx.float32) * logp, axis=-1))

    step_fn = nn.value_and_grad(model, loss_fn)

    first_loss = None
    final_loss = None
    for step in range(steps):
        idx = step % len(inputs)
        loss, grads = step_fn(inputs[idx], teacher[idx])
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        final_loss = float(loss.item())
        if first_loss is None:
            first_loss = final_loss

    model.freeze()
    return {
        "trained_tensors": n_trainable,
        "first_loss": first_loss,
        "final_loss": final_loss,
    }
