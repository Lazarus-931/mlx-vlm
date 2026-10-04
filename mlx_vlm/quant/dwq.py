"""In-house DWQ (distilled weight quantization).

Recovers low-bit quality by distilling a quantized student toward the
full-precision teacher: the packed integer weights are frozen and only the
continuous quantization ``scales``/``biases`` are optimized to match the
teacher's output distribution over a calibration set. Pure mlx.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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


def save_teacher_targets(
    forward: Callable[[mx.array], mx.array],
    inputs: List[mx.array],
    target_dir: Union[str, Path],
    split: str = "train",
    top_k: int = 1024,
) -> None:
    """Persist compact teacher targets so the teacher can be unloaded.

    Only the largest ``top_k`` logits are retained.  This mirrors mlx-lm's
    DWQ target format and avoids materializing full-vocabulary probabilities
    for every calibration token.
    """
    path = Path(target_dir) / split
    path.mkdir(parents=True, exist_ok=True)
    for index, ids in enumerate(inputs):
        logits = mx.stop_gradient(forward(ids).astype(mx.float32), stream=mx.cpu)
        mx.eval(logits)
        k = min(top_k, logits.shape[-1])
        token_ids = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
        values = mx.take_along_axis(logits, token_ids, axis=-1)
        mx.save_safetensors(
            path / f"{index:010d}.safetensors",
            {"logits": values, "indices": token_ids},
        )


def load_teacher_target(
    target_dir: Union[str, Path], index: int, split: str = "train"
) -> Tuple[mx.array, mx.array]:
    target = mx.load(Path(target_dir) / split / f"{index:010d}.safetensors")
    return target["logits"], target["indices"]


def apply_dwq(
    model: nn.Module,
    forward: Callable[[mx.array], mx.array],
    inputs: List[mx.array],
    teacher: Union[List[mx.array], Callable[[int], object]],
    steps: int = 200,
    lr: float = 3e-6,
    temp: float = 1.0,
    gradient_checkpoint: bool = False,
) -> Dict[str, object]:
    """Distill the quantized model's scales/biases toward cached teacher outputs."""
    model.freeze()
    model.unfreeze(recurse=True, keys=["scales", "biases"])
    if gradient_checkpoint:
        try:
            from mlx_lm.tuner.trainer import grad_checkpoint

            layers = getattr(model, "layers", None)
            if not layers:
                raise AttributeError("decoder layers are not exposed")
            grad_checkpoint(layers[0])
        except (ImportError, AttributeError, TypeError) as error:
            raise ValueError(
                "Gradient checkpointing is not supported by this model layout."
            ) from error
    n_trainable = len(tree_flatten(model.trainable_parameters()))
    if n_trainable == 0 or steps <= 0:
        return {"trained_tensors": n_trainable, "final_loss": None, "first_loss": None}

    opt = optim.Adam(learning_rate=lr)

    def loss_fn(ids, target):
        logits = forward(ids).astype(mx.float32) / temp
        if isinstance(target, tuple):
            target, indices = target
            logits = mx.take_along_axis(logits, indices, axis=-1)
            tprob = mx.softmax(target.astype(mx.float32) / temp, axis=-1)
        else:
            tprob = target
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.mean(mx.sum(tprob.astype(mx.float32) * logp, axis=-1))

    step_fn = nn.value_and_grad(model, loss_fn)

    first_loss = None
    final_loss = None
    for step in range(steps):
        idx = step % len(inputs)
        target = teacher(idx) if callable(teacher) else teacher[idx]
        loss, grads = step_fn(inputs[idx], target)
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
