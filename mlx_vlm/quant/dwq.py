"""In-house DWQ (distilled weight quantization).

Recovers low-bit quality by distilling a quantized student toward the
full-precision teacher: the packed integer weights are frozen and only the
continuous quantization ``scales``/``biases`` are optimized to match the
teacher's output distribution over a calibration set. Pure mlx.
"""

import argparse
import hashlib
import json
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
    metadata: Optional[dict] = None,
) -> None:
    """Persist compact teacher targets so the teacher can be unloaded.

    Only the largest ``top_k`` logits are retained.  This mirrors mlx-lm's
    DWQ target format and avoids materializing full-vocabulary probabilities
    for every calibration token.
    """
    path = Path(target_dir) / split
    path.mkdir(parents=True, exist_ok=True)
    input_hashes = [
        hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest()
        for ids in inputs
    ]
    manifest = {
        "format_version": 1,
        "split": split,
        "samples": len(inputs),
        "top_k": top_k,
        "input_hashes": input_hashes,
        **(metadata or {}),
    }
    (Path(target_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
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


def validate_teacher_targets(target_dir, inputs, expected=None):
    manifest_path = Path(target_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing DWQ target manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    actual_hashes = [
        hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest()
        for ids in inputs
    ]
    required = {"format_version": 1, "samples": len(inputs), "input_hashes": actual_hashes}
    required.update(expected or {})
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in required.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"DWQ target manifest mismatch: {mismatches}")
    return manifest


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


def main():
    """Run the standalone, two-stage DWQ conversion workflow."""
    # Keep orchestration imports lazy: convert imports the quant package.
    from ..convert import convert, fetch_from_hub, skip_multimodal_module
    from ..quant_utils import quantize_model
    from ..utils import get_model_path
    from .calibration import text_calibration_inputs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mlx-path", default="mlx_model")
    parser.add_argument("--target-dir")
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--teacher-bits", type=int, choices=(2, 3, 4, 6, 8))
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 6, 8), default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--grad-checkpoint", action="store_true")
    args = parser.parse_args()

    if args.targets_only and not args.target_dir:
        parser.error("--targets-only requires --target-dir")

    if args.targets_only:
        model_path = get_model_path(args.model)
        model, config, processor = fetch_from_hub(model_path, lazy=True)
        if args.teacher_bits is not None:
            target = (
                model.language_model._model
                if getattr(model, "_is_text_model", False)
                else model
            )
            quantize_model(
                target,
                config,
                group_size=args.group_size,
                bits=args.teacher_bits,
                quant_predicate=lambda path, module: not skip_multimodal_module(path),
            )
        forward = logits_forward(model)
        if forward is None:
            raise RuntimeError("Model does not expose a language-logits forward pass.")
        save_teacher_targets(
            forward,
            text_calibration_inputs(processor, args.max_seq_length),
            Path(args.target_dir),
            metadata={
                "model": args.model,
                "max_seq_length": args.max_seq_length,
                "teacher_bits": args.teacher_bits,
                "group_size": args.group_size,
                "calibration": "built-in-text-v1",
            },
        )
        return

    convert(
        args.model,
        mlx_path=args.mlx_path,
        quantize=True,
        q_bits=args.bits,
        q_group_size=args.group_size,
        quant_method="dwq",
        dwq_steps=args.steps,
        dwq_lr=args.learning_rate,
        dwq_target_dir=args.target_dir,
        dwq_grad_checkpoint=args.grad_checkpoint,
        dwq_max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
