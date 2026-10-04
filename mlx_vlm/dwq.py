"""Standalone distilled-weight-quantization workflow for MLX-VLM."""

import argparse
from pathlib import Path

import mlx.core as mx

from .convert import convert, fetch_from_hub, skip_multimodal_module
from .quant import (
    DEFAULT_CALIBRATION_TEXT,
    logits_forward,
    save_teacher_targets,
)
from .utils import get_model_path
from .quant_utils import quantize_model


def _text_inputs(processor, max_seq_length):
    tokenizer = getattr(processor, "tokenizer", processor)
    return [
        mx.array([tokenizer.encode(text)[:max_seq_length]])
        for text in DEFAULT_CALIBRATION_TEXT
    ]


def main():
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument(
        "--calibration", choices=("text", "multimodal"), default="text"
    )
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("The built-in calibration corpus currently uses batch size 1.")
    if args.calibration == "multimodal":
        raise ValueError(
            "Multimodal DWQ requires an explicit calibration dataset; the default "
            "corpus is text-only."
        )
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
            _text_inputs(processor, args.max_seq_length),
            Path(args.target_dir),
        )
        return

    # Conversion remains the owner of model copying, quantization predicates,
    # processor/config persistence, and multimodal tower exclusion.
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
