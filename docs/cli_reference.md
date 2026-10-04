# CLI Reference

MLX-VLM provides several command line entry points:

Recommended invocation style:

```bash
python -m mlx_vlm <subcommand> ...
```

For example:

```bash
python -m mlx_vlm convert --help
python -m mlx_vlm generate --help
```

- `mlx_vlm.convert` – convert Hugging Face models to MLX format.
- `mlx_vlm.dwq` – cache teacher targets and run distilled weight quantization.
- `mlx_vlm.generate` – run inference on images, audio, or video.
- `mlx_vlm.chat_ui` – start an interactive Gradio UI.
- `mlx_vlm.server` – run the FastAPI server.

Each command accepts `--help` for full usage information.

## Weight quantization

`mlx_vlm.convert` supports round-to-nearest (`rtn`), activation-aware (`awq`),
distilled (`dwq`), and combined (`awq+dwq`) weight quantization. For example:

```bash
mlx_vlm.convert \
  --hf-path MODEL \
  --mlx-path OUTPUT \
  --quantize --q-bits 4 --q-group-size 64 \
  --quant-method awq
```

DWQ can run directly during conversion, or as a two-stage workflow that caches
the teacher logits before loading and optimizing the quantized student:

```bash
mlx_vlm.dwq \
  --model MODEL \
  --target-dir TARGETS \
  --targets-only

mlx_vlm.dwq \
  --model MODEL \
  --target-dir TARGETS \
  --mlx-path OUTPUT
```

The target cache contains a manifest and is validated against the calibration
inputs and conversion settings before training starts. Use
`mlx_vlm.convert --help` and `mlx_vlm.dwq --help` for tuning options.
