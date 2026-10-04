# Local DWQ smoke report

No checkpoints were downloaded and the source checkpoints were not modified.
Derived files and cached targets are in `/tmp/dwq-*`.

## Configuration

- Student quantization: affine, 4 bits, group size 64.
- Calibration: built-in text corpus, sequence length 32, batch size 1.
- Distillation: one step. These runs verify plumbing and reloadability, not
  converged model quality.
- Cached targets: top 1024 teacher logits per token.

## Results

| Model | Source | Derived output | Effective BPW | First loss | Original / derived output | Peak memory original / derived |
|---|---|---|---:|---:|---|---:|
| Qwen3-0.6B BF16 | HF cache `42096995...` | `/tmp/dwq-qwen3-student` | 4.501 | 3.1919 | `The capital of France is **Paris**` / same | 1.249 / 0.419 GB |
| LFM2.5-VL-450M BF16 | HF cache `ed71acda...` | `/tmp/dwq-lfm25-student` | 6.917 | 1.6080 | `The capital of France is Paris.` / same | 0.915 / 0.422 GB |
| Granite-4.0-H-350M BF16 | HF cache `66fed81b...` | `/tmp/dwq-granite-student` | 4.512 | 2.8608 | `The capital of France is Paris.` / same | 0.756 / 0.270 GB |

Observed one-step conversion command wall times were approximately 1.71 s,
1.42 s, and 1.47 s respectively. They include process startup and saving.
Generation throughput is intentionally not treated as a benchmark because each
case was a single short run.

An 8-bit-teacher target-cache run succeeded for Qwen3 (reported effective
teacher BPW 8.501), as did the gradient-checkpointed one-step student path.

Focused quantization regression result: `9 passed`.

## Reproduction

Use the existing project environment and local source tree:

```bash
export PYTHONPATH="$PWD"
PY=.venv/bin/python

$PY -m mlx_vlm.quant.dwq --model MODEL_PATH --target-dir TARGET_DIR \
  --targets-only --max-seq-length 32

$PY -m mlx_vlm.quant.dwq --model MODEL_PATH --target-dir TARGET_DIR \
  --mlx-path OUTPUT_DIR --max-seq-length 32 --steps 1

$PY -m mlx_vlm.generate --model OUTPUT_DIR \
  --prompt 'The capital of France is' --max-tokens 8 --temp 0

$PY -m pytest mlx_vlm/tests/test_quant.py -q
```

The first-loss values are single-step observations. They cannot establish that
DWQ improves final perplexity or task quality; a longer train/validation run is
required for that conclusion.
