import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

class _AWQAttn(nn.Module):
    def __init__(self, d, dk):
        super().__init__()
        self.q_proj = nn.Linear(d, dk, bias=False)
        self.k_proj = nn.Linear(d, dk, bias=False)
        self.v_proj = nn.Linear(d, dk, bias=False)
        self.o_proj = nn.Linear(dk, d, bias=False)


class _AWQMLP(nn.Module):
    def __init__(self, d, di):
        super().__init__()
        self.gate_proj = nn.Linear(d, di, bias=False)
        self.up_proj = nn.Linear(d, di, bias=False)
        self.down_proj = nn.Linear(di, d, bias=False)


class _AWQBlock(nn.Module):
    def __init__(self, d=64, dk=32, di=128, eps=1e-5):
        super().__init__()
        self.self_attn = _AWQAttn(d, dk)
        self.mlp = _AWQMLP(d, di)
        self.input_layernorm = nn.RMSNorm(d, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=eps)

    def __call__(self, x):
        a = self.input_layernorm(x)
        v = self.self_attn.v_proj(a)
        v = (
            v
            + 0.0 * self.self_attn.q_proj(a).sum()
            + 0.0 * self.self_attn.k_proj(a).sum()
        )
        h = x + self.self_attn.o_proj(v)
        b = self.post_attention_layernorm(h)
        return h + self.mlp.down_proj(
            nn.silu(self.mlp.gate_proj(b)) * self.mlp.up_proj(b)
        )


class TestAWQ(unittest.TestCase):
    def test_fold_into_norm_is_identity(self):
        from mlx_vlm.quant.awq import _fold_into_norm

        mx.random.seed(0)
        d = 32
        norm = nn.RMSNorm(d)
        norm.weight = mx.random.uniform(0.5, 1.5, (d,))
        lin = nn.Linear(d, 8, bias=False)
        x = mx.random.normal((4, d))
        y0 = lin(norm(x))
        s = mx.random.uniform(0.3, 3.0, (d,))
        _fold_into_norm(norm, [lin], s)
        self.assertLess(float(mx.max(mx.abs(lin(norm(x)) - y0))), 1e-3)

    def test_fold_into_linear_is_identity(self):
        from mlx_vlm.quant.awq import _fold_into_linear

        mx.random.seed(0)
        up = nn.Linear(16, 24, bias=False)
        down = nn.Linear(24, 16, bias=False)
        x = mx.random.normal((4, 16))
        y0 = down(up(x))
        s = mx.random.uniform(0.3, 3.0, (24,))
        _fold_into_linear(up, [down], s)
        self.assertLess(float(mx.max(mx.abs(down(up(x)) - y0))), 1e-3)

    def test_apply_awq_preserves_output_and_transforms_all_groups(self):
        from mlx_vlm.quant import apply_awq, collect_activation_stats

        mx.random.seed(0)
        block = _AWQBlock()
        mx.eval(block.parameters())
        x = mx.random.normal((2, 6, 64))
        y0 = block(x)
        mx.eval(y0)

        def run():
            for _ in range(3):
                mx.eval(block(mx.random.normal((2, 6, 64))))

        stats = collect_activation_stats(block, run)
        summary = apply_awq(block, stats, bits=4, group_size=32)
        self.assertEqual(summary["groups"], 4)
        self.assertLess(float(mx.max(mx.abs(block(x) - y0))), 2e-3)

    def test_apply_awq_no_stats_is_noop(self):
        from mlx_vlm.quant import apply_awq

        mx.random.seed(0)
        block = _AWQBlock()
        mx.eval(block.parameters())
        x = mx.random.normal((2, 6, 64))
        y0 = block(x)
        summary = apply_awq(block, {}, bits=4, group_size=32)
        self.assertEqual(summary["groups"], 0)
        self.assertLess(float(mx.max(mx.abs(block(x) - y0))), 1e-6)


class _DWQTiny(nn.Module):
    def __init__(self, vocab=128, dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc = nn.Linear(dim, dim, bias=False)
        self.head = nn.Linear(dim, vocab, bias=False)

    def __call__(self, ids):
        return self.head(self.fc(self.embed(ids)))


class TestDWQ(unittest.TestCase):
    def _setup(self):
        mx.random.seed(0)
        model = _DWQTiny()
        mx.eval(model.parameters())
        ids = mx.array([[1, 5, 9, 3, 7, 2]])
        return model, ids, (lambda x: model(x))

    def _quantize(self, model):
        def pred(path, mod):
            return hasattr(mod, "to_quantized") and int(mod.weight.shape[-1]) % 32 == 0

        nn.quantize(model, 32, 4, class_predicate=pred)

    def test_loss_decreases(self):
        from mlx_vlm.quant import apply_dwq, capture_teacher

        model, ids, fwd = self._setup()
        teacher = capture_teacher(fwd, [ids])
        self._quantize(model)
        summary = apply_dwq(model, fwd, [ids], teacher, steps=100, lr=1e-3)
        self.assertIsNotNone(summary["final_loss"])
        self.assertLess(summary["final_loss"], summary["first_loss"])

    def test_only_scales_biases_change(self):
        from mlx_vlm.quant import apply_dwq, capture_teacher

        model, ids, fwd = self._setup()
        teacher = capture_teacher(fwd, [ids])
        self._quantize(model)
        weight_before = mx.array(model.head.weight)
        scales_before = mx.array(model.head.scales)
        apply_dwq(model, fwd, [ids], teacher, steps=40, lr=1e-3)
        self.assertTrue(mx.array_equal(model.head.weight, weight_before))
        self.assertFalse(mx.array_equal(model.head.scales, scales_before))

    def test_shapes_preserved(self):
        from mlx_vlm.quant import apply_dwq, capture_teacher

        model, ids, fwd = self._setup()
        teacher = capture_teacher(fwd, [ids])
        self._quantize(model)
        apply_dwq(model, fwd, [ids], teacher, steps=10, lr=1e-3)
        self.assertEqual(fwd(ids).shape, (1, 6, 128))

    def test_zero_steps_is_noop(self):
        from mlx_vlm.quant import apply_dwq, capture_teacher

        model, ids, fwd = self._setup()
        teacher = capture_teacher(fwd, [ids])
        self._quantize(model)
        scales_before = mx.array(model.head.scales)
        summary = apply_dwq(model, fwd, [ids], teacher, steps=0, lr=1e-3)
        self.assertIsNone(summary["final_loss"])
        self.assertTrue(mx.array_equal(model.head.scales, scales_before))

    def test_target_manifest_rejects_incompatible_inputs(self):
        from mlx_vlm.quant.dwq import (
            save_teacher_targets,
            validate_teacher_targets,
        )

        model, ids, fwd = self._setup()
        with tempfile.TemporaryDirectory() as directory:
            save_teacher_targets(
                fwd,
                [ids],
                Path(directory),
                top_k=16,
                metadata={"max_seq_length": 6, "calibration": "test"},
            )
            manifest = validate_teacher_targets(
                directory,
                [ids],
                {"max_seq_length": 6, "calibration": "test"},
            )
            self.assertEqual(manifest["top_k"], 16)
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                validate_teacher_targets(directory, [ids[:, :-1]])
