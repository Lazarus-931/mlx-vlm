"""Shared weight-sanitization helpers for the Gemma 4 family.

Used by ``gemma4`` and ``gemma4_unified``. Every converter here is
idempotent: since always-on sanitization (#1498), ``load_model`` runs
``Model.sanitize`` on MLX-format checkpoints too, so converters must leave
already-converted weights unchanged.
"""


def remap_gemma4_key(key: str) -> str:
    """Strip the HF ``model.`` prefix and normalize ``language_model.*`` keys."""
    if key.startswith("model."):
        key = key[len("model.") :]
    if key.startswith("language_model.") and not key.startswith(
        "language_model.model."
    ):
        key = "language_model.model." + key[len("language_model.") :]
    return key


def convert_conv2d_weight(v):
    """PyTorch ``[out, in, kH, kW]`` -> MLX ``[out, kH, kW, in]``, idempotent.

    A weight already in MLX layout carries its (equal) spatial dims on the
    middle axes; PyTorch layout carries them on the trailing axes. The Gemma 4
    audio subsample convs have in-channels of 1 or 128 against 3x3 kernels, so
    the two layouts are always distinguishable by shape.
    """
    if v.ndim == 4 and not (v.shape[1] == v.shape[2] and v.shape[0] >= v.shape[1]):
        return v.transpose(0, 2, 3, 1)
    return v


def convert_conv1d_weight(v):
    """PyTorch ``[out, in, kW]`` -> MLX ``[out, kW, in]``, idempotent.

    Only used for depthwise weights, where the per-group in-channel dim is 1:
    PyTorch layout is ``[C, 1, k]`` and MLX layout is ``[C, k, 1]``.
    """
    if v.ndim == 3 and v.shape[-1] != 1:
        return v.transpose(0, 2, 1)
    return v


def split_moe_gate_up(new_key, v, sanitized):
    """Map fused ``.experts.*`` tensors onto ``switch_glu`` weights.

    Returns True when the key was consumed (already written to ``sanitized``).
    Renamed ``switch_glu`` keys never re-match, so re-sanitization is a no-op.
    """
    if new_key.endswith(".experts.down_proj"):
        sanitized[
            new_key.replace(
                ".experts.down_proj", ".experts.switch_glu.down_proj.weight"
            )
        ] = v
        return True
    if new_key.endswith(".experts.gate_up_proj"):
        gate_key = new_key.replace(
            ".experts.gate_up_proj", ".experts.switch_glu.gate_proj.weight"
        )
        up_key = new_key.replace(
            ".experts.gate_up_proj", ".experts.switch_glu.up_proj.weight"
        )
        v = v.swapaxes(-1, -2)
        mid_dim = v.shape[-1] // 2
        sanitized[gate_key] = v[..., :mid_dim].swapaxes(-1, -2)
        sanitized[up_key] = v[..., mid_dim:].swapaxes(-1, -2)
        return True
    return False
