"""Public quantization API, loaded lazily to keep CLI modules executable."""

_EXPORTS = {
    "apply_awq": (".awq", "apply_awq"),
    "apply_dwq": (".dwq", "apply_dwq"),
    "capture_teacher": (".dwq", "capture_teacher"),
    "load_teacher_target": (".dwq", "load_teacher_target"),
    "logits_forward": (".dwq", "logits_forward"),
    "save_teacher_targets": (".dwq", "save_teacher_targets"),
    "validate_teacher_targets": (".dwq", "validate_teacher_targets"),
    "collect_activation_stats": (".calibration", "collect_activation_stats"),
    "DEFAULT_CALIBRATION_TEXT": (".calibration", "DEFAULT_CALIBRATION_TEXT"),
    "text_calibration_inputs": (".calibration", "text_calibration_inputs"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
