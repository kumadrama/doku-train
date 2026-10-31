"""Convert TensorFlow Serving column outputs into ordered four-task rows."""

import math

from .artifact_loader import TASKS


def format_response(result, version, size):
    outputs = result.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != TASKS:
        raise ValueError("model output tasks differ")
    if any(not isinstance(v, list) or len(v) != size for v in outputs.values()):
        raise ValueError("model output batch differs")
    rows = []
    for index in range(size):
        row = {}
        for name, values in outputs.items():
            value = values[index]
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("nonfinite or nonscalar prediction")
            row[name] = value
        rows.append(row)
    return {"model_version": version, "predictions": rows}
