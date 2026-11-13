"""Convert TensorFlow Serving column outputs into signature-declared prediction rows."""

import math


def format_response(result, version, size, tasks):
    outputs = result.get("outputs")
    if isinstance(outputs, list) and len(tasks) == 1:
        outputs = {next(iter(tasks)): outputs}
    if not tasks or not isinstance(outputs, dict) or set(outputs) != set(tasks):
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
