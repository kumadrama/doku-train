"""Frozen numeric split thresholds, independent of any model implementation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

VERSION = "tree-buckets/v1"


def validate_buckets(artifact: dict[str, Any]) -> dict[str, list[float]]:
    if artifact.get("version") != VERSION or artifact.get("comparison") != "float32_right":
        raise ValueError("invalid tree bucket artifact version or comparison")
    features = artifact.get("features")
    if not isinstance(features, dict):
        raise ValueError("tree bucket features must be a mapping")
    result = {}
    for name, cuts in features.items():
        if not isinstance(name, str) or not isinstance(cuts, list):
            raise ValueError("invalid tree bucket feature")
        if any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in cuts):
            raise ValueError("tree thresholds must be finite numbers")
        rounded = [float(np.float32(x)) for x in cuts]
        if any(not math.isfinite(x) for x in rounded) or rounded != sorted(set(rounded)):
            raise ValueError("tree thresholds must be strictly increasing float32 values")
        result[name] = rounded
    return result
