"""Configuration identity, independent of model algorithms and cloud runtimes."""

import copy
import hashlib
import json
import re


def model_name(config, *, required=False):
    name = config.get("model_name")
    if name is None and not required:
        return None
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
        raise ValueError("model_name must be a stable snake_case identifier (1..64 characters)")
    return name


def recipe_sha256(config):
    # Preserve unknown fields conservatively: a new algorithm option must change identity.
    # Only these transport/runtime fields are excluded; model and all training options remain.
    recipe = copy.deepcopy(config)
    for key in ("model_name", "prepared_identity"):
        recipe.pop(key, None)
    dataset = recipe.get("dataset_config", {})
    for key in ("dataset", "execution"):
        dataset.pop(key, None)
    split = dataset.get("split", {})
    for key in list(split):
        if key.startswith(("train_", "validation_", "test_")):
            del split[key]
    if not split:
        dataset.pop("split", None)
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
