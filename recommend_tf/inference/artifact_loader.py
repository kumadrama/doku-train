"""Validate the immutable inference artifact before exposing its model."""

import hashlib
import json
from pathlib import Path

TASKS = {"completion", "fast_swipe", "immersive_click", "watch_seconds"}


def checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(root):
    root = Path(root).resolve()
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("model symlinks forbidden")
        if path.is_file() and path != root / "manifest.json":
            actual[path.relative_to(root).as_posix()] = checksum(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("version") != "doku-tf-inference/v1" or manifest["files"] != actual:
        raise ValueError("invalid artifact inventory")
    signature = json.loads((root / "signature.json").read_text())
    if (
        signature.get("version") != "doku-tf-signature/v1"
        or signature.get("model_version") != manifest["model_version"]
        or signature.get("input_stage") != "organized_padded_features"
        or signature.get("response_order") != "request_order"
        or set(signature["outputs"]) != TASKS
        or not signature["inputs"]
        or not (root / "saved_model/saved_model.pb").is_file()
    ):
        raise ValueError("unsupported model signature")
    config = json.loads((root / "feature-config.json").read_text())
    return {
        "root": root,
        "version": manifest["model_version"],
        "signature": signature,
        "config": config,
    }
