"""One-shot container adapter for pinned config and explicit infrastructure bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from recommend.train.comm.artifact_store import ObjectStore
from recommend.train.comm.model_params_validator import load_config


def read_config(uri: str, sha256: str, model_name: str) -> dict[str, Any]:
    parent, key = uri.rsplit("/", 1)
    body = ObjectStore(parent, max_bytes=8 * 1024 * 1024).get(key)[0]
    if hashlib.sha256(body).hexdigest() != sha256:
        raise ValueError("scheduled config digest mismatch")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_bytes(body)
        config = load_config(path)
    if config["model_name"] != model_name or config["model_type"] != "rerank_tree":
        raise ValueError("scheduled model identity mismatch")
    return config


def execute() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["daily", "cleanup"])
    parser.add_argument("--cutoff")
    args = parser.parse_args()
    root = os.environ["DOKU_TRAIN_ARTIFACT_ROOT"]
    name = os.environ["DOKU_TRAIN_MODEL_NAME"]
    # Fail before any writes if the explicit lake encryption binding is missing.
    if not os.environ["DOKU_TRAIN_S3_KMS_KEY_ARN"]:
        raise ValueError("KMS key binding is empty")
    if args.operation == "daily":
        if not args.cutoff:
            parser.error("daily requires --cutoff")
        from recommend.train.run import daily_run

        config = read_config(
            os.environ["DOKU_TRAIN_CONFIG_URI"],
            os.environ["DOKU_TRAIN_CONFIG_SHA256"],
            name,
        )
        from recommend.train.publish_model import distribution_from_environment

        result = daily_run(
            config,
            daily_root=os.environ["DOKU_TRAIN_DAILY_ROOT"],
            frozen_root=os.environ["DOKU_TRAIN_FROZEN_ROOT"],
            cutoff=args.cutoff,
            artifact_root=root,
            export_root=Path("/work/exports"),
            distribution=distribution_from_environment(root, name),
        )
    else:
        from recommend.train.comm.publication import Registry
        from recommend.train.publish_model import distribution_from_environment

        distribution = distribution_from_environment(root, name)
        result = (
            distribution.cleanup(apply=True)
            if distribution
            else Registry(ObjectStore(root), name).cleanup(apply=True)
        )
    print(json.dumps({"operation": args.operation, "completed": True, "result": result}))


def main() -> None:
    try:
        execute()
    except Exception:
        import sys

        operation = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        print(json.dumps({"operation": operation, "completed": False}), flush=True)
        raise


if __name__ == "__main__":
    main()
