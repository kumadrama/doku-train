"""Repackage an existing SavedModel for AWS's unmodified TF Serving container.

No training, export, upload or deployment is performed by this module.
"""

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

from recommend_tf.inference.artifact_loader import checksum, load
from recommend_tf.inference.request_adapter import prepare

PROTOCOL = "tensorflow-serving/v1"
LEGACY_PROTOCOL = "doku-tf-request/v1"


def model_name(version):
    if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("exact lowercase model digest required")
    return "doku_tf_" + version


def custom_attributes(version):
    return f"tfs-model-name={model_name(version)},tfs-model-version=1,tfs-method=predict"


def protocol(binding):
    value = binding.get("inference_protocol", LEGACY_PROTOCOL)
    if value not in (LEGACY_PROTOCOL, PROTOCOL):
        raise ValueError("unsupported inference protocol")
    return value


def environment(binding, version):
    env = dict(binding.get("container_environment", {}))
    if protocol(binding) == PROTOCOL:
        # Hooks would silently reintroduce a Python handler in the request path.
        if any(
            k.startswith("DOKU_") or k in {"SAGEMAKER_PROGRAM", "SAGEMAKER_SUBMIT_DIRECTORY"}
            for k in env
        ):
            raise ValueError("official serving cannot use custom Python handlers")
        env["SAGEMAKER_TFS_DEFAULT_MODEL_NAME"] = model_name(version)
    else:
        env["DOKU_MODEL_VERSION"] = version
    return env


def package(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    artifact = load(source)
    name = model_name(artifact["version"])
    # Retain the canonical inventory unchanged as provenance, outside the TFS
    # model directory. Its paths describe the source artifact, not this archive.
    entries = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts[0] == "saved_model":
            target = f"{name}/1/" + Path(*relative.parts[1:]).as_posix()
        else:
            target = "doku-contract/" + relative.as_posix()
        entries[target] = path
    if destination.resolve().is_relative_to(source):
        raise ValueError("deployment package must be outside canonical artifact")
    # Exclusive creation protects both immutable training artifacts and retries.
    with destination.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|") as archive:
                for target, path in sorted(entries.items()):
                    info = tarfile.TarInfo(target)
                    info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    return {
        "inference_protocol": PROTOCOL,
        "model_version": artifact["version"],
        "served_model_name": name,
        "served_model_version": 1,
        "source_manifest_sha256": checksum(source / "manifest.json"),
        "deployment_archive_sha256": checksum(destination),
        "custom_attributes": custom_attributes(artifact["version"]),
        "container_environment": environment({"inference_protocol": PROTOCOL}, artifact["version"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe", type=Path, help="optional synthetic legacy request JSON")
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--max-history", type=int, default=10000)
    args = parser.parse_args()
    evidence = package(args.source, args.output)
    if args.probe:
        artifact = load(args.source)
        request, size = prepare(
            json.loads(args.probe.read_text()),
            artifact,
            max_batch=args.max_batch,
            max_history=args.max_history,
        )
        evidence["probe"] = {
            "fixture_kind": "synthetic",
            "content_type": "application/json",
            "request": request,
            "expected_rows": size,
            "tasks": sorted(artifact["signature"]["outputs"]),
        }
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
