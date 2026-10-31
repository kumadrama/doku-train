"""Prepare immutable release files locally. No upload, registration or deployment."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(args):
    binding = json.loads(Path(args.binding).read_text())
    applied = json.loads(Path(args.terraform_outputs).read_text())["tf_runtime_binding"]["value"]
    if binding["state_uri"] != applied["state_uri"]:
        raise ValueError("applied infrastructure differs from existing model state")
    for image in (args.pipeline_image, args.training_image):
        if not re.fullmatch(
            r"[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[^@]+@sha256:[a-f0-9]{64}", image
        ):
            raise ValueError("immutable ECR image digest required")
    config_raw = Path(args.config).read_bytes()
    config = json.loads(config_raw)
    if config.get("model_name") != binding["model_name"]:
        raise ValueError("recipe/model binding mismatch")
    binding.update(applied)
    binding["training_request"]["AlgorithmSpecification"]["TrainingImage"] = args.training_image
    if args.allow_legacy_ready:
        binding["legacy_ready_config_sha256"] = digest(config_raw)
    binding_raw = (json.dumps(binding, sort_keys=True, indent=2) + "\n").encode()
    prefix = args.config_prefix.rstrip("/")
    if not prefix.startswith("s3://"):
        raise ValueError("explicit configuration object prefix required")
    release = dict(
        schema_version="doku-tf-runtime-release/v1",
        source_revision=args.source_revision,
        pipeline_image=args.pipeline_image,
        training_image=args.training_image,
        inference_image=binding["deployment"]["image_uri"],
        config_uri=prefix + "/" + digest(config_raw) + "/daily-config.json",
        config_sha256=digest(config_raw),
        binding_uri=prefix + "/" + digest(binding_raw) + "/binding.json",
        binding_sha256=digest(binding_raw),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "binding.json").write_bytes(binding_raw)
    (output / "daily-config.json").write_bytes(config_raw)
    (output / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    task = json.loads(
        (
            Path(__file__).resolve().parents[1] / "tf-pipeline/deploy/prod/task-definition.yml"
        ).read_text()
    )
    container = task["containerDefinitions"][0]
    for entry in container["environment"]:
        key = entry["name"].removeprefix("DOKU_").lower()
        if key in release:
            entry["value"] = release[key]
    (output / "task-definition.yml").write_text(json.dumps(task, indent=2) + "\n")
    return release


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "binding",
        "terraform-outputs",
        "config",
        "config-prefix",
        "pipeline-image",
        "training-image",
        "source-revision",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--allow-legacy-ready", action="store_true")
    prepare(parser.parse_args())
