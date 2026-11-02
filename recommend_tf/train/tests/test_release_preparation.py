import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.prepare_tf_release import prepare


def test_release_freezes_binding_and_task_defaults(tmp_path):
    config = tmp_path / "recipe.json"
    config.write_text(
        json.dumps(
            {
                "model_name": "test_model",
                "online_config": {"training_date": 28, "validation_date": 1},
            }
        )
    )
    binding = tmp_path / "base.json"
    binding.write_text(
        json.dumps(
            dict(
                model_name="test_model",
                state_uri="s3://bucket/state",
                training_request={"AlgorithmSpecification": {}},
                deployment={"image_uri": "frozen-inference"},
            )
        )
    )
    outputs = tmp_path / "outputs.json"
    outputs.write_text(
        json.dumps(
            {
                "tf_runtime_binding": {
                    "value": {
                        "state_uri": "s3://bucket/state",
                        "runtime_policy": None,
                        "deferred_uri": "s3://bucket/deferred",
                    }
                }
            }
        )
    )
    image = "123456789012.dkr.ecr.us-east-2.amazonaws.com/train@sha256:" + "a" * 64
    provenance = tmp_path / "training-provenance.json"
    provenance.write_text(
        json.dumps(
            dict(
                image_uri=image,
                source_revision="a" * 40,
                source_sha256="b" * 64,
                lock_sha256="c" * 64,
            )
        )
    )
    args = SimpleNamespace(
        binding=str(binding),
        config=str(config),
        terraform_outputs=str(outputs),
        pipeline_image=image,
        training_image=image,
        training_provenance=str(provenance),
        source_revision="a" * 40,
        config_prefix="s3://bucket/config",
        allow_legacy_ready=True,
        output=str(tmp_path / "release"),
    )
    release = prepare(args)
    result = tmp_path / "release"
    assert (
        hashlib.sha256((result / "binding.json").read_bytes()).hexdigest()
        == release["binding_sha256"]
    )
    assert (result / "daily-config.json").read_bytes() == config.read_bytes()
    task = json.loads((result / "task-definition.yml").read_text())
    env = {
        entry["name"]: entry["value"] for entry in task["containerDefinitions"][0]["environment"]
    }
    assert env["DOKU_BINDING_SHA256"] == release["binding_sha256"]
    assert (
        json.loads((result / "binding.json").read_text())["deployment"]["image_uri"]
        == "frozen-inference"
    )
    exported = json.loads((result / "binding.json").read_text())
    assert exported["training_request"]["Environment"] == {
        "DOKU_SOURCE_SHA256": "b" * 64,
        "DOKU_LOCK_SHA256": "c" * 64,
    }
    args.output = str(tmp_path / "wrong-source")
    args.source_revision = "d" * 40
    with pytest.raises(ValueError, match="provenance"):
        prepare(args)
    assert not (tmp_path / "wrong-source").exists()
    args.source_revision = "a" * 40
    args.output = str(tmp_path / "bad-release")
    args.training_image = "mutable:latest"
    with pytest.raises(ValueError, match="immutable"):
        prepare(args)
    assert not (tmp_path / "bad-release").exists()
