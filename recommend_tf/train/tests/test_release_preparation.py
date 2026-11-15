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


def test_family_enrollment_keeps_legacy_operation_and_recipe(tmp_path):
    from scripts.prepare_tf_daily_deployment import prepare_daily

    from recommend_tf.platforms.sagemaker.daily_pipeline import operation_identity
    from recommend_tf.train.comm.model_identity import recipe_sha256

    outputs = {
        "tf_daily_deployment": {
            "value": {
                "schema_version": "doku-tf-daily-deployment/v1",
                "environment": "prod",
                "region": "region",
                "account_id": "123",
                "families": {
                    "family": {
                        "role": "worker",
                        "domain": "Idea",
                        "sources": [
                            {"kind": "daily_runs", "uri": "s3://bucket/runs/job-"},
                            {"kind": "journals", "uri": "s3://bucket/control"},
                        ],
                    }
                },
                "monitoring": {
                    "catalog_uri": "s3://bucket/catalog",
                    "outbox_uri": "s3://bucket/outbox",
                    "kms_key": "key",
                },
            }
        }
    }
    task = {
        "family": "family",
        "taskRoleArn": "worker",
        "containerDefinitions": [
            {
                "environment": [
                    {"name": "DOKU_" + k, "value": "previous"}
                    for k in ("CONFIG_SHA256", "CONFIG_URI", "BINDING_SHA256", "BINDING_URI")
                ]
            }
        ],
    }
    config = {"model_name": "idea_model"}
    original = {
        "environment": "prod",
        "region": "region",
        "work_prefix": "s3://bucket/work",
        "runs_prefix": "s3://bucket/runs",
        "job_name_prefix": "job-",
        "state_uri": "s3://bucket/control/state",
    }
    args = SimpleNamespace(config_prefix="s3://bucket/releases", output=tmp_path / "result")
    for name, value in dict(
        task_definition=task, monitoring_outputs=outputs, binding=original, config=config
    ).items():
        path = tmp_path / name
        path.write_text(json.dumps(value))
        setattr(args, name, path)
    prepare_daily(args)
    bound = json.loads((args.output / "binding.json").read_bytes())
    assert bound["monitoring"]["model_name"] == config["model_name"]
    assert bound["monitoring"]["task_family"] == "family"
    assert operation_identity(config, {}, original) == operation_identity(config, {}, bound)
    frozen = args.output / "daily-config.json"
    assert frozen.read_bytes() == args.config.read_bytes()
    assert recipe_sha256(json.loads(frozen.read_bytes())) == recipe_sha256(config)
    task["taskRoleArn"] = "another-role"
    args.task_definition.write_text(json.dumps(task))
    with pytest.raises(ValueError, match="role"):
        prepare_daily(args)
