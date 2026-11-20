import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from recommend_tf.platforms.sagemaker.monitoring_context import attach_context, validate_environment
from recommend_tf.platforms.sagemaker.publish_model import ReleaseState
from recommend_tf.platforms.sagemaker.repair_monitoring_context import repair
from recommend_tf.train.comm.monitoring_context import read_context
from recommend_tf.train.tests.test_daily_publication import S3


def test_file_transport_preserves_large_context_and_checks_integrity(tmp_path):
    context = {"training_job_name": "synthetic", "metric_mapping": {"task" * 150: "auc"}}
    s3 = S3()
    request = {"Environment": {}}
    attach_context(s3, request, context, prefix="s3://bucket/daily", work=tmp_path, kms_key="key")
    validate_environment(request)
    env = dict(
        request["Environment"],
        DOKU_MONITORING_CONTEXT_PATH=str(tmp_path / "monitoring-context.json"),
    )
    assert read_context(env) == context
    assert read_context({"DOKU_MONITORING_CONTEXT": json.dumps(context)}) == context
    assert read_context({}) is None
    with pytest.raises(ValueError, match="ambiguous"):
        read_context(dict(env, DOKU_MONITORING_CONTEXT="{}"))
    with pytest.raises(ValueError, match="incomplete"):
        read_context({"DOKU_MONITORING_CONTEXT_PATH": env["DOKU_MONITORING_CONTEXT_PATH"]})
    (tmp_path / "monitoring-context.json").write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_context(env)
    (tmp_path / "monitoring-context.json").unlink()
    with pytest.raises(FileNotFoundError):
        read_context(env)
    with pytest.raises(ValueError, match="immutable"):
        attach_context(
            s3,
            request,
            {"different": True},
            prefix="s3://bucket/daily",
            work=tmp_path,
            kms_key="key",
        )


@pytest.mark.parametrize("length,valid", [(512, True), (513, False)])
def test_environment_value_boundary(length, valid):
    request = {"Environment": {"synthetic": "x" * length}}
    if valid:
        validate_environment(request)
    else:
        with pytest.raises(ValueError, match="synthetic"):
            validate_environment(request)


def setup_repair():
    s3 = S3()
    binding = {
        "state_uri": "s3://bucket/state",
        "kms_key": "key",
        "ecs_cluster": "cluster",
        "work_prefix": "s3://bucket/work",
    }
    state = ReleaseState(s3, binding["state_uri"], "key")
    state.claim("op", "2026-11-19")
    context = {"training_job_name": "job", "outputs": {"task" * 150: "probability"}}
    request = {
        "TrainingJobName": "job",
        "AlgorithmSpecification": {"TrainingImage": "old@sha256:" + "a" * 64},
        "Environment": {
            "DOKU_MONITORING_CONTEXT": json.dumps(context),
            "DOKU_SOURCE_SHA256": "b" * 64,
            "DOKU_CONFIG_SHA256": hashlib.sha256(b"{}").hexdigest(),
        },
        "InputDataConfig": [
            {
                "ChannelName": "daily",
                "InputMode": "File",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://bucket/work/op/daily/",
                    }
                },
            }
        ],
    }
    state.value["active"].update(
        stage="training", executor="executor", training_request=request, run_uri="s3://bucket/run"
    )
    state.save()
    s3.put_object(Bucket="bucket", Key="work/op/daily/config.json", Body=b"{}")
    ecs = SimpleNamespace(
        describe_tasks=lambda **kwargs: {
            "tasks": [{"taskArn": "executor", "lastStatus": "STOPPED"}]
        }
    )
    sm = SimpleNamespace(describe_training_job=lambda **kwargs: None)
    return s3, sm, ecs, binding, copy.deepcopy(state.value)


def invoke(s3, sm, ecs, binding, tmp_path, **kwargs):
    return repair(
        s3,
        sm,
        ecs,
        binding,
        operation="op",
        image="new@sha256:" + "c" * 64,
        source_sha256="d" * 64,
        work=tmp_path,
        **kwargs,
    )


def test_repair_plan_is_read_only_apply_preserves_frozen_identity(tmp_path):
    s3, sm, ecs, binding, before = setup_repair()
    objects = copy.deepcopy(s3.objects)
    plan = invoke(s3, sm, ecs, binding, tmp_path)
    assert not plan["applied"] and s3.objects == objects
    result = invoke(s3, sm, ecs, binding, tmp_path, apply=True)
    after = ReleaseState(s3, binding["state_uri"], "key").value
    active = after["active"]
    assert (
        active.pop("monitoring_context_repair")["original_request"]
        == before["active"]["training_request"]
    )
    request = active.pop("training_request")
    original = before["active"].pop("training_request")
    assert after == before
    assert result["replacement_request"] == request
    assert request["InputDataConfig"] == original["InputDataConfig"]
    assert (
        request["Environment"]["DOKU_CONFIG_SHA256"]
        == original["Environment"]["DOKU_CONFIG_SHA256"]
    )
    assert json.loads(
        s3.objects["bucket", "work/op/daily/monitoring-context.json"][0]
    ) == json.loads(original["Environment"]["DOKU_MONITORING_CONTEXT"])
    with pytest.raises(ValueError, match="oversized legacy"):
        invoke(s3, sm, ecs, binding, tmp_path, apply=True)


@pytest.mark.parametrize(
    "blocker",
    [
        "existing_job",
        "running_executor",
        "unknown_executor",
        "different_operation",
        "trained",
        "bad_config",
        "access_denied",
    ],
)
def test_repair_rejects_unsafe_state_without_writes(tmp_path, blocker):
    s3, sm, ecs, binding, _ = setup_repair()
    if blocker == "existing_job":
        sm.describe_training_job = lambda **kwargs: {"TrainingJobStatus": "Completed"}
    elif blocker == "running_executor":
        ecs.describe_tasks = lambda **kwargs: {
            "tasks": [{"taskArn": "executor", "lastStatus": "RUNNING"}]
        }
    elif blocker == "unknown_executor":
        ecs.describe_tasks = lambda **kwargs: {"failures": [{"reason": "MISSING"}]}
    elif blocker in {"different_operation", "trained"}:
        state = ReleaseState(s3, binding["state_uri"], "key")
        state.value["active"]["operation" if blocker == "different_operation" else "stage"] = (
            blocker
        )
        state.save()
    elif blocker == "bad_config":
        s3.put_object(Bucket="bucket", Key="work/op/daily/config.json", Body=b"changed")
    else:

        def denied(**kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "DescribeTrainingJob")

        sm.describe_training_job = denied
    objects = copy.deepcopy(s3.objects)
    with pytest.raises((ValueError, ClientError)):
        invoke(s3, sm, ecs, binding, tmp_path, apply=True)
    assert s3.objects == objects


def test_repair_loses_cas_race_without_overwriting_new_executor(tmp_path):
    s3, sm, ecs, binding, _ = setup_repair()
    put = s3.put_object

    def race(**kwargs):
        if kwargs["Key"].endswith("monitoring-context.json"):
            state = ReleaseState(s3, binding["state_uri"], "key")
            state.value["active"]["executor"] = "new-executor"
            state.save()
        return put(**kwargs)

    s3.put_object = race
    with pytest.raises(ClientError, match="PreconditionFailed"):
        invoke(s3, sm, ecs, binding, tmp_path, apply=True)
    active = ReleaseState(s3, binding["state_uri"], "key").value["active"]
    assert active["executor"] == "new-executor"
    assert "DOKU_MONITORING_CONTEXT" in active["training_request"]["Environment"]


def test_repair_requires_upgraded_training_reader(tmp_path):
    s3, sm, ecs, binding, before = setup_repair()
    original = before["active"]["training_request"]
    objects = copy.deepcopy(s3.objects)
    with pytest.raises(ValueError, match="upgraded training image"):
        repair(
            s3,
            sm,
            ecs,
            binding,
            operation="op",
            image=original["AlgorithmSpecification"]["TrainingImage"],
            source_sha256="d" * 64,
            work=tmp_path,
            apply=True,
        )
    assert s3.objects == objects
