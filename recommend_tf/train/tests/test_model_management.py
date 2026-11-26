"""Behavioral tests for configuration revisions, admission races and unchanged serving."""

import copy
import io
import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from recommend_tf.platforms.model_management.contracts import (
    GROUP,
    REGISTRY,
    binding_for,
    digest,
    validate_group,
    validate_registry,
)
from recommend_tf.platforms.model_management.dispatch import launch
from recommend_tf.platforms.model_management.publication import ManagedPublication
from recommend_tf.platforms.model_management.registry import Registry
from recommend_tf.platforms.model_management.requests import cancel, enqueue, validate_request
from recommend_tf.platforms.model_management.storage import Documents
from recommend_tf.platforms.model_management.worker import run
from recommend_tf.platforms.sagemaker.publish_model import ReleaseState
from recommend_tf.train.tests.test_daily_publication import deployment_fixture, fixture


class S3:
    def __init__(self):
        self.objects = {}
        self.counter = 0

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body, etag = self.objects[Bucket, Key]
        return {"Body": io.BytesIO(body), "ETag": etag, "ContentLength": len(body)}

    def put_object(self, **args):
        key = args["Bucket"], args["Key"]
        old = self.objects.get(key)
        if (args.get("IfNoneMatch") and old) or (
            args.get("IfMatch") and (not old or old[1] != args["IfMatch"])
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        body = args["Body"]
        if hasattr(body, "read"):
            body = body.read()
        self.counter += 1
        self.objects[key] = body, str(self.counter)
        return {"ETag": str(self.counter)}

    def get_paginator(self, operation):
        def paginate(Bucket, Prefix):
            yield {
                "Contents": [
                    {"Key": k}
                    for b, k in sorted(self.objects)
                    if b == Bucket and k.startswith(Prefix)
                ]
            }

        return SimpleNamespace(paginate=paginate)


@pytest.fixture
def setup(tmp_path):
    config, ready = fixture(tmp_path)
    outputs = {
        "completion": "probability",
        "fast_swipe": "probability",
        "immersive_click": "probability",
        "watch_seconds": "seconds",
    }
    group = {
        "schema_version": GROUP,
        "environment": "dte",
        "region": "us-east-2",
        "name": "fyp",
        "registry": {
            "application_id": "app",
            "environment_id": "env",
            "profile_id": "registry",
            "deployment_strategy_id": "strategy",
        },
        "storage_prefix": "s3://bucket/manage/fyp",
        "kms_key": "synthetic",
        "endpoint_prefix": "doku-dte-fyp-",
        "worker": {
            "cluster": "doku-dte",
            "task_definition": "arn:aws:ecs:us-east-2:123456789012:task-definition/daily:1",
            "container": "pipeline",
            "network_configuration": {"awsvpcConfiguration": {"subnets": ["synthetic"]}},
        },
        "data_source": {"kind": "ready_manifest", "bucket": "bucket", "daily_prefix": "data"},
        "read_prefixes": ["s3://bucket/runs"],
        "binding_template": {
            "environment": "dte",
            "region": "us-east-2",
            "ecs_cluster": "doku-dte",
            "state_uri": "s3://bucket/state/${model_name}.json",
            "work_prefix": "s3://bucket/work/${model_name}",
            "model_prefix": "s3://bucket/models/${model_name}",
            "kms_key": "synthetic",
            "deployment": {**deployment_fixture()[2], "endpoint_name": "${endpoint_name}"},
            "poll_seconds": 5,
            "max_wait_seconds": 600,
            "download_limit_bytes": 100_000_000,
            "export_timeout_seconds": 60,
            "training_request": {
                "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.c5.xlarge"},
                "AlgorithmSpecification": {"TrainingImage": "image@sha256:" + "1" * 64},
            },
        },
    }
    model = {
        "group": "fyp",
        "config": config,
        "config_revision": digest(config),
        "service": {
            "outputs": outputs,
            "probe": {"fixture_kind": "synthetic", "tasks": sorted(outputs)},
        },
        "daily_enabled": True,
        "admission": "managed",
        "first_cutoff": ready["cutoff"],
    }
    registry = {
        "schema_version": REGISTRY,
        "environment": "dte",
        "group": "fyp",
        "revision": 1,
        "models": {config["model_name"]: model},
    }
    validate_group(group)
    validate_registry(registry, group)
    s3 = S3()
    return Documents(s3, group), group, registry, config["model_name"], ready


def queued(setup):
    docs, group, registry, name, ready = setup
    record = enqueue(
        docs, group, registry, name, cutoff=ready["cutoff"], ready_builder=lambda *a: ready
    )
    request = docs.pinned(record["request"])
    return record, request


def test_old_request_survives_config_change_and_repeated_trigger(setup):
    docs, group, registry, name, ready = setup
    original, request = queued(setup)
    registry = copy.deepcopy(registry)
    registry["revision"] += 1
    model = registry["models"][name]
    model["config"]["max_epochs"] = 15
    model["config_revision"] = digest(model["config"])
    repeated = enqueue(
        docs,
        group,
        registry,
        name,
        cutoff=ready["cutoff"],
        ready_builder=lambda *a: pytest.fail("must not reread data"),
    )
    assert repeated == original
    _, values = validate_request(docs, group, original["request"])
    assert values["config"]["max_epochs"] == 1
    later = enqueue(
        docs,
        group,
        registry,
        name,
        cutoff="2026-10-08",
        ready_builder=lambda *a: {**ready, "cutoff": "2026-10-08"},
    )
    assert validate_request(docs, group, later["request"])[1]["config"]["max_epochs"] == 15
    assert request["config_revision"] != model["config_revision"]


class ECS:
    def __init__(self):
        self.calls = []
        self.lost_response = False
        self.arn = "arn:aws:ecs:us-east-2:123456789012:task/doku-dte/synthetic"

    def run_task(self, **kwargs):
        self.calls.append(kwargs)
        if self.lost_response:
            self.lost_response = False
            raise TimeoutError("lost response")
        return {"tasks": [{"taskArn": self.arn}], "failures": []}

    def list_tasks(self, **kwargs):
        return {"taskArns": []}

    def describe_tasks(self, **kwargs):
        return {"tasks": [{"taskArn": self.arn, "lastStatus": "STOPPED"}], "failures": []}


def test_cancel_and_launch_are_exclusive(setup):
    docs, group, _, _, _ = setup
    original, request = queued(setup)
    cancel(docs, request["slot_uri"], original["request"])
    ecs = ECS()
    assert launch(docs, group, ecs, request["slot_uri"])["status"] == "cancelled"
    assert not ecs.calls


def test_lost_ecs_response_reuses_token_and_expired_token_fails_closed(setup):
    docs, group, _, _, _ = setup
    _, request = queued(setup)
    ecs = ECS()
    ecs.lost_response = True
    with pytest.raises(TimeoutError):
        launch(docs, group, ecs, request["slot_uri"], now=lambda: 100)
    outcome = launch(docs, group, ecs, request["slot_uri"], now=lambda: 4000)
    assert "unknown" in outcome["waiting"] and len(ecs.calls) == 1
    outcome = launch(docs, group, ecs, request["slot_uri"], now=lambda: 101)
    assert outcome["status"] == "submitted"
    assert ecs.calls[0] == ecs.calls[1]
    with pytest.raises(ValueError, match="unstarted"):
        cancel(docs, request["slot_uri"], outcome["request"])


def test_worker_uses_exact_frozen_inputs_and_marks_failure(setup):
    docs, group, _, _, _ = setup
    record, request = queued(setup)
    ecs = ECS()
    launch(docs, group, ecs, request["slot_uri"])

    def execute(args):
        assert json.load(open(args.config))["max_epochs"] == 1
        assert "managed_model" in json.load(open(args.binding))
        raise RuntimeError("synthetic training failure")

    with pytest.raises(RuntimeError):
        run(
            docs, group, record["request"], identity=(ecs.arn, "doku-dte", "daily"), execute=execute
        )
    assert docs.read(request["slot_uri"])[0]["status"] == "failed"


def test_registry_plan_is_stale_safe_and_changes_only_recipe(setup):
    docs, group, registry, name, _ = setup
    store = Registry(docs, group, None, None)
    store.read = lambda: registry
    config = copy.deepcopy(registry["models"][name]["config"])
    config["max_epochs"] = 15
    plan = store.plan("configure", name, {"config": config}, digest(registry))
    intent = docs.pinned(plan["plan"])
    assert intent["before"] == registry
    assert intent["after"]["models"][name]["service"] == registry["models"][name]["service"]
    assert {v["path"] for v in plan["changes"]} == {"/config/max_epochs", "/config_revision"}
    with pytest.raises(ValueError, match="changed"):
        store.plan("configure", name, {"config": config}, "0" * 64)
    registry["revision"] += 1
    with pytest.raises(ValueError, match="stale"):
        store.apply(plan["plan"])


def test_registered_identity_cannot_inject_runtime_or_consumer_route(setup):
    docs, group, registry, name, _ = setup
    bad = copy.deepcopy(registry)
    bad["models"][name]["service"]["role_arn"] = "other"
    with pytest.raises(ValueError):
        validate_registry(bad, group)
    group["binding_template"]["idea_appconfig"] = {}
    with pytest.raises(ValueError, match="consumer"):
        validate_group(group)


def test_contract_change_rejected_before_endpoint_mutation(setup, tmp_path):
    docs, group, registry, name, _ = setup
    binding = binding_for(group, name, registry["models"][name])
    state = ReleaseState(docs.s3, binding["state_uri"], group["kms_key"])
    state.claim("one", "2026-10-07")
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "signature.json").write_text(
        json.dumps(
            {"model_version": "1" * 64, "inputs": {"id": "int64"}, "outputs": {"score": "float32"}}
        )
    )
    (root / "feature-config.json").write_text('{"features":["id"]}')
    guard = ManagedPublication(docs.s3, binding)
    guard.prepare(state, root, tmp_path)
    # Weight identity changes are compatible; deleting a required input is not.
    sig = json.loads((root / "signature.json").read_text())
    sig["model_version"] = "2" * 64
    (root / "signature.json").write_text(json.dumps(sig))
    guard.prepare(state, root, tmp_path)
    sig["inputs"] = {}
    (root / "signature.json").write_text(json.dumps(sig))
    with pytest.raises(ValueError, match="incompatible"):
        guard.prepare(state, root, tmp_path)
    assert state.value["active"]["stage"] == "export_contract_failed"


def test_appconfig_lost_response_resumes_same_publication(setup):
    from recommend_tf.inference.tests.test_idea_publication import AppConfig

    docs, group, registry, name, _ = setup
    # Existing bootstrap can have noncanonical JSON formatting.
    api = AppConfig({"registry": json.dumps(registry, indent=2).encode()})
    store = Registry(docs, group, api, api)
    config = copy.deepcopy(registry["models"][name]["config"])
    config["max_epochs"] = 15
    plan = store.plan("configure", name, {"config": config}, digest(registry))
    api.lose_create = True
    with pytest.raises(Exception):
        store.apply(plan["plan"])
    for _ in range(5):
        result = store.apply(plan["plan"])
        if result["status"] == "complete":
            break
    assert result["status"] == "complete"
    assert api.starts == 1
    assert store.read()["models"][name]["config"]["max_epochs"] == 15
    assert store.apply(plan["plan"]) == result


def test_model_serializes_two_queued_dates_before_worker_claim(setup):
    docs, group, registry, name, ready = setup
    _, first = queued(setup)
    later = enqueue(
        docs,
        group,
        registry,
        name,
        cutoff="2026-10-08",
        ready_builder=lambda *a: {**ready, "cutoff": "2026-10-08"},
    )
    second = docs.pinned(later["request"])
    ecs = ECS()
    launch(docs, group, ecs, first["slot_uri"])
    outcome = launch(docs, group, ecs, second["slot_uri"])
    assert outcome["waiting"] == "previous_model_request_not_finished"
    assert len(ecs.calls) == 1


def test_idea_admission_freezes_mapped_pointers(setup):
    from recommend_tf.platforms.model_management.requests import ready_for

    docs, group, registry, name, _ = setup
    group["data_source"] = {"kind": "idea_mapped", "uri": "s3://bucket/mapped"}
    config = registry["models"][name]["config"]
    config["online_config"] = {"training_date": 1, "validation_date": 1}
    for day in ["2026-10-06", "2026-10-07"]:
        docs.put(
            "s3://bucket/mapped/date=" + day + "/_SUCCESS",
            {
                "manifest_uri": "s3://bucket/mapped/date=" + day + "/run_id=one/manifest.json",
                "manifest_sha256": "1" * 64,
            },
        )
    ready = ready_for(docs.s3, group, registry["models"][name], "2026-10-07")
    assert ready["version"] == "idea-mapped-ready/v1"
    assert [x["date"] for x in ready["sources"]] == ["2026-10-06", "2026-10-07"]


def test_incomplete_handover_cannot_admit_model(setup):
    docs, group, registry, name, ready = setup
    registry["models"][name]["adoption"] = {
        "legacy_rule": "old",
        "binding": {},
        "evidence_uri": docs.uri("evidence"),
    }
    with pytest.raises(ValueError, match="handover incomplete"):
        enqueue(docs, group, registry, name, cutoff=ready["cutoff"])


def test_controller_imports_without_training_dependencies(tmp_path):
    import subprocess
    import sys

    code = "import sys; from recommend_tf.platforms.model_management.api import handle; assert not {'tensorflow','pyarrow','numpy'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


def test_managed_export_uses_real_cpu_training_and_identity_signature(setup, tmp_path):
    from recommend_tf.platforms.sagemaker.daily_pipeline import step
    from recommend_tf.train.comm.datasvr.daily_dataset import freeze
    from recommend_tf.train.daily_train_model import execute as train
    from recommend_tf.train.tests.test_daily_publication import deployment_fixture

    docs, group, registry, name, ready = setup
    config = registry["models"][name]["config"]
    frozen = tmp_path / "frozen"
    resolved = freeze(config, ready, frozen)
    run_root = tmp_path / "trained"
    train(resolved, run_root, cutoff=ready["cutoff"], frozen_input=frozen / "input")
    evidence = json.loads((run_root / "COMMITTED.json").read_text())
    for filename in [*evidence["files"], "COMMITTED.json"]:
        docs.s3.put_object(
            Bucket="bucket", Key="runs/one/" + filename, Body=(run_root / filename).read_bytes()
        )
    _, sm, deployment = deployment_fixture()
    sm.endpoint = None
    sm.create_endpoint = lambda **kwargs: setattr(
        sm,
        "endpoint",
        {"EndpointStatus": "Creating", "EndpointConfigName": kwargs["EndpointConfigName"]},
    )
    binding = {
        "download_limit_bytes": 100_000_000,
        "export_timeout_seconds": 60,
        "outputs": registry["models"][name]["service"]["outputs"],
        "model_prefix": "s3://bucket/models",
        "kms_key": "synthetic",
        "deployment": deployment,
        "probe": {},
        "managed_model": {
            "config_revision": digest(config),
            "identity_signature": "serving_with_model_version",
        },
    }
    state = ReleaseState(docs.s3, "s3://bucket/managed-release", "synthetic")
    state.claim("bootstrap", ready["cutoff"])
    state.value["active"].update(stage="trained", run_uri="s3://bucket/runs/one")
    state.save()
    work = tmp_path / "work"
    work.mkdir()
    assert (
        step(
            docs.s3,
            sm,
            None,
            state,
            config,
            ready,
            binding,
            work=work,
            bootstrap_run="s3://bucket/runs/one",
            model_publication=ManagedPublication(docs.s3, binding),
        )
        == "pending"
    )
    assert state.value["managed_contract"]["signature"]["outputs"]
    assert state.value["active"]["managed_first_deployment"] is True
    manifest = json.loads((work / "package/manifest.json").read_text())
    assert manifest["verification"]["serving_version_signature"] == "serving_with_model_version"


@pytest.mark.parametrize("legacy_monitoring_identity", [False, True])
@pytest.mark.parametrize("early_fence", [False, True])
def test_adoption_waits_for_retry_drain_and_keeps_endpoint(
    setup, legacy_monitoring_identity, early_fence
):
    import hashlib
    import tarfile

    from recommend_tf.inference.tests.test_idea_publication import AppConfig
    from recommend_tf.platforms.model_management import adoption
    from recommend_tf.platforms.model_management.contracts import canonical

    docs, group, registry, name, _ = setup
    model = registry["models"][name]
    old_binding = binding_for(group, name, model)
    old_binding.pop("managed_model")
    if legacy_monitoring_identity:
        old_binding.pop("model_name")
        old_binding["monitoring"] = {"model_name": name}
    old_binding["download_limit_bytes"] = 100_000
    group["binding_template"]["download_limit_bytes"] = 100_000
    group["legacy_rules"] = ["old-daily"]
    group["adoption_prefixes"] = ["s3://bucket"]
    config_pin = docs.immutable("legacy-config", model["config"])
    binding_pin = docs.immutable("legacy-binding", old_binding)
    uri = "s3://bucket/models/old/model.tar.gz"
    files = {
        "signature.json": canonical({"model_version": "a" * 64, "inputs": {"id": "int64"}}),
        "feature-config.json": canonical({"features": ["id"]}),
    }
    zipped = io.BytesIO()
    with tarfile.open(fileobj=zipped, mode="w:gz") as tar:
        for filename, body in files.items():
            info = tarfile.TarInfo(filename)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    docs.s3.put_object(Bucket="bucket", Key="models/old/model.tar.gz", Body=zipped.getvalue())
    docs.put(
        "s3://bucket/models/old/manifest.json",
        {"files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}},
    )
    state = {
        "version": "doku-tf-release/v1",
        "active": None,
        "high_water": "2026-10-07",
        "online": {"model_version": "a" * 64, "endpoint_config": "original"},
        "history": [{"stage": "deployed", "model_version": "a" * 64, "model_uri": uri}],
    }
    docs.put(old_binding["state_uri"], state)

    class Events:
        state = "ENABLED"

        def list_targets_by_rule(self, **kwargs):
            return {
                "Targets": [
                    {
                        "Id": "worker",
                        "EcsParameters": {"TaskDefinitionArn": group["worker"]["task_definition"]},
                        "RetryPolicy": {"MaximumEventAgeInSeconds": 86400 if early_fence else 60},
                    }
                ]
            }

        def describe_rule(self, **kwargs):
            return {"State": self.state}

        def disable_rule(self, **kwargs):
            self.state = "DISABLED"

        def enable_rule(self, **kwargs):
            self.state = "ENABLED"

    class LegacyECS(ECS):
        retired = False

        def get_paginator(self, operation):
            key = "taskDefinitionArns" if operation == "list_task_definitions" else "serviceArns"
            return SimpleNamespace(paginate=lambda **kw: [{key: []}])

        def describe_task_definition(self, **kwargs):
            env = {
                "DOKU_CONFIG_URI": config_pin["uri"],
                "DOKU_CONFIG_SHA256": config_pin["sha256"],
                "DOKU_BINDING_URI": binding_pin["uri"],
                "DOKU_BINDING_SHA256": binding_pin["sha256"],
            }
            return {
                "taskDefinition": {
                    "status": "INACTIVE" if self.retired else "ACTIVE",
                    "taskDefinitionArn": group["worker"]["task_definition"],
                    "containerDefinitions": [
                        {
                            "name": "pipeline",
                            "environment": [{"name": k, "value": v} for k, v in env.items()],
                        }
                    ],
                }
            }

    sm = SimpleNamespace(
        describe_endpoint=lambda **kw: {
            "EndpointStatus": "InService",
            "EndpointConfigName": "original",
        },
        describe_endpoint_config=lambda **kw: {"ProductionVariants": [{"ModelName": "old"}]},
        describe_model=lambda **kw: {"PrimaryContainer": {"ModelDataUrl": uri}},
    )
    events, ecs = Events(), LegacyECS()
    before = {**registry, "models": {}}
    app = AppConfig({"registry": canonical(before)})
    store = Registry(docs, group, app, app)
    pin = adoption.plan(docs, group, before, name, "old-daily", events=events, ecs=ecs, sm=sm)[
        "plan"
    ]
    result = adoption.apply(docs, group, store, pin, events=events, ecs=ecs, sm=sm, now=lambda: 100)
    assert result["waiting"] == "legacy_delivery_retries_draining"
    assert store.read()["models"] == {}
    assert events.state == "DISABLED"
    adoption.abort(docs, group, pin, events=events)
    assert events.state == "ENABLED"
    restarted = adoption.apply(
        docs, group, store, pin, events=events, ecs=ecs, sm=sm, now=lambda: 101
    )
    assert restarted["drain_after"] == (86501 if early_fence else 161)
    complete_at = 161
    if early_fence:
        with pytest.raises(ValueError, match="INACTIVE"):
            adoption.fence(docs, group, pin, events=events, ecs=ecs, now=lambda: 102)
        ecs.retired = True
        result = adoption.fence(docs, group, pin, events=events, ecs=ecs, now=lambda: 102)
        assert result["drain_after"] == 86501  # Original barrier is not rewritten.
        assert result["revision_fence"]["ready_after"] == 702
        with pytest.raises(ValueError, match="replacement release"):
            adoption.abort(docs, group, pin, events=events)
        result = adoption.apply(
            docs, group, store, pin, events=events, ecs=ecs, sm=sm, now=lambda: 701
        )
        assert result["waiting"] == "legacy_revision_fence_propagating"
        assert store.read()["models"] == {}
        complete_at = 702
    for _ in range(5):
        result = adoption.apply(
            docs, group, store, pin, events=events, ecs=ecs, sm=sm, now=lambda: complete_at
        )
        if result["status"] == "managed":
            break
    assert result["status"] == "managed"
    assert docs.read(old_binding["state_uri"])[0] == state
    assert store.read()["models"][name]["config"] == model["config"]
    assert events.state == "DISABLED"
    with pytest.raises(ValueError, match="activated"):
        adoption.abort(docs, group, pin, events=events)


@pytest.mark.parametrize("actual_status", ["RUNNING", "STOPPING", "STOPPED"])
def test_adoption_waits_for_actual_exit_after_stop_request(setup, actual_status):
    from recommend_tf.platforms.model_management.adoption import idle

    docs, group, _, _, _ = setup
    state = {"active": None, "high_water": "2026-10-07"}
    binding = {"state_uri": "s3://bucket/state/legacy.json"}
    docs.put(binding["state_uri"], state)
    evidence = {
        "state_sha256": digest(state),
        "snapshot": {
            "task_definition": "arn:aws:ecs:us-east-2:123456789012:task-definition/legacy:1"
        },
    }

    class StoppingECS:
        def list_tasks(self, **kwargs):
            if kwargs["desiredStatus"] == "RUNNING":
                return {"taskArns": []}
            if not kwargs.get("nextToken"):
                return {"taskArns": [], "nextToken": "page2"}
            return {"taskArns": ["stopping-task"]}

        def describe_tasks(self, **kwargs):
            assert kwargs["tasks"] == ["stopping-task"]
            return {"tasks": [{"lastStatus": actual_status}], "failures": []}

    if actual_status == "STOPPED":
        idle(docs, group, evidence, binding, StoppingECS())
    else:
        with pytest.raises(ValueError, match="executor has not stopped"):
            idle(docs, group, evidence, binding, StoppingECS())


@pytest.mark.parametrize("case", ["other_model", "target_model", "unknown", "service"])
def test_revision_fence_rejects_remaining_legacy_paths(setup, case):
    from recommend_tf.platforms.model_management.adoption_fence import verify

    docs, group, _, name, _ = setup
    group["adoption_prefixes"] = ["s3://bucket"]
    pin = docs.immutable("config", {"model_name": name if case == "target_model" else "other"})
    original = group["worker"]["task_definition"]
    active = original.rsplit(":", 1)[0] + ":2"
    evidence = {"snapshot": {"task_definition": original}}

    class Inventory:
        def get_paginator(self, operation):
            key = "taskDefinitionArns" if operation == "list_task_definitions" else "serviceArns"
            values = (
                [active] if key == "taskDefinitionArns" else (["svc"] if case == "service" else [])
            )
            return SimpleNamespace(paginate=lambda **kw: [{key: []}, {key: values}])

        def describe_task_definition(self, taskDefinition):
            env = (
                []
                if case == "unknown"
                else [
                    {"name": "DOKU_CONFIG_URI", "value": pin["uri"]},
                    {"name": "DOKU_CONFIG_SHA256", "value": pin["sha256"]},
                ]
            )
            return {
                "taskDefinition": {
                    "family": "daily",
                    "status": "INACTIVE" if taskDefinition == original else "ACTIVE",
                    "containerDefinitions": [{"name": "pipeline", "environment": env}],
                }
            }

        def describe_services(self, **kwargs):
            return {"services": [{"taskDefinition": original}], "failures": []}

    if case == "other_model":
        assert verify(docs, group, evidence, name, Inventory())["target"] == original
    else:
        with pytest.raises(ValueError):
            verify(docs, group, evidence, name, Inventory())


def test_platform_upgrade_preserves_queued_runtime(setup):
    docs, group, _, _, _ = setup
    original, request = queued(setup)
    changed = copy.deepcopy(group)
    changed["worker"]["task_definition"] = (
        group["worker"]["task_definition"].rsplit(":", 1)[0] + ":2"
    )
    changed["binding_template"]["training_request"]["AlgorithmSpecification"]["TrainingImage"] = (
        "image@sha256:" + "2" * 64
    )
    ecs = ECS()
    launch(docs, changed, ecs, request["slot_uri"])
    assert ecs.calls[0]["taskDefinition"] == group["worker"]["task_definition"]
    _, values = validate_request(docs, changed, original["request"])
    assert values["binding"]["training_request"]["AlgorithmSpecification"][
        "TrainingImage"
    ].endswith("1" * 64)


def test_stopped_worker_recovery_keeps_request_and_uses_new_attempt(setup):
    from recommend_tf.platforms.model_management.dispatch import tick

    docs, group, registry, name, _ = setup
    original, request = queued(setup)
    ecs = ECS()
    launch(docs, group, ecs, request["slot_uri"])
    # Disable only creation of future requests; the already submitted one remains runnable.
    registry["models"][name]["daily_enabled"] = False
    tick(docs, group, registry, ecs)
    current, _ = docs.read(request["slot_uri"])
    assert current["request"] == original["request"]
    assert current["launch_attempt"] == 1
    assert ecs.calls[0]["clientToken"] != ecs.calls[1]["clientToken"]


def test_bootstrap_inventory_is_frozen_before_start(setup):
    docs, group, registry, name, ready = setup
    sealed = {"version": "doku-tf-run/v1", "files": {"experiment-config.json": "a" * 64}}
    docs.put("s3://bucket/runs/source/COMMITTED.json", sealed)
    original = enqueue(
        docs,
        group,
        registry,
        name,
        cutoff=ready["cutoff"],
        bootstrap_run="s3://bucket/runs/source",
        operation_id="first",
    )
    raw, etag = docs.read("s3://bucket/runs/source/COMMITTED.json")
    docs.put("s3://bucket/runs/source/COMMITTED.json", {**raw, "files": {}}, etag)
    _, values = validate_request(docs, group, original["request"])
    assert values["bootstrap_evidence"] == sealed


def test_rebuilt_failed_slot_can_replace_its_old_launch_gate(setup, monkeypatch):
    from recommend_tf.platforms.model_management import requests

    docs, group, registry, _, ready = setup
    original, request = queued(setup)
    ecs = ECS()
    launch(docs, group, ecs, request["slot_uri"])
    record, etag = docs.read(request["slot_uri"])
    record["status"] = "failed"
    docs.put(request["slot_uri"], record, etag)
    monkeypatch.setattr(requests, "ready_for", lambda *args: ready)
    rebuilt = requests.rebuild(
        docs, group, registry, request["slot_uri"], original["request"], retry_id="retry_one"
    )
    outcome = launch(docs, group, ecs, request["slot_uri"])
    assert outcome["status"] == "submitted"
    assert outcome["request"] == rebuilt["request"]
    assert ecs.calls[0]["clientToken"] != ecs.calls[1]["clientToken"]


def test_api_registration_enrolls_resources_without_launching_training(setup):
    from recommend_tf.inference.tests.test_idea_publication import AppConfig
    from recommend_tf.platforms.model_management.api import handle
    from recommend_tf.platforms.model_management.contracts import API, canonical

    docs, group, registry, name, ready = setup
    before = {**registry, "models": {}}
    group["resource_manager"] = "arn:aws:lambda:us-east-2:123456789012:function:synthetic"
    app = AppConfig({"registry": canonical(before)})
    ecs = ECS()
    calls = []

    def enroll(**args):
        request = json.loads(args["Payload"])
        calls.append(request)
        return {
            "StatusCode": 200,
            "Payload": io.BytesIO(
                canonical({"endpoint_name": request["endpoint_name"], "alarm_names": ["health"]})
            ),
        }

    clients = dict(
        s3=docs.s3,
        control=app,
        data=app,
        ecs=ecs,
        events=None,
        sm=None,
        resource_client=SimpleNamespace(invoke=enroll),
    )
    plan = handle(
        {
            "schema_version": API,
            "operation": "plan",
            "arguments": {
                "action": "register",
                "model_name": name,
                "expected_registry_sha256": digest(before),
                "payload": {
                    "config": registry["models"][name]["config"],
                    "service": registry["models"][name]["service"],
                    "first_cutoff": ready["cutoff"],
                },
            },
        },
        group,
        **clients,
    )
    for _ in range(5):
        result = handle(
            {"schema_version": API, "operation": "apply", "arguments": {"plan": plan["plan"]}},
            group,
            **clients,
        )
        if result["status"] == "complete":
            break
    assert result["status"] == "complete"
    assert calls and not ecs.calls
    after = Registry(docs, group, app, app).read()
    assert after["models"][name]["daily_enabled"] is False
    assert group["worker"]["task_definition"].endswith(":1")


def test_bootstrap_recovery_matches_frozen_source_inventory(setup):
    from recommend_tf.platforms.model_management.dispatch import recover
    from recommend_tf.platforms.sagemaker.operation_identity import operation_identity

    docs, group, registry, name, ready = setup
    sealed = {"version": "doku-tf-run/v1", "files": {"experiment-config.json": "a" * 64}}
    docs.put("s3://bucket/runs/source/COMMITTED.json", sealed)
    original = enqueue(
        docs,
        group,
        registry,
        name,
        cutoff=ready["cutoff"],
        bootstrap_run="s3://bucket/runs/source",
        operation_id="first",
    )
    request, values = validate_request(docs, group, original["request"])
    ecs = ECS()
    launch(docs, group, ecs, request["slot_uri"])
    state = ReleaseState(docs.s3, values["binding"]["state_uri"], group["kms_key"])
    op = operation_identity(
        values["config"],
        values["ready"],
        values["binding"],
        bootstrap_run=request["bootstrap_run"],
        evidence=sealed,
    )
    state.claim(op, ready["cutoff"])
    result = recover(docs, group, ecs, request["slot_uri"], original["request"])
    assert result["status"] == "submitted"
    assert result["request"] == original["request"]


def test_graceful_pause_can_resume_after_ecs_task_metadata_expires(setup):
    from recommend_tf.platforms.model_management.dispatch import recover

    docs, group, _, _, _ = setup
    original, request = queued(setup)
    ecs = ECS()
    launch(docs, group, ecs, request["slot_uri"])
    record, etag = docs.read(request["slot_uri"])
    record["status"] = "deferred_publication"
    docs.put(request["slot_uri"], record, etag)
    ecs.describe_tasks = lambda **kw: pytest.fail("graceful release is already durably proven")
    outcome = recover(docs, group, ecs, request["slot_uri"], original["request"])
    assert outcome["status"] == "submitted"
    assert outcome["request"] == original["request"]


def test_failed_release_reconciliation_remains_explicitly_rebuildable(setup):
    from recommend_tf.platforms.model_management.dispatch import recover
    from recommend_tf.platforms.sagemaker.operation_identity import operation_identity

    docs, group, _, _, _ = setup
    original, request = queued(setup)
    _, values = validate_request(docs, group, original["request"])
    ecs = ECS()
    launch(docs, group, ecs, request["slot_uri"])
    state = ReleaseState(docs.s3, values["binding"]["state_uri"], group["kms_key"])
    operation = operation_identity(values["config"], values["ready"], values["binding"])
    failed = state.claim(operation, request["cutoff"])
    failed["stage"] = "training_failed"
    state.finish(failed)
    result = recover(docs, group, ecs, request["slot_uri"], original["request"])
    assert result["status"] == "failed"
    assert result["release_stage"] == "training_failed"
    assert len(ecs.calls) == 1


def test_adoption_preserves_exact_legacy_endpoint_and_role(setup):
    _, group, registry, name, _ = setup
    model = registry["models"][name]
    legacy = binding_for(group, name, model)
    legacy["deployment"]["endpoint_name"] = "doku-dte-legacy-official"
    legacy["deployment"]["execution_role_arn"] = "arn:aws:iam::123456789012:role/old-inference"
    group["adoption_prefixes"] = ["s3://bucket/state", "s3://bucket/work", "s3://bucket/models"]
    group["legacy_rules"] = ["legacy-daily"]
    model["adoption"] = {
        "binding": legacy,
        "legacy_rule": "legacy-daily",
        "evidence_uri": "s3://bucket/manage/fyp/evidence.json",
    }
    with pytest.raises(ValueError, match="outside group"):
        binding_for(group, name, model)
    group["adoption_services"] = {
        "doku-dte-legacy-official": {
            "execution_role_arn": legacy["deployment"]["execution_role_arn"]
        }
    }
    validate_group(group)
    validate_registry(registry, group)
    assert binding_for(group, name, model)["deployment"] == legacy["deployment"]
    legacy["deployment"]["execution_role_arn"] = "arn:aws:iam::123456789012:role/not-enrolled"
    with pytest.raises(ValueError, match="inference role"):
        binding_for(group, name, model)
    group["adoption_services"]["*"] = {
        "execution_role_arn": "arn:aws:iam::123456789012:role/old-inference"
    }
    with pytest.raises(ValueError, match="explicit adopted Endpoint"):
        validate_group(group)
