"""One-shot ECS orchestrator. Explicit bindings only; never starts a local timer."""

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from recommend_tf.inference import official
from recommend_tf.train.comm.artifact_integrity import checksum
from recommend_tf.train.comm.datasvr.daily_contracts import digest
from recommend_tf.train.comm.datasvr.daily_dataset import freeze
from recommend_tf.train.comm.early_stop import validate_early_stop
from recommend_tf.train.comm.export_tasks import export_outputs

from .publish_model import (
    ReleaseState,
    contains,
    deploy_step,
    describe,
    immutable_put,
    location,
    publish_package,
    read,
    request_rollback,
    rollback_step,
    verify_probe,
)


def fetch_run(s3, prefix, target, limit, evidence=None):
    """Fetch the checksum inventory first; refuse traversal and bound local disk use."""
    if evidence is None:
        body, _ = read(s3, prefix.rstrip("/") + "/COMMITTED.json")
        record = json.loads(body)
    else:
        record = evidence
        body = json.dumps(record, sort_keys=True).encode()
    if record.get("version") != "doku-tf-run/v1":
        raise ValueError("successful sealed training output required")
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    used = 0
    for name, expected in record["files"].items():
        path = target / name
        if not path.resolve().is_relative_to(target) or Path(name).is_absolute():
            raise ValueError("invalid artifact path")
        bucket, key = location(prefix.rstrip("/") + "/" + name)
        response = s3.get_object(Bucket=bucket, Key=key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with response["Body"] as stream, path.open("xb") as out:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                used += len(chunk)
                if used > limit:
                    raise ValueError("run download exceeds explicit disk budget")
                out.write(chunk)
        if checksum(path) != expected:
            raise ValueError("downloaded training output checksum differs")
    (target / "COMMITTED.json").write_bytes(body)


def freeze_input(config, ready, target, binding, s3):
    source = binding.get("data_source", {"freeze": "ready_manifest"})
    if source["freeze"] == "ready_manifest":
        return freeze(config, ready, target)
    if source["freeze"] == "idea_mapped":
        from .idea_mapped_input import freeze as freeze_mapped

        return freeze_mapped(s3, config, ready, source, target)
    raise ValueError("unknown data_source.freeze")


def prepare_job(s3, config, ready, binding, operation, work):
    frozen = work / "frozen"
    resolved = freeze_input(config, ready, frozen, binding, s3)
    prefix = binding["work_prefix"].rstrip("/") + "/" + operation
    manifest_uri = prefix + "/daily/input/manifest.json"
    resolved["dataset_config"]["dataset"] = {"manifest_uri": manifest_uri}
    record_path = frozen / "input/record.json"
    record = json.loads(record_path.read_text())
    record["manifest_uri"] = manifest_uri
    record_path.write_text(json.dumps(record, sort_keys=True))
    config_path = frozen / "config.json"
    config_path.write_text(json.dumps(resolved, sort_keys=True))
    for path in sorted((frozen / "input").rglob("*")):
        if path.is_file():
            immutable_put(
                s3,
                prefix + "/daily/input/" + path.relative_to(frozen / "input").as_posix(),
                path,
                binding["kms_key"],
            )
    immutable_put(s3, prefix + "/daily/config.json", config_path, binding["kms_key"])
    immutable_put(s3, prefix + "/ready.json", frozen / "ready.json", binding["kms_key"])
    request = copy.deepcopy(binding["training_request"])
    if (
        request["ResourceConfig"].get("InstanceCount") != 1
        or not request["AlgorithmSpecification"]["TrainingImage"]
    ):
        raise ValueError("single instance explicit training image required")
    instance = request["ResourceConfig"]["InstanceType"]
    if instance.startswith(("ml.p", "ml.g", "ml.trn", "ml.inf")):
        raise ValueError("daily training requires CPU")
    if "@sha256:" not in request["AlgorithmSpecification"]["TrainingImage"]:
        raise ValueError("training image must be immutable")
    name_prefix = binding.get("job_name_prefix", "doku-daily-")
    request["TrainingJobName"] = name_prefix + operation[: min(40, 63 - len(name_prefix))]
    run_uri = (
        (binding["runs_prefix"].rstrip("/") + "/" + request["TrainingJobName"] + "/live")
        if "runs_prefix" in binding
        else prefix + "/run"
    )
    if "runs_prefix" in binding:
        request["OutputDataConfig"]["S3OutputPath"] = run_uri.rsplit("/", 1)[0] + "/artifacts/"
    if any(c["ChannelName"] == "daily" for c in request.get("InputDataConfig", [])):
        raise ValueError("daily channel is reserved")
    request.setdefault("InputDataConfig", []).append(
        {
            "ChannelName": "daily",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": prefix + "/daily/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "InputMode": "File",
        }
    )
    request.setdefault("Environment", {}).update(
        {
            "DOKU_MODE": "daily",
            "DOKU_CONFIG": "/opt/ml/input/data/daily/config.json",
            "DOKU_CONFIG_SHA256": checksum(config_path),
            "DOKU_FROZEN_INPUT": "/opt/ml/input/data/daily/input",
            "DOKU_CUTOFF": ready["cutoff"],
            "DOKU_RUN_ROOT": "/opt/ml/work/run",
            "DOKU_ARTIFACT_URI": run_uri,
            "DOKU_KMS_KEY": binding["kms_key"],
        }
    )
    if binding.get("monitoring"):
        context = {key: binding[key] for key in ("model_name", "environment", "region")}
        context["outputs"] = binding.get("outputs") or export_outputs(config, {})
        context["training_job_name"] = request["TrainingJobName"]
        context["deployment"] = {"endpoint_name": binding["deployment"]["endpoint_name"]}
        context["monitoring"] = {"metric_mapping": binding["monitoring"].get("metric_mapping", {})}
        request["Environment"]["DOKU_MONITORING_CONTEXT"] = json.dumps(context, sort_keys=True)
    for name in ("DOKU_SOURCE_SHA256", "DOKU_LOCK_SHA256"):
        if not request["Environment"].get(name):
            raise ValueError("training image provenance required: " + name)
    return request, run_uri


def step(
    s3,
    sm,
    runtime,
    state,
    config,
    ready,
    binding,
    *,
    work,
    bootstrap_run=None,
    bootstrap_evidence=None,
    policy=None,
):
    active = state.value["active"]
    stage = active["stage"]
    if stage == "claimed":
        if not bootstrap_run and policy and not policy.allowed("training"):
            return "deferred_training"
        if bootstrap_run:
            active.update(stage="trained", run_uri=bootstrap_run)
        else:
            try:
                request, run_uri = prepare_job(
                    s3, config, ready, binding, active["operation"], work
                )
            except ValueError as error:
                # No training API call is possible before the durable training stage.
                active.update(stage="input_failed", error_type=type(error).__name__)
                state.save()
                return "input_failed"
            active.update(stage="training", training_request=request, run_uri=run_uri)
        state.save()
    return advance_release(
        s3,
        sm,
        runtime,
        state,
        binding,
        work=work,
        evidence=bootstrap_evidence,
        policy=policy,
        selection_options=validate_early_stop(config) if binding.get("verify_selection") else None,
    )


def advance_release(
    s3,
    sm,
    runtime,
    state,
    binding,
    *,
    work,
    evidence=None,
    policy=None,
    broadcast_feat=None,
    selection_options=None,
):
    active = state.value["active"]
    if active["stage"] == "training":
        request = active["training_request"]
        observed = describe(sm, "describe_training_job", TrainingJobName=request["TrainingJobName"])
        if observed is None:
            if policy and not policy.allowed("training"):
                return "deferred_training"
            sm.create_training_job(**request)
            return "pending"
        # Reconcile deterministic names against the actual submitted configuration.
        if any(
            not contains(observed.get(key), value)
            for key, value in request.items()
            if key != "Tags"
        ):
            raise ValueError("training job identity differs")
        if observed["TrainingJobStatus"] in {"Failed", "Stopped"}:
            active.update(stage="training_failed", failure_reason=observed["TrainingJobStatus"])
            state.save()
            return active["stage"]
        if observed["TrainingJobStatus"] != "Completed":
            return "pending"
        active["stage"] = "trained"
        if "TrainingEndTime" in observed:
            active["training_completed_at_ms"] = int(observed["TrainingEndTime"].timestamp() * 1000)
        state.save()
    if active["stage"] in {"trained", "exported"}:
        run = work / "run"
        fetch_run(s3, active["run_uri"], run, binding["download_limit_bytes"], evidence)
        run_config = json.loads((run / "experiment-config.json").read_text())
        if run_config["dataset_config"]["split"]["validation_to"] != active["cutoff"]:
            raise ValueError("release cutoff differs from the selected run validation cutoff")
        statuses = list(run.glob("*-status.json"))
        if not statuses or any(
            json.loads(p.read_text()).get("status") != "completed" for p in statuses
        ):
            active["stage"] = "no_selected_model"
            state.save()
            return active["stage"]
        if selection_options is not None:
            status = json.loads((run / "daily-status.json").read_text())
            selection = json.loads((run / "training/selection.json").read_text())
            if (
                status["cutoff"] != active["cutoff"]
                or status["selection"] != selection
                or any(selection.get(key) != selection_options[key] for key in ("metric", "weight"))
            ):
                raise ValueError("daily selection differs from the configured validation objective")
            active.update(selected_epoch=selection["epoch"], score=selection["score"])
        if binding.get("monitoring"):
            try:
                from recommend_tf.platforms.monitoring.report_metrics import report

                report_binding = (
                    binding
                    if binding.get("outputs")
                    else {**binding, "outputs": export_outputs(run_config, {})}
                )
                report(s3, run, report_binding, run_uri=active["run_uri"])
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "event": "monitoring_failed",
                            "operation": active["operation"],
                            "error_type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
        output = work / "package"
        options = work / "outputs.json"
        options.write_text(json.dumps(binding.get("outputs", {})))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "recommend_tf.train.export_model",
                "--run",
                str(run),
                "--output",
                str(output),
                "--outputs",
                str(options),
                *(["--broadcast-feat", broadcast_feat] if broadcast_feat else []),
            ],
            check=True,
            timeout=binding.get("export_timeout_seconds"),
        )
        manifest = json.loads((output / "manifest.json").read_text())
        export_digest = checksum(output / "manifest.json")
        if active.get("export_manifest_sha256", export_digest) != export_digest:
            raise ValueError("retry produced a different export; retain original evidence")
        active.update(
            stage="exported",
            model_version=manifest["model_version"],
            export_manifest_sha256=export_digest,
            export_tasks=list(json.loads((output / "signature.json").read_text())["outputs"]),
        )
        state.save()
        active["model_uri"] = publish_package(
            s3,
            output,
            binding["model_prefix"],
            binding["kms_key"],
            inference_protocol=official.protocol(binding["deployment"]),
        )
        active.update(stage="uploaded", model_version=manifest["model_version"])
        state.save()
    if active["stage"] in {"uploaded", "deploying"}:
        if active["stage"] == "uploaded" and policy and not policy.allowed("publication"):
            return "deferred_publication"
        return deploy_step(
            sm,
            runtime,
            state,
            binding["deployment"],
            probe=lambda client, endpoint, version: verify_probe(
                client,
                endpoint,
                version,
                {**binding["probe"], "tasks": active["export_tasks"]}
                if "export_tasks" in active
                else binding["probe"],
                inference_protocol=official.protocol(binding["deployment"]),
            ),
        )
    if active["stage"] == "rollback_pending":
        return rollback_step(sm, state, binding["deployment"])
    return active["stage"]


def acquire_executor(state, ecs, cluster, task_arn):
    """A new ECS executor may take over only after AWS confirms the old task stopped."""
    active = state.value["active"]
    old = active.get("executor")
    if old and old != task_arn:
        response = ecs.describe_tasks(cluster=cluster, tasks=[old])
        if (
            response.get("failures")
            or len(response.get("tasks", [])) != 1
            or response["tasks"][0]["lastStatus"] != "STOPPED"
        ):
            raise ValueError("previous ECS executor is not confirmed stopped")
    active["executor"] = task_arn
    state.save()


def validate_binding(binding, config=None):
    for name in (
        "environment",
        "region",
        "ecs_cluster",
        "state_uri",
        "work_prefix",
        "model_prefix",
        "kms_key",
    ):
        if not isinstance(binding.get(name), str) or not binding[name]:
            raise ValueError("explicit platform binding required: " + name)
    for name in (
        "poll_seconds",
        "max_wait_seconds",
        "download_limit_bytes",
        "export_timeout_seconds",
    ):
        if type(binding.get(name)) is not int or binding[name] <= 0:
            raise ValueError("positive execution budget required: " + name)
    deployment = binding["deployment"]
    if official.protocol(deployment) == official.PROTOCOL:
        # Reject legacy/hook environment settings before any training is submitted.
        # The placeholder digest is used only for validation, never deployment.
        official.environment(deployment, "0" * 64)
        probe_request = binding["probe"]["request"]
        if probe_request.get("signature_name") != "serving_default" or not probe_request.get(
            "inputs"
        ):
            raise ValueError("official probe requires prepared tensors")
    if (
        type(deployment["instance_count"]) is not int
        or deployment["instance_count"] < 1
        or deployment["instance_type"].startswith(("ml.p", "ml.g", "ml.inf", "ml.trn"))
    ):
        raise ValueError("explicit positive CPU inference capacity required")
    if config is not None and "export_tasks" in config:
        export_outputs(config, binding.get("outputs", {}))
    elif set(binding["probe"]["tasks"]) != set(binding["outputs"]):
        raise ValueError("probe must verify all configured primary tasks")
    if binding["probe"].get("fixture_kind") != "synthetic":
        raise ValueError("deployment probe must be explicitly synthetic")


def operation_identity(
    config, ready, binding, *, bootstrap_run=None, evidence=None, rollback_version=None, retry_id=""
):
    # Execution budgets may be increased when resuming the same immutable operation.
    stable_binding = {
        k: v
        for k, v in binding.items()
        if k
        not in {
            "monitoring",
            "runtime_policy",
            "legacy_ready_config_sha256",
            "deferred_uri",
            "poll_seconds",
            "max_wait_seconds",
            "download_limit_bytes",
            "export_timeout_seconds",
        }
    }
    return digest(
        {
            "config": config,
            "ready": ready,
            "binding": stable_binding,
            "bootstrap_run": bootstrap_run,
            "bootstrap_evidence": evidence,
            "rollback_version": rollback_version,
            "retry_id": retry_id,
        }
    )


def preflight_ready(state, config, ready, operation, *, skip=False, binding=None, s3=None):
    active = state.value["active"]
    if skip or any(r["operation"] == operation for r in state.value["history"]):
        return
    if active and (active["operation"] != operation or active["stage"] != "claimed"):
        return
    # A missing day must not strand an Endpoint claim. Completed/resuming jobs
    # use their frozen input and never reread an upstream daily publication.
    with tempfile.TemporaryDirectory(prefix="tf-ready-preflight-") as directory:
        freeze_input(config, ready, Path(directory) / "frozen", binding or {}, s3)


def main():
    parser = argparse.ArgumentParser()
    for key in ("config", "ready", "binding", "executor-arn"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument(
        "--bootstrap-run", help="Sealed run S3 prefix; bypass daily training for first release"
    )
    parser.add_argument(
        "--bootstrap-evidence", help="Audited local checksum inventory for historical run"
    )
    parser.add_argument("--rollback-version", help="Explicit retained model version to restore")
    parser.add_argument(
        "--retry-id", default="", help="Explicit new attempt after a terminal training failure"
    )
    args = parser.parse_args()
    return execute(args)


def execute(args):
    """Run the same durable workflow from CLI or an ECS adapter."""
    evidence = (
        json.loads(Path(args.bootstrap_evidence).read_text()) if args.bootstrap_evidence else None
    )
    config, binding = [json.loads(Path(p).read_text()) for p in (args.config, args.binding)]
    ready = json.loads(Path(args.ready).read_text()) if args.ready else None
    validate_binding(binding, config)
    if binding.get("monitoring"):
        from recommend_tf.train.comm.model_identity import model_name

        if model_name(config, required=True) != binding.get("model_name"):
            raise ValueError("monitoring binding and configuration model_name differ")
    if evidence is not None and not args.bootstrap_run:
        raise ValueError("bootstrap evidence requires bootstrap run")
    import fcntl

    import boto3

    # Prevent two local processes in the same ECS executor from sharing its identity.
    with open("/tmp/doku-tf-daily-executor.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = boto3.Session(region_name=binding["region"])
        s3, sm, runtime, ecs = [
            session.client(name) for name in ("s3", "sagemaker", "sagemaker-runtime", "ecs")
        ]
        policy = None
        if binding.get("runtime_policy"):
            from botocore.config import Config

            from recommend_tf.platforms.runtime_policy import RuntimePolicy

            policy = RuntimePolicy(
                session.client(
                    "appconfigdata",
                    config=Config(
                        connect_timeout=3,
                        read_timeout=5,
                        retries={"mode": "standard", "max_attempts": 2},
                    ),
                ),
                binding["runtime_policy"],
            )
        state = ReleaseState(s3, binding["state_uri"], binding["kms_key"])
        if ready is None:
            from .idea_mapped_input import next_ready

            source = binding["data_source"]
            if source["freeze"] != "idea_mapped":
                raise ValueError("this source requires a pinned ready envelope")
            ready = next_ready(s3, state, source, config)
            if ready is None:
                return "caught_up"
        if binding.get("monitoring", {}).get("catalog_uri"):
            from recommend_tf.train.comm.model_identity import recipe_sha256

            catalog_uri = (
                binding["monitoring"]["catalog_uri"].rstrip("/")
                + "/"
                + config["model_name"]
                + ".json"
            )
            catalog = None
            try:
                catalog = json.loads(read(s3, catalog_uri)[0])
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "event": "monitoring_catalog_unavailable",
                            "error_type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
            if catalog and catalog.get("recipe_sha256", recipe_sha256(config)) != recipe_sha256(
                config
            ):
                raise ValueError(
                    "model_name already identifies a different recipe; use a new model_name"
                )
        operation = operation_identity(
            config,
            ready,
            binding,
            bootstrap_run=args.bootstrap_run,
            evidence=evidence,
            rollback_version=args.rollback_version,
            retry_id=args.retry_id,
        )
        active = state.value["active"]
        if (
            args.rollback_version
            and policy
            and not (active and active["operation"] == operation)
            and not policy.allowed("publication")
        ):
            return "deferred_publication"
        preflight_ready(
            state,
            config,
            ready,
            operation,
            skip=bool(args.bootstrap_run or args.rollback_version),
            binding=binding,
            s3=s3,
        )
        result = (
            request_rollback(
                state, sm, binding["deployment"]["endpoint_name"], args.rollback_version, operation
            )
            if args.rollback_version
            else state.claim(
                operation,
                ready["cutoff"],
                ready=ready
                if binding.get("data_source", {}).get("freeze") == "idea_mapped"
                else None,
            )
        )
        if state.value["active"] is None:
            print(json.dumps(result))
            if result["stage"] != "deployed" and not (
                args.rollback_version and result["stage"] == "rolled_back"
            ):
                raise RuntimeError("previous attempt failed; explicit retry-id required")
            return result["stage"]
        acquire_executor(state, ecs, binding["ecs_cluster"], args.executor_arn)
        started = time.monotonic()
        while time.monotonic() - started < binding["max_wait_seconds"]:
            with tempfile.TemporaryDirectory(prefix="doku-tf-daily-") as directory:
                outcome = step(
                    s3,
                    sm,
                    runtime,
                    state,
                    config,
                    ready,
                    binding,
                    work=Path(directory),
                    bootstrap_run=args.bootstrap_run,
                    bootstrap_evidence=evidence,
                    policy=policy,
                )
            if outcome.startswith("deferred_"):
                # A graceful pause owns no in-flight AWS operation. Release only this
                # executor, so recovery after a long pause does not require expired
                # ECS STOPPED metadata. Journal CAS still protects concurrent resumes.
                if state.value["active"].get("executor") != args.executor_arn:
                    raise ValueError("cannot defer another executor")
                state.value["active"].pop("executor")
                state.save()
                print(json.dumps({"operation": operation, "status": outcome}), flush=True)
                return outcome
            if outcome != "pending":
                state.finish(state.value["active"], binding if binding.get("monitoring") else None)
                if binding.get("monitoring"):
                    try:
                        from recommend_tf.platforms.monitoring.report_metrics import report_release

                        report_release(s3, state, binding)
                    except Exception as error:
                        print(
                            json.dumps(
                                {
                                    "event": "monitoring_failed",
                                    "operation": operation,
                                    "error_type": type(error).__name__,
                                }
                            ),
                            flush=True,
                        )
                print(json.dumps({"operation": operation, "status": outcome}))
                if outcome != "deployed" and not (
                    args.rollback_version and outcome == "rolled_back"
                ):
                    raise RuntimeError("daily pipeline did not deploy: " + outcome)
                return outcome
            time.sleep(binding["poll_seconds"])
        raise TimeoutError(
            "execution budget exhausted; persistent state retained for reconciliation"
        )


if __name__ == "__main__":
    main()
