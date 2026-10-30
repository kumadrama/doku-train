"""Immutable S3 packages and restartable all-at-once Endpoint deployment."""

import hashlib
import json
import math
import tarfile
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from recommend_tf.train.comm.artifact_integrity import checksum, files


def location(uri):
    value = urlsplit(uri)
    if (
        value.scheme != "s3"
        or not value.netloc
        or not value.path.strip("/")
        or value.query
        or value.fragment
        or ".." in value.path.split("/")
    ):
        raise ValueError("explicit S3 object URI required")
    return value.netloc, value.path.lstrip("/")


def missing(error):
    return getattr(error, "response", {}).get("Error", {}).get("Code") in {
        "NoSuchKey",
        "404",
        "ResourceNotFound",
        "ResourceNotFoundException",
    }


def read(s3, uri):
    bucket, key = location(uri)
    result = s3.get_object(Bucket=bucket, Key=key)
    with result["Body"] as stream:
        return stream.read(), result["ETag"]


def immutable_put(s3, uri, path, kms_key):
    bucket, key = location(uri)
    expected = checksum(path)
    try:
        with Path(path).open("rb") as stream:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=stream,
                IfNoneMatch="*",
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId=kms_key,
                Metadata={"sha256": expected},
            )
    except Exception as error:
        if getattr(error, "response", {}).get("Error", {}).get("Code") not in {
            "PreconditionFailed",
            "412",
        }:
            raise
    response = s3.get_object(Bucket=bucket, Key=key)
    digest = hashlib.sha256()
    with response["Body"] as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("immutable S3 object differs")


class ReleaseState:
    """One conditional object per Endpoint; no expiring lease or unsafe takeover."""

    def __init__(self, s3, uri, kms_key):
        self.s3, self.uri, self.kms_key = s3, uri, kms_key
        self.etag = None
        try:
            body, self.etag = read(s3, uri)
            self.value = json.loads(body)
        except Exception as error:
            if not missing(error):
                raise
            self.value = {
                "version": "doku-tf-release/v1",
                "active": None,
                "high_water": None,
                "history": [],
            }
        if self.value.get("version") != "doku-tf-release/v1":
            raise ValueError("unknown release state")

    def save(self):
        bucket, key = location(self.uri)
        result = self.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(self.value, sort_keys=True, allow_nan=False).encode(),
            **({"IfMatch": self.etag} if self.etag else {"IfNoneMatch": "*"}),
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=self.kms_key,
        )
        self.etag = result["ETag"]

    def claim(self, operation, cutoff):
        date.fromisoformat(cutoff)
        active = self.value["active"]
        if active:
            if active["operation"] != operation:
                raise ValueError(
                    "Endpoint already has an active release; reconcile its executor first"
                )
            return active
        prior = next((r for r in self.value["history"] if r["operation"] == operation), None)
        if prior:
            return prior
        if self.value["high_water"] and cutoff <= self.value["high_water"]:
            raise ValueError("automatic release cannot replace equal/older data cutoff")
        active = {"operation": operation, "cutoff": cutoff, "stage": "claimed"}
        self.value["active"] = active
        self.save()
        return active

    def finish(self, result):
        if result["stage"] == "deployed":
            self.value["high_water"] = max(
                self.value["high_water"] or result["cutoff"], result["cutoff"]
            )
            self.value["online"] = {
                key: result[key] for key in ("model_version", "endpoint_config", "cutoff")
            }
        if result["stage"] == "rolled_back" and result.get("rollback_target"):
            self.value["online"] = result["rollback_target"]
        self.value["history"].append(dict(result))
        self.value["active"] = None
        self.save()


def publish_package(s3, package, prefix, kms_key):
    package = Path(package)
    manifest = json.loads((package / "manifest.json").read_text())
    actual = files(package)
    actual.pop("manifest.json")
    if manifest.get("version") != "doku-tf-inference/v1" or actual != manifest["files"]:
        raise ValueError("invalid inference package")
    uri = (
        prefix.rstrip("/")
        + "/"
        + manifest["model_version"]
        + "/"
        + checksum(package / "manifest.json")
    )
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "model.tar.gz"
        # Deterministic tar metadata and gzip header make retry contents identical.
        import gzip

        with (
            archive.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as zipped,
            tarfile.open(fileobj=zipped, mode="w|") as tar,
        ):
            for name in sorted(files(package)):
                path = package / name
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                with path.open("rb") as body:
                    tar.addfile(info, body)
        immutable_put(s3, uri + "/model.tar.gz", archive, kms_key)
    immutable_put(s3, uri + "/manifest.json", package / "manifest.json", kms_key)
    return uri + "/model.tar.gz"


def describe(client, method, **kwargs):
    try:
        return getattr(client, method)(**kwargs)
    except Exception as error:
        # SageMaker often reports missing named resources as ValidationException.
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        message = getattr(error, "response", {}).get("Error", {}).get("Message", "").lower()
        if missing(error) or (
            code == "ValidationException"
            and ("could not find" in message or "does not exist" in message)
        ):
            return None
        raise


def contains(actual, expected):
    """AWS descriptions may add default fields; all requested fields must match."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and contains(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(contains(a, e) for a, e in zip(actual, expected))
        )
    return actual == expected


def ensure_resource(client, kind, request):
    name_key = {"model": "ModelName", "endpoint_config": "EndpointConfigName"}[kind]
    observed = describe(client, "describe_" + kind, **{name_key: request[name_key]})
    if observed is None:
        getattr(client, "create_" + kind)(**request)
    else:
        for key, expected in request.items():
            if key == "Tags":
                continue
            if not contains(observed.get(key), expected):
                raise ValueError("existing SageMaker resource has different settings: " + key)


def deploy_step(client, runtime, state, binding, *, probe):
    """One deployment tick. Caller holds an exclusive executor for this release state."""
    active = state.value["active"]
    endpoint = binding["endpoint_name"]
    image = binding["image_uri"]
    if "@sha256:" not in image or not binding["alarm_names"] or binding["wait_seconds"] <= 0:
        raise ValueError("deployment requires pinned image, alarms and positive observation period")
    if binding["artifact_contract"] != "doku-tf-inference/v1":
        raise ValueError("serving image contract is not confirmed")
    name = (
        "tf-"
        + hashlib.sha256(
            (
                endpoint
                + active["model_version"]
                + active["model_uri"]
                + json.dumps(binding, sort_keys=True)
            ).encode()
        ).hexdigest()[:48]
    )
    model = {
        "ModelName": name,
        "ExecutionRoleArn": binding["execution_role_arn"],
        "PrimaryContainer": {
            "Image": image,
            "ModelDataUrl": active["model_uri"],
            "Environment": {"DOKU_MODEL_VERSION": active["model_version"]},
        },
        "EnableNetworkIsolation": True,
    }
    if binding.get("vpc_config"):
        model["VpcConfig"] = binding["vpc_config"]
    endpoint_config = {
        "EndpointConfigName": name,
        "ProductionVariants": [
            {
                "VariantName": "primary",
                "ModelName": name,
                "InitialInstanceCount": binding["instance_count"],
                "InstanceType": binding["instance_type"],
                "InitialVariantWeight": 1.0,
            }
        ],
    }
    if active["stage"] == "uploaded":
        ensure_resource(client, "model", model)
        ensure_resource(client, "endpoint_config", endpoint_config)
        current = describe(client, "describe_endpoint", EndpointName=endpoint)
        if current and current["EndpointStatus"] != "InService":
            raise ValueError("Endpoint has another unfinished operation")
        active.update(
            stage="deploying",
            endpoint_config=name,
            previous_config=current["EndpointConfigName"] if current else None,
            submitted=False,
        )
        state.save()  # Durable intent before a possibly ambiguous API response.
    if active["stage"] == "deploying":
        current = describe(client, "describe_endpoint", EndpointName=endpoint)
        status = current["EndpointStatus"] if current else None
        pending = (current or {}).get("PendingDeploymentSummary", {}).get("EndpointConfigName")
        if pending == name or (
            current and current["EndpointConfigName"] == name and status != "InService"
        ):
            if not active.get("observed_progress"):
                active["observed_progress"] = True
                state.save()
        if current and current["EndpointConfigName"] == name and status == "InService":
            try:
                probe(runtime, endpoint, active["model_version"])
            except Exception as error:
                active.update(
                    stage="rollback_pending" if active["previous_config"] else "deployment_failed",
                    probe_error=type(error).__name__,
                )
                state.save()
                return (
                    rollback_step(client, state, binding)
                    if active["previous_config"]
                    else "deployment_failed"
                )
            active["stage"] = "deployed"
            state.save()
            return "deployed"
        if status in {"Failed", "UpdateRollbackFailed"} and active["previous_config"]:
            active["stage"] = "rollback_pending"
            state.save()
            return rollback_step(client, state, binding)
        if active["submitted"]:
            if status in {"Failed", "UpdateRollbackFailed"} or (
                status == "InService"
                and active.get("observed_progress")
                and current["EndpointConfigName"] == active["previous_config"]
            ):
                active["stage"] = (
                    "rolled_back"
                    if active["previous_config"] and status == "InService"
                    else "deployment_failed"
                )
                state.save()
                return active["stage"]
            return "pending"
        if pending == name or (current and current["EndpointConfigName"] == name):
            active["submitted"] = True
            state.save()
            return "pending"
        if status not in {None, "InService"}:
            raise ValueError("ambiguous deployment state requires reconciliation")
        if (current["EndpointConfigName"] if current else None) != active["previous_config"]:
            raise ValueError("Endpoint changed outside this release; refusing to overwrite")
        if active.get("attempted"):
            raise ValueError("ambiguous prior deployment submission; reconcile before retry")
        kwargs = {"EndpointName": endpoint, "EndpointConfigName": name}
        kwargs["DeploymentConfig"] = {
            "BlueGreenUpdatePolicy": {
                "TrafficRoutingConfiguration": {
                    "Type": "ALL_AT_ONCE",
                    "WaitIntervalInSeconds": binding["wait_seconds"],
                },
                "TerminationWaitInSeconds": binding["termination_wait_seconds"],
                "MaximumExecutionTimeoutInSeconds": binding["timeout_seconds"],
            },
            "AutoRollbackConfiguration": {
                "Alarms": [{"AlarmName": n} for n in binding["alarm_names"]]
            },
        }
        active["attempted"] = True
        state.save()
        if current:
            client.update_endpoint(**kwargs)
        else:
            client.create_endpoint(**kwargs)
        active["submitted"] = True
        state.save()
    return "pending"


def verify_probe(runtime, endpoint, version, probe):
    response = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType=probe["content_type"],
        Body=json.dumps(probe["request"]).encode(),
    )
    with response["Body"] as stream:
        body = json.load(stream)
    if body[probe["version_field"]] != version:
        raise ValueError("Endpoint reported wrong model version")
    rows = body[probe["outputs_field"]]
    if len(rows) != probe["expected_rows"]:
        raise ValueError("Endpoint output row count differs")
    for row in rows:
        if set(row) != set(probe["tasks"]) or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in row.values()
        ):
            raise ValueError("Endpoint output contract differs")


def rollback_step(client, state, binding):
    """Restore the retained EndpointConfig after a failed probe or explicit rollback."""
    active = state.value["active"]
    previous = active.get("previous_config")
    if not previous:
        raise ValueError("no previous Endpoint configuration to restore")
    current = describe(client, "describe_endpoint", EndpointName=binding["endpoint_name"])
    if current is None:
        raise ValueError("Endpoint disappeared during rollback")
    if current["EndpointStatus"] == "InService" and current["EndpointConfigName"] == previous:
        active["stage"] = "rolled_back"
        state.save()
        return "rolled_back"
    if current["EndpointConfigName"] not in {active["endpoint_config"], previous}:
        raise ValueError("Endpoint changed outside this rollback; refusing to overwrite")
    if current["EndpointStatus"] in {"Updating", "RollingBack", "Creating", "SystemUpdating"}:
        return "pending"
    if active.get("rollback_attempted"):
        raise ValueError("rollback outcome unresolved; reconcile before retry")
    active.update(stage="rollback_pending", rollback_attempted=True)
    state.save()
    client.update_endpoint(EndpointName=binding["endpoint_name"], EndpointConfigName=previous)
    return "pending"


def request_rollback(state, client, endpoint, version, operation):
    """Explicit rollback may lower online data date, never the automatic high-water mark."""
    if state.value["active"]:
        if state.value["active"]["operation"] != operation:
            raise ValueError("another release is active")
        return state.value["active"]
    prior = next((r for r in state.value["history"] if r["operation"] == operation), None)
    if prior:
        return prior
    candidates = [
        r
        for r in state.value["history"]
        if r["stage"] == "deployed" and r["model_version"] == version
    ]
    if not candidates:
        raise ValueError("rollback target is not a retained successful release")
    target = candidates[-1]
    current = describe(client, "describe_endpoint", EndpointName=endpoint)
    if current is None or current["EndpointStatus"] != "InService":
        raise ValueError("explicit rollback requires a stable existing Endpoint")
    active = {
        "operation": operation,
        "stage": "rollback_pending",
        "cutoff": target["cutoff"],
        "model_version": version,
        "previous_config": target["endpoint_config"],
        "endpoint_config": current["EndpointConfigName"],
        "rollback_target": {k: target[k] for k in ("model_version", "endpoint_config", "cutoff")},
    }
    state.value["active"] = active
    state.save()
    return active
