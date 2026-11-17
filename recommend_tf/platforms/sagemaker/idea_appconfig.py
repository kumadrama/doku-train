"""Idea model-list/model-definition document contract and guarded publication.

This module consumes versioned JSON contracts, never ai-engine implementation
types. Existing runtime budgets, feature processing and dump settings come from
the captured active definition; only immutable model identity is replaced.
"""

import copy
import hashlib
import json
import time
from pathlib import Path

from recommend_tf.inference.artifact_loader import checksum, load

MAX_DOCUMENT_BYTES = 2 << 20
MODEL_LIST_SCHEMA = "ai-engine/idea-model-list/v1"
MODELS_SCHEMA = "ai-engine/models-config/v1"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    if not 0 < len(raw) < MAX_DOCUMENT_BYTES:
        raise ValueError("Idea AppConfig document exceeds consumer protocol limit")
    return raw


def decode(raw, schema, environment):
    if not 0 < len(raw) < MAX_DOCUMENT_BYTES:
        raise ValueError("Idea AppConfig document size outside consumer protocol")
    value = json.loads(raw)
    if value.get("schema_version") != schema or value.get("environment") != environment:
        raise ValueError("Idea AppConfig schema/environment differs")
    return value


def active_definition(list_raw, models_raw, environment):
    listing = decode(list_raw, MODEL_LIST_SCHEMA, environment)
    models = decode(models_raw, MODELS_SCHEMA, environment)
    if len(listing.get("models", [])) != 1:
        raise ValueError("Idea publisher requires one unambiguous mapped route")
    route = listing["models"][0]
    if route["service_name"] != "idea_tab_rank" or route["config_name"] != "idea_models_config":
        raise ValueError("publisher cannot target a non-Idea consumer")
    key = route["model_key"]
    definition = models["models"][key]
    release = definition["release"]
    if (
        release["id"] != key
        or release["service"] != route["service_name"]
        or release["model"] != route["model_name"]
    ):
        raise ValueError("Idea active route and immutable definition differ")
    prefix = release["artifact_dir"] + "/"
    for name, value in definition["files"].items():
        if (
            not name.startswith(prefix)
            or ".." in Path(name).parts
            or Path(name).is_absolute()
            or sha256(value["content"].encode()) != value["sha256"]
        ):
            raise ValueError("invalid embedded Idea model file")
    return listing, models, key, definition


def candidate_documents(list_raw, models_raw, candidate, model_uri, environment, endpoint):
    """Build bounded old/new definitions while preserving the current template."""
    listing, models, old_key, old = active_definition(list_raw, models_raw, environment)
    artifact = load(candidate)
    root, version = artifact["root"], artifact["version"]
    old_release = old["release"]
    if old_release.get("warmup_file") or old_release.get("pipeline_file"):
        raise ValueError("Idea daily template must use its existing feature-chain files")
    if version == old_release["model_version"]:
        raise ValueError("candidate is already the active immutable model")
    old_prefix = old_release["artifact_dir"] + "/"
    old_config = old["files"][old_prefix + "feature-config.json"]["content"].encode()
    if old_config != (root / "feature-config.json").read_bytes():
        raise ValueError("daily feature contract changed; review its processing chain first")
    old_signature = json.loads(old["files"][old_prefix + "signature.json"]["content"])
    new_signature = copy.deepcopy(artifact["signature"])
    old_signature.pop("model_version", None)
    new_signature.pop("model_version", None)
    if old_signature != new_signature:
        raise ValueError("daily model signature changed; consumer review required")
    template = copy.deepcopy(old_release)
    for key in ("id", "artifact_dir", "model_version", "manifest_sha256"):
        template.pop(key)
    template_digest = sha256(
        encode(
            {
                "release": template,
                "feature_chain": old["feature_chain"],
                "feature_chain_config": models["feature_chains"][old["feature_chain"]],
                "model_call": old["model_call"],
            }
        )
    )
    key = "idea-online-" + version[:16] + "-" + template_digest[:12]
    directory = "mapped-" + version
    new = copy.deepcopy(old)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest_sha = checksum(root / "manifest.json")
    embedded = {}
    for path, previous in old["files"].items():
        relative = path[len(old_prefix) :]
        replacement = root / relative
        raw = replacement.read_bytes() if replacement.is_file() else previous["content"].encode()
        if relative in manifest["files"] and sha256(raw) != manifest["files"][relative]:
            raise ValueError("candidate model file differs from canonical manifest")
        embedded[directory + "/" + relative] = {"sha256": sha256(raw), "content": raw.decode()}
    new["files"] = embedded
    new["release"].update(
        id=key,
        artifact_dir=directory,
        model_version=version,
        manifest_sha256=manifest_sha,
        endpoint=endpoint,
    )
    new["provenance"] = {"source": model_uri, "source_manifest_sha256": manifest_sha}
    lineage_path = root / "lineage.json"
    if lineage_path.is_file():
        lineage = json.loads(lineage_path.read_text())
        for field in ("selected_epoch", "source_digest"):
            if field in lineage:
                new["provenance"][field] = str(lineage[field])
        if lineage.get("input", {}).get("manifest_sha256"):
            new["provenance"]["input_manifest_sha256"] = lineage["input"]["manifest_sha256"]
    if key in models["models"] and models["models"][key] != new:
        raise ValueError("immutable candidate release id already identifies different content")
    # Retain the complete old definition for mixed reads and rollback. Its
    # separate endpoint is never replaced or removed by this publication.
    models["models"] = {old_key: old, key: new}
    listing["models"][0]["model_key"] = key
    return {
        "model_list": encode(listing),
        "models_config": encode(models),
        "previous_release": old_key,
        "candidate_release": key,
        "previous_version": old_release["model_version"],
        "candidate_version": version,
        "manifest_sha256": manifest_sha,
        "template_sha256": template_digest,
    }


class DeploymentRolledBack(ValueError):
    """The recorded deployment has conclusively returned to its prior content."""


class Publisher:
    """One profile tick with durable intent, CAS and response reconciliation."""

    def __init__(self, control, data, binding, *, label_prefix="idea"):
        self.control, self.data, self.binding = control, data, binding
        self.label_prefix = label_prefix
        metadata = getattr(control, "meta", None)
        if (
            metadata is not None
            and "LatestDeploymentNumber"
            not in metadata.service_model.operation_model("StartDeployment").input_shape.members
        ):
            raise ValueError("AppConfig SDK lacks deployment CAS; upgrade the pinned worker SDK")

    def read(self, profile):
        token = self.data.start_configuration_session(
            ApplicationIdentifier=self.binding["application_id"],
            EnvironmentIdentifier=self.binding["environment_id"],
            ConfigurationProfileIdentifier=profile,
        )["InitialConfigurationToken"]
        response = self.data.get_latest_configuration(ConfigurationToken=token)
        with response["Configuration"] as stream:
            raw = stream.read(MAX_DOCUMENT_BYTES + 1)
        if not 0 < len(raw) < MAX_DOCUMENT_BYTES:
            raise ValueError("model configuration outside consumer document limit")
        return raw

    def versions(self, profile, label=None):
        request = {
            "ApplicationId": self.binding["application_id"],
            "ConfigurationProfileId": profile,
            "MaxResults": 50,
        }
        if label:
            request["VersionLabel"] = label
        result = self.control.list_hosted_configuration_versions(**request)
        return result["Items"]

    def _hosted(self, profile, number, expected):
        response = self.control.get_hosted_configuration_version(
            ApplicationId=self.binding["application_id"],
            ConfigurationProfileId=profile,
            VersionNumber=number,
        )
        with response["Content"] as stream:
            raw = stream.read(MAX_DOCUMENT_BYTES + 1)
        if raw != expected:
            raise ValueError("reconciled hosted version has different contents")

    def environment_head(self):
        response = self.control.list_deployments(
            ApplicationId=self.binding["application_id"],
            EnvironmentId=self.binding["environment_id"],
            MaxResults=50,
        )
        items = response["Items"]
        if not items:
            raise ValueError(
                "model publication requires an existing AppConfig environment deployment"
            )
        return max(item["DeploymentNumber"] for item in items)

    def _deployments(self, label):
        request = {
            "ApplicationId": self.binding["application_id"],
            "EnvironmentId": self.binding["environment_id"],
            "MaxResults": 50,
        }
        # The label is unique per profile operation. This scan is needed only
        # when StartDeployment's response was lost before the journal save.
        for _ in range(20):
            response = self.control.list_deployments(**request)
            matches = [d for d in response["Items"] if d.get("VersionLabel") == label]
            if matches:
                if len(matches) != 1:
                    raise ValueError("ambiguous repeated AppConfig deployment")
                return matches[0]["DeploymentNumber"]
            if not response.get("NextToken"):
                return None
            request["NextToken"] = response["NextToken"]
        raise ValueError("deployment reconciliation exceeds bounded metadata scan")

    def step(self, state, slot, profile, before, after, *, companion=None):
        active = state.value["active"]
        publications = active["online"].setdefault("publications", {})
        label = (
            self.label_prefix
            + "-"
            + sha256((active["operation"] + slot).encode())[:20]
            + "-"
            + sha256(after)[:12]
        )
        record = publications.get(slot)
        if record is None:
            head = self.environment_head()
            if companion:
                companion()
            if self.read(profile) != before or self.environment_head() != head:
                raise ValueError("model configuration changed outside this release")
            versions = self.versions(profile)
            if not versions:
                raise ValueError("existing model hosted profile required")
            record = {
                "profile": profile,
                "label": label,
                "before_sha256": sha256(before),
                "after_sha256": sha256(after),
                "latest_version": max(v["VersionNumber"] for v in versions),
                "environment_head": head,
                "created_at_ms": time.time_ns() // 1_000_000,
            }
            publications[slot] = record
            state.save()
        if (
            record["profile"] != profile
            or record["label"] != label
            or record["before_sha256"] != sha256(before)
            or record["after_sha256"] != sha256(after)
        ):
            raise ValueError("publication intent differs from frozen documents")
        if "version" not in record:
            matches = [v for v in self.versions(profile, label) if v.get("VersionLabel") == label]
            if len(matches) > 1:
                raise ValueError("duplicate immutable configuration label")
            if matches:
                record["version"] = matches[0]["VersionNumber"]
                self._hosted(profile, record["version"], after)
            else:
                if companion:
                    companion()
                if self.read(profile) != before:
                    raise ValueError("Idea profile changed before hosting")
                latest = max(v["VersionNumber"] for v in self.versions(profile))
                if latest != record["latest_version"]:
                    raise ValueError("another hosted version appeared during publication")
                # Unique VersionLabel plus LatestVersionNumber makes a lost
                # create response reconcilable without allocating a second ID.
                response = self.control.create_hosted_configuration_version(
                    ApplicationId=self.binding["application_id"],
                    ConfigurationProfileId=profile,
                    LatestVersionNumber=record["latest_version"],
                    Content=after,
                    ContentType="application/json",
                    VersionLabel=label,
                    Description=self.label_prefix
                    + " immutable model publication "
                    + active["operation"],
                )
                record["version"] = response["VersionNumber"]
            state.save()
        if "deployment" not in record:
            found = self._deployments(label)
            if found is not None:
                record["deployment"] = found
                state.save()
            else:
                if companion:
                    companion()
                if self.read(profile) != before:
                    raise ValueError("Idea profile changed before deployment")
                latest = max(v["VersionNumber"] for v in self.versions(profile))
                if latest != record["version"]:
                    raise ValueError("concurrent hosted configuration appeared")
                if self.environment_head() != record["environment_head"]:
                    # Another operation may be unrelated, but cannot be safely
                    # overwritten. Keep the journal and serving bundle intact.
                    raise ValueError("AppConfig environment advanced; reconcile this release")
                record["start_attempted"] = True
                state.save()
                # Server-side deployment-number CAS makes replay of this exact
                # intent safe even after a crash before/after the API call.
                response = self.control.start_deployment(
                    ApplicationId=self.binding["application_id"],
                    EnvironmentId=self.binding["environment_id"],
                    ConfigurationProfileId=profile,
                    ConfigurationVersion=str(record["version"]),
                    DeploymentStrategyId=self.binding["deployment_strategy_id"],
                    LatestDeploymentNumber=record["environment_head"],
                    Description=self.label_prefix + " immutable model publication " + label,
                )
                record["deployment"] = response["DeploymentNumber"]
                state.save()
        response = self.control.get_deployment(
            ApplicationId=self.binding["application_id"],
            EnvironmentId=self.binding["environment_id"],
            DeploymentNumber=record["deployment"],
        )
        if response["ConfigurationProfileId"] != profile or response[
            "ConfigurationVersion"
        ] not in {str(record["version"]), label}:
            raise ValueError("deployment identity differs from recorded intent")
        status = response["State"]
        if status in {"ROLLED_BACK", "REVERTED"}:
            raise DeploymentRolledBack("model configuration deployment rolled back")
        if status != "COMPLETE":
            return False
        if companion:
            companion()
        observed = self.read(profile)
        if observed != after:
            # COMPLETE can precede propagation to AppConfig Data. Only the
            # known prior content under our unchanged deployment is pending.
            if observed == before and self.environment_head() == record["deployment"]:
                return False
            raise ValueError("completed model configuration differs on readback")
        if not record.get("complete"):
            record["complete"] = True
            record["completed_at_ms"] = time.time_ns() // 1_000_000
            state.save()
        return True
