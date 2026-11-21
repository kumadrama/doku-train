"""Adapt an existing FYP daily artifact to the Discovery AppConfig contract."""

import copy
import json
from pathlib import Path

import yaml

from recommend_tf.inference import official
from recommend_tf.inference.artifact_loader import checksum, load
from recommend_tf.train.comm.export_tasks import DISCOVERY_SERVING_SIGNATURE

from .idea_appconfig import MAX_DOCUMENT_BYTES, MODELS_SCHEMA, encode, sha256
from .model_publication import ModelPublication

ALIAS = "discovery_fyp_nn"
BOOTSTRAP_ALIAS = "discovery_bootstrap_nn"
SERVICE = "discovery_episode_rank"
MODEL = "tf_rerank_ln_lr002_3feat"
LIST_SCHEMA = "ai-engine/discovery-model-list/v1"
MODELS_PROFILE = "discovery_models_config"
ENDPOINT_CALLER = "sagemaker-tfs-identified/v1"


def decode(raw, schema, environment):
    if not 0 < len(raw) < MAX_DOCUMENT_BYTES:
        raise ValueError("model configuration size outside consumer protocol")
    value = yaml.safe_load(raw)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != schema
        or value.get("environment") != environment
    ):
        raise ValueError("Discovery model schema or environment differs")
    return value


def active_definition(list_raw, models_raw, environment):
    listing = decode(list_raw, LIST_SCHEMA, environment)
    models = decode(models_raw, MODELS_SCHEMA, environment)
    routes = listing["models"]
    if [r["model_name"] for r in routes] not in ([ALIAS], [ALIAS, BOOTSTRAP_ALIAS]):
        raise ValueError(
            "Discovery model list requires its active alias and optional bootstrap alias"
        )
    if models.get("inference_dump", {}).get("enabled") or models.get("training_dump") is not None:
        raise ValueError("Discovery configuration cannot write FYP dumps")
    for route in routes:
        definition = models["models"][route["model_key"]]
        release = definition["release"]
        if (
            route["service_name"] != SERVICE
            or route["config_name"] != MODELS_PROFILE
            or release["service"] != SERVICE
            or release["model"] != MODEL
            or release["scenes"] != ["discovery"]
            or release.get("require_dump") is not True
        ):
            raise ValueError("Discovery route and release identity differ")
        directory = release["artifact_dir"] + "/"
        for name, file in definition["files"].items():
            if (
                not name.startswith(directory)
                or ".." in Path(name).parts
                or Path(name).is_absolute()
                or sha256(file["content"].encode()) != file["sha256"]
            ):
                raise ValueError("invalid embedded Discovery model file")
    key = routes[0]["model_key"]
    return listing, models, key, models["models"][key]


def next_definition(models, old, candidate, model_uri, endpoint, release_prefix):
    artifact = load(candidate)
    root, version = artifact["root"], artifact["version"]
    previous = old["release"]
    caller = old.get("model_call", {}).get("type")
    legacy_caller = "sagemaker-tfs/v1"
    discovery = release_prefix == "discovery"
    if previous.get("pipeline_file") or (
        caller != legacy_caller and not (discovery and caller == ENDPOINT_CALLER)
    ):
        raise ValueError("Discovery daily model requires its existing TF Serving feature chain")
    old_prefix = previous["artifact_dir"] + "/"
    if (
        old["files"][old_prefix + "feature-config.json"]["content"].encode()
        != (root / "feature-config.json").read_bytes()
    ):
        raise ValueError("daily feature config changed; review the feature chain")
    previous_signature = json.loads(old["files"][old_prefix + "signature.json"]["content"])
    candidate_signature = copy.deepcopy(artifact["signature"])
    previous_signature.pop("model_version", None)
    candidate_signature.pop("model_version", None)
    if previous_signature != candidate_signature:
        raise ValueError("daily model signature changed; review Discovery scoring")

    manifest_path = root / "manifest.json"
    manifest_sha = checksum(manifest_path)
    model_call = old["model_call"]
    if discovery:
        manifest = json.loads(manifest_path.read_bytes())
        identified = (
            manifest.get("verification", {}).get("serving_version_signature")
            == DISCOVERY_SERVING_SIGNATURE
        )
        if caller == ENDPOINT_CALLER and not identified:
            raise ValueError("fixed Discovery Endpoint requires the model identity signature")
        if identified:
            model_call = {**model_call, "type": ENDPOINT_CALLER}
    previous_version = previous.get("model_version")
    if previous_version is None:
        embedded = old["files"][old_prefix + "manifest.json"]
        if embedded["sha256"] != previous["manifest_sha256"]:
            raise ValueError("Discovery contract manifest differs from its pinned digest")
        previous_version = json.loads(embedded["content"])["model_version"]
    template = copy.deepcopy(previous)
    for name in ("id", "model_version", "manifest_sha256", "artifact_dir", "warmup_file"):
        template.pop(name, None)
    template["endpoint"] = endpoint
    identity = sha256(
        encode(
            {
                "release": template,
                "manifest": manifest_sha,
                "feature_chain": models["feature_chains"][old["feature_chain"]],
                "model_call": model_call,
            }
        )
    )
    key = release_prefix + "-" + version[:16] + "-" + identity[:12]
    directory = "models/" + key
    new = copy.deepcopy(old)
    new["model_call"] = model_call
    new["files"] = {}
    for filename in (
        "manifest.json",
        "feature-config.json",
        "signature.json",
        "feature-state.json",
    ):
        raw = (root / filename).read_bytes()
        new["files"][directory + "/" + filename] = {"content": raw.decode(), "sha256": sha256(raw)}
    new["release"].update(
        id=key,
        model_version=version,
        manifest_sha256=manifest_sha,
        artifact_dir=directory,
        endpoint=endpoint,
    )
    if model_call["type"] == ENDPOINT_CALLER:
        new["release"].pop("model_version")
    if warmup := previous.get("warmup_file"):
        target = directory + "/warmup.json"
        new["files"][target] = copy.deepcopy(old["files"][warmup])
        new["release"]["warmup_file"] = target
    else:
        raise ValueError("Discovery model requires its existing synthetic warmup")
    new["provenance"] = {"model_data_url": model_uri, "source_manifest_sha256": manifest_sha}
    if key in models["models"] and models["models"][key] != new:
        raise ValueError("immutable model release already has different content")
    proposed = {
        "previous_release": previous["id"],
        "candidate_release": key,
        "previous_version": previous_version,
        "candidate_version": version,
        "manifest_sha256": manifest_sha,
        "template_sha256": identity,
    }
    return key, new, proposed


def candidate_documents(list_raw, models_raw, candidate, model_uri, environment, endpoint):
    listing, models, old_key, old = active_definition(list_raw, models_raw, environment)
    key, new, proposed = next_definition(models, old, candidate, model_uri, endpoint, "discovery")
    if old_key == key or (
        old["model_call"]["type"] == ENDPOINT_CALLER and old["release"]["endpoint"] == endpoint
    ):
        proposed["candidate_release"] = old_key
        return {**proposed, "model_list": list_raw, "models_config": models_raw}

    # Keep definitions referenced by a cached old list during the two updates.
    models["models"] = {r["model_key"]: models["models"][r["model_key"]] for r in listing["models"]}
    models["models"][key] = new
    route = listing["models"][0]
    bootstrap = next((r for r in listing["models"] if r["model_name"] == BOOTSTRAP_ALIAS), None)
    if bootstrap is None and old["release"]["endpoint"] != endpoint:
        bootstrap = {**route, "model_name": BOOTSTRAP_ALIAS}
    listing["models"] = [{**route, "model_key": key}]
    if bootstrap is not None:
        listing["models"].append(bootstrap)
    return {**proposed, "model_list": encode(listing), "models_config": encode(models)}


class DiscoveryPublication(ModelPublication):
    binding_key = "discovery_appconfig"
    endpoint_ready_stage = "discovery_endpoint_ready"
    create_only = False
    active_definition = staticmethod(active_definition)
    candidate_documents = staticmethod(candidate_documents)

    def deployment(self, version):
        environment = self.binding["environment"]
        deployment = self.binding["deployment"]
        if (
            environment not in {"dte", "prod"}
            or self.binding.get("model_name") != MODEL
            or deployment["endpoint_name"] != f"doku-{environment}-tf-rerank-official"
            or official.protocol(deployment) != official.PROTOCOL
        ):
            raise ValueError(
                "Discovery AppConfig publication requires the existing official FYP daily model"
            )
        return deployment
