"""Pure management contracts. Program resources come only from a trusted group binding."""

import copy
import hashlib
import json
import re
from datetime import date
from urllib.parse import urlsplit

REGISTRY = "doku-model-registry/v1"
GROUP = "doku-model-group/v1"
REQUEST = "doku-model-request/v1"
API = "doku-model-management/v1"
MAX_BYTES = 2 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise ValueError("invalid model/group identifier")
    return value


def exact(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(required) - set(optional)
        or set(required) - value.keys()
    ):
        raise ValueError("management contract fields differ")


def under(uri, prefixes):
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or ".." in parsed.path.split("/")
    ):
        raise ValueError("explicit safe S3 URI required")
    if not any(uri.startswith(p.rstrip("/") + "/") for p in prefixes):
        raise ValueError("URI outside task group resources")
    return uri


def validate_group(group):
    exact(
        group,
        (
            "schema_version",
            "environment",
            "region",
            "name",
            "registry",
            "storage_prefix",
            "kms_key",
            "endpoint_prefix",
            "binding_template",
            "worker",
            "data_source",
        ),
        (
            "adoption_prefixes",
            "read_prefixes",
            "legacy_rules",
            "resource_manager",
            "legacy_ingress",
            "adoption_services",
        ),
    )
    if group["schema_version"] != GROUP:
        raise ValueError("unknown task group contract")
    identifier(group["name"])
    for endpoint, service in group.get("adoption_services", {}).items():
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,62}", endpoint):
            raise ValueError("explicit adopted Endpoint required")
        exact(service, ("execution_role_arn",))
        if not re.fullmatch(
            r"arn:aws:iam::[0-9]{12}:role/[a-zA-Z0-9+=,.@_/-]+", service["execution_role_arn"]
        ):
            raise ValueError("explicit adopted inference role required")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,23}", group["environment"]):
        raise ValueError("explicit environment required")
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,39}-", group["endpoint_prefix"]):
        raise ValueError("bounded endpoint prefix required")
    exact(
        group["registry"],
        ("application_id", "environment_id", "profile_id", "deployment_strategy_id"),
    )
    for value in group["registry"].values():
        if not isinstance(value, str) or not value:
            raise ValueError("explicit AppConfig resources required")
    template = group["binding_template"]
    if template["environment"] != group["environment"] or template["region"] != group["region"]:
        raise ValueError("task group environment mismatch")
    if any(
        k in template
        for k in ("fyp_appconfig", "fyp_split_appconfig", "idea_appconfig", "discovery_appconfig")
    ):
        raise ValueError("managed model service must not own consumer routing")
    for key in ("state_uri", "work_prefix", "model_prefix"):
        if "${model_name}" not in template[key]:
            raise ValueError("per-model storage namespace required: " + key)
    if template["deployment"]["endpoint_name"] != "${endpoint_name}":
        raise ValueError("per-model Endpoint template required")
    request = template["training_request"]
    if request["ResourceConfig"]["InstanceCount"] != 1 or request["ResourceConfig"][
        "InstanceType"
    ].startswith(("ml.g", "ml.p", "ml.inf", "ml.trn")):
        raise ValueError("single-instance CPU training required")
    if "@sha256:" not in request["AlgorithmSpecification"]["TrainingImage"]:
        raise ValueError("immutable training runtime required")
    if group["data_source"]["kind"] not in {"ready_manifest", "idea_mapped"}:
        raise ValueError("unsupported data source")
    worker = group["worker"]
    exact(worker, ("cluster", "task_definition", "container", "network_configuration"))
    if not re.fullmatch(
        r"arn:aws:ecs:[^:]+:[0-9]{12}:task-definition/[^:]+:[0-9]+", worker["task_definition"]
    ):
        raise ValueError("explicit worker revision required")
    return group


def endpoint_name(group, name):
    prefix = group["endpoint_prefix"]
    return prefix + name.replace("_", "-")[: 63 - len(prefix) - 9] + "-" + digest(name)[:8]


def validate_config(config, name):
    from recommend_tf.train.comm.early_stop import validate_early_stop
    from recommend_tf.train.comm.export_tasks import validate_export_tasks
    from recommend_tf.train.comm.learning_rate import validate_schedules
    from recommend_tf.train.models.rerank.model_params import parameters

    if config.get("model_name") != name:
        raise ValueError("config model_name differs from registration")
    if type(config.get("max_epochs")) is not int or config["max_epochs"] < 1:
        raise ValueError("positive max_epochs required")
    if type(config.get("batch_size")) is not int or config["batch_size"] < 1:
        raise ValueError("positive batch_size required")
    validate_export_tasks(config)
    validate_early_stop(config)
    validate_schedules(config["model"], config["max_epochs"])
    parameters(config["model"])
    online = config["online_config"]
    exact(online, ("training_date", "validation_date"), ("first_day",))
    if any(
        type(online[k]) is not int or online[k] <= 0 for k in ("training_date", "validation_date")
    ):
        raise ValueError("positive training/validation windows required")
    if "first_day" in online:
        date.fromisoformat(online["first_day"])


def validate_model(name, model, group):
    identifier(name)
    exact(
        model,
        (
            "group",
            "config",
            "config_revision",
            "service",
            "daily_enabled",
            "admission",
            "first_cutoff",
        ),
        ("adoption",),
    )
    if model["group"] != group["name"] or model["config_revision"] != digest(model["config"]):
        raise ValueError("model group/config revision mismatch")
    validate_config(model["config"], name)
    if type(model["daily_enabled"]) is not bool or model["admission"] not in {"managed", "legacy"}:
        raise ValueError("invalid admission settings")
    date.fromisoformat(model["first_cutoff"])
    exact(model["service"], ("outputs", "probe"), ("broadcast_feat", "identity_signature"))
    if model["service"].get("identity_signature", "serving_with_model_version") not in {
        "serving_with_model_version",
        "serving_default",
    }:
        raise ValueError("unsupported identity signature")
    from recommend_tf.train.comm.export_tasks import publication_outputs

    publication_outputs(model["config"], model["service"])
    if model["service"]["probe"].get("fixture_kind") != "synthetic":
        raise ValueError("synthetic probe required")
    if "adoption" in model:
        exact(model["adoption"], ("binding", "legacy_rule", "evidence_uri"))
        if model["adoption"]["legacy_rule"] not in group.get("legacy_rules", []):
            raise ValueError("legacy rule not enrolled by platform")
        under(model["adoption"]["evidence_uri"], [group["storage_prefix"]])
    from ..sagemaker.runtime_binding import validate_binding

    validate_binding(binding_for(group, name, model), model["config"])
    return model


def validate_registry(registry, group):
    exact(registry, ("schema_version", "environment", "group", "revision", "models"))
    if (
        registry["schema_version"] != REGISTRY
        or registry["environment"] != group["environment"]
        or registry["group"] != group["name"]
    ):
        raise ValueError("registry identity mismatch")
    if (
        type(registry["revision"]) is not int
        or registry["revision"] < 0
        or len(canonical(registry)) >= MAX_BYTES
    ):
        raise ValueError("registry revision/size invalid")
    identities = set()
    states = set()
    for name, model in registry["models"].items():
        validate_model(name, model, group)
        binding = binding_for(group, name, model)
        endpoint = binding["deployment"]["endpoint_name"]
        if endpoint in identities or binding["state_uri"] in states:
            raise ValueError("independent models cannot share Endpoint or release journal")
        identities.add(endpoint)
        states.add(binding["state_uri"])
    return registry


def binding_for(group, name, model):
    """Model knobs cannot replace IAM, networking, runtime image or group resources."""
    template = copy.deepcopy(group["binding_template"])
    endpoint = endpoint_name(group, name)

    def expand(value):
        if isinstance(value, str):
            return value.replace("${model_name}", name).replace("${endpoint_name}", endpoint)
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        return value

    binding = expand(template)
    if adoption := model.get("adoption"):
        old = adoption["binding"]
        # Only verified identity locations survive adoption; privileged runtime stays group-owned.
        for key in ("state_uri", "work_prefix", "model_prefix", "deferred_uri"):
            if key in old:
                under(old[key], group.get("adoption_prefixes", []))
                binding[key] = old[key]
        endpoint = old["deployment"]["endpoint_name"]
        enrolled = group.get("adoption_services", {}).get(endpoint)
        if not endpoint.startswith(group["endpoint_prefix"]) and not enrolled:
            raise ValueError("adopted endpoint outside group IAM boundary")
        expected_role = (enrolled or binding["deployment"]).get("execution_role_arn")
        if old["deployment"].get("execution_role_arn") != expected_role:
            raise ValueError("adopted inference role differs from enrolled group")
        binding["deployment"] = copy.deepcopy(old["deployment"])
    if binding["deployment"]["endpoint_name"] != endpoint:
        raise ValueError("template endpoint identity mismatch")
    binding.update(
        model_name=name, outputs=model["service"]["outputs"], probe=model["service"]["probe"]
    )
    binding.pop("deferred_uri", None)  # Managed request journal replaces the legacy recovery queue.
    binding["managed_model"] = {
        "schema_version": "doku-managed-model/v1",
        "group": group["name"],
        "config_revision": model["config_revision"],
        "identity_signature": model["service"].get(
            "identity_signature", "serving_with_model_version"
        ),
    }
    if "adoption" in model:
        binding["managed_model"]["adoption_evidence_uri"] = model["adoption"]["evidence_uri"]
    if "broadcast_feat" in model["service"]:
        binding["managed_model"]["broadcast_feat"] = model["service"]["broadcast_feat"]
    if "monitoring" in binding:
        binding["monitoring"]["model_name"] = name
    if group["data_source"]["kind"] == "idea_mapped":
        binding["data_source"] = {
            "freeze": "idea_mapped",
            "uri": group["data_source"]["uri"],
            "bootstrap_cutoff": model["first_cutoff"],
        }
    return binding
