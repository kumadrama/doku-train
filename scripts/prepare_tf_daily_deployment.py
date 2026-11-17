"""Prepare a TF daily task's immutable monitoring binding for its existing deployment owner."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def bind_daily(config, binding, task, registration):
    family = task["family"]
    if registration["schema_version"] != "doku-tf-daily-deployment/v1":
        raise ValueError("unsupported daily deployment contract")
    entry = registration["families"][family]
    if (registration["environment"], registration["region"]) != (
        binding["environment"],
        binding["region"],
    ):
        raise ValueError("daily task and monitoring environment differ")
    role = task["taskRoleArn"]
    expected_arn = f"arn:aws:iam::{registration['account_id']}:role/{entry['role']}"
    if role not in {entry["role"], expected_arn}:
        raise ValueError("daily task role differs from registered monitoring producer")
    name = config["model_name"]
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or binding.get("model_name", name) != name:
        raise ValueError("invalid or conflicting model_name")
    run_prefix = binding["work_prefix"].rstrip("/") + "/"
    if "runs_prefix" in binding:
        run_prefix = (
            binding["runs_prefix"].rstrip("/") + "/" + binding.get("job_name_prefix", "doku-daily-")
        )
    sources = entry["sources"]
    if not any(
        s["kind"] in {"runs", "daily_runs"}
        and run_prefix.startswith(
            s["uri"] if s["kind"] == "daily_runs" else s["uri"].rstrip("/") + "/"
        )
        for s in sources
    ):
        raise ValueError("daily runs are outside registered monitoring sources")
    if not any(
        s["kind"] == "journals" and binding["state_uri"].startswith(s["uri"].rstrip("/") + "/")
        for s in sources
    ):
        raise ValueError("daily release journal is outside registered monitoring sources")
    options = {
        key: registration["monitoring"][key] for key in ("outbox_uri", "catalog_uri", "kms_key")
    }
    if not all(options.values()) or not all(
        options[k].startswith("s3://") for k in ("outbox_uri", "catalog_uri")
    ):
        raise ValueError("incomplete monitoring destinations")
    return dict(
        binding,
        monitoring={
            **binding.get("monitoring", {}),
            **options,
            "task_family": family,
            "model_name": name,
        },
    )


def prepare_daily(args):
    config_raw = Path(args.config).read_bytes()
    config = json.loads(config_raw)
    binding = json.loads(Path(args.binding).read_bytes())
    task = json.loads(Path(args.task_definition).read_bytes())
    registration = json.loads(Path(args.monitoring_outputs).read_bytes())["tf_daily_deployment"][
        "value"
    ]
    binding = bind_daily(config, binding, task, registration)
    if getattr(args, "online_release_outputs", None):
        output = json.loads(Path(args.online_release_outputs).read_bytes())[
            "discovery_online_release_binding"
        ]["value"]
        if (output["environment"], output["region"]) != (
            binding["environment"],
            binding["region"],
        ) or "idea_appconfig" in binding:
            raise ValueError("Discovery publication binding differs from this daily worker")
        resources = output["discovery_appconfig"]
        required = {
            "application_id",
            "environment_id",
            "deployment_strategy_id",
            "model_list_profile",
            "models_config_profile",
        }
        if (
            set(resources) != required
            or not all(
                isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._-]+", value)
                for value in resources.values()
            )
            or resources["model_list_profile"] == resources["models_config_profile"]
        ):
            raise ValueError("invalid Discovery publication resources")
        if "discovery_appconfig" in binding and binding["discovery_appconfig"] != resources:
            raise ValueError("Discovery publication resources already differ")
        binding["discovery_appconfig"] = resources
    prefix = args.config_prefix.rstrip("/")
    if not prefix.startswith("s3://") or not prefix[5:].strip("/"):
        raise ValueError("configuration S3 prefix required")
    raw = (json.dumps(binding, sort_keys=True, indent=2) + "\n").encode()
    files = {"config": ("daily-config.json", config_raw), "binding": ("binding.json", raw)}
    pins = {}
    for key, (name, body) in files.items():
        sha = hashlib.sha256(body).hexdigest()
        pins["DOKU_" + key.upper() + "_SHA256"] = sha
        pins["DOKU_" + key.upper() + "_URI"] = f"{prefix}/{sha}/{name}"
    containers = [
        c
        for c in task["containerDefinitions"]
        if any(e["name"] == "DOKU_BINDING_URI" for e in c.get("environment", []))
    ]
    if len(containers) != 1:
        raise ValueError("exactly one daily executor container must own the frozen pins")
    env = containers[0]["environment"]
    if any(sum(e["name"] == key for e in env) != 1 for key in pins):
        raise ValueError("daily executor requires unique config and binding pins")
    for field in env:
        if field["name"] in pins:
            field["value"] = pins[field["name"]]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for name, body in files.values():
        (root / name).write_bytes(body)
    (root / "task-definition.yml").write_text(json.dumps(task, indent=2) + "\n")
    return pins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "config",
        "binding",
        "task-definition",
        "monitoring-outputs",
        "config-prefix",
        "output",
    ):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--online-release-outputs", help="applied IaC Discovery publisher output")
    print(json.dumps(prepare_daily(parser.parse_args()), indent=2))
