"""Observe retired legacy revisions before replacing the delivery drain barrier."""

from .contracts import digest

PROPAGATION_SECONDS = 600


def verify(documents, group, evidence, name, ecs):
    from .adoption import legacy_pin

    original = evidence["snapshot"]["task_definition"]
    family = original.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    target = ecs.describe_task_definition(taskDefinition=original)["taskDefinition"]
    if target["status"] != "INACTIVE":
        raise ValueError("legacy target revision must be INACTIVE")
    others = []
    for page in ecs.get_paginator("list_task_definitions").paginate(
        familyPrefix=family, status="ACTIVE"
    ):
        for arn in page["taskDefinitionArns"]:
            td = ecs.describe_task_definition(taskDefinition=arn)["taskDefinition"]
            if td["family"] != family:
                continue
            containers = [c for c in td["containerDefinitions"] if c["name"] == "pipeline"]
            if len(containers) != 1:
                raise ValueError("unidentified active legacy revision")
            env = {v["name"]: v["value"] for v in containers[0].get("environment", [])}
            if not all(k in env for k in ("DOKU_CONFIG_URI", "DOKU_CONFIG_SHA256")):
                raise ValueError("unidentified active legacy revision: " + arn)
            config = legacy_pin(
                documents,
                group,
                {"uri": env["DOKU_CONFIG_URI"], "sha256": env["DOKU_CONFIG_SHA256"]},
            )
            if not config.get("model_name") or config["model_name"] == name:
                raise ValueError("active revision can execute adopted model: " + arn)
            others.append({"arn": arn, "model_name": config["model_name"]})
    cluster = group["worker"]["cluster"]
    for page in ecs.get_paginator("list_services").paginate(cluster=cluster):
        arns = page["serviceArns"]
        for start in range(0, len(arns), 10):
            result = ecs.describe_services(cluster=cluster, services=arns[start : start + 10])
            if result.get("failures") or len(result["services"]) != len(arns[start : start + 10]):
                raise ValueError("legacy service inventory incomplete")
            for service in result["services"]:
                definitions = [service.get("taskDefinition", "")]
                definitions += [d.get("taskDefinition", "") for d in service.get("deployments", [])]
                definitions += [d.get("taskDefinition", "") for d in service.get("taskSets", [])]
                if any(d.rsplit("/", 1)[-1].rsplit(":", 1)[0] == family for d in definitions):
                    raise ValueError("legacy family still referenced by service")
    return {"family": family, "target": original, "other_active_revisions_sha256": digest(others)}
