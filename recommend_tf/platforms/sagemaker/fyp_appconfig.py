"""Follow the FYP daily artifact while retaining an independently served rollback."""

import json

from .discovery_appconfig import MODEL, DiscoveryPublication, decode, next_definition
from .idea_appconfig import MODELS_SCHEMA, encode
from .model_publication import ModelPublication

SERVICE = "fyp_episode_rank"
ROLLBACK = MODEL + "__rollback"
LIST_SCHEMA = "ai-engine/fyp-model-list/v1"


def active_definition(list_raw, models_raw, environment):
    listing = decode(list_raw, LIST_SCHEMA, environment)
    models = decode(models_raw, MODELS_SCHEMA, environment)
    routes = [route for route in listing["models"] if route["model_name"] == MODEL]
    if len(routes) != 1:
        raise ValueError("FYP requires exactly one active daily alias")
    route = routes[0]
    definition = route_definition(route, models, environment)
    return listing, models, route["model_key"], definition


def route_definition(route, models, environment):
    definition = models["models"][route["model_key"]]
    release = definition["release"]
    if (
        route["service_name"] != SERVICE
        or route["config_name"] != "models-config-" + environment
        or release["service"] != SERVICE
        or release["model"] != MODEL
        or release["scenes"] != ["for_you"]
        or release.get("require_dump") is not True
        or definition.get("model_call", {}).get("type") != "sagemaker-tfs/v1"
        or not release["endpoint"].startswith(f"doku-{environment}-")
    ):
        raise ValueError("incompatible FYP daily route")
    return definition


def candidate_documents(list_raw, models_raw, candidate, model_uri, environment, endpoint):
    listing, models, old_key, old = active_definition(list_raw, models_raw, environment)
    routes = listing["models"]
    active = next(route for route in routes if route["model_name"] == MODEL)
    rollback = [route for route in routes if route["model_name"] == ROLLBACK]
    if len(rollback) > 1:
        raise ValueError("ambiguous FYP rollback alias")
    if not rollback:
        rollback = [{**active, "model_name": ROLLBACK}]
    rollback = rollback[0]
    backup = route_definition(rollback, models, environment)
    if backup["release"]["endpoint"] == endpoint:
        raise ValueError("FYP rollback must use an independent endpoint")
    old_feature = old["release"]["artifact_dir"] + "/feature-config.json"
    backup_feature = backup["release"]["artifact_dir"] + "/feature-config.json"
    if old["files"][old_feature] != backup["files"][backup_feature]:
        raise ValueError("FYP rollback has a different feature contract")
    signatures = []
    for definition in (old, backup):
        path = definition["release"]["artifact_dir"] + "/signature.json"
        signature = json.loads(definition["files"][path]["content"])
        signature.pop("model_version", None)
        signatures.append(signature)
    if signatures[0] != signatures[1]:
        raise ValueError("FYP rollback has a different scoring signature")

    key, new, report = next_definition(models, old, candidate, model_uri, endpoint, "fyp")
    if key in models["models"] and models["models"][key] != new:
        raise ValueError("immutable FYP release already has different content")
    if key == old_key:
        return {**report, "model_list": list_raw, "models_config": models_raw}

    listing["models"] = [{**active, "model_key": key} if r == active else r for r in routes]
    if rollback not in listing["models"]:
        listing["models"].append(rollback)

    # A definition on the shared official endpoint can only serve until the
    # next endpoint update. A rollback always uses the independent endpoint.
    if old["release"]["endpoint"] == endpoint:
        fallback_routes = [
            {**active, "model_key": rollback["model_key"]} if r == active else r for r in routes
        ]
        if rollback not in fallback_routes:
            fallback_routes.append(rollback)
        report["fallback_list"] = encode({**listing, "models": fallback_routes})
        report["fallback_release"] = backup["release"]["id"]
    definitions = models["models"]
    retained = {route["model_key"] for route in (*routes, rollback)}
    for model_key, definition in list(definitions.items()):
        release = definition["release"]
        if (
            model_key not in retained
            and model_key.startswith("fyp-")
            and release["service"] == SERVICE
            and release["endpoint"] == endpoint
        ):
            del definitions[model_key]
    definitions[key] = new
    return {**report, "model_list": encode(listing), "models_config": encode(models)}


class FypPublication(ModelPublication):
    binding_key = "fyp_appconfig"
    journal_key = "fyp_online"
    endpoint_ready_stage = DiscoveryPublication.endpoint_ready_stage
    create_only = False
    active_definition = staticmethod(active_definition)
    candidate_documents = staticmethod(candidate_documents)

    def __init__(self, s3, publisher, binding, *, ecs, logs):
        self.ecs, self.logs = ecs, logs
        super().__init__(s3, publisher, binding)

    @classmethod
    def from_session(cls, session, s3, binding):
        from .idea_appconfig import Publisher

        return cls(
            s3,
            Publisher(
                session.client("appconfig"),
                session.client("appconfigdata"),
                binding[cls.binding_key],
                label_prefix="fyp",
            ),
            binding,
            ecs=session.client("ecs"),
            logs=session.client("logs"),
        )

    def deployment(self, version):
        return DiscoveryPublication.deployment(self, version)

    def before_endpoint(self, state):
        journal = state.value["active"][self.journal_key]
        if "fallback_release" not in journal:
            return True
        docs = self.read_documents(journal)
        models = self.resources["models_config_profile"]

        def companion():
            if self.publisher.read(models) != docs["before_models"]:
                raise ValueError("FYP definitions changed before endpoint rotation")

        if not self.publisher.step(
            state,
            "protect",
            self.resources["model_list_profile"],
            docs["before_list"],
            docs["fallback_list"],
            companion=companion,
            journal_key=self.journal_key,
        ):
            return False
        record = journal["publications"]["protect"]
        return self.rollback_loaded(record["created_at_ms"], journal["fallback_release"])

    def rollback_loaded(self, since, release):
        environment = self.binding["environment"]
        cluster = self.binding["ecs_cluster"]
        service_name = self.resources.get("ecs_service", "doku-ai-engine")
        response = self.ecs.describe_services(cluster=cluster, services=[service_name])
        if response.get("failures") or len(response["services"]) != 1:
            raise ValueError("cannot verify FYP serving replicas before endpoint rotation")
        service = response["services"][0]
        arns = [
            arn
            for page in self.ecs.get_paginator("list_tasks").paginate(
                cluster=cluster, serviceName=service_name, desiredStatus="RUNNING"
            )
            for arn in page["taskArns"]
        ]
        if not 0 < service["desiredCount"] == service["runningCount"] == len(arns):
            return False
        tasks = {arn.rsplit("/", 1)[-1] for arn in arns}
        latest = {}
        for page in self.logs.get_paginator("filter_log_events").paginate(
            logGroupName=f"/ecs/doku-{environment}/{service_name}",
            startTime=since,
            filterPattern='{ $.event = "ai_engine.model_generation_ready" }',
        ):
            for event in page["events"]:
                task = event["logStreamName"].rsplit("/", 1)[-1]
                if task not in tasks or task in latest and latest[task][0] > event["timestamp"]:
                    continue
                row = json.loads(event["message"])
                latest[task] = (
                    event["timestamp"],
                    row.get("model_routes", {}).get(SERVICE, {}).get(MODEL),
                )
        return all(latest.get(task, (0, None))[1] == release for task in tasks)


class FypAndDiscoveryPublication(DiscoveryPublication):
    """Publish the same probed artifact to both workloads in one durable operation."""

    @classmethod
    def from_session(cls, session, s3, binding):
        publication = super().from_session(session, s3, binding)
        publication.fyp = FypPublication.from_session(session, s3, binding)
        for key in ("application_id", "environment_id"):
            if publication.resources[key] != publication.fyp.resources[key]:
                raise ValueError("FYP and Discovery Config must share the serving environment")
        if {
            publication.resources["model_list_profile"],
            publication.resources["models_config_profile"],
        } & {
            publication.fyp.resources["model_list_profile"],
            publication.fyp.resources["models_config_profile"],
        }:
            raise ValueError("FYP and Discovery require independent AppConfig profiles")
        return publication

    def prepare(self, state, artifact, work):
        self.fyp.prepare(state, artifact, work)
        super().prepare(state, artifact, work)

    def before_endpoint(self, state):
        return self.fyp.before_endpoint(state)

    def publish(self, state, *, finalize=True):
        # Discovery was using the shared endpoint during rotation. Publish its
        # matching release first while FYP still serves the rollback model.
        journal = state.value["active"][self.journal_key]
        if not journal.get("released"):
            if super().publish(state, finalize=False) != "deployed":
                return "pending"
            journal["released"] = True
            state.save()
        return self.fyp.publish(state, finalize=finalize)
