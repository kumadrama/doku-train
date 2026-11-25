"""Follow the FYP daily artifact while retaining an independently served rollback."""

import json
import re
import time
from pathlib import Path

from recommend_tf.inference import official
from recommend_tf.train.comm.export_tasks import DISCOVERY_SERVING_SIGNATURE

from .discovery_appconfig import (
    ENDPOINT_CALLER,
    MODEL,
    DiscoveryPublication,
    decode,
    next_definition,
    serving_identity_signature,
)
from .idea_appconfig import MODELS_SCHEMA, DeploymentRolledBack, encode
from .model_publication import ModelPublication

SERVICE = "fyp_episode_rank"
ROLLBACK = MODEL + "__rollback"
LIST_SCHEMA = "ai-engine/fyp-model-list/v1"
SPLIT_MODELS = {
    "inner_watch_lhuc_m512_e16_t64",
    "inner_watch_lr003",
    "inner_watch_w02_lr003",
    "tf_rerank_ln_lr002_3feat_us_or_unknown",
}


def split_endpoint_prefix(base):
    # Keep room for the version under the SageMaker 63-character limit.
    return base[:48] + "-v"


def active_definition(list_raw, models_raw, environment):
    listing = decode(list_raw, LIST_SCHEMA, environment)
    models = decode(models_raw, MODELS_SCHEMA, environment)
    routes = [route for route in listing["models"] if route["model_name"] == MODEL]
    if len(routes) != 1:
        raise ValueError("FYP requires exactly one active daily alias")
    route = routes[0]
    definition = route_definition(route, models, environment)
    return listing, models, route["model_key"], definition


def route_definition(route, models, environment, model_name=MODEL):
    definition = models["models"][route["model_key"]]
    release = definition["release"]
    if (
        route["service_name"] != SERVICE
        or route["config_name"] != "models-config-" + environment
        or release["service"] != SERVICE
        or release["model"] != model_name
        or release["scenes"] != ["for_you"]
        or release.get("require_dump") is not True
        or definition.get("model_call", {}).get("type") not in {"sagemaker-tfs/v1", ENDPOINT_CALLER}
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

    key, new, report = next_definition(
        models, old, candidate, model_uri, endpoint, "fyp", endpoint_identity=True
    )
    if old["model_call"]["type"] == ENDPOINT_CALLER and old["release"]["endpoint"] == endpoint:
        report.update(candidate_release=old_key, reuse_route=True)
        return {**report, "model_list": list_raw, "models_config": models_raw}
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
            if journal.get("serving_signature"):
                docs = self.read_documents(journal)
                if (
                    self.publisher.read(self.resources["model_list_profile"]) != docs["before_list"]
                    or self.publisher.read(self.resources["models_config_profile"])
                    != docs["before_models"]
                ):
                    raise ValueError("FYP routes changed before endpoint deployment")
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

    def rollback_loaded(self, since, release, *, model_name=MODEL):
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
                    row.get("model_routes", {}).get(SERVICE, {}).get(model_name),
                )
        return all(latest.get(task, (0, None))[1] == release for task in tasks)


class SplitFypPublication(FypPublication):
    """Migrate one FYP experiment to its dedicated fixed endpoint."""

    binding_key = "fyp_split_appconfig"
    journal_key = "fyp_split_online"
    endpoint_ready_stage = "fyp_split_endpoint_ready"
    create_only = True
    readiness_timeout_seconds = 300

    def deployment(self, version):
        model = self.binding.get("model_name")
        environment = self.binding["environment"]
        base = self.binding["deployment"]
        if (
            model not in SPLIT_MODELS
            or environment not in {"dte", "prod"}
            or self.lock is None
            or official.protocol(base) != official.PROTOCOL
            or base["endpoint_name"] != f"doku-{environment}-fyp-{model.replace('_', '-')}"
        ):
            raise ValueError("split FYP publication requires a supported model and shared lock")
        endpoint = split_endpoint_prefix(base["endpoint_name"]) + version[:12]
        if len(endpoint) > 63 or not re.fullmatch(r"[a-f0-9]{64}", version):
            raise ValueError("exact version and valid split FYP endpoint required")
        return {**base, "endpoint_name": endpoint}

    def candidate_deployment(self, version, old, artifact):
        base = self.binding["deployment"]
        if old["model_call"]["type"] == ENDPOINT_CALLER:
            if old["release"]["endpoint"] != base["endpoint_name"]:
                raise ValueError("fixed FYP route differs from its configured Endpoint")
            return base
        if (
            old["release"]["endpoint"] != base["endpoint_name"]
            and serving_identity_signature(Path(artifact) / "manifest.json")
            == DISCOVERY_SERVING_SIGNATURE
        ):
            return base
        # A pinned release on the base needs one normal immutable cutover first.
        # The next candidate can then be probed on the base before changing the route.
        return self.deployment(version)

    def deployment_for_state(self, active):
        base = self.binding["deployment"]
        pinned = active[self.journal_key]["endpoint"]
        if pinned == base["endpoint_name"]:
            return base
        versioned = self.deployment(active["model_version"])
        if pinned != versioned["endpoint_name"]:
            raise ValueError("frozen split FYP Endpoint differs from its binding")
        return versioned

    def creates_versioned_endpoint(self, deployment):
        return deployment["endpoint_name"] != self.binding["deployment"]["endpoint_name"]

    def active_definition(self, list_raw, models_raw, environment):
        listing = decode(list_raw, LIST_SCHEMA, environment)
        models = decode(models_raw, MODELS_SCHEMA, environment)
        name = self.binding["model_name"]
        aliases = [row for row in listing["models"] if row["model_name"] == name]
        if len(aliases) != 1:
            raise ValueError("split FYP requires exactly one active alias")
        alias = aliases[0]
        old = route_definition(alias, models, environment, model_name=name)
        if old["release"]["id"] != alias["model_key"]:
            raise ValueError("split FYP alias points to a different immutable definition")
        return listing, models, alias["model_key"], old

    def candidate_documents(
        self, list_raw, models_raw, candidate, model_uri, environment, endpoint
    ):
        listing, models, old_key, old = self.active_definition(list_raw, models_raw, environment)
        fixed_endpoint = self.binding["deployment"]["endpoint_name"]
        key, definition, report = next_definition(
            models,
            old,
            candidate,
            model_uri,
            endpoint,
            "fyp-daily",
            endpoint_identity=endpoint == fixed_endpoint,
        )
        if old["model_call"]["type"] == ENDPOINT_CALLER:
            report.update(candidate_release=old_key, reuse_route=True)
            return {**report, "model_list": list_raw, "models_config": models_raw}
        retained = {route["model_key"] for route in listing["models"]}
        for name, previous in list(models["models"].items()):
            release = previous["release"]
            if (
                name not in retained
                and release["model"] == self.binding["model_name"]
                and release["service"] == SERVICE
                and (
                    (
                        name.startswith("fyp-daily-")
                        and release["endpoint"].startswith(split_endpoint_prefix(fixed_endpoint))
                    )
                    or (
                        release["endpoint"] == old["release"]["endpoint"] == fixed_endpoint
                        and release.get("model_version") != old["release"].get("model_version")
                    )
                )
            ):
                del models["models"][name]
        models["models"][key] = definition
        for route in listing["models"]:
            if route["model_name"] == self.binding["model_name"]:
                route["model_key"] = key
        return {**report, "model_list": encode(listing), "models_config": encode(models)}

    def publish(self, state):
        active = state.value["active"]
        journal = active[self.journal_key]
        if journal.get("reuse_route"):
            return super().publish(state)
        docs = self.read_documents(journal)
        deployments = journal.get("publications", {})
        route = deployments.get("route", {})
        model_profile = self.resources["models_config_profile"]
        list_profile = self.resources["model_list_profile"]

        def verify_definitions():
            if self.publisher.read(model_profile) != docs["after_models"]:
                raise ValueError("FYP model definitions changed before rollback")

        if not route.get("complete"):
            try:
                if super().publish(state, finalize=False) != "deployed":
                    return "pending"
            except DeploymentRolledBack:
                if self.publisher.read(list_profile) != docs["before_list"]:
                    raise
                active["stage"] = "cutover_rolled_back"
                state.save()
                return active["stage"]
            route = journal["publications"]["route"]

        now = time.time_ns() // 1_000_000
        if not journal.get("rollback_requested_at_ms"):
            verify_definitions()
            if self.publisher.read(list_profile) != docs["after_list"]:
                raise ValueError("FYP alias changed before cutover verification")
            if self.rollback_loaded(
                route["created_at_ms"],
                journal["candidate_release"],
                model_name=self.binding["model_name"],
            ):
                active["stage"] = "deployed"
                state.save()
                return active["stage"]
            if now - route["completed_at_ms"] < self.readiness_timeout_seconds * 1000:
                return "pending"
            journal["rollback_requested_at_ms"] = now
            state.save()

        if not self.publisher.step(
            state,
            "rollback",
            list_profile,
            docs["after_list"],
            docs["before_list"],
            companion=verify_definitions,
            journal_key=self.journal_key,
        ):
            return "pending"
        rollback = journal["publications"]["rollback"]
        if not self.rollback_loaded(
            rollback["created_at_ms"],
            journal["previous_release"],
            model_name=self.binding["model_name"],
        ):
            return "pending"
        active["stage"] = "cutover_rolled_back"
        state.save()
        return active["stage"]


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
