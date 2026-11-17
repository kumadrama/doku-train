"""Idea adapter for the existing FYP-style version endpoint and hot config path."""

import re
from pathlib import Path

from .idea_appconfig import (
    MAX_DOCUMENT_BYTES,
    Publisher,
    active_definition,
    candidate_documents,
    sha256,
)
from .publish_model import immutable_put, location


def deployment_for(binding, version):
    """FYP bindings retain their original deployment identity and behavior."""
    if "idea_appconfig" not in binding:
        return binding["deployment"]
    if binding.get("data_source", {}).get("freeze") != "idea_mapped":
        raise ValueError("Idea AppConfig adapter requires the Idea mapped input")
    if not re.fullmatch("[a-f0-9]{64}", version):
        raise ValueError("exact model version required")
    name = binding["idea_appconfig"]["endpoint_prefix"] + version[:12]
    if len(name) > 63 or not re.fullmatch(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}", name):
        raise ValueError("invalid version endpoint name")
    return {**binding["deployment"], "endpoint_name": name}


class IdeaPublication:
    def __init__(self, s3, publisher, binding):
        self.s3, self.publisher, self.binding = s3, publisher, binding
        self.resources = binding["idea_appconfig"]
        deployment_for(
            binding, "0" * 64
        )  # Reject incorrect domain/resource wiring before training.
        if self.resources["model_list_profile"] == self.resources["models_config_profile"]:
            raise ValueError("Idea profiles must be separate")

    @classmethod
    def from_session(cls, session, s3, binding):
        return cls(
            s3,
            Publisher(
                session.client("appconfig"),
                session.client("appconfigdata"),
                binding["idea_appconfig"],
            ),
            binding,
        )

    def prepare(self, state, artifact, work):
        active = state.value["active"]
        if active.get("online"):
            return  # Frozen documents are already part of the same durable operation.
        listing = self.publisher.read(self.resources["model_list_profile"])
        models = self.publisher.read(self.resources["models_config_profile"])
        _, _, _, old = active_definition(listing, models, self.binding["environment"])
        endpoint = deployment_for(self.binding, active["model_version"])["endpoint_name"]
        if old["release"]["endpoint"] == endpoint:
            raise ValueError("candidate endpoint must be independent of the active model")
        proposed = candidate_documents(
            listing, models, artifact, active["model_uri"], self.binding["environment"], endpoint
        )
        docs = {
            "before_list": listing,
            "before_models": models,
            "after_list": proposed.pop("model_list"),
            "after_models": proposed.pop("models_config"),
        }
        pins = {}
        for name, raw in docs.items():
            path = Path(work) / (name + ".json")
            path.write_bytes(raw)
            uri = (
                self.binding["work_prefix"].rstrip("/")
                + "/"
                + active["operation"]
                + "/appconfig/"
                + sha256(raw)
                + "/"
                + path.name
            )
            immutable_put(self.s3, uri, path, self.binding["kms_key"])
            pins[name] = {"uri": uri, "sha256": sha256(raw)}
        active["online"] = {**proposed, "documents": pins, "endpoint": endpoint}
        active["endpoint_name"] = endpoint
        state.save()

    def publish(self, state):
        """Existing endpoint is ready; definitions first, then the active route."""
        active = state.value["active"]
        if active["stage"] not in {"idea_endpoint_ready", "deployed"}:
            raise ValueError("Idea endpoint must pass the existing deployment probe first")
        journal = active["online"]
        docs = {}
        for name, pin in journal["documents"].items():
            bucket, key = location(pin["uri"])
            response = self.s3.get_object(Bucket=bucket, Key=key)
            with response["Body"] as stream:
                raw = stream.read(MAX_DOCUMENT_BYTES + 1)
            if len(raw) >= MAX_DOCUMENT_BYTES or sha256(raw) != pin["sha256"]:
                raise ValueError("frozen Idea AppConfig content differs")
            docs[name] = raw

        def companion(profile, expected):
            if self.publisher.read(profile) != expected:
                raise ValueError("Idea companion profile changed outside this publication")

        if not journal.get("models_complete"):
            if not self.publisher.step(
                state,
                "models",
                self.resources["models_config_profile"],
                docs["before_models"],
                docs["after_models"],
                companion=lambda: companion(
                    self.resources["model_list_profile"], docs["before_list"]
                ),
            ):
                return "pending"
            journal["models_complete"] = True
            state.save()
        if not self.publisher.step(
            state,
            "route",
            self.resources["model_list_profile"],
            docs["before_list"],
            docs["after_list"],
            companion=lambda: companion(
                self.resources["models_config_profile"], docs["after_models"]
            ),
        ):
            return "pending"
        active["stage"] = "deployed"
        state.save()
        return "deployed"
