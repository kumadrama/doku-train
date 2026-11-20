"""Idea version Endpoint wiring for the shared AppConfig publication driver."""

import re

from recommend_tf.inference import official

from .idea_appconfig import (
    active_definition,
    candidate_documents,
    compatible_candidate,
    encode,
    sha256,
)
from .model_publication import ModelPublication


def deployment_for(binding, version):
    """FYP bindings retain their original deployment identity and behavior."""
    if "idea_appconfig" not in binding:
        return binding["deployment"]
    if binding.get("data_source", {}).get("freeze") != "idea_mapped":
        raise ValueError("Idea AppConfig adapter requires the Idea mapped input")
    if not re.fullmatch("[a-f0-9]{64}", version):
        raise ValueError("exact model version required")
    options = binding["idea_appconfig"]
    deployment = binding["deployment"]
    if options.get("fixed_endpoint") is True:
        if (
            not re.fullmatch(
                rf"doku-{re.escape(binding['environment'])}-idea-mapped(?:-v-[0-9a-f]{{12}})?",
                deployment["endpoint_name"],
            )
            or official.protocol(deployment) != official.PROTOCOL
        ):
            raise ValueError("fixed Idea publication requires the mapped TensorFlow Endpoint")
        return deployment
    name = options["endpoint_prefix"] + version[:12]
    if len(name) > 63 or not re.fullmatch(r"[A-Za-z0-9](?:-*[A-Za-z0-9]){0,62}", name):
        raise ValueError("invalid version endpoint name")
    return {**binding["deployment"], "endpoint_name": name}


class IdeaPublication(ModelPublication):
    binding_key = "idea_appconfig"
    endpoint_ready_stage = "idea_endpoint_ready"
    active_definition = staticmethod(active_definition)
    candidate_documents = staticmethod(candidate_documents)

    def deployment(self, version):
        return deployment_for(self.binding, version)

    @property
    def fixed_endpoint(self):
        return self.resources.get("fixed_endpoint") is True

    @property
    def create_only(self):
        return not self.fixed_endpoint

    def fixed_route(self):
        listing = self.publisher.read(self.resources["model_list_profile"])
        models = self.publisher.read(self.resources["models_config_profile"])
        _, config, key, definition = self.active_definition(
            listing, models, self.binding["environment"]
        )
        endpoint = self.binding["deployment"]["endpoint_name"]
        if definition["release"]["endpoint"] != endpoint:
            raise ValueError("online Idea route has not been migrated to its fixed Endpoint")
        digest = sha256(
            encode(
                {
                    "release": definition,
                    "feature_chain": config["feature_chains"][definition["feature_chain"]],
                }
            )
        )
        return {"release": key, "sha256": digest, "endpoint": endpoint}, definition

    def prepare(self, state, artifact, work):
        if not self.fixed_endpoint:
            return super().prepare(state, artifact, work)
        active = state.value["active"]
        route, definition = self.fixed_route()
        candidate, _ = compatible_candidate(definition, artifact)
        if candidate["version"] != active["model_version"]:
            raise ValueError("fixed Idea candidate differs from the journal")
        if active.get("fixed_idea_route", route) != route:
            raise ValueError("fixed Idea route changed during export")
        active["fixed_idea_route"] = route
        active["endpoint_name"] = route["endpoint"]
        state.save()

    def before_endpoint(self, state):
        if self.fixed_endpoint and state.value["active"]["stage"] == "uploaded":
            route, _ = self.fixed_route()
            if state.value["active"].get("fixed_idea_route") != route:
                raise ValueError("fixed Idea route changed after export")
        return True
