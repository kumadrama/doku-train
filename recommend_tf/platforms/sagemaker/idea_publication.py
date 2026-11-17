"""Idea version Endpoint wiring for the shared AppConfig publication driver."""

import re

from .idea_appconfig import active_definition, candidate_documents
from .model_publication import ModelPublication


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


class IdeaPublication(ModelPublication):
    binding_key = "idea_appconfig"
    endpoint_ready_stage = "idea_endpoint_ready"
    create_only = True
    active_definition = staticmethod(active_definition)
    candidate_documents = staticmethod(candidate_documents)

    def deployment(self, version):
        return deployment_for(self.binding, version)
