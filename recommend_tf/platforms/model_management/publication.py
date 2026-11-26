"""Fixed-Endpoint publication guard. Never reads/writes consumer routes or AB."""

import json
from pathlib import Path

from ..sagemaker.publish_model import read
from .contracts import canonical, digest


class ManagedPublication:
    lock = None  # Existing per-model ReleaseState owns Endpoint mutation.
    create_only = False
    endpoint_ready_stage = "deployed"

    def __init__(self, s3, binding):
        self.s3, self.binding = s3, binding

    def acquire(self, state):
        return True

    def release(self, state):
        return True

    def before_endpoint(self, state):
        return True

    def deployment_for_state(self, active):
        self.create_only = active.get("managed_first_deployment", False)
        return self.binding["deployment"]

    def creates_versioned_endpoint(self, deployment):
        return self.create_only

    def prepare(self, state, artifact, work):
        active = state.value["active"]
        signature = json.loads((Path(artifact) / "signature.json").read_text())
        signature.pop("model_version", None)
        contract = {
            "signature": signature,
            "feature_config": json.loads((Path(artifact) / "feature-config.json").read_text()),
        }
        previous = state.value.get("online")
        active["managed_first_deployment"] = previous is None
        prior_contract = state.value.get("managed_contract")
        if prior_contract is None and previous:
            evidence_uri = self.binding["managed_model"].get("adoption_evidence_uri")
            adoption = (
                json.loads(read(self.s3, evidence_uri)[0])["contract"] if evidence_uri else None
            )
            if not adoption:
                raise ValueError("existing Endpoint requires a verified adoption contract")
            raw = read(self.s3, adoption["uri"])[0]
            if digest(json.loads(raw)) != adoption["sha256"]:
                raise ValueError("adopted serving contract digest mismatch")
            prior_contract = json.loads(raw)
        if prior_contract is not None and canonical(prior_contract) != canonical(contract):
            active.update(
                stage="export_contract_failed", error_type="ManagedServingContractMismatch"
            )
            state.save()
            raise ValueError("incompatible same-name service contract; register a new model_name")
        # Anchor at the first attempted export; recovery cannot change the contract silently.
        state.value["managed_contract"] = contract
        active["endpoint_name"] = self.binding["deployment"]["endpoint_name"]
        active["config_revision"] = self.binding["managed_model"]["config_revision"]
        state.save()

    def publish(self, state):
        return "deployed"
