"""Freeze and publish model documents with the existing AppConfig CAS journal."""

import json
from pathlib import Path

from .idea_appconfig import MAX_DOCUMENT_BYTES, Publisher, sha256
from .publish_model import immutable_put, location, read


class PublicationLock:
    """One durable owner for coordinated FYP profile writes across workers."""

    def __init__(self, s3, uri, kms_key):
        self.s3, self.uri, self.kms_key = s3, uri, kms_key
        self.bucket, self.key = location(uri)

    def write(self, owner, etag=None):
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self.key,
                Body=json.dumps(owner, sort_keys=True).encode(),
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId=self.kms_key,
                **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}),
            )
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code not in {"ConditionalRequestConflict", "409", "PreconditionFailed", "412"}:
                raise
            return False
        return True

    def change(self, state, *, release=False):
        active = state.value["active"]
        locked = active.get("fyp_publication_locked", False)
        owner = {"operation": active["operation"], "journal": state.uri}
        if release and not locked:
            return True
        if not locked and self.write(owner):
            active["fyp_publication_locked"] = True
            state.save()
            return True
        body, etag = read(self.s3, self.uri)
        current = json.loads(body)
        if release:
            return current != owner or self.write(None, etag)
        if current != owner and (current is not None or not self.write(owner, etag)):
            return False
        if not locked:
            active["fyp_publication_locked"] = True
            state.save()
        return True


class ModelPublication:
    journal_key = "online"

    def before_endpoint(self, state):
        return True

    def candidate_deployment(self, version, old, artifact):
        return self.deployment(version)

    def deployment_for_state(self, active):
        return self.deployment(active["model_version"])

    def creates_versioned_endpoint(self, deployment):
        return self.create_only

    def __init__(self, s3, publisher, binding):
        self.s3, self.publisher, self.binding = s3, publisher, binding
        self.resources = binding[self.binding_key]
        lock_uri = binding.get("fyp_publication_lock_uri")
        self.lock = PublicationLock(s3, lock_uri, binding["kms_key"]) if lock_uri else None
        self.deployment("0" * 64)
        if self.resources["model_list_profile"] == self.resources["models_config_profile"]:
            raise ValueError("model list and definitions require separate profiles")

    def acquire(self, state):
        return self.lock is None or self.lock.change(state)

    def release(self, state):
        return self.lock is None or self.lock.change(state, release=True)

    @classmethod
    def from_session(cls, session, s3, binding):
        return cls(
            s3,
            Publisher(
                session.client("appconfig"),
                session.client("appconfigdata"),
                binding[cls.binding_key],
                label_prefix=cls.binding_key.removesuffix("_appconfig"),
            ),
            binding,
        )

    def prepare(self, state, artifact, work):
        active = state.value["active"]
        if active.get(self.journal_key):
            return  # Frozen documents are already part of the same durable operation.
        listing = self.publisher.read(self.resources["model_list_profile"])
        models = self.publisher.read(self.resources["models_config_profile"])
        _, _, _, old = self.active_definition(listing, models, self.binding["environment"])
        deployment = self.candidate_deployment(active["model_version"], old, artifact)
        endpoint = deployment["endpoint_name"]
        if self.creates_versioned_endpoint(deployment) and old["release"]["endpoint"] == endpoint:
            raise ValueError("candidate endpoint must be independent of the active model")
        proposed = self.candidate_documents(
            listing, models, artifact, active["model_uri"], self.binding["environment"], endpoint
        )
        docs = {
            "before_list": listing,
            "before_models": models,
            "after_list": proposed.pop("model_list"),
            "after_models": proposed.pop("models_config"),
        }
        if "fallback_list" in proposed:
            docs["fallback_list"] = proposed.pop("fallback_list")
        pins = {}
        for name, raw in docs.items():
            path = Path(work) / (name + ".json")
            path.write_bytes(raw)
            uri = (
                self.binding["work_prefix"].rstrip("/")
                + "/"
                + active["operation"]
                + "/appconfig/"
                + (self.journal_key + "/" if self.journal_key != "online" else "")
                + sha256(raw)
                + "/"
                + path.name
            )
            immutable_put(self.s3, uri, path, self.binding["kms_key"])
            pins[name] = {"uri": uri, "sha256": sha256(raw)}
        active[self.journal_key] = {**proposed, "documents": pins, "endpoint": endpoint}
        if self.journal_key == "online" or self.create_only:
            active["endpoint_name"] = endpoint
        state.save()

    def read_documents(self, journal):
        docs = {}
        for name, pin in journal["documents"].items():
            bucket, key = location(pin["uri"])
            response = self.s3.get_object(Bucket=bucket, Key=key)
            with response["Body"] as stream:
                raw = stream.read(MAX_DOCUMENT_BYTES + 1)
            if len(raw) >= MAX_DOCUMENT_BYTES or sha256(raw) != pin["sha256"]:
                raise ValueError("frozen AppConfig content differs")
            docs[name] = raw
        return docs

    def publish(self, state, *, finalize=True):
        """Existing endpoint is ready; definitions first, then the active route."""
        active = state.value["active"]
        if active["stage"] not in {self.endpoint_ready_stage, "deployed"}:
            raise ValueError("model endpoint must pass the existing deployment probe first")
        journal = active[self.journal_key]
        docs = self.read_documents(journal)
        live_list = docs.get("fallback_list", docs["before_list"])

        def companion(profile, expected):
            if self.publisher.read(profile) != expected:
                raise ValueError("companion profile changed outside this publication")

        if (
            docs["before_models"] == docs["after_models"]
            and docs["before_list"] == docs["after_list"]
        ):
            companion(self.resources["model_list_profile"], docs["before_list"])
            companion(self.resources["models_config_profile"], docs["before_models"])
            if finalize:
                active["stage"] = "deployed"
                state.save()
            return "deployed"

        if not journal.get("models_complete"):
            if not self.publisher.step(
                state,
                "models",
                self.resources["models_config_profile"],
                docs["before_models"],
                docs["after_models"],
                companion=lambda: companion(self.resources["model_list_profile"], live_list),
                journal_key=self.journal_key,
            ):
                return "pending"
            journal["models_complete"] = True
            state.save()
        if not self.publisher.step(
            state,
            "route",
            self.resources["model_list_profile"],
            live_list,
            docs["after_list"],
            companion=lambda: companion(
                self.resources["models_config_profile"], docs["after_models"]
            ),
            journal_key=self.journal_key,
        ):
            return "pending"
        if finalize:
            active["stage"] = "deployed"
            state.save()
        return "deployed"
