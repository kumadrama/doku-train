"""AppConfig admission policy; no dependency from model or offline training code."""

import hashlib
import json
import time


class PolicyUnavailable(RuntimeError):
    pass


def validate(value):
    fields = {
        "schema_version",
        "training_enabled",
        "publication_enabled",
        "recovery_enabled",
        "recovery_max_items",
    }
    if set(value) != fields or value["schema_version"] != "doku-tf-runtime/v1":
        raise ValueError("unsupported TF runtime policy")
    for name in ("training_enabled", "publication_enabled", "recovery_enabled"):
        if type(value[name]) is not bool:
            raise ValueError("policy switch must be boolean")
    if type(value["recovery_max_items"]) is not int or not 1 <= value["recovery_max_items"] <= 100:
        raise ValueError("invalid recovery budget")
    return value


class RuntimePolicy:
    def __init__(self, client, binding, *, clock=time.monotonic):
        self.client, self.binding, self.clock = client, binding, clock
        self.token = None
        self.snapshot = None
        self.next_poll = 0
        self.fetched = None
        self.version = None

    def read(self):
        now = self.clock()
        if self.snapshot is not None and now < self.next_poll:
            if now - self.fetched > 60:
                raise PolicyUnavailable("runtime policy expired")
            return dict(self.snapshot)
        try:
            if self.token is None:
                result = self.client.start_configuration_session(
                    ApplicationIdentifier=self.binding["application_id"],
                    EnvironmentIdentifier=self.binding["environment_id"],
                    ConfigurationProfileIdentifier=self.binding["profile_id"],
                    RequiredMinimumPollIntervalInSeconds=15,
                )
                self.token = result["InitialConfigurationToken"]
            result = self.client.get_latest_configuration(ConfigurationToken=self.token)
            self.token = result["NextPollConfigurationToken"]
            self.next_poll = now + result["NextPollIntervalInSeconds"]
            raw = result["Configuration"].read(65537)
            if len(raw) > 65536:
                raise ValueError("policy too large")
            if raw:
                value = validate(json.loads(raw))
                self.snapshot = value
                self.version = result.get("VersionLabel") or hashlib.sha256(raw).hexdigest()
            if self.snapshot is None:
                raise ValueError("empty initial policy")
            self.fetched = now
            return dict(self.snapshot)
        except Exception as error:
            self.snapshot = None
            self.token = None
            raise PolicyUnavailable("runtime policy unavailable: " + type(error).__name__) from None

    def allowed(self, operation):
        value = self.read()
        result = value[operation + "_enabled"]
        print(
            json.dumps(
                {
                    "event": "runtime_policy",
                    "version": self.version,
                    "operation": operation,
                    "allowed": result,
                }
            ),
            flush=True,
        )
        return result
