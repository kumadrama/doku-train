"""Stable operation identity shared by admission and the executing pipeline."""

import hashlib
import json


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def operation_identity(
    config, ready, binding, *, bootstrap_run=None, evidence=None, rollback_version=None, retry_id=""
):
    # Execution budgets may be increased when resuming the same immutable operation.
    stable_binding = {
        k: v
        for k, v in binding.items()
        if k
        not in {
            "monitoring",
            "runtime_policy",
            "executor_stop_events",
            "legacy_ready_config_sha256",
            "deferred_uri",
            "poll_seconds",
            "max_wait_seconds",
            "download_limit_bytes",
            "export_timeout_seconds",
        }
    }
    return digest(
        {
            "config": config,
            "ready": ready,
            "binding": stable_binding,
            "bootstrap_run": bootstrap_run,
            "bootstrap_evidence": evidence,
            "rollback_version": rollback_version,
            "retry_id": retry_id,
        }
    )
