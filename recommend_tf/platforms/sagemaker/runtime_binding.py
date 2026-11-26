"""Pure validation of platform resources before a job is admitted."""

from recommend_tf.inference import official
from recommend_tf.train.comm.export_tasks import export_outputs


def validate_binding(binding, config=None):
    for name in (
        "environment",
        "region",
        "ecs_cluster",
        "state_uri",
        "work_prefix",
        "model_prefix",
        "kms_key",
    ):
        if not isinstance(binding.get(name), str) or not binding[name]:
            raise ValueError("explicit platform binding required: " + name)
    if "executor_stop_events" in binding:
        evidence = binding["executor_stop_events"]
        if (
            not ("idea_appconfig" in binding or "managed_model" in binding)
            or not isinstance(evidence, dict)
            or any(
                not isinstance(evidence.get(name), str) or not evidence[name]
                for name in ("log_group", "cluster_arn", "task_group")
            )
        ):
            raise ValueError("explicit Idea executor stop event binding required")
    for name in (
        "poll_seconds",
        "max_wait_seconds",
        "download_limit_bytes",
        "export_timeout_seconds",
    ):
        if type(binding.get(name)) is not int or binding[name] <= 0:
            raise ValueError("positive execution budget required: " + name)
    deployment = binding["deployment"]
    if official.protocol(deployment) == official.PROTOCOL:
        # Reject legacy/hook environment settings before any training is submitted.
        # The placeholder digest is used only for validation, never deployment.
        official.environment(deployment, "0" * 64)
        probe_request = binding["probe"]["request"]
        if probe_request.get("signature_name") != "serving_default" or not probe_request.get(
            "inputs"
        ):
            raise ValueError("official probe requires prepared tensors")
    if (
        type(deployment["instance_count"]) is not int
        or deployment["instance_count"] < 1
        or deployment["instance_type"].startswith(("ml.p", "ml.g", "ml.inf", "ml.trn"))
    ):
        raise ValueError("explicit positive CPU inference capacity required")
    if config is not None and "export_tasks" in config:
        export_outputs(config, binding.get("outputs", {}))
    elif set(binding["probe"]["tasks"]) != set(binding["outputs"]):
        raise ValueError("probe must verify all configured primary tasks")
    if binding["probe"].get("fixture_kind") != "synthetic":
        raise ValueError("deployment probe must be explicitly synthetic")
