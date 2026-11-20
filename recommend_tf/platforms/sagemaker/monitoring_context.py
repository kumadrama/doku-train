"""Immutable file transport for daily monitoring metadata."""

import json

from recommend_tf.train.comm.artifact_integrity import checksum

from .publish_model import immutable_put


def attach_context(s3, request, context, *, prefix, work, kms_key):
    path = work / "monitoring-context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context, sort_keys=True, allow_nan=False))
    immutable_put(s3, prefix.rstrip("/") + "/monitoring-context.json", path, kms_key)
    env = request["Environment"]
    env.pop("DOKU_MONITORING_CONTEXT", None)
    env["DOKU_MONITORING_CONTEXT_PATH"] = "/opt/ml/input/data/daily/monitoring-context.json"
    env["DOKU_MONITORING_CONTEXT_SHA256"] = checksum(path)


def validate_environment(request):
    for name, value in request.get("Environment", {}).items():
        if not isinstance(value, str) or len(value) > 512:
            raise ValueError("SageMaker environment value exceeds contract: " + name)
