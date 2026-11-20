"""Read and verify frozen monitoring metadata before training starts."""

import hashlib
import json
from pathlib import Path


def read_context(environment):
    path = environment.get("DOKU_MONITORING_CONTEXT_PATH")
    expected = environment.get("DOKU_MONITORING_CONTEXT_SHA256")
    inline = environment.get("DOKU_MONITORING_CONTEXT")
    if path or expected:
        if not path or not expected or inline is not None:
            raise ValueError("ambiguous or incomplete monitoring context transport")
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("monitoring context digest mismatch")
        context = json.loads(payload)
    elif inline is not None:
        # Existing valid frozen requests retain their original transport.
        context = json.loads(inline)
    else:
        return None
    if not isinstance(context, dict) or not context.get("training_job_name"):
        raise ValueError("invalid monitoring context")
    return context
