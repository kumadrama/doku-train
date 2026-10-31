"""Generate synthetic parity evidence or verify it against a running container."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def fixture(root):
    import tensorflow.compat.v1 as tf
    from recommend_tf.inference.request_adapter import prepare
    from recommend_tf.inference.response_adapter import format_response
    from recommend_tf.train.tests.test_inference_contract import exported_fixture

    root.mkdir(parents=True, exist_ok=False)
    _, artifact, request = exported_fixture(root)
    payload, size = prepare(request, artifact, max_batch=128, max_history=10000)
    with tf.Graph().as_default(), tf.Session() as session:
        meta = tf.saved_model.loader.load(session, ["serve"], str(root / "package/saved_model"))
        signature = meta.signature_def["serving_default"]
        outputs = session.run(
            {k: session.graph.get_tensor_by_name(v.name) for k, v in signature.outputs.items()},
            {
                session.graph.get_tensor_by_name(signature.inputs[k].name): v
                for k, v in payload["inputs"].items()
            },
        )
    expected = format_response(
        {"outputs": {k: v.tolist() for k, v in outputs.items()}}, artifact["version"], size
    )
    (root / "request.json").write_text(json.dumps(request))
    (root / "expected.json").write_text(json.dumps(expected))


def verify(root, url):
    import numpy as np

    deadline = time.monotonic() + 120
    while True:
        try:
            with urlopen(url + "/ping", timeout=3) as response:
                assert response.status == 200
            break
        except (OSError, AssertionError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)
    request = Request(
        url + "/invocations",
        data=(root / "request.json").read_bytes(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=50) as response:
        result = json.load(response)
    expected = json.loads((root / "expected.json").read_text())
    assert result["model_version"] == expected["model_version"]
    assert len(result["predictions"]) == len(expected["predictions"])
    for actual, reference in zip(result["predictions"], expected["predictions"]):
        assert actual.keys() == reference.keys()
        for key in actual:
            np.testing.assert_allclose(actual[key], reference[key], rtol=1e-6, atol=1e-7)
    bad = Request(
        url + "/invocations",
        data=b'{"schema_version":"wrong","instances":[]}',
        headers={"Content-Type": "application/json"},
    )
    try:
        urlopen(bad, timeout=5)
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError("invalid request accepted")
    print(
        json.dumps(
            {
                "status": "passed",
                "model_version": result["model_version"],
                "rows": len(result["predictions"]),
                "invalid_request": 400,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fixture", "verify"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    args = parser.parse_args()
    if args.mode == "fixture":
        fixture(args.root)
    else:
        verify(args.root, args.url)


if __name__ == "__main__":
    main()
