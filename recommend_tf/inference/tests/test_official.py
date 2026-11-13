"""Serving-only regression tests: no TensorFlow import or training execution."""

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from recommend_tf.inference import official
from recommend_tf.inference.artifact_loader import checksum
from recommend_tf.platforms.sagemaker.publish_model import publish_package, verify_probe


def fixture(root):
    source = root / "canonical"
    (source / "saved_model/variables").mkdir(parents=True)
    (source / "saved_model/saved_model.pb").write_bytes(b"synthetic protobuf placeholder")
    (source / "saved_model/variables/variables.index").write_bytes(b"synthetic variables")
    (source / "feature-config.json").write_text("{}")
    (source / "signature.json").write_text(
        json.dumps(
            {
                "version": "doku-tf-signature/v1",
                "model_version": "a" * 64,
                "input_stage": "organized_padded_features",
                "response_order": "request_order",
                "inputs": {"item": {}},
                "output_units": dict.fromkeys(
                    ["completion", "fast_swipe", "immersive_click", "watch_seconds"], "unit"
                ),
                "outputs": dict.fromkeys(
                    ["completion", "fast_swipe", "immersive_click", "watch_seconds"], {}
                ),
            }
        )
    )
    inventory = {
        p.relative_to(source).as_posix(): checksum(p) for p in source.rglob("*") if p.is_file()
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {"version": "doku-tf-inference/v1", "model_version": "a" * 64, "files": inventory}
        )
    )
    return source


class MemoryS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, **kw):
        key = (kw["Bucket"], kw["Key"])
        data = kw["Body"].read()
        if key in self.objects and self.objects[key] != data:
            raise AssertionError("immutable overwrite")
        self.objects[key] = data

    def get_object(self, **kw):
        return {"Body": io.BytesIO(self.objects[(kw["Bucket"], kw["Key"])])}


class OfficialTest(unittest.TestCase):
    def test_legacy_probe_remains_version_checked(self):
        probe = {
            "content_type": "application/json",
            "request": {"instances": [{}]},
            "version_field": "model_version",
            "outputs_field": "predictions",
            "expected_rows": 1,
            "tasks": ["completion"],
        }
        runtime = Mock()
        runtime.invoke_endpoint.return_value = {
            "Body": io.BytesIO(b'{"model_version":"old","predictions":[{"completion":0.5}]}')
        }
        verify_probe(runtime, "synthetic", "old", probe)
        self.assertNotIn("CustomAttributes", runtime.invoke_endpoint.call_args.kwargs)
        runtime.invoke_endpoint.return_value = {
            "Body": io.BytesIO(b'{"model_version":"wrong","predictions":[{"completion":0.5}]}')
        }
        with self.assertRaises(ValueError):
            verify_probe(runtime, "synthetic", "old", probe)

    def test_package_is_deterministic_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = fixture(root)
            before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            first = official.package(source, root / "one.tar.gz")
            second = official.package(source, root / "two.tar.gz")
            self.assertEqual(first, second)
            with tarfile.open(root / "one.tar.gz") as tar:
                names = tar.getnames()
                self.assertIn("doku_tf_" + "a" * 64 + "/1/saved_model.pb", names)
                self.assertIn("doku-contract/manifest.json", names)
                self.assertFalse(any(n.startswith("code/") for n in names))
            self.assertEqual(
                before, {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            )
            with self.assertRaises(FileExistsError):
                official.package(source, root / "one.tar.gz")
            (source / "saved_model/saved_model.pb").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                official.package(source, root / "bad.tar.gz")

    def test_publish_uses_distinct_immutable_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = fixture(Path(tmp))
            s3 = MemoryS3()
            old = publish_package(s3, source, "s3://synthetic/models", "key")
            new = publish_package(
                s3, source, "s3://synthetic/models", "key", inference_protocol=official.PROTOCOL
            )
            self.assertNotEqual(old, new)
            self.assertIn("/tensorflow-serving-v1/model.tar.gz", new)
            self.assertEqual(
                new,
                publish_package(
                    s3, source, "s3://synthetic/models", "key", inference_protocol=official.PROTOCOL
                ),
            )

    def test_protocol_environment_and_model_identity(self):
        binding = {"inference_protocol": official.PROTOCOL}
        self.assertEqual(
            official.environment(binding, "a" * 64),
            {"SAGEMAKER_TFS_DEFAULT_MODEL_NAME": "doku_tf_" + "a" * 64},
        )
        self.assertEqual(official.environment({}, "a" * 64), {"DOKU_MODEL_VERSION": "a" * 64})
        for value in ("A" * 64, "wrong", "../model"):
            with self.assertRaises(ValueError):
                official.model_name(value)
        for key in ("DOKU_CONCURRENCY", "SAGEMAKER_PROGRAM"):
            with self.assertRaises(ValueError):
                official.environment(dict(binding, container_environment={key: "1"}), "a" * 64)

    def test_probe_pins_actual_model_and_validates_outputs(self):
        probe = {
            "content_type": "application/json",
            "request": {"signature_name": "serving_default", "inputs": {"x": [1]}},
            "expected_rows": 1,
            "tasks": ["completion"],
        }
        runtime = Mock()
        runtime.invoke_endpoint.return_value = {
            "Body": io.BytesIO(b'{"outputs":{"completion":[0.5]}}')
        }
        verify_probe(runtime, "synthetic", "a" * 64, probe, inference_protocol=official.PROTOCOL)
        self.assertEqual(
            runtime.invoke_endpoint.call_args.kwargs["CustomAttributes"],
            official.custom_attributes("a" * 64),
        )
        runtime.invoke_endpoint.return_value = {"Body": io.BytesIO(b'{"outputs":[0.5]}')}
        verify_probe(runtime, "synthetic", "a" * 64, probe, inference_protocol=official.PROTOCOL)
        for outputs in (
            {"other": [1]},
            {"completion": []},
            {"completion": [None]},
            {"completion": [[1]]},
        ):
            runtime.invoke_endpoint.return_value = {
                "Body": io.BytesIO(json.dumps({"outputs": outputs}).encode())
            }
            with self.assertRaises(ValueError):
                verify_probe(
                    runtime, "synthetic", "a" * 64, probe, inference_protocol=official.PROTOCOL
                )


if __name__ == "__main__":
    unittest.main()
