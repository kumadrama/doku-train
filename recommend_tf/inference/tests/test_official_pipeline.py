"""Publication integration with synthetic artifacts; no training or cloud calls."""

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from recommend_tf.inference import official
from recommend_tf.inference.tests.test_official import MemoryS3, fixture
from recommend_tf.platforms.sagemaker import build_context, daily_pipeline


def binding():
    return {
        "environment": "synthetic",
        "region": "us-east-1",
        "ecs_cluster": "synthetic",
        "state_uri": "s3://synthetic/state",
        "work_prefix": "s3://synthetic/work",
        "model_prefix": "s3://synthetic/models",
        "kms_key": "synthetic-key",
        "poll_seconds": 1,
        "max_wait_seconds": 10,
        "download_limit_bytes": 10000,
        "export_timeout_seconds": 10,
        "deployment": {
            "endpoint_name": "synthetic",
            "image_uri": "synthetic/tfs@sha256:" + "b" * 64,
            "execution_role_arn": "synthetic-role",
            "artifact_contract": "doku-tf-inference/v1",
            "instance_count": 1,
            "instance_type": "ml.m5.large",
            "alarm_names": ["synthetic-alarm"],
            "wait_seconds": 1,
            "termination_wait_seconds": 1,
            "timeout_seconds": 10,
            "inference_protocol": official.PROTOCOL,
            "container_environment": {"SAGEMAKER_TFS_INSTANCE_COUNT": "1"},
        },
        "outputs": dict.fromkeys(
            ["completion", "fast_swipe", "immersive_click", "watch_seconds"], "unit"
        ),
        "probe": {
            "fixture_kind": "synthetic",
            "content_type": "application/json",
            "request": {"signature_name": "serving_default", "inputs": {"item": [1]}},
            "expected_rows": 1,
            "tasks": ["completion", "fast_swipe", "immersive_click", "watch_seconds"],
        },
    }


class OfficialPipelineTest(unittest.TestCase):
    def test_binding_rejects_incompatible_protocol_before_execution(self):
        valid = binding()
        daily_pipeline.validate_binding(valid)
        for deployment in (
            {"inference_protocol": "unknown"},
            {"container_environment": {"DOKU_CONCURRENCY": "20"}},
            {"container_environment": {"SAGEMAKER_PROGRAM": "inference.py"}},
            {"container_environment": {"SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"}},
        ):
            invalid = copy.deepcopy(valid)
            invalid["deployment"].update(deployment)
            with self.subTest(deployment=deployment), self.assertRaises(ValueError):
                daily_pipeline.validate_binding(invalid)
        invalid = copy.deepcopy(valid)
        invalid["probe"]["request"] = {"instances": [{}]}
        with self.assertRaisesRegex(ValueError, "prepared tensors"):
            daily_pipeline.validate_binding(invalid)
        # Legacy bindings continue to accept the raw-feature probe.
        invalid["deployment"].pop("inference_protocol")
        daily_pipeline.validate_binding(invalid)

    def test_completed_run_uses_official_package_environment_and_probe(self):
        options = binding()
        # An existing probe binding can have different names from the exported signature.
        options["probe"]["tasks"] = ["old_task"]
        state = SimpleNamespace(
            value={
                "active": {
                    "stage": "trained",
                    "run_uri": "s3://synthetic/completed-run",
                    "cutoff": "2026-01-01",
                }
            },
            save=Mock(),
        )
        sm, runtime = Mock(), Mock()
        sm.describe_model.return_value = None
        sm.describe_endpoint_config.return_value = None
        sm.describe_endpoint.return_value = None
        s3 = MemoryS3()

        def fetch(_s3, _uri, target, _limit, _evidence):
            target.mkdir()
            (target / "experiment-config.json").write_text(
                json.dumps({"dataset_config": {"split": {"validation_to": "2026-01-01"}}})
            )
            (target / "experiment-status.json").write_text('{"status":"completed"}')

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            source = fixture(work)

            def export(args, **_):
                self.assertEqual(args[1:3], ["-m", "recommend_tf.train.export_model"])
                shutil.copytree(source, Path(args[args.index("--output") + 1]))

            with (
                patch.object(daily_pipeline, "fetch_run", side_effect=fetch),
                patch.object(daily_pipeline.subprocess, "run", side_effect=export),
            ):
                result = daily_pipeline.step(s3, sm, runtime, state, {}, {}, options, work=work)
            self.assertEqual(result, "pending")
            container = sm.create_model.call_args.kwargs["PrimaryContainer"]
            self.assertIn("/tensorflow-serving-v1/model.tar.gz", container["ModelDataUrl"])
            self.assertEqual(
                container["Environment"],
                {
                    "SAGEMAKER_TFS_INSTANCE_COUNT": "1",
                    "SAGEMAKER_TFS_DEFAULT_MODEL_NAME": "doku_tf_" + "a" * 64,
                },
            )
            sm.describe_endpoint.return_value = {
                "EndpointStatus": "InService",
                "EndpointConfigName": state.value["active"]["endpoint_config"],
            }
            runtime.invoke_endpoint.return_value = {
                "Body": io.BytesIO(
                    json.dumps({"outputs": dict.fromkeys(options["outputs"], [0.5])}).encode()
                )
            }
            self.assertEqual(
                daily_pipeline.step(s3, sm, runtime, state, {}, {}, options, work=work),
                "deployed",
            )
        sm.create_training_job.assert_not_called()
        self.assertEqual(
            runtime.invoke_endpoint.call_args.kwargs["CustomAttributes"],
            official.custom_attributes("a" * 64),
        )

    def test_pipeline_build_context_imports_publication_without_source_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(build_context.subprocess, "run"):
                context = build_context.prepare(Path(tmp) / "context", kind="pipeline")
            self.assertTrue((context / "recommend_tf/inference/official.py").is_file())
            self.assertFalse((context / "recommend_tf/inference/tests").exists())
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from recommend_tf.platforms.sagemaker import daily_pipeline; "
                    "assert 'tensorflow' not in sys.modules; "
                    "print(daily_pipeline.official.PROTOCOL)",
                ],
                cwd=context,
                env={**os.environ, "PYTHONPATH": str(context), "PYTHONDONTWRITEBYTECODE": "1"},
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), official.PROTOCOL)


if __name__ == "__main__":
    unittest.main()
