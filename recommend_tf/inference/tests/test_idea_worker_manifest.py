"""Keep a source-only change from silently reusing an older Idea release image."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "idea_source_package", ROOT / "idea-mapped-daily/build/package_source.py"
)
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class WorkerManifestTests(unittest.TestCase):
    def test_manifest_pins_current_runtime_and_original_worker(self):
        body, _ = PACKAGE.source_archive(ROOT)
        manifest = json.loads(
            (ROOT / "idea-mapped-daily/deploy/prod/task-definition.yml").read_text()
        )
        container = manifest["containerDefinitions"][0]
        values = {item["name"]: item["value"] for item in container["environment"]}
        digest = hashlib.sha256(body).hexdigest()
        self.assertEqual(values["ARTIFACT_SHA"], digest)
        self.assertTrue(values["ARTIFACT_KEY"].endswith("/" + digest + "/worker.tar.gz"))
        self.assertEqual(manifest["family"], "doku-prod-idea-mapped-daily")
        self.assertEqual(manifest["taskRoleArn"], "doku-prod-idea-mapped-daily-worker")
        self.assertFalse((ROOT / "idea-mapped-daily/deploy/prod/task-target.yml").exists())
        self.assertIn("worker artifact digest differs", container["command"][0])

    def test_runtime_changes_affect_archive_but_tests_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recommend_tf/platforms/sagemaker/idea_publication.py"
            source.parent.mkdir(parents=True)
            source.write_text("value = 1\n")
            before, _ = PACKAGE.source_archive(root)
            self.assertEqual(before, PACKAGE.source_archive(root)[0])
            tests = root / "recommend_tf/train/tests/test_example.py"
            tests.parent.mkdir(parents=True)
            tests.write_text("test = True\n")
            self.assertEqual(before, PACKAGE.source_archive(root)[0])
            source.write_text("value = 2\n")
            self.assertNotEqual(before, PACKAGE.source_archive(root)[0])
