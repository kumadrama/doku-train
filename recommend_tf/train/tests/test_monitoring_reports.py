import io
import json
from datetime import datetime, timezone

import pytest
from botocore.exceptions import ClientError

from recommend_tf.platforms.monitoring.report_metrics import observation, put_event, release_event
from recommend_tf.train.comm.artifact_integrity import seal_run
from recommend_tf.train.tests.test_selected_metrics import selected_run


class Store:
    def __init__(self):
        self.objects = {}

    def put_object(self, **kwargs):
        key = kwargs["Bucket"], kwargs["Key"]
        if key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[key] = kwargs["Body"]
        return {"ETag": "one"}

    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(self.objects[kwargs["Bucket"], kwargs["Key"]]), "ETag": "one"}


def fixture(root):
    config = selected_run(root)
    config.update(
        model_name="model_a", dataset_config={"split": {"validation_dates": ["2026-10-29"]}}
    )
    (root / "input").mkdir()
    (root / "input/manifest.json").write_text("{}")
    (root / "experiment-config.json").write_text(json.dumps(config))
    (root / "resolved-config.json").write_text("{}")
    (root / "daily-status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "cutoff": "2026-10-29",
                "training_completed_at_ms": 1793300000000,
            }
        )
    )
    seal_run(root, "daily-status.json")
    return {
        "environment": "synthetic",
        "model_name": "model_a",
        "region": "us-east-2",
        "deployment": {"endpoint_name": "synthetic"},
        "outputs": {"click": "probability", "watch": "seconds"},
    }


def test_atomic_projection_and_immutable_retry(tmp_path):
    binding = fixture(tmp_path)
    event = observation(tmp_path, binding, run_uri="s3://synthetic/run")
    assert [t["value"] for t in event["tasks"]] == [None, 32.0]
    assert event["selected_epoch"] == 2
    s3 = Store()
    options = {"outbox_uri": "s3://synthetic/outbox", "kms_key": "synthetic"}
    put_event(s3, event, options)
    put_event(s3, event, options)
    assert len(s3.objects) == 1
    with pytest.raises(ValueError, match="conflicts"):
        put_event(s3, dict(event, selection_score=99), options)


def test_sealed_evidence_and_ambiguous_mapping(tmp_path):
    binding = fixture(tmp_path)
    binding["model_name"] = "other"
    with pytest.raises(ValueError, match="differ"):
        observation(tmp_path, binding, run_uri="s3://synthetic/run")
    (tmp_path / "training/epoch-metrics/0002.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        observation(tmp_path, binding, run_uri="s3://synthetic/run")


def test_historical_identity_does_not_mutate_run(tmp_path):
    binding = fixture(tmp_path)
    path = tmp_path / "experiment-config.json"
    config = json.loads(path.read_text())
    del config["model_name"]
    path.write_text(json.dumps(config))
    status = tmp_path / "daily-status.json"
    record = json.loads(status.read_text())
    del record["training_completed_at_ms"]
    status.write_text(json.dumps(record))
    seal_run(tmp_path, "daily-status.json")
    from recommend_tf.train.comm.artifact_integrity import checksum, files

    before = files(tmp_path)
    event = observation(
        tmp_path,
        binding,
        run_uri="s3://synthetic/run",
        historical_identity={"model_name": "model_a", "config_sha256": checksum(path)},
        training_job={
            "TrainingJobStatus": "Completed",
            "TrainingJobName": "old-job",
            "Environment": {"DOKU_ARTIFACT_URI": "s3://synthetic/run"},
            "TrainingEndTime": datetime(2026, 10, 30, tzinfo=timezone.utc),
        },
    )
    assert event["model_name"] == "model_a"
    assert files(tmp_path) == before


def test_failed_release_never_claims_online(tmp_path):
    binding = fixture(tmp_path)
    record = {
        "operation": "one",
        "release_revision": 1,
        "finished_at_ms": 123,
        "stage": "deployment_failed",
    }
    event = release_event("s3://bucket/state", record, binding)
    assert event["status"] == "failed"
    assert event["model_version"] is None


def test_preseal_event_equals_postseal_report_without_scanning_prepared(tmp_path):
    from recommend_tf.platforms.monitoring.report_metrics import source_evidence

    binding = fixture(tmp_path)
    evidence = source_evidence(tmp_path)
    assert all(not name.startswith("prepared/") for name in evidence["files"])
    expected = observation(tmp_path, binding, run_uri="s3://synthetic/run")
    (tmp_path / "COMMITTED.json").unlink()
    actual = observation(tmp_path, binding, run_uri="s3://synthetic/run", evidence=evidence)
    assert actual == expected
