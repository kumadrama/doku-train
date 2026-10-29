import json

import pytest
import tensorflow.compat.v1 as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from recommend_tf.train.comm.artifact_upload import ArtifactUpload
from recommend_tf.train.comm.epoch_metrics import save_epoch
from recommend_tf.train.experiment import execute
from recommend_tf.train.models.rerank.data_utils import migrate_experiment
from recommend_tf.train.tests.experiment_fixture import make_experiment


def test_epoch_events_and_null_metrics(tmp_path):
    tf.disable_eager_execution()
    record = {
        "epoch": 3,
        "train_loss": {"a": 0.5},
        "train_objective_batch_mean": {"dense_reg_loss": 0.25, "sparse_reg_loss": 0.125},
        "validation_loss": {"a": 0.6},
        "validation_metrics": {"a": {"auc": {"value": None, "reason": "single_class"}}},
        "stage_seconds": {"train_with_input": 2.0},
    }
    save_epoch(tmp_path, record)
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("train/a/loss")[0].step == 3
    assert events.Scalars("train/a/loss")[0].value == 0.5
    assert events.Scalars("train/objective_batch_mean/dense_reg_loss")[0].value == 0.25
    assert events.Scalars("train/objective_batch_mean/sparse_reg_loss")[0].value == 0.125
    assert "validation/a/auc" not in events.Tags()["scalars"]
    assert json.loads((tmp_path / "epoch-metrics/0003.json").read_text()) == record


def test_upload_excludes_data_and_retries_failed_file(tmp_path):
    for name in [
        "prepared/train/part.parquet",
        "training/epoch-metrics/0001.json",
        "training/model.index",
        "training/live.sqlite",
        "experiment-status.json",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")

    class Client:
        def __init__(self):
            self.calls = []
            self.fail = True

        def upload_file(self, path, bucket, key, ExtraArgs):
            if self.fail:
                self.fail = False
                raise OSError("upload failed")
            self.calls.append(key)

    client = Client()
    upload = ArtifactUpload(client, "s3://fixture/run-1/", "fixture-key")
    with pytest.raises(OSError):
        upload(tmp_path)
    upload(tmp_path)
    assert len(client.calls) == 3
    assert client.calls[-1] == "run-1/experiment-status.json"
    assert not any("prepared" in k or "sqlite" in k for k in client.calls)
    upload(tmp_path)
    assert len(client.calls) == 3


def test_train_select_test_and_partial_epoch_persistence(tmp_path):
    from urllib.parse import unquote, urlsplit

    import pyarrow.parquet as pq

    source = make_experiment(tmp_path / "fixture", "deepfm", rows_per_day=4)
    source["max_epochs"] = 2
    manifest = json.loads((tmp_path / "fixture/dataset/manifest.json").read_text())
    schema = pq.read_schema(unquote(urlsplit(manifest["partitions"][0]["uri"]).path))
    dtypes = {
        field.name: "str"
        if str(field.type) == "string"
        else "int64"
        if "int64" in str(field.type)
        else "float32"
        for field in schema
        if field.name != "impression_time"
    }
    cfg = migrate_experiment(
        source,
        dtypes,
        {"optimizer": "adamw", "learning_rate": 0.004, "optimizer_args": {"weight_decay": 0.0001}},
        16,
    )
    seen = []

    def publish(root):
        seen.append(len(list((root / "training/epoch-metrics").glob("*.json"))))

    status = execute(cfg, tmp_path / "success", publish=publish)
    assert status["status"] == "completed" and seen == [1, 2, 2]
    root = tmp_path / "success"
    selection = json.loads((root / "training/selection.json").read_text())
    test = json.loads((root / "test/metrics.json").read_text())
    assert test["checkpoint"] == selection["checkpoint"]
    events = EventAccumulator(str(root / "training/tensorboard")).Reload()
    assert [x.step for x in events.Scalars("train/completion/loss")] == [1, 2]

    def fail_after_epoch(root):
        raise RuntimeError("simulated interrupted upload")

    with pytest.raises(RuntimeError, match="simulated"):
        execute(cfg, tmp_path / "failed", publish=fail_after_epoch)
    root = tmp_path / "failed"
    assert (root / "training/epoch-metrics/0001.json").exists()
    assert (root / "training/epoch-0001/checkpoint.json").exists()
    assert not (root / "training/metrics.json").exists()
    assert not (root / "test").exists()
    assert json.loads((root / "experiment-status.json").read_text())["status"] == "failed"
