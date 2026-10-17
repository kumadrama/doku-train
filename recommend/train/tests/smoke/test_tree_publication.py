from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recommend.train.comm.artifact_store import ObjectStore
from recommend.train.comm.datasvr.published_dataset import digest
from recommend.train.comm.publication import Registry, encode
from recommend.train.run import daily_run
from recommend.train.tests.experiment_fixture import make_experiment


def daily_fixture(tmp_path):
    config = make_experiment(tmp_path / "fixture", "xgboost", rows_per_day=32)
    dataset = tmp_path / "fixture/dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    projection = digest(json.loads((dataset / "contracts/features.json").read_text()))
    daily = ObjectStore((tmp_path / "daily").as_uri())
    for part in manifest["partitions"]:
        day = part["event_time_from"][:10]
        base = f"date={day}/run_id=abc"
        body = Path(part["uri"].removeprefix("file://")).read_bytes()
        daily.put(base + "/part.parquet", body)
        single = {
            **manifest,
            "dataset_version": "abc",
            "projection_sha256": projection,
            "label_config_version": "fixture",
            "row_count": part["rows"],
            "partitions": [{**part, "uri": daily.root + "/" + base + "/part.parquet"}],
        }
        single.pop("content_sha256")
        single["content_sha256"] = digest(single)
        raw = encode(single)
        daily.put(base + "/manifest.json", raw)
        for name in ["features", "labels"]:
            daily.put(
                base + f"/contracts/{name}.json", (dataset / f"contracts/{name}.json").read_bytes()
            )
        daily.put(
            f"date={day}/_SUCCESS",
            encode(
                {
                    "pointer_version": "doku-train-day/v1",
                    "date": day,
                    "projection_sha256": projection,
                    "manifest_uri": daily.root + "/" + base + "/manifest.json",
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                }
            ),
        )
    config["daily"] = {"training_window_days": 30}
    config["publication"] = {}
    for name, task in config["tasks"].items():
        task["output_unit"] = "probability" if task["objective"] == "binary:logistic" else "seconds"
        task["metrics"]["publish_rank"] = {
            "func": "auc" if task["objective"] == "binary:logistic" else "regression_auc"
        }
        config["publication"][name] = {"metric": "publish_rank", "minimum": 0.0}
    return config, daily


def test_daily_export_publish_and_repeat(tmp_path):
    config, daily = daily_fixture(tmp_path)
    kwargs = {
        "daily_root": daily.root,
        "frozen_root": (tmp_path / "frozen").as_uri(),
        "cutoff": "2026-10-08",
        "artifact_root": (tmp_path / "artifacts").as_uri(),
        "export_root": tmp_path / "exports",
    }
    observations = []
    result = daily_run(config, test_observer=observations.extend, **kwargs)
    assert result["status"] == "published"
    registry = Registry(ObjectStore(kwargs["artifact_root"]), config["model_name"])
    state = registry.read()[0]
    assert state["current"] == result["model_version"]
    assert len(state["versions"][state["current"]]["objects"]) > 10
    assert observations
    assert any(
        key.endswith("training/test-auc-events.json")
        for key in state["versions"][state["current"]]["objects"]
    )
    count = len(observations)
    again = daily_run(config, test_observer=observations.extend, **kwargs)
    assert again == {"status": "skipped", "reason": "duplicate_input"}
    assert len(observations) == count
    from recommend.train.eval_rerank import execute_test

    root = Path(result["run_dir"])
    before = (root / "checkpoint.xgb.zip").read_bytes()
    execute_test(root)
    assert (root / "checkpoint.xgb.zip").read_bytes() == before
    from recommend.train.export_model import export_model

    (root / "fitted-feature-state/state.json").write_text("{}")
    with pytest.raises(ValueError, match="feature state"):
        export_model(root, tmp_path / "bad")


def test_insufficient_days_and_rejection_do_not_publish(tmp_path):
    config, daily = daily_fixture(tmp_path)
    args = {
        "daily_root": daily.root,
        "frozen_root": (tmp_path / "frozen").as_uri(),
        "cutoff": "2026-10-02",
        "artifact_root": (tmp_path / "artifacts").as_uri(),
        "export_root": tmp_path / "exports",
    }
    observations = []
    assert (
        daily_run(config, test_observer=observations.extend, **args)["reason"]
        == "insufficient_days"
    )
    assert not observations
    for rule in config["publication"].values():
        rule["minimum"] = 1.0
    args["cutoff"] = "2026-10-08"
    result = daily_run(config, test_observer=observations.extend, **args)
    assert result["status"] == "rejected"
    assert observations
    registry = Registry(ObjectStore(args["artifact_root"]), config["model_name"])
    assert registry.read()[0]["current"] is None
    assert json.loads((Path(result["run_dir"]) / "status.json").read_text())["status"] == "complete"
    assert any(
        key.endswith("training/test-auc-events.json")
        for version in registry.read()[0]["versions"].values()
        for key in version["objects"]
    )


@pytest.mark.parametrize("paused", ["publication", "redis"])
def test_pause_after_training_archives_without_advancing_current(tmp_path, monkeypatch, paused):
    from recommend.train import runtime_policy, train_rerank
    from recommend.train.tests.unit.test_runtime_policy import document

    config, daily = daily_fixture(tmp_path)
    settings = document()
    monkeypatch.setattr(runtime_policy, "policy_for", lambda root: settings)
    original = train_rerank.execute_train

    def train_then_pause(value):
        result = original(value)
        settings[paused]["enabled"] = False
        return result

    monkeypatch.setattr(train_rerank, "execute_train", train_then_pause)
    artifact_root = (tmp_path / "artifacts").as_uri()
    result = daily_run(
        config,
        daily_root=daily.root,
        frozen_root=(tmp_path / "frozen").as_uri(),
        cutoff="2026-10-08",
        artifact_root=artifact_root,
        export_root=tmp_path / "exports",
    )
    assert result["reason"] == f"{paused}_disabled"
    state = Registry(ObjectStore(artifact_root), config["model_name"]).read()[0]
    assert state["current"] is None
    assert all(not v["published"] for v in state["versions"].values())
    assert any(v.get("kind") == "diagnostic" for v in state["versions"].values())
    assert (Path(result["run_dir"]) / "checkpoint.xgb.zip").exists()
