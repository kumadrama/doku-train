"""Regression coverage for experiment lifecycle and discarded-source isolation."""

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from recommend_tf.train.comm.datasvr.daily_contracts import digest
from recommend_tf.train.experiment import execute
from recommend_tf.train.tests.test_mixed_integration import mixed_config
from recommend_tf.train.train_rerank import train


def test_all_skipped_experiment_publishes_evidence_not_success(tmp_path):
    cfg = mixed_config(tmp_path)
    for source in cfg["data_sources"]:
        source["manifest_uri"] = (tmp_path / (source["name"] + "-missing.json")).as_uri()
    uploaded = []

    def publish(root):
        uploaded.append(json.loads((root / "input/sources.json").read_text()))
        assert json.loads((root / "experiment-status.json").read_text())["status"] == "skipped"
        assert not (root / "COMMITTED.json").exists()

    assert execute(cfg, tmp_path / "run", publish=publish)["status"] == "skipped"
    assert len(uploaded) == 1
    assert all(source["status"] == "skipped" for source in uploaded[0]["sources"])


def test_experiment_reports_empty_test_as_skipped(tmp_path):
    cfg = mixed_config(tmp_path)
    cfg["dataset_config"]["split"].update(test_from="2030-01-01", test_to="2030-01-01")
    result = execute(cfg, tmp_path / "run")
    assert result["status"] == "completed"
    assert result["test"] == "skipped"
    assert result["test_reason"] == "empty"
    assert (tmp_path / "run/COMMITTED.json").exists()


def test_discarded_source_dtypes_do_not_poison_valid_source(tmp_path):
    cfg = mixed_config(tmp_path)
    path = tmp_path / "fyp/dataset/manifest.json"
    manifest = json.loads(path.read_text())
    contract = path.parent / "contracts/features.json"
    features = json.loads(contract.read_text())
    field = next(x for x in features["features"] if x["name"] == "i_drama_id")
    field["dtype"] = "float64"
    raw = json.dumps(features).encode()
    contract.write_bytes(raw)
    if "contract_sha256" in manifest:
        manifest["contract_sha256"]["features"] = hashlib.sha256(raw).hexdigest()
    manifest.pop("content_sha256")
    manifest["content_sha256"] = digest(manifest)
    path.write_text(json.dumps(manifest))
    shard = Path(unquote(urlsplit(manifest["partitions"][0]["uri"]).path))
    shard.write_bytes(b"corrupt")
    result = train(cfg, tmp_path / "run")
    assert len(result) == 1
    evidence = json.loads((tmp_path / "run/input/sources.json").read_text())
    assert [s["status"] for s in evidence["sources"]] == ["skipped", "ready"]
