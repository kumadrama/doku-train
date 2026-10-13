import json
from pathlib import Path

import pytest

from recommend.train.comm.datasvr.published_dataset import PublishedDataset, digest
from recommend.train.comm.model_params_validator import validate_config
from recommend.train.tests.experiment_fixture import make_experiment


def test_freeze_survives_source_change_and_detects_corruption(tmp_path):
    config = make_experiment(tmp_path)
    validate_config(config)
    data = PublishedDataset(config)
    frozen = tmp_path / "frozen"
    data.freeze(frozen)
    source = Path(config["dataset"]["manifest_uri"].removeprefix("file://"))
    source.write_text("changed")
    restored = PublishedDataset(config, frozen_input=frozen)
    assert restored.manifest == data.manifest
    assert list(restored.rows("train", {"completion"})) == list(data.rows("train", {"completion"}))
    (frozen / "contracts/features.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        PublishedDataset(config, frozen_input=frozen)


def test_manifest_integrity_and_required_identity(tmp_path):
    config = make_experiment(tmp_path)
    source = Path(config["dataset"]["manifest_uri"].removeprefix("file://"))
    m = json.loads(source.read_text())
    m["row_count"] += 1
    source.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="digest"):
        PublishedDataset(config)
    m.pop("dataset_version")
    m["content_sha256"] = digest({k: v for k, v in m.items() if k != "content_sha256"})
    source.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="identity"):
        PublishedDataset(config)


def test_old_dataset_options_are_rejected(tmp_path):
    config = make_experiment(tmp_path)
    config["dataset"]["aws_profile"] = "not-a-config-option"
    with pytest.raises(Exception, match="Additional properties"):
        validate_config(config)
