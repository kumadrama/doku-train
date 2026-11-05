"""Independent exact daily manifests use aggregate train coverage."""

import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from recommend_tf.train.comm.datasvr.daily_contracts import digest
from recommend_tf.train.tests.test_mixed_integration import mixed_config
from recommend_tf.train.train_rerank import train


def test_multiple_day_manifests_train_without_fyp_daily_metadata(tmp_path):
    cfg = mixed_config(tmp_path)
    cfg["data_sources"] = cfg["data_sources"][:1]
    source = cfg["data_sources"][0]
    path = Path(unquote(urlsplit(source.pop("manifest_uri")).path))
    original = json.loads(path.read_text())
    uris = []
    for i, part in enumerate(original["partitions"]):
        manifest = dict(original, partitions=[part], row_count=part["rows"])
        manifest.pop("content_sha256")
        manifest["content_sha256"] = digest(manifest)
        daily = path.with_name(f"day-{i}.json")
        daily.write_text(json.dumps(manifest))
        uris.append(daily.as_uri())
    source["manifest_uris"] = uris
    result = train(cfg, tmp_path / "run")
    assert len(result) == 1
    record = json.loads((tmp_path / "run/input/sources.json").read_text())
    assert record["sources"][0]["status"] == "ready"


def test_overlapping_manifests_are_configuration_error(tmp_path):
    cfg = mixed_config(tmp_path)
    source = cfg["data_sources"][0]
    path = Path(unquote(urlsplit(source.pop("manifest_uri")).path))
    duplicate = path.with_name("duplicate.json")
    duplicate.write_bytes(path.read_bytes())
    source["manifest_uris"] = [path.as_uri(), duplicate.as_uri()]
    with pytest.raises(ValueError, match="Overlapping"):
        train(cfg, tmp_path / "run")
