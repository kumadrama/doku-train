import pytest

from recommend_tf.train.comm.datasvr.daily_contracts import merged_dataset_identity


def test_identity_boundaries():
    def source(identity, schema="fyp-rank-features/v5", label="v1"):
        return dict(dataset_id=identity, feature_schema_version=schema, label_config_version=label)

    v5 = source("fyp-rank-train/fyp-rank-features-v5")
    v6 = source("fyp-rank-train/fyp-rank-features-v6", "fyp-rank-features/v6")
    assert merged_dataset_identity([v5, v6]) == "fyp-rank-train"
    assert merged_dataset_identity([v5, v5]) == v5["dataset_id"]
    assert merged_dataset_identity([source("other"), source("other")]) == "other"
    for bad in [
        source("unrelated"),
        source("fyp-rank-train"),
        source(v6["dataset_id"]),
        source(v5["dataset_id"], label="v2"),
    ]:
        with pytest.raises(ValueError):
            merged_dataset_identity([v5, bad])


def test_tf_versioned_family_freeze_restore(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    from recommend_tf.train.comm.datasvr.daily_contracts import digest
    from recommend_tf.train.comm.datasvr.daily_dataset import freeze
    from recommend_tf.train.comm.datasvr.published_dataset import PublishedDataset
    from recommend_tf.train.tests.test_daily_publication import fixture

    config, ready = fixture(tmp_path)
    for i, source in enumerate(ready["sources"]):
        version = 6 if i == 6 else 5
        path = Path(source["manifest_uri"].removeprefix("file://"))
        feature_path = path.parent / "contracts/features.json"
        features = json.loads(feature_path.read_text())
        features["feature_spec_version"] = f"fyp-rank-features/v{version}"
        if version == 6:
            features["features"].append({"name": "u_deep_link_drama_id", "dtype": "int64"})
        body = json.dumps(features).encode()
        feature_path.write_bytes(body)
        manifest = json.loads(path.read_text())
        manifest.update(
            dataset_id=f"fyp-rank-train/fyp-rank-features-v{version}",
            feature_schema_version=features["feature_spec_version"],
            projection_sha256=digest(features),
        )
        manifest["contract_sha256"]["features"] = hashlib.sha256(body).hexdigest()
        manifest["content_sha256"] = digest(
            {k: v for k, v in manifest.items() if k != "content_sha256"}
        )
        raw = json.dumps(manifest).encode()
        path.write_bytes(raw)
        source.update(
            manifest_sha256=hashlib.sha256(raw).hexdigest(),
            contract_sha256=manifest["contract_sha256"],
        )
    target = tmp_path / "frozen"
    resolved = freeze(config, ready, target)
    dataset = PublishedDataset(resolved["dataset_config"], frozen_input=target / "input")
    assert dataset.manifest["dataset_id"] == "fyp-rank-train"
    rows = list(dataset.rows("train", {"i_episode_dur_s"}))
    assert rows and all(not r["defaulted_features"] for r in dataset.source_records)
    for source in ready["sources"]:
        Path(source["manifest_uri"].removeprefix("file://")).unlink()
    restored = PublishedDataset(resolved["dataset_config"], frozen_input=target / "input")
    assert list(restored.rows("train", {"i_episode_dur_s"})) == rows
    assert restored.source_records[-1]["dataset_id"].endswith("-v6")
