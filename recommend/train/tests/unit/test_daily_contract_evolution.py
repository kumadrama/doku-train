"""Cross-day contract changes use synthetic immutable daily inputs only."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from recommend.train.comm.daily_publication import prepare
from recommend.train.comm.datasvr.published_dataset import PublishedDataset, digest
from recommend.train.tests.smoke.test_tree_publication import daily_fixture


def rewrite_day(daily, day, *, features=None, labels=None, drop=()):
    base = Path(daily.root.removeprefix("file://")) / f"date={day}/run_id=abc"
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, change in [("features", features), ("labels", labels)]:
        path = base / f"contracts/{name}.json"
        contract = json.loads(path.read_text())
        if change:
            change(contract)
        path.write_text(json.dumps(contract))
        if name == "features":
            manifest["feature_schema_version"] = contract["feature_spec_version"]
            manifest["projection_sha256"] = digest(contract)
        else:
            manifest["label_definition_version"] = ",".join(contract["label_definition_versions"])
    if drop:
        path = base / "part.parquet"
        pq.write_table(pq.ParquetFile(path).read().drop(list(drop)), path)
        manifest["partitions"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.pop("content_sha256")
    manifest["content_sha256"] = digest(manifest)
    raw = json.dumps(manifest).encode()
    manifest_path.write_bytes(raw)
    pointer_path = base.parent / "_SUCCESS"
    pointer = json.loads(pointer_path.read_text())
    pointer.update(
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        projection_sha256=manifest["projection_sha256"],
    )
    pointer_path.write_text(json.dumps(pointer))


def evolve(tmp_path):
    config, daily = daily_fixture(tmp_path)
    removed = {"i_episode_dur_s", "u_skipped__drama_id"}

    def old_features(c):
        c["features"] = [f for f in c["features"] if f["name"] not in removed]
        c["feature_spec_version"] = "old-features"

    rewrite_day(daily, "2026-10-01", features=old_features, drop=removed)
    rewrite_day(
        daily,
        "2026-10-08",
        labels=lambda c: c["label_definition_versions"].extend(["labels-v5", "labels-v6"]),
    )
    config["feature_default_values"]["i_episode_dur_s"] = -1
    config["feature_default_values"]["u_skipped__drama_id"] = []
    return config, daily


def prepared(tmp_path, config, daily):
    return prepare(config, daily.root, (tmp_path / "frozen").as_uri(), "2026-10-08", window_days=30)


def test_merge_defaults_and_restore_without_source_metadata(tmp_path):
    config, daily = evolve(tmp_path)
    config = prepared(tmp_path, config, daily)
    dataset = PublishedDataset(config)
    columns = {"i_episode_dur_s", "u_skipped__drama_id", "completion"}
    rows = list(dataset.rows("train", columns))
    assert rows[0]["i_episode_dur_s"] == -1
    assert rows[0]["u_skipped__drama_id"] == []
    rows[0]["u_skipped__drama_id"].append(999)
    assert rows[1]["u_skipped__drama_id"] == []
    assert rows[32]["i_episode_dur_s"] == 0
    assert dataset.source_records[0]["defaulted_features"] == sorted(columns - {"completion"})
    assert dataset.source_records[0]["feature_schema_version"] == "old-features"
    assert set(dataset.manifest["label_definition_version"].split(",")) == {
        "fixed-5m-playback-stream-v4",
        "labels-v5",
        "labels-v6",
    }
    # The merged identity is independent of wall time or pointer formatting.
    again = prepared(tmp_path, config, daily)
    assert again["dataset"] == config["dataset"]
    frozen = tmp_path / "run-input"
    dataset.freeze(frozen)
    for path in Path(daily.root.removeprefix("file://")).rglob("*"):
        if path.is_file() and path.suffix != ".parquet":
            path.unlink()
    for path in (tmp_path / "frozen").rglob("*"):
        if path.is_file():
            path.unlink()
    restored = PublishedDataset(config, frozen_input=frozen)
    assert restored.source_records == dataset.source_records
    assert list(restored.rows("train", columns))[1:] == rows[1:]


def test_missing_explicit_default_fails(tmp_path):
    config, daily = evolve(tmp_path)
    del config["feature_default_values"]["i_episode_dur_s"]
    config = prepared(tmp_path, config, daily)
    with pytest.raises(ValueError, match="explicit feature defaults"):
        PublishedDataset(config)


@pytest.mark.parametrize("column", ["i_episode_dur_s", "completion", "impression_time"])
def test_declared_feature_or_required_column_missing_fails(tmp_path, column):
    config, daily = daily_fixture(tmp_path)
    rewrite_day(daily, "2026-10-01", drop=[column])
    dataset = PublishedDataset(prepared(tmp_path, config, daily))
    with pytest.raises(ValueError, match="missing required columns"):
        list(dataset.rows("train", {column}))


@pytest.mark.parametrize("change", ["dtype", "source", "unit", "labels", "metadata"])
def test_semantic_conflicts_rejected(tmp_path, change):
    config, daily = daily_fixture(tmp_path)

    def features(c):
        if change in {"dtype", "source", "unit"}:
            c["features"][0][change] = "different"
        elif change == "metadata":
            c["snapshot_schema_version"] = "different"

    def labels(c):
        if change == "labels":
            c["label_units"] = {"watch_duration_ms": "seconds"}

    rewrite_day(daily, "2026-10-08", features=features, labels=labels)
    with pytest.raises(ValueError, match="incompatible daily"):
        prepared(tmp_path, config, daily)


def test_unselected_new_feature_needs_no_default(tmp_path):
    config, daily = daily_fixture(tmp_path)
    rewrite_day(
        daily,
        "2026-10-08",
        features=lambda c: c["features"].append({"name": "unused", "dtype": "float64"}),
    )
    dataset = PublishedDataset(prepared(tmp_path, config, daily))
    assert list(dataset.rows("train", {"i_episode_dur_s"}))
    assert all(not r["defaulted_features"] for r in dataset.source_records)


@pytest.mark.parametrize("change", ["contract", "partition", "versions"])
def test_frozen_evidence_cannot_disagree_with_merged_manifest(tmp_path, change):
    config, daily = evolve(tmp_path)
    config = prepared(tmp_path, config, daily)
    path = Path(config["dataset"]["manifest_uri"].removeprefix("file://"))
    manifest = json.loads(path.read_text())
    if change == "contract":
        manifest["source_manifests"][0]["contracts"]["features"] = "{}"
    elif change == "partition":
        manifest["partitions"][0]["sha256"] = "a" * 64
    else:
        manifest["label_definition_version"] = "labels-v6"
    manifest.pop("content_sha256")
    manifest["content_sha256"] = digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"checksum|evidence"):
        PublishedDataset(config)


def test_plain_manifest_accepts_declared_label_version_set(tmp_path):
    from recommend.train.tests.experiment_fixture import make_experiment

    config = make_experiment(tmp_path)
    path = tmp_path / "dataset/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["label_definition_version"] += ",labels-v5"
    manifest.pop("content_sha256")
    manifest["content_sha256"] = digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="versions differ"):
        PublishedDataset(config)
    path = tmp_path / "dataset/contracts/labels.json"
    labels = json.loads(path.read_text())
    labels["label_definition_versions"].append("labels-v5")
    path.write_text(json.dumps(labels))
    assert PublishedDataset(config).labels
