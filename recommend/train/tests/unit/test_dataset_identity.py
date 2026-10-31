import pytest

from recommend.train.comm.datasvr.daily_contracts import merged_dataset_identity


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
