"""Published features must be reproducible without the archived training config."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from recommend.train.comm.feature_plan import FeaturePlan
from recommend.train.run import daily_run
from recommend.train.tests.smoke.test_tree_publication import daily_fixture


@pytest.mark.parametrize("with_history", [True, False])
def test_exported_contract_reproduces_training_features(tmp_path, with_history):
    config, daily = daily_fixture(tmp_path)
    if not with_history:
        config["tree_inputs"]["numeric"] = [
            f for f in config["tree_inputs"]["numeric"] if f not in config["history_scalars"]
        ]
        config["history_scalars"] = {}
    result = daily_run(
        config,
        daily_root=daily.root,
        frozen_root=(tmp_path / "frozen").as_uri(),
        cutoff="2026-10-08",
        artifact_root=(tmp_path / "artifacts").as_uri(),
        export_root=tmp_path / "exports",
    )
    assert result["status"] == "published"
    (contract_path,) = (tmp_path / "exports").rglob("feature-contract.json")
    contract = json.loads(contract_path.read_text())
    manifest = json.loads((contract_path.parent / "manifest.json").read_text())
    state = json.loads((contract_path.parent / "fitted-feature-state/state.json").read_text())
    assert (
        manifest["files"]["feature-contract.json"]
        == hashlib.sha256(contract_path.read_bytes()).hexdigest()
    )
    assert contract["encoding"]["history_scalars"] == config["history_scalars"]
    assert (
        contract["columns"]
        == config["tree_inputs"]["numeric"] + config["tree_inputs"]["categorical"]
    )
    # Only inference fields come from the published contract; no training archive fallback.
    inference = {**contract["encoding"], "item_filter": "True", "tasks": {}}
    serving = FeaturePlan(inference, state["dtypes"])
    serving.restore(state)
    training = FeaturePlan({**config, "tasks": {}}, state["dtypes"])
    training.restore(state)
    base = {f: None for f in config["features"]}
    base.update(video_finish_seen=1, watch_duration_ms=0)
    cases = [
        ({}, [0, None, 0, None, 0]),
        (
            {
                "i_drama_id": 123,
                "u_skipped__drama_id": [100, 123, 123],
                "u_drama_interests__id": [123, 100, 123],
                "u_drama_interests__weight": [1, 2, 3],
                "i_genre_v2_ids": [1, 1, 2],
                "u_onboarding_genre_ids": [1, 1, 3],
            },
            [3, 1, 2, 2 / 3, 1],
        ),
        ({"i_drama_id": 0, "u_skipped__drama_id": list(range(101))}, [100, 0, 0, 0, 0]),
        (
            {
                "i_drama_id": 123,
                "u_drama_interests__id": [123, 100, 123],
                "u_drama_interests__weight": [0, 0, 0],
            },
            [0, 0, 0, 0, 0],
        ),
        (
            {
                "i_drama_id": 100,
                "u_drama_interests__id": list(reversed(range(101))),
                "u_drama_interests__weight": [1] * 101,
            },
            [0, 0, 100, 0, 0],
        ),
        ({"i_drama_id": 2**63 - 2, "u_skipped__drama_id": [2**63 - 1]}, [1, 0, 0, 0, 0]),
    ]
    names = [
        "skip_count",
        "candidate_in_skip",
        "interest_count",
        "candidate_interest_share",
        "genre_overlap_count",
    ]
    for changes, expected in cases:
        source = {**base, **changes}
        actual_row = serving.row(source)
        expected_row = training.row(source)
        assert actual_row is not None and expected_row is not None
        actual = serving.encode(actual_row)["tree"]
        np.testing.assert_array_equal(actual, training.encode(expected_row)["tree"])
        if with_history:
            indices = [contract["columns"].index(f) for f in names]
            np.testing.assert_array_equal(
                actual[indices],
                np.asarray([np.nan if v is None else v for v in expected], dtype=np.float32),
            )
