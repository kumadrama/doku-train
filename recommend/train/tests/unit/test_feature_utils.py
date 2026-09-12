from __future__ import annotations

import inspect
import json

import pyarrow as pa
import pytest

from recommend.train.models.rerank.feature_utils import (
    FittedFeatureState,
    fit_feature_state,
    transform_batch,
)


def training_batch() -> pa.RecordBatch:
    return pa.record_batch(
        {
            "watch_7d": [0.0, None, 2.0],
            "country": ["US", None, "JP"],
            "effective_watch": [1.0, 0.0, 1.0],
            "effective_watch_valid": [True, True, True],
            "non_fast_swipe": [1.0, 1.0, 0.0],
            "non_fast_swipe_valid": [True, True, True],
            "immersive_click": [0.0, 1.0, 0.0],
            "immersive_click_valid": [True, True, True],
            "watch_duration_ms": [94.0, 95.0, None],
            "content_duration_ms": [100.0, 100.0, 100.0],
        }
    )


def test_missing_numeric_is_distinct_from_real_zero_and_category_oov_is_frozen() -> None:
    state = fit_feature_state(
        [training_batch()],
        numeric_features=("watch_7d",),
        categorical_features=("country",),
        feature_schema_version="features-v1",
    )
    validation = pa.record_batch(
        {
            "watch_7d": [0.0, None, 2.0],
            "country": ["US", "XX", None],
            "effective_watch": [1.0, 0.0, 1.0],
            "effective_watch_valid": [True, True, True],
            "non_fast_swipe": [1.0, 1.0, 0.0],
            "non_fast_swipe_valid": [True, True, True],
            "immersive_click": [0.0, 1.0, 0.0],
            "immersive_click_valid": [True, True, True],
            "watch_duration_ms": [94.0, 95.0, 96.0],
            "content_duration_ms": [100.0, 100.0, 100.0],
        }
    )
    transformed = transform_batch(validation, state)
    assert transformed.numeric[:, 0].tolist() == [-1.0, 0.0, 1.0]
    assert transformed.numeric[:, 1].tolist() == [0.0, 1.0, 0.0]
    assert transformed.categorical[:, 0].tolist() == [3, 1, 0]
    assert state.categorical[0].vocabulary == ("JP", "US")


def test_feature_state_round_trips_canonical_json() -> None:
    state = fit_feature_state(
        [training_batch()],
        numeric_features=("watch_7d",),
        categorical_features=("country",),
        feature_schema_version="features-v1",
    )
    encoded = state.to_json_bytes()
    assert encoded == state.to_json_bytes()
    assert json.loads(encoded)["feature_schema_version"] == "features-v1"
    assert FittedFeatureState.from_json_bytes(encoded) == state


def test_production_fit_is_independent_from_evaluation_state() -> None:
    evaluation = fit_feature_state(
        [training_batch()],
        numeric_features=("watch_7d",),
        categorical_features=("country",),
        feature_schema_version="features-v1",
    )
    production_batch = training_batch().set_column(0, "watch_7d", pa.array([10.0, 10.0, 10.0]))
    production = fit_feature_state(
        [production_batch],
        numeric_features=("watch_7d",),
        categorical_features=("country",),
        feature_schema_version="features-v1",
    )
    assert evaluation.numeric[0].mean == 1.0
    assert production.numeric[0].mean == 10.0


def test_feature_fit_keeps_numeric_statistics_streaming_and_rejects_non_finite() -> None:
    source = inspect.getsource(fit_feature_state)
    assert ".extend(" not in source
    assert "list[float]" not in source
    invalid = training_batch().set_column(0, "watch_7d", pa.array([0.0, float("nan"), 2.0]))
    with pytest.raises(ValueError, match="NUMERICAL_FAILURE"):
        fit_feature_state(
            [invalid],
            numeric_features=("watch_7d",),
            categorical_features=("country",),
            feature_schema_version="features-v1",
        )
