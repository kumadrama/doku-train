from types import SimpleNamespace
from unittest.mock import Mock

from recommend_tf.platforms.sagemaker.idea_mapped_input import next_ready
from recommend_tf.train.comm.datasvr.daily_dataset import resolve_config, window


def test_growing_window_and_frozen_recipe():
    config = {
        "online_config": {"first_day": "2026-11-02", "training_date": 30, "validation_date": 1},
        "dataset_config": {"split": {"timezone": "UTC"}},
    }
    for cutoff, first, count in [
        ("2026-11-12", "2026-11-02", 10),
        ("2026-12-12", "2026-11-12", 30),
    ]:
        periods = window(config, cutoff)
        assert periods["train"][0] == first and len(periods["train"]) == count
        assert periods["validation"] == [cutoff]
        assert window(resolve_config(config, cutoff), cutoff) == periods


def test_reconcile_resumes_frozen_ready_or_waits_for_publication():
    s3 = Mock()
    ready = {"cutoff": "2026-11-12"}
    state = SimpleNamespace(
        value={"active": {"ready": ready}, "high_water": "2026-11-12", "history": []}
    )
    source = {"uri": "s3://bucket/mapped"}
    config = {"online_config": {"training_date": 1, "validation_date": 1}}
    assert next_ready(s3, state, source, config) == ready
    assert not s3.mock_calls
    state.value["active"] = None
    s3.list_objects_v2.return_value = {}
    assert next_ready(s3, state, source, config) is None
