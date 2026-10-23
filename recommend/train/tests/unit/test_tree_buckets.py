from __future__ import annotations

import json

import numpy as np
import pytest

from recommend.train.comm.datasvr.published_dataset import PublishedDataset
from recommend.train.comm.expressions import dependencies, evaluate, tree_feature
from recommend.train.comm.feature_plan import FeaturePlan
from recommend.train.comm.model_params_validator import load_config, validate_config
from recommend.train.comm.tree_buckets import validate_buckets
from recommend.train.tests.tree_binning_fixture import FIELD, make_tree_experiment


def artifact(cuts):
    return {"version": "tree-buckets/v1", "comparison": "float32_right", "features": {FIELD: cuts}}


def test_tree_expression_edges_and_precision():
    expr = f"tree({FIELD})"
    cuts = {FIELD: [1.0, 5.0]}
    assert dependencies(expr) == {FIELD}
    for value, expected in [(-10, 0), (1, 1), (3, 1), (5, 2), (999, 2), (None, None)]:
        assert evaluate(expr, {FIELD: value}, tree_thresholds=cuts) == expected
    below = float(np.nextafter(np.float32(1), np.float32(0)))
    assert evaluate(expr, {FIELD: below}, tree_thresholds=cuts) == 0
    assert evaluate(expr, {FIELD: 1 - 1e-12}, tree_thresholds=cuts) == 1
    assert evaluate(expr, {FIELD: 100}, tree_thresholds={FIELD: []}) == 0
    with pytest.raises(ValueError, match="missing tree bucket"):
        evaluate(expr, {FIELD: None}, tree_thresholds={})
    for value in [float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite"):
            evaluate(expr, {FIELD: value}, tree_thresholds=cuts)
    for expr in ["tree(x + 1)", "tree()", "tree(x, y)", "tree('x')", "log1p(tree(x))"]:
        with pytest.raises(ValueError):
            tree_feature(expr)


def test_all_bucket_ids_fitted_and_restored(tmp_path):
    config = make_tree_experiment(tmp_path)
    data = PublishedDataset(config)
    plan = FeaturePlan(config, data.dtypes, tree_buckets=artifact([10.0, 20.0]))
    plan.fit(data.rows("train", plan.columns))  # All synthetic values are below 10.
    target = FIELD + "_tree"
    assert list(plan.vocabs[target]) == [0, 1, 2]
    row = next(data.rows("train", plan.columns))
    row[FIELD] = 20.0
    processed = plan.row(row)
    assert processed[target] == 2
    assert plan.encode(processed)[target][0] == 4  # independent row, not OOV
    restored = FeaturePlan(config, data.dtypes)
    restored.restore(json.loads(json.dumps(plan.state())))
    assert restored.encode(restored.row(row))[target][0] == 4
    row[FIELD] = None
    assert restored.row(row)[target] == 0  # configured default precedes tree()
    with pytest.raises(ValueError, match="missing tree bucket features"):
        FeaturePlan(config, data.dtypes, tree_buckets=artifact([]) | {"features": {}})


def test_schema_and_relative_path(tmp_path):
    import yaml

    config = make_tree_experiment(tmp_path)
    config["tree_binning"]["config_path"] = "teacher.yaml"
    path = tmp_path / "nn.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["tree_binning"]["config_path"] == str(tmp_path / "teacher.yaml")
    config["tree_binning"]["enabled"] = False
    with pytest.raises(ValueError, match="requires enabled"):
        validate_config(config)


@pytest.mark.parametrize("cuts", [[2.0, 1.0], [1.0, 1.0], [float("nan")], ["bad"]])
def test_reject_invalid_thresholds(cuts):
    with pytest.raises(ValueError):
        validate_buckets(artifact(cuts))


def test_tree_config_binding(tmp_path):
    import yaml

    from recommend.train.comm.tree_pretraining import load_tree_config

    config = make_tree_experiment(tmp_path)
    teacher = load_tree_config(config)
    assert teacher is not None
    teacher["split"]["train_to"] = "2026-10-05"
    (tmp_path / "teacher.yaml").write_text(yaml.safe_dump(teacher))
    with pytest.raises(ValueError, match="same dataset"):
        load_tree_config(config)
