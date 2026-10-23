from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema import ValidationError

from recommend.train.comm.datasvr.published_dataset import PublishedDataset
from recommend.train.comm.expressions import evaluate
from recommend.train.comm.feature_plan import FeaturePlan
from recommend.train.comm.model_params_validator import validate_config
from recommend.train.comm.search import compare, expand_search
from recommend.train.tests.experiment_fixture import make_experiment


def test_expressions_reject_execution_and_handle_null():
    assert evaluate("where(is_null(a), 0, log1p(a))", {"a": None}) == 0
    assert evaluate("a > 0", {"a": None}) is None
    with pytest.raises(ValueError):
        evaluate('__import__("os")', {})
    with pytest.raises(ValueError):
        evaluate("a.__class__", {"a": 0})


def test_schema_rejects_unknown_and_derived_defaults(tmp_path):
    c = make_experiment(tmp_path)
    validate_config(c)
    c["tasks"]["completion"]["typo"] = 1
    with pytest.raises(ValidationError):
        validate_config(c)
    del c["tasks"]["completion"]["typo"]
    c["feature_default_values"]["fake_log"] = 0
    with pytest.raises(ValueError):
        validate_config(c)


def test_train_vocab_and_history(tmp_path):
    c = make_experiment(tmp_path)
    data = PublishedDataset(c)
    plan = FeaturePlan(c, data.dtypes)
    plan.fit(data.rows("train", plan.columns))
    old = plan.state()
    row = next(data.rows("validation", plan.columns))
    row["i_drama_id"] = 123456
    row["u_drama_interests__id"] = [8, 8, 9]
    row["u_drama_interests__weight"] = [1e-81, 1e-81, 2e-81]
    processed = plan.row(row)
    assert processed["u_drama_interests__weight"] == [0.5, 0.5]
    encoded = plan.encode(processed)
    assert encoded["i_drama_id"][0] == 1
    assert plan.state() == old


def test_grid_identity(tmp_path):
    import yaml

    from recommend.train.tests.synthetic_config import synthetic_search

    for model in ["dnn", "deepfm", "xgboost"]:
        make_experiment(tmp_path, model)
    p = tmp_path / "search.yaml"
    p.write_text(yaml.safe_dump(synthetic_search()))
    _, trials = expand_search(p, tmp_path / "search")
    assert len(trials) == 9
    assert len({x["output"]["run_id"] for _, x in trials}) == 9
    assert all(x["model_type"] == "rerank" for _, x in trials[:-1])


def test_tradeoff_is_not_win():
    a = {"metrics": {"auc": {"value": 0.8}, "logloss": {"value": 0.7}}}
    b = {"metrics": {"auc": {"value": 0.7}, "logloss": {"value": 0.6}}}
    rule = {"loss": "logloss", "loss_direction": "min", "auc": "auc", "auc_direction": "max"}
    assert compare(a, b, rule)["status"] == "tradeoff"


def test_checksum_and_time_partition_integrity(tmp_path):
    config = make_experiment(tmp_path)
    dataset = PublishedDataset(config)
    part = dataset.partitions("train")[0]
    path = Path(part["uri"].removeprefix("file://"))
    with path.open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        list(dataset.rows("train", {"completion"}))


def test_single_class_auc_keeps_logloss():
    import numpy as np

    from recommend.train.comm.task_metrics import metric_values

    task = {"metrics": {"auc": {"func": "auc"}, "logloss": {"func": "logloss"}}}
    arrays = {
        "metric:t": np.zeros((3, 1)),
        "mask:t": np.ones((3, 1), dtype=bool),
        "valid:t:auc": np.ones((3, 1), dtype=bool),
        "valid:t:logloss": np.ones((3, 1), dtype=bool),
    }
    result = metric_values(
        task,
        "t",
        arrays,
        np.array([0.1, 0.2, 0.3]),
        logits=np.log(np.array([0.1, 0.2, 0.3]) / np.array([0.9, 0.8, 0.7])),
    )
    assert result["auc"]["value"] is None
    assert result["auc"]["reason"] == "single_class"
    assert result["logloss"]["value"] > 0


def test_original_defaults_precede_log(tmp_path):
    config = make_experiment(tmp_path)
    dataset = PublishedDataset(config)
    plan = FeaturePlan(config, dataset.dtypes)
    row = next(dataset.rows("train", plan.columns))
    field = "u_expose_count_7d_all"
    row[field] = None
    value = plan.row(row)
    assert value[field] == 0 and value[field + "_log_bucket"] == 0
    assert not any(k.endswith("_missing") for k in value)


def test_derived_bucket_embedding_state_and_oov(tmp_path):
    import json

    import numpy as np

    config = make_experiment(tmp_path)
    data = PublishedDataset(config)
    source_types = dict(data.dtypes)
    plan = FeaturePlan(config, data.dtypes)
    plan.fit(data.rows("train", plan.columns))
    field = "i_episode_dur_s_bucket"
    assert field not in plan.columns
    assert data.dtypes == source_types
    assert plan.dtypes[field] == "int64"
    row = next(data.rows("train", plan.columns))
    row["i_episode_dur_s"] = 0
    encoded = plan.encode(plan.row(row))
    # Bucket zero is a learned category, not PAD=0 or OOV=1.
    assert encoded[field].dtype == np.int64
    assert encoded[field][0] >= 2
    row["i_episode_dur_s"] = 211
    before = json.loads(json.dumps(plan.state()))
    encoded = plan.encode(plan.row(row))
    assert encoded[field][0] == 1  # Unseen validation bucket cannot fit the vocabulary.
    restored = FeaturePlan(config, data.dtypes)
    restored.restore(before)
    assert restored.state() == before
    np.testing.assert_array_equal(restored.encode(restored.row(row))[field], encoded[field])


@pytest.mark.parametrize("model", ["dnn", "deepfm"])
def test_bucket_embeddings_forward_and_gradient(tmp_path, model):
    import numpy as np
    import torch

    from recommend.train.models.rerank.rerank_model import ConfiguredRerank

    config = make_experiment(tmp_path, model)
    data = PublishedDataset(config)
    plan = FeaturePlan(config, data.dtypes)
    plan.fit(data.rows("train", plan.columns))
    row = plan.row(next(data.rows("train", plan.columns)))
    batch = {k: torch.from_numpy(np.stack([v, v])) for k, v in plan.encode(row).items()}
    network = ConfiguredRerank(config, plan.state())
    output = network(batch)
    sum(v.sum() for v in output.values()).backward()
    for field in config["embed_name"]:
        if field.endswith("_bucket"):
            table = network.embeddings[network.table_names.index(config["embed_name"][field])]
            assert table.embedding_dim == 4
            assert table.weight.grad is not None
            assert torch.isfinite(table.weight.grad).all()
    assert all(v.shape[0] == 2 and torch.isfinite(v).all() for v in output.values())


def test_history_limit_and_stable_interest_order(tmp_path):
    config = make_experiment(tmp_path)
    data = PublishedDataset(config)
    plan = FeaturePlan(config, data.dtypes)
    row = next(data.rows("train", plan.columns))
    row["u_skipped__drama_id"] = list(range(150))
    row["u_drama_interests__id"] = list(range(149, -1, -1))
    row["u_drama_interests__weight"] = [1.0] * 150
    value = plan.row(row)
    assert value["u_skipped__drama_id"] == list(range(50, 150))
    assert value["u_drama_interests__id"] == list(range(100))
    assert abs(sum(value["u_drama_interests__weight"]) - 1) < 1e-12


def test_no_test_reads_during_training(tmp_path):
    from recommend.train.train_rerank import execute_train

    c = make_experiment(tmp_path)
    # Test-only corruption must not affect train/validation or trigger test reads.
    path = tmp_path / "dataset/day-7.parquet"
    path.write_bytes(b"not parquet")
    assert execute_train(c)["status"] == "complete"


def test_zero_reader_workers_reads_synchronously(tmp_path, monkeypatch):
    import recommend.train.comm.datasvr.published_dataset as reader

    config = make_experiment(tmp_path)
    config["execution"]["reader_workers"] = 0
    dataset = PublishedDataset(config)

    def unexpected_pool(*args, **kwargs):
        raise AssertionError("reader_workers=0 must not create a worker pool")

    monkeypatch.setattr(reader, "ThreadPoolExecutor", unexpected_pool)
    assert len(list(dataset.rows("train", {"completion"}))) == 6 * 32


def test_partition_uses_impression_time_when_request_crosses_midnight(tmp_path):
    config = make_experiment(tmp_path)
    dataset = PublishedDataset(config)
    rows = list(dataset.rows("train", {"request_time_ms"}))
    assert len(rows) == 6 * 32
    assert rows[0]["request_time_ms"] < rows[0]["impression_time"].timestamp() * 1000


def test_dropout_keep_probability_and_eval_identity():
    import torch

    from recommend.train.models.rerank.rerank_model import layers

    cfg = {"dims": [32], "acts": ["relu"], "norms": [""]}
    torch.manual_seed(42)
    baseline, _ = layers(16, cfg)
    torch.manual_seed(42)
    dropped, _ = layers(16, {**cfg, "layer_dropout": 0.9})
    assert isinstance(dropped[-1], torch.nn.Dropout)
    assert dropped[-1].p == pytest.approx(0.1)
    x = torch.ones(256, 16)
    dropped.eval()
    assert torch.equal(baseline(x), dropped(x))
    dropped.train()
    torch.manual_seed(7)
    first = dropped(x)
    torch.manual_seed(7)
    assert torch.equal(first, dropped(x))
    assert not torch.equal(first, baseline(x))
    torch.manual_seed(42)
    explicit, _ = layers(16, {**cfg, "layer_dropout": 1.0})
    assert list(explicit.state_dict()) == list(baseline.state_dict())
    assert torch.equal(explicit(x), baseline(x))


@pytest.mark.parametrize("value", [0, -0.1, 1.1])
def test_dropout_schema_rejects_invalid_keep_probability(tmp_path, value):
    config = make_experiment(tmp_path)
    for layer in [config["output_layer"], config["tasks"]["completion"]]:
        layer["layer_dropout"] = value
        with pytest.raises(ValidationError):
            validate_config(config)
        del layer["layer_dropout"]
