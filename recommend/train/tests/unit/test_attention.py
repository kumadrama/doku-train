from __future__ import annotations

import copy
import math

import pytest
import torch
from jsonschema import ValidationError

from recommend.train.comm.feature_plan import FeaturePlan
from recommend.train.comm.model_params_validator import validate_config
from recommend.train.models.rerank.attention import history_attention
from recommend.train.models.rerank.rerank_model import ConfiguredRerank
from recommend.train.tests.synthetic_config import synthetic_config


def attention_fixture():
    config = synthetic_config("dnn")
    config["input_features"] = {
        "history": {
            "features": ["h"],
            "combiner": "concat",
            "history_pooling": "attention",
            "hist_attn_query": "candidate",
        },
        "candidate": {"features": ["q"], "combiner": "concat"},
    }
    config["embed_dim"] = {"*": 2}
    config["dnn_inputs"] = ["history"]
    config["output_layer"] = {"dims": [], "acts": [], "norms": []}
    config["tasks"] = {"score": {"dims": [], "acts": [], "num_outputs": 1}}
    config["attn_scale"] = 1.0
    state = {
        "vocabs": {"items": [10, 20]},
        "tables": {"q": "items", "h": "items"},
        "dtypes": {"q": "int64", "h": "list<int64>"},
    }
    return config, state


def set_embeddings(model):
    with torch.no_grad():
        model.embeddings[0].weight.copy_(
            torch.tensor([[0.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
        )


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_attention_hand_calculation_and_padding(scale):
    q = torch.tensor([[1.0, 0.0]])
    h = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1000.0, 1000.0]]])
    valid = torch.tensor([[True, True, False]])
    a = math.exp(1 / scale) / (math.exp(1 / scale) + 1)
    expected = torch.tensor([[a, 2 * (1 - a)]])
    torch.testing.assert_close(history_attention(q, h, valid, scale), expected)
    torch.testing.assert_close(history_attention(q, h[:, :2], valid[:, :2], scale), expected)
    order = [1, 2, 0]
    torch.testing.assert_close(history_attention(q, h[:, order], valid[:, order], scale), expected)


def test_empty_history_and_gradients_are_finite():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    h = torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]], requires_grad=True)
    valid = torch.tensor([[True, True], [False, False]])
    out = history_attention(q, h, valid, 1.0)
    assert torch.equal(out[1], torch.zeros(2))
    out.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(h.grad).all()
    assert q.grad[0].abs().sum() > 0 and h.grad[0].abs().sum() > 0
    assert torch.count_nonzero(q.grad[1]) == torch.count_nonzero(h.grad[1]) == 0
    assert torch.equal(history_attention(q, h[:, :0], valid[:, :0], 1.0), torch.zeros_like(q))


def test_attention_oov_fm_and_no_new_parameters():
    config, state = attention_fixture()
    config["fm_layers"] = [{"fm_type": "embed", "features": [["candidate", "history"]]}]
    torch.manual_seed(7)
    model = ConfiguredRerank(config, state)
    baseline_config = copy.deepcopy(config)
    baseline_config["input_features"]["history"]["history_pooling"] = "mean"
    del baseline_config["input_features"]["history"]["hist_attn_query"]
    torch.manual_seed(7)
    baseline = ConfiguredRerank(baseline_config, state)
    assert model.state_dict().keys() == baseline.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, baseline.state_dict()[name])
    set_embeddings(model)
    with torch.no_grad():
        model.towers[0][-1].weight.copy_(torch.tensor([[0.0, 0.0, 1.0]]))
        model.towers[0][-1].bias.zero_()
    batch = {"q": torch.tensor([[2]]), "h": torch.tensor([[2, 1, 0]])}
    a = math.exp(1) / (math.exp(1) + 1)
    torch.testing.assert_close(model(batch)["score"], torch.tensor([[a]]))
    model(batch)["score"].sum().backward()
    assert model.embeddings[0].weight.grad[1].abs().sum() > 0
    assert model.embeddings[0].weight.grad[2].abs().sum() > 0
    assert model.embeddings[0].weight.grad[0].abs().sum() == 0


@pytest.mark.parametrize("combiner", ["concat", "sum", "avg"])
@pytest.mark.parametrize("pooling", ["mean", "weighted_mean", "attention", "none"])
def test_field_combination_then_history_reduction(combiner, pooling):
    config, state = attention_fixture()
    history = config["input_features"]["history"]
    candidate = config["input_features"]["candidate"]
    history.update(features=["h", "h2"], combiner=combiner, history_pooling=pooling)
    candidate.update(features=["q", "q2"], combiner=combiner)
    state["tables"].update(h2="items", q2="items")
    state["dtypes"].update(h2="list<int64>", q2="int64")
    if pooling != "attention":
        del history["hist_attn_query"]
    if pooling == "weighted_mean":
        history["weight_feature"] = "w"
    if pooling == "none":
        config["dnn_inputs"] = ["candidate"]
    model = ConfiguredRerank(config, state)
    set_embeddings(model)
    batch = {
        "q": torch.tensor([[2]]),
        "q2": torch.tensor([[3]]),
        "h": torch.tensor([[2, 3, 0]]),
        "h2": torch.tensor([[3, 3, 0]]),
        "w": torch.tensor([[1.0, 3.0, 100.0]]),
    }
    # Independent hand-combined positions: each field is combined at the same timestamp.
    if combiner == "concat":
        positions = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
        logits = [2.0, 1.0]
    elif combiner == "sum":
        positions = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
        logits = [2.0, 2.0]
    else:
        positions = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
        logits = [0.5, 0.5]
    if pooling == "none":
        expected = torch.cat([positions, torch.zeros_like(positions[:1])]).unsqueeze(0)
    else:
        if pooling == "attention":
            a = math.exp(logits[0]) / sum(math.exp(x) for x in logits)
        else:
            a = 0.25 if pooling == "weighted_mean" else 0.5
        expected = (a * positions[0] + (1 - a) * positions[1]).unsqueeze(0)
    torch.testing.assert_close(model.input_groups(batch)["history"], expected)
    # Query group is declared after history; no order-dependent lookup or premature mean.
    assert model.group_dims["history"] == positions.shape[-1]


def test_alignment_and_unreduced_history_checks():
    config, state = attention_fixture()
    config["input_features"]["history"]["features"].append("h2")
    config["input_features"]["history"]["combiner"] = "sum"
    state["tables"]["h2"] = "items"
    state["dtypes"]["h2"] = "list<int64>"
    model = ConfiguredRerank(config, state)
    with pytest.raises(ValueError, match="aligned"):
        model.input_groups(
            {"q": torch.tensor([[2]]), "h": torch.tensor([[2, 0]]), "h2": torch.tensor([[2, 3]])}
        )
    config["input_features"]["history"]["history_pooling"] = "none"
    del config["input_features"]["history"]["hist_attn_query"]
    with pytest.raises(ValueError, match="unreduced"):
        ConfiguredRerank(config, state)


def test_multi_field_history_truncation_and_alignment():
    config, _ = attention_fixture()
    config.update(
        features=["q", "h", "h2"],
        feature_default_values={"h": [], "h2": []},
        feature_transformations=[],
        item_filter="True",
        tasks={},
        embed_name={"q": "items", "h": "items", "h2": "items"},
    )
    config["input_features"]["history"]["features"] = ["h", "h2"]
    config["hist_options"] = {
        "history": {"max_hist_size": 2, "truncate": "tail", "duplicates": "preserve"}
    }
    plan = FeaturePlan(config, {"q": "int64", "h": "list<int64>", "h2": "list<int64>"})
    row = {"q": 10, "h": [1, 2, 3], "h2": [4, 5, 6]}
    processed = plan.row(row)
    assert processed["h"] == [2, 3] and processed["h2"] == [5, 6]
    plan.fit(iter([row]))
    encoded = plan.encode(processed)
    assert encoded["h"].shape == encoded["h2"].shape == (2,)
    with pytest.raises(ValueError, match="aligned lengths"):
        plan.row({"q": 10, "h": [1], "h2": [4, 5]})


@pytest.mark.parametrize("scale", [0, -1, True, "1", float("inf"), float("nan")])
def test_schema_rejects_invalid_attention_scale(scale):
    config = synthetic_config("dnn")
    config["attn_scale"] = scale
    with pytest.raises((ValidationError, ValueError)):
        validate_config(config)


@pytest.mark.parametrize("model", ["dnn", "deepfm"])
def test_schema_separates_combination_and_pooling(model):
    config = synthetic_config(model)
    validate_config(config)
    group = config["input_features"]["hist_drama_skip"]
    for old in ["mean", "weighted_mean"]:
        group["combiner"] = old
        with pytest.raises(ValidationError):
            validate_config(config)
    group["combiner"] = "avg"
    group["history_pooling"] = "attention"
    with pytest.raises(ValidationError):
        validate_config(config)
    group["hist_attn_query"] = "item_id"
    validate_config(config)
    group["skip_hist_attn"] = False
    with pytest.raises(ValidationError):
        validate_config(config)
    del group["skip_hist_attn"]
    group["history_pooling"] = "mean"
    with pytest.raises(ValidationError):
        validate_config(config)
    del group["hist_attn_query"]
    del group["features"]
    with pytest.raises(ValidationError):
        validate_config(config)


def test_tree_schema_does_not_gain_nn_pooling():
    config = synthetic_config("xgboost")
    config["input_features"]["hist_drama_skip"]["history_pooling"] = "attention"
    with pytest.raises(ValidationError):
        validate_config(config)


@pytest.mark.parametrize(
    "problem",
    [
        "unknown_query",
        "sequence_query",
        "scalar_history",
        "dimensions",
        "weighted",
        "no_pooling",
        "mixed",
        "sum_dims",
    ],
)
def test_invalid_group_semantics(problem):
    config, state = attention_fixture()
    history = config["input_features"]["history"]
    if problem == "unknown_query":
        history["hist_attn_query"] = "unknown"
    elif problem == "sequence_query":
        state["dtypes"]["q"] = "list<int64>"
        config["input_features"]["candidate"]["history_pooling"] = "mean"
    elif problem == "scalar_history":
        state["dtypes"]["h"] = "int64"
    elif problem == "dimensions":
        state["tables"]["q"] = "other"
        config["embed_dim"]["other"] = 3
    elif problem == "weighted":
        history["weight_feature"] = "w"
    elif problem == "no_pooling":
        del history["history_pooling"]
    elif problem == "mixed":
        history["features"].append("q")
    else:
        config["input_features"]["candidate"].update(features=["q", "n"], combiner="sum")
        state["dtypes"]["n"] = "float32"
    with pytest.raises(ValueError):
        ConfiguredRerank(config, state)


def test_shared_history_capacity_is_independent_of_group_order():
    config, _ = attention_fixture()
    config.update(
        features=["q", "h"],
        feature_default_values={"h": []},
        feature_transformations=[],
        item_filter="True",
        tasks={},
        embed_name={"q": "items", "h": "items"},
    )
    config["input_features"]["second_view"] = {
        "features": ["h"],
        "combiner": "concat",
        "history_pooling": "mean",
    }
    config["hist_options"] = {
        "history": {"max_hist_size": 5, "truncate": "tail", "duplicates": "preserve"}
    }
    row = {"q": 1, "h": [1, 2]}
    for reverse in [False, True]:
        config["input_features"] = dict(sorted(config["input_features"].items(), reverse=reverse))
        plan = FeaturePlan(config, {"q": "int64", "h": "list<int64>"})
        plan.fit(iter([row]))
        assert plan.encode(plan.row(row))["h"].shape == (5,)
    config["hist_options"]["second_view"] = {
        "max_hist_size": 3,
        "truncate": "tail",
        "duplicates": "preserve",
    }
    with pytest.raises(ValueError, match="conflicting truncation"):
        FeaturePlan(config, {"q": "int64", "h": "list<int64>"})
