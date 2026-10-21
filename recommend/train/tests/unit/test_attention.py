from __future__ import annotations

import copy
import math

import pytest
import torch
from jsonschema import ValidationError

from recommend.train.comm.model_params_validator import validate_config
from recommend.train.models.rerank.attention import history_attention
from recommend.train.models.rerank.rerank_model import ConfiguredRerank
from recommend.train.tests.synthetic_config import synthetic_config


def attention_fixture():
    config = synthetic_config("dnn")
    config["input_features"] = {
        "history": {"features": ["h"], "combiner": "mean"},
        "candidate": {"features": ["q"], "combiner": "concat"},
    }
    config["embed_dim"] = {"*": 2}
    config["dnn_inputs"] = ["history"]
    config["output_layer"] = {"dims": [], "acts": [], "norms": []}
    config["tasks"] = {"score": {"dims": [], "acts": [], "num_outputs": 1}}
    config["hist_attn_configs"] = {"history": {"query": "candidate", "attn_scale": 1.0}}
    state = {
        "vocabs": {"items": [10, 20]},
        "tables": {"q": "items", "h": "items"},
        "dtypes": {"q": "int64", "h": "list<int64>"},
    }
    return config, state


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
    empty = history_attention(q, h[:, :0], valid[:, :0], 1.0)
    assert torch.equal(empty, torch.zeros_like(q))


def test_attention_group_order_oov_fm_and_no_new_parameters():
    config, state = attention_fixture()
    config["fm_layers"] = [{"fm_type": "embed", "features": [["candidate", "history"]]}]
    torch.manual_seed(7)
    model = ConfiguredRerank(config, state)
    no_attention = copy.deepcopy(config)
    del no_attention["hist_attn_configs"]
    torch.manual_seed(7)
    baseline = ConfiguredRerank(no_attention, state)
    assert model.state_dict().keys() == baseline.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, baseline.state_dict()[name])
    with torch.no_grad():
        model.embeddings[0].weight.copy_(
            torch.tensor([[0.0, 0.0], [0.0, 2.0], [1.0, 0.0], [2.0, 0.0]])
        )
        model.towers[0][-1].weight.copy_(torch.tensor([[0.0, 0.0, 1.0]]))
        model.towers[0][-1].bias.zero_()
    batch = {"q": torch.tensor([[2]]), "h": torch.tensor([[2, 1, 0]])}
    # OOV=1 participates; padding=0 does not. History group precedes query in YAML.
    a = math.exp(1) / (math.exp(1) + 1)
    # Isolate the FM output: dot(candidate, attention-pooled history).
    torch.testing.assert_close(model(batch)["score"], torch.tensor([[a]]))
    config["fm_layers"] = []
    plain = ConfiguredRerank(config, state)
    with torch.no_grad():
        plain.embeddings[0].weight.copy_(model.embeddings[0].weight)
        plain.towers[0][-1].weight.copy_(torch.tensor([[1.0, 0.0]]))
        plain.towers[0][-1].bias.zero_()
    torch.testing.assert_close(plain(batch)["score"], torch.tensor([[a]]))
    plain(batch)["score"].sum().backward()
    assert plain.embeddings[0].weight.grad[1].abs().sum() > 0
    assert plain.embeddings[0].weight.grad[2].abs().sum() > 0
    assert plain.embeddings[0].weight.grad[0].abs().sum() == 0
    training_output = plain(batch)["score"].detach()
    plain.eval()
    assert torch.equal(training_output, plain(batch)["score"])


def test_absent_attention_preserves_mean_weighted_mean():
    config, state = attention_fixture()
    del config["hist_attn_configs"]
    for combiner in ["mean", "weighted_mean"]:
        config["input_features"]["history"]["combiner"] = combiner
        if combiner == "weighted_mean":
            config["input_features"]["history"]["weight_feature"] = "w"
        model = ConfiguredRerank(config, state)
        with torch.no_grad():
            model.embeddings[0].weight.copy_(
                torch.tensor([[0.0, 0.0], [0.0, 2.0], [1.0, 0.0], [2.0, 0.0]])
            )
            model.towers[0][-1].weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.towers[0][-1].bias.zero_()
        batch = {
            "q": torch.tensor([[2]]),
            "h": torch.tensor([[2, 3, 0]]),
            "w": torch.tensor([[1.0, 3.0, 100.0]]),
        }
        expected = 1.5 if combiner == "mean" else 1.75
        torch.testing.assert_close(model(batch)["score"], torch.tensor([[expected]]))


@pytest.mark.parametrize("scale", [0, -1, True, "1", float("inf"), float("nan")])
def test_schema_rejects_invalid_attention_scale(scale):
    config = synthetic_config("dnn")
    config["hist_attn_configs"] = {"hist_drama_skip": {"query": "item_id", "attn_scale": scale}}
    with pytest.raises((ValidationError, ValueError)):
        validate_config(config)


def test_schema_supports_nn_only_and_rejects_unknown_options():
    for model in ["dnn", "deepfm"]:
        config = synthetic_config(model)
        config["hist_attn_configs"] = {"hist_drama_skip": {"query": "item_id"}}
        validate_config(config)
        config["hist_attn_configs"]["hist_drama_skip"]["projection"] = True
        with pytest.raises(ValidationError):
            validate_config(config)
    config = synthetic_config("xgboost")
    config["hist_attn_configs"] = {}
    with pytest.raises(ValidationError):
        validate_config(config)


@pytest.mark.parametrize(
    "problem",
    [
        "unknown_history",
        "unknown_query",
        "self",
        "sequence_query",
        "scalar_history",
        "numeric",
        "dimensions",
        "weighted",
        "multiple",
    ],
)
def test_invalid_attention_semantics(problem):
    config, state = attention_fixture()
    spec = config["hist_attn_configs"]["history"]
    if problem == "unknown_history":
        config["hist_attn_configs"] = {"unknown": spec}
    elif problem == "unknown_query":
        spec["query"] = "unknown"
    elif problem == "self":
        spec["query"] = "history"
    elif problem == "sequence_query":
        state["dtypes"]["q"] = "list<int64>"
    elif problem == "scalar_history":
        state["dtypes"]["h"] = "int64"
    elif problem == "numeric":
        del state["tables"]["q"]
    elif problem == "dimensions":
        state["tables"]["q"] = "other"
        state["vocabs"]["other"] = [10]
        config["embed_dim"]["other"] = 3
    elif problem == "weighted":
        config["input_features"]["history"]["weight_feature"] = "w"
    else:
        config["input_features"]["history"]["features"].append("q")
    with pytest.raises(ValueError, match="attention"):
        ConfiguredRerank(config, state)
