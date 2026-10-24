import copy

import pytest
import torch
from jsonschema import ValidationError

from recommend.train.comm.model_params_validator import validate_config
from recommend.train.models.rerank.runner import optimizer_parameters
from recommend.train.tests.experiment_fixture import make_experiment


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(3, 2)
        self.shared = torch.nn.Linear(2, 2)
        self.task_names = ["click", "finish"]
        self.towers = torch.nn.ModuleList([torch.nn.Linear(2, 1) for _ in self.task_names])


@pytest.mark.parametrize("cls", [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD])
def test_groups_update_and_resume(cls):
    model = TinyModel()
    opt = {
        "weight_decay": 0.1,
        "weight_decay_overrides": {"embeddings": 0, "tasks": {"click": 0.01}},
    }
    groups = optimizer_parameters(model, opt)
    assert [g["weight_decay"] for g in groups] == [0, 0.1, 0.01, 0.1]
    optimizer = cls(groups, lr=0.01)
    reference = copy.deepcopy(model)
    refopt = cls(
        [
            {"params": reference.embeddings.parameters(), "weight_decay": 0},
            {"params": reference.shared.parameters(), "weight_decay": 0.1},
            {"params": reference.towers[0].parameters(), "weight_decay": 0.01},
            {"params": reference.towers[1].parameters(), "weight_decay": 0.1},
        ],
        lr=0.01,
    )

    def step(m, o):
        for p in m.parameters():
            p.grad = torch.ones_like(p) * 0.2
        o.step()

    step(model, optimizer)
    step(reference, refopt)
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), reference.parameters(), strict=True)
    )
    restored = copy.deepcopy(model)
    resumed = cls(optimizer_parameters(restored, opt), lr=0.01)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(restored, resumed)
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters(), strict=True)
    )
    assert [id(p) for p in optimizer_parameters(model, {"weight_decay": 0.1})] == [
        id(p) for p in model.parameters()
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"shared": -1},
        {"embedding": 0.1},
        {"tasks": {"missing": 0.1}},
        {"tasks": {"completion": -1}},
    ],
)
def test_invalid_overrides(tmp_path, override):
    config = make_experiment(tmp_path)
    config["optimizer"]["weight_decay_overrides"] = override
    with pytest.raises((ValueError, ValidationError)):
        validate_config(config)


def test_valid_overrides(tmp_path):
    config = make_experiment(tmp_path)
    config["optimizer"]["weight_decay_overrides"] = {
        "embeddings": 0,
        "shared": 0.01,
        "tasks": {"completion": 0.001},
    }
    validate_config(config)
