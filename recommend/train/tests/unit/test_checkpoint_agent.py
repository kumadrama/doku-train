from __future__ import annotations

from pathlib import Path

import pytest
import torch

from recommend.train.comm.checkpoint_agent import (
    CheckpointAgent,
    CheckpointLineage,
    CheckpointRole,
)


def lineage(config: str = "config-a") -> CheckpointLineage:
    return CheckpointLineage(config, "dataset-a", "model-a")


def test_checkpoint_paths_are_role_separated_and_atomic(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    agent = CheckpointAgent(tmp_path)
    evaluation = agent.save(
        "run-1",
        CheckpointRole.EVALUATION,
        model,
        optimizer,
        epoch=2,
        step=10,
        lineage=lineage(),
        training_state={"best_loss": 0.4, "stale_epochs": 0},
    )
    production = agent.save(
        "run-1",
        CheckpointRole.PRODUCTION,
        model,
        optimizer,
        epoch=2,
        step=20,
        lineage=lineage(),
        training_state={},
    )
    assert evaluation != production
    assert "/evaluation/" in evaluation.as_posix()
    assert "/production/" in production.as_posix()
    assert not list(tmp_path.rglob("*.tmp"))


def test_restore_rejects_lineage_or_role_mismatch_before_loading(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    agent = CheckpointAgent(tmp_path)
    agent.save(
        "run-1",
        CheckpointRole.EVALUATION,
        model,
        optimizer,
        epoch=2,
        step=10,
        lineage=lineage(),
        training_state={},
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="lineage"):
        agent.restore(
            "run-1",
            CheckpointRole.EVALUATION,
            model,
            optimizer,
            lineage("config-b"),
        )
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    body = agent.path("run-1", CheckpointRole.EVALUATION).read_bytes()
    with pytest.raises(ValueError, match="role"):
        agent.load_bytes(
            body,
            CheckpointRole.PRODUCTION,
            model,
            optimizer,
            lineage(),
        )


def test_production_checkpoint_round_trips_from_bytes(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(2.0)
    source_optimizer = torch.optim.AdamW(source.parameters())
    agent = CheckpointAgent(tmp_path)
    path = agent.save(
        "run-1",
        CheckpointRole.PRODUCTION,
        source,
        source_optimizer,
        epoch=3,
        step=30,
        lineage=lineage(),
        training_state={},
    )
    target = torch.nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target.parameters())
    metadata = agent.load_bytes(
        path.read_bytes(),
        CheckpointRole.PRODUCTION,
        target,
        target_optimizer,
        lineage(),
    )
    assert metadata.epoch == 3
    assert metadata.step == 30
    assert torch.equal(source.weight, target.weight)
