from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
import torch

from recommend.train.comm.checkpoint_agent import CheckpointAgent, CheckpointLineage
from recommend.train.comm.eval.gates import GateResult, GateSeverity
from recommend.train.comm.training_pipeline import BatchFactories, PipelineBlocked, TrainingPipeline


class Adapter:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1)

    def loss(self, batch: torch.Tensor) -> torch.Tensor:
        return self.module(batch).square().mean()

    def logits(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"score": self.module(batch).squeeze(1)}


def batches() -> Callable[[], Iterable[torch.Tensor]]:
    return lambda: (torch.ones(2, 1),)


def pipeline(
    tmp_path: Path,
    *,
    factory: Callable[[str], Adapter],
    gate: GateResult,
    memory_limit_bytes: int = 1_000_000_000_000,
) -> TrainingPipeline[torch.Tensor, dict[str, float]]:
    return TrainingPipeline(
        adapter_factory=factory,
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        evaluator=lambda _adapter, _batches: {"auc": 0.4},
        gate_builder=lambda _metrics: (gate,),
        checkpoint_agent=CheckpointAgent(tmp_path),
        run_id="run-1",
        lineage=CheckpointLineage("config", "dataset", "model"),
        seed=7,
        max_epochs=1,
        patience=1,
        memory_limit_bytes=memory_limit_bytes,
    )


def test_success_refits_fresh_model_and_saves_production_checkpoint(tmp_path: Path) -> None:
    created: list[Adapter] = []

    def factory(_role: str) -> Adapter:
        created.append(Adapter())
        return created[-1]

    result = pipeline(
        tmp_path,
        factory=factory,
        gate=GateResult("auc", False, GateSeverity.WARN, "soft warning"),
    ).run(BatchFactories(batches(), batches(), batches()))
    assert len(created) == 2
    assert created[0] is not created[1]
    assert result.production_adapter is created[1]
    assert result.selected_epoch == 1
    assert result.production_checkpoint.is_file()
    assert "/production/" in result.production_checkpoint.as_posix()
    assert [trace.role for trace in result.traces] == ["evaluation", "production"]


def test_blocking_gate_prevents_refit(tmp_path: Path) -> None:
    created: list[Adapter] = []
    runner = pipeline(
        tmp_path,
        factory=lambda _role: created.append(Adapter()) or created[-1],
        gate=GateResult("labels", False, GateSeverity.BLOCK, "not evaluable"),
    )
    with pytest.raises(PipelineBlocked, match="EVALUATION_GATE_FAILED") as captured:
        runner.run(BatchFactories(batches(), batches(), batches()))
    assert captured.value.stage == "evaluate"
    assert len(created) == 1


def test_non_finite_loss_stops_before_refit(tmp_path: Path) -> None:
    class NonFiniteAdapter(Adapter):
        def loss(self, batch: torch.Tensor) -> torch.Tensor:
            return super().loss(batch) * torch.tensor(float("nan"))

    runner = pipeline(
        tmp_path,
        factory=lambda _role: NonFiniteAdapter(),
        gate=GateResult("ok", True, GateSeverity.PASS, "ok"),
    )
    with pytest.raises(PipelineBlocked, match="NUMERICAL_FAILURE"):
        runner.run(BatchFactories(batches(), batches(), batches()))


def test_non_finite_refit_reports_refit_stage(tmp_path: Path) -> None:
    class NonFiniteAdapter(Adapter):
        def loss(self, batch: torch.Tensor) -> torch.Tensor:
            return super().loss(batch) * torch.tensor(float("nan"))

    runner = pipeline(
        tmp_path,
        factory=lambda role: Adapter() if role == "evaluation" else NonFiniteAdapter(),
        gate=GateResult("ok", True, GateSeverity.PASS, "ok"),
    )
    with pytest.raises(PipelineBlocked, match="NUMERICAL_FAILURE") as captured:
        runner.run(BatchFactories(batches(), batches(), batches()))
    assert captured.value.stage == "refit"


def test_memory_budget_stops_the_active_stage(tmp_path: Path) -> None:
    runner = pipeline(
        tmp_path,
        factory=lambda _role: Adapter(),
        gate=GateResult("ok", True, GateSeverity.PASS, "ok"),
        memory_limit_bytes=1,
    )
    with pytest.raises(PipelineBlocked, match="RESOURCE_BUDGET_EXCEEDED") as captured:
        runner.run(BatchFactories(batches(), batches(), batches()))
    assert captured.value.stage == "evaluate"


def test_pipeline_has_no_exporter_or_pre_refit_callback() -> None:
    parameters = inspect.signature(TrainingPipeline.__init__).parameters
    assert "exporter" not in parameters
    assert "pre_refit_validator" not in parameters
    source = inspect.getsource(TrainingPipeline).casefold()
    assert "onnx" not in source
    assert "model_exporter" not in source
