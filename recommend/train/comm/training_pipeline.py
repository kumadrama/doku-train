from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch

from recommend.train.comm.checkpoint_agent import (
    CheckpointAgent,
    CheckpointLineage,
    CheckpointRole,
    seed_everything,
)
from recommend.train.comm.eval.gates import GateResult, GateSeverity
from recommend.train.comm.model_protocol import AdapterFactory, ModelAdapter


class PipelineBlocked(RuntimeError):
    def __init__(self, message: str, *, stage: str) -> None:
        self.stage = stage
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BatchFactories[BatchT]:
    train: Callable[[], Iterable[BatchT]]
    validation: Callable[[], Iterable[BatchT]]
    refit: Callable[[], Iterable[BatchT]]


@dataclass(frozen=True, slots=True)
class EpochTrace:
    role: str
    epoch: int
    step_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PipelineResult[BatchT, MetricT]:
    selected_epoch: int
    metrics: MetricT
    gates: tuple[GateResult, ...]
    production_adapter: ModelAdapter[BatchT]
    production_checkpoint: Path
    traces: tuple[EpochTrace, ...]


class TrainingPipeline[BatchT, MetricT]:
    def __init__(
        self,
        *,
        adapter_factory: AdapterFactory[BatchT],
        optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        evaluator: Callable[[ModelAdapter[BatchT], Iterable[BatchT]], MetricT],
        gate_builder: Callable[[MetricT], tuple[GateResult, ...]],
        checkpoint_agent: CheckpointAgent,
        run_id: str,
        lineage: CheckpointLineage,
        seed: int,
        max_epochs: int,
        patience: int,
        memory_limit_bytes: int,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._optimizer_factory = optimizer_factory
        self._evaluator = evaluator
        self._gate_builder = gate_builder
        self._checkpoint_agent = checkpoint_agent
        self._run_id = run_id
        self._lineage = lineage
        self._seed = seed
        self._max_epochs = max_epochs
        self._patience = patience
        self._memory_limit_bytes = memory_limit_bytes

    def _check_memory(self, stage: str) -> None:
        if psutil.Process().memory_info().rss > self._memory_limit_bytes:
            raise PipelineBlocked("RESOURCE_BUDGET_EXCEEDED", stage=stage)

    def _train_epoch(
        self,
        adapter: ModelAdapter[BatchT],
        optimizer: torch.optim.Optimizer,
        batches: Iterable[BatchT],
        *,
        stage: str,
    ) -> tuple[float, tuple[float, ...]]:
        adapter.module.train()
        losses: list[float] = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = adapter.loss(batch)
            if not torch.isfinite(loss):
                raise PipelineBlocked("NUMERICAL_FAILURE", stage=stage)
            torch.autograd.backward(loss)
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in adapter.module.parameters()
            ):
                raise PipelineBlocked("NUMERICAL_FAILURE", stage=stage)
            optimizer.step()
            if any(
                not torch.isfinite(parameter).all() for parameter in adapter.module.parameters()
            ):
                raise PipelineBlocked("NUMERICAL_FAILURE", stage=stage)
            losses.append(float(loss.detach()))
            self._check_memory(stage)
        if not losses:
            raise PipelineBlocked("DATA_EXHAUSTED", stage=stage)
        return sum(losses) / len(losses), tuple(losses)

    @staticmethod
    def _mean_loss(
        adapter: ModelAdapter[BatchT], batches: Iterable[BatchT], *, stage: str
    ) -> float:
        adapter.module.eval()
        values: list[float] = []
        with torch.inference_mode():
            for batch in batches:
                loss = adapter.loss(batch)
                if not torch.isfinite(loss):
                    raise PipelineBlocked("NUMERICAL_FAILURE", stage=stage)
                values.append(float(loss))
        if not values:
            raise PipelineBlocked("DATA_EXHAUSTED", stage=stage)
        return sum(values) / len(values)

    def run(self, batches: BatchFactories[BatchT]) -> PipelineResult[BatchT, MetricT]:
        seed_everything(self._seed)
        evaluation_adapter = self._adapter_factory("evaluation")
        evaluation_optimizer = self._optimizer_factory(evaluation_adapter.module.parameters())
        selected_epoch = 0
        best_loss = math.inf
        stale_epochs = 0
        evaluation_step = 0
        traces: list[EpochTrace] = []
        for epoch in range(1, self._max_epochs + 1):
            _, step_losses = self._train_epoch(
                evaluation_adapter,
                evaluation_optimizer,
                batches.train(),
                stage="evaluate",
            )
            evaluation_step += len(step_losses)
            traces.append(EpochTrace("evaluation", epoch, step_losses))
            validation_loss = self._mean_loss(
                evaluation_adapter,
                batches.validation(),
                stage="evaluate",
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                selected_epoch = epoch
                stale_epochs = 0
                self._checkpoint_agent.save(
                    self._run_id,
                    CheckpointRole.EVALUATION,
                    evaluation_adapter.module,
                    evaluation_optimizer,
                    epoch=epoch,
                    step=evaluation_step,
                    lineage=self._lineage,
                    training_state={"best_loss": best_loss, "stale_epochs": stale_epochs},
                )
            else:
                stale_epochs += 1
            if stale_epochs >= self._patience:
                break
        if selected_epoch == 0:
            raise PipelineBlocked("DATA_EXHAUSTED", stage="evaluate")
        self._checkpoint_agent.restore(
            self._run_id,
            CheckpointRole.EVALUATION,
            evaluation_adapter.module,
            evaluation_optimizer,
            self._lineage,
        )
        metrics = self._evaluator(evaluation_adapter, batches.validation())
        gates = self._gate_builder(metrics)
        blocked = [gate.name for gate in gates if gate.severity is GateSeverity.BLOCK]
        if blocked:
            raise PipelineBlocked(
                "EVALUATION_GATE_FAILED:" + ",".join(blocked),
                stage="evaluate",
            )

        seed_everything(self._seed + 1)
        production_adapter = self._adapter_factory("production")
        production_optimizer = self._optimizer_factory(production_adapter.module.parameters())
        production_step = 0
        for epoch in range(1, selected_epoch + 1):
            _, step_losses = self._train_epoch(
                production_adapter,
                production_optimizer,
                batches.refit(),
                stage="refit",
            )
            production_step += len(step_losses)
            traces.append(EpochTrace("production", epoch, step_losses))
        production_checkpoint = self._checkpoint_agent.save(
            self._run_id,
            CheckpointRole.PRODUCTION,
            production_adapter.module,
            production_optimizer,
            epoch=selected_epoch,
            step=production_step,
            lineage=self._lineage,
            training_state={},
        )
        return PipelineResult(
            selected_epoch=selected_epoch,
            metrics=metrics,
            gates=gates,
            production_adapter=production_adapter,
            production_checkpoint=production_checkpoint,
            traces=tuple(traces),
        )
