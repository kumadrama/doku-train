from __future__ import annotations

import copy
import io
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CheckpointRole(StrEnum):
    EVALUATION = "evaluation"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class CheckpointLineage:
    config_digest: str
    dataset_manifest_digest: str
    model_structure_digest: str


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    role: CheckpointRole
    epoch: int
    step: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class CheckpointAgent:
    FORMAT_VERSION = "1"

    def __init__(self, root: Path) -> None:
        self._root = root

    def path(self, run_id: str, role: CheckpointRole) -> Path:
        if not run_id or "/" in run_id or ".." in run_id:
            raise ValueError("invalid run_id")
        return self._root / run_id / "checkpoints" / role.value / "checkpoint.pt"

    def save(
        self,
        run_id: str,
        role: CheckpointRole,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        step: int,
        lineage: CheckpointLineage,
        training_state: dict[str, float | int],
    ) -> Path:
        path = self.path(run_id, role)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "role": role.value,
            "epoch": epoch,
            "step": step,
            "lineage": asdict(lineage),
            "training_state": training_state,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix="checkpoint-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return path

    def restore(
        self,
        run_id: str,
        role: CheckpointRole,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expected: CheckpointLineage,
    ) -> CheckpointMetadata:
        return self._load_payload(
            torch.load(self.path(run_id, role), map_location="cpu", weights_only=False),
            role,
            model,
            optimizer,
            expected,
            restore_rng=True,
        )

    def load_bytes(
        self,
        body: bytes,
        role: CheckpointRole,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expected: CheckpointLineage,
    ) -> CheckpointMetadata:
        payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)
        return self._load_payload(payload, role, model, optimizer, expected, restore_rng=False)

    def _load_payload(
        self,
        payload: dict[str, Any],
        role: CheckpointRole,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expected: CheckpointLineage,
        *,
        restore_rng: bool,
    ) -> CheckpointMetadata:
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("CHECKPOINT_INCOMPATIBLE: format version")
        if payload.get("role") != role.value:
            raise ValueError("CHECKPOINT_INCOMPATIBLE: role")
        if payload.get("lineage") != asdict(expected):
            raise ValueError("CHECKPOINT_INCOMPATIBLE: lineage")

        model_before = copy.deepcopy(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        try:
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
        except Exception:
            model.load_state_dict(model_before, strict=True)
            optimizer.load_state_dict(optimizer_before)
            raise
        if restore_rng:
            torch.set_rng_state(payload["torch_rng_state"])
            np.random.set_state(payload["numpy_rng_state"])
            random.setstate(payload["python_rng_state"])
        return CheckpointMetadata(
            role=role,
            epoch=int(payload["epoch"]),
            step=int(payload["step"]),
        )
