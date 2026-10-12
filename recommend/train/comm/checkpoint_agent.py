"""Framework-owned checkpoint I/O; model code only serializes in-memory payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CheckpointAgent:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save_resume(self, payload: dict[str, Any]) -> None:
        import torch

        temporary = self.directory / "resume.partial"
        torch.save(payload, temporary)
        temporary.replace(self.directory / "resume.pt")

    def save_bytes(self, name: str, payload: bytes) -> None:
        if name not in {"checkpoint.pt", "checkpoint.xgb.zip"}:
            raise ValueError("unexpected checkpoint name")
        with (self.directory / name).open("xb") as f:
            f.write(payload)

    @staticmethod
    def read_resume(path: Path) -> dict[str, Any]:
        import torch

        value: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        return value
