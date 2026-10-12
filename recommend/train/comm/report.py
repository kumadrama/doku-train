"""Aggregate only explicitly evaluated, frozen runs into a final report."""

from __future__ import annotations

import json
from pathlib import Path


def write_final_report(run_dirs: list[Path], output: Path) -> None:
    lines = [
        "# Final test report",
        "",
        "Each row uses one selected checkpoint per run.",
        "",
        "| Run | Task | Metric | Value | Reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for root in run_dirs:
        status = json.loads((root / "status.json").read_text())
        if status["status"] != "complete" or status["selection"]["status"] != "selected":
            raise ValueError("report requires successfully selected runs")
        metrics = json.loads((root / "test-metrics.json").read_text())
        for task, values in metrics.items():
            for metric, value in values["metrics"].items():
                if isinstance(value, dict):
                    lines.append(
                        f"| {root.name} | {task} | {metric} | "
                        f"{value['value']} | {value['reason']} |"
                    )
    with output.open("x") as f:
        f.write("\n".join(lines) + "\n")
