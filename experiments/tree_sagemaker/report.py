"""Summarize verified validation results; never auto-select feature configurations."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.artifacts / "e00/metrics.json").read_text())
    config = json.loads((args.artifacts / "e00/resolved-config.json").read_text())
    lines = [
        "# Tree experiment validation report",
        "",
        "Validation only. Feature decisions remain with the user. No online publication.",
        "",
        "| Experiment | Task | AUC | Δ AUC | Selection metric | Value | Δ value | Rounds |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    candidates = {}

    def render(x):
        return "null" if x is None else f"{x:.6f}"

    for path in sorted(args.artifacts.glob("*/metrics.json")):
        name = path.parent.name
        metrics = json.loads(path.read_text())
        for task, values in metrics["validation"].items():
            selection = config["selection"]["per_task_metrics"][task]
            value = values["metrics"][selection]["value"]
            auc = values["metrics"].get("auc", {}).get("value")
            original = baseline["validation"][task]["metrics"]
            base_auc = original.get("auc", {}).get("value")
            base_value = original[selection]["value"]
            delta_auc = auc - base_auc if auc is not None and base_auc is not None else None
            delta = value - base_value if value is not None and base_value is not None else None
            lines.append(
                f"| {name} | {task} | {render(auc)} | {render(delta_auc)} | "
                f"{selection} | {render(value)} | {render(delta)} | "
                f"{metrics['selection']['rounds'][task]} |"
            )
            if "-t" in name and value is not None:
                candidates.setdefault(task, []).append((value, name))
    lines += [
        "",
        "## Parameter recommendations",
        "",
        "Lowest declared validation metric among single-task runs. "
        "Exploratory recommendations, not independent test evidence.",
        "",
    ]
    for task, values in candidates.items():
        value, name = min(values)
        lines.append(
            f"- {task}: {name}, {config['selection']['per_task_metrics'][task]}={value:.6f}."
        )
    lines += ["", "## Single-task baseline reproducibility", ""]

    def boosters(path):
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("metadata.json"))
            return {task: archive.read(f"{i}.ubj") for i, task in enumerate(metadata["tasks"])}

    reference = boosters(args.artifacts / "e00/checkpoint.xgb.zip")
    for task in config["tasks"]:
        path = args.artifacts / (task.replace("_", "-") + "-t00") / "checkpoint.xgb.zip"
        if path.exists():
            matches = boosters(path)[task] == reference[task]
            comparison = "identical" if matches else "different; investigate equivalence"
            lines.append(f"- {task}: Booster bytes {comparison}.")
    group_rows = []
    directory = args.artifacts.parent / "reports"
    group_baseline_path = directory / "e00-groups.json"
    if group_baseline_path.exists():
        group_baseline = json.loads(group_baseline_path.read_text())["groups"]
        for path in sorted(directory.glob("*-groups.json")):
            name = path.name.removesuffix("-groups.json")
            for group, details in json.loads(path.read_text())["groups"].items():
                for task, result in (details["metrics"] or {}).items():
                    for metric, record in result["metrics"].items():
                        if not isinstance(record, dict):
                            continue
                        value = record["value"]
                        reference = group_baseline[group]["metrics"]
                        before = reference[task]["metrics"][metric]["value"] if reference else None
                        group_rows.append(
                            dict(
                                experiment=name,
                                group=group,
                                rows=details["rows"],
                                task=task,
                                metric=metric,
                                value=value,
                                delta=value - before
                                if value is not None and before is not None
                                else None,
                                reason=record["reason"],
                                valid_rows=record["valid_rows"],
                                **result.get("binary_counts", {}),
                            )
                        )
        if group_rows:
            fields = [
                "experiment",
                "group",
                "rows",
                "task",
                "metric",
                "value",
                "delta",
                "reason",
                "valid_rows",
                "positive",
                "negative",
            ]
            with (args.output.parent / "group-validation.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(group_rows)
            lines += ["", "Detailed group metrics and baseline deltas: group-validation.csv", ""]
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
