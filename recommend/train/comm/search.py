"""Deterministic small grids and per-task comparison; no automatic quality gates."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from recommend.train.comm.model_params_validator import load_config, validate_config
from recommend.train.comm.training_pipeline import write_json


def expand_search(
    path: Path, output: Path
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    spec = yaml.safe_load(path.read_text())
    schema = yaml.safe_load(Path(__file__).with_name("search.schema.yml").read_text())
    Draft202012Validator(schema).validate(spec)
    if (
        set(spec) != {"base_configs", "baseline_config", "search", "comparison"}
        or spec["search"]["method"] != "grid"
    ):
        raise ValueError("invalid search configuration")
    if set(spec["search"]) != {"method", "parameters", "fixed"}:
        raise ValueError("unknown search fields")
    space = spec["search"]["parameters"]
    trials = []
    identities = set()
    for name in spec["base_configs"]:
        identity = Path(name).stem
        if identity in identities:
            raise ValueError("duplicate experiment identity")
        identities.add(identity)
        base = load_config(path.parent / name)
        for i, values in enumerate(itertools.product(*space.values())):
            c = copy.deepcopy(base)
            for key, value in {
                **dict(zip(space, values, strict=True)),
                **spec["search"]["fixed"],
            }.items():
                parent = c
                parts = key.split(".")
                for part in parts[:-1]:
                    parent = parent[part]
                if parts[-1] not in parent:
                    raise ValueError("search override refers to unknown field")
                parent[parts[-1]] = value
            label = f"{identity}-{i + 1:02d}"
            c["output"] = {"root": str(output), "run_id": label}
            trials.append((label, validate_config(c)))
    base = load_config(path.parent / spec["baseline_config"])
    if spec["comparison"]["baseline"] != base["model_name"]:
        raise ValueError("comparison baseline identity mismatch")
    for _, candidate in trials:
        if candidate["dataset"] != base["dataset"] or candidate["split"] != base["split"]:
            raise ValueError("search candidates must use the same frozen data and splits")
        for name, rule in spec["comparison"]["tasks"].items():
            left, right = candidate["tasks"][name], base["tasks"][name]
            if (
                left.get("metric_label", left["label"]) != right.get("metric_label", right["label"])
                or left["mask"] != right["mask"]
            ):
                raise ValueError("comparison label or mask semantics differ")
            for key in [rule["loss"]] + ([rule["auc"]] if "auc" in rule else []):
                if left["metrics"][key] != right["metrics"][key]:
                    raise ValueError("comparison metrics differ")
    base["output"] = {
        "root": str(output),
        "run_id": Path(spec["baseline_config"]).stem + "-baseline",
    }
    trials.append((base["output"]["run_id"], validate_config(base)))
    return spec, trials


def compare(
    candidate: dict[str, Any], baseline: dict[str, Any], rule: dict[str, Any]
) -> dict[str, Any]:
    keys = [rule["loss"]] + ([rule["auc"]] if "auc" in rule else [])
    differences: dict[str, float] = {}
    signs = []
    for i, key in enumerate(keys):
        a = candidate["metrics"][key]["value"]
        b = baseline["metrics"][key]["value"]
        if a is None or b is None:
            return {"status": "unavailable", "deltas": differences}
        delta = a - b
        differences[key] = delta
        direction = rule["loss_direction"] if i == 0 else rule["auc_direction"]
        signs.append(0 if delta == 0 else 1 if (delta < 0) == (direction == "min") else -1)
    status = (
        "win"
        if all(x == 1 for x in signs)
        else "loss"
        if all(x == -1 for x in signs)
        else "tie"
        if not any(signs)
        else "tradeoff"
        if 1 in signs and -1 in signs
        else "partial"
    )
    return {"status": status, "deltas": differences}


def execute_search(
    path: Path, output: Path, run: Any, *, local_root: Path | None = None
) -> dict[str, Any]:
    spec, trials = expand_search(path, output)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for label, c in trials:
        print(f"starting {label}", flush=True)
        try:
            results[label] = run(c, local_root=local_root)
        except Exception as e:
            results[label] = {"status": "failed", "error": str(e)}
    baseline = results[trials[-1][0]]
    comparisons = {}
    if baseline["status"] == "complete":
        for label, result in results.items():
            if result["status"] == "complete":
                comparisons[label] = {
                    name: compare(
                        result["metrics"]["validation"][name],
                        baseline["metrics"]["validation"][name],
                        rule,
                    )
                    for name, rule in spec["comparison"]["tasks"].items()
                }
    report = {"results": results, "comparison": comparisons, "test_evaluated": False}
    write_json(output / "search-report.json", report)
    lines = [
        "# Validation search report",
        "",
        "No test data used. No automatic fine search or overall winner.",
        "",
        "| Experiment | Task | Result | Metric deltas vs baseline |",
        "| --- | --- | --- | --- |",
    ]
    lines += ["", "## Configurations and resources", ""]
    for label, cfg in trials:
        result = results[label]
        lines.append(
            f"- {label}: status={result['status']}, "
            f"dims={cfg.get('output_layer', {}).get('dims')}, "
            f"learning_rate={cfg.get('optimizer', {}).get('learning_rate')}, "
            f"resources={result.get('metrics', {}).get('resources', {})}"
        )
    lines += [
        "",
        "## Per-task validation",
        "",
        "| Experiment | Task | Result | Metric deltas vs baseline |",
        "| --- | --- | --- | --- |",
    ]
    for label, tasks in comparisons.items():
        for task, value in tasks.items():
            lines.append(f"| {label} | {task} | {value['status']} | {value['deltas']} |")
    lines += ["", "## Absolute metrics", ""]
    for label, result in results.items():
        if result["status"] == "complete":
            lines.append(f"### {label}")
            for task, values in result["metrics"]["validation"].items():
                lines.append(f"- {task}: " + json.dumps(values, sort_keys=True))
    (output / "search-report.md").write_text("\n".join(lines) + "\n")
    return report
