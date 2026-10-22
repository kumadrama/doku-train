"""Runtime schema validation and semantic compilation of flat model configurations."""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from recommend.train.comm.expressions import dependencies, transformations

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    json.dumps(config, allow_nan=False)
    common = yaml.safe_load((ROOT / "comm/config.schema.yml").read_text())
    name = config.get("model_type")
    if name not in {"rerank", "rerank_tree"}:
        raise ValueError("unknown model_type; old configuration format is not supported")
    plugin = yaml.safe_load((ROOT / "models" / name / "config.schema.yml").read_text())
    schema = copy.deepcopy(common)
    for key, value in plugin["properties"].items():
        if key == "input_features":
            # Model schemas may add group options without duplicating or weakening common fields.
            group = schema["properties"][key]["additionalProperties"]
            extension = value["additionalProperties"]
            if group["properties"].keys() & extension["properties"].keys():
                raise ValueError("input feature schema ownership conflict")
            group["properties"].update(copy.deepcopy(extension["properties"]))
            group.setdefault("allOf", []).extend(copy.deepcopy(extension.get("allOf", [])))
            continue
        if key in schema["properties"] and key != "selection":
            raise ValueError("schema ownership conflict")
        schema["properties"][key] = value
    schema["required"] = list(set(schema["required"]) | set(plugin["required"]))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(config)
    split = config["split"]
    if split["timezone"] != "UTC":
        raise ValueError("source manifest uses UTC event partitions")
    previous = None
    for part in ["train", "validation", "test"]:
        start, end = (date.fromisoformat(split[f"{part}_{edge}"]) for edge in ["from", "to"])
        if start > end or (previous is not None and start <= previous):
            raise ValueError("date splits must be ordered and disjoint")
        explicit = split.get(f"{part}_dates")
        if explicit is not None:
            dates = [date.fromisoformat(d) for d in explicit]
            if dates != sorted(set(dates)) or dates[0] != start or dates[-1] != end:
                raise ValueError("explicit dates must be sorted and match split bounds")
        previous = end
    features = set(config["features"])
    if features & {"u_account_id", "user_id", "ctx_score", "ctx_position"}:
        raise ValueError("unapproved identifier or serving feature")
    if not set(config["feature_default_values"]) <= features:
        raise ValueError("defaults are allowed only for raw features")
    derived = features.copy()
    for target, expr in transformations(config):
        dependencies(expr)
        derived.add(target)
    for group, options in config["input_features"].items():
        if not set(options["features"]) <= derived:
            raise ValueError(f"unknown input field in {group}")
    if not set(config["hist_options"]) <= set(config["input_features"]):
        raise ValueError("history options reference unknown group")
    for task in config["tasks"].values():
        for key in ["label", "mask", "loss_weight"]:
            dependencies(task[key])
        if task["loss_type"] == "softmax_soft":
            if (
                "soft_label" not in task
                or task.get("num_outputs", 0) < 2
                or task["predict_fn"] != "expectation"
            ):
                raise ValueError("invalid soft-label task")
        elif "soft_label" in task:
            raise ValueError("soft_label requires softmax_soft")
    if name == "rerank":
        if not set(config["dnn_inputs"]) <= set(config["input_features"]):
            raise ValueError("unknown dnn input group")
        for layer in [config["output_layer"], *config["tasks"].values()]:
            if (
                not all(k in layer for k in ["dims", "acts", "norms"])
                or len(layer["dims"]) != len(layer["acts"])
                or len(layer["dims"]) != len(layer["norms"])
            ):
                raise ValueError("each layer requires aligned dims/acts/norms")
        if not set(config["selection"]["task_losses"]) <= set(config["tasks"]):
            raise ValueError("selection references unknown task")
    else:
        if not set(config["selection"]["per_task_metrics"]) == set(config["tasks"]):
            raise ValueError("tree selection must describe every task")
    if "publication" in config:
        if set(config["publication"]) != set(config["tasks"]):
            raise ValueError("publication must cover all tasks")
        for task, rule in config["publication"].items():
            if not config["tasks"][task].get("output_unit"):
                raise ValueError("published task requires explicit output_unit")
            metric = config["tasks"][task]["metrics"].get(rule["metric"])
            if metric is None or metric["func"] not in {"auc", "regression_auc"}:
                raise ValueError("publication requires an explicit ranking metric")
    return copy.deepcopy(config)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    return validate_config(value)
