"""Generate local, schema-validated experiment configs from an existing frozen run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from recommend.train.comm.model_params_validator import validate_config


def experiments(base):
    for name in ["e00", "e01", "e02", "e03", "e04"]:
        config = copy.deepcopy(base)
        if name == "e01":
            config["tree_inputs"]["categorical"].remove("i_drama_id")
        elif name == "e02":
            config["features"].append("i_episode_id")
            for behavior in ["watched", "finished", "skipped"]:
                field = f"u_{behavior}__episode_id"
                group = f"hist_episode_{behavior}"
                scalar = f"candidate_episode_{behavior}"
                config["features"].append(field)
                config["feature_default_values"][field] = []
                config["input_features"][group] = {"features": [field]}
                config["history_scalars"][scalar] = {
                    "history": group,
                    "func": "contains",
                    "candidate": "i_episode_id",
                }
                config["tree_inputs"]["numeric"].append(scalar)
        elif name == "e03":
            config["hist_options"]["hist_drama_interest"]["max_hist_size"] = 200
        elif name == "e04":
            for group in ["hist_drama_skip", "hist_drama_finish", "hist_drama_immersive_enter"]:
                config["hist_options"].pop(group)
        yield name, config
    changes = [
        {},
        {"max_depth": 4},
        {"max_depth": 8},
        {"min_child_weight": 100},
        {"reg_lambda": 20},
        {"num_boost_round": 1600},
    ]
    for task in base["tasks"]:
        for i, change in enumerate(changes):
            config = copy.deepcopy(base)
            config["tasks"] = {task: config["tasks"][task]}
            config["selection"]["per_task_metrics"] = {
                task: config["selection"]["per_task_metrics"][task]
            }
            config["model_params"].update(change)
            yield f"{task.replace('_', '-')}-t{i:02d}", config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    base = json.loads(args.baseline.read_text())
    args.destination.mkdir(parents=True, exist_ok=False)
    inventory = []
    for name, config in experiments(base):
        config.pop("publication", None)
        config["output"] = {
            "root": str(args.destination / "local-runs"),
            "run_id": f"{args.prefix}-{name}",
        }
        validate_config(config)
        path = args.destination / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        inventory.append(
            {
                "name": name,
                "config": str(path.resolve()),
                "run_id": config["output"]["run_id"],
                "conditional": name == "e04",
            }
        )
    (args.destination / "inventory.json").write_text(json.dumps(inventory, indent=2))
    print(f"Validated {len(inventory)} configs; e04 conditional on audit.")


if __name__ == "__main__":
    main()
