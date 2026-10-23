from __future__ import annotations

import copy

import yaml

from recommend.train.tests.experiment_fixture import make_experiment
from recommend.train.tests.synthetic_config import synthetic_config

FIELD = "u_expose_count_7d_all"


def configure_tree_binning(config, path):
    teacher = synthetic_config("xgboost")
    teacher["dataset"] = copy.deepcopy(config["dataset"])
    teacher["split"] = copy.deepcopy(config["split"])
    teacher["execution"] = copy.deepcopy(config["execution"])
    teacher["tree_inputs"]["numeric"] = [FIELD, "i_episode_dur_s"]
    teacher["model_params"].update(num_boost_round=3, max_depth=2, min_child_weight=1)
    path.write_text(yaml.safe_dump(teacher))
    config["tree_binning"] = {"enabled": True, "config_path": str(path)}
    target = FIELD + "_tree"
    config["feature_transformations"].append(
        {"expr": "[*]_tree = tree([*])", "feature_list": [FIELD]}
    )
    config["embed_name"][target] = target
    config["input_features"]["tree_stats"] = {"features": [target], "combiner": "concat"}
    config["dnn_inputs"].append("tree_stats")
    return config


def make_tree_experiment(root):
    return configure_tree_binning(make_experiment(root, "deepfm"), root / "teacher.yaml")
