"""Explicit Doku experiment → Finder current-path configuration adapter."""

from copy import deepcopy

from .model_params import parameters


def migrate_experiment(experiment, raw_dtypes, optimizer_settings, shuffle_buffer_size):
    """No implicit conversion of Torch weight_decay; caller selects Finder settings."""
    allowed_optimizer = {
        "optimizer",
        "learning_rate",
        "optimizer_args",
        "sparse_optimizer",
        "sparse_learning_rate",
        "sparse_optimizer_args",
    }
    if set(optimizer_settings) - allowed_optimizer or "optimizer" not in optimizer_settings:
        raise ValueError("Explicit Finder optimizer configuration required")
    source = deepcopy(experiment)
    model = {
        key: deepcopy(source[key]) for key in ("embed_name", "embed_dim", "output_layer", "tasks")
    }
    model["embed_initializer_args"] = source["embed_initializer_args"]
    if source.get("embed_initializer", "uniform") != "uniform":
        raise ValueError("Only current uniform embedding initializer is in scope")
    model["attn_scale"] = source.get("attn_scale", 1.0)
    groups = {}
    replacements = {}
    for name, group in source["input_features"].items():
        group = deepcopy(group)
        if group.get("history_pooling"):
            group.update(input_type="embed", **{"from": name})
            if any(f not in model["embed_name"] for f in group["features"]):
                raise ValueError("History fields require embedding declarations")
            groups[name] = group
            replacements[name] = [name]
            continue
        # Preserve adjacent type runs, not a blanket scalar-first reorder.
        runs = []
        for field in group["features"]:
            kind = "embed" if field in model["embed_name"] else "scalar"
            if not runs or runs[-1][0] != kind:
                runs.append((kind, []))
            runs[-1][1].append(field)
        replacements[name] = []
        for index, (kind, fields) in enumerate(runs):
            target = name if len(runs) == 1 else name + "_" + kind + "_" + str(index)
            groups[target] = {**group, "input_type": kind, "features": fields, "from": "item"}
            replacements[name].append(target)
    model["input_features"] = groups
    model["dnn_inputs"] = [n for name in source["dnn_inputs"] for n in replacements[name]]
    model["fm_layers"] = []
    for block in source.get("fm_layers", []):
        if block["fm_type"] != "embed":
            raise ValueError("Only embedding FM is in scope")
        for pair in block["features"]:
            if any(len(replacements[name]) != 1 for name in pair):
                raise ValueError("Cannot implicitly split FM group")
            model["fm_layers"].append([replacements[name][0] for name in pair])
    model["feature_transformations"] = []
    for trans in source["feature_transformations"]:
        model["feature_transformations"].append(
            {
                "expr": trans["expr"],
                "apply_to_features": trans.get("feature_list", []),
                "apply_to": ["item"],
            }
        )
    for head in model["tasks"].values():
        # Explicitly approved removal of the old nullable mask policy.
        if head.get("mask") not in (None, "True", "not is_null(watch_seconds)"):
            raise ValueError("Business mask must be explicitly expressed in loss_weight")
        head.pop("mask", None)
        head["from"] = "features"
    model.update(optimizer_settings)
    parameters(model)
    dataset = deepcopy(source)
    dataset["feature_default_values"]["watch_duration_ms"] = 0
    execution = source["execution"]
    return {
        "dataset_config": dataset,
        "model": model,
        "dtypes": raw_dtypes,
        "histories": source["hist_options"],
        "item_filter": source["item_filter"],
        "batch_size": source["batch_size"],
        "max_epochs": source["max_epochs"],
        "seed": source["seed"],
        "intra_op_threads": execution["intra_op_threads"],
        "inter_op_threads": execution["inter_op_threads"],
        "shuffle_buffer_size": shuffle_buffer_size,
        "selection": source.get("selection"),
    }
