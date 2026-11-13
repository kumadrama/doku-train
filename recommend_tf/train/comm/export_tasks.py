"""Resolve the public prediction heads independently of training and selection."""


def validate_export_tasks(config):
    if "export_tasks" not in config:
        return
    names = config["export_tasks"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or name not in config["model"]["tasks"] for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("export_tasks must be a nonempty list of distinct model.tasks names")


def export_outputs(config, units):
    validate_export_tasks(config)
    tasks = config["model"]["tasks"]
    names = config.get("export_tasks", list(units))
    if not names or any(name not in tasks for name in names):
        raise ValueError("export requires configured model tasks")
    outputs = {}
    for name in names:
        unit = units.get(name)
        if name not in units and tasks[name].get("predict_fn") == "sigmoid":
            unit = "probability"
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"missing export output unit for {name}")
        outputs[name] = unit
    return outputs


def publication_outputs(config, binding):
    units = binding.get("outputs", {})
    if "export_tasks" in config:
        return export_outputs(config, units)
    return units
