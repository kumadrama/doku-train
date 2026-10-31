"""Versioned raw-feature wire protocol; no train imports or feature-state fitting."""

import math


def scalar(value, dtype):
    if dtype in {"str", "string"}:
        if not isinstance(value, str):
            raise ValueError("string feature required")
    elif dtype in {"int32", "int64"}:
        bits = 32 if dtype == "int32" else 64
        if type(value) is not int or not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
            raise ValueError("integer feature out of range")
    elif dtype in {"float32", "float64"}:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("finite numeric feature required")
        if dtype == "float32" and abs(value) > 3.4028235e38:
            raise ValueError("float32 overflow")
    else:
        raise ValueError("unsupported input dtype")
    return value


def prepare(payload, artifact, *, max_batch, max_history):
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "instances"}:
        raise ValueError("request requires schema_version and instances")
    if payload["schema_version"] != "doku-tf-request/v1":
        raise ValueError("unsupported request version")
    rows = payload["instances"]
    if not isinstance(rows, list) or not 0 < len(rows) <= max_batch:
        raise ValueError("batch outside configured limit")
    cfg, signature = artifact["config"], artifact["signature"]
    groups = cfg["input_features"]
    # Only groups reachable from the serving signature are evaluated.
    active = {v["source_group"] for v in signature["inputs"].values()} - {"item"}
    histories = {
        n: g for n, g in groups.items() if g.get("history_pooling") and g["from"] in active
    }
    required = set(cfg["required_raw_fields"])
    for group in histories.values():
        required.update(group["features"])
        if "weight_feature" in group:
            required.add(group["weight_feature"])
    sequence_fields = {
        f
        for g in histories.values()
        for f in g["features"] + ([g["weight_feature"]] if "weight_feature" in g else [])
    }
    organized = []
    for original in rows:
        if not isinstance(original, dict) or set(original) - required:
            raise ValueError("unknown input fields")
        row = dict(original)
        for group in histories.values():
            if "weight_feature" in group:
                ids, weights = row.get(group["features"][0]), row.get(group["weight_feature"])
                if (ids is None) != (weights is None):
                    raise ValueError("history IDs and weights must be supplied together")
        for name in required:
            value = row.get(name)
            if value is None:
                value = cfg["defaults"].get(name)
            dtype = cfg["dtypes"].get(name)
            if name in sequence_fields:
                if not isinstance(value, list) or len(value) > max_history:
                    raise ValueError("invalid history size or type")
                row[name] = [scalar(v, dtype) for v in value]
            else:
                row[name] = scalar(value, dtype)
        lengths = {}
        for name, group in histories.items():
            fields = group["features"]
            weight = group.get("weight_feature")
            all_fields = fields + ([weight] if weight else [])
            if len({len(row[f]) for f in all_fields}) != 1:
                raise ValueError("unaligned history")
            policy = cfg["histories"].get(name)
            if policy:
                cap = policy["max_hist_size"]
                if type(cap) is not int or cap <= 0:
                    raise ValueError("invalid history cap")
                if policy["truncate"] == "tail" and policy["duplicates"] == "preserve":
                    for field in all_fields:
                        row[field] = row[field][-cap:]
                elif (policy["truncate"], policy["duplicates"], policy.get("tie_break")) == (
                    "top_weight",
                    "sum_weights",
                    "raw_id_ascending",
                ):
                    if len(fields) != 1 or not weight:
                        raise ValueError("invalid weighted history")
                    merged = {}
                    for key, value in zip(row[fields[0]], row[weight]):
                        if value < 0:
                            raise ValueError("negative history weight")
                        merged[key] = merged.get(key, 0.0) + value
                    pairs = sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))[:cap]
                    row[fields[0]] = [key for key, _ in pairs]
                    row[weight] = [scalar(value, cfg["dtypes"][weight]) for _, value in pairs]
                else:
                    raise ValueError("unsupported history policy")
            size = len(row[fields[0]])
            source = group["from"]
            if source in lengths and lengths[source] != size:
                raise ValueError("ambiguous shared history length")
            lengths[source] = size
        organized.append((row, lengths))
    tensors = {}
    for key, spec in signature["inputs"].items():
        field, group = spec["source_field"], spec["source_group"]
        if group == "item":
            values = [row[field] for row, _ in organized]
        elif field == "length":
            values = [lengths[group] for _, lengths in organized]
        else:
            width = max(1, max(lengths[group] for _, lengths in organized))
            pad = "" if spec["dtype"] == "string" else 0
            values = [row[field] + [pad] * (width - len(row[field])) for row, _ in organized]
        rank = 2 if group != "item" and field != "length" else 1
        if len(spec["shape"]) != rank:
            raise ValueError("unsupported signature input rank")
        tensors[key] = values
    return {"signature_name": "serving_default", "inputs": tensors}, len(rows)
