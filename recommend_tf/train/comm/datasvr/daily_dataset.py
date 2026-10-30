"""Resolve a versioned upstream completion envelope into frozen daily input."""

import copy
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .daily_contracts import digest, merge_contracts
from .published_dataset import PublishedDataset


def window(config, cutoff):
    options = config["online_config"]
    if set(options) != {"training_date", "validation_date"} or any(
        type(value) is not int or value <= 0 for value in options.values()
    ):
        raise ValueError("online_config requires positive integer day counts")
    end = date.fromisoformat(cutoff)
    validation = [end - timedelta(days=i) for i in reversed(range(options["validation_date"]))]
    training = [
        validation[0] - timedelta(days=i) for i in reversed(range(1, options["training_date"] + 1))
    ]
    return {
        "train": [d.isoformat() for d in training],
        "validation": [d.isoformat() for d in validation],
    }


def resolve_config(config, cutoff):
    result = copy.deepcopy(config)
    split = result["dataset_config"]["split"]
    for key in list(split):
        if key.startswith(("train_", "validation_", "test_")):
            del split[key]
    for name, days in window(config, cutoff).items():
        split.update({f"{name}_dates": days, f"{name}_from": days[0], f"{name}_to": days[-1]})
    return result


def freeze(config, ready, target):
    """Consume exact immutable URIs, never discover latest or infer readiness from files.

    The envelope must be emitted by the upstream success adapter, not by S3 object events.
    Its own URI/digest is pinned by the platform caller. UTC is the current reader contract.
    """
    if ready.get("version") != "doku-tf-dataset-ready/v1" or ready.get("timezone") != "UTC":
        raise ValueError("unsupported ready envelope or day timezone")
    resolved = resolve_config(config, ready["cutoff"])
    days = sum(window(config, ready["cutoff"]).values(), [])
    sources = ready["sources"]
    if [source["date"] for source in sources] != days:
        raise ValueError("ready envelope must cover the exact contiguous window in order")
    manifests, bundles, evidence = [], [], []
    for source in sources:
        if source.get("status") != "published" or not source.get("maturity_evidence"):
            raise ValueError("missing upstream publication/maturity evidence")
        options = copy.deepcopy(resolved["dataset_config"])
        options["dataset"] = {"manifest_uri": source["manifest_uri"]}
        dataset = PublishedDataset(options)
        if hashlib.sha256(dataset.manifest_bytes).hexdigest() != source["manifest_sha256"]:
            raise ValueError("source manifest file digest mismatch")
        manifest = dataset.manifest
        if "source_manifests" in manifest:
            raise ValueError("ready sources must be single-day manifests")
        if dataset.contract_digests != source["contract_sha256"]:
            raise ValueError("ready source contracts differ")
        start = datetime.fromisoformat(source["date"]).replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        if not manifest["partitions"]:
            raise ValueError("empty day publication")
        for part in manifest["partitions"]:
            lo, hi = [
                datetime.fromisoformat(part[f"event_time_{x}"].replace("Z", "+00:00"))
                for x in ("from", "to")
            ]
            if not start <= lo < hi <= end:
                raise ValueError("partition outside published UTC day")
        if manifests and any(
            manifest[k] != manifests[0][k] for k in ("dataset_id", "label_config_version")
        ):
            raise ValueError("incompatible daily dataset identity")
        # Existing additive daily-contract verifier requires these versioned fields.
        for key in ("label_config_version", "projection_sha256"):
            if not manifest.get(key):
                raise ValueError("daily manifest missing " + key)
        manifests.append(manifest)
        bundles.append({k: json.loads(v) for k, v in dataset.contract_bytes.items()})
        evidence.append(
            {
                "date": source["date"],
                "uri": source["manifest_uri"],
                "manifest": dataset.manifest_bytes.decode(),
                "sha256": source["manifest_sha256"],
                "contracts": {k: v.decode() for k, v in dataset.contract_bytes.items()},
                "contract_sha256": dataset.contract_digests,
            }
        )
    contracts = merge_contracts(bundles)
    contract_bytes = {k: json.dumps(v, sort_keys=True).encode() for k, v in contracts.items()}
    merged = copy.deepcopy(manifests[-1])
    merged.update(
        dataset_version="daily-" + digest(evidence),
        feature_schema_version=contracts["features"]["feature_spec_version"],
        label_definition_version=",".join(sorted(contracts["labels"]["label_definition_versions"])),
        projection_sha256=digest(contracts["features"]),
        partitions=[p for m in manifests for p in m["partitions"]],
        daily_contracts_version="doku-daily-contracts/v1",
        source_manifests=evidence,
        contract_sha256={k: hashlib.sha256(v).hexdigest() for k, v in contract_bytes.items()},
        as_of_ms=max(m["as_of_ms"] for m in manifests),
    )
    merged["row_count"] = sum(p["rows"] for p in merged["partitions"])
    merged["content_sha256"] = digest({k: v for k, v in merged.items() if k != "content_sha256"})
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=False)
    (target / "contracts").mkdir()
    for key, body in contract_bytes.items():
        (target / "contracts" / f"{key}.json").write_bytes(body)
    (target / "manifest.json").write_text(json.dumps(merged, sort_keys=True))
    (target / "ready.json").write_text(json.dumps(ready, sort_keys=True))
    resolved["dataset_config"]["dataset"] = {"manifest_uri": (target / "manifest.json").as_uri()}
    dataset = PublishedDataset(resolved["dataset_config"])
    for split in ("train", "validation"):
        dataset.partitions(split)
    dataset.freeze(target / "input")
    (target / "config.json").write_text(json.dumps(resolved, sort_keys=True))
    return resolved
