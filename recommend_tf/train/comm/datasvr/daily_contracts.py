"""Validate source contracts and merge additive daily schema changes."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()
    ).hexdigest()


def label_versions(value: Any) -> set[str]:
    if not isinstance(value, str) or not value or any(not x.strip() for x in value.split(",")):
        raise ValueError("invalid label version set")
    return set(value.split(","))


def merged_dataset_identity(manifests: list[dict[str, Any]]) -> str:
    """Normalize only the known upstream versioned family; retain source evidence."""
    identities = []
    families = []
    for manifest in manifests:
        identity = manifest["dataset_id"]
        match = re.fullmatch(r"fyp-rank-train/fyp-rank-features-v([1-9][0-9]*)", identity)
        if match:
            if manifest["feature_schema_version"] != "fyp-rank-features/v" + match[1]:
                raise ValueError("daily dataset identity/schema mismatch")
            family = "fyp-rank-train"
        else:
            # Unknown identities must never alias the canonical merged family.
            family = "exact:" + identity
        identities.append(identity)
        families.append(family)
    if (
        not manifests
        or len(set(families)) != 1
        or len({m["label_config_version"] for m in manifests}) != 1
    ):
        raise ValueError("incompatible daily dataset identity or label config")
    return identities[0] if len(set(identities)) == 1 else families[0]


def validate_contracts(manifest: dict[str, Any], contracts: dict[str, Any]) -> None:
    features, labels = contracts["features"], contracts["labels"]
    declared = labels.get("label_definition_versions")
    if (
        not isinstance(declared, list)
        or not declared
        or any(not isinstance(v, str) or not v or "," in v for v in declared)
    ):
        raise ValueError("invalid label contract versions")
    if features["feature_spec_version"] != manifest["feature_schema_version"] or not label_versions(
        manifest["label_definition_version"]
    ) <= set(declared):
        raise ValueError("source contract versions differ")
    if "projection_sha256" in manifest and digest(features) != manifest["projection_sha256"]:
        raise ValueError("projection contract digest mismatch")
    names = [f["name"] for f in features["features"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature contract name")


def merge_contracts(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    result = copy.deepcopy(bundles[0])
    definitions: dict[str, Any] = {}
    versions: set[str] = set()
    feature_digests = set()
    for bundle in bundles:
        for name, excluded in [
            ("features", {"features", "feature_spec_version"}),
            ("labels", {"label_definition_versions"}),
        ]:
            if {k: v for k, v in bundle[name].items() if k not in excluded} != {
                k: v for k, v in result[name].items() if k not in excluded
            }:
                raise ValueError(f"incompatible daily {name} contract metadata")
        for feature in bundle["features"]["features"]:
            name = feature["name"]
            if name in definitions and definitions[name] != feature:
                raise ValueError(f"incompatible daily feature definition: {name}")
            definitions[name] = feature
        versions.update(bundle["labels"]["label_definition_versions"])
        feature_digests.add(digest(bundle["features"]))
    if len(feature_digests) > 1:
        result["features"]["feature_spec_version"] = "merged-" + digest(sorted(feature_digests))
        result["features"]["features"] = [definitions[n] for n in sorted(definitions)]
    result["labels"]["label_definition_versions"] = sorted(versions)
    return result


def verify_daily_evidence(
    manifest: dict[str, Any], contracts: dict[str, Any]
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    """Recheck frozen source evidence; never follow a mutable pointer on resume."""
    declared_by_uri: dict[str, set[str]] = {}
    parts = []
    bundles = []
    records = []
    originals = []
    versions: set[str] = set()
    for source in manifest["source_manifests"]:
        raw = source["manifest"].encode()
        original = json.loads(raw)
        if (
            hashlib.sha256(raw).hexdigest() != source["sha256"]
            or digest({k: v for k, v in original.items() if k != "content_sha256"})
            != original["content_sha256"]
        ):
            raise ValueError("daily source manifest checksum mismatch")
        bundle = {name: json.loads(body) for name, body in source["contracts"].items()}
        hashes = {
            name: hashlib.sha256(body.encode()).hexdigest()
            for name, body in source["contracts"].items()
        }
        if hashes != source["contract_sha256"] or any(
            hashes.get(k) != v for k, v in original.get("contract_sha256", {}).items()
        ):
            raise ValueError("daily source contract checksum mismatch")
        validate_contracts(original, bundle)
        originals.append(original)
        if original["label_config_version"] != manifest["label_config_version"]:
            raise ValueError("daily source identity mismatch")
        fields = {f["name"] for f in bundle["features"]["features"]}
        for part in original["partitions"]:
            if part["uri"] in declared_by_uri or part["event_time_from"][:10] != source["date"]:
                raise ValueError("daily source partition mismatch")
            declared_by_uri[part["uri"]] = fields
            parts.append(part)
        bundles.append(bundle)
        versions.update(label_versions(original["label_definition_version"]))
        records.append(
            {
                "dataset_id": original["dataset_id"],
                "date": source["date"],
                "uri": source["uri"],
                "sha256": source["sha256"],
                "feature_schema_version": original["feature_schema_version"],
                "label_definition_version": original["label_definition_version"],
                "projection_sha256": original["projection_sha256"],
                "contract_sha256": hashes,
            }
        )
    if merged_dataset_identity(originals) != manifest["dataset_id"]:
        raise ValueError("daily source identity mismatch")
    if not bundles or parts != manifest["partitions"] or merge_contracts(bundles) != contracts:
        raise ValueError("daily merged contracts or partitions differ from evidence")
    if versions != label_versions(manifest["label_definition_version"]):
        raise ValueError("daily merged label versions differ from evidence")
    return declared_by_uri, records
