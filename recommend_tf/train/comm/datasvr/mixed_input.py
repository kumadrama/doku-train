"""Atomic preparation of independently frozen, versioned input sources."""

import copy
import json
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from .prepared import digest, prepare
from .published_dataset import PublishedDataset
from .source_manifest import resolve_source


class SourceUnavailable(Exception):
    """An input source failed validation or could not be read."""


def validate_sources(sources):
    if not isinstance(sources, list) or not sources:
        raise ValueError("data_sources must be a nonempty list")
    ids, names = set(), set()
    for source in sources:
        if set(source) not in (
            {"name", "source_id", "format", "manifest_uri"},
            {"name", "source_id", "format", "manifest_uris"},
        ):
            raise ValueError("Each source requires name/source_id/format/manifest_uri")
        if not isinstance(source["name"], str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", source["name"]
        ):
            raise ValueError("Invalid source name")
        sid = source["source_id"]
        if type(sid) is not int or sid < 0 or sid in ids or source["name"] in names:
            raise ValueError("Source IDs and names must be unique")
        if source["format"] != "doku-train-dataset-manifest/v1":
            raise ValueError("Unsupported source format")
        uris = source.get("manifest_uris", [source.get("manifest_uri")])
        if (
            not isinstance(uris, list)
            or not uris
            or any(not isinstance(u, str) or not u for u in uris)
            or len(set(uris)) != len(uris)
        ):
            raise ValueError("Source manifest URI is required")
        ids.add(sid)
        names.add(source["name"])


def prepare_sources(
    config, root, params, splits, source_revision, local_root=None, frozen_input=None
):
    sources = config["data_sources"]
    validate_sources(sources)
    root = Path(root)
    frozen = Path(frozen_input) if frozen_input else None
    previous = json.loads((frozen / "sources.json").read_text()) if frozen else None
    if previous and previous["configuration"] != sources:
        raise ValueError("Frozen source configuration differs")
    identity = digest({"config": config, "revision": source_revision})
    records, manifests = [], {split: [] for split in splits}
    columns = set(config["dtypes"])
    defaults = config["dataset_config"]["feature_default_values"]
    known_dtypes = {}
    for source in sources:
        sid = source["source_id"]
        target = root / "prepared" / "sources" / str(sid)
        record = dict(source, status="skipped")
        old = (
            next((r for r in previous["sources"] if r["source_id"] == sid), None)
            if previous
            else None
        )
        if old and old["status"] != "ready":
            records.append(old)
            continue
        cfg = copy.deepcopy(config["dataset_config"])
        cfg["features"] = []
        cfg["optional_audit_defaults"] = True
        datasets = []
        try:
            if old:
                uris = old["resolved_manifest_uris"]
                resolution = old.get("resolution", [])
            else:
                uris, resolution = resolve_source(source, cfg, splits)
            record["resolved_manifest_uris"] = uris
            record["resolution"] = resolution
            for index, uri in enumerate(uris):
                cfg["dataset"] = {"manifest_uri": uri}
                dataset = PublishedDataset(
                    cfg,
                    local_root=local_root,
                    frozen_input=frozen / str(sid) / str(index) if frozen else None,
                )
                expected = next(
                    (r["manifest_sha256"] for r in resolution if r.get("manifest_uri") == uri), None
                )
                if expected:
                    import hashlib

                    if hashlib.sha256(dataset.manifest_bytes).hexdigest() != expected:
                        raise ValueError("Resolved manifest changed before freezing")
                dataset.freeze(root / "input" / str(sid) / str(index))
                datasets.append(dataset)
        except (OSError, ValueError, KeyError, BotoCoreError, ClientError) as exc:
            record["reason"] = type(exc).__name__
            records.append(record)
            continue
        record["manifest_digests"] = [d.manifest["content_sha256"] for d in datasets]
        # Distinct manifests must not claim intersecting event intervals. Shards
        # within a single manifest may share a partition interval legitimately.
        intervals = []
        all_days = set()
        for index, dataset in enumerate(datasets):
            days = {p["event_time_from"][:10] for p in dataset.manifest["partitions"]}
            all_days.update(days)
            for part in dataset.manifest["partitions"]:
                lower = datetime.fromisoformat(part["event_time_from"].replace("Z", "+00:00"))
                upper = datetime.fromisoformat(part["event_time_to"].replace("Z", "+00:00"))
                if any(
                    other != index and lower < end and start < upper
                    for other, start, end in intervals
                ):
                    raise ValueError("Overlapping source manifests")
                intervals.append((index, lower, upper))
            opts = dataset.config["split"]
            for split in splits:
                lower, upper = (
                    date.fromisoformat(opts[f"{split}_{edge}"]) for edge in ("from", "to")
                )
                declared = set(
                    opts.get(
                        f"{split}_dates",
                        [
                            (lower + timedelta(days=i)).isoformat()
                            for i in range((upper - lower).days + 1)
                        ],
                    )
                )
                opts[f"{split}_dates"] = sorted(declared & days)
        if "train" in splits:
            opts = config["dataset_config"]["split"]
            lower, upper = (date.fromisoformat(opts[f"train_{edge}"]) for edge in ("from", "to"))
            declared = set(
                opts.get(
                    "train_dates",
                    [
                        (lower + timedelta(days=i)).isoformat()
                        for i in range((upper - lower).days + 1)
                    ],
                )
            )
            if not declared <= all_days:
                record["reason"] = "missing_train_dates"
                records.append(record)
                continue
        source_dtypes = {}
        for dataset in datasets:
            for name, dtype in dataset.dtypes.items():
                if name in columns and name in source_dtypes and source_dtypes[name] != dtype:
                    raise ValueError("Conflicting source dtype: " + name)
                if name in columns:
                    source_dtypes[name] = dtype
            available = (
                set(dataset.dtypes)
                | dataset.labels
                | {"impression_id", "impression_time", "request_time_ms"}
            )
            missing = columns - available - {"datasvr_source"}
            for name in missing:
                if name not in defaults:
                    raise ValueError("Missing explicit source default: " + name)
            dataset.config["features"] = sorted(
                set(config["dataset_config"]["features"]) & available
            )
        staged = {}
        try:
            for split in splits:

                def blocks():
                    for dataset in datasets:
                        available = (
                            set(dataset.dtypes)
                            | dataset.labels
                            | {"impression_id", "impression_time", "request_time_ms"}
                        )
                        missing = columns - available - {"datasvr_source"}
                        iterator = iter(dataset.iter_batches(split, columns & available))
                        while True:
                            try:
                                block = next(iterator)
                            except StopIteration:
                                break
                            except (
                                OSError,
                                ValueError,
                                KeyError,
                                BotoCoreError,
                                ClientError,
                            ) as exc:
                                raise SourceUnavailable(type(exc).__name__) from exc
                            count = len(next(iter(block.values())))
                            for name in missing:
                                block[name] = [copy.deepcopy(defaults[name]) for _ in range(count)]
                            block["datasvr_source"] = [sid] * count
                            yield block

                staged[split] = prepare(
                    blocks(),
                    target / split,
                    defaults,
                    params,
                    config["histories"],
                    config["item_filter"],
                    identity,
                    cfg["execution"]["cache_limit_bytes"],
                    allow_empty=True,
                    budget_root=root / "prepared",
                )
            if "train" in staged and not staged["train"]["rows"]:
                raise SourceUnavailable("empty_train")
        except SourceUnavailable as exc:
            shutil.rmtree(target, ignore_errors=True)
            record["reason"] = str(exc)
            records.append(record)
            continue
        for name, dtype in source_dtypes.items():
            if name in known_dtypes and known_dtypes[name] != dtype:
                raise ValueError("Conflicting source dtype: " + name)
        known_dtypes.update(source_dtypes)
        record["status"] = "ready"
        record["rows"] = {split: m["rows"] for split, m in staged.items()}
        records.append(record)
        for split, manifest in staged.items():
            manifests[split].append((sid, manifest))
    (root / "input").mkdir(exist_ok=True)
    evidence = {"configuration": sources, "sources": records}
    (root / "input" / "sources.json").write_text(json.dumps(evidence, sort_keys=True))
    identity = digest({"preparation": identity, "sources": evidence})
    for split, entries in manifests.items():
        directory = root / "prepared" / split
        directory.mkdir(parents=True, exist_ok=True)
        files, source_rows = [], {}
        for sid, manifest in entries:
            source_rows[str(sid)] = manifest["rows"]
            files.extend(
                dict(f, file=f"../sources/{sid}/{split}/{f['file']}", source_id=sid)
                for f in manifest["files"]
            )
        manifest = {
            "identity": identity,
            "rows": sum(source_rows.values()),
            "files": files,
            "source_rows": source_rows,
        }
        (directory / "complete.json").write_text(json.dumps(manifest, sort_keys=True))
    return identity, sum(r.get("rows", {}).get("train", 0) for r in records)
