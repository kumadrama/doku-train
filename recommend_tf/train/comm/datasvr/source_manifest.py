"""Resolve an explicit date template once; never discover latest or list storage."""

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit


def resolve_source(source, config, splits):
    uri = source.get("manifest_uri", "")
    if "{date}" not in uri:
        return source.get("manifest_uris", [uri]), []
    options = config["split"]
    days = set()
    train_days = set()
    for split in splits:
        start, end = (date.fromisoformat(options[f"{split}_{x}"]) for x in ("from", "to"))
        selected = options.get(
            f"{split}_dates",
            [str(start + timedelta(days=i)) for i in range((end - start).days + 1)],
        )
        days.update(selected)
        if split == "train":
            train_days.update(selected)
    limit = config["execution"]["memory_limit_bytes"]
    client = None

    def read(address):
        nonlocal client
        parsed = urlsplit(address)
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("Invalid manifest address")
        if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
            with Path(unquote(parsed.path)).open("rb") as stream:
                body = stream.read(limit + 1)
        elif parsed.scheme == "s3":
            if client is None:
                import boto3

                client = boto3.Session().client("s3")
            with client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))[
                "Body"
            ] as stream:
                body = stream.read(limit + 1)
        else:
            raise ValueError("Unsupported manifest address")
        if len(body) > limit:
            raise ValueError("Manifest exceeds memory budget")
        return body

    from botocore.exceptions import BotoCoreError, ClientError

    resolved, evidence = [], []
    for day in sorted(days):
        address = uri.replace("{date}", day)
        try:
            raw = read(address)
            value = json.loads(raw)
            target = address
            if "manifest_uri" in value:
                target = value["manifest_uri"]
                base, dest = urlsplit(address), urlsplit(target)
                prefix = unquote(base.path).rsplit("/", 1)[0] + "/run_id="
                target_path = unquote(dest.path)
                if (
                    (base.scheme, base.netloc) != (dest.scheme, dest.netloc)
                    or not target_path.startswith(prefix)
                    or not target_path.endswith("/manifest.json")
                    or not target_path[len(prefix) : -len("/manifest.json")].isalnum()
                ):
                    raise ValueError("Daily pointer must stay within its immutable date run")
                if "date" in value and value["date"] != day:
                    raise ValueError("Daily pointer date mismatch")
                manifest_raw = read(target)
                if hashlib.sha256(manifest_raw).hexdigest() != value.get("manifest_sha256"):
                    raise ValueError("Daily manifest checksum mismatch")
            else:
                manifest_raw = raw
            manifest = json.loads(manifest_raw)
            for part in manifest["partitions"]:
                start = datetime.combine(date.fromisoformat(day), datetime.min.time(), timezone.utc)
                lower, upper = (
                    datetime.fromisoformat(part[f"event_time_{edge}"].replace("Z", "+00:00"))
                    for edge in ("from", "to")
                )
                if not start <= lower < upper <= start + timedelta(days=1):
                    raise ValueError("Daily manifest date mismatch")
            if read(address) != raw:
                raise ValueError("Daily input changed while resolving")
        except (OSError, ValueError, KeyError, BotoCoreError, ClientError) as exc:
            if day in train_days:
                raise
            evidence.append(
                {"date": day, "uri": address, "status": "skipped", "reason": type(exc).__name__}
            )
            continue
        resolved.append(target)
        evidence.append(
            {
                "date": day,
                "uri": address,
                "raw": raw.decode(),
                "manifest_uri": target,
                "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "status": "ready",
            }
        )
    for entry in evidence:
        if entry["status"] == "ready" and read(entry["uri"]) != entry["raw"].encode():
            raise ValueError("Daily input changed while resolving")
    return resolved, evidence
