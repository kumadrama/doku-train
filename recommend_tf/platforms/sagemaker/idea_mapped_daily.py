"""Reconcile the next contiguous Idea publication and its independent endpoint."""

import argparse
import copy
import hashlib
import json
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

import boto3

from recommend_tf.inference import official
from recommend_tf.train import export_model
from recommend_tf.train.comm.artifact_integrity import checksum

from .daily_pipeline import acquire_executor, fetch_run
from .idea_mapped_input import dates, freeze
from .publish_model import (
    ReleaseState,
    contains,
    deploy_step,
    describe,
    immutable_put,
    location,
    publish_package,
    rollback_step,
    verify_probe,
)


def recipe_request(s3, cfg, day, root):
    days = dates(cfg["first_day"], day)
    merged, contracts, manifest_uri = freeze(s3, cfg, days, root)
    recipe = copy.deepcopy(cfg["recipe"])
    recipe["online_config"] = {"training_date": len(days) - 1, "validation_date": 1}
    recipe["model"]["broadcast_feat"] = "item/idea_drama_id"
    recipe["dataset_config"]["dataset"] = {"manifest_uri": manifest_uri}
    recipe["dataset_config"]["split"].update(
        train_from=days[0], train_to=days[-2], validation_from=day, validation_to=day
    )
    body = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(body).hexdigest()
    name = f"doku-exp-idea-mapped-daily-{day.replace('-', '')}-{digest[:8]}"
    prefix = cfg["input_uri"] + "/jobs/" + name + "/recipe/"
    manifest = root / "manifest.json"
    record = dict(
        manifest_uri=manifest_uri,
        manifest_file_sha256=checksum(manifest),
        manifest_sha256=merged["content_sha256"],
        contract_digests=merged["contract_sha256"],
    )
    for key, data in {
        "config.json": body,
        "input/record.json": json.dumps(record).encode(),
        "input/manifest.json": manifest.read_bytes(),
        **{"input/contracts/" + k + ".json": v for k, v in contracts.items()},
    }.items():
        path = root / "upload"
        path.write_bytes(data)
        immutable_put(s3, prefix + key, path, cfg["kms_key"])
    req = copy.deepcopy(cfg["template"])
    req["TrainingJobName"] = name
    run = cfg["runs_uri"] + "/" + name
    req["Environment"].update(
        DOKU_CUTOFF=day, DOKU_CONFIG_SHA256=digest, DOKU_ARTIFACT_URI=run + "/live/"
    )
    req["OutputDataConfig"]["S3OutputPath"] = run + "/artifacts/"
    req["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] = prefix
    return req


def tick(cfg, s3, sm, ecs, rt, state, executor):
    active = state.value["active"]
    if active is None:
        last = state.value["high_water"] or cfg["bootstrap_cutoff"]
        if state.value["history"] and state.value["history"][-1]["stage"] != "deployed":
            raise ValueError("previous release failed; explicit recovery required")
        day = (date.fromisoformat(last) + timedelta(days=1)).isoformat()
        bucket, key = location(cfg["source_uri"] + "/date=" + day + "/_SUCCESS")
        published = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        if not any(obj["Key"] == key for obj in published.get("Contents", [])):
            return "caught_up"
        active = state.claim("idea-mapped-" + day, day)
    acquire_executor(state, ecs, cfg["cluster"], executor)
    day = active["cutoff"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        if active["stage"] == "claimed":
            req = (
                cfg["first_request"]
                if day == cfg["first_request"]["Environment"]["DOKU_CUTOFF"]
                else recipe_request(s3, cfg, day, root)
            )
            active.update(stage="training", request=req)
            state.save()
        if active["stage"] == "training":
            req = active["request"]
            job = describe(sm, "describe_training_job", TrainingJobName=req["TrainingJobName"])
            if job is None:
                sm.create_training_job(**req)
                return "pending"
            if any(not contains(job.get(k), v) for k, v in req.items() if k != "Tags"):
                raise ValueError("existing training job differs")
            if job["TrainingJobStatus"] in ("Failed", "Stopped"):
                active["stage"] = "training_failed"
                state.finish(active)
                return "failed"
            if job["TrainingJobStatus"] != "Completed":
                return "pending"
            active["stage"] = "trained"
            state.save()
        if active["stage"] == "trained":
            run = root / "run"
            fetch_run(
                s3,
                active["request"]["Environment"]["DOKU_ARTIFACT_URI"],
                run,
                cfg["download_limit_bytes"],
            )
            status = json.loads((run / "daily-status.json").read_bytes())
            if status["cutoff"] != day or status["status"] != "completed":
                raise ValueError("daily run not selected")
            selection = json.loads((run / "training/selection.json").read_bytes())
            weights = selection.get("weight", {})
            if (
                selection.get("metric") != "auc"
                or set(weights) != set(cfg["outputs"])
                or any(abs(v - 1 / 3) > 1e-10 for v in weights.values())
            ):
                raise ValueError("expected equal three-task validation AUC")
            out = root / "export"
            result = export_model.execute(
                run, out, outputs=cfg["outputs"], broadcast_feat="item/idea_drama_id"
            )
            src = publish_package(
                s3, out, cfg["models_uri"], cfg["kms_key"], inference_protocol=official.PROTOCOL
            )
            active.update(
                stage="uploaded",
                model_uri=src,
                model_version=result["model_version"],
                selected_epoch=selection["epoch"],
                score=selection["score"],
            )
            state.save()
        if active["stage"] in ("uploaded", "deploying"):
            result = deploy_step(
                sm,
                rt,
                state,
                cfg["deployment"],
                probe=lambda client, endpoint, version: verify_probe(
                    client, endpoint, version, cfg["probe"], inference_protocol=official.PROTOCOL
                ),
            )
        elif active["stage"] == "rollback_pending":
            result = rollback_step(sm, state, cfg["deployment"])
        else:
            raise ValueError("unknown stage")
        if result != "pending":
            state.finish(active)
        return result


def main():
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.binding).read_bytes())
    with urlopen(os.environ["ECS_CONTAINER_METADATA_URI_V4"] + "/task", timeout=5) as response:
        executor = json.load(response)["TaskARN"]
    session = boto3.Session(region_name=cfg["region"])
    s3, sm, ecs, rt = [session.client(k) for k in ("s3", "sagemaker", "ecs", "sagemaker-runtime")]
    while True:
        state = ReleaseState(s3, cfg["state_uri"], cfg["kms_key"])
        result = tick(cfg, s3, sm, ecs, rt, state, executor)
        if result == "caught_up":
            return
        if result not in ("pending", "deployed"):
            raise RuntimeError(result)
        time.sleep(30 if result == "pending" else 1)


if __name__ == "__main__":
    main()
