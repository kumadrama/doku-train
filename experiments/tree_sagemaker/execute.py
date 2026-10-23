"""Bounded, resumable experiment submission; stages and inputs come from frozen configs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import boto3
import yaml
from botocore.exceptions import (
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
    SSLError,
)

from recommend.train.platforms.sagemaker.config import load_runtime
from recommend.train.platforms.sagemaker.jobs import fetch_result
from recommend.train.platforms.sagemaker.submit import prepare, submit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--baseline-input", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1:
        raise ValueError("experiment concurrency must be positive")
    runtime, binding = load_runtime(args.runtime)
    if runtime.stages != ["train"]:
        raise ValueError("experiments must not evaluate test")
    session = boto3.Session(profile_name=args.profile, region_name=binding.region)
    if session.client("sts").get_caller_identity()["Account"] != binding.account_id:
        raise ValueError("wrong AWS account")
    s3, sm = session.client("s3"), session.client("sagemaker")
    expected = json.loads((args.baseline_input / "manifest.json").read_text())["content_sha256"]
    inventory = json.loads(args.inventory.read_text())
    run_e04 = json.loads(args.audit.read_text())["run_e04"]
    selected = [x for x in inventory if not x["conditional"] or run_e04]
    if args.baseline_only:
        selected = [x for x in selected if x["name"] == "e00"]
    args.work.mkdir(parents=True, exist_ok=True)
    state_path = args.work / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if not args.baseline_only and state.get("e00", {}).get("status") != "Completed":
        raise ValueError("verified baseline must complete before batch submission")
    if any(x["status"] in ["Failed", "Stopped"] for x in state.values()):
        raise ValueError("failed/stopped jobs require explicit investigation before resuming")
    pending = [x for x in selected if x["name"] not in state]
    while pending or any(
        x["status"] in ["Submitted", "InProgress", "Stopping"] for x in state.values()
    ):
        active = sum(x["status"] in ["Submitted", "InProgress", "Stopping"] for x in state.values())
        while pending and active < args.parallel:
            item = pending.pop(0)
            dest = args.work / "prepared" / item["name"]
            if not dest.exists():
                original = Path(item["config"]).read_text()
                prepare(
                    yaml.safe_load(original),
                    runtime,
                    binding,
                    item["run_id"],
                    s3,
                    dest,
                    original_yaml=original,
                )
            package = json.loads((dest / "package.json").read_text())
            if json.loads(package["input"]["manifest.json"])["content_sha256"] != expected:
                raise ValueError("experiment manifest changed")
            if package["runtime"] != runtime.model_dump():
                raise ValueError("prepared runtime changed")
            submit(dest, s3, sm)
            state[item["name"]] = {"job": f"doku-exp-{item['run_id']}", "status": "Submitted"}
            state_path.write_text(json.dumps(state, indent=2))
            print("SUBMITTED", item["name"], flush=True)
            active += 1
        for name, record in state.items():
            if record["status"] not in ["Submitted", "InProgress", "Stopping"]:
                continue
            job = sm.describe_training_job(TrainingJobName=record["job"])
            status = job["TrainingJobStatus"]
            if status == "Completed":
                result = fetch_result(sm, s3, record["job"], binding)
                result_path = args.work / f"{name}-result.json"
                result_path.write_text(json.dumps(result, indent=2))
                record["artifact_uri"] = job["ModelArtifacts"]["S3ModelArtifacts"]
            elif status in ["Failed", "Stopped"]:
                record["failure"] = job.get("FailureReason", status)
                record["status"] = status
                state_path.write_text(json.dumps(state, indent=2))
                raise RuntimeError(f"{name}: {record['failure']}; no automatic retry")
            if status != record["status"]:
                print(name, status, flush=True)
            record["status"] = status
        state_path.write_text(json.dumps(state, indent=2))
        if pending or any(
            x["status"] in ["Submitted", "InProgress", "Stopping"] for x in state.values()
        ):
            time.sleep(30)
    if any(state.get(x["name"], {}).get("status") != "Completed" for x in selected):
        raise RuntimeError("not all selected experiments completed")
    print("VERIFIED COMPLETE", len(selected), flush=True)


def run_with_network_recovery(action=main, sleep=time.sleep):
    """Resume immutable prepared jobs after bounded transport failures only."""
    for attempt in range(4):
        try:
            return action()
        except (SSLError, EndpointConnectionError, ConnectionClosedError, ReadTimeoutError) as exc:
            if attempt == 3:
                raise
            delay = 30 * (attempt + 1)
            print(
                f"NETWORK_RECOVERY {type(exc).__name__} attempt={attempt + 1} delay={delay}s",
                flush=True,
            )
            sleep(delay)


if __name__ == "__main__":
    run_with_network_recovery()
