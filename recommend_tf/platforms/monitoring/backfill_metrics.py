"""Explicit sealed daily runs only; dry-run is the default. Never trains or deploys."""

import argparse
import json
import tempfile
from pathlib import Path

from recommend_tf.platforms.sagemaker.daily_pipeline import fetch_run

from .report_metrics import observation, put_event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument(
        "--runs",
        required=True,
        help="JSON list: run_uri, optional training_job_name and historical_identity",
    )
    parser.add_argument("--cutoff-from")
    parser.add_argument("--cutoff-to")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    from datetime import date

    if bool(args.cutoff_from) != bool(args.cutoff_to):
        parser.error("cutoff-from and cutoff-to must be supplied together")
    if args.cutoff_from and date.fromisoformat(args.cutoff_from) > date.fromisoformat(
        args.cutoff_to
    ):
        parser.error("cutoff range is reversed")
    binding = json.loads(Path(args.binding).read_text())
    import boto3

    session = boto3.Session(region_name=binding["region"])
    s3, sm = session.client("s3"), session.client("sagemaker")
    failures = []
    for entry in json.loads(Path(args.runs).read_text()):
        uri = entry["run_uri"]
        if not uri.startswith(binding["work_prefix"].rstrip("/") + "/"):
            raise ValueError("run outside authorized work prefix")
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "run"
                fetch_run(s3, uri, root, binding["download_limit_bytes"])
                job = (
                    sm.describe_training_job(TrainingJobName=entry["training_job_name"])
                    if entry.get("training_job_name")
                    else None
                )
                event = observation(
                    root,
                    binding,
                    run_uri=uri,
                    training_job=job,
                    historical_identity=entry.get("historical_identity"),
                )
                if (
                    args.cutoff_from
                    and not args.cutoff_from <= event["data_cutoff"] <= args.cutoff_to
                ):
                    print(json.dumps({"run_uri": uri, "status": "outside_requested_cutoff_range"}))
                    continue
                result = put_event(s3, event, binding["monitoring"]) if args.write else event
                print(json.dumps({"run_uri": uri, "result": result}, sort_keys=True))
        except Exception as error:
            failures.append(uri)
            print(json.dumps({"run_uri": uri, "error_type": type(error).__name__}))
    if failures:
        raise RuntimeError(f"{len(failures)} runs require attention")


if __name__ == "__main__":
    main()
