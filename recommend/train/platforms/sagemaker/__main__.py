from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3

from recommend.train.comm.model_params_validator import load_config
from recommend.train.platforms.sagemaker.config import load_runtime
from recommend.train.platforms.sagemaker.jobs import fetch_result, view_events
from recommend.train.platforms.sagemaker.submit import prepare, submit


def main() -> None:
    parser = argparse.ArgumentParser(description="Doku single-instance CPU training on SageMaker")
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument(
        "--profile", required=True, help="explicit AWS profile, no production default"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("submit")
    sub.add_argument("--config", type=Path)
    sub.add_argument("--run-id")
    sub.add_argument("--prepared", type=Path, required=True)
    sub.add_argument("--dry-run", action="store_true")
    for name in ["status", "stop"]:
        commands.add_parser(name).add_argument("--job-name", required=True)
    commands.add_parser("compare").add_argument("--job-names", nargs="+", required=True)
    tb = commands.add_parser("tensorboard")
    tb.add_argument("--run-ids", nargs="+", required=True)
    tb.add_argument("--directory", type=Path, required=True)
    tb.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    runtime, binding = load_runtime(args.runtime_config)
    session = boto3.Session(profile_name=args.profile, region_name=binding.region)
    actual = session.client("sts").get_caller_identity()["Account"]
    if actual != binding.account_id:
        raise ValueError("AWS caller account differs from binding")
    s3, sm = session.client("s3"), session.client("sagemaker")
    if args.command == "submit":
        if args.config is not None:
            if not args.run_id:
                parser.error("--config requires --run-id")
            original_yaml = args.config.read_text()
            request = prepare(
                load_config(args.config),
                runtime,
                binding,
                args.run_id,
                s3,
                args.prepared,
                original_yaml=original_yaml,
            )
        else:
            package = json.loads((args.prepared / "package.json").read_text())
            if (
                package["runtime"] != runtime.model_dump()
                or package["binding"] != binding.model_dump()
            ):
                raise ValueError("prepared submission runtime/binding mismatch")
            request = json.loads((args.prepared / "request.json").read_text())
        result = request if args.dry_run else submit(args.prepared, s3, sm)
        print(json.dumps(result, indent=2, default=str))
        print(
            f"https://{binding.region}.console.aws.amazon.com/sagemaker/home?region={binding.region}#/jobs/{request['TrainingJobName']}"
        )
    elif args.command == "status":
        result = sm.describe_training_job(TrainingJobName=args.job_name)
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in [
                        "TrainingJobName",
                        "TrainingJobStatus",
                        "SecondaryStatus",
                        "FailureReason",
                        "TrainingTimeInSeconds",
                        "BillableTimeInSeconds",
                        "ModelArtifacts",
                    ]
                },
                indent=2,
                default=str,
            )
        )
    elif args.command == "stop":
        sm.stop_training_job(TrainingJobName=args.job_name)
        print("Stop requested; inspect status to confirm termination.")
    elif args.command == "compare":
        print(json.dumps([fetch_result(sm, s3, n, binding) for n in args.job_names], indent=2))
    elif args.command == "tensorboard":
        view_events(
            s3, binding, args.run_ids, args.directory, runtime.sync_interval_seconds, args.port
        )


if __name__ == "__main__":
    main()
