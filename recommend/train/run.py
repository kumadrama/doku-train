from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from recommend.train.comm.artifact_store import ObjectStore
from recommend.train.comm.daily_publication import archive_unpublished_run, prepare, read_uri
from recommend.train.comm.datasvr.published_dataset import digest
from recommend.train.comm.model_params_validator import load_config, validate_config
from recommend.train.comm.publication import Registry, encode
from recommend.train.comm.redis_distribution import Distribution
from recommend.train.comm.search import execute_search, expand_search
from recommend.train.comm.training_pipeline import source_identity
from recommend.train.runtime_policy import OperationPaused, check, skipped
from recommend.train.train_rerank import execute_test_local, execute_train, execute_train_local


def daily_run(
    config: dict[str, Any],
    *,
    daily_root: str,
    frozen_root: str,
    cutoff: str,
    artifact_root: str,
    export_root: Path,
    force: bool = False,
    distribution: Distribution | None = None,
    test_observer: Callable[[list[dict[str, Any]]], None] | None = None,
) -> dict[str, Any]:
    from recommend.train.eval_rerank import execute_test
    from recommend.train.export_model import execute_export, gate
    from recommend.train.publish_model import publish_prepared, sync_version
    from recommend.train.train_rerank import execute_train

    try:
        check(artifact_root, "training")
    except OperationPaused as error:
        return skipped(error)
    config = validate_config(config)
    if config["model_type"] != "rerank_tree" or "publication" not in config:
        raise ValueError("daily publication requires tree model and publication rules")
    if "daily" not in config:
        raise ValueError("daily.training_window_days must be declared in YAML")
    prepared = prepare(
        config, daily_root, frozen_root, cutoff, window_days=config["daily"]["training_window_days"]
    )
    if prepared is None:
        return {"status": "skipped", "reason": "insufficient_days"}
    config = prepared
    logical = {k: v for k, v in config.items() if k != "output"}
    logical["execution"] = {k: v for k, v in config["execution"].items() if k != "cache_dir"}
    fingerprint = digest(
        {
            "config": logical,
            "source": source_identity(),
            "manifest": hashlib.sha256(
                read_uri(
                    config["dataset"]["manifest_uri"], config["execution"]["memory_limit_bytes"]
                )
            ).hexdigest(),
        }
    )
    owner = uuid.uuid4().hex
    registry = Registry(
        ObjectStore(artifact_root, max_bytes=config["execution"]["memory_limit_bytes"]),
        config["model_name"],
    )
    if distribution is not None and (
        distribution.registry.name != registry.name
        or distribution.registry.store.root != registry.store.root
    ):
        raise ValueError("distribution binding mismatch")
    state = registry.read()[0]
    if state["format"] == "doku-publication/v2" and distribution is None:
        raise ValueError("Redis-enabled model requires distribution binding")
    if not force:
        runs = {
            rid
            for rid, r in state["runs"].items()
            if r["fingerprint"] == fingerprint and r["status"] in {"published", "rejected"}
        }
        if runs:
            versions = [
                v
                for v, e in state["versions"].items()
                if e["producer"] in runs and e["published"] and e["state"] == "available"
            ]
            if distribution is not None and versions:
                # Never undo a later publication or explicit rollback on an old duplicate trigger.
                current = state["current"]
                if current in versions:
                    return sync_version(distribution, current)
                return {"status": "skipped", "reason": "superseded_input"}
            return {"status": "skipped", "reason": "duplicate_input"}
    if not registry.start(owner, fingerprint, force=force):
        return {"status": "skipped", "reason": "duplicate_input"}
    config["output"]["run_id"] = owner
    status = "failed"
    try:
        trained = execute_train(config)
        root = Path(trained["run_dir"])
        metrics = execute_test(root)
        from recommend.train.comm.auc_events import record_events

        try:
            events = record_events(root, config, metrics)
            if test_observer is not None:
                test_observer(events)
        except Exception as error:
            # Observability must not change training or publication outcomes.
            import sys

            print(
                json.dumps(
                    {"event": "test_auc_reporting_failed", "error_type": type(error).__name__}
                ),
                file=sys.stderr,
                flush=True,
            )
        decision = gate(config, metrics)
        (root / "publication-gate.json").write_bytes(encode(decision))
        if not decision["passed"]:
            status = "rejected"
            return {"status": status, "run_dir": str(root), "gate": decision}
        package = execute_export(root, export_root, version=owner)
        version = publish_prepared(package, registry, owner, run_dir=root)
        status = "published"
        result = {"status": status, "model_version": version, "run_dir": str(root)}
    except OperationPaused as error:
        status = "paused"
        return {**skipped(error), "run_dir": str(root)}
    finally:
        try:
            if status != "published":
                root = Path(config["output"]["root"]) / owner
                archive_unpublished_run(root, config, registry, owner)
        finally:
            registry.finish(owner, status)

    if distribution is not None:
        result.update(sync_version(distribution, version, policy_checked=True))
    return result


app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command("validate")
def validate(config: Annotated[Path, typer.Option("--config")]) -> None:
    value = load_config(config)
    typer.echo(
        json.dumps(
            {"status": "valid", "model_type": value["model_type"], "tasks": list(value["tasks"])}
        )
    )


@app.command("train")
def train(
    config: Annotated[Path, typer.Option("--config")],
    local_root: Annotated[Path | None, typer.Option("--local-root")] = None,
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
) -> None:
    result = execute_train_local(load_config(config), local_root=local_root, resume=resume)
    typer.echo(json.dumps({"status": result["status"], "run_dir": result["run_dir"]}))


@app.command("test")
def test(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    local_root: Annotated[Path | None, typer.Option("--local-root")] = None,
) -> None:
    typer.echo(json.dumps(execute_test_local(run_dir, local_root=local_root)))


@app.command("search")
def search(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
    local_root: Annotated[Path | None, typer.Option("--local-root")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    if dry_run:
        _, trials = expand_search(config, output)
        typer.echo(json.dumps({"trial_count": len(trials), "experiments": [n for n, _ in trials]}))
    else:
        execute_search(config, output, execute_train, local_root=local_root)
        typer.echo(str(output / "search-report.md"))


@app.command("report")
def report(
    run_dir: Annotated[list[Path], typer.Option("--run-dir")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    from recommend.train.comm.report import write_final_report

    write_final_report(run_dir, output)
    typer.echo(str(output))


@app.command("export")
def export(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    output: Annotated[Path, typer.Option("--output")],
    version: Annotated[str | None, typer.Option("--version")] = None,
) -> None:
    from recommend.train.export_model import export_model

    typer.echo(str(export_model(run_dir, output, version=version)))


@app.command("daily")
def daily(
    config: Annotated[Path, typer.Option("--config")],
    daily_root: Annotated[str, typer.Option("--daily-root")],
    frozen_root: Annotated[str, typer.Option("--frozen-root")],
    cutoff: Annotated[str, typer.Option("--cutoff")],
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    export_root: Annotated[Path, typer.Option("--export-root")],
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    try:
        check(artifact_root, "training")
    except OperationPaused as error:
        typer.echo(json.dumps(skipped(error)))
        return

    from recommend.train.publish_model import distribution_from_environment

    value = load_config(config)
    result = daily_run(
        value,
        daily_root=daily_root,
        frozen_root=frozen_root,
        cutoff=cutoff,
        artifact_root=artifact_root,
        export_root=export_root,
        force=force,
        distribution=distribution_from_environment(artifact_root, value["model_name"]),
    )
    typer.echo(json.dumps(result))


@app.command("cleanup")
def cleanup(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    apply: Annotated[bool, typer.Option("--apply")] = False,
    run_root: Annotated[Path | None, typer.Option("--run-root")] = None,
    export_root: Annotated[Path | None, typer.Option("--export-root")] = None,
) -> None:
    try:
        check(artifact_root, "cleanup")
    except OperationPaused as error:
        typer.echo(json.dumps(skipped(error)))
        return

    from recommend.train.comm.artifact_store import ObjectStore
    from recommend.train.comm.publication import Registry, cleanup_local

    if (run_root is None) != (export_root is None):
        raise ValueError("both local roots required")
    registry = Registry(ObjectStore(artifact_root), model_name)
    from recommend.train.publish_model import distribution_from_environment

    distribution = distribution_from_environment(artifact_root, model_name)
    result = distribution.cleanup(apply=apply) if distribution else registry.cleanup(apply=apply)
    if run_root is not None and export_root is not None:
        result["local"] = cleanup_local(registry, run_root, export_root, apply=apply)
    typer.echo(json.dumps(result))


@app.command("rollback")
def rollback(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    version: Annotated[str, typer.Option("--version")],
) -> None:
    try:
        check(artifact_root, "publication")
    except OperationPaused as error:
        typer.echo(json.dumps(skipped(error)))
        return

    from recommend.train.comm.artifact_store import ObjectStore
    from recommend.train.comm.publication import Registry, verify_objects
    from recommend.train.publish_model import distribution_from_environment

    distribution = distribution_from_environment(artifact_root, model_name)
    if distribution is not None:
        typer.echo(json.dumps(distribution.sync(version, rollback=True)))
        return
    registry = Registry(ObjectStore(artifact_root), model_name)
    verify_objects(registry.store, registry.read()[0]["versions"][version])
    registry.publish(version, "explicit-rollback", rollback=True)
    typer.echo(json.dumps({"status": "published", "model_version": version}))


@app.command("pin")
def pin(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    version: Annotated[str, typer.Option("--version")],
    owner: Annotated[str, typer.Option("--owner")],
    release: Annotated[bool, typer.Option("--release")] = False,
) -> None:
    from recommend.train.comm.artifact_store import ObjectStore
    from recommend.train.comm.publication import Registry

    Registry(ObjectStore(artifact_root), model_name).pin(version, owner, release=release)
    typer.echo(json.dumps({"status": "released" if release else "pinned"}))


@app.command("consumer-protocol-ready")
def consumer_protocol_ready(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
) -> None:
    """Enable deletion only after every consumer implements the persistent pin protocol."""
    from recommend.train.comm.artifact_store import ObjectStore
    from recommend.train.comm.publication import Registry

    Registry(ObjectStore(artifact_root), model_name).consumer_ready()
    typer.echo(json.dumps({"consumer_protocol_ready": True}))


@app.command("publish")
def publish(
    package: Annotated[Path, typer.Option("--package")],
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    run_dir: Annotated[Path | None, typer.Option("--run-dir")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    try:
        check(artifact_root, "publication")
    except OperationPaused as error:
        typer.echo(json.dumps(skipped(error)))
        return

    from recommend.train.publish_model import distribution_from_environment, execute_publish

    name = json.loads((package / "manifest.json").read_bytes())["model_name"]
    result = execute_publish(
        package,
        artifact_root,
        run_dir=run_dir,
        force=force,
        distribution=distribution_from_environment(artifact_root, name),
    )
    typer.echo(json.dumps(result))


@app.command("recover-run")
def recover_run(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    owner: Annotated[str, typer.Option("--owner")],
    confirmed_stopped: Annotated[bool, typer.Option("--confirmed-stopped")] = False,
) -> None:
    """Release an orphaned producer only after the operator verifies it has stopped."""
    from recommend.train.comm.artifact_store import ObjectStore
    from recommend.train.comm.publication import Registry

    if not confirmed_stopped:
        raise ValueError("confirm producer termination before releasing ownership")
    registry = Registry(ObjectStore(artifact_root), model_name)
    state, _ = registry.read()
    published = any(e["producer"] == owner and e["published"] for e in state["versions"].values())
    registry.finish(owner, "published" if published else "aborted")
    typer.echo(json.dumps({"status": "recovered", "owner": owner}))


def require_distribution(artifact_root: str, model_name: str) -> Distribution:
    from recommend.train.publish_model import distribution_from_environment

    distribution = distribution_from_environment(artifact_root, model_name)
    if distribution is None:
        raise ValueError("explicit Redis runtime bindings required")
    return distribution


@app.command("redis-sync")
def redis_sync(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    version: Annotated[str, typer.Option("--version")],
    stage_only: Annotated[bool, typer.Option("--stage-only")] = False,
) -> None:
    try:
        check(artifact_root, "publication")
    except OperationPaused as error:
        typer.echo(json.dumps(skipped(error)))
        return

    from recommend.train.publish_model import sync_version

    distribution = require_distribution(artifact_root, model_name)
    result = (
        distribution.sync(version, stage_only=True)
        if stage_only
        else sync_version(distribution, version)
    )
    typer.echo(json.dumps(result))


@app.command("redis-cancel")
def redis_cancel(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    operation_id: Annotated[str, typer.Option("--operation-id")],
    confirmed_stopped: Annotated[bool, typer.Option("--confirmed-stopped")] = False,
) -> None:
    typer.echo(
        json.dumps(
            require_distribution(artifact_root, model_name).cancel(
                operation_id, confirmed_stopped=confirmed_stopped
            )
        )
    )


@app.command("redis-resume")
def redis_resume(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    operation_id: Annotated[str, typer.Option("--operation-id")],
    rebuild_redis: Annotated[bool, typer.Option("--rebuild-redis")] = False,
    confirmed_stopped: Annotated[bool, typer.Option("--confirmed-stopped")] = False,
) -> None:
    try:
        check(artifact_root, "publication")
    except OperationPaused as error:
        typer.echo(json.dumps(skipped(error)))
        return

    typer.echo(
        json.dumps(
            require_distribution(artifact_root, model_name).resume(
                operation_id, confirmed_stopped=confirmed_stopped, rebuild_redis=rebuild_redis
            )
        )
    )


@app.command("redis-rebuild")
def redis_rebuild(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    confirmed_stopped: Annotated[bool, typer.Option("--confirmed-stopped")] = False,
) -> None:
    try:
        check(artifact_root, "publication")
    except OperationPaused as error:
        typer.echo(json.dumps(skipped(error)))
        return

    typer.echo(
        json.dumps(
            require_distribution(artifact_root, model_name).rebuild(
                confirmed_stopped=confirmed_stopped
            )
        )
    )


@app.command("distribution-protocol-ready")
def distribution_protocol_ready(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
) -> None:
    require_distribution(artifact_root, model_name).protocol_ready()
    typer.echo(json.dumps({"distribution_protocol_ready": True}))


if __name__ == "__main__":
    app()
