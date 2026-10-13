from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from recommend.train.comm.model_params_validator import load_config
from recommend.train.comm.search import execute_search, expand_search
from recommend.train.train_rerank import execute_test_local, execute_train, execute_train_local

app = typer.Typer(no_args_is_help=True)


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
    from recommend.train.comm.daily_publication import daily_run

    result = daily_run(
        load_config(config),
        daily_root=daily_root,
        frozen_root=frozen_root,
        cutoff=cutoff,
        artifact_root=artifact_root,
        export_root=export_root,
        force=force,
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
    from recommend.train.comm.artifact_store import ObjectStore
    from recommend.train.comm.publication import Registry, cleanup_local

    if (run_root is None) != (export_root is None):
        raise ValueError("both local roots required")
    registry = Registry(ObjectStore(artifact_root), model_name)
    result = registry.cleanup(apply=apply)
    if run_root is not None and export_root is not None:
        result["local"] = cleanup_local(registry, run_root, export_root, apply=apply)
    typer.echo(json.dumps(result))


@app.command("rollback")
def rollback(
    artifact_root: Annotated[str, typer.Option("--artifact-root")],
    model_name: Annotated[str, typer.Option("--model-name")],
    version: Annotated[str, typer.Option("--version")],
) -> None:
    from recommend.train.comm.artifact_store import ObjectStore
    from recommend.train.comm.publication import Registry, verify_objects

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
    import hashlib
    import uuid

    from recommend.train.comm.artifact_store import ObjectStore
    from recommend.train.comm.daily_publication import publish_package
    from recommend.train.comm.publication import Registry, verify_objects

    raw = (package / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    registry = Registry(ObjectStore(artifact_root), manifest["model_name"])
    existing = registry.read()[0]["versions"].get(manifest["model_version"])
    if existing is not None and existing["published"]:
        if existing["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("version collision")
        if existing["state"] in {"deleting", "deleted"}:
            raise ValueError("published version has retired")
        verify_objects(registry.store, existing)
        typer.echo(json.dumps({"status": "skipped", "reason": "already_published"}))
        return
    owner = uuid.uuid4().hex
    if not registry.start(owner, hashlib.sha256(raw).hexdigest(), force=force):
        typer.echo(json.dumps({"status": "skipped", "reason": "duplicate_package"}))
        return
    status = "failed"
    try:
        version = publish_package(package, registry, owner, run_dir=run_dir)
        status = "published"
        typer.echo(json.dumps({"status": status, "model_version": version}))
    finally:
        registry.finish(owner, status)


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


if __name__ == "__main__":
    app()
