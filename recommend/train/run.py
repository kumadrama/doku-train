from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from recommend.train.comm.errors import classify_error
from recommend.train.train_rerank import execute_backtest, execute_train, validate_input

app = typer.Typer(no_args_is_help=True)
rerank_app = typer.Typer(no_args_is_help=True)
app.add_typer(rerank_app, name="rerank")


def _fail(error: Exception, *, run_id: str, stage: str) -> None:
    typer.echo(classify_error(error, run_id=run_id, stage=stage).to_json())
    raise typer.Exit(code=1)


@rerank_app.command("validate")
def validate_command(
    manifest: Annotated[str, typer.Option("--manifest")],
    config: Annotated[Path, typer.Option("--config")],
    run_id: Annotated[str, typer.Option("--run-id")] = "validate",
) -> None:
    try:
        validated = validate_input(manifest_uri=manifest, config_path=config)
    except Exception as error:
        _fail(error, run_id=run_id, stage="validate")
    typer.echo(
        json.dumps(
            {
                "status": "VALID",
                "dataset_manifest_digest": validated.manifest.content_sha256,
                "row_count": validated.manifest.row_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


@rerank_app.command("train")
def train_command(
    manifest: Annotated[str, typer.Option("--manifest")],
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[str, typer.Option("--output")],
    run_id: Annotated[str, typer.Option("--run-id")],
) -> None:
    try:
        result = execute_train(
            manifest_uri=manifest,
            config_path=config,
            output_prefix=output,
            run_id=run_id,
        )
    except Exception as error:
        _fail(error, run_id=run_id, stage="train")
    typer.echo(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))


@rerank_app.command("backtest")
def backtest_command(
    manifest: Annotated[str, typer.Option("--manifest")],
    config: Annotated[Path, typer.Option("--config")],
    run_id: Annotated[str, typer.Option("--run-id")] = "backtest",
) -> None:
    try:
        windows = execute_backtest(manifest_uri=manifest, config_path=config)
    except Exception as error:
        _fail(error, run_id=run_id, stage="backtest")
    typer.echo(
        json.dumps(
            {"status": "VALID", "fold_count": len(windows)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
