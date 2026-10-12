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
            {"status": "valid", "model_name": value["model_name"], "tasks": list(value["tasks"])}
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


if __name__ == "__main__":
    app()
