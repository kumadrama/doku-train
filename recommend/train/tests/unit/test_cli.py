from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from recommend.train.run import app
from recommend.train.tests.fixtures import build_training_fixture

runner = CliRunner()


def test_validate_and_backtest_commands_use_finder_style_entrypoint(tmp_path: Path) -> None:
    fixture = build_training_fixture(tmp_path)
    validate = runner.invoke(
        app,
        [
            "rerank",
            "validate",
            "--manifest",
            fixture.manifest_uri,
            "--config",
            str(fixture.config_path),
        ],
    )
    assert validate.exit_code == 0, validate.output
    assert json.loads(validate.output)["status"] == "VALID"
    backtest = runner.invoke(
        app,
        [
            "rerank",
            "backtest",
            "--manifest",
            fixture.manifest_uri,
            "--config",
            str(fixture.config_path),
        ],
    )
    assert backtest.exit_code == 0, backtest.output
    assert json.loads(backtest.output)["fold_count"] == 1


def test_cli_failure_is_nonzero_structured_and_redacted(tmp_path: Path) -> None:
    fixture = build_training_fixture(tmp_path)
    result = runner.invoke(
        app,
        [
            "rerank",
            "validate",
            "--manifest",
            f"{fixture.manifest_uri}?token=secret",
            "--config",
            str(fixture.config_path),
            "--run-id",
            "bad-run",
        ],
    )
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["run_id"] == "bad-run"
    assert payload["category"] == "INPUT_CONTRACT_INVALID"
    assert "secret" not in result.output
