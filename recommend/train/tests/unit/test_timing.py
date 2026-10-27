from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from recommend.train.comm import timing
from recommend.train.comm.model_params_validator import validate_config
from recommend.train.tests.synthetic_config import synthetic_config


def summaries(root):
    return [json.loads(p.read_text()) for p in (root / "timing").glob("*-summary.json")]


def test_disabled_does_not_read_clock_or_write(tmp_path, monkeypatch, capsys):
    def forbidden():
        raise AssertionError("disabled profiling read clock")

    monkeypatch.setattr(timing.time, "perf_counter", forbidden)
    with timing.session({}, tmp_path, "train") as timer, timer.span("data", report=True):
        assert list(timer.iterate("read", iter([1, 2]))) == [1, 2]
        timer.epoch(1)
        timer.emit("ignored")
    assert not (tmp_path / "timing").exists()
    assert capsys.readouterr().out == ""


def test_clock_hierarchy_progress_epoch_and_stdout(tmp_path, monkeypatch, capsys):
    clock = [0.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])
    with timing.session(
        {"profiling": {"enabled": True, "print_logs": True}}, tmp_path, "train"
    ) as timer:
        with timer.span("data", report=True):
            with timer.span("decode", count=12, unit="rows"):
                clock[0] = 59
            assert '"event": "progress"' not in capsys.readouterr().out
            with timer.span("decode", count=8, unit="rows"):
                clock[0] = 61
        timer.epoch(1)
        with timer.span("forward"):
            clock[0] = 64
        timer.epoch(2)
    events = [
        json.loads(line)
        for p in (tmp_path / "timing").glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert len([e for e in events if e["event"] == "progress"]) == 1
    assert "DOKU_TIMING " in capsys.readouterr().out
    summary = summaries(tmp_path)[0]
    assert summary["elapsed_seconds"] == 64
    assert summary["stages"]["data/decode"]["count"] == 20
    assert summary["stages"]["data/decode"]["elapsed_seconds"] == 61
    assert summary["stages"]["data"]["elapsed_seconds"] == 61  # inclusive, not additive
    epochs = [e for e in events if e["event"] == "epoch_summary"]
    assert epochs[1]["delta_seconds"]["forward"] == 3
    assert epochs[1]["delta_seconds"]["data"] == 0


def test_generator_measurement_excludes_consumer_and_failure_survives(
    tmp_path, monkeypatch, capsys
):
    clock = [0.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])

    def source():
        for i in range(2):
            clock[0] += 2
            yield i

    with (
        pytest.raises(ValueError, match="private sample"),
        timing.session({"profiling": {"enabled": True}}, tmp_path, "test") as timer,
        timer.span("test", report=True),
    ):
        for _ in timer.iterate("fetch", source()):
            clock[0] += 100
        raise ValueError("private sample must not appear in timing")
    summary = summaries(tmp_path)[0]
    assert summary["status"] == "failed"
    assert summary["stages"]["test/fetch"]["elapsed_seconds"] == 4
    assert summary["stages"]["test"]["elapsed_seconds"] == 204
    assert "private sample" not in next((tmp_path / "timing").glob("*.jsonl")).read_text()
    assert capsys.readouterr().out == ""
    assert not timing.current().enabled


def test_explicit_collector_supports_worker_accumulation(tmp_path):
    with timing.session({"profiling": {"enabled": True}}, tmp_path, "train") as timer:

        def work(_):
            with timer.span("shard_download", unit="shards"):
                pass

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(work, range(10)))
    assert summaries(tmp_path)[0]["stages"]["shard_download"]["count"] == 10


def test_profiling_config_rejects_invalid_combinations():
    config = synthetic_config("deepfm")
    assert "profiling" not in validate_config(config)
    config["profiling"] = {"enabled": False, "print_logs": True}
    with pytest.raises(ValueError, match="requires"):
        validate_config(config)
    config["profiling"] = {"enabled": True, "print_logs": False}
    validate_config(config)
    config["fm_layers"] = []
    with pytest.raises(ValueError, match="only for DeepFM"):
        validate_config(config)
    tree = synthetic_config("xgboost")
    tree["profiling"] = {"enabled": True}
    with pytest.raises(Exception, match="profiling"):
        validate_config(tree)


def test_process_wrapper_forwards_only_enabled_timing(capsys):
    import sys

    from recommend.train.comm.process_logs import run_captured

    args = [
        sys.executable,
        "-c",
        "import sys; print('ordinary'); print('DOKU_TIMING {}'); "
        "print('error', file=sys.stderr); sys.exit(7)",
    ]
    off = run_captured(args, {})
    assert off.returncode == 7
    assert "ordinary" in off.stdout
    assert capsys.readouterr().out == ""
    on = run_captured(args, {"profiling": {"enabled": True, "print_logs": True}})
    assert on.returncode == 7
    assert "error" in on.stderr
    assert capsys.readouterr().out == "DOKU_TIMING {}\n"
