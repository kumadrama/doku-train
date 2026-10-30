"""Train, select using validation only, then evaluate the selected checkpoint on test."""

import json
from pathlib import Path

from .comm.artifact_integrity import seal_run
from .comm.training_pipeline import evaluate_checkpoint
from .train_rerank import train


def execute(config, root, publish=None, local_root=None):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)

    owned = False

    def created():
        nonlocal owned
        owned = True

    def sync(record=None):
        config_path = root / "experiment-config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(config, indent=2))
        if publish:
            publish(root)

    try:
        train(config, root, local_root=local_root, on_epoch=sync, on_created=created)
        selection = json.loads((root / "training/selection.json").read_text())
        if "checkpoint" not in selection:
            status = {
                "status": "trained_without_selection",
                "test": "not_run",
                "selection": selection,
            }
        else:
            runtime = json.loads((root / "resolved-config.json").read_text())
            checkpoint = Path(selection["checkpoint"])
            measured = evaluate_checkpoint(runtime, root / "prepared", checkpoint, root / "test")
            status = {
                "status": "completed",
                "selected_epoch": selection["epoch"],
                "test": "completed",
                "checkpoint": str(checkpoint),
                "test_identity": measured["identity"],
            }
        early_status = root / "training/early-stop.json"
        if early_status.exists():
            status["early_stop"] = json.loads(early_status.read_text())
        (root / "experiment-status.json").write_text(json.dumps(status, indent=2))
        (root / "experiment-config.json").write_text(json.dumps(config, sort_keys=True))
        seal_run(root, "experiment-status.json")
        sync()
        return status
    except Exception as error:
        if owned:
            (root / "COMMITTED.json").unlink(missing_ok=True)
            (root / "experiment-status.json").write_text(
                json.dumps({"status": "failed", "error_type": type(error).__name__})
            )
            try:
                sync()
            except Exception as upload_error:
                print(
                    json.dumps(
                        {
                            "event": "failure_upload_failed",
                            "error_type": type(upload_error).__name__,
                        }
                    ),
                    flush=True,
                )
        raise
