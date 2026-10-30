"""Export sealed successful runs without running training or implementing serving."""

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

from .comm.artifact_integrity import checksum, files, reserve_output, verify_run
from .comm.datasvr.prepared import digest, organize, tensor_batch
from .models.rerank.export_adapter import export
from .models.rerank.model_params import parameters


def exporter_identity():
    return digest(
        {
            str(p.relative_to(Path(__file__).parent)): checksum(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))
            if "tests" not in p.parts
        }
    )


def synthetic_batches(config, vocabs, feature_contract):
    """Non-user fixtures for empty/default, OOV, duplicate and truncated histories."""
    params = parameters(config["model"], vocabs)
    definitions = {f["name"]: f["dtype"] for f in feature_contract["features"]}
    defaults = config["dataset_config"]["feature_default_values"]
    row = {}
    for name, dtype in config["dtypes"].items():
        row[name] = (
            [] if definitions.get(name, "").startswith("list<") else "" if dtype == "str" else 0
        )
    batches = []
    for case in range(3):
        rows = []
        for index in range(case + 1):
            current = copy.deepcopy(row)
            for name in defaults:
                if name in current:
                    current[name] = None
            for group_name, group in params.input_features.items():
                for name in group["features"]:
                    if name not in current:
                        continue  # Derived in graph.
                    dtype = config["dtypes"][name]
                    value = "__synthetic_oov__" if dtype == "str" else 987654321
                    table = vocabs.get(params.embed_name.get(name), {})
                    if case == 2 and table.get("keys"):
                        value = table["keys"][0]
                    if group.get("history_pooling"):
                        cap = config["histories"].get(group_name, {}).get("max_hist_size", 2)
                        if cap > 10000:
                            raise ValueError("synthetic verification history budget exceeded")
                        current[name] = [] if case == 0 else [value] * (cap + 1)
                        if "weight_feature" in group:
                            current[group["weight_feature"]] = [1.0] * len(current[name])
                    elif definitions.get(name, "").startswith("list<"):
                        current[name] = [value]
                    elif case:
                        current[name] = value
            rows.append(organize(current, defaults, params.input_features, config["histories"]))
        batches.append(tensor_batch(rows, params, config["dtypes"]))
    return batches


def execute(run_dir, output, *, outputs, evidence=None):
    root, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    sealed = verify_run(root, evidence)
    status_name = (
        "daily-status.json" if (root / "daily-status.json").exists() else "experiment-status.json"
    )
    status = json.loads((root / status_name).read_text())
    if status.get("status") != "completed":
        raise ValueError("export requires a successful selected run")
    selection = json.loads((root / "training/selection.json").read_text())
    epoch = selection["epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("invalid selected epoch")
    expected_name = f"epoch-{epoch:04d}"
    if (
        Path(selection["checkpoint"]).name != expected_name
        or ".." in Path(selection["checkpoint"]).parts
    ):
        raise ValueError("invalid checkpoint selection path")
    checkpoint = root / "training" / expected_name
    config = json.loads((root / "experiment-config.json").read_text())
    runtime = json.loads((root / "resolved-config.json").read_text())
    vocabs = json.loads((root / "training/feature-state.json").read_text())
    lineage = json.loads((root / "training/lineage.json").read_text())
    record = json.loads((checkpoint / "checkpoint.json").read_text())
    if runtime != {
        **{
            k: v
            for k, v in config.items()
            if k not in ("dataset_config", "histories", "item_filter", "online_config")
        },
        "prepared_identity": runtime["prepared_identity"],
    }:
        raise ValueError("export config differs from resolved training config")
    source = {k: lineage[k] for k in ("source_digest", "lock_digest", "python", "tensorflow")}
    if (
        record["epoch"] != epoch
        or record["identity"] != lineage["identity"]
        or digest({"config": runtime, "vocabs": vocabs, "source": source}) != record["identity"]
    ):
        raise ValueError("checkpoint/config/feature state identity mismatch")
    if (
        len(outputs) != 4
        or not set(outputs) <= config["model"]["tasks"].keys()
        or any(not isinstance(v, str) or not v for v in outputs.values())
    ):
        raise ValueError("exactly four configured outputs with explicit units are required")
    required = {
        status_name,
        "experiment-config.json",
        "resolved-config.json",
        "training/selection.json",
        "training/feature-state.json",
        "training/lineage.json",
        "training/metrics.json",
        "input/record.json",
        "input/manifest.json",
        "input/contracts/features.json",
        "input/contracts/labels.json",
    }
    required.update(
        f"training/{expected_name}/{p.name}" for p in checkpoint.iterdir() if p.is_file()
    )
    if (
        not required <= sealed["files"].keys()
        or not (checkpoint / "model.index").exists()
        or not list(checkpoint.glob("model.data-*"))
    ):
        raise ValueError("incomplete sealed run evidence")
    manifest = json.loads((root / "input/manifest.json").read_text())
    frozen = json.loads((root / "input/record.json").read_text())
    prepared_identity = digest(
        {
            "manifest": manifest,
            "source": source,
            "model": config["model"],
            "defaults": config["dataset_config"]["feature_default_values"],
            "histories": config["histories"],
            "item_filter": config["item_filter"],
        }
    )
    if (
        prepared_identity != runtime["prepared_identity"]
        or prepared_identity != lineage["prepared_identity"]
    ):
        raise ValueError("frozen feature preparation differs from training identity")
    if frozen["manifest_file_sha256"] != checksum(root / "input/manifest.json") or any(
        frozen["contract_digests"][k] != checksum(root / f"input/contracts/{k}.json")
        for k in ("features", "labels")
    ):
        raise ValueError("frozen input record differs from sealed input")
    exporter_source = exporter_identity()
    feature_contract = json.loads((root / "input/contracts/features.json").read_text())
    labels = json.loads((root / "input/contracts/labels.json").read_text())["labels"]
    batches = synthetic_batches(config, vocabs, feature_contract)
    version = digest(
        {
            "identity": record["identity"],
            "checkpoint": {
                k: v
                for k, v in sealed["files"].items()
                if k.startswith(f"training/{expected_name}/model.")
            },
            "outputs": outputs,
            "exporter_source_digest": exporter_source,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        reserve_output(output),
        tempfile.TemporaryDirectory(prefix=".export-", dir=output.parent) as directory,
    ):
        package = Path(directory) / "package"
        package.mkdir()
        signature = export(
            runtime,
            vocabs,
            checkpoint,
            package / "saved_model",
            batches,
            outputs,
            set(labels) - set(config["dataset_config"]["features"]),
            version,
        )
        signature.update(
            version="doku-tf-signature/v1",
            output_units=outputs,
            model_version=version,
            input_stage="organized_padded_features",
            response_order="request_order",
        )
        raw_fields = {value["source_field"] for value in signature["inputs"].values()} - {"length"}
        feature_config = {
            "required_raw_fields": sorted(raw_fields),
            "defaults": {
                k: v
                for k, v in config["dataset_config"]["feature_default_values"].items()
                if k in raw_fields
            },
            "histories": config["histories"],
            "input_features": config["model"]["input_features"],
            "dtypes": {k: v for k, v in config["dtypes"].items() if k in raw_fields},
            "transformations_in_graph": config["model"].get("feature_transformations", []),
        }
        for name, value in {
            "signature.json": signature,
            "feature-state.json": vocabs,
            "feature-config.json": feature_config,
            "lineage.json": {
                **lineage,
                "selected_epoch": epoch,
                "source_evidence_digest": digest(sealed),
                "input": json.loads((root / "input/record.json").read_text()),
                "dataset": {
                    k: json.loads((root / "input/manifest.json").read_text())[k]
                    for k in (
                        "dataset_id",
                        "dataset_version",
                        "feature_schema_version",
                        "label_definition_version",
                        "as_of_ms",
                    )
                },
                "exporter_source_digest": exporter_source,
            },
        }.items():
            (package / name).write_text(json.dumps(value, sort_keys=True, allow_nan=False))
        manifest = {
            "version": "doku-tf-inference/v1",
            "model_version": version,
            "files": files(package),
            "verification": {
                "kind": "synthetic_eval_savedmodel_parity",
                "batches": len(batches),
                "rtol": 1e-6,
                "atol": 1e-7,
            },
        }
        (package / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        # Replace only our exclusively reserved empty directory with the complete package.
        os.rename(package, output)
    return {
        "model_version": version,
        "manifest_sha256": checksum(output / "manifest.json"),
        "path": str(output),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "output", "outputs"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--evidence", help="Audited checksum inventory for historical unsealed runs"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            execute(
                args.run,
                args.output,
                outputs=json.loads(Path(args.outputs).read_text()),
                evidence=json.loads(Path(args.evidence).read_text()) if args.evidence else None,
            )
        )
    )


if __name__ == "__main__":
    main()
