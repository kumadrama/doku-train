from __future__ import annotations

import inspect

from recommend.train.comm import model_exporter
from recommend.train.comm.model_exporter import ExportRequest, ModelExporter
from recommend.train.comm.training_output import TrainingOutputUris


def test_model_exporter_is_protocol_only_and_accepts_successful_training_outputs() -> None:
    outputs = TrainingOutputUris("checkpoint", "metrics", "features", "lineage")
    request = ExportRequest(run_id="run-1", training_outputs=outputs)
    assert request.training_outputs is outputs
    assert ModelExporter._is_protocol is True
    assert not any(
        inspect.isclass(value)
        and value.__module__ == model_exporter.__name__
        and issubclass(value, ModelExporter)
        and value is not ModelExporter
        for value in vars(model_exporter).values()
    )


def test_export_boundary_has_no_onnx_dependency_or_pipeline_registration() -> None:
    source = inspect.getsource(model_exporter).casefold()
    assert "onnx" not in source
    from recommend.train.comm import training_pipeline

    assert "model_exporter" not in inspect.getsource(training_pipeline)
