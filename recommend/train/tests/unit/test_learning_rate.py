import copy

import pytest
from jsonschema import ValidationError

from recommend.train.comm.learning_rate import learning_rate_for_epoch
from recommend.train.comm.model_params_validator import validate_config
from recommend.train.tests.experiment_fixture import make_experiment


def test_schedule_boundaries():
    opt = {"learning_rate": 0.004}
    assert learning_rate_for_epoch(opt, 40) == 0.004
    opt["learning_rate_schedule"] = {
        "type": "piecewise_linear",
        "points": [
            {"epoch": 1, "learning_rate": 0.004},
            {"epoch": 10, "learning_rate": 0.004},
            {"epoch": 30, "learning_rate": 0.001},
        ],
    }
    for epoch, expected in [
        (1, 0.004),
        (10, 0.004),
        (11, 0.00385),
        (20, 0.0025),
        (30, 0.001),
        (31, 0.001),
        (40, 0.001),
    ]:
        assert learning_rate_for_epoch(opt, epoch) == pytest.approx(expected)
    opt["learning_rate_schedule"]["type"] = "piecewise_constant"
    assert learning_rate_for_epoch(opt, 29) == 0.004
    assert learning_rate_for_epoch(opt, 30) == 0.001


@pytest.mark.parametrize(
    "case", ["first", "duplicate", "order", "beyond", "mismatch", "zero", "nan", "unknown", "empty"]
)
def test_invalid_schedule(tmp_path, case):
    c = make_experiment(tmp_path, "dnn")
    c["max_epochs"] = 40
    lr = c["optimizer"]["learning_rate"]
    schedule = {
        "type": "piecewise_linear",
        "points": [{"epoch": 1, "learning_rate": lr}, {"epoch": 20, "learning_rate": lr / 2}],
    }
    c["optimizer"]["learning_rate_schedule"] = copy.deepcopy(schedule)
    s = c["optimizer"]["learning_rate_schedule"]
    if case == "first":
        s["points"][0]["epoch"] = 2
    elif case == "duplicate":
        s["points"][1]["epoch"] = 1
    elif case == "order":
        s["points"].append({"epoch": 10, "learning_rate": lr})
    elif case == "beyond":
        s["points"][1]["epoch"] = 41
    elif case == "mismatch":
        s["points"][0]["learning_rate"] = lr * 2
    elif case in ("zero", "nan"):
        s["points"][1]["learning_rate"] = 0 if case == "zero" else float("nan")
    elif case == "unknown":
        s["type"] = "unknown"
    else:
        s["points"] = []
    with pytest.raises((ValueError, ValidationError)):
        validate_config(c)
