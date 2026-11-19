"""Validate takeover from expired ECS metadata and exact terminal events."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from recommend_tf.platforms.sagemaker.daily_pipeline import acquire_executor, operation_identity


@pytest.mark.parametrize(
    "ecs_reason,detail_override,allowed",
    [
        ("MISSING", {}, True),
        ("MISSING", {"taskArn": "another-task"}, False),
        ("MISSING", {"clusterArn": "another-cluster"}, False),
        ("MISSING", {"group": "family:other"}, False),
        ("MISSING", {"lastStatus": "RUNNING"}, False),
        ("ACCESS_DENIED", {}, False),
    ],
)
def test_executor_takeover_after_ecs_metadata_expires(ecs_reason, detail_override, allowed):
    state = SimpleNamespace(value={"active": {}}, save=Mock())
    acquire_executor(state, None, "cluster", "old")
    ecs, logs = Mock(), Mock()
    ecs.describe_tasks.return_value = {"failures": [{"arn": "old", "reason": ecs_reason}]}
    event = {
        "id": "event-id",
        "source": "aws.ecs",
        "detail-type": "ECS Task State Change",
        "detail": {
            "taskArn": "old",
            "clusterArn": "cluster-arn",
            "group": "family:idea",
            "lastStatus": "STOPPED",
            **detail_override,
        },
    }
    paginate = logs.get_paginator.return_value.paginate
    paginate.return_value = [{"events": []}, {"events": [{"message": json.dumps(event)}]}]
    binding = {
        "log_group": "/events/idea-failures",
        "cluster_arn": "cluster-arn",
        "task_group": "family:idea",
    }
    if allowed:
        acquire_executor(state, ecs, "cluster", "new", logs=logs, stop_events=binding)
        assert state.value["active"]["executor"] == "new"
        assert state.save.call_count == 2
        paginate.assert_called_once_with(
            logGroupName="/events/idea-failures", filterPattern='{ $.detail.taskArn = "old" }'
        )
    else:
        with pytest.raises(ValueError, match="not confirmed stopped"):
            acquire_executor(state, ecs, "cluster", "new", logs=logs, stop_events=binding)
        assert state.value["active"]["executor"] == "old"
        state.save.assert_called_once()
        if ecs_reason != "MISSING":
            paginate.assert_not_called()


def test_idea_stop_events_do_not_change_in_flight_operation_identity():
    ready = {"cutoff": "2026-11-18"}
    original = {"idea_appconfig": {"model_list_profile": "idea"}}
    updated = {**original, "executor_stop_events": {"log_group": "/events/idea-failures"}}
    assert operation_identity({}, ready, updated) == operation_identity({}, ready, original)
