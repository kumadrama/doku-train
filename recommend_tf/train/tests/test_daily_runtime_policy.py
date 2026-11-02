from types import SimpleNamespace
from unittest.mock import Mock, patch

from recommend_tf.platforms.sagemaker.daily_pipeline import step


def call(stage, tmp_path, policy, sm=None):
    state = SimpleNamespace(value={"active": {"stage": stage}}, save=Mock())
    result = step(Mock(), sm or Mock(), Mock(), state, {}, {}, {}, work=tmp_path, policy=policy)
    return result, state


def test_pause_before_training_has_no_training_side_effect(tmp_path):
    policy = Mock()
    policy.allowed.return_value = False
    sm = Mock()
    result, state = call("claimed", tmp_path, policy, sm)
    assert result == "deferred_training"
    state.save.assert_not_called()
    sm.create_training_job.assert_not_called()


def test_pause_before_publication_preserves_uploaded_state(tmp_path):
    policy = Mock()
    policy.allowed.return_value = False
    sm = Mock()
    result, state = call("uploaded", tmp_path, policy, sm)
    assert result == "deferred_publication"
    assert state.value["active"]["stage"] == "uploaded"
    sm.update_endpoint.assert_not_called()


def test_running_deployment_still_finishes_when_policy_is_closed(tmp_path):
    policy = Mock()
    policy.allowed.return_value = False
    state = SimpleNamespace(value={"active": {"stage": "deploying"}})
    with patch(
        "recommend_tf.platforms.sagemaker.daily_pipeline.deploy_step", return_value="pending"
    ) as deploy:
        result = step(
            Mock(), Mock(), Mock(), state, {}, {}, {"deployment": {}}, work=tmp_path, policy=policy
        )
    assert result == "pending"
    deploy.assert_called_once()
    policy.allowed.assert_not_called()


def test_started_rollback_finishes_without_new_policy_admission(tmp_path):
    policy = Mock()
    policy.allowed.side_effect = RuntimeError("configuration unavailable")
    state = SimpleNamespace(value={"active": {"stage": "rollback_pending"}})
    with patch(
        "recommend_tf.platforms.sagemaker.daily_pipeline.rollback_step", return_value="rolled_back"
    ):
        assert (
            step(
                Mock(),
                Mock(),
                Mock(),
                state,
                {},
                {},
                {"deployment": {}},
                work=tmp_path,
                policy=policy,
            )
            == "rolled_back"
        )
    policy.allowed.assert_not_called()
