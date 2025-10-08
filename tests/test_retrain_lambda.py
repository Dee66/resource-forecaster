
import os
import json
import importlib.util
from pathlib import Path

import pytest


# Dynamically load the retrain handler module (module name 'lambda' would clash with keyword)
ROOT = Path(__file__).resolve().parent.parent
handler_path = ROOT / "lambda" / "retrain" / "handler.py"
spec = importlib.util.spec_from_file_location("retrain_handler", str(handler_path))
retrain_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain_module)  # type: ignore
retrain_handler = retrain_module


class DummySFNClient:
    def __init__(self, raise_exc: bool = False):
        self.raise_exc = raise_exc
        self.called = False
        self.kwargs = None

    def start_execution(self, **kwargs):
        self.called = True
        self.kwargs = kwargs
        if self.raise_exc:
            raise RuntimeError("start failed")
        return {"executionArn": "arn:aws:states:us-east-1:123456789012:execution:sm:exec-1"}


def test_retrain_lambda_happy_path(monkeypatch):
    # Arrange
    monkeypatch.setenv("STATE_MACHINE_ARN", "arn:aws:states:us-east-1:123:stateMachine:sm")
    dummy = DummySFNClient()

    # Act
    result = retrain_handler.lambda_handler({"some": "payload"}, clients={"stepfunctions": dummy})

    # Assert
    assert result["status"] == "started"
    assert "executionArn" in result
    assert dummy.called is True
    # Validate the state machine ARN used and that input contains our trigger
    assert dummy.kwargs["stateMachineArn"] == "arn:aws:states:us-east-1:123:stateMachine:sm"
    payload = json.loads(dummy.kwargs["input"])
    assert payload["trigger"] == "rmse_alarm"


def test_retrain_lambda_start_execution_failure(monkeypatch):
    # Arrange
    monkeypatch.setenv("STATE_MACHINE_ARN", "arn:aws:states:us-east-1:123:stateMachine:sm")
    dummy = DummySFNClient(raise_exc=True)

    # Act
    result = retrain_handler.lambda_handler({"detail": "alarm"}, clients={"stepfunctions": dummy})

    # Assert
    assert result["status"] == "error"
    assert "message" in result
    assert dummy.called is True


def test_retrain_lambda_no_state_machine_env(monkeypatch):
    # Ensure env var is not set
    monkeypatch.delenv("STATE_MACHINE_ARN", raising=False)

    # Act
    result = retrain_handler.lambda_handler({})

    # Assert
    assert result["status"] == "error"
    assert result["message"] == "no state machine configured"
