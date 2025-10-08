import os
import time
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
handler_path = ROOT / "lambda" / "shutdown" / "handler.py"
spec = importlib.util.spec_from_file_location("shutdown_handler", str(handler_path))
shutdown_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shutdown_module)  # type: ignore
handler = shutdown_module


class DummyDDB:
    def __init__(self, fail_times: int = 0):
        self.calls = 0
        self.fail_times = fail_times

    def put_item(self, TableName=None, Item=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return {"ConsumedCapacity": {}}


def test_write_audit_event_success():
    dummy = DummyDDB()
    table = "audit-table"
    event = {"recommendation_id": "r1", "timestamp": int(time.time()), "payload": {}}

    # Should not raise
    handler.write_audit_event(table, event, dynamodb_client=dummy, retries=3)
    assert dummy.calls == 1


def test_write_audit_event_retries_and_fail():
    dummy = DummyDDB(fail_times=5)
    table = "audit-table"
    event = {"recommendation_id": "r2", "timestamp": int(time.time()), "payload": {}}

    try:
        handler.write_audit_event(table, event, dynamodb_client=dummy, retries=3)
    except Exception:
        # Expected to raise after retries exhausted
        assert dummy.calls == 3
    else:
        raise AssertionError("Expected write_audit_event to raise after retries")


def test_lambda_handler_audit_flow(monkeypatch):
    # Ensure audit is enabled and forced so dry_run doesn't block write
    monkeypatch.setenv("AUDIT_ENABLED", "true")
    monkeypatch.setenv("AUDIT_FORCE", "true")
    monkeypatch.setenv("AUDIT_TABLE_NAME", "audit-table")
    monkeypatch.setenv("ENV", "test")

    dummy = DummyDDB()

    result = handler.lambda_handler({"resources": ["i-1234567890abcdef0"]}, dry_run_override=True, clients={"dynamodb": dummy})
    assert result["status"] == "success"
    assert result["results"].get("_audit", {}).get("status") == "stored"
    assert dummy.calls == 1
