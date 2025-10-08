import importlib.util
from pathlib import Path
from unittest.mock import Mock
import os


def _load_handler_module():
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        candidate = parent / "lambda" / "shutdown" / "handler.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("shutdown_handler", candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            return module
    raise FileNotFoundError("lambda/shutdown/handler.py not found")


def test_audit_written_on_non_dry_run():
    handler = _load_handler_module()
    old_audit_table = os.environ.get("AUDIT_TABLE_NAME")
    os.environ["AUDIT_TABLE_NAME"] = "RightsizingAudit"
    os.environ["AUDIT_ENABLED"] = "true"
    os.environ["AUDIT_FORCE"] = "false"
    os.environ["DRY_RUN"] = "false"

    dynamo = Mock()
    dynamo.put_item.return_value = {}
    ec2 = Mock()
    rds = Mock()
    ecs = Mock()

    event = {"resources": ["i-123"]}
    res = handler.lambda_handler(event, None, clients={"dynamodb": dynamo, "ec2": ec2, "rds": rds, "ecs": ecs})
    assert res["status"] == "success"
    assert "_audit" in res["results"]
    assert res["results"]["_audit"]["status"] == "stored"
    dynamo.put_item.assert_called_once()

    # restore env
    if old_audit_table is None:
        del os.environ["AUDIT_TABLE_NAME"]
    else:
        os.environ["AUDIT_TABLE_NAME"] = old_audit_table


def test_no_audit_on_dry_run_unless_forced():
    handler = _load_handler_module()
    old_audit_table = os.environ.get("AUDIT_TABLE_NAME")
    os.environ["AUDIT_TABLE_NAME"] = "RightsizingAudit"
    os.environ["AUDIT_ENABLED"] = "true"
    os.environ["AUDIT_FORCE"] = "false"
    os.environ["DRY_RUN"] = "true"

    dynamo = Mock()
    ec2 = Mock()
    rds = Mock()
    ecs = Mock()

    event = {"resources": ["i-123"]}
    res = handler.lambda_handler(event, None, clients={"dynamodb": dynamo, "ec2": ec2, "rds": rds, "ecs": ecs})
    assert res["status"] == "success"
    assert "_audit" not in res["results"]
    dynamo.put_item.assert_not_called()

    # Now force audit even during dry-run
    os.environ["AUDIT_FORCE"] = "true"
    res2 = handler.lambda_handler(event, None, clients={"dynamodb": dynamo})
    assert "_audit" in res2["results"]

    # restore env
    if old_audit_table is None:
        del os.environ["AUDIT_TABLE_NAME"]
    else:
        os.environ["AUDIT_TABLE_NAME"] = old_audit_table


def test_audit_retry_on_transient_failure():
    handler = _load_handler_module()
    old_audit_table = os.environ.get("AUDIT_TABLE_NAME")
    os.environ["AUDIT_TABLE_NAME"] = "RightsizingAudit"
    os.environ["AUDIT_ENABLED"] = "true"
    os.environ["AUDIT_FORCE"] = "false"
    os.environ["DRY_RUN"] = "false"

    dynamo = Mock()
    # First call raises, second call succeeds
    dynamo.put_item.side_effect = [Exception("transient"), {}]
    ec2 = Mock()
    rds = Mock()
    ecs = Mock()

    event = {"resources": ["i-xyz"]}
    res = handler.lambda_handler(event, None, clients={"dynamodb": dynamo, "ec2": ec2, "rds": rds, "ecs": ecs})
    assert res["status"] == "success"
    assert "_audit" in res["results"]
    dynamo.put_item.assert_called()

    # restore env
    if old_audit_table is None:
        del os.environ["AUDIT_TABLE_NAME"]
    else:
        os.environ["AUDIT_TABLE_NAME"] = old_audit_table
