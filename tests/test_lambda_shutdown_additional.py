import importlib.util
from pathlib import Path
from unittest.mock import Mock


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


def test_invalid_resources_type_returns_error():
    handler = _load_handler_module()
    result = handler.lambda_handler({"resources": "not-a-list"}, None)
    assert result["status"] == "error"
    assert "message" in result


def test_unhandled_resource_is_reported():
    handler = _load_handler_module()
    event = {"resources": ["some-unknown-resource"]}
    res = handler.lambda_handler(event, None, dry_run_override=True)
    assert res["status"] == "success"
    assert "some-unknown-resource" in res["results"]
    assert res["results"]["some-unknown-resource"]["status"] == "unhandled"


def test_ec2_multiple_instances_result_in_multiple_calls():
    handler = _load_handler_module()
    ec2_mock = Mock()
    # Call with two instance ids; use dry_run_override=False so clients are invoked
    event = {"resources": ["i-aaa", "i-bbb"]}
    handler.lambda_handler(event, None, dry_run_override=False, clients={"ec2": ec2_mock})
    assert ec2_mock.stop_instances.call_count == 2
    calls = ec2_mock.stop_instances.call_args_list
    assert calls[0].kwargs["InstanceIds"] == ["i-aaa"]
    assert calls[1].kwargs["InstanceIds"] == ["i-bbb"]


def test_rds_arn_parsed_and_stop_called_with_identifier():
    handler = _load_handler_module()
    rds_mock = Mock()
    arn = "arn:aws:rds:us-east-1:123456789012:db:mydb"
    handler.lambda_handler({"resources": [arn]}, None, dry_run_override=False, clients={"rds": rds_mock})
    rds_mock.stop_db_instance.assert_called_once_with(DBInstanceIdentifier="mydb")
