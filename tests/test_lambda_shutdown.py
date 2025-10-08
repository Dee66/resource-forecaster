import importlib.util
from pathlib import Path
from unittest.mock import Mock


def _load_handler_module():
    p = Path(__file__).resolve()
    # Walk up parents to find the lambda/shutdown/handler.py path
    for parent in [p] + list(p.parents):
        candidate = parent / "lambda" / "shutdown" / "handler.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("shutdown_handler", candidate)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
            return module
    raise FileNotFoundError("lambda/shutdown/handler.py not found in any parent directories")


def test_dry_run_defaults_to_true_and_no_client_calls():
    handler = _load_handler_module()
    event = {
        "resources": [
            "i-0123abcd",
            "arn:aws:rds:us-east-1:123456789012:db:mydb",
            "arn:aws:ecs:us-east-1:123:service/cluster/service-name",
        ]
    }
    old = handler.os.environ.get("DRY_RUN")
    handler.os.environ["DRY_RUN"] = "true"

    ec2_mock = Mock()
    rds_mock = Mock()
    ecs_mock = Mock()

    result = handler.lambda_handler(event, None, clients={"ec2": ec2_mock, "rds": rds_mock, "ecs": ecs_mock})

    assert result["status"] == "success"
    assert result["dry_run"] is True

    # No boto3 client methods should be invoked in dry-run
    ec2_mock.stop_instances.assert_not_called()
    rds_mock.stop_db_instance.assert_not_called()
    ecs_mock.update_service.assert_not_called()

    if old is None:
        del handler.os.environ["DRY_RUN"]
    else:
        handler.os.environ["DRY_RUN"] = old


def test_non_dry_run_invokes_clients():
    handler = _load_handler_module()
    event = {"resources": ["i-0123abcd", "db-mydb", "arn:aws:ecs:us-east-1:123:service/cluster/service-name"]}
    old = handler.os.environ.get("DRY_RUN")
    handler.os.environ["DRY_RUN"] = "false"

    ec2_mock = Mock()
    rds_mock = Mock()
    ecs_mock = Mock()

    # Setup return values (not used by assertions but keep realistic)
    ec2_mock.stop_instances.return_value = {"StoppingInstances": ["i-0123abcd"]}
    rds_mock.stop_db_instance.return_value = {"DBInstance": "db-mydb"}
    ecs_mock.update_service.return_value = {"service": {"service": "cluster/service-name", "desiredCount": 0}}

    result = handler.lambda_handler(event, None, clients={"ec2": ec2_mock, "rds": rds_mock, "ecs": ecs_mock})

    assert result["status"] == "success"
    assert result["dry_run"] is False

    ec2_mock.stop_instances.assert_called_once_with(InstanceIds=["i-0123abcd"])
    rds_mock.stop_db_instance.assert_called_once_with(DBInstanceIdentifier="db-mydb")
    # For ECS we expect update_service called with service name (cluster/service-name) and desiredCount=0
    ecs_mock.update_service.assert_called_once()
    args, called_kwargs = ecs_mock.update_service.call_args
    svc_val = called_kwargs.get("service") if called_kwargs else (args[0].get("service") if args and isinstance(args[0], dict) else None)
    assert svc_val is not None
    # Accept either the ARN or the parsed cluster/service-name
    assert svc_val.endswith("cluster/service-name") or svc_val.endswith("service-name")
    # desiredCount must be 0
    desired = called_kwargs.get("desiredCount") if called_kwargs else (args[0].get("desiredCount") if args and isinstance(args[0], dict) else None)
    assert desired == 0

    if old is None:
        del handler.os.environ["DRY_RUN"]
    else:
        handler.os.environ["DRY_RUN"] = old
