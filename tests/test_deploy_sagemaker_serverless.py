import sys
from unittest import mock

import boto3
from botocore.stub import Stubber


import scripts.deploy_sagemaker_serverless as deploy_mod


def test_dry_run_default(monkeypatch):
    # No ALLOW_AWS_DEPLOY and no --apply should exit early (dry-run)
    monkeypatch.delenv("ALLOW_AWS_DEPLOY", raising=False)
    monkeypatch.setattr(sys, "argv", ["deploy_sagemaker_serverless.py", "--image-uri", "x", "--endpoint-name", "rf-demo"])
    # main() should return early (dry-run) and not raise
    deploy_mod.main()


def test_apply_with_stubbed_clients(tmp_path, monkeypatch):
    # When ALLOW_AWS_DEPLOY=1 and --apply passed, the script will attempt AWS calls.
    monkeypatch.setenv("ALLOW_AWS_DEPLOY", "1")

    session = boto3.Session()
    sagemaker = session.client("sagemaker", region_name="us-east-1")
    s3 = session.client("s3", region_name="us-east-1")
    runtime = session.client("sagemaker-runtime", region_name="us-east-1")

    with Stubber(sagemaker) as sm_stubber, Stubber(s3) as s3_stubber, Stubber(runtime) as rt_stubber:
        # Provide minimal responses for create_model, create_endpoint_config, create_endpoint, describe_endpoint
        sm_stubber.add_response(
            "create_model",
            {"ModelArn": "arn:aws:sagemaker:us-east-1:123456789012:model/rf-model"},
            expected_params=None,
        )
        sm_stubber.add_response(
            "create_endpoint_config",
            {"EndpointConfigArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint-config/rf-demo-config"},
            expected_params=None,
        )
        sm_stubber.add_response(
            "create_endpoint",
            {"EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/rf-demo"},
            expected_params={"EndpointName": "rf-demo", "EndpointConfigName": "rf-demo-config"},
        )
        sm_stubber.add_response(
            "describe_endpoint",
            {
                "EndpointName": "rf-demo",
                "EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/rf-demo",
                "CreationTime": "2025-10-08T00:00:00",
                "LastModifiedTime": "2025-10-08T00:00:00",
                "EndpointStatus": "InService",
            },
            expected_params={"EndpointName": "rf-demo"},
        )

        # Monkeypatch boto3.Session to return our stubbed clients
        real_Session = boto3.Session

        class DummySession:
            def __init__(self, *a, **k):
                pass

            def client(self, svc, region_name=None):
                if svc == "sagemaker":
                    return sagemaker
                if svc == "s3":
                    return s3
                if svc == "sagemaker-runtime":
                    return runtime
                return real_Session().client(svc, region_name=region_name)

        monkeypatch.setattr("boto3.Session", DummySession)

        # Set argv and call main() in-process so our monkeypatches apply
        monkeypatch.setattr(sys, "argv", ["deploy_sagemaker_serverless.py", "--image-uri", "x", "--endpoint-name", "rf-demo", "--apply", "--role-arn", "arn:aws:iam::123456789012:role/demo"])
        deploy_mod.main()
