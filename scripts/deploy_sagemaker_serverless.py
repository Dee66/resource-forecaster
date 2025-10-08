"""
Deploy a packaged model to Amazon SageMaker Serverless Inference (demo script).

This is a safe, example script intended for a portfolio demo. It:
 - uploads a model tarball to S3
 - creates a SageMaker Model pointing at the tarball and a provided container
 - creates a Serverless EndpointConfig and Endpoint
 - optionally invokes the endpoint with a sample payload
 - optionally deletes the endpoint and related resources

Notes:
 - Requires AWS credentials and appropriate SageMaker/ECR/IAM permissions.
 - For a real deployment, prefer using CDK or CloudFormation and tighten IAM.

Usage (example):
  python scripts/deploy_sagemaker_serverless.py \
    --s3-bucket my-bucket --s3-prefix models/forecaster \
    --model-artifact ./artifacts/model_package-20251007-150642.zip \
    --image-uri 123456789012.dkr.ecr.us-east-1.amazonaws.com/forecaster-inference:latest \
    --endpoint-name rf-portfolio-demo --region us-east-1

Cleanup (delete endpoint):
  python scripts/deploy_sagemaker_serverless.py --endpoint-name rf-portfolio-demo --cleanup --region us-east-1

This script intentionally keeps error handling minimal for readability — treat it as an example.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from typing import Optional

import boto3

LOG = logging.getLogger("deploy_sagemaker_serverless")
logging.basicConfig(level=logging.INFO)


def upload_model(s3_client, bucket: str, key: str, artifact_path: str) -> str:
    """Upload the local artifact to S3 and return the s3 uri."""
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(artifact_path)
    LOG.info("Uploading %s to s3://%s/%s", artifact_path, bucket, key)
    s3_client.upload_file(artifact_path, bucket, key)
    return f"s3://{bucket}/{key}"


def create_model(sagemaker_client, model_name: str, image_uri: str, model_data_url: Optional[str], role_arn: Optional[str]):
    LOG.info("Creating SageMaker model %s", model_name)
    primary_container = {"Image": image_uri}
    if model_data_url:
        primary_container["ModelDataUrl"] = model_data_url

    kwargs = {
        "ModelName": model_name,
        "PrimaryContainer": primary_container,
    }
    if role_arn:
        kwargs["ExecutionRoleArn"] = role_arn

    response = sagemaker_client.create_model(**kwargs)
    return response


def create_serverless_endpoint_config(sagemaker_client, config_name: str, model_name: str, max_concurrency: int = 4):
    LOG.info("Creating serverless endpoint config %s", config_name)
    resp = sagemaker_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.m5.large",
                # Serverless specific configuration
                "ServerlessConfig": {
                    "MemorySizeInMB": 4096,
                    "MaxConcurrency": max_concurrency,
                },
            }
        ],
    )
    return resp


def create_endpoint(sagemaker_client, endpoint_name: str, endpoint_config_name: str):
    LOG.info("Creating endpoint %s", endpoint_name)
    resp = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    return resp


def wait_for_endpoint(sagemaker_client, endpoint_name: str, poll_interval: int = 10):
    LOG.info("Waiting for endpoint %s to be InService", endpoint_name)
    while True:
        resp = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        LOG.info("Endpoint status: %s", status)
        if status in ("InService", "Failed", "OutOfService"):
            return resp
        time.sleep(poll_interval)


def invoke_endpoint(runtime_client, endpoint_name: str, payload: dict) -> dict:
    LOG.info("Invoking endpoint %s", endpoint_name)
    resp = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    body = resp["Body"].read()
    return json.loads(body.decode("utf-8"))


def delete_endpoint(sagemaker_client, endpoint_name: str, endpoint_config_name: Optional[str] = None, model_name: Optional[str] = None):
    LOG.info("Deleting endpoint %s", endpoint_name)
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    except Exception:
        LOG.exception("Failed to delete endpoint")
    if endpoint_config_name:
        try:
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        except Exception:
            LOG.exception("Failed to delete endpoint config")
    if model_name:
        try:
            sagemaker_client.delete_model(ModelName=model_name)
        except Exception:
            LOG.exception("Failed to delete model")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--s3-bucket", required=False)
    p.add_argument("--s3-prefix", default="models/forecaster")
    p.add_argument("--model-artifact", required=False)
    p.add_argument("--image-uri", required=True, help="ECR image uri for the inference container")
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--role-arn", default=None, help="Execution role ARN used by SageMaker")
    p.add_argument("--region", default=None)
    p.add_argument("--cleanup", action="store_true")
    p.add_argument("--invoke", action="store_true")
    p.add_argument("--apply", action="store_true", help="Actually perform AWS changes (default: dry-run)")
    return p.parse_args()


def main():
    args = parse_args()

    # Safety guard: require explicit approval to perform AWS operations.
    allowed = args.apply or os.environ.get("ALLOW_AWS_DEPLOY") == "1"
    if not allowed:
        LOG.warning(
            "Dry-run: no AWS operations will be performed. To apply changes, pass --apply and set ALLOW_AWS_DEPLOY=1 in your environment."
        )
        print(
            "DRY-RUN: No AWS calls were made. To enable actual deploys, set ALLOW_AWS_DEPLOY=1 and pass --apply."
        )
        return

    region = args.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    session = boto3.Session(region_name=region)
    s3 = session.client("s3")
    sagemaker = session.client("sagemaker")
    runtime = session.client("sagemaker-runtime")

    model_name = args.model_name or f"rf-model-{int(time.time())}"
    endpoint_config_name = f"{args.endpoint_name}-config"

    model_data_url = None
    try:
        if args.model_artifact:
            if not args.s3_bucket:
                raise SystemExit("--s3-bucket required when uploading model artifact")
            key = f"{args.s3_prefix}/{os.path.basename(args.model_artifact)}"
            model_data_url = upload_model(s3, args.s3_bucket, key, args.model_artifact)

        if not args.role_arn:
            LOG.warning("No role ARN provided — attempting to create model without ExecutionRoleArn may fail")

        create_model(sagemaker, model_name, args.image_uri, model_data_url, args.role_arn)

        create_serverless_endpoint_config(sagemaker, endpoint_config_name, model_name)

        create_endpoint(sagemaker, args.endpoint_name, endpoint_config_name)

        resp = wait_for_endpoint(sagemaker, args.endpoint_name)
        LOG.info("Endpoint final status: %s", resp.get("EndpointStatus"))

        if args.invoke:
            sample = {"start_date": "2025-10-01", "end_date": "2025-10-07", "granularity": "daily"}
            try:
                out = invoke_endpoint(runtime, args.endpoint_name, sample)
                print(json.dumps(out, indent=2))
            except Exception:
                LOG.exception("Invocation failed")
                sys.exit(1)

    except Exception:
        LOG.exception("Deployment failed")
        sys.exit(1)

    if args.cleanup:
        try:
            delete_endpoint(sagemaker, args.endpoint_name, endpoint_config_name=endpoint_config_name, model_name=model_name)
        except Exception:
            LOG.exception("Cleanup failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
