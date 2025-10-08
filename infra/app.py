#!/usr/bin/env python3
"""
CDK Application for Resource Forecaster

Deploys the complete Resource Forecaster infrastructure across environments.
"""

import os
import aws_cdk as cdk
from forecaster_stack import ForecasterStack

# Import the repo config loader (safe, pure-Python, no AWS calls)
from src.forecaster.config import load_config


def main():
    """Main CDK application entry point."""
    app = cdk.App()
    
    # Get environment from context or environment variable
    environment = app.node.try_get_context("environment") or os.environ.get("ENVIRONMENT", "dev")
    
    # Environment-specific configurations
    env_configs = {
        "dev": {
            "account": app.node.try_get_context("dev_account") or os.environ.get("CDK_DEFAULT_ACCOUNT"),
            "region": app.node.try_get_context("dev_region") or os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
        },
        "staging": {
            "account": app.node.try_get_context("staging_account") or os.environ.get("CDK_DEFAULT_ACCOUNT"),
            "region": app.node.try_get_context("staging_region") or os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
        },
        "prod": {
            "account": app.node.try_get_context("prod_account") or os.environ.get("CDK_DEFAULT_ACCOUNT"),
            "region": app.node.try_get_context("prod_region") or os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
        }
    }
    
    env_config = env_configs.get(environment, env_configs["dev"])

    # Load environment-specific YAML config (optional). This is a local file read
    # and will not touch AWS. It provides bucket names, VPC ids, and other overrides.
    try:
        config = load_config(environment)
    except FileNotFoundError:
        config = None
    
    # Create the stack
    # Prepare stack kwargs and pass through config-driven overrides where present
    stack_kwargs = dict(
        environment=environment,
        env=cdk.Environment(
            account=env_config["account"],
            region=env_config["region"]
        ),
        description=f"Resource Forecaster infrastructure for {environment} environment"
    )

    if config is not None:
        # Only pass through a few safe overrides: explicit bucket names and VPC id
        infra = config.infrastructure
        # pydantic model from load_config returns proper attribute access
        if getattr(infra, "model_bucket", None):
            stack_kwargs["model_bucket_name"] = infra.model_bucket
        if getattr(infra, "data_bucket", None):
            stack_kwargs["data_bucket_name"] = infra.data_bucket
        if getattr(infra, "vpc_id", None):
            stack_kwargs["vpc_id"] = infra.vpc_id

    stack = ForecasterStack(app, f"resource-forecaster-{environment}", **stack_kwargs)
    
    # Add tags to all resources (use kebab-case project name)
    cdk.Tags.of(stack).add("App", "resource-forecaster")
    cdk.Tags.of(stack).add("Environment", environment)
    cdk.Tags.of(stack).add("CostCenter", "MLOps")
    cdk.Tags.of(stack).add("Owner", "DataScience")
    cdk.Tags.of(stack).add("Project", "FinOpsForecasting")
    
    app.synth()


if __name__ == "__main__":
    main()