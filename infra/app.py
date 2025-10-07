#!/usr/bin/env python3
"""
CDK Application for Resource Forecaster

Deploys the complete Resource Forecaster infrastructure across environments.
"""

import os
import aws_cdk as cdk
from forecaster_stack import ForecasterStack


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
    
    # Create the stack
    stack = ForecasterStack(
        app, 
        f"ResourceForecaster-{environment.title()}",
        environment=environment,
        env=cdk.Environment(
            account=env_config["account"],
            region=env_config["region"]
        ),
        description=f"Resource Forecaster infrastructure for {environment} environment"
    )
    
    # Add tags to all resources
    cdk.Tags.of(stack).add("App", "ResourceForecaster")
    cdk.Tags.of(stack).add("Environment", environment)
    cdk.Tags.of(stack).add("CostCenter", "MLOps")
    cdk.Tags.of(stack).add("Owner", "DataScience")
    cdk.Tags.of(stack).add("Project", "FinOpsForecasting")
    
    app.synth()


if __name__ == "__main__":
    main()