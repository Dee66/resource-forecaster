"""Pytest configuration and shared fixtures for Resource Forecaster tests."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.forecaster.config import ForecasterConfig


@pytest.fixture
def sample_config() -> ForecasterConfig:
    """Sample configuration for testing."""
    config_data = {
        "environment": "dev",
        "aws_region": "us-east-1",
        "data_source": {
            "cur_bucket": "test-cur-bucket",
            "athena_output_bucket": "test-athena-bucket",
            "athena_database": "test_database",
            "athena_workgroup": "primary",
        },
        "infrastructure": {"model_bucket": "test-model-bucket", "data_bucket": "test-data-bucket"},
    }
    return ForecasterConfig(**config_data)


@pytest.fixture
def sample_cost_data() -> pd.DataFrame:
    """Sample cost data for testing."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D")

    data = {
        "usage_date": dates,
        "daily_cost": [10.5 + i * 0.1 for i in range(len(dates))],
        "service": ["EC2"] * len(dates),
        "cost_center": ["engineering"] * len(dates),
        "project": ["forecaster"] * len(dates),
        "environment": ["dev"] * len(dates),
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for testing."""
    mock_client = MagicMock()
    mock_client.start_query_execution.return_value = {"QueryExecutionId": "test-execution-id"}
    mock_client.get_query_execution.return_value = {
        "QueryExecution": {
            "Status": {"State": "SUCCEEDED"},
            "ResultConfiguration": {"OutputLocation": "s3://test-bucket/results/test-results.csv"},
        }
    }
    return mock_client


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    mock_client = MagicMock()
    # Mock CSV data
    csv_data = "usage_date,daily_cost,service\n2023-01-01,10.50,EC2\n2023-01-02,11.25,EC2"
    mock_client.get_object.return_value = {
        "Body": type("MockBody", (), {"read": lambda: csv_data.encode()})()
    }
    return mock_client


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean up environment variables after each test."""
    yield
    # Clean up any environment variables that might affect tests
    test_env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def sample_athena_results() -> pd.DataFrame:
    """Sample Athena query results for testing."""
    return pd.DataFrame(
        {
            "usage_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "daily_cost": [10.50, 15.75, 12.30],
            "service": ["EC2", "S3", "Lambda"],
            "cost_center": ["engineering", "data", "ml"],
            "project": ["forecaster", "pipeline", "training"],
            "environment": ["dev", "staging", "prod"],
        }
    )
