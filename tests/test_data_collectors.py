"""Unit tests for data collectors."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.forecaster.data.collectors import CloudWatchCollector, CURDataCollector
from src.forecaster.exceptions import DataSourceError


class TestCURDataCollector:
    """Test cases for CUR data collector."""

    def test_init(self, sample_config):
        """Test collector initialization."""
        collector = CURDataCollector(sample_config)
        assert collector.config == sample_config
        assert collector._athena_client is None
        assert collector._s3_client is None

    @patch("boto3.client")
    def test_athena_client_property(self, mock_boto3, sample_config):
        """Test lazy loading of Athena client."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client

        collector = CURDataCollector(sample_config)
        client = collector.athena_client

        assert client == mock_client
        assert collector._athena_client == mock_client
        mock_boto3.assert_called_once_with("athena", region_name="us-east-1")

    @patch("boto3.client")
    def test_s3_client_property(self, mock_boto3, sample_config):
        """Test lazy loading of S3 client."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client

        collector = CURDataCollector(sample_config)
        client = collector.s3_client

        assert client == mock_client
        assert collector._s3_client == mock_client
        mock_boto3.assert_called_once_with("s3", region_name="us-east-1")

    def test_build_daily_cost_query(self, sample_config):
        """Test SQL query building."""
        collector = CURDataCollector(sample_config)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        query = collector._build_daily_cost_query(start_date, end_date)

        assert "2023-01-01" in query
        assert "2023-01-31" in query
        assert "test_database" in query
        assert "line_item_blended_cost" in query
        assert "GROUP BY" in query
        assert "ORDER BY" in query

    def test_build_daily_cost_query_with_filters(self, sample_config):
        """Test SQL query building with filters."""
        collector = CURDataCollector(sample_config)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        cost_center = "engineering"
        project = "forecaster"

        query = collector._build_daily_cost_query(start_date, end_date, cost_center, project)

        assert "engineering" in query
        assert "forecaster" in query
        assert "resource_tags_user_cost_center" in query
        assert "resource_tags_user_project" in query

    def test_process_cost_data(self, sample_config, sample_athena_results):
        """Test cost data processing."""
        collector = CURDataCollector(sample_config)

        processed_df = collector._process_cost_data(sample_athena_results)

        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(processed_df["usage_date"])

        # Check positive costs only
        assert all(processed_df["daily_cost"] > 0)

        # Check sorting
        assert processed_df["usage_date"].is_monotonic_increasing

        # Check filled missing values
        for col in ["cost_center", "project", "environment"]:
            if col in processed_df.columns:
                assert not processed_df[col].isna().any()

    @patch("src.forecaster.data.collectors.CURDataCollector._execute_athena_query")
    @patch("src.forecaster.data.collectors.CURDataCollector._get_query_results")
    @patch("src.forecaster.data.collectors.CURDataCollector._process_cost_data")
    def test_collect_daily_costs_success(
        self, mock_process, mock_get_results, mock_execute, sample_config, sample_cost_data
    ):
        """Test successful cost data collection."""
        # Setup mocks
        mock_execute.return_value = "test-execution-id"
        mock_get_results.return_value = sample_cost_data
        mock_process.return_value = sample_cost_data

        collector = CURDataCollector(sample_config)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        result = collector.collect_daily_costs(start_date, end_date)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Verify method calls
        mock_execute.assert_called_once()
        mock_get_results.assert_called_once_with("test-execution-id")
        mock_process.assert_called_once()

    @patch("src.forecaster.data.collectors.CURDataCollector._execute_athena_query")
    def test_collect_daily_costs_failure(self, mock_execute, sample_config):
        """Test cost data collection failure."""
        mock_execute.side_effect = Exception("Athena error")

        collector = CURDataCollector(sample_config)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)

        with pytest.raises(DataSourceError) as exc_info:
            collector.collect_daily_costs(start_date, end_date)

        assert "Failed to collect CUR data" in str(exc_info.value)
        assert exc_info.value.source_type == "CUR"


class TestCloudWatchCollector:
    """Test cases for CloudWatch collector."""

    def test_init(self, sample_config):
        """Test collector initialization."""
        collector = CloudWatchCollector(sample_config)
        assert collector.config == sample_config
        assert collector._cloudwatch_client is None

    @patch("boto3.client")
    def test_cloudwatch_client_property(self, mock_boto3, sample_config):
        """Test lazy loading of CloudWatch client."""
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client

        collector = CloudWatchCollector(sample_config)
        client = collector.cloudwatch_client

        assert client == mock_client
        assert collector._cloudwatch_client == mock_client
        mock_boto3.assert_called_once_with("cloudwatch", region_name="us-east-1")

    @patch("boto3.client")
    def test_collect_metric_success(self, mock_boto3, sample_config):
        """Test successful metric collection."""
        # Setup mock CloudWatch response
        mock_client = MagicMock()
        mock_response = {
            "Datapoints": [
                {"Timestamp": datetime(2023, 1, 1, 12, 0), "Average": 45.5, "Maximum": 78.2},
                {"Timestamp": datetime(2023, 1, 1, 13, 0), "Average": 52.1, "Maximum": 85.7},
            ]
        }
        mock_client.get_metric_statistics.return_value = mock_response
        mock_boto3.return_value = mock_client

        collector = CloudWatchCollector(sample_config)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)
        namespace = "AWS/EC2"
        metric_name = "CPUUtilization"

        result = collector._collect_metric(namespace, metric_name, start_date, end_date)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "MetricName" in result.columns
        assert "Namespace" in result.columns
        assert "average_value" in result.columns
        assert "maximum_value" in result.columns
        assert all(result["MetricName"] == metric_name)
        assert all(result["Namespace"] == namespace)

    @patch("boto3.client")
    def test_collect_metric_no_data(self, mock_boto3, sample_config):
        """Test metric collection with no data."""
        mock_client = MagicMock()
        mock_client.get_metric_statistics.return_value = {"Datapoints": []}
        mock_boto3.return_value = mock_client

        collector = CloudWatchCollector(sample_config)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)

        result = collector._collect_metric("AWS/EC2", "CPUUtilization", start_date, end_date)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.forecaster.data.collectors.CloudWatchCollector._collect_metric")
    def test_collect_resource_metrics_success(self, mock_collect_metric, sample_config):
        """Test successful resource metrics collection."""
        # Setup mock metric data
        metric_df = pd.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1, 12, 0)],
                "average_value": [45.5],
                "maximum_value": [78.2],
                "MetricName": ["CPUUtilization"],
                "Namespace": ["AWS/EC2"],
            }
        )
        mock_collect_metric.return_value = metric_df

        collector = CloudWatchCollector(sample_config)

        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)

        result = collector.collect_resource_metrics(start_date, end_date)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Should be called for each metric in the default config
        assert mock_collect_metric.call_count == len(sample_config.data_source.cloudwatch_metrics)
