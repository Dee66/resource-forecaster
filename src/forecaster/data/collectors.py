"""Data collectors for Resource Forecaster.

Provides collection of historical cost data from AWS sources
including Cost and Usage Reports (CUR) and CloudWatch metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

from ..config import ForecasterConfig
from ..exceptions import DataSourceError

logger = logging.getLogger(__name__)


class CURDataCollector:
    """Collects cost data from AWS Cost and Usage Reports via Athena."""
    
    def __init__(self, config: ForecasterConfig):
        """Initialize CUR data collector.
        
        Args:
            config: Forecaster configuration
        """
        self.config = config
        self._athena_client = None
        self._s3_client = None
    
    @property
    def athena_client(self):
        """Lazy-loaded Athena client."""
        if self._athena_client is None:
            self._athena_client = boto3.client(
                'athena',
                region_name=self.config.aws_region
            )
        return self._athena_client
    
    @property
    def s3_client(self):
        """Lazy-loaded S3 client."""
        if self._s3_client is None:
            self._s3_client = boto3.client(
                's3',
                region_name=self.config.aws_region
            )
        return self._s3_client
    
    def collect_daily_costs(
        self, 
        start_date: datetime, 
        end_date: datetime,
        cost_center: Optional[str] = None,
        project: Optional[str] = None
    ) -> pd.DataFrame:
        """Collect daily cost data from CUR.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            cost_center: Optional cost center filter
            project: Optional project filter
            
        Returns:
            DataFrame with daily cost data
            
        Raises:
            DataSourceError: If data collection fails
        """
        try:
            logger.info(
                f"Collecting CUR data from {start_date.date()} to {end_date.date()}"
            )
            
            # Build Athena query
            query = self._build_daily_cost_query(
                start_date, end_date, cost_center, project
            )
            
            # Execute query
            execution_id = self._execute_athena_query(query)
            
            # Get results
            results_df = self._get_query_results(execution_id)
            
            # Validate and process results
            processed_df = self._process_cost_data(results_df)
            
            logger.info(f"Collected {len(processed_df)} daily cost records")
            return processed_df
            
        except Exception as e:
            raise DataSourceError(
                f"Failed to collect CUR data: {str(e)}",
                source_type="CUR",
                details={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "cost_center": cost_center,
                    "project": project
                }
            ) from e
    
    def _build_daily_cost_query(
        self,
        start_date: datetime,
        end_date: datetime,
        cost_center: Optional[str] = None,
        project: Optional[str] = None
    ) -> str:
        """Build Athena SQL query for daily cost data."""
        
        base_query = f"""
        SELECT 
            line_item_usage_start_date AS usage_date,
            SUM(line_item_blended_cost) AS daily_cost,
            product_product_name AS service,
            resource_tags_user_cost_center AS cost_center,
            resource_tags_user_project AS project,
            resource_tags_user_environment AS environment,
            line_item_resource_id AS resource_id,
            line_item_usage_type AS usage_type
        FROM {self.config.data_source.athena_database}.{self.config.data_source.cur_prefix}cost_usage_reports
        WHERE 
            line_item_usage_start_date >= '{start_date.strftime('%Y-%m-%d')}'
            AND line_item_usage_start_date <= '{end_date.strftime('%Y-%m-%d')}'
            AND line_item_blended_cost > 0
        """
        
        # Add filters if specified
        filters = []
        if cost_center:
            filters.append(f"resource_tags_user_cost_center = '{cost_center}'")
        if project:
            filters.append(f"resource_tags_user_project = '{project}'")
        
        if filters:
            base_query += "AND " + " AND ".join(filters)
        
        base_query += """
        GROUP BY 
            line_item_usage_start_date,
            product_product_name,
            resource_tags_user_cost_center,
            resource_tags_user_project,
            resource_tags_user_environment,
            line_item_resource_id,
            line_item_usage_type
        ORDER BY line_item_usage_start_date DESC
        """
        
        return base_query
    
    def _execute_athena_query(self, query: str) -> str:
        """Execute Athena query and return execution ID."""
        try:
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': self.config.data_source.athena_database
                },
                ResultConfiguration={
                    'OutputLocation': f"s3://{self.config.data_source.athena_output_bucket}/forecaster-queries/"
                },
                WorkGroup=self.config.data_source.athena_workgroup
            )
            
            execution_id = response['QueryExecutionId']
            logger.info(f"Started Athena query execution: {execution_id}")
            
            # Wait for completion
            self._wait_for_query_completion(execution_id)
            
            return execution_id
            
        except ClientError as e:
            raise DataSourceError(
                f"Athena query execution failed: {e}",
                source_type="Athena"
            ) from e
    
    def _wait_for_query_completion(self, execution_id: str, timeout: int = 300) -> None:
        """Wait for Athena query to complete."""
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.athena_client.get_query_execution(
                QueryExecutionId=execution_id
            )
            
            status = response['QueryExecution']['Status']['State']
            
            if status == 'SUCCEEDED':
                logger.info(f"Athena query {execution_id} completed successfully")
                return
            elif status in ['FAILED', 'CANCELLED']:
                reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
                raise DataSourceError(
                    f"Athena query failed: {reason}",
                    source_type="Athena",
                    details={'execution_id': execution_id, 'status': status}
                )
            
            time.sleep(5)  # Poll every 5 seconds
        
        raise DataSourceError(
            f"Athena query timeout after {timeout} seconds",
            source_type="Athena",
            details={'execution_id': execution_id}
        )
    
    def _get_query_results(self, execution_id: str) -> pd.DataFrame:
        """Get results from completed Athena query."""
        try:
            # Get result location
            response = self.athena_client.get_query_execution(
                QueryExecutionId=execution_id
            )
            
            result_location = response['QueryExecution']['ResultConfiguration']['OutputLocation']
            
            # Parse S3 location
            bucket = result_location.split('/')[2]
            key = '/'.join(result_location.split('/')[3:])
            
            # Download results
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            # Read CSV results
            df = pd.read_csv(obj['Body'])
            
            logger.info(f"Retrieved {len(df)} rows from Athena query")
            return df
            
        except Exception as e:
            raise DataSourceError(
                f"Failed to retrieve Athena results: {e}",
                source_type="Athena",
                details={'execution_id': execution_id}
            ) from e
    
    def _process_cost_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate cost data."""
        # Convert date column
        df['usage_date'] = pd.to_datetime(df['usage_date'])
        
        # Ensure positive costs
        filtered_df = df[df['daily_cost'] > 0].copy()
        
        # Fill missing tags
        tag_columns = ['cost_center', 'project', 'environment']
        for col in tag_columns:
            if col in filtered_df.columns:
                filtered_df[col] = filtered_df[col].fillna('untagged')
        
        # Sort by date
        filtered_df = filtered_df.sort_values('usage_date')
        
        return filtered_df


class CloudWatchCollector:
    """Collects utilization metrics from CloudWatch."""
    
    def __init__(self, config: ForecasterConfig):
        """Initialize CloudWatch collector.
        
        Args:
            config: Forecaster configuration
        """
        self.config = config
        self._cloudwatch_client = None
    
    @property
    def cloudwatch_client(self):
        """Lazy-loaded CloudWatch client."""
        if self._cloudwatch_client is None:
            self._cloudwatch_client = boto3.client(
                'cloudwatch',
                region_name=self.config.aws_region
            )
        return self._cloudwatch_client
    
    def collect_resource_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        namespace: Optional[str] = None,
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Collect resource utilization metrics.
        
        Args:
            start_date: Start date for metrics
            end_date: End date for metrics
            namespace: CloudWatch namespace (defaults to config)
            metric_names: List of metrics to collect (defaults to config)
            
        Returns:
            DataFrame with resource metrics
            
        Raises:
            DataSourceError: If metrics collection fails
        """
        try:
            namespace = namespace or self.config.data_source.cloudwatch_namespace
            metric_names = metric_names or self.config.data_source.cloudwatch_metrics
            
            logger.info(
                f"Collecting CloudWatch metrics from {start_date.date()} to {end_date.date()}"
            )
            
            all_metrics = []
            
            for metric_name in metric_names:
                metric_data = self._collect_metric(
                    namespace, metric_name, start_date, end_date
                )
                all_metrics.append(metric_data)
            
            # Combine all metrics
            if all_metrics:
                combined_df = pd.concat(all_metrics, ignore_index=True)
                logger.info(f"Collected {len(combined_df)} metric records")
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            raise DataSourceError(
                f"Failed to collect CloudWatch metrics: {e}",
                source_type="CloudWatch",
                details={
                    "namespace": namespace,
                    "metrics": metric_names,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                }
            ) from e
    
    def _collect_metric(
        self,
        namespace: str,
        metric_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect a specific metric."""
        try:
            # Get metric statistics
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                StartTime=start_date,
                EndTime=end_date,
                Period=3600,  # 1 hour periods
                Statistics=['Average', 'Maximum']
            )
            
            # Convert to DataFrame
            datapoints = response.get('Datapoints', [])
            if not datapoints:
                logger.warning(f"No data found for metric {namespace}/{metric_name}")
                return pd.DataFrame()
            
            df = pd.DataFrame(datapoints)
            df['MetricName'] = metric_name
            df['Namespace'] = namespace
            
            # Rename columns
            df = df.rename(columns={
                'Timestamp': 'timestamp',
                'Average': 'average_value',
                'Maximum': 'maximum_value'
            })
            
            return df
            
        except ClientError as e:
            logger.error(f"Failed to collect metric {namespace}/{metric_name}: {e}")
            return pd.DataFrame()