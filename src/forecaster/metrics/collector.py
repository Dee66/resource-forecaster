"""Training metrics capture and CloudWatch integration.

Provides comprehensive metrics collection, storage, and monitoring
for Resource Forecaster model training and validation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from ..config import ForecasterConfig
from ..exceptions import MetricsError

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    model_type: str
    model_version: str
    training_start: datetime
    training_end: datetime
    dataset_size: int
    training_samples: int
    validation_samples: int
    
    # Core metrics
    mse: float
    rmse: float
    mae: float
    mape: float
    r2_score: float
    
    # Prediction windows
    short_term_mape: float  # 7-day forecast
    medium_term_mape: float  # 30-day forecast
    long_term_mape: float   # 90-day forecast
    
    # Training performance
    training_duration_seconds: float
    convergence_epoch: Optional[int] = None
    final_loss: Optional[float] = None
    
    # Hyperparameters
    hyperparameters: Optional[Dict[str, Any]] = None
    
    # Validation metrics
    cross_validation_score: Optional[float] = None
    stability_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable values."""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        if isinstance(data['training_start'], datetime):
            data['training_start'] = data['training_start'].isoformat()
        if isinstance(data['training_end'], datetime):
            data['training_end'] = data['training_end'].isoformat()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """Create instance from dictionary."""
        # Convert ISO strings back to datetime objects
        if isinstance(data['training_start'], str):
            data['training_start'] = datetime.fromisoformat(data['training_start'])
        if isinstance(data['training_end'], str):
            data['training_end'] = datetime.fromisoformat(data['training_end'])
            
        return cls(**data)


@dataclass
class InferenceMetrics:
    """Container for inference/prediction metrics."""
    prediction_id: str
    model_version: str
    prediction_timestamp: datetime
    forecast_horizon_days: int
    input_data_points: int
    
    # Performance metrics
    prediction_latency_ms: float
    memory_usage_mb: float
    
    # Forecast metrics (if ground truth available)
    actual_vs_predicted_mape: Optional[float] = None
    prediction_confidence: Optional[float] = None
    
    # Business metrics
    cost_center: Optional[str] = None
    environment: Optional[str] = None
    forecast_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable values."""
        data = asdict(self)
        
        if isinstance(data['prediction_timestamp'], datetime):
            data['prediction_timestamp'] = data['prediction_timestamp'].isoformat()
            
        return data


class MetricsCollector:
    """Collects and manages training and inference metrics."""
    
    def __init__(self, config: ForecasterConfig):
        """Initialize metrics collector.
        
        Args:
            config: Forecaster configuration
        """
        self.config = config
        self.cloudwatch = None
        self.s3_client = None
        
        # Initialize AWS clients if in AWS environment
        try:
            self.cloudwatch = boto3.client('cloudwatch')
            self.s3_client = boto3.client('s3')
            logger.info("AWS clients initialized for metrics collection")
        except Exception as e:
            logger.warning(f"AWS clients not available: {e}")
    
    def record_training_metrics(
        self, 
        metrics: TrainingMetrics,
        save_local: bool = True,
        send_to_cloudwatch: bool = True,
        save_to_s3: bool = False
    ) -> None:
        """Record training metrics to various destinations.
        
        Args:
            metrics: Training metrics to record
            save_local: Save to local JSON file
            send_to_cloudwatch: Send metrics to CloudWatch
            save_to_s3: Save detailed metrics to S3
            
        Raises:
            MetricsError: If metrics recording fails
        """
        try:
            logger.info(f"Recording training metrics for model {metrics.model_type} v{metrics.model_version}")
            
            # Save locally
            if save_local:
                self._save_training_metrics_local(metrics)
            
            # Send to CloudWatch
            if send_to_cloudwatch and self.cloudwatch:
                self._send_training_metrics_cloudwatch(metrics)
            
            # Save to S3
            if save_to_s3 and self.s3_client:
                self._save_training_metrics_s3(metrics)
                
            logger.info("Training metrics recorded successfully")
            
        except Exception as e:
            raise MetricsError(f"Failed to record training metrics: {e}") from e
    
    def record_inference_metrics(
        self,
        metrics: InferenceMetrics,
        send_to_cloudwatch: bool = True
    ) -> None:
        """Record inference metrics.
        
        Args:
            metrics: Inference metrics to record
            send_to_cloudwatch: Send metrics to CloudWatch
            
        Raises:
            MetricsError: If metrics recording fails
        """
        try:
            logger.debug(f"Recording inference metrics for prediction {metrics.prediction_id}")
            
            # Send to CloudWatch
            if send_to_cloudwatch and self.cloudwatch:
                self._send_inference_metrics_cloudwatch(metrics)
                
        except Exception as e:
            raise MetricsError(f"Failed to record inference metrics: {e}") from e
    
    def get_training_history(
        self,
        model_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TrainingMetrics]:
        """Retrieve training metrics history.
        
        Args:
            model_type: Filter by model type
            start_date: Filter by training start date
            end_date: Filter by training end date
            
        Returns:
            List of training metrics
        """
        try:
            # Load from local storage
            metrics_dir = Path("metrics/training")
            if not metrics_dir.exists():
                return []
            
            training_history = []
            
            for metrics_file in metrics_dir.glob("*.json"):
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    metrics = TrainingMetrics.from_dict(data)
                    
                    # Apply filters
                    if model_type and metrics.model_type != model_type:
                        continue
                    if start_date and metrics.training_start < start_date:
                        continue
                    if end_date and metrics.training_end > end_date:
                        continue
                    
                    training_history.append(metrics)
            
            # Sort by training start time
            training_history.sort(key=lambda x: x.training_start, reverse=True)
            
            return training_history
            
        except Exception as e:
            logger.warning(f"Failed to retrieve training history: {e}")
            return []
    
    def get_model_performance_trends(
        self,
        model_type: str,
        days: int = 30
    ) -> Dict[str, List[float]]:
        """Get model performance trends over time.
        
        Args:
            model_type: Model type to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with metric trends
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        history = self.get_training_history(
            model_type=model_type,
            start_date=start_date,
            end_date=end_date
        )
        
        if not history:
            return {}
        
        # Extract trends
        trends = {
            'dates': [m.training_start.isoformat() for m in history],
            'rmse': [m.rmse for m in history],
            'mape': [m.mape for m in history],
            'r2_score': [m.r2_score for m in history],
            'short_term_mape': [m.short_term_mape for m in history],
            'medium_term_mape': [m.medium_term_mape for m in history],
            'long_term_mape': [m.long_term_mape for m in history],
            'training_duration': [m.training_duration_seconds for m in history]
        }
        
        return trends
    
    def generate_training_report(
        self,
        metrics: TrainingMetrics,
        comparison_metrics: Optional[List[TrainingMetrics]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive training report.
        
        Args:
            metrics: Current training metrics
            comparison_metrics: Previous metrics for comparison
            
        Returns:
            Training report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': metrics.model_type,
                'version': metrics.model_version,
                'training_duration': f"{metrics.training_duration_seconds:.2f}s"
            },
            'performance_metrics': {
                'rmse': metrics.rmse,
                'mape': f"{metrics.mape:.2f}%",
                'r2_score': f"{metrics.r2_score:.3f}",
                'mae': metrics.mae
            },
            'forecast_accuracy': {
                'short_term_7d': f"{metrics.short_term_mape:.2f}%",
                'medium_term_30d': f"{metrics.medium_term_mape:.2f}%",
                'long_term_90d': f"{metrics.long_term_mape:.2f}%"
            },
            'data_info': {
                'dataset_size': metrics.dataset_size,
                'training_samples': metrics.training_samples,
                'validation_samples': metrics.validation_samples
            },
            'quality_assessment': self._assess_model_quality(metrics),
            'recommendations': self._generate_training_recommendations(metrics)
        }
        
        # Add comparison if available
        if comparison_metrics:
            report['performance_comparison'] = self._compare_with_previous(
                metrics, comparison_metrics
            )
        
        return report
    
    def _save_training_metrics_local(self, metrics: TrainingMetrics) -> None:
        """Save training metrics to local JSON file."""
        metrics_dir = Path("metrics/training")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        filename = (
            f"{metrics.model_type}_{metrics.model_version}_"
            f"{metrics.training_start.strftime('%Y%m%d_%H%M%S')}.json"
        )
        filepath = metrics_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Training metrics saved to {filepath}")
    
    def _send_training_metrics_cloudwatch(self, metrics: TrainingMetrics) -> None:
        """Send training metrics to CloudWatch."""
        try:
            namespace = f"ResourceForecaster/{metrics.model_type}"
            
            # Core metrics
            metric_data = [
                {
                    'MetricName': 'RMSE',
                    'Value': metrics.rmse,
                    'Unit': 'None',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version},
                        {'Name': 'ModelType', 'Value': metrics.model_type}
                    ]
                },
                {
                    'MetricName': 'MAPE',
                    'Value': metrics.mape,
                    'Unit': 'Percent',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version},
                        {'Name': 'ModelType', 'Value': metrics.model_type}
                    ]
                },
                {
                    'MetricName': 'R2Score',
                    'Value': metrics.r2_score,
                    'Unit': 'None',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version},
                        {'Name': 'ModelType', 'Value': metrics.model_type}
                    ]
                },
                {
                    'MetricName': 'TrainingDuration',
                    'Value': metrics.training_duration_seconds,
                    'Unit': 'Seconds',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version},
                        {'Name': 'ModelType', 'Value': metrics.model_type}
                    ]
                }
            ]
            
            # Forecast horizon metrics
            for horizon, mape_value in [
                ('7d', metrics.short_term_mape),
                ('30d', metrics.medium_term_mape),
                ('90d', metrics.long_term_mape)
            ]:
                metric_data.append({
                    'MetricName': 'ForecastMAPE',
                    'Value': mape_value,
                    'Unit': 'Percent',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version},
                        {'Name': 'ModelType', 'Value': metrics.model_type},
                        {'Name': 'ForecastHorizon', 'Value': horizon}
                    ]
                })
            
            # Send metrics in batches (CloudWatch limit is 20 per request)
            batch_size = 20
            for i in range(0, len(metric_data), batch_size):
                batch = metric_data[i:i + batch_size]
                
                self.cloudwatch.put_metric_data(
                    Namespace=namespace,
                    MetricData=batch
                )
            
            logger.info(f"Sent {len(metric_data)} metrics to CloudWatch")
            
        except ClientError as e:
            logger.error(f"Failed to send metrics to CloudWatch: {e}")
            raise
    
    def _send_inference_metrics_cloudwatch(self, metrics: InferenceMetrics) -> None:
        """Send inference metrics to CloudWatch."""
        try:
            namespace = "ResourceForecaster/Inference"
            
            metric_data = [
                {
                    'MetricName': 'PredictionLatency',
                    'Value': metrics.prediction_latency_ms,
                    'Unit': 'Milliseconds',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version}
                    ]
                },
                {
                    'MetricName': 'MemoryUsage',
                    'Value': metrics.memory_usage_mb,
                    'Unit': 'Megabytes',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version}
                    ]
                },
                {
                    'MetricName': 'ForecastHorizon',
                    'Value': metrics.forecast_horizon_days,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version}
                    ]
                }
            ]
            
            # Add accuracy metrics if available
            if metrics.actual_vs_predicted_mape is not None:
                metric_data.append({
                    'MetricName': 'PredictionAccuracy',
                    'Value': 100 - metrics.actual_vs_predicted_mape,  # Convert MAPE to accuracy
                    'Unit': 'Percent',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version}
                    ]
                })
            
            # Add business context if available
            if metrics.cost_center:
                metric_data.append({
                    'MetricName': 'PredictionCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'ModelVersion', 'Value': metrics.model_version},
                        {'Name': 'CostCenter', 'Value': metrics.cost_center}
                    ]
                })
            
            self.cloudwatch.put_metric_data(
                Namespace=namespace,
                MetricData=metric_data
            )
            
        except ClientError as e:
            logger.warning(f"Failed to send inference metrics to CloudWatch: {e}")
    
    def _save_training_metrics_s3(self, metrics: TrainingMetrics) -> None:
        """Save detailed training metrics to S3."""
        try:
            bucket = self.config.metrics_s3_bucket
            if not bucket:
                logger.warning("S3 bucket not configured for metrics storage")
                return
            
            # Create S3 key with partitioning
            key = (
                f"training-metrics/"
                f"year={metrics.training_start.year}/"
                f"month={metrics.training_start.month:02d}/"
                f"day={metrics.training_start.day:02d}/"
                f"{metrics.model_type}_{metrics.model_version}_"
                f"{metrics.training_start.strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(metrics.to_dict(), indent=2, default=str),
                ContentType='application/json'
            )
            
            logger.info(f"Training metrics saved to S3: s3://{bucket}/{key}")
            
        except ClientError as e:
            logger.error(f"Failed to save metrics to S3: {e}")
    
    def _assess_model_quality(self, metrics: TrainingMetrics) -> Dict[str, str]:
        """Assess model quality based on metrics."""
        assessment = {}
        
        # RMSE assessment
        if metrics.rmse < 100:
            assessment['rmse'] = 'Excellent'
        elif metrics.rmse < 500:
            assessment['rmse'] = 'Good'
        elif metrics.rmse < 1000:
            assessment['rmse'] = 'Fair'
        else:
            assessment['rmse'] = 'Poor'
        
        # MAPE assessment
        if metrics.mape < 5:
            assessment['mape'] = 'Excellent'
        elif metrics.mape < 10:
            assessment['mape'] = 'Good'
        elif metrics.mape < 20:
            assessment['mape'] = 'Fair'
        else:
            assessment['mape'] = 'Poor'
        
        # R2 assessment
        if metrics.r2_score > 0.9:
            assessment['r2'] = 'Excellent'
        elif metrics.r2_score > 0.7:
            assessment['r2'] = 'Good'
        elif metrics.r2_score > 0.5:
            assessment['r2'] = 'Fair'
        else:
            assessment['r2'] = 'Poor'
        
        # Overall assessment
        scores = {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
        avg_score = sum(scores[v] for v in assessment.values()) / len(assessment)
        
        if avg_score >= 3.5:
            assessment['overall'] = 'Excellent'
        elif avg_score >= 2.5:
            assessment['overall'] = 'Good'
        elif avg_score >= 1.5:
            assessment['overall'] = 'Fair'
        else:
            assessment['overall'] = 'Poor'
        
        return assessment
    
    def _generate_training_recommendations(self, metrics: TrainingMetrics) -> List[str]:
        """Generate recommendations based on training metrics."""
        recommendations = []
        
        # MAPE-based recommendations
        if metrics.mape > 20:
            recommendations.append("High MAPE suggests poor model fit. Consider feature engineering or different model architecture.")
        
        # RMSE-based recommendations
        if metrics.rmse > 1000:
            recommendations.append("High RMSE indicates large prediction errors. Review data quality and outlier handling.")
        
        # R2-based recommendations
        if metrics.r2_score < 0.5:
            recommendations.append("Low R-squared suggests poor explanatory power. Consider additional features or different model.")
        
        # Forecast horizon analysis
        if metrics.long_term_mape > metrics.short_term_mape * 2:
            recommendations.append("Long-term forecast accuracy degrades significantly. Consider ensemble methods or shorter horizons.")
        
        # Training efficiency
        if metrics.training_duration_seconds > 3600:  # 1 hour
            recommendations.append("Long training time detected. Consider model simplification or distributed training.")
        
        # Cross-validation
        if metrics.cross_validation_score and metrics.cross_validation_score < 0.7:
            recommendations.append("Low cross-validation score suggests overfitting. Consider regularization or simpler model.")
        
        return recommendations
    
    def _compare_with_previous(
        self,
        current: TrainingMetrics,
        previous_list: List[TrainingMetrics]
    ) -> Dict[str, Any]:
        """Compare current metrics with previous training runs."""
        if not previous_list:
            return {}
        
        # Get most recent previous metrics
        previous = max(previous_list, key=lambda x: x.training_start)
        
        comparison = {
            'previous_version': previous.model_version,
            'previous_training_date': previous.training_start.isoformat(),
            'improvements': {},
            'regressions': {}
        }
        
        # Compare key metrics
        metrics_to_compare = [
            ('rmse', 'lower_is_better'),
            ('mape', 'lower_is_better'),
            ('r2_score', 'higher_is_better'),
            ('short_term_mape', 'lower_is_better'),
            ('medium_term_mape', 'lower_is_better'),
            ('long_term_mape', 'lower_is_better')
        ]
        
        for metric_name, direction in metrics_to_compare:
            current_value = getattr(current, metric_name)
            previous_value = getattr(previous, metric_name)
            
            if current_value != previous_value:
                change_pct = ((current_value - previous_value) / previous_value) * 100
                
                if direction == 'lower_is_better':
                    if current_value < previous_value:
                        comparison['improvements'][metric_name] = f"{abs(change_pct):.2f}% better"
                    else:
                        comparison['regressions'][metric_name] = f"{change_pct:.2f}% worse"
                else:  # higher_is_better
                    if current_value > previous_value:
                        comparison['improvements'][metric_name] = f"{change_pct:.2f}% better"
                    else:
                        comparison['regressions'][metric_name] = f"{abs(change_pct):.2f}% worse"
        
        return comparison