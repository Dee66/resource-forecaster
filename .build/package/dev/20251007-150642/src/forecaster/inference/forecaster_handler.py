"""
Forecaster Handler for Resource Forecasting Service

Provides real-time cost forecasting and recommendation logic for AWS resources.
Handles data preparation, model loading, prediction logic, and recommendation generation.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
import boto3
from botocore.exceptions import ClientError

from ..config import ForecasterConfig
from ..data.collectors import CURDataCollector, CloudWatchCollector
from ..data.processors import CostDataProcessor, FeatureEngineer
from ..data.validators import DataQualityValidator
from ..train.model_factory import ModelFactory
from ..exceptions import (
    ModelLoadingError, 
    DataProcessingError, 
    PredictionError, 
    RecommendationError
)

logger = logging.getLogger(__name__)


class ModelArtifactManager:
    """Manages model artifact storage and retrieval from S3."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.s3_client = boto3.client('s3')
        
    def save_model(self, model: Any, model_name: str, version: str = None) -> str:
        """Save model artifact to S3.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Model version (defaults to timestamp)
            
        Returns:
            S3 URI of saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        key = f"models/{model_name}/{version}/model.pkl"
        
        try:
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.config.storage.model_bucket,
                Key=key,
                Body=model_bytes,
                Metadata={
                    'model_name': model_name,
                    'version': version,
                    'created_at': datetime.now().isoformat(),
                    'config_hash': str(hash(str(self.config.dict())))
                }
            )
            
            s3_uri = f"s3://{self.config.storage.model_bucket}/{key}"
            logger.info(f"Model saved to {s3_uri}")
            return s3_uri
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to save model to S3: {str(e)}")
            
    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load model artifact from S3.
        
        Args:
            model_name: Name of the model
            version: Model version (or 'latest')
            
        Returns:
            Loaded model object
        """
        try:
            if version == "latest":
                # Find the latest version
                prefix = f"models/{model_name}/"
                response = self.s3_client.list_objects_v2(
                    Bucket=self.config.storage.model_bucket,
                    Prefix=prefix,
                    Delimiter='/'
                )
                
                if 'CommonPrefixes' not in response:
                    raise ModelLoadingError(f"No models found for {model_name}")
                    
                versions = [
                    prefix['Prefix'].split('/')[-2] 
                    for prefix in response['CommonPrefixes']
                ]
                version = max(versions)  # Latest version lexicographically
                
            key = f"models/{model_name}/{version}/model.pkl"
            
            # Download from S3
            response = self.s3_client.get_object(
                Bucket=self.config.storage.model_bucket,
                Key=key
            )
            
            # Deserialize model
            model = pickle.loads(response['Body'].read())
            
            logger.info(f"Model loaded from s3://{self.config.storage.model_bucket}/{key}")
            return model
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise ModelLoadingError(f"Model not found: {model_name} version {version}")
            else:
                raise ModelLoadingError(f"Failed to load model from S3: {str(e)}")
        except Exception as e:
            raise ModelLoadingError(f"Failed to deserialize model: {str(e)}")


class RecommendationEngine:
    """Generates cost optimization recommendations based on forecasts."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.ce_client = boto3.client('ce')  # Cost Explorer
        self.pricing_client = boto3.client('pricing', region_name='us-east-1')
        
    def generate_recommendations(
        self, 
        forecast_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        forecast_horizon_days: int = 30
    ) -> Dict[str, Any]:
        """Generate cost optimization recommendations.
        
        Args:
            forecast_data: Forecasted cost data
            historical_data: Historical cost data for baseline
            forecast_horizon_days: Forecast horizon in days
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            recommendations = {
                'rightsizing': self._generate_rightsizing_recommendations(forecast_data, historical_data),
                'savings_plans': self._generate_savings_plan_recommendations(forecast_data, historical_data),
                'reserved_instances': self._generate_ri_recommendations(forecast_data, historical_data),
                'scheduling': self._generate_scheduling_recommendations(forecast_data, historical_data),
                'cost_anomalies': self._detect_cost_anomalies(forecast_data, historical_data),
                'summary': self._generate_summary(forecast_data, historical_data, forecast_horizon_days)
            }
            
            return recommendations
            
        except Exception as e:
            raise RecommendationError(f"Failed to generate recommendations: {str(e)}")
            
    def _generate_rightsizing_recommendations(
        self, 
        forecast_data: pd.DataFrame, 
        historical_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate rightsizing recommendations."""
        recommendations = []
        
        try:
            # Get rightsizing recommendations from Cost Explorer
            response = self.ce_client.get_rightsizing_recommendation(
                Configuration={
                    'BenefitsConsidered': True,
                    'RecommendationType': 'TERMINATE'
                }
            )
            
            # Process AWS recommendations
            for rec in response.get('RightsizingRecommendations', []):
                if rec.get('RightsizingType') == 'Terminate':
                    recommendations.append({
                        'type': 'terminate',
                        'resource_id': rec.get('CurrentInstance', {}).get('ResourceId'),
                        'instance_type': rec.get('CurrentInstance', {}).get('InstanceType'),
                        'monthly_savings': rec.get('TerminateRecommendationDetail', {}).get('EstimatedMonthlySavings'),
                        'confidence': 'high',
                        'reason': 'Low utilization detected'
                    })
                elif rec.get('RightsizingType') == 'Modify':
                    recommendations.append({
                        'type': 'resize',
                        'resource_id': rec.get('CurrentInstance', {}).get('ResourceId'),
                        'current_type': rec.get('CurrentInstance', {}).get('InstanceType'),
                        'recommended_type': rec.get('ModifyRecommendationDetail', {}).get('TargetInstances', [{}])[0].get('InstanceType'),
                        'monthly_savings': rec.get('ModifyRecommendationDetail', {}).get('EstimatedMonthlySavings'),
                        'confidence': 'high',
                        'reason': 'Over-provisioned resources'
                    })
                    
        except ClientError as e:
            logger.warning(f"Failed to get rightsizing recommendations: {str(e)}")
            
        # Add custom rightsizing logic based on forecast patterns
        overprovisioned = self._identify_overprovisioned_resources(forecast_data, historical_data)
        recommendations.extend(overprovisioned)
        
        return recommendations
        
    def _generate_savings_plan_recommendations(
        self, 
        forecast_data: pd.DataFrame, 
        historical_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate Savings Plan recommendations."""
        recommendations = []
        
        try:
            # Get Savings Plans recommendations from Cost Explorer
            response = self.ce_client.get_savings_plans_purchase_recommendation(
                SavingsPlansType='EC2_INSTANCE',
                TermInYears='ONE_YEAR',
                PaymentOption='NO_UPFRONT',
                LookbackPeriodInDays='THIRTY_DAYS'
            )
            
            for rec in response.get('SavingsPlansRecommendations', []):
                recommendations.append({
                    'type': 'savings_plan',
                    'plan_type': rec.get('SavingsPlansType'),
                    'term': rec.get('TermInYears'),
                    'payment_option': rec.get('PaymentOption'),
                    'hourly_commitment': rec.get('HourlyCommitment'),
                    'estimated_savings': rec.get('EstimatedSavings'),
                    'confidence': 'medium',
                    'reason': 'Consistent usage pattern identified'
                })
                
        except ClientError as e:
            logger.warning(f"Failed to get Savings Plans recommendations: {str(e)}")
            
        return recommendations
        
    def _generate_ri_recommendations(
        self, 
        forecast_data: pd.DataFrame, 
        historical_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate Reserved Instance recommendations."""
        recommendations = []
        
        try:
            # Get RI recommendations from Cost Explorer
            response = self.ce_client.get_reservation_purchase_recommendation(
                Service='Amazon Elastic Compute Cloud - Compute',
                TermInYears='ONE_YEAR',
                PaymentOption='NO_UPFRONT',
                LookbackPeriodInDays='THIRTY_DAYS'
            )
            
            for rec in response.get('Recommendations', []):
                recommendations.append({
                    'type': 'reserved_instance',
                    'instance_type': rec.get('InstanceDetails', {}).get('EC2InstanceDetails', {}).get('InstanceType'),
                    'platform': rec.get('InstanceDetails', {}).get('EC2InstanceDetails', {}).get('Platform'),
                    'region': rec.get('InstanceDetails', {}).get('EC2InstanceDetails', {}).get('Region'),
                    'quantity': rec.get('RecommendationDetails', {}).get('InstanceQuantity'),
                    'estimated_savings': rec.get('RecommendationDetails', {}).get('EstimatedMonthlySavingsAmount'),
                    'confidence': 'medium',
                    'reason': 'Steady-state workload detected'
                })
                
        except ClientError as e:
            logger.warning(f"Failed to get RI recommendations: {str(e)}")
            
        return recommendations
        
    def _generate_scheduling_recommendations(
        self, 
        forecast_data: pd.DataFrame, 
        historical_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate resource scheduling recommendations."""
        recommendations = []
        
        # Analyze usage patterns for scheduling opportunities
        if 'hour' in historical_data.columns:
            hourly_usage = historical_data.groupby('hour')['cost'].mean()
            
            # Identify low-usage hours
            low_usage_threshold = hourly_usage.mean() * 0.3
            low_usage_hours = hourly_usage[hourly_usage < low_usage_threshold].index.tolist()
            
            if low_usage_hours:
                recommendations.append({
                    'type': 'scheduling',
                    'action': 'stop_non_production',
                    'hours': low_usage_hours,
                    'estimated_savings_percent': 20,
                    'confidence': 'medium',
                    'reason': f'Low usage detected during hours: {low_usage_hours}'
                })
                
        # Analyze day-of-week patterns
        if 'day_of_week' in historical_data.columns:
            daily_usage = historical_data.groupby('day_of_week')['cost'].mean()
            
            # Weekend optimization
            weekend_avg = daily_usage[[5, 6]].mean()  # Saturday, Sunday
            weekday_avg = daily_usage[[0, 1, 2, 3, 4]].mean()  # Monday-Friday
            
            if weekend_avg < weekday_avg * 0.5:
                recommendations.append({
                    'type': 'scheduling',
                    'action': 'weekend_shutdown',
                    'estimated_savings_percent': 15,
                    'confidence': 'high',
                    'reason': 'Low weekend usage detected'
                })
                
        return recommendations
        
    def _detect_cost_anomalies(
        self, 
        forecast_data: pd.DataFrame, 
        historical_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Detect cost anomalies and unusual patterns."""
        anomalies = []
        
        try:
            # Statistical anomaly detection
            recent_costs = historical_data['cost'].tail(30)  # Last 30 data points
            mean_cost = recent_costs.mean()
            std_cost = recent_costs.std()
            
            # Z-score based anomaly detection
            z_scores = np.abs((recent_costs - mean_cost) / std_cost)
            anomaly_threshold = 2.5
            
            anomalous_indices = recent_costs[z_scores > anomaly_threshold].index
            
            for idx in anomalous_indices:
                anomalies.append({
                    'type': 'cost_spike',
                    'date': historical_data.loc[idx, 'ds'].isoformat() if 'ds' in historical_data.columns else str(idx),
                    'cost': historical_data.loc[idx, 'cost'],
                    'z_score': z_scores.loc[idx],
                    'severity': 'high' if z_scores.loc[idx] > 3 else 'medium',
                    'reason': 'Unusual cost spike detected'
                })
                
            # Forecast vs historical comparison
            if not forecast_data.empty:
                forecast_mean = forecast_data['yhat'].mean()
                historical_mean = historical_data['cost'].tail(30).mean()
                
                variance_percent = ((forecast_mean - historical_mean) / historical_mean) * 100
                
                if abs(variance_percent) > 20:
                    anomalies.append({
                        'type': 'forecast_deviation',
                        'variance_percent': variance_percent,
                        'forecast_mean': forecast_mean,
                        'historical_mean': historical_mean,
                        'severity': 'high' if abs(variance_percent) > 50 else 'medium',
                        'reason': f'Forecast deviates {variance_percent:.1f}% from historical average'
                    })
                    
        except Exception as e:
            logger.warning(f"Failed to detect cost anomalies: {str(e)}")
            
        return anomalies
        
    def _identify_overprovisioned_resources(
        self, 
        forecast_data: pd.DataFrame, 
        historical_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Identify overprovisioned resources based on usage patterns."""
        recommendations = []
        
        # This would typically integrate with CloudWatch metrics
        # For now, we'll provide a simple pattern-based analysis
        
        if 'service' in historical_data.columns:
            service_costs = historical_data.groupby('service')['cost'].agg(['mean', 'std']).reset_index()
            
            # Identify services with consistently low cost variance (potential over-provisioning)
            low_variance_services = service_costs[
                (service_costs['std'] / service_costs['mean']) < 0.1
            ]
            
            for _, service in low_variance_services.iterrows():
                recommendations.append({
                    'type': 'rightsizing',
                    'service': service['service'],
                    'current_avg_cost': service['mean'],
                    'potential_savings_percent': 25,
                    'confidence': 'medium',
                    'reason': 'Consistent low variance suggests over-provisioning'
                })
                
        return recommendations
        
    def _generate_summary(
        self, 
        forecast_data: pd.DataFrame, 
        historical_data: pd.DataFrame,
        forecast_horizon_days: int
    ) -> Dict[str, Any]:
        """Generate recommendation summary."""
        
        total_historical_cost = historical_data['cost'].sum()
        total_forecast_cost = forecast_data['yhat'].sum() if not forecast_data.empty else 0
        
        forecast_vs_historical = ((total_forecast_cost - total_historical_cost) / total_historical_cost * 100) if total_historical_cost > 0 else 0
        
        return {
            'forecast_horizon_days': forecast_horizon_days,
            'historical_total_cost': total_historical_cost,
            'forecast_total_cost': total_forecast_cost,
            'forecast_vs_historical_percent': forecast_vs_historical,
            'potential_savings_identified': True,
            'confidence_level': 'medium',
            'last_updated': datetime.now().isoformat()
        }


class ForecasterHandler:
    """Main handler for cost forecasting requests."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.data_collector = CURDataCollector(config)
        self.cloudwatch_collector = CloudWatchCollector(config)
        self.data_processor = CostDataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.data_validator = DataQualityValidator(config)
        self.model_manager = ModelArtifactManager(config)
        self.recommendation_engine = RecommendationEngine(config)
        self.model_factory = ModelFactory(config)
        
        # Cache for loaded models
        self._model_cache = {}
        
    def predict(
        self, 
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle prediction request.
        
        Args:
            request: Prediction request containing parameters
            
        Returns:
            Prediction response with forecast and recommendations
        """
        try:
            # Extract request parameters
            forecast_horizon_days = request.get('forecast_horizon_days', 30)
            model_name = request.get('model_name', 'prophet')
            include_recommendations = request.get('include_recommendations', True)
            account_id = request.get('account_id')
            service_filter = request.get('service_filter')
            
            logger.info(f"Processing prediction request for {forecast_horizon_days} days")
            
            # Collect and prepare data
            data = self._prepare_data(
                account_id=account_id,
                service_filter=service_filter,
                lookback_days=90  # Use 90 days of historical data
            )
            
            # Generate forecast
            forecast = self._generate_forecast(
                data=data,
                model_name=model_name,
                forecast_horizon_days=forecast_horizon_days
            )
            
            # Generate recommendations if requested
            recommendations = None
            if include_recommendations:
                recommendations = self.recommendation_engine.generate_recommendations(
                    forecast_data=forecast,
                    historical_data=data,
                    forecast_horizon_days=forecast_horizon_days
                )
                
            # Build response
            response = {
                'request_id': request.get('request_id', str(datetime.now().timestamp())),
                'forecast': {
                    'model_used': model_name,
                    'forecast_horizon_days': forecast_horizon_days,
                    'predictions': forecast.to_dict('records') if not forecast.empty else [],
                    'generated_at': datetime.now().isoformat()
                },
                'recommendations': recommendations,
                'metadata': {
                    'data_points_used': len(data),
                    'account_id': account_id,
                    'service_filter': service_filter,
                    'confidence_interval': '80%'  # Default Prophet confidence interval
                }
            }
            
            logger.info(f"Prediction request completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Prediction request failed: {str(e)}")
            raise PredictionError(f"Failed to process prediction request: {str(e)}")
            
    def _prepare_data(
        self, 
        account_id: Optional[str] = None,
        service_filter: Optional[List[str]] = None,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """Prepare data for forecasting.
        
        Args:
            account_id: AWS account ID filter
            service_filter: List of services to include
            lookback_days: Number of days of historical data to collect
            
        Returns:
            Prepared DataFrame ready for modeling
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Collect cost data
            logger.info(f"Collecting cost data from {start_date} to {end_date}")
            raw_data = self.data_collector.collect_cost_data(
                start_date=start_date,
                end_date=end_date,
                account_filter=account_id,
                service_filter=service_filter
            )
            
            if raw_data.empty:
                raise DataProcessingError("No cost data collected")
                
            # Process data
            logger.info("Processing cost data")
            processed_data = self.data_processor.process_cost_data(raw_data)
            
            # Engineer features
            logger.info("Engineering features")
            features_data = self.feature_engineer.engineer_features(processed_data)
            
            # Validate data quality
            logger.info("Validating data quality")
            validation_results = self.data_validator.validate_cost_data(features_data)
            
            if not validation_results['is_valid']:
                logger.warning(f"Data quality issues detected: {validation_results['issues']}")
                # Continue with processing but log warnings
                
            logger.info(f"Data preparation completed. {len(features_data)} records processed")
            return features_data
            
        except Exception as e:
            raise DataProcessingError(f"Failed to prepare data: {str(e)}")
            
    def _generate_forecast(
        self, 
        data: pd.DataFrame,
        model_name: str,
        forecast_horizon_days: int
    ) -> pd.DataFrame:
        """Generate forecast using specified model.
        
        Args:
            data: Historical data
            model_name: Name of the model to use
            forecast_horizon_days: Number of days to forecast
            
        Returns:
            DataFrame containing forecast results
        """
        try:
            # Load or create model
            model = self._get_model(model_name)
            
            # Generate forecast
            logger.info(f"Generating forecast for {forecast_horizon_days} days using {model_name}")
            forecast = model.predict(data, forecast_horizon_days)
            
            logger.info(f"Forecast generated successfully")
            return forecast
            
        except Exception as e:
            raise PredictionError(f"Failed to generate forecast: {str(e)}")
            
    def _get_model(self, model_name: str):
        """Get model from cache or load from storage.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded model object
        """
        if model_name in self._model_cache:
            return self._model_cache[model_name]
            
        try:
            # Try to load from S3
            model = self.model_manager.load_model(model_name)
            self._model_cache[model_name] = model
            return model
            
        except ModelLoadingError:
            # Fallback: create new model with default parameters
            logger.warning(f"Could not load {model_name} from storage, creating new model")
            model = self.model_factory.create_model(model_name)
            return model
            
    def batch_predict(
        self, 
        requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Handle batch prediction requests.
        
        Args:
            requests: List of prediction requests
            
        Returns:
            List of prediction responses
        """
        logger.info(f"Processing batch prediction with {len(requests)} requests")
        
        responses = []
        for i, request in enumerate(requests):
            try:
                request['request_id'] = f"batch_{i}_{datetime.now().timestamp()}"
                response = self.predict(request)
                responses.append(response)
            except Exception as e:
                logger.error(f"Batch request {i} failed: {str(e)}")
                responses.append({
                    'request_id': request.get('request_id', f"batch_{i}"),
                    'error': str(e),
                    'status': 'failed'
                })
                
        logger.info(f"Batch prediction completed. {len([r for r in responses if 'error' not in r])} successful")
        return responses
        
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the forecasting service.
        
        Returns:
            Health status and metrics
        """
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': self.config.version,
                'checks': {}
            }
            
            # Check data sources
            try:
                # Test CUR data access
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)
                test_data = self.data_collector.collect_cost_data(start_date, end_date)
                health_status['checks']['data_access'] = 'ok'
            except Exception as e:
                health_status['checks']['data_access'] = f'error: {str(e)}'
                health_status['status'] = 'degraded'
                
            # Check model availability
            try:
                available_models = list(self._model_cache.keys())
                health_status['checks']['models_loaded'] = len(available_models)
                health_status['checks']['available_models'] = available_models
            except Exception as e:
                health_status['checks']['models'] = f'error: {str(e)}'
                health_status['status'] = 'degraded'
                
            # Check AWS services
            try:
                # Test Cost Explorer access
                self.recommendation_engine.ce_client.get_dimension_values(
                    TimePeriod={
                        'Start': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                        'End': datetime.now().strftime('%Y-%m-%d')
                    },
                    Dimension='SERVICE'
                )
                health_status['checks']['cost_explorer'] = 'ok'
            except Exception as e:
                health_status['checks']['cost_explorer'] = f'error: {str(e)}'
                health_status['status'] = 'degraded'
                
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }