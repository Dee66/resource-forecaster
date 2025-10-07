"""Simplified workflow orchestration for testing.

Basic implementation of FinOps workflow components for validation.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    success: bool
    step: str
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time: Optional[float] = None


class SimpleWorkflowOrchestrator:
    """Simplified orchestrator for FinOps workflows."""
    
    def __init__(self, config: Any):
        self.config = config
        
    def handle_workflow_event(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Handle Step Functions workflow events."""
        try:
            action = event.get('action', 'unknown')
            logger.info(f"Processing workflow action: {action}")
            
            if action == 'collect_data':
                return self._collect_data_step(event)
            elif action == 'generate_forecast':
                return self._generate_forecast_step(event)
            elif action == 'evaluate_forecast':
                return self._evaluate_forecast_step(event)
            elif action == 'send_alerts':
                return self._send_alerts_step(event)
            else:
                raise ValueError(f"Unknown workflow action: {action}")
                
        except Exception as e:
            logger.error(f"Workflow step failed: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'success': False,
                    'error': str(e),
                    'step': action
                })
            }
    
    def _collect_data_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Collect historical cost and usage data."""
        start_time = datetime.now()
        
        # Simulate data collection
        cost_data = [
            {'date': '2024-01-01', 'cost': 100.0},
            {'date': '2024-01-02', 'cost': 120.0}
        ]
        
        metrics_data = [
            {'metric': 'CPUUtilization', 'value': 75.0}
        ]
        
        result_data = {
            'cost_data': cost_data,
            'metrics_data': metrics_data,
            'date_range': {
                'start': '2024-01-01',
                'end': '2024-01-02'
            }
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'step': 'collect_data',
                'data': result_data,
                'execution_time': execution_time
            })
        }
    
    def _generate_forecast_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost forecasts using collected data."""
        start_time = datetime.now()
        
        # Simulate forecast generation
        forecast_result = {
            'predictions': [100.0, 110.0, 105.0],
            'accuracy_metrics': {'mape': 5.2, 'rmse': 8.1},
            'anomaly_score': 0.3,
            'recommendations': [
                {'type': 'rightsizing', 'description': 'Resize EC2 instance'}
            ]
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'step': 'generate_forecast',
                'data': {
                    'forecast_result': forecast_result
                },
                'execution_time': execution_time
            })
        }
    
    def _evaluate_forecast_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate forecast results and determine if alerts are needed."""
        start_time = datetime.now()
        
        # Extract data from previous step
        body = json.loads(event.get('body', '{}'))
        data = body.get('data', {})
        forecast_result = data.get('forecast_result', {})
        
        anomaly_score = forecast_result.get('anomaly_score', 0.0)
        needs_alerts = anomaly_score > 0.8
        
        alert_data = {
            'needs_alerts': needs_alerts,
            'anomaly_score': anomaly_score,
            'cost_increase_ratio': 0.1,
            'forecast_summary': {
                'total_predicted_cost': 315.0,
                'average_daily_cost': 105.0,
                'recommendations': forecast_result.get('recommendations', [])
            }
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'step': 'evaluate_forecast',
                'data': {
                    'alert_data': alert_data,
                    'forecast_result': forecast_result
                },
                'execution_time': execution_time
            })
        }
    
    def _send_alerts_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Send alerts and notifications based on evaluation results."""
        start_time = datetime.now()
        
        # Extract data from previous step
        body = json.loads(event.get('body', '{}'))
        data = body.get('data', {})
        alert_data = data.get('alert_data', {})
        
        if not alert_data.get('needs_alerts', False):
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'step': 'send_alerts',
                    'data': {'alerts_sent': 0},
                    'execution_time': 0
                })
            }
        
        # Simulate sending alerts
        alerts_sent = [
            {'type': 'sns', 'target': 'test-topic', 'message_id': 'msg-123'}
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'step': 'send_alerts',
                'data': {
                    'alerts_sent': len(alerts_sent),
                    'alert_details': alerts_sent
                },
                'execution_time': execution_time
            })
        }


class SimpleBudgetManager:
    """Simplified budget alert manager for testing."""
    
    def __init__(self, config: Any):
        self.config = config
        
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status and spending trends."""
        return {
            'test-budget-monthly': {
                'limit': 1000.0,
                'actual_spend': 750.0,
                'forecasted_spend': 950.0,
                'actual_percentage': 75.0,
                'forecasted_percentage': 95.0
            }
        }
    
    def get_cost_recommendations(self, days_back: int = 30) -> List[Any]:
        """Get cost optimization recommendations."""
        from types import SimpleNamespace
        
        return [
            SimpleNamespace(
                recommendation_type='rightsizing',
                projected_savings=100.0,
                priority='high',
                resource_id='i-123',
                action_required='Resize instance'
            ),
            SimpleNamespace(
                recommendation_type='reserved_instance',
                projected_savings=200.0,
                priority='medium',
                resource_id='m5.large',
                action_required='Purchase RI'
            )
        ]
    
    def create_cost_explorer_playbook(self) -> Dict[str, Any]:
        """Create automated Cost Explorer analysis playbook."""
        return {
            'total_period_cost': 1000.0,
            'cost_trend': {
                'trend_percentage': 5.0,
                'trend_direction': 'increasing'
            }
        }


class SimpleJobManager:
    """Simplified scheduled job manager for testing."""
    
    def __init__(self, config: Any):
        self.config = config
        self.orchestrator = SimpleWorkflowOrchestrator(config)
        self.budget_manager = SimpleBudgetManager(config)
        
    def _is_low_cost_window(self) -> bool:
        """Check if current time is within low-cost execution window."""
        # For testing, always return True
        return True
    
    def _handle_forecast_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle daily forecast job execution."""
        # Simulate starting workflow
        execution_arn = f"arn:aws:states:us-east-1:123456789012:execution:test:exec-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'execution_arn': execution_arn,
                'job_type': 'forecast',
                'workflow_input': {
                    'forecast_days': parameters.get('forecast_days', 30)
                }
            })
        }
    
    def _handle_budget_review_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle weekly budget review job execution."""
        budget_status = self.budget_manager.get_budget_status()
        recommendations = self.budget_manager.get_cost_recommendations(parameters.get('review_period_days', 7))
        playbook = self.budget_manager.create_cost_explorer_playbook()
        
        total_potential_savings = sum(rec.projected_savings for rec in recommendations)
        
        review_result = {
            'review_period_days': parameters.get('review_period_days', 7),
            'budget_status': budget_status,
            'recommendations_count': len(recommendations),
            'total_potential_savings': total_potential_savings,
            'high_priority_recommendations': len([rec for rec in recommendations if rec.priority == 'high']),
            'cost_explorer_playbook': playbook,
            'review_timestamp': datetime.now().isoformat()
        }
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'job_type': 'budget_review',
                'review_result': review_result
            })
        }
    
    def _handle_anomaly_check_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle real-time anomaly check job execution."""
        budget_status = self.budget_manager.get_budget_status()
        
        anomalies_detected = []
        
        for budget_name, status in budget_status.items():
            if status.get('actual_percentage', 0) > 90:
                anomalies_detected.append({
                    'type': 'budget_threshold',
                    'budget_name': budget_name,
                    'percentage': status['actual_percentage'],
                    'severity': 'high'
                })
            elif status.get('actual_percentage', 0) > 75:
                anomalies_detected.append({
                    'type': 'budget_warning',
                    'budget_name': budget_name,
                    'percentage': status['actual_percentage'],
                    'severity': 'medium'
                })
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'job_type': 'anomaly_check',
                'anomalies_detected': len(anomalies_detected),
                'anomalies': anomalies_detected,
                'check_time': datetime.now().isoformat()
            })
        }