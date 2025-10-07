"""Step Functions orchestration handler for FinOps workflows.

Coordinates daily forecasting, evaluation, and alert routing workflows
for automated cost optimization and budget management.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError

from ..data.collectors import CURCollector, CloudWatchCollector
from ..inference.forecaster_handler import ForecasterHandler
from ..metrics.cloudwatch_metrics import CloudWatchMetrics
from ..config import ForecasterConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    success: bool
    step: str
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time: Optional[float] = None


class StepFunctionsOrchestrator:
    """Orchestrates FinOps workflows using AWS Step Functions."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.stepfunctions = boto3.client('stepfunctions')
        self.budgets = boto3.client('budgets')
        self.sns = boto3.client('sns')
        
        # Initialize components
        self.cost_collector = CostCollector(config)
        self.cloudwatch_collector = CloudWatchCollector(config)
        self.forecaster = ForecasterHandler(config)
        self.metrics = CloudWatchMetrics(config)
        
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
            elif action == 'update_budgets':
                return self._update_budgets_step(event)
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
        
        try:
            # Get date range from event or use defaults
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)
            
            if 'date_range' in event:
                start_date = datetime.fromisoformat(event['date_range']['start']).date()
                end_date = datetime.fromisoformat(event['date_range']['end']).date()
            
            # Collect cost data
            logger.info(f"Collecting cost data from {start_date} to {end_date}")
            cost_data = self.cost_collector.collect_cost_data(start_date, end_date)
            
            # Collect CloudWatch metrics
            logger.info("Collecting CloudWatch metrics")
            metrics_data = self.cloudwatch_collector.collect_metrics(start_date, end_date)
            
            # Store data for next step
            result_data = {
                'cost_data': cost_data,
                'metrics_data': metrics_data,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics.record_workflow_step_duration('collect_data', execution_time)
            self.metrics.record_data_collection_volume(len(cost_data))
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'step': 'collect_data',
                    'data': result_data,
                    'execution_time': execution_time
                })
            }
            
        except Exception as e:
            logger.error(f"Data collection failed: {str(e)}")
            self.metrics.record_workflow_step_error('collect_data', str(e))
            raise
    
    def _generate_forecast_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost forecasts using collected data."""
        start_time = datetime.now()
        
        try:
            # Extract data from previous step
            body = json.loads(event.get('body', '{}'))
            input_data = body.get('data', {})
            
            if not input_data:
                raise ValueError("No input data from previous step")
            
            # Generate forecasts
            logger.info("Generating cost forecasts")
            forecast_request = {
                'account_id': 'current',
                'forecast_days': 30,
                'include_recommendations': True,
                'cost_data': input_data.get('cost_data'),
                'metrics_data': input_data.get('metrics_data')
            }
            
            forecast_result = self.forecaster.generate_forecast(forecast_request)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics.record_workflow_step_duration('generate_forecast', execution_time)
            self.metrics.record_forecast_accuracy(forecast_result.get('accuracy_metrics', {}))
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'step': 'generate_forecast',
                    'data': {
                        'forecast_result': forecast_result,
                        'input_data': input_data
                    },
                    'execution_time': execution_time
                })
            }
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {str(e)}")
            self.metrics.record_workflow_step_error('generate_forecast', str(e))
            raise
    
    def _evaluate_forecast_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate forecast results and determine if alerts are needed."""
        start_time = datetime.now()
        
        try:
            # Extract data from previous step
            body = json.loads(event.get('body', '{}'))
            data = body.get('data', {})
            forecast_result = data.get('forecast_result', {})
            
            # Evaluation criteria
            anomaly_threshold = 0.8
            cost_increase_threshold = 0.2  # 20% increase
            
            anomaly_score = forecast_result.get('anomaly_score', 0.0)
            predicted_costs = forecast_result.get('predictions', [])
            
            # Calculate cost trend
            if predicted_costs and len(predicted_costs) >= 7:
                recent_avg = sum(predicted_costs[-7:]) / 7
                historical_avg = sum(predicted_costs[:7]) / 7 if len(predicted_costs) >= 14 else recent_avg
                cost_increase_ratio = (recent_avg - historical_avg) / historical_avg if historical_avg > 0 else 0
            else:
                cost_increase_ratio = 0
            
            # Determine if alerts are needed
            needs_alerts = (
                anomaly_score > anomaly_threshold or 
                cost_increase_ratio > cost_increase_threshold
            )
            
            # Prepare alert data
            alert_data = {
                'needs_alerts': needs_alerts,
                'anomaly_score': anomaly_score,
                'cost_increase_ratio': cost_increase_ratio,
                'forecast_summary': {
                    'total_predicted_cost': sum(predicted_costs) if predicted_costs else 0,
                    'average_daily_cost': sum(predicted_costs) / len(predicted_costs) if predicted_costs else 0,
                    'recommendations': forecast_result.get('recommendations', [])
                }
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics.record_workflow_step_duration('evaluate_forecast', execution_time)
            if needs_alerts:
                self.metrics.record_alert_triggered('cost_anomaly', anomaly_score)
            
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
            
        except Exception as e:
            logger.error(f"Forecast evaluation failed: {str(e)}")
            self.metrics.record_workflow_step_error('evaluate_forecast', str(e))
            raise
    
    def _send_alerts_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Send alerts and notifications based on evaluation results."""
        start_time = datetime.now()
        
        try:
            # Extract data from previous step
            body = json.loads(event.get('body', '{}'))
            data = body.get('data', {})
            alert_data = data.get('alert_data', {})
            
            if not alert_data.get('needs_alerts', False):
                logger.info("No alerts needed, skipping notification step")
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'success': True,
                        'step': 'send_alerts',
                        'data': {'alerts_sent': 0},
                        'execution_time': 0
                    })
                }
            
            alerts_sent = []
            
            # Send SNS notifications
            if hasattr(self.config, 'alerts_topic_arn') and self.config.alerts_topic_arn:
                message = self._create_alert_message(alert_data)
                
                response = self.sns.publish(
                    TopicArn=self.config.alerts_topic_arn,
                    Subject=f"Cost Forecast Alert - {datetime.now().strftime('%Y-%m-%d')}",
                    Message=message
                )
                
                alerts_sent.append({
                    'type': 'sns',
                    'target': self.config.alerts_topic_arn,
                    'message_id': response['MessageId']
                })
            
            # Send to CloudWatch for dashboard alerts
            if alert_data.get('anomaly_score', 0) > 0.8:
                self.metrics.record_custom_metric(
                    'CostAnomalyAlert',
                    alert_data['anomaly_score'],
                    unit='None'
                )
                
                alerts_sent.append({
                    'type': 'cloudwatch',
                    'metric': 'CostAnomalyAlert',
                    'value': alert_data['anomaly_score']
                })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics.record_workflow_step_duration('send_alerts', execution_time)
            self.metrics.record_alerts_sent(len(alerts_sent))
            
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
            
        except Exception as e:
            logger.error(f"Alert sending failed: {str(e)}")
            self.metrics.record_workflow_step_error('send_alerts', str(e))
            raise
    
    def _update_budgets_step(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Update AWS Budgets based on forecast results."""
        start_time = datetime.now()
        
        try:
            # Extract data from previous step
            body = json.loads(event.get('body', '{}'))
            data = body.get('data', {})
            forecast_result = data.get('forecast_result', {})
            
            predictions = forecast_result.get('predictions', [])
            if not predictions:
                logger.warning("No predictions available for budget updates")
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'success': True,
                        'step': 'update_budgets',
                        'data': {'budgets_updated': 0},
                        'execution_time': 0
                    })
                }
            
            # Calculate next month's predicted spend
            monthly_prediction = sum(predictions[:30]) if len(predictions) >= 30 else sum(predictions)
            
            # Add 10% buffer for budget setting
            budget_amount = monthly_prediction * 1.1
            
            budgets_updated = []
            
            # Update monthly budget if configured
            if hasattr(self.config, 'budget_name') and self.config.budget_name:
                try:
                    # Update budget amount
                    next_month = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1)
                    
                    budget_response = self.budgets.modify_budget(
                        AccountId=self.config.aws_account_id,
                        Budget={
                            'BudgetName': self.config.budget_name,
                            'BudgetLimit': {
                                'Amount': str(budget_amount),
                                'Unit': 'USD'
                            },
                            'TimeUnit': 'MONTHLY',
                            'TimePeriod': {
                                'Start': next_month.date(),
                                'End': (next_month + timedelta(days=31)).date()
                            },
                            'BudgetType': 'COST'
                        }
                    )
                    
                    budgets_updated.append({
                        'budget_name': self.config.budget_name,
                        'new_amount': budget_amount,
                        'period': next_month.strftime('%Y-%m')
                    })
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NotFoundException':
                        logger.warning(f"Budget {self.config.budget_name} not found, skipping update")
                    else:
                        raise
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics.record_workflow_step_duration('update_budgets', execution_time)
            self.metrics.record_budget_updates(len(budgets_updated))
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'step': 'update_budgets',
                    'data': {
                        'budgets_updated': len(budgets_updated),
                        'budget_details': budgets_updated,
                        'predicted_monthly_cost': monthly_prediction
                    },
                    'execution_time': execution_time
                })
            }
            
        except Exception as e:
            logger.error(f"Budget update failed: {str(e)}")
            self.metrics.record_workflow_step_error('update_budgets', str(e))
            raise
    
    def _create_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Create formatted alert message for notifications."""
        forecast_summary = alert_data.get('forecast_summary', {})
        anomaly_score = alert_data.get('anomaly_score', 0)
        cost_increase_ratio = alert_data.get('cost_increase_ratio', 0)
        
        message = f"""
🚨 COST FORECAST ALERT 🚨

Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

ANOMALY DETECTION:
• Anomaly Score: {anomaly_score:.2f} (Threshold: 0.8)
• Cost Increase: {cost_increase_ratio:.1%} (Threshold: 20%)

FORECAST SUMMARY:
• Total Predicted Cost (30 days): ${forecast_summary.get('total_predicted_cost', 0):,.2f}
• Average Daily Cost: ${forecast_summary.get('average_daily_cost', 0):,.2f}

RECOMMENDATIONS:
"""
        
        recommendations = forecast_summary.get('recommendations', [])
        for i, rec in enumerate(recommendations[:5], 1):  # Limit to top 5
            message += f"  {i}. {rec.get('type', 'Unknown')}: {rec.get('description', 'No description')}\n"
            message += f"     Estimated Savings: ${rec.get('estimated_savings', 0):,.2f}\n"
        
        message += f"""
🔗 View detailed analysis in the Resource Forecaster dashboard.

This alert was generated automatically by the FinOps Resource Forecaster.
        """
        
        return message.strip()
    
    def start_workflow_execution(self, state_machine_arn: str, input_data: Dict[str, Any]) -> str:
        """Start a new workflow execution."""
        try:
            execution_name = f"forecaster-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            response = self.stepfunctions.start_execution(
                stateMachineArn=state_machine_arn,
                name=execution_name,
                input=json.dumps(input_data)
            )
            
            logger.info(f"Started workflow execution: {execution_name}")
            return response['executionArn']
            
        except Exception as e:
            logger.error(f"Failed to start workflow execution: {str(e)}")
            raise
    
    def get_execution_status(self, execution_arn: str) -> Dict[str, Any]:
        """Get the status of a workflow execution."""
        try:
            response = self.stepfunctions.describe_execution(
                executionArn=execution_arn
            )
            
            return {
                'status': response['status'],
                'start_date': response['startDate'].isoformat(),
                'stop_date': response.get('stopDate', {}).isoformat() if response.get('stopDate') else None,
                'input': json.loads(response['input']),
                'output': json.loads(response.get('output', '{}'))
            }
            
        except Exception as e:
            logger.error(f"Failed to get execution status: {str(e)}")
            raise