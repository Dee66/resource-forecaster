"""Scheduled job manager for automated FinOps workflows.

Provides time-based concurrency windows and automated execution
of forecasting, optimization, and alert workflows.
"""

import json
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError

from .step_functions_orchestrator import StepFunctionsOrchestrator
from .budget_alert_manager import BudgetAlertManager
from ..config import ForecasterConfig

logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """Represents a scheduled FinOps job."""
    job_name: str
    cron_expression: str
    description: str
    enabled: bool = True
    job_type: str = "forecast"
    parameters: Optional[Dict[str, Any]] = None


class ScheduledJobManager:
    """Manages scheduled execution of FinOps workflows."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.events = boto3.client('events')
        self.stepfunctions = boto3.client('stepfunctions')
        self.lambda_client = boto3.client('lambda')
        
        self.orchestrator = StepFunctionsOrchestrator(config)
        self.budget_manager = BudgetAlertManager(config)
        
    def setup_scheduled_jobs(self, state_machine_arn: str) -> Dict[str, Any]:
        """Setup all scheduled jobs for automated FinOps workflows."""
        try:
            environment = getattr(self.config, 'environment', 'dev')
            
            # Define scheduled jobs
            jobs = [
                ScheduledJob(
                    job_name=f"daily-forecast-{environment}",
                    cron_expression="cron(0 6 * * ? *)",  # 6 AM UTC daily
                    description="Daily cost forecasting and optimization analysis",
                    job_type="forecast",
                    parameters={"forecast_days": 30, "include_recommendations": True}
                ),
                ScheduledJob(
                    job_name=f"weekly-budget-review-{environment}",
                    cron_expression="cron(0 8 ? * MON *)",  # 8 AM UTC every Monday
                    description="Weekly budget review and alert configuration update",
                    job_type="budget_review",
                    parameters={"review_period_days": 7}
                ),
                ScheduledJob(
                    job_name=f"monthly-optimization-report-{environment}",
                    cron_expression="cron(0 9 1 * ? *)",  # 9 AM UTC on 1st of month
                    description="Monthly cost optimization report and recommendations",
                    job_type="optimization_report",
                    parameters={"report_period_days": 30}
                ),
                ScheduledJob(
                    job_name=f"real-time-anomaly-check-{environment}",
                    cron_expression="rate(30 minutes)",  # Every 30 minutes
                    description="Real-time cost anomaly detection and alerting",
                    job_type="anomaly_check",
                    parameters={"lookback_hours": 2}
                )
            ]
            
            # Disable real-time checks in non-prod environments
            if environment != 'prod':
                jobs = [job for job in jobs if job.job_type != 'anomaly_check']
            
            created_jobs = []
            
            for job in jobs:
                try:
                    # Create EventBridge rule
                    rule_response = self.events.put_rule(
                        Name=job.job_name,
                        ScheduleExpression=job.cron_expression,
                        Description=job.description,
                        State='ENABLED' if job.enabled else 'DISABLED'
                    )
                    
                    # Create target for Step Functions
                    target_input = {
                        "job_type": job.job_type,
                        "job_name": job.job_name,
                        "parameters": job.parameters or {},
                        "scheduled_time": "$.time",
                        "environment": environment
                    }
                    
                    self.events.put_targets(
                        Rule=job.job_name,
                        Targets=[
                            {
                                'Id': '1',
                                'Arn': state_machine_arn,
                                'RoleArn': self._get_events_role_arn(),
                                'Input': json.dumps(target_input)
                            }
                        ]
                    )
                    
                    created_jobs.append({
                        'job_name': job.job_name,
                        'schedule': job.cron_expression,
                        'type': job.job_type,
                        'rule_arn': rule_response['RuleArn']
                    })
                    
                    logger.info(f"Created scheduled job: {job.job_name}")
                    
                except ClientError as e:
                    logger.error(f"Failed to create job {job.job_name}: {str(e)}")
            
            return {
                'jobs_created': len(created_jobs),
                'job_details': created_jobs,
                'environment': environment
            }
            
        except Exception as e:
            logger.error(f"Failed to setup scheduled jobs: {str(e)}")
            raise
    
    def handle_scheduled_event(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Handle incoming scheduled events from EventBridge."""
        try:
            job_type = event.get('job_type', 'unknown')
            job_name = event.get('job_name', 'unknown')
            parameters = event.get('parameters', {})
            
            logger.info(f"Processing scheduled job: {job_name} (type: {job_type})")
            
            # Check if we're in low-cost time window
            if not self._is_low_cost_window():
                logger.info("Skipping job execution - not in low-cost time window")
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'success': True,
                        'skipped': True,
                        'reason': 'Outside low-cost time window'
                    })
                }
            
            # Route to appropriate handler
            if job_type == 'forecast':
                return self._handle_forecast_job(parameters)
            elif job_type == 'budget_review':
                return self._handle_budget_review_job(parameters)
            elif job_type == 'optimization_report':
                return self._handle_optimization_report_job(parameters)
            elif job_type == 'anomaly_check':
                return self._handle_anomaly_check_job(parameters)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
                
        except Exception as e:
            logger.error(f"Scheduled job failed: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'success': False,
                    'error': str(e),
                    'job_name': event.get('job_name', 'unknown')
                })
            }
    
    def _is_low_cost_window(self) -> bool:
        """Check if current time is within low-cost execution window."""
        current_time = datetime.now().time()
        
        # Define low-cost windows (UTC times when AWS usage is typically lower)
        low_cost_windows = [
            (time(2, 0), time(8, 0)),   # 2 AM - 8 AM UTC
            (time(14, 0), time(16, 0))  # 2 PM - 4 PM UTC
        ]
        
        for start_time, end_time in low_cost_windows:
            if start_time <= current_time <= end_time:
                return True
        
        return False
    
    def _handle_forecast_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle daily forecast job execution."""
        try:
            # Prepare workflow input
            workflow_input = {
                'action': 'collect_data',
                'forecast_days': parameters.get('forecast_days', 30),
                'include_recommendations': parameters.get('include_recommendations', True),
                'date_range': {
                    'start': (datetime.now().date() - timedelta(days=90)).isoformat(),
                    'end': datetime.now().date().isoformat()
                }
            }
            
            # Start workflow execution
            state_machine_arn = self._get_state_machine_arn()
            execution_arn = self.orchestrator.start_workflow_execution(
                state_machine_arn,
                workflow_input
            )
            
            logger.info(f"Started forecast workflow: {execution_arn}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'execution_arn': execution_arn,
                    'job_type': 'forecast',
                    'workflow_input': workflow_input
                })
            }
            
        except Exception as e:
            logger.error(f"Forecast job failed: {str(e)}")
            raise
    
    def _handle_budget_review_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle weekly budget review job execution."""
        try:
            review_period_days = parameters.get('review_period_days', 7)
            
            # Get current budget status
            budget_status = self.budget_manager.get_budget_status()
            
            # Get cost recommendations
            recommendations = self.budget_manager.get_cost_recommendations(review_period_days)
            
            # Create Cost Explorer playbook
            playbook = self.budget_manager.create_cost_explorer_playbook()
            
            # Calculate review metrics
            total_potential_savings = sum(rec.projected_savings for rec in recommendations)
            high_priority_count = len([rec for rec in recommendations if rec.priority == 'high'])
            
            review_result = {
                'review_period_days': review_period_days,
                'budget_status': budget_status,
                'recommendations_count': len(recommendations),
                'total_potential_savings': total_potential_savings,
                'high_priority_recommendations': high_priority_count,
                'cost_explorer_playbook': playbook,
                'review_timestamp': datetime.now().isoformat()
            }
            
            # Send weekly summary if significant savings available
            if total_potential_savings > 500:  # $500 threshold
                self._send_weekly_summary(review_result)
            
            logger.info(f"Completed budget review - {len(recommendations)} recommendations, ${total_potential_savings:.2f} potential savings")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'job_type': 'budget_review',
                    'review_result': review_result
                })
            }
            
        except Exception as e:
            logger.error(f"Budget review job failed: {str(e)}")
            raise
    
    def _handle_optimization_report_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle monthly optimization report job execution."""
        try:
            report_period_days = parameters.get('report_period_days', 30)
            
            # Generate comprehensive optimization report
            recommendations = self.budget_manager.get_cost_recommendations(report_period_days)
            playbook = self.budget_manager.create_cost_explorer_playbook()
            budget_status = self.budget_manager.get_budget_status()
            
            # Calculate monthly metrics
            total_spend = sum(day['cost'] for day in playbook.get('daily_costs', []))
            total_potential_savings = sum(rec.projected_savings for rec in recommendations)
            savings_percentage = (total_potential_savings / total_spend * 100) if total_spend > 0 else 0
            
            # Categorize recommendations
            recommendations_by_type = {}
            for rec in recommendations:
                if rec.recommendation_type not in recommendations_by_type:
                    recommendations_by_type[rec.recommendation_type] = []
                recommendations_by_type[rec.recommendation_type].append(rec)
            
            report = {
                'report_period': {
                    'days': report_period_days,
                    'start_date': (datetime.now().date() - timedelta(days=report_period_days)).isoformat(),
                    'end_date': datetime.now().date().isoformat()
                },
                'cost_summary': {
                    'total_spend': total_spend,
                    'average_daily_spend': total_spend / report_period_days if report_period_days > 0 else 0,
                    'cost_trend': playbook.get('cost_trend', {})
                },
                'optimization_summary': {
                    'total_recommendations': len(recommendations),
                    'total_potential_savings': total_potential_savings,
                    'savings_percentage': savings_percentage,
                    'recommendations_by_type': {
                        rec_type: len(recs) for rec_type, recs in recommendations_by_type.items()
                    }
                },
                'top_recommendations': [
                    {
                        'type': rec.recommendation_type,
                        'resource': rec.resource_id,
                        'current_cost': rec.current_cost,
                        'projected_savings': rec.projected_savings,
                        'action': rec.action_required,
                        'priority': rec.priority
                    }
                    for rec in sorted(recommendations, key=lambda x: x.projected_savings, reverse=True)[:10]
                ],
                'budget_status': budget_status,
                'generated_at': datetime.now().isoformat()
            }
            
            # Send monthly report
            self._send_monthly_report(report)
            
            logger.info(f"Generated monthly optimization report - {savings_percentage:.1f}% potential savings")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'job_type': 'optimization_report',
                    'report': report
                })
            }
            
        except Exception as e:
            logger.error(f"Optimization report job failed: {str(e)}")
            raise
    
    def _handle_anomaly_check_job(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle real-time anomaly check job execution."""
        try:
            lookback_hours = parameters.get('lookback_hours', 2)
            
            # Get recent cost data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)
            
            # Simple anomaly check using recent vs historical averages
            # In production, this would use more sophisticated anomaly detection
            
            # For now, check budget status for immediate alerts
            budget_status = self.budget_manager.get_budget_status()
            
            anomalies_detected = []
            
            for budget_name, status in budget_status.items():
                # Check for rapid spending increases
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
            
            # Send alerts for high severity anomalies
            for anomaly in anomalies_detected:
                if anomaly['severity'] == 'high':
                    self._send_anomaly_alert(anomaly)
            
            logger.info(f"Anomaly check completed - {len(anomalies_detected)} anomalies detected")
            
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
            
        except Exception as e:
            logger.error(f"Anomaly check job failed: {str(e)}")
            raise
    
    def _send_weekly_summary(self, review_result: Dict[str, Any]) -> None:
        """Send weekly budget review summary."""
        try:
            if not hasattr(self.config, 'alerts_topic_arn') or not self.config.alerts_topic_arn:
                return
            
            sns = boto3.client('sns')
            
            alerts_topic_arn = getattr(self.config, 'alerts_topic_arn', None)
            if not alerts_topic_arn:
                return
            
            sns = boto3.client('sns')
            
            message = f"""
📊 WEEKLY FINOPS SUMMARY 📊

Review Period: {review_result['review_period_days']} days
Generated: {review_result['review_timestamp']}

💰 OPTIMIZATION OPPORTUNITIES:
• Total Recommendations: {review_result['recommendations_count']}
• High Priority Actions: {review_result['high_priority_recommendations']}
• Potential Monthly Savings: ${review_result['total_potential_savings']:,.2f}

📈 BUDGET STATUS:
"""
            
            for budget_name, status in review_result.get('budget_status', {}).items():
                message += f"• {budget_name}: {status.get('actual_percentage', 0):.1f}% utilized\n"
            
            message += f"""
🔗 View detailed analysis in the Resource Forecaster dashboard.

Next weekly review: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}
            """
            
            sns.publish(
                TopicArn=alerts_topic_arn,
                Subject="📊 Weekly FinOps Summary",
                Message=message.strip()
            )
            
            sns.publish(
                TopicArn=alerts_topic_arn,
                Subject="📊 Weekly FinOps Summary",
                Message=message.strip()
            )
            
            logger.info("Sent weekly summary notification")
            
        except Exception as e:
            logger.warning(f"Failed to send weekly summary: {str(e)}")
    
    def _send_monthly_report(self, report: Dict[str, Any]) -> None:
        """Send monthly optimization report."""
        try:
            alerts_topic_arn = getattr(self.config, 'alerts_topic_arn', None)
            if not alerts_topic_arn:
                return
            
            sns = boto3.client('sns')
            
            cost_summary = report['cost_summary']
            opt_summary = report['optimization_summary']
            
            message = f"""
📈 MONTHLY FINOPS OPTIMIZATION REPORT 📈

Period: {report['report_period']['start_date']} to {report['report_period']['end_date']}

💵 COST SUMMARY:
• Total Spend: ${cost_summary['total_spend']:,.2f}
• Average Daily: ${cost_summary['average_daily_spend']:,.2f}
• Cost Trend: {cost_summary.get('cost_trend', {}).get('trend_direction', 'stable')} ({cost_summary.get('cost_trend', {}).get('trend_percentage', 0):.1f}%)

🎯 OPTIMIZATION POTENTIAL:
• Total Recommendations: {opt_summary['total_recommendations']}
• Potential Savings: ${opt_summary['total_potential_savings']:,.2f}
• Savings Percentage: {opt_summary['savings_percentage']:.1f}%

🏆 TOP OPPORTUNITIES:
"""
            
            for i, rec in enumerate(report['top_recommendations'][:5], 1):
                message += f"  {i}. {rec['type'].title().replace('_', ' ')}: ${rec['projected_savings']:,.2f}/month\n"
            
            message += f"""
📊 Full report available in the Resource Forecaster dashboard.
🎯 Target: 40% cost reduction through predictive analytics

Report generated: {report['generated_at']}
            """
            
            sns.publish(
                TopicArn=alerts_topic_arn,
                Subject="📈 Monthly FinOps Optimization Report",
                Message=message.strip()
            )
            
            logger.info("Sent monthly report notification")
            
        except Exception as e:
            logger.warning(f"Failed to send monthly report: {str(e)}")
    
    def _send_anomaly_alert(self, anomaly: Dict[str, Any]) -> None:
        """Send real-time anomaly alert."""
        try:
            alerts_topic_arn = getattr(self.config, 'alerts_topic_arn', None)
            if not alerts_topic_arn:
                return
            
            sns = boto3.client('sns')
            
            message = f"""
🚨 REAL-TIME COST ANOMALY DETECTED 🚨

Type: {anomaly['type'].replace('_', ' ').title()}
Severity: {anomaly['severity'].upper()}
Detected: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

DETAILS:
Budget: {anomaly.get('budget_name', 'Unknown')}
Current Utilization: {anomaly.get('percentage', 0):.1f}%

⚡ IMMEDIATE ACTION RECOMMENDED ⚡

🔗 Check Resource Forecaster dashboard for detailed analysis.
🎯 Review active resources and optimization recommendations.
            """
            
            sns.publish(
                TopicArn=alerts_topic_arn,
                Subject=f"🚨 Cost Anomaly: {anomaly['severity'].upper()}",
                Message=message.strip()
            )
            
            logger.info(f"Sent anomaly alert for {anomaly['type']}")
            
        except Exception as e:
            logger.warning(f"Failed to send anomaly alert: {str(e)}")
    
    def _get_events_role_arn(self) -> str:
        """Get or create IAM role for EventBridge to invoke Step Functions."""
        # In practice, this would be created by CDK
        # For now, return a placeholder that would be replaced by actual role ARN
        return f"arn:aws:iam::{self.config.aws_account_id}:role/ResourceForecaster-EventsRole"
    
    def _get_state_machine_arn(self) -> str:
        """Get the Step Functions state machine ARN."""
        # In practice, this would be stored in SSM or passed as environment variable
        environment = getattr(self.config, 'environment', 'dev')
        return f"arn:aws:states:{self.config.aws_region}:{self.config.aws_account_id}:stateMachine:ResourceForecaster-{environment}"
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get status of all scheduled jobs."""
        try:
            environment = getattr(self.config, 'environment', 'dev')
            
            # List all forecaster-related rules
            response = self.events.list_rules(
                NamePrefix=f"daily-forecast-{environment}"
            )
            
            job_status = {}
            
            for rule in response.get('Rules', []):
                rule_name = rule['Name']
                
                # Get rule details
                job_status[rule_name] = {
                    'schedule': rule.get('ScheduleExpression', 'Unknown'),
                    'state': rule.get('State', 'Unknown'),
                    'description': rule.get('Description', ''),
                    'last_modified': rule.get('EventBusName', 'default')
                }
                
                # Get recent executions (if available)
                try:
                    targets_response = self.events.list_targets_by_rule(Rule=rule_name)
                    job_status[rule_name]['targets'] = len(targets_response.get('Targets', []))
                except ClientError:
                    job_status[rule_name]['targets'] = 0
            
            return {
                'environment': environment,
                'jobs': job_status,
                'total_jobs': len(job_status),
                'status_checked_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get job status: {str(e)}")
            return {}