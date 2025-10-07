"""Budget alert manager for automated cost optimization.

Integrates with AWS Budgets, Cost Explorer, and SNS to provide
proactive cost management and optimization recommendations.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError

from ..config import ForecasterConfig

logger = logging.getLogger(__name__)


@dataclass
class BudgetAlert:
    """Represents a budget alert configuration."""
    budget_name: str
    threshold_percentage: float
    alert_type: str
    recipients: List[str]
    enabled: bool = True


@dataclass
class CostRecommendation:
    """Represents a cost optimization recommendation."""
    recommendation_type: str
    resource_id: str
    current_cost: float
    projected_savings: float
    confidence_score: float
    action_required: str
    priority: str = "medium"


class BudgetAlertManager:
    """Manages budget alerts and cost optimization workflows."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.budgets = boto3.client('budgets')
        self.ce = boto3.client('ce')  # Cost Explorer
        self.sns = boto3.client('sns')
        self.ssm = boto3.client('ssm')
        
    def setup_forecaster_budgets(self, monthly_budget: float) -> Dict[str, Any]:
        """Setup comprehensive budgets for the forecaster system."""
        try:
            account_id = self.config.aws_account_id
            environment = getattr(self.config, 'environment', 'dev')
            
            # Main monthly budget
            main_budget = {
                'BudgetName': f'ResourceForecaster-Monthly-{environment}',
                'BudgetLimit': {
                    'Amount': str(monthly_budget),
                    'Unit': 'USD'
                },
                'TimeUnit': 'MONTHLY',
                'BudgetType': 'COST',
                'CostFilters': {
                    'TagKey': ['App'],
                    'TagValue': ['ResourceForecaster']
                }
            }
            
            # Daily budget (1/30th of monthly)
            daily_budget = {
                'BudgetName': f'ResourceForecaster-Daily-{environment}',
                'BudgetLimit': {
                    'Amount': str(monthly_budget / 30),
                    'Unit': 'USD'
                },
                'TimeUnit': 'DAILY',
                'BudgetType': 'COST',
                'CostFilters': {
                    'TagKey': ['App'],
                    'TagValue': ['ResourceForecaster']
                }
            }
            
            budgets_created = []
            
            # Create budgets
            for budget in [main_budget, daily_budget]:
                try:
                    self.budgets.create_budget(
                        AccountId=account_id,
                        Budget=budget
                    )
                    budgets_created.append(budget['BudgetName'])
                    logger.info(f"Created budget: {budget['BudgetName']}")
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'DuplicateRecordException':
                        logger.info(f"Budget {budget['BudgetName']} already exists, updating...")
                        self.budgets.modify_budget(
                            AccountId=account_id,
                            Budget=budget
                        )
                        budgets_created.append(budget['BudgetName'])
                    else:
                        raise
            
            # Setup budget alerts
            alert_configs = [
                {'threshold': 50, 'type': 'ACTUAL'},
                {'threshold': 80, 'type': 'ACTUAL'},
                {'threshold': 100, 'type': 'FORECASTED'},
                {'threshold': 120, 'type': 'FORECASTED'}
            ]
            
            alerts_created = []
            
            for budget_name in budgets_created:
                for alert_config in alert_configs:
                    alert = {
                        'AlertType': 'COST',
                        'ComparisonOperator': 'GREATER_THAN',
                        'Threshold': alert_config['threshold'],
                        'ThresholdType': 'PERCENTAGE',
                        'NotificationType': alert_config['type']
                    }
                    
                    # Add subscribers if SNS topic is configured
                    subscribers = []
                    if hasattr(self.config, 'alerts_topic_arn') and self.config.alerts_topic_arn:
                        subscribers.append({
                            'SubscriptionType': 'SNS',
                            'Address': self.config.alerts_topic_arn
                        })
                    
                    if hasattr(self.config, 'alert_emails'):
                        for email in self.config.alert_emails:
                            subscribers.append({
                                'SubscriptionType': 'EMAIL',
                                'Address': email
                            })
                    
                    if subscribers:
                        alert['Subscribers'] = subscribers
                        
                        try:
                            self.budgets.create_budget_action(
                                AccountId=account_id,
                                BudgetName=budget_name,
                                NotificationType=alert_config['type'],
                                ActionType='SNS_TOPIC',
                                ActionThreshold={
                                    'ActionThresholdValue': alert_config['threshold'],
                                    'ActionThresholdType': 'PERCENTAGE'
                                },
                                Definition={
                                    'SnsActionDefinition': {
                                        'TopicArn': self.config.alerts_topic_arn
                                    }
                                } if hasattr(self.config, 'alerts_topic_arn') else {},
                                Subscribers=subscribers
                            )
                            alerts_created.append(f"{budget_name}-{alert_config['threshold']}%")
                            
                        except ClientError as e:
                            if e.response['Error']['Code'] == 'DuplicateRecordException':
                                logger.info(f"Alert for {budget_name} at {alert_config['threshold']}% already exists")
                            else:
                                logger.warning(f"Failed to create alert: {str(e)}")
            
            return {
                'budgets_created': budgets_created,
                'alerts_created': alerts_created,
                'monthly_budget': monthly_budget
            }
            
        except Exception as e:
            logger.error(f"Failed to setup budgets: {str(e)}")
            raise
    
    def get_cost_recommendations(self, days_back: int = 30) -> List[CostRecommendation]:
        """Get cost optimization recommendations from AWS Cost Explorer."""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            recommendations = []
            
            # Get rightsizing recommendations
            try:
                response = self.ce.get_rightsizing_recommendation(
                    Service='AmazonEC2',
                    PageSize=100,
                    Configuration={
                        'BenefitsConsidered': True,
                        'RecommendationTarget': 'SAME_INSTANCE_FAMILY'
                    }
                )
                
                for rec in response.get('RightsizingRecommendations', []):
                    if rec.get('RightsizingType') == 'Modify':
                        current_instance = rec.get('CurrentInstance', {})
                        recommended_instance = rec.get('ModifyRecommendationDetail', {}).get('TargetInstances', [{}])[0]
                        
                        current_cost = float(current_instance.get('MonthlyCost', '0'))
                        projected_cost = float(recommended_instance.get('EstimatedMonthlyCost', '0'))
                        savings = current_cost - projected_cost
                        
                        if savings > 0:
                            recommendations.append(CostRecommendation(
                                recommendation_type="rightsizing",
                                resource_id=current_instance.get('ResourceId', 'unknown'),
                                current_cost=current_cost,
                                projected_savings=savings,
                                confidence_score=0.85,
                                action_required=f"Resize from {current_instance.get('InstanceType', 'unknown')} to {recommended_instance.get('InstanceType', 'unknown')}",
                                priority="high" if savings > 100 else "medium"
                            ))
                        
            except ClientError as e:
                logger.warning(f"Failed to get rightsizing recommendations: {str(e)}")
            
            # Get Reserved Instance recommendations
            try:
                response = self.ce.get_reservation_purchase_recommendation(
                    Service='AmazonEC2',
                    PaymentOption='NO_UPFRONT',
                    TermInYears='ONE_YEAR'
                )
                
                for rec in response.get('Recommendations', []):
                    recommendation_details = rec.get('RecommendationDetails', {})
                    estimated_savings = float(recommendation_details.get('EstimatedMonthlySavingsAmount', '0'))
                    
                    if estimated_savings > 0:
                        recommendations.append(CostRecommendation(
                            recommendation_type="reserved_instance",
                            resource_id=recommendation_details.get('InstanceType', 'unknown'),
                            current_cost=float(recommendation_details.get('EstimatedMonthlyOnDemandCost', '0')),
                            projected_savings=estimated_savings,
                            confidence_score=0.9,
                            action_required=f"Purchase Reserved Instance: {recommendation_details.get('InstanceType', 'unknown')}",
                            priority="high" if estimated_savings > 200 else "medium"
                        ))
                        
            except ClientError as e:
                logger.warning(f"Failed to get RI recommendations: {str(e)}")
            
            # Get Savings Plans recommendations
            try:
                response = self.ce.get_savings_plans_purchase_recommendation(
                    SavingsPlansType='COMPUTE_SP',
                    TermInYears='ONE_YEAR',
                    PaymentOption='NO_UPFRONT'
                )
                
                for rec in response.get('SavingsPlansEstimatedGeneralMonthlyCost', []):
                    estimated_savings = float(rec.get('EstimatedMonthlySavings', '0'))
                    
                    if estimated_savings > 0:
                        recommendations.append(CostRecommendation(
                            recommendation_type="savings_plan",
                            resource_id="compute-sp",
                            current_cost=float(rec.get('EstimatedOnDemandCost', '0')),
                            projected_savings=estimated_savings,
                            confidence_score=0.85,
                            action_required="Purchase Compute Savings Plan",
                            priority="high" if estimated_savings > 150 else "medium"
                        ))
                        
            except ClientError as e:
                logger.warning(f"Failed to get Savings Plans recommendations: {str(e)}")
            
            # Sort by projected savings (highest first)
            recommendations.sort(key=lambda x: x.projected_savings, reverse=True)
            
            logger.info(f"Generated {len(recommendations)} cost recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get cost recommendations: {str(e)}")
            return []
    
    def process_budget_alert(self, alert_event: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming budget alert and trigger optimization actions."""
        try:
            budget_name = alert_event.get('budgetName', 'Unknown')
            threshold = alert_event.get('threshold', 0)
            actual_spend = alert_event.get('actualSpend', 0)
            forecasted_spend = alert_event.get('forecastedSpend', 0)
            
            logger.info(f"Processing budget alert for {budget_name}, threshold: {threshold}%")
            
            # Get current recommendations
            recommendations = self.get_cost_recommendations()
            
            # Calculate potential savings
            total_potential_savings = sum(rec.projected_savings for rec in recommendations)
            
            # Prepare alert response
            alert_response = {
                'budget_name': budget_name,
                'threshold_exceeded': threshold,
                'actual_spend': actual_spend,
                'forecasted_spend': forecasted_spend,
                'recommendations_count': len(recommendations),
                'total_potential_savings': total_potential_savings,
                'high_priority_recommendations': [
                    {
                        'type': rec.recommendation_type,
                        'resource': rec.resource_id,
                        'savings': rec.projected_savings,
                        'action': rec.action_required
                    }
                    for rec in recommendations if rec.priority == 'high'
                ][:5],  # Top 5 high priority
                'timestamp': datetime.now().isoformat()
            }
            
            # Send detailed alert if significant threshold exceeded
            if threshold >= 80:
                self._send_detailed_alert(alert_response)
            
            # Store alert data for dashboard
            self._store_alert_data(alert_response)
            
            return alert_response
            
        except Exception as e:
            logger.error(f"Failed to process budget alert: {str(e)}")
            raise
    
    def _send_detailed_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send detailed alert with recommendations."""
        try:
            message = self._format_budget_alert_message(alert_data)
            
            # Send to SNS if configured
            if hasattr(self.config, 'alerts_topic_arn') and self.config.alerts_topic_arn:
                self.sns.publish(
                    TopicArn=self.config.alerts_topic_arn,
                    Subject=f"🚨 Budget Alert: {alert_data['budget_name']}",
                    Message=message
                )
                
                logger.info(f"Sent detailed budget alert for {alert_data['budget_name']}")
                
        except Exception as e:
            logger.error(f"Failed to send detailed alert: {str(e)}")
    
    def _format_budget_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format budget alert message for notifications."""
        message = f"""
🚨 BUDGET THRESHOLD EXCEEDED 🚨

Budget: {alert_data['budget_name']}
Threshold Exceeded: {alert_data['threshold_exceeded']}%
Current Spend: ${alert_data['actual_spend']:,.2f}
Forecasted Spend: ${alert_data['forecasted_spend']:,.2f}

💰 OPTIMIZATION OPPORTUNITIES:
Total Potential Savings: ${alert_data['total_potential_savings']:,.2f}
High Priority Recommendations: {len(alert_data['high_priority_recommendations'])}

TOP RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(alert_data['high_priority_recommendations'], 1):
            message += f"  {i}. {rec['type'].title().replace('_', ' ')}: {rec['resource']}\n"
            message += f"     Action: {rec['action']}\n"
            message += f"     Monthly Savings: ${rec['savings']:,.2f}\n\n"
        
        message += f"""
📊 View detailed analysis in the Resource Forecaster dashboard.
🔗 Access Cost Explorer for additional insights.

Alert generated: {alert_data['timestamp']}
        """
        
        return message.strip()
    
    def _store_alert_data(self, alert_data: Dict[str, Any]) -> None:
        """Store alert data in SSM Parameter Store for dashboard access."""
        try:
            parameter_name = f"/forecaster/{self.config.environment}/alerts/latest"
            
            self.ssm.put_parameter(
                Name=parameter_name,
                Value=json.dumps(alert_data),
                Type='String',
                Overwrite=True,
                Description=f"Latest budget alert data for Resource Forecaster"
            )
            
            logger.info(f"Stored alert data in parameter: {parameter_name}")
            
        except Exception as e:
            logger.warning(f"Failed to store alert data: {str(e)}")
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status and spending trends."""
        try:
            account_id = self.config.aws_account_id
            environment = getattr(self.config, 'environment', 'dev')
            
            budget_names = [
                f'ResourceForecaster-Monthly-{environment}',
                f'ResourceForecaster-Daily-{environment}'
            ]
            
            budget_status = {}
            
            for budget_name in budget_names:
                try:
                    response = self.budgets.describe_budget(
                        AccountId=account_id,
                        BudgetName=budget_name
                    )
                    
                    budget = response['Budget']
                    calculated_spend = budget.get('CalculatedSpend', {})
                    
                    budget_status[budget_name] = {
                        'limit': float(budget['BudgetLimit']['Amount']),
                        'actual_spend': float(calculated_spend.get('ActualSpend', {}).get('Amount', '0')),
                        'forecasted_spend': float(calculated_spend.get('ForecastedSpend', {}).get('Amount', '0')),
                        'time_unit': budget['TimeUnit'],
                        'budget_type': budget['BudgetType']
                    }
                    
                    # Calculate utilization percentage
                    limit = budget_status[budget_name]['limit']
                    actual = budget_status[budget_name]['actual_spend']
                    forecasted = budget_status[budget_name]['forecasted_spend']
                    
                    budget_status[budget_name]['actual_percentage'] = (actual / limit * 100) if limit > 0 else 0
                    budget_status[budget_name]['forecasted_percentage'] = (forecasted / limit * 100) if limit > 0 else 0
                    
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NotFoundException':
                        logger.warning(f"Budget {budget_name} not found")
                    else:
                        raise
            
            return budget_status
            
        except Exception as e:
            logger.error(f"Failed to get budget status: {str(e)}")
            return {}
    
    def create_cost_explorer_playbook(self) -> Dict[str, Any]:
        """Create automated Cost Explorer analysis playbook."""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            # Get cost and usage data
            response = self.ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            )
            
            # Analyze cost trends
            daily_costs = []
            service_costs = {}
            
            for result in response['ResultsByTime']:
                date = result['TimePeriod']['Start']
                total_cost = 0
                
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    total_cost += cost
                    
                    if service not in service_costs:
                        service_costs[service] = []
                    service_costs[service].append(cost)
                
                daily_costs.append({
                    'date': date,
                    'cost': total_cost
                })
            
            # Calculate trends
            if len(daily_costs) >= 7:
                recent_avg = sum(day['cost'] for day in daily_costs[-7:]) / 7
                previous_avg = sum(day['cost'] for day in daily_costs[-14:-7]) / 7 if len(daily_costs) >= 14 else recent_avg
                trend_percentage = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            else:
                trend_percentage = 0
            
            # Identify top cost services
            service_totals = {
                service: sum(costs) for service, costs in service_costs.items()
            }
            top_services = sorted(service_totals.items(), key=lambda x: x[1], reverse=True)[:10]
            
            playbook = {
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'cost_trend': {
                    'trend_percentage': trend_percentage,
                    'trend_direction': 'increasing' if trend_percentage > 0 else 'decreasing',
                    'is_significant': abs(trend_percentage) > 10
                },
                'top_services': [
                    {
                        'service': service,
                        'total_cost': cost,
                        'percentage_of_total': (cost / sum(service_totals.values()) * 100) if sum(service_totals.values()) > 0 else 0
                    }
                    for service, cost in top_services
                ],
                'daily_costs': daily_costs,
                'total_period_cost': sum(day['cost'] for day in daily_costs),
                'average_daily_cost': sum(day['cost'] for day in daily_costs) / len(daily_costs) if daily_costs else 0,
                'generated_at': datetime.now().isoformat()
            }
            
            return playbook
            
        except Exception as e:
            logger.error(f"Failed to create Cost Explorer playbook: {str(e)}")
            return {}