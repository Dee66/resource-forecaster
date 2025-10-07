"""Tests for workflow orchestration components.

Test suite for Step Functions integration, budget alerts,
and scheduled job management in the FinOps workflow.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from .simple_workflow import (
    SimpleWorkflowOrchestrator,
    SimpleBudgetManager, 
    SimpleJobManager
)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock()
    config.aws_account_id = "123456789012"
    config.aws_region = "us-east-1"
    config.environment = "test"
    return config


@pytest.fixture
def orchestrator(mock_config):
    """Create SimpleWorkflowOrchestrator instance for testing."""
    return SimpleWorkflowOrchestrator(mock_config)


@pytest.fixture
def budget_manager(mock_config):
    """Create SimpleBudgetManager instance for testing."""
    return SimpleBudgetManager(mock_config)


@pytest.fixture
def job_manager(mock_config):
    """Create SimpleJobManager instance for testing."""
    return SimpleJobManager(mock_config)


class TestSimpleWorkflowOrchestrator:
    """Test cases for Step Functions orchestration."""
    
    def test_handle_collect_data_workflow(self, orchestrator):
        """Test data collection workflow step."""
        # Test event
        event = {
            'action': 'collect_data',
            'date_range': {
                'start': '2024-01-01',
                'end': '2024-01-02'
            }
        }
        
        # Execute
        result = orchestrator.handle_workflow_event(event, None)
        
        # Verify
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['success'] is True
        assert body['step'] == 'collect_data'
        assert 'cost_data' in body['data']
        assert 'metrics_data' in body['data']
    
    def test_handle_generate_forecast_workflow(self, orchestrator):
        """Test forecast generation workflow step."""
        # Test event
        event = {
            'action': 'generate_forecast',
            'body': json.dumps({
                'data': {
                    'cost_data': [{'date': '2024-01-01', 'cost': 100.0}],
                    'metrics_data': [{'metric': 'CPUUtilization', 'value': 75.0}]
                }
            })
        }
        
        # Execute
        result = orchestrator.handle_workflow_event(event, None)
        
        # Verify
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['success'] is True
        assert body['step'] == 'generate_forecast'
        assert 'forecast_result' in body['data']
    
    def test_handle_evaluate_forecast_workflow(self, orchestrator):
        """Test forecast evaluation workflow step."""
        # Test event with high anomaly score
        event = {
            'action': 'evaluate_forecast',
            'body': json.dumps({
                'data': {
                    'forecast_result': {
                        'predictions': [100.0, 200.0, 180.0],  # High variation
                        'anomaly_score': 0.9,  # High anomaly
                        'recommendations': [
                            {'type': 'rightsizing', 'estimated_savings': 50.0}
                        ]
                    }
                }
            })
        }
        
        # Execute
        result = orchestrator.handle_workflow_event(event, None)
        
        # Verify
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['success'] is True
        assert body['step'] == 'evaluate_forecast'
        assert body['data']['alert_data']['needs_alerts'] is True
    
    def test_start_workflow_execution(self, orchestrator):
        """Test basic workflow execution capabilities."""
        # Test that we can handle basic workflow events
        event = {'action': 'collect_data'}
        result = orchestrator.handle_workflow_event(event, None)
        
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['success'] is True


class TestSimpleBudgetManager:
    """Test cases for budget alert management."""
    
    def test_get_budget_status(self, budget_manager):
        """Test getting budget status."""
        status = budget_manager.get_budget_status()
        
        assert 'test-budget-monthly' in status
        budget = status['test-budget-monthly']
        assert budget['limit'] == 1000.0
        assert budget['actual_spend'] == 750.0
        assert budget['actual_percentage'] == 75.0
    
    def test_get_cost_recommendations(self, budget_manager):
        """Test getting cost optimization recommendations."""
        recommendations = budget_manager.get_cost_recommendations()
        
        assert len(recommendations) == 2
        assert recommendations[0].recommendation_type == "rightsizing"
        assert recommendations[0].projected_savings == 100.0
        assert recommendations[1].recommendation_type == "reserved_instance"
    
    def test_create_cost_explorer_playbook(self, budget_manager):
        """Test creating Cost Explorer playbook."""
        playbook = budget_manager.create_cost_explorer_playbook()
        
        assert playbook['total_period_cost'] == 1000.0
        assert playbook['cost_trend']['trend_direction'] == 'increasing'


class TestSimpleJobManager:
    """Test cases for scheduled job management."""
    
    def test_handle_forecast_job(self, job_manager):
        """Test handling scheduled forecast job."""
        parameters = {'forecast_days': 30, 'include_recommendations': True}
        
        result = job_manager._handle_forecast_job(parameters)
        
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['success'] is True
        assert body['job_type'] == 'forecast'
        assert 'execution_arn' in body
    
    def test_handle_budget_review_job(self, job_manager):
        """Test handling scheduled budget review job."""
        parameters = {'review_period_days': 7}
        
        result = job_manager._handle_budget_review_job(parameters)
        
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['success'] is True
        assert body['job_type'] == 'budget_review'
        
        review_result = body['review_result']
        assert review_result['total_potential_savings'] == 300.0  # 100 + 200
        assert review_result['high_priority_recommendations'] == 1
    
    def test_low_cost_window_detection(self, job_manager):
        """Test low-cost time window detection."""
        # Simplified version always returns True for testing
        assert job_manager._is_low_cost_window() is True
    
    def test_handle_anomaly_check_job(self, job_manager):
        """Test handling real-time anomaly check job."""
        parameters = {'lookback_hours': 2}
        
        result = job_manager._handle_anomaly_check_job(parameters)
        
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert body['success'] is True
        # Budget has 75% utilization, so should NOT trigger anomaly (threshold is >75)
        assert body['anomalies_detected'] == 0


if __name__ == "__main__":
    pytest.main([__file__])