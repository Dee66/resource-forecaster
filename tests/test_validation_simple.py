"""Simple tests for validation functionality."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.forecaster.validation.validator import (
    ModelValidator, ValidationMetrics, BacktestResult
)


class TestValidationMetrics:
    """Test ValidationMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ValidationMetrics(
            mape=5.5,
            rmse=100.0,
            mae=75.0,
            r2=0.85,
            directional_accuracy=0.75,
            bias=5.0
        )
        
        expected = {
            'mape': 5.5,
            'rmse': 100.0,
            'mae': 75.0,
            'r2': 0.85,
            'directional_accuracy': 0.75,
            'bias': 5.0
        }
        
        assert metrics.to_dict() == expected


class TestModelValidatorCore:
    """Test core validation functionality without full config."""
    
    @pytest.fixture
    def simple_validator(self):
        """Create validator with minimal config."""
        # Mock config object
        class MockConfig:
            def __init__(self):
                self.metrics_s3_bucket = None
        
        validator = ModelValidator.__new__(ModelValidator)
        validator.config = MockConfig()
        validator.quality_gates = {
            'mape': 15.0,  # 15% MAPE threshold  
            'rmse': 1000,
            'r2': 0.6,
            'directional_accuracy': 0.65
        }
        validator.cloudwatch = None
        validator.s3_client = None
        
        return validator
    
    def test_calculate_metrics_basic(self, simple_validator):
        """Test basic metric calculation."""
        y_true = pd.Series([100, 110, 105, 120, 115])
        y_pred = pd.Series([98, 108, 107, 118, 117])
        
        metrics = simple_validator.calculate_metrics(y_true, y_pred)
        
        assert isinstance(metrics, ValidationMetrics)
        assert metrics.mape > 0
        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert -1 <= metrics.r2 <= 1
        assert 0 <= metrics.directional_accuracy <= 1
    
    def test_calculate_metrics_perfect_prediction(self, simple_validator):
        """Test metrics with perfect predictions."""
        y_true = pd.Series([100, 110, 105, 120, 115])
        y_pred = y_true.copy()
        
        metrics = simple_validator.calculate_metrics(y_true, y_pred)
        
        assert metrics.mape == 0.0
        assert metrics.rmse == 0.0
        assert metrics.mae == 0.0
        assert metrics.r2 == 1.0
        assert abs(metrics.bias) < 1e-10
    
    def test_calculate_metrics_with_dates(self, simple_validator):
        """Test metric calculation with date information."""
        y_true = pd.Series([100, 110, 105, 120, 115])
        y_pred = pd.Series([98, 108, 107, 118, 117])
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        
        metrics = simple_validator.calculate_metrics(y_true, y_pred, dates)
        
        assert isinstance(metrics, ValidationMetrics)
        assert 0 <= metrics.directional_accuracy <= 1
    
    def test_calculate_metrics_length_mismatch(self, simple_validator):
        """Test error handling for length mismatch."""
        y_true = pd.Series([100, 110, 105])
        y_pred = pd.Series([98, 108])
        
        with pytest.raises(ValueError):
            simple_validator.calculate_metrics(y_true, y_pred)
    
    def test_calculate_metrics_nan_values(self, simple_validator):
        """Test handling of NaN values."""
        y_true = pd.Series([100, np.nan, 105, 120])
        y_pred = pd.Series([98, 108, np.nan, 118])
        
        metrics = simple_validator.calculate_metrics(y_true, y_pred)
        
        # Should only use the 100/98 and 120/118 pairs
        assert isinstance(metrics, ValidationMetrics)
    
    def test_directional_accuracy_perfect(self, simple_validator):
        """Test perfect directional accuracy."""
        # Both increase then decrease
        y_true = pd.Series([100, 110, 105])
        y_pred = pd.Series([98, 108, 103])
        
        accuracy = simple_validator._calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_directional_accuracy_opposite(self, simple_validator):
        """Test opposite directional trends."""
        # True increases, predicted decreases
        y_true = pd.Series([100, 110, 120])
        y_pred = pd.Series([120, 110, 100])
        
        accuracy = simple_validator._calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_validate_against_quality_gates_pass(self, simple_validator):
        """Test quality gate validation - passing case."""
        metrics = ValidationMetrics(
            mape=10.0,    # Below 15% threshold
            rmse=500.0,   # Below 1000 threshold
            mae=300.0,
            r2=0.8,       # Above 0.6 threshold
            directional_accuracy=0.7,  # Above 0.65 threshold
            bias=10.0
        )
        
        result = simple_validator.validate_against_quality_gates(metrics)
        
        assert result['passed'] is True
        assert len(result['failures']) == 0
        assert result['metrics'] == metrics.to_dict()
    
    def test_validate_against_quality_gates_fail(self, simple_validator):
        """Test quality gate validation - failing case."""
        metrics = ValidationMetrics(
            mape=20.0,    # Above 15% threshold
            rmse=1500.0,  # Above 1000 threshold
            mae=800.0,
            r2=0.4,       # Below 0.6 threshold
            directional_accuracy=0.5,  # Below 0.65 threshold
            bias=100.0
        )
        
        result = simple_validator.validate_against_quality_gates(metrics)
        
        assert result['passed'] is False
        assert len(result['failures']) == 4  # All gates should fail
        
        # Check specific failures
        failure_metrics = [f['metric'] for f in result['failures']]
        assert 'mape' in failure_metrics
        assert 'rmse' in failure_metrics
        assert 'r2' in failure_metrics
        assert 'directional_accuracy' in failure_metrics
    
    def test_generate_test_periods(self, simple_validator):
        """Test generation of test periods."""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'usage_date': dates,
            'daily_cost': np.random.uniform(1000, 2000, 100)
        })
        
        test_periods = simple_validator._generate_test_periods(sample_data, forecast_horizon=7)
        
        assert len(test_periods) > 0
        assert all(isinstance(period, tuple) for period in test_periods)
        assert all(len(period) == 2 for period in test_periods)
    
    def test_split_data_for_period(self, simple_validator):
        """Test data splitting for backtesting."""
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'usage_date': dates,
            'daily_cost': np.random.uniform(1000, 2000, 100)
        })
        
        start_date = "2024-02-01"
        end_date = "2024-02-07"
        
        train_data, test_data = simple_validator._split_data_for_period(
            sample_data, start_date, end_date, forecast_horizon=7
        )
        
        assert len(train_data) > 0
        assert len(test_data) > 0
        assert train_data['usage_date'].max() < pd.to_datetime(start_date)
        assert test_data['usage_date'].min() >= pd.to_datetime(start_date)
        assert test_data['usage_date'].max() <= pd.to_datetime(end_date)
    
    def test_generate_training_recommendations(self, simple_validator):
        """Test training recommendation generation."""
        # Create results with various metric levels that will trigger recommendations
        high_mape_metrics = ValidationMetrics(25.0, 500.0, 300.0, 0.8, 0.7, 5.0)
        low_r2_metrics = ValidationMetrics(16.0, 500.0, 300.0, 0.3, 0.7, 5.0)  # Slightly higher MAPE
        high_bias_metrics = ValidationMetrics(10.0, 500.0, 300.0, 0.8, 0.7, 150.0)
        
        predictions = pd.DataFrame({'actual': [100], 'predicted': [95]})
        
        results = [
            BacktestResult("Period 1", high_mape_metrics, predictions),
            BacktestResult("Period 2", low_r2_metrics, predictions),
            BacktestResult("Period 3", high_bias_metrics, predictions)
        ]
        
        recommendations = simple_validator._generate_validation_recommendations(results, None)
        
        assert len(recommendations) > 0
        
        # Check for specific recommendation types
        rec_text = ' '.join(recommendations)
        # Should trigger based on mean_mape > 15 and mean_bias > 100
        assert 'MAPE' in rec_text or 'mape' in rec_text or 'bias' in rec_text or 'Bias' in rec_text
    
    def test_extract_feature_importance_sklearn(self, simple_validator):
        """Test feature importance extraction for sklearn models."""
        from unittest.mock import Mock
        
        # Mock sklearn model
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.6, 0.4])
        mock_model.feature_names_in_ = ['feature1', 'feature2']
        
        importance = simple_validator._extract_feature_importance(mock_model)
        
        expected = {'feature1': 0.6, 'feature2': 0.4}
        assert importance == expected
    
    def test_extract_feature_importance_linear(self, simple_validator):
        """Test feature importance extraction for linear models."""
        from unittest.mock import Mock
        
        # Mock linear model
        mock_model = Mock()
        mock_model.coef_ = np.array([0.8, -0.6])
        mock_model.feature_names_in_ = ['feature1', 'feature2']
        # Remove feature_importances_ attribute
        if hasattr(mock_model, 'feature_importances_'):
            delattr(mock_model, 'feature_importances_')
        
        importance = simple_validator._extract_feature_importance(mock_model)
        
        expected = {'feature1': 0.8, 'feature2': 0.6}  # Absolute values
        assert importance == expected
    
    def test_extract_feature_importance_no_support(self, simple_validator):
        """Test feature importance extraction when not supported."""
        from unittest.mock import Mock
        
        # Mock model without feature importance
        mock_model = Mock()
        
        importance = simple_validator._extract_feature_importance(mock_model)
        
        assert importance is None