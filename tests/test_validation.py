"""Tests for model validation and backtesting functionality."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.forecaster.validation.validator import (
    ModelValidator, ValidationMetrics, BacktestResult
)
from src.forecaster.config import ForecasterConfig
from src.forecaster.exceptions import ForecastValidationError


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


class TestBacktestResult:
    """Test BacktestResult dataclass."""
    
    def test_to_dict_basic(self):
        """Test basic conversion to dictionary."""
        metrics = ValidationMetrics(5.5, 100.0, 75.0, 0.85, 0.75, 5.0)
        predictions = pd.DataFrame({
            'actual': [100, 110, 105],
            'predicted': [98, 108, 107]
        })
        
        result = BacktestResult(
            test_period="2024-01-01 to 2024-01-07",
            metrics=metrics,
            predictions=predictions
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['test_period'] == "2024-01-01 to 2024-01-07"
        assert result_dict['prediction_count'] == 3
        assert 'metrics' in result_dict
        assert 'feature_importance' not in result_dict
    
    def test_to_dict_with_feature_importance(self):
        """Test conversion with feature importance."""
        metrics = ValidationMetrics(5.5, 100.0, 75.0, 0.85, 0.75, 5.0)
        predictions = pd.DataFrame({'actual': [100], 'predicted': [98]})
        feature_importance = {'feature1': 0.6, 'feature2': 0.4}
        
        result = BacktestResult(
            test_period="2024-01-01 to 2024-01-07",
            metrics=metrics,
            predictions=predictions,
            feature_importance=feature_importance
        )
        
        result_dict = result.to_dict()
        assert result_dict['feature_importance'] == feature_importance


class TestModelValidator:
    """Test ModelValidator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ForecasterConfig(
            environment="dev",
            data_source={
                "cur_bucket": "test-cur-bucket",
                "athena_output_bucket": "test-athena-bucket"
            },
            infrastructure={
                "model_bucket": "test-model-bucket",
                "data_bucket": "test-data-bucket"
            }
        )
    
    @pytest.fixture
    def validator(self, config):
        """Create model validator instance."""
        return ModelValidator(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        trend = np.linspace(1000, 1200, 100)
        noise = np.random.normal(0, 50, 100)
        costs = trend + noise
        
        return pd.DataFrame({
            'usage_date': dates,
            'daily_cost': costs,
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
    
    def test_calculate_metrics_basic(self, validator):
        """Test basic metric calculation."""
        y_true = pd.Series([100, 110, 105, 120, 115])
        y_pred = pd.Series([98, 108, 107, 118, 117])
        
        metrics = validator.calculate_metrics(y_true, y_pred)
        
        assert isinstance(metrics, ValidationMetrics)
        assert metrics.mape > 0
        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert -1 <= metrics.r2 <= 1
        assert 0 <= metrics.directional_accuracy <= 1
    
    def test_calculate_metrics_with_dates(self, validator):
        """Test metric calculation with date information."""
        y_true = pd.Series([100, 110, 105, 120, 115])
        y_pred = pd.Series([98, 108, 107, 118, 117])
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        
        metrics = validator.calculate_metrics(y_true, y_pred, dates)
        
        assert isinstance(metrics, ValidationMetrics)
        assert 0 <= metrics.directional_accuracy <= 1
    
    def test_calculate_metrics_length_mismatch(self, validator):
        """Test error handling for length mismatch."""
        y_true = pd.Series([100, 110, 105])
        y_pred = pd.Series([98, 108])
        
        with pytest.raises(ForecastValidationError):
            validator.calculate_metrics(y_true, y_pred)
    
    def test_calculate_metrics_nan_values(self, validator):
        """Test handling of NaN values."""
        y_true = pd.Series([100, np.nan, 105, 120])
        y_pred = pd.Series([98, 108, np.nan, 118])
        
        metrics = validator.calculate_metrics(y_true, y_pred)
        
        # Should only use the 100/98 and 120/118 pairs
        assert isinstance(metrics, ValidationMetrics)
    
    def test_calculate_metrics_all_nan(self, validator):
        """Test error handling when all values are NaN."""
        y_true = pd.Series([np.nan, np.nan, np.nan])
        y_pred = pd.Series([np.nan, np.nan, np.nan])
        
        with pytest.raises(ForecastValidationError):
            validator.calculate_metrics(y_true, y_pred)
    
    def test_directional_accuracy_perfect(self, validator):
        """Test perfect directional accuracy."""
        # Both increase then decrease
        y_true = pd.Series([100, 110, 105])
        y_pred = pd.Series([98, 108, 103])
        
        accuracy = validator._calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_directional_accuracy_opposite(self, validator):
        """Test opposite directional trends."""
        # True increases, predicted decreases
        y_true = pd.Series([100, 110, 120])
        y_pred = pd.Series([120, 110, 100])
        
        accuracy = validator._calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_directional_accuracy_short_series(self, validator):
        """Test directional accuracy with short series."""
        y_true = pd.Series([100])
        y_pred = pd.Series([98])
        
        accuracy = validator._calculate_directional_accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_generate_test_periods(self, validator, sample_data):
        """Test generation of test periods."""
        test_periods = validator._generate_test_periods(sample_data, forecast_horizon=7)
        
        assert len(test_periods) > 0
        assert all(isinstance(period, tuple) for period in test_periods)
        assert all(len(period) == 2 for period in test_periods)
    
    def test_split_data_for_period(self, validator, sample_data):
        """Test data splitting for backtesting."""
        start_date = "2024-02-01"
        end_date = "2024-02-07"
        
        train_data, test_data = validator._split_data_for_period(
            sample_data, start_date, end_date, forecast_horizon=7
        )
        
        assert len(train_data) > 0
        assert len(test_data) > 0
        assert train_data['usage_date'].max() < pd.to_datetime(start_date)
        assert test_data['usage_date'].min() >= pd.to_datetime(start_date)
        assert test_data['usage_date'].max() <= pd.to_datetime(end_date)
    
    def test_validate_against_quality_gates_pass(self, validator):
        """Test quality gate validation - passing case."""
        metrics = ValidationMetrics(
            mape=10.0,    # Below 15% threshold
            rmse=500.0,   # Below 1000 threshold
            mae=300.0,
            r2=0.8,       # Above 0.6 threshold
            directional_accuracy=0.7,  # Above 0.65 threshold
            bias=10.0
        )
        
        result = validator.validate_against_quality_gates(metrics)
        
        assert result['passed'] is True
        assert len(result['failures']) == 0
        assert result['metrics'] == metrics.to_dict()
    
    def test_validate_against_quality_gates_fail(self, validator):
        """Test quality gate validation - failing case."""
        metrics = ValidationMetrics(
            mape=20.0,    # Above 15% threshold
            rmse=1500.0,  # Above 1000 threshold
            mae=800.0,
            r2=0.4,       # Below 0.6 threshold
            directional_accuracy=0.5,  # Below 0.65 threshold
            bias=100.0
        )
        
        result = validator.validate_against_quality_gates(metrics)
        
        assert result['passed'] is False
        assert len(result['failures']) == 4  # All gates should fail
        
        # Check specific failures
        failure_metrics = [f['metric'] for f in result['failures']]
        assert 'mape' in failure_metrics
        assert 'rmse' in failure_metrics
        assert 'r2' in failure_metrics
        assert 'directional_accuracy' in failure_metrics
    
    def test_summarize_backtest_results(self, validator):
        """Test backtesting results summarization."""
        # Create mock backtest results
        metrics1 = ValidationMetrics(5.0, 100.0, 75.0, 0.8, 0.7, 5.0)
        metrics2 = ValidationMetrics(7.0, 120.0, 85.0, 0.75, 0.65, 8.0)
        
        predictions1 = pd.DataFrame({'actual': [100], 'predicted': [95]})
        predictions2 = pd.DataFrame({'actual': [110], 'predicted': [103]})
        
        results = [
            BacktestResult("Period 1", metrics1, predictions1),
            BacktestResult("Period 2", metrics2, predictions2)
        ]
        
        summary = validator._summarize_backtest_results(results)
        
        assert summary['n_periods'] == 2
        assert summary['mean_mape'] == 6.0  # (5.0 + 7.0) / 2
        assert summary['mean_rmse'] == 110.0  # (100.0 + 120.0) / 2
        assert summary['std_mape'] == 1.0  # std([5.0, 7.0])
    
    def test_aggregate_cv_results(self, validator):
        """Test cross-validation results aggregation."""
        fold_results = [
            {
                'fold': 1,
                'train_size': 70,
                'test_size': 30,
                'metrics': {'mape': 5.0, 'rmse': 100.0}
            },
            {
                'fold': 2,
                'train_size': 70,
                'test_size': 30,
                'metrics': {'mape': 7.0, 'rmse': 120.0}
            }
        ]
        
        cv_results = validator._aggregate_cv_results(fold_results)
        
        assert cv_results['n_folds'] == 2
        assert cv_results['statistics']['mean_mape'] == 6.0
        assert cv_results['statistics']['mean_rmse'] == 110.0
        assert cv_results['statistics']['std_mape'] == 1.0
    
    def test_generate_validation_recommendations(self, validator):
        """Test validation recommendation generation."""
        # Create results with various metric levels
        high_mape_metrics = ValidationMetrics(25.0, 500.0, 300.0, 0.8, 0.7, 5.0)
        low_r2_metrics = ValidationMetrics(10.0, 500.0, 300.0, 0.3, 0.7, 5.0)
        high_bias_metrics = ValidationMetrics(10.0, 500.0, 300.0, 0.8, 0.7, 200.0)
        
        predictions = pd.DataFrame({'actual': [100], 'predicted': [95]})
        
        results = [
            BacktestResult("Period 1", high_mape_metrics, predictions),
            BacktestResult("Period 2", low_r2_metrics, predictions),
            BacktestResult("Period 3", high_bias_metrics, predictions)
        ]
        
        recommendations = validator._generate_validation_recommendations(results, None)
        
        assert len(recommendations) > 0
        
        # Check for specific recommendation types
        rec_text = ' '.join(recommendations)
        assert 'MAPE' in rec_text or 'mape' in rec_text
        assert 'R-squared' in rec_text or 'explanatory power' in rec_text
        assert 'bias' in rec_text or 'Bias' in rec_text
    
    @patch('src.forecaster.validation.validator.ModelValidator._generate_backtest_predictions')
    def test_backtest_model_basic(self, mock_predictions, validator, sample_data):
        """Test basic backtesting functionality."""
        # Mock model
        mock_model = Mock()
        
        # Mock predictions
        mock_predictions.return_value = pd.DataFrame({
            'usage_date': sample_data['usage_date'][:7],
            'actual': sample_data['daily_cost'][:7],
            'predicted': sample_data['daily_cost'][:7] * 0.95
        })
        
        # Run backtest with a single test period
        test_periods = [("2024-01-01", "2024-01-07")]
        
        results = validator.backtest_model(
            model=mock_model,
            data=sample_data,
            target_column='daily_cost',
            test_periods=test_periods,
            forecast_horizon=7
        )
        
        assert len(results) == 1
        assert isinstance(results[0], BacktestResult)
        assert results[0].test_period == "2024-01-01 to 2024-01-07"
    
    def test_generate_validation_report(self, validator):
        """Test validation report generation."""
        # Create mock backtest results
        metrics = ValidationMetrics(5.0, 100.0, 75.0, 0.8, 0.7, 5.0)
        predictions = pd.DataFrame({'actual': [100], 'predicted': [95]})
        results = [BacktestResult("Period 1", metrics, predictions)]
        
        # Mock CV results
        cv_results = {
            'n_folds': 3,
            'statistics': {'mean_mape': 6.0, 'std_mape': 1.0}
        }
        
        report = validator.generate_validation_report(
            backtest_results=results,
            cv_results=cv_results
        )
        
        assert 'timestamp' in report
        assert 'backtest_summary' in report
        assert 'cross_validation' in report
        assert 'quality_gates' in report
        assert 'recommendations' in report
        assert 'overall_quality_check' in report
    
    def test_extract_feature_importance_sklearn(self, validator):
        """Test feature importance extraction for sklearn models."""
        # Mock sklearn model
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.6, 0.4])
        mock_model.feature_names_in_ = ['feature1', 'feature2']
        
        importance = validator._extract_feature_importance(mock_model)
        
        expected = {'feature1': 0.6, 'feature2': 0.4}
        assert importance == expected
    
    def test_extract_feature_importance_linear(self, validator):
        """Test feature importance extraction for linear models."""
        # Mock linear model
        mock_model = Mock()
        mock_model.coef_ = np.array([0.8, -0.6])
        mock_model.feature_names_in_ = ['feature1', 'feature2']
        # Remove feature_importances_ attribute
        if hasattr(mock_model, 'feature_importances_'):
            delattr(mock_model, 'feature_importances_')
        
        importance = validator._extract_feature_importance(mock_model)
        
        expected = {'feature1': 0.8, 'feature2': 0.6}  # Absolute values
        assert importance == expected
    
    def test_extract_feature_importance_no_support(self, validator):
        """Test feature importance extraction when not supported."""
        # Mock model without feature importance
        mock_model = Mock()
        
        importance = validator._extract_feature_importance(mock_model)
        
        assert importance is None


@pytest.mark.integration
class TestModelValidatorIntegration:
    """Integration tests for ModelValidator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ForecasterConfig(
            environment="dev",
            data_source={
                "cur_bucket": "test-cur-bucket",
                "athena_output_bucket": "test-athena-bucket"
            },
            infrastructure={
                "model_bucket": "test-model-bucket",
                "data_bucket": "test-data-bucket"
            }
        )
    
    @pytest.fixture
    def validator(self, config):
        """Create model validator instance."""
        return ModelValidator(config)
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic time series data with trends and seasonality."""
        dates = pd.date_range('2023-01-01', '2024-03-31', freq='D')
        n_days = len(dates)
        
        # Create realistic cost pattern
        np.random.seed(42)
        
        # Base trend (slight increase over time)
        trend = 1000 + np.linspace(0, 200, n_days)
        
        # Weekly seasonality (lower on weekends)
        weekly_pattern = 50 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        
        # Monthly seasonality (higher at month end)
        monthly_pattern = 30 * np.sin(2 * np.pi * np.arange(n_days) / 30)
        
        # Random noise
        noise = np.random.normal(0, 25, n_days)
        
        # Combine components
        daily_cost = trend + weekly_pattern + monthly_pattern + noise
        
        # Ensure no negative costs
        daily_cost = np.maximum(daily_cost, 50)
        
        return pd.DataFrame({
            'usage_date': dates,
            'daily_cost': daily_cost,
            'day_of_week': dates.dayofweek,
            'is_weekend': dates.dayofweek >= 5,
            'month': dates.month
        })
    
    def test_end_to_end_validation_workflow(self, validator, realistic_data):
        """Test complete validation workflow with realistic data."""
        # Mock model that predicts based on simple trend
        class MockForecastModel:
            def __init__(self, config):
                self.config = config
                self.trend_coef = None
            
            def fit(self, data):
                # Simple trend-based model
                X = np.arange(len(data)).reshape(-1, 1)
                y = data['daily_cost'].values
                self.trend_coef = np.polyfit(X.flatten(), y, 1)[0] if len(data) > 1 else 0.0
            
            def predict(self, start_date, end_date):
                # Generate predictions based on trend
                pred_dates = pd.date_range(start_date, end_date, freq='D')
                n_pred = len(pred_dates)
                
                # Simple linear trend prediction
                base_value = 1100  # Approximate middle value
                trend_coef = self.trend_coef if self.trend_coef is not None else 0.0
                predictions = base_value + trend_coef * np.arange(n_pred)
                
                return pd.DataFrame({
                    'usage_date': pred_dates,
                    'predicted': predictions
                })
        
        # Perform backtesting
        with patch.object(validator, '_generate_backtest_predictions') as mock_pred:
            # Mock prediction generation
            def generate_predictions(model, train_data, test_data, target_col, horizon):
                actual = test_data[target_col].values
                # Add some realistic prediction error
                predicted = actual * (0.95 + 0.1 * np.random.random(len(actual)))
                
                return pd.DataFrame({
                    'usage_date': test_data['usage_date'],
                    'actual': actual,
                    'predicted': predicted
                })
            
            mock_pred.side_effect = generate_predictions
            
            # Run backtest
            mock_model = MockForecastModel(validator.config)
            results = validator.backtest_model(
                model=mock_model,
                data=realistic_data,
                target_column='daily_cost',
                forecast_horizon=7
            )
            
            # Validate results
            assert len(results) > 0
            
            # Check metrics are reasonable
            for result in results:
                assert 0 < result.metrics.mape < 100
                assert result.metrics.rmse > 0
                assert result.metrics.mae > 0
                assert result.metrics.r2 > -1
                assert 0 <= result.metrics.directional_accuracy <= 1
        
        # Generate validation report
        report = validator.generate_validation_report(results)
        
        # Validate report structure
        assert 'timestamp' in report
        assert 'backtest_summary' in report
        assert 'quality_gates' in report
        assert 'recommendations' in report
        assert isinstance(report['recommendations'], list)
        
        # Check quality gate validation
        if 'overall_quality_check' in report:
            quality_check = report['overall_quality_check']
            assert 'passed' in quality_check
            assert 'metrics' in quality_check
            assert 'thresholds' in quality_check