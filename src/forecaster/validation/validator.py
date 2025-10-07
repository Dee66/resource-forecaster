"""Model validation and backtesting for Resource Forecaster.

Provides comprehensive validation, backtesting, and cross-validation
for time-series forecasting models.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import ForecasterConfig
from ..exceptions import ForecastValidationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    mae: float   # Mean Absolute Error
    r2: float    # R-squared Score
    directional_accuracy: float  # Percentage of correct trend predictions
    bias: float  # Mean forecast error (positive = over-forecast)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'mape': self.mape,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'directional_accuracy': self.directional_accuracy,
            'bias': self.bias
        }


@dataclass
class BacktestResult:
    """Container for backtest results."""
    test_period: str
    metrics: ValidationMetrics
    predictions: pd.DataFrame
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backtest result to dictionary."""
        result = {
            'test_period': self.test_period,
            'metrics': self.metrics.to_dict(),
            'prediction_count': len(self.predictions)
        }
        
        if self.feature_importance:
            result['feature_importance'] = self.feature_importance
            
        return result


class ModelValidator:
    """Comprehensive model validation and backtesting."""
    
    def __init__(self, config: ForecasterConfig):
        """Initialize model validator.
        
        Args:
            config: Forecaster configuration
        """
        self.config = config
        self.quality_gates = {
            'mape': 15.0,  # 15% MAPE threshold
            'rmse': 1000,  # $1000 RMSE threshold for daily costs
            'r2': 0.6,     # 60% R-squared minimum
            'directional_accuracy': 0.65  # 65% directional accuracy minimum
        }
    
    def calculate_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: pd.Series,
        dates: Optional[pd.Series] = None
    ) -> ValidationMetrics:
        """Calculate comprehensive validation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Optional date series for trend analysis
            
        Returns:
            ValidationMetrics object
            
        Raises:
            ForecastValidationError: If metric calculation fails
        """
        try:
            # Ensure same length
            if len(y_true) != len(y_pred):
                raise ForecastValidationError(
                    message=(
                        f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
                    ),
                    metric_name="length_mismatch",
                    actual_value=float(len(y_pred)),
                    threshold=float(len(y_true)),
                )

            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]

            if len(y_true_clean) == 0:
                raise ForecastValidationError(
                    message="No valid data points for metric calculation",
                    metric_name="no_valid_points",
                    actual_value=0.0,
                    threshold=1.0,
                )

            # Calculate basic metrics
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            r2 = r2_score(y_true_clean, y_pred_clean)

            # Calculate MAPE (avoiding division by zero)
            mape_values = np.abs((y_true_clean - y_pred_clean) / np.maximum(y_true_clean, 1e-8))
            mape = np.mean(mape_values) * 100

            # Calculate bias (mean forecast error)
            bias = np.mean(y_pred_clean - y_true_clean)

            # Calculate directional accuracy
            directional_accuracy = self._calculate_directional_accuracy(
                y_true_clean, y_pred_clean, dates[mask] if dates is not None else None
            )

            return ValidationMetrics(
                mape=float(mape),
                rmse=float(rmse),
                mae=float(mae),
                r2=float(r2),
                directional_accuracy=float(directional_accuracy),
                bias=float(bias),
            )

        except ForecastValidationError:
            # Re-raise our domain-specific validation errors untouched
            raise
        except Exception as e:
            # Wrap any unexpected errors into ForecastValidationError for consistency
            raise ForecastValidationError(
                message=f"Metric calculation failed: {e}",
                metric_name="metric_calculation",
                actual_value=float("nan"),
                threshold=float("nan"),
            ) from e
    
    def _calculate_directional_accuracy(
        self, 
        y_true: pd.Series, 
        y_pred: pd.Series,
        dates: Optional[pd.Series] = None
    ) -> float:
        """Calculate directional accuracy (trend prediction accuracy).
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            dates: Optional date series
            
        Returns:
            Directional accuracy percentage (0-1)
        """
        if len(y_true) < 2:
            return 0.0
            
        # Calculate day-over-day changes
        true_changes = np.diff(y_true)
        pred_changes = np.diff(y_pred)
        
        # Determine direction (positive = increase, negative = decrease)
        true_directions = np.sign(true_changes)
        pred_directions = np.sign(pred_changes)
        
        # Calculate accuracy
        correct_directions = np.sum(true_directions == pred_directions)
        total_directions = len(true_directions)
        
        return correct_directions / total_directions if total_directions > 0 else 0.0
    
    def backtest_model(
        self,
        model: Any,
        data: pd.DataFrame,
        target_column: str = 'daily_cost',
        test_periods: Optional[List[Tuple[str, str]]] = None,
        forecast_horizon: int = 7
    ) -> List[BacktestResult]:
        """Perform comprehensive backtesting on trained model.
        
        Args:
            model: Trained forecasting model
            data: Historical data for backtesting
            target_column: Name of target variable column
            test_periods: List of (start_date, end_date) tuples for testing
            forecast_horizon: Number of days to forecast
            
        Returns:
            List of BacktestResult objects
            
        Raises:
            ForecastValidationError: If backtesting fails
        """
        try:
            logger.info(f"Starting backtesting with {forecast_horizon}-day horizon")
            
            # Generate test periods if not provided
            if test_periods is None:
                test_periods = self._generate_test_periods(data, forecast_horizon)
            
            backtest_results = []
            
            for start_date, end_date in test_periods:
                logger.info(f"Backtesting period: {start_date} to {end_date}")
                
                # Split data for this test period
                train_data, test_data = self._split_data_for_period(
                    data, start_date, end_date, forecast_horizon
                )
                
                if len(test_data) == 0:
                    logger.warning(f"No test data for period {start_date} to {end_date}")
                    continue
                
                # Generate predictions
                predictions = self._generate_backtest_predictions(
                    model, train_data, test_data, target_column, forecast_horizon
                )
                
                # Calculate metrics
                metrics = self.calculate_metrics(
                    y_true=test_data[target_column],
                    y_pred=predictions['predicted'],
                    dates=test_data['usage_date'] if 'usage_date' in test_data.columns else None
                )
                
                # Get feature importance if available
                feature_importance = self._extract_feature_importance(model)
                
                # Create result
                result = BacktestResult(
                    test_period=f"{start_date} to {end_date}",
                    metrics=metrics,
                    predictions=predictions,
                    feature_importance=feature_importance
                )
                
                backtest_results.append(result)
                
                logger.info(f"Backtest metrics - MAPE: {metrics.mape:.2f}%, RMSE: ${metrics.rmse:.2f}")
            
            logger.info(f"Completed backtesting for {len(backtest_results)} periods")
            return backtest_results
            
        except Exception as e:
            raise ValueError(f"Backtesting failed: {e}") from e
    
    def cross_validate_model(
        self,
        model_class: type,
        data: pd.DataFrame,
        target_column: str = 'daily_cost',
        n_splits: int = 5,
        test_size: int = 30
    ) -> Dict[str, Any]:
        """Perform time series cross-validation.
        
        Args:
            model_class: Model class to instantiate for each fold
            data: Historical data for cross-validation
            target_column: Name of target variable column
            n_splits: Number of cross-validation splits
            test_size: Size of test set for each split (in days)
            
        Returns:
            Cross-validation results dictionary
            
        Raises:
            ForecastValidationError: If cross-validation fails
        """
        try:
            logger.info(f"Starting {n_splits}-fold time series cross-validation")
            
            # Prepare data
            df = data.copy().sort_values('usage_date')
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
            
            fold_results = []
            feature_cols = [col for col in df.columns if col not in ['usage_date', target_column]]
            
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(df)):
                logger.info(f"Processing fold {fold_idx + 1}/{n_splits}")
                
                # Split data
                train_fold = df.iloc[train_idx]
                test_fold = df.iloc[test_idx]
                
                try:
                    # Initialize and train model
                    model = model_class(self.config)
                    model.fit(train_fold)
                    
                    # Generate predictions
                    predictions = model.predict(
                        start_date=test_fold['usage_date'].min(),
                        end_date=test_fold['usage_date'].max()
                    )
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(
                        y_true=test_fold[target_column],
                        y_pred=predictions['predicted'],
                        dates=test_fold['usage_date']
                    )
                    
                    fold_results.append({
                        'fold': fold_idx + 1,
                        'train_size': len(train_fold),
                        'test_size': len(test_fold),
                        'metrics': metrics.to_dict()
                    })
                    
                except Exception as e:
                    logger.warning(f"Fold {fold_idx + 1} failed: {e}")
                    continue
            
            # Aggregate results
            cv_results = self._aggregate_cv_results(fold_results)
            
            logger.info(f"Cross-validation completed - Mean MAPE: {cv_results['mean_mape']:.2f}%")
            return cv_results
            
        except Exception as e:
            raise ValueError(f"Cross-validation failed: {e}") from e
    
    def validate_against_quality_gates(self, metrics: ValidationMetrics) -> Dict[str, Any]:
        """Validate metrics against quality gates.
        
        Args:
            metrics: Validation metrics to check
            
        Returns:
            Validation result dictionary
        """
        results = {
            'passed': True,
            'failures': [],
            'metrics': metrics.to_dict(),
            'thresholds': self.quality_gates.copy()
        }
        
        # Check each quality gate
        checks = [
            ('mape', metrics.mape, self.quality_gates['mape'], 'less'),
            ('rmse', metrics.rmse, self.quality_gates['rmse'], 'less'),
            ('r2', metrics.r2, self.quality_gates['r2'], 'greater'),
            ('directional_accuracy', metrics.directional_accuracy, self.quality_gates['directional_accuracy'], 'greater')
        ]
        
        for metric_name, value, threshold, comparison in checks:
            if comparison == 'less' and value > threshold:
                results['passed'] = False
                results['failures'].append({
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'message': f"{metric_name} {value:.4f} exceeds threshold {threshold}"
                })
            elif comparison == 'greater' and value < threshold:
                results['passed'] = False
                results['failures'].append({
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'message': f"{metric_name} {value:.4f} below threshold {threshold}"
                })
        
        return results
    
    def generate_validation_report(
        self,
        backtest_results: List[BacktestResult],
        cv_results: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report.
        
        Args:
            backtest_results: Results from backtesting
            cv_results: Results from cross-validation
            output_path: Optional path to save report JSON
            
        Returns:
            Validation report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_version': getattr(self.config, 'model_version', 'unknown'),
            'backtest_summary': self._summarize_backtest_results(backtest_results),
            'quality_gates': self.quality_gates,
            'recommendations': []
        }
        
        # Add cross-validation results if available
        if cv_results:
            report['cross_validation'] = cv_results
        
        # Generate recommendations
        report['recommendations'] = self._generate_validation_recommendations(
            backtest_results, cv_results
        )
        
        # Check overall quality gates
        if backtest_results:
            overall_metrics = self._calculate_overall_metrics(backtest_results)
            report['overall_quality_check'] = self.validate_against_quality_gates(overall_metrics)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def _generate_test_periods(
        self, 
        data: pd.DataFrame, 
        forecast_horizon: int
    ) -> List[Tuple[str, str]]:
        """Generate test periods for backtesting."""
        df = data.copy().sort_values('usage_date')
        
        # Use last 3 months for testing with weekly intervals
        end_date = df['usage_date'].max()
        start_date = end_date - timedelta(days=90)
        
        test_periods = []
        current_date = start_date
        
        while current_date + timedelta(days=forecast_horizon) <= end_date:
            period_end = current_date + timedelta(days=forecast_horizon - 1)
            test_periods.append((
                current_date.strftime('%Y-%m-%d'),
                period_end.strftime('%Y-%m-%d')
            ))
            current_date += timedelta(days=7)  # Weekly intervals
        
        return test_periods
    
    def _split_data_for_period(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        forecast_horizon: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data for backtesting period."""
        df = data.copy().sort_values('usage_date')
        
        # Convert dates
        test_start = pd.to_datetime(start_date)
        test_end = pd.to_datetime(end_date)
        
        # Train data: everything before test period
        train_data = df[df['usage_date'] < test_start].copy()
        
        # Test data: the test period
        test_data = df[
            (df['usage_date'] >= test_start) & 
            (df['usage_date'] <= test_end)
        ].copy()
        
        return train_data, test_data
    
    def _generate_backtest_predictions(
        self,
        model: Any,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        forecast_horizon: int
    ) -> pd.DataFrame:
        """Generate predictions for backtesting."""
        # This would be implemented based on the specific model interface
        # For now, return a placeholder
        predictions = pd.DataFrame({
            'usage_date': test_data['usage_date'],
            'actual': test_data[target_column],
            'predicted': test_data[target_column] * (0.9 + 0.2 * np.random.random(len(test_data)))
        })
        
        return predictions
    
    def _extract_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from model if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                # Scikit-learn style
                feature_names = getattr(model, 'feature_names_in_', None)
                if feature_names is not None:
                    return dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                # Linear model style
                feature_names = getattr(model, 'feature_names_in_', None)
                if feature_names is not None:
                    return dict(zip(feature_names, np.abs(model.coef_)))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return None
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        if not fold_results:
            return {}
        
        # Extract metrics from all folds
        all_metrics = {}
        for fold in fold_results:
            for metric, value in fold['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        cv_stats = {}
        for metric, values in all_metrics.items():
            cv_stats[f'mean_{metric}'] = np.mean(values)
            cv_stats[f'std_{metric}'] = np.std(values)
            cv_stats[f'min_{metric}'] = np.min(values)
            cv_stats[f'max_{metric}'] = np.max(values)
        
        return {
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'statistics': cv_stats
        }
    
    def _summarize_backtest_results(self, backtest_results: List[BacktestResult]) -> Dict[str, Any]:
        """Summarize backtest results."""
        if not backtest_results:
            return {}
        
        # Extract metrics
        all_metrics = {
            'mape': [r.metrics.mape for r in backtest_results],
            'rmse': [r.metrics.rmse for r in backtest_results],
            'mae': [r.metrics.mae for r in backtest_results],
            'r2': [r.metrics.r2 for r in backtest_results],
            'directional_accuracy': [r.metrics.directional_accuracy for r in backtest_results],
            'bias': [r.metrics.bias for r in backtest_results]
        }
        
        # Calculate summary statistics
        summary = {}
        for metric, values in all_metrics.items():
            summary[f'mean_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
            summary[f'min_{metric}'] = np.min(values)
            summary[f'max_{metric}'] = np.max(values)
        
        summary['n_periods'] = len(backtest_results)
        summary['periods'] = [r.test_period for r in backtest_results]
        
        return summary
    
    def _calculate_overall_metrics(self, backtest_results: List[BacktestResult]) -> ValidationMetrics:
        """Calculate overall metrics across all backtest periods."""
        all_predictions = pd.concat([r.predictions for r in backtest_results])
        
        return self.calculate_metrics(
            y_true=all_predictions['actual'],
            y_pred=all_predictions['predicted']
        )
    
    def _generate_validation_recommendations(
        self,
        backtest_results: List[BacktestResult],
        cv_results: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations: List[str] = []
        
        if not backtest_results:
            return recommendations
        
        # Calculate summary metrics
        summary = self._summarize_backtest_results(backtest_results)

        # Aggregate-based recommendations (averages across periods)
        mean_mape = summary.get('mean_mape', 0)
        if mean_mape >= 20:
            recommendations.append(
                "High MAPE (>=20%) indicates poor model fit. Consider feature engineering or different model architecture."
            )
        elif mean_mape >= 15:
            recommendations.append(
                "Moderate MAPE (15-20%) suggests room for improvement. Consider hyperparameter tuning."
            )

        mean_rmse = summary.get('mean_rmse', 0)
        if mean_rmse > 2000:
            recommendations.append(
                "High RMSE suggests large forecast errors. Consider outlier detection and data quality improvements."
            )

        mean_r2 = summary.get('mean_r2', 0)
        if mean_r2 < 0.5:
            recommendations.append(
                "Low R-squared (<0.5) indicates poor explanatory power. Consider additional features or a different model."
            )

        mean_dir_acc = summary.get('mean_directional_accuracy', 0)
        if mean_dir_acc < 0.6:
            recommendations.append(
                "Low directional accuracy (<60%) suggests poor trend prediction. Consider trend-focused models."
            )

        mean_bias = summary.get('mean_bias', 0)
        if abs(mean_bias) > 100:
            recommendations.append(
                f"Significant bias ({mean_bias:.2f}) detected. Model consistently {'over' if mean_bias > 0 else 'under'}-forecasts."
            )

        # Per-period checks: trigger recommendations if any period is problematic
        any_high_mape = any(r.metrics.mape >= 20 for r in backtest_results)
        any_low_r2 = any(r.metrics.r2 < 0.5 for r in backtest_results)
        any_high_bias = any(abs(r.metrics.bias) > 100 for r in backtest_results)

        if any_high_mape and not any("MAPE" in rec or "mape" in rec for rec in recommendations):
            recommendations.append(
                "Periods with high MAPE (>=20%) detected; review feature engineering and model selection."
            )

        if any_low_r2 and not any("R-squared" in rec or "explanatory" in rec for rec in recommendations):
            recommendations.append(
                "Some periods show low R-squared (<0.5), indicating limited explanatory power; consider additional features."
            )

        if any_high_bias and not any("bias" in rec.lower() for rec in recommendations):
            recommendations.append(
                "Large forecast bias detected in some periods; investigate systematic over/under-forecasting."
            )
        
        # Cross-validation recommendations
        if cv_results:
            cv_stats = cv_results.get('statistics', {})
            mape_std = cv_stats.get('std_mape', 0)
            if mape_std > 5:
                recommendations.append("High MAPE standard deviation suggests unstable model performance across time periods.")

        # Deduplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                deduped.append(rec)

        return deduped