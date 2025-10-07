"""Data quality validators for Resource Forecaster.

Provides validation checks for cost data quality and consistency.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import ForecasterConfig
from ..exceptions import ForecastValidationError

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validates data quality for cost forecasting."""
    
    def __init__(self, config: ForecasterConfig):
        """Initialize data quality validator.
        
        Args:
            config: Forecaster configuration
        """
        self.config = config
    
    def validate_cost_data(
        self, 
        df: pd.DataFrame,
        target_column: str = 'daily_cost'
    ) -> Dict[str, Any]:
        """Validate cost data quality.
        
        Args:
            df: DataFrame to validate
            target_column: Cost column to validate
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ForecastValidationError: If critical validation failures occur
        """
        logger.info(f"Validating {len(df)} cost records")
        
        validation_results = {
            'total_records': len(df),
            'validation_passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Check for negative costs
            negative_costs = self._check_negative_costs(df, target_column)
            if negative_costs['count'] > 0:
                validation_results['errors'].append(negative_costs)
                validation_results['validation_passed'] = False
            
            # Check for missing timestamps
            missing_timestamps = self._check_missing_timestamps(df)
            if missing_timestamps['count'] > 0:
                validation_results['warnings'].append(missing_timestamps)
            
            # Check for data gaps
            data_gaps = self._check_data_gaps(df)
            if data_gaps['count'] > 0:
                validation_results['warnings'].append(data_gaps)
            
            # Check for outliers
            outliers = self._check_outliers(df, target_column)
            validation_results['metrics']['outliers'] = outliers
            
            # Check data completeness
            completeness = self._check_completeness(df)
            validation_results['metrics']['completeness'] = completeness
            
            # Check temporal consistency
            temporal_consistency = self._check_temporal_consistency(df)
            validation_results['metrics']['temporal_consistency'] = temporal_consistency
            
            logger.info(
                f"Validation completed: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}"
            )
            
            return validation_results
            
        except Exception as e:
            raise ForecastValidationError(
                f"Data validation failed: {e}",
                metric_name="data_quality",
                actual_value=0.0,
                threshold=1.0
            ) from e
    
    def _check_negative_costs(
        self, 
        df: pd.DataFrame, 
        cost_column: str
    ) -> Dict[str, Any]:
        """Check for negative cost values."""
        if cost_column not in df.columns:
            return {'count': 0, 'message': f'Cost column {cost_column} not found'}
        
        negative_mask = df[cost_column] < 0
        negative_count = negative_mask.sum()
        
        return {
            'check': 'negative_costs',
            'count': int(negative_count),
            'percentage': float(negative_count / len(df) * 100),
            'message': f'Found {negative_count} records with negative costs',
            'severity': 'error' if negative_count > 0 else 'info'
        }
    
    def _check_missing_timestamps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing timestamp values."""
        date_columns = ['usage_date', 'timestamp', 'date']
        date_column = None
        
        for col in date_columns:
            if col in df.columns:
                date_column = col
                break
        
        if date_column is None:
            return {
                'check': 'missing_timestamps',
                'count': len(df),
                'message': 'No timestamp column found',
                'severity': 'warning'
            }
        
        missing_count = df[date_column].isna().sum()
        
        return {
            'check': 'missing_timestamps',
            'count': int(missing_count),
            'percentage': float(missing_count / len(df) * 100),
            'message': f'Found {missing_count} records with missing timestamps',
            'severity': 'warning' if missing_count > 0 else 'info'
        }
    
    def _check_data_gaps(
        self, 
        df: pd.DataFrame,
        date_column: str = 'usage_date'
    ) -> Dict[str, Any]:
        """Check for gaps in daily data."""
        if date_column not in df.columns:
            return {'count': 0, 'message': f'Date column {date_column} not found'}
        
        # Convert to datetime
        dates = pd.to_datetime(df[date_column]).dt.date
        unique_dates = sorted(dates.unique())
        
        if len(unique_dates) < 2:
            return {'count': 0, 'message': 'Insufficient data for gap analysis'}
        
        # Check for gaps
        date_range = pd.date_range(
            start=unique_dates[0], 
            end=unique_dates[-1], 
            freq='D'
        ).date
        
        missing_dates = set(date_range) - set(unique_dates)
        gap_count = len(missing_dates)
        
        return {
            'check': 'data_gaps',
            'count': gap_count,
            'percentage': float(gap_count / len(date_range) * 100),
            'message': f'Found {gap_count} missing days in date range',
            'missing_dates': sorted(list(missing_dates))[:10],  # First 10
            'severity': 'warning' if gap_count > 0 else 'info'
        }
    
    def _check_outliers(
        self, 
        df: pd.DataFrame, 
        cost_column: str
    ) -> Dict[str, Any]:
        """Check for statistical outliers in cost data."""
        if cost_column not in df.columns:
            return {'count': 0, 'message': f'Cost column {cost_column} not found'}
        
        # IQR method
        Q1 = df[cost_column].quantile(0.25)
        Q3 = df[cost_column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (
            (df[cost_column] < lower_bound) | 
            (df[cost_column] > upper_bound)
        )
        
        outlier_count = outliers_mask.sum()
        
        return {
            'check': 'outliers',
            'method': 'IQR',
            'count': int(outlier_count),
            'percentage': float(outlier_count / len(df) * 100),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'q1': float(Q1),
            'q3': float(Q3),
            'severity': 'info'
        }
    
    def _check_completeness(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check data completeness across columns."""
        completeness = {}
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            completeness[column] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_count / len(df) * 100),
                'completeness_percentage': float((len(df) - missing_count) / len(df) * 100)
            }
        
        return completeness
    
    def _check_temporal_consistency(
        self, 
        df: pd.DataFrame,
        date_column: str = 'usage_date'
    ) -> Dict[str, Any]:
        """Check temporal consistency of the data."""
        if date_column not in df.columns:
            return {'status': 'no_date_column'}
        
        try:
            dates = pd.to_datetime(df[date_column])
            
            return {
                'status': 'valid',
                'date_range': {
                    'start': dates.min().isoformat(),
                    'end': dates.max().isoformat(),
                    'span_days': (dates.max() - dates.min()).days
                },
                'unique_dates': int(dates.dt.date.nunique()),
                'total_records': len(df),
                'records_per_day_avg': float(len(df) / dates.dt.date.nunique()) if dates.dt.date.nunique() > 0 else 0
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Temporal consistency check failed: {e}'
            }