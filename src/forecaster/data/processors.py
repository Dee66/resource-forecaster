"""Data processors for Resource Forecaster.

Provides data preprocessing, feature engineering, and validation
for cost forecasting models.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ..config import ForecasterConfig, FeatureEngineeringConfig
from ..exceptions import DataSourceError, ForecastValidationError

logger = logging.getLogger(__name__)


class CostDataProcessor:
    """Processes raw cost data for time-series forecasting."""
    
    def __init__(self, config: ForecasterConfig):
        """Initialize cost data processor.
        
        Args:
            config: Forecaster configuration
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
    
    def preprocess_cost_data(
        self, 
        raw_data: pd.DataFrame,
        target_column: str = 'daily_cost'
    ) -> pd.DataFrame:
        """Preprocess raw cost data for modeling.
        
        Args:
            raw_data: Raw cost data from collectors
            target_column: Name of the cost column to predict
            
        Returns:
            Preprocessed DataFrame ready for modeling
            
        Raises:
            DataSourceError: If preprocessing fails
        """
        try:
            logger.info(f"Preprocessing {len(raw_data)} cost records")
            
            df = raw_data.copy()
            
            # Validate required columns
            required_cols = ['usage_date', target_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise DataSourceError(
                    f"Missing required columns: {missing_cols}",
                    source_type="preprocessing"
                )
            
            # Convert date column
            df['usage_date'] = pd.to_datetime(df['usage_date'])
            
            # Remove outliers
            df = self._remove_cost_outliers(df, target_column)
            
            # Handle missing values
            df = self._handle_missing_values(df, target_column)
            
            # Aggregate to daily level if needed
            df = self._aggregate_daily_costs(df, target_column)
            
            # Process FinOps tagging metadata
            df = self._process_finops_tags(df)
            
            # Sort by date
            df = df.sort_values('usage_date').reset_index(drop=True)
            
            logger.info(f"Preprocessed data shape: {df.shape}")
            return df
            
        except Exception as e:
            raise DataSourceError(
                f"Cost data preprocessing failed: {e}",
                source_type="preprocessing"
            ) from e
    
    def _remove_cost_outliers(
        self, 
        df: pd.DataFrame, 
        cost_column: str,
        method: str = 'iqr'
    ) -> pd.DataFrame:
        """Remove cost outliers using IQR method."""
        if method == 'iqr':
            Q1 = df[cost_column].quantile(0.25)
            Q3 = df[cost_column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (
                (df[cost_column] < lower_bound) | 
                (df[cost_column] > upper_bound)
            )
            
            outliers_count = outliers_mask.sum()
            if outliers_count > 0:
                logger.info(f"Removing {outliers_count} cost outliers")
                df = df[~outliers_mask].copy()
        
        return df
    
    def _handle_missing_values(
        self, 
        df: pd.DataFrame, 
        cost_column: str
    ) -> pd.DataFrame:
        """Handle missing values in cost data."""
        # Remove rows with missing costs
        before_count = len(df)
        df = df.dropna(subset=[cost_column])
        after_count = len(df)
        
        if before_count != after_count:
            logger.info(f"Removed {before_count - after_count} rows with missing costs")
        
        # Fill missing categorical values
        categorical_cols = ['cost_center', 'project', 'environment', 'service']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def _aggregate_daily_costs(
        self, 
        df: pd.DataFrame, 
        cost_column: str
    ) -> pd.DataFrame:
        """Aggregate costs to daily level."""
        # Group by date and sum costs
        daily_df = df.groupby('usage_date').agg({
            cost_column: 'sum',
            'cost_center': lambda x: x.mode().iloc[0] if not x.mode().empty else 'mixed',
            'project': lambda x: x.mode().iloc[0] if not x.mode().empty else 'mixed',
            'environment': lambda x: x.mode().iloc[0] if not x.mode().empty else 'mixed'
        }).reset_index()
        
        return daily_df


class FeatureEngineer:
    """Creates time-series features for cost forecasting."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        """Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
    
    def create_features(
        self, 
        df: pd.DataFrame,
        date_column: str = 'usage_date',
        target_column: str = 'daily_cost'
    ) -> pd.DataFrame:
        """Create time-series features for forecasting.
        
        Args:
            df: Input DataFrame with cost data
            date_column: Name of the date column
            target_column: Name of the target cost column
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            DataSourceError: If feature engineering fails
        """
        try:
            logger.info("Creating time-series features")
            
            features_df = df.copy()
            
            # Ensure date column is datetime
            features_df[date_column] = pd.to_datetime(features_df[date_column])
            
            # Time-based features
            if self.config.include_day_of_week:
                features_df['day_of_week'] = features_df[date_column].dt.dayofweek
                features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
            
            if self.config.include_month:
                features_df['month'] = features_df[date_column].dt.month
                features_df['quarter'] = features_df[date_column].dt.quarter
            
            if self.config.include_quarter:
                features_df['quarter'] = features_df[date_column].dt.quarter
            
            # Holiday features
            if self.config.include_holidays:
                features_df = self._add_holiday_features(features_df, date_column)
            
            # Lag features
            features_df = self._add_lag_features(
                features_df, target_column, self.config.lag_days
            )
            
            # Rolling window features
            features_df = self._add_rolling_features(
                features_df, target_column, self.config.rolling_windows
            )
            
            # FinOps tag features
            features_df = self._encode_categorical_features(features_df)
            
            logger.info(f"Created features: {list(features_df.columns)}")
            return features_df
            
        except Exception as e:
            raise DataSourceError(
                f"Feature engineering failed: {e}",
                source_type="feature_engineering"
            ) from e
    
    def _add_holiday_features(
        self, 
        df: pd.DataFrame, 
        date_column: str
    ) -> pd.DataFrame:
        """Add holiday indicator features."""
        try:
            import holidays
            
            # US holidays (can be configurable)
            us_holidays = holidays.UnitedStates()
            
            df['is_holiday'] = df[date_column].apply(
                lambda x: x.date() in us_holidays
            ).astype(int)
            
            # Day before/after holiday
            df['day_before_holiday'] = df[date_column].apply(
                lambda x: (x + timedelta(days=1)).date() in us_holidays
            ).astype(int)
            
            df['day_after_holiday'] = df[date_column].apply(
                lambda x: (x - timedelta(days=1)).date() in us_holidays
            ).astype(int)
            
        except ImportError:
            logger.warning("holidays package not available, skipping holiday features")
            df['is_holiday'] = 0
            df['day_before_holiday'] = 0
            df['day_after_holiday'] = 0
        
        return df
    
    def _add_lag_features(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        lag_days: List[int]
    ) -> pd.DataFrame:
        """Add lagged cost features."""
        for lag in lag_days:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        return df
    
    def _add_rolling_features(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        windows: List[int]
    ) -> pd.DataFrame:
        """Add rolling window statistics."""
        for window in windows:
            # Rolling mean
            df[f'{target_column}_rolling_mean_{window}'] = (
                df[target_column].rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling standard deviation
            df[f'{target_column}_rolling_std_{window}'] = (
                df[target_column].rolling(window=window, min_periods=1).std()
            )
            
            # Rolling min/max
            df[f'{target_column}_rolling_min_{window}'] = (
                df[target_column].rolling(window=window, min_periods=1).min()
            )
            
            df[f'{target_column}_rolling_max_{window}'] = (
                df[target_column].rolling(window=window, min_periods=1).max()
            )
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical FinOps features."""
        categorical_cols = ['cost_center', 'project', 'environment']
        
        for col in categorical_cols:
            if col in df.columns:
                # One-hot encode with limited categories to avoid explosion
                top_categories = df[col].value_counts().head(10).index
                for category in top_categories:
                    df[f'{col}_{category}'] = (df[col] == category).astype(int)
        
        return df
    
    def _process_finops_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and standardize FinOps tagging metadata.
        
        Extracts cost center, project, and environment information from
        resource tags and creates normalized features for forecasting.
        
        Args:
            df: DataFrame with cost data and resource tags
            
        Returns:
            DataFrame with FinOps tag features added
        """
        logger.info("Processing FinOps tagging metadata")
        
        # Process cost center tags
        self._process_cost_center_tags(df)
        
        # Process project tags
        self._process_project_tags(df)
        
        # Process environment tags
        self._process_environment_tags(df)
        
        # Create additional FinOps features
        self._create_finops_features(df)
        
        logger.info("FinOps tagging processing completed")
        return df
    
    def _process_cost_center_tags(self, df: pd.DataFrame) -> None:
        """Process and standardize cost center tagging metadata.
        
        Extracts cost center information from resource tags and normalizes
        for consistent FinOps reporting and chargeback.
        """
        # Extract cost center from resource tags
        cost_center_columns = [
            'resource_tags_user_cost_center',
            'resource_tags_user_costcenter', 
            'resource_tags_user_cost_centre',
            'resource_tags_user_department',
            'resource_tags_user_dept'
        ]
        
        df['cost_center'] = 'untagged'
        for col in cost_center_columns:
            if col in df.columns:
                df['cost_center'] = df[col].fillna(df['cost_center'])
                break
                
        # Standardize cost center names
        cost_center_mapping = {
            'ml': 'MLOps',
            'mlops': 'MLOps', 
            'machine-learning': 'MLOps',
            'data-science': 'DataScience',
            'ds': 'DataScience',
            'analytics': 'DataScience',
            'engineering': 'Engineering',
            'eng': 'Engineering',
            'platform': 'Engineering',
            'infrastructure': 'Infrastructure',
            'infra': 'Infrastructure',
            'ops': 'Infrastructure',
            'devops': 'Infrastructure',
            'security': 'Security',
            'sec': 'Security',
            'finance': 'Finance',
            'fin': 'Finance',
            'product': 'Product',
            'prod': 'Product',
            'marketing': 'Marketing',
            'sales': 'Sales',
            'hr': 'HumanResources',
            'human-resources': 'HumanResources',
            'legal': 'Legal',
            'compliance': 'Compliance'
        }
        
        df['cost_center_normalized'] = df['cost_center'].str.lower().map(
            cost_center_mapping
        ).fillna(df['cost_center'])
        
        # Create cost center encoding for ML features
        cost_center_counts = df['cost_center_normalized'].value_counts()
        df['cost_center_frequency'] = df['cost_center_normalized'].map(cost_center_counts)
        
        # Flag high-spend cost centers for special attention
        if 'daily_cost' in df.columns:
            cost_center_spend = df.groupby('cost_center_normalized')['daily_cost'].sum()
        elif 'cost' in df.columns:
            cost_center_spend = df.groupby('cost_center_normalized')['cost'].sum()
        else:
            # Use the target column
            cost_columns = [col for col in df.columns if 'cost' in col.lower()]
            if cost_columns:
                cost_center_spend = df.groupby('cost_center_normalized')[cost_columns[0]].sum()
            else:
                cost_center_spend = pd.Series(dtype=float)
        
        if not cost_center_spend.empty:
            high_spend_threshold = cost_center_spend.quantile(0.8)
            high_spend_centers = cost_center_spend[cost_center_spend >= high_spend_threshold].index
            df['is_high_spend_cost_center'] = df['cost_center_normalized'].isin(high_spend_centers)
        else:
            df['is_high_spend_cost_center'] = False
        
        logger.info(f"Processed {df['cost_center_normalized'].nunique()} unique cost centers")
    
    def _process_project_tags(self, df: pd.DataFrame) -> None:
        """Process and standardize project tagging metadata.
        
        Extracts project information for project-level cost allocation
        and forecasting.
        """
        # Extract project from resource tags
        project_columns = [
            'resource_tags_user_project',
            'resource_tags_user_project_name', 
            'resource_tags_user_application',
            'resource_tags_user_app',
            'resource_tags_user_service',
            'resource_tags_user_workload'
        ]
        
        df['project'] = 'untagged'
        for col in project_columns:
            if col in df.columns:
                df['project'] = df[col].fillna(df['project'])
                break
                
        # Clean and standardize project names
        df['project'] = df['project'].str.lower().str.replace('[^a-z0-9-]', '-', regex=True)
        df['project'] = df['project'].str.strip('-')
        
        # Create project lifecycle indicators
        if 'usage_date' in df.columns:
            project_history = df.groupby('project')['usage_date'].agg(['min', 'max', 'count'])
            project_history['days_active'] = (project_history['max'] - project_history['min']).dt.days
            project_history['is_established'] = (
                (project_history['days_active'] > 30) & 
                (project_history['count'] > 10)
            )
            
            df = df.merge(
                project_history[['is_established']].reset_index(),
                on='project',
                how='left'
            )
        else:
            df['is_established'] = True
        
        # Create project spend categories
        if 'daily_cost' in df.columns:
            project_spend = df.groupby('project')['daily_cost'].sum().sort_values(ascending=False)
        elif 'cost' in df.columns:
            project_spend = df.groupby('project')['cost'].sum().sort_values(ascending=False)
        else:
            # Use the target column
            cost_columns = [col for col in df.columns if 'cost' in col.lower()]
            if cost_columns:
                project_spend = df.groupby('project')[cost_columns[0]].sum().sort_values(ascending=False)
            else:
                project_spend = pd.Series(dtype=float)
        
        if not project_spend.empty:
            df['project_spend_rank'] = df['project'].map(
                dict(zip(project_spend.index, range(1, len(project_spend) + 1)))
            )
            
            # Categorize projects by spend level
            project_spend_percentiles = project_spend.quantile([0.8, 0.95])
            df['project_spend_category'] = 'low'
            df.loc[df['project'].isin(
                project_spend[project_spend >= project_spend_percentiles[0.8]].index
            ), 'project_spend_category'] = 'medium'
            df.loc[df['project'].isin(
                project_spend[project_spend >= project_spend_percentiles[0.95]].index  
            ), 'project_spend_category'] = 'high'
        else:
            df['project_spend_rank'] = 1
            df['project_spend_category'] = 'low'
        
        logger.info(f"Processed {df['project'].nunique()} unique projects")
    
    def _process_environment_tags(self, df: pd.DataFrame) -> None:
        """Process and standardize environment tagging metadata.
        
        Extracts environment information for environment-specific
        cost optimization and forecasting.
        """
        # Extract environment from resource tags
        env_columns = [
            'resource_tags_user_environment',
            'resource_tags_user_env',
            'resource_tags_user_stage',
            'resource_tags_user_tier',
            'resource_tags_user_lifecycle'
        ]
        
        df['environment'] = 'unknown'
        for col in env_columns:
            if col in df.columns:
                df['environment'] = df[col].fillna(df['environment'])
                break
                
        # Standardize environment names
        env_mapping = {
            'prod': 'production',
            'production': 'production',
            'prd': 'production',
            'live': 'production',
            'main': 'production',
            'staging': 'staging', 
            'stage': 'staging',
            'stg': 'staging',
            'uat': 'staging',
            'preprod': 'staging',
            'dev': 'development',
            'development': 'development',
            'develop': 'development',
            'test': 'testing',
            'testing': 'testing',
            'qa': 'testing',
            'quality': 'testing',
            'sandbox': 'sandbox',
            'sb': 'sandbox',
            'experimental': 'sandbox',
            'demo': 'demo',
            'training': 'training',
            'research': 'research'
        }
        
        df['environment_normalized'] = df['environment'].str.lower().map(
            env_mapping
        ).fillna('unknown')
        
        # Create environment-specific features
        env_priorities = {
            'production': 1,
            'staging': 2, 
            'testing': 3,
            'development': 4,
            'sandbox': 5,
            'demo': 6,
            'training': 7,
            'research': 8,
            'unknown': 9
        }
        
        df['environment_priority'] = df['environment_normalized'].map(env_priorities)
        
        # Flag production workloads for special handling
        df['is_production'] = df['environment_normalized'] == 'production'
        
        # Environment utilization patterns (weekday vs weekend)
        if 'usage_date' in df.columns:
            df['is_weekend'] = df['usage_date'].dt.weekday >= 5
            
            # Calculate environment weekend usage ratios
            if 'daily_cost' in df.columns:
                cost_col = 'daily_cost'
            elif 'cost' in df.columns:
                cost_col = 'cost'
            else:
                cost_columns = [col for col in df.columns if 'cost' in col.lower()]
                cost_col = cost_columns[0] if cost_columns else None
            
            if cost_col:
                env_weekend_ratio = df.groupby(['environment_normalized', 'is_weekend'])[cost_col].sum().unstack(fill_value=0)
                if not env_weekend_ratio.empty and True in env_weekend_ratio.columns and False in env_weekend_ratio.columns:
                    env_weekend_ratio['weekend_ratio'] = (
                        env_weekend_ratio[True] / 
                        (env_weekend_ratio[True] + env_weekend_ratio[False])
                    ).fillna(0)
                    
                    df = df.merge(
                        env_weekend_ratio[['weekend_ratio']].reset_index(),
                        on='environment_normalized',
                        how='left'
                    )
                else:
                    df['weekend_ratio'] = 0.0
            else:
                df['weekend_ratio'] = 0.0
        else:
            df['is_weekend'] = False
            df['weekend_ratio'] = 0.0
            
        logger.info(f"Processed {df['environment_normalized'].nunique()} unique environments")
    
    def _create_finops_features(self, df: pd.DataFrame) -> None:
        """Create additional FinOps-specific features for forecasting."""
        
        # Create tagging completeness score
        tag_columns = ['cost_center', 'project', 'environment']
        df['tagging_completeness'] = 0
        
        for col in tag_columns:
            if col in df.columns:
                df['tagging_completeness'] += (
                    (~df[col].isin(['untagged', 'unknown'])).astype(int)
                )
        
        df['tagging_completeness'] = df['tagging_completeness'] / len(tag_columns)
        
        # Create well-tagged flag (all required tags present)
        df['is_well_tagged'] = df['tagging_completeness'] == 1.0
        
        # Create business criticality score
        # Production environment + high spend cost center = high criticality
        df['business_criticality'] = 0
        
        if 'is_production' in df.columns:
            df['business_criticality'] += df['is_production'].astype(int) * 3
            
        if 'is_high_spend_cost_center' in df.columns:
            df['business_criticality'] += df['is_high_spend_cost_center'].astype(int) * 2
            
        if 'project_spend_category' in df.columns:
            criticality_mapping = {'high': 2, 'medium': 1, 'low': 0}
            df['business_criticality'] += df['project_spend_category'].map(criticality_mapping).fillna(0)
        
        # Normalize business criticality to 0-1 scale
        if df['business_criticality'].max() > 0:
            df['business_criticality'] = df['business_criticality'] / df['business_criticality'].max()
        
        # Create FinOps maturity indicators
        df['finops_maturity_score'] = (
            df['tagging_completeness'] * 0.4 +
            (df['is_established'].astype(int) if 'is_established' in df.columns else 0) * 0.3 +
            (1 - (df['environment_normalized'] == 'unknown').astype(int)) * 0.3
        )
        
        logger.info("Created FinOps-specific features for forecasting")