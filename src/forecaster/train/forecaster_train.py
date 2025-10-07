"""
Resource Forecaster - Training Module

Implements time-series forecasting models for AWS cost prediction using Prophet
and ensemble methods. Targets RMSE ≤ 5% for accurate FinOps automation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from pathlib import Path

from ..config import Config
from ..data.collectors import CURDataCollector, CloudWatchCollector
from ..data.processors import CostDataProcessor, FeatureEngineer
from ..exceptions import ModelTrainingError, DataValidationError

logger = logging.getLogger(__name__)


class ForecastModel:
    """Base class for forecasting models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the model on time-series data."""
        raise NotImplementedError
        
    def predict(self, periods: int, future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate predictions for future periods."""
        raise NotImplementedError
        
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance."""
        raise NotImplementedError


class ProphetModel(ForecastModel):
    """Prophet-based time-series forecasting model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("prophet", config)
        
        # Prophet configuration
        prophet_config = config.get("prophet", {})
        self.model = Prophet(
            yearly_seasonality=prophet_config.get("yearly_seasonality", True),
            weekly_seasonality=prophet_config.get("weekly_seasonality", True),
            daily_seasonality=prophet_config.get("daily_seasonality", False),
            seasonality_mode=prophet_config.get("seasonality_mode", "multiplicative"),
            changepoint_prior_scale=prophet_config.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=prophet_config.get("seasonality_prior_scale", 10.0),
            holidays_prior_scale=prophet_config.get("holidays_prior_scale", 10.0),
            mcmc_samples=prophet_config.get("mcmc_samples", 0),
            interval_width=prophet_config.get("interval_width", 0.95),
            uncertainty_samples=prophet_config.get("uncertainty_samples", 1000)
        )
        
        # Add custom seasonalities
        self._add_custom_seasonalities()
        
    def _add_custom_seasonalities(self):
        """Add custom seasonalities for cloud cost patterns."""
        # Monthly seasonality for billing cycles
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        # Quarter seasonality for budget cycles
        self.model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=3
        )
        
    def _prepare_prophet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data in Prophet format (ds, y columns)."""
        if 'date' not in df.columns or 'cost' not in df.columns:
            raise DataValidationError("DataFrame must contain 'date' and 'cost' columns")
            
        prophet_df = df[['date', 'cost']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Add regressors if available
        regressor_cols = [col for col in df.columns if col.startswith('feature_')]
        for col in regressor_cols:
            prophet_df[col] = df[col]
            self.model.add_regressor(col)
            
        return prophet_df.sort_values('ds')
        
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet model on cost data."""
        try:
            logger.info(f"Training Prophet model on {len(df)} data points")
            
            # Prepare data
            prophet_df = self._prepare_prophet_data(df)
            
            # Train model
            self.model.fit(prophet_df)
            self.is_trained = True
            
            # Calculate training metrics
            in_sample_forecast = self.model.predict(prophet_df)
            self.training_metrics = self._calculate_metrics(
                prophet_df['y'].values,
                in_sample_forecast['yhat'].values
            )
            
            logger.info(f"Prophet training completed. RMSE: {self.training_metrics['rmse']:.4f}")
            return self.training_metrics
            
        except Exception as e:
            raise ModelTrainingError(f"Prophet training failed: {str(e)}", model_type="prophet")
            
    def predict(self, periods: int, future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate Prophet predictions."""
        if not self.is_trained:
            raise ModelTrainingError("Model must be trained before prediction")
            
        try:
            # Create future dataframe
            if future_df is not None:
                future = self._prepare_prophet_data(future_df)[['ds'] + 
                    [col for col in future_df.columns if col.startswith('feature_')]]
            else:
                future = self.model.make_future_dataframe(periods=periods, freq='D')
                
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Return relevant columns
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
        except Exception as e:
            raise ModelTrainingError(f"Prophet prediction failed: {str(e)}", model_type="prophet")
            
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate Prophet model on test data."""
        if not self.is_trained:
            raise ModelTrainingError("Model must be trained before evaluation", model_type="prophet")
            
        try:
            prophet_test = self._prepare_prophet_data(test_df)
            forecast = self.model.predict(prophet_test)
            
            return self._calculate_metrics(
                prophet_test['y'].values,
                forecast['yhat'].values
            )
            
        except Exception as e:
            raise ModelTrainingError(f"Prophet evaluation failed: {str(e)}", model_type="prophet")
            
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }


class EnsembleModel(ForecastModel):
    """Ensemble of ML models for cost forecasting."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ensemble", config)
        
        # Initialize base models
        ensemble_config = config.get("ensemble", {})
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=ensemble_config.get("rf_n_estimators", 100),
                max_depth=ensemble_config.get("rf_max_depth", 10),
                random_state=42
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=ensemble_config.get("gbm_n_estimators", 100),
                learning_rate=ensemble_config.get("gbm_learning_rate", 0.1),
                max_depth=ensemble_config.get("gbm_max_depth", 6),
                random_state=42
            ),
            'linear': LinearRegression()
        }
        
        self.weights = ensemble_config.get("weights", [0.4, 0.4, 0.2])
        self.feature_columns = []
        
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for ML models."""
        # Use feature columns created by FeatureEngineer
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            raise DataValidationError("No feature columns found in data")
            
        self.feature_columns = feature_cols
        X = df[feature_cols].values
        y = df['cost'].values
        
        return X, y
        
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train ensemble models."""
        try:
            logger.info(f"Training ensemble models on {len(df)} data points")
            
            # Prepare features
            X, y = self._prepare_features(df)
            
            # Train each model
            model_metrics = {}
            for name, model in self.models.items():
                logger.info(f"Training {name} model")
                model.fit(X, y)
                
                # Calculate in-sample metrics
                y_pred = model.predict(X)
                model_metrics[name] = self._calculate_metrics(y, y_pred)
                
            self.is_trained = True
            self.training_metrics = model_metrics
            
            logger.info("Ensemble training completed")
            return model_metrics
            
        except Exception as e:
            raise ModelTrainingError(f"Ensemble training failed: {str(e)}", model_type="ensemble")
            
    def predict(self, periods: int, future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate ensemble predictions."""
        if not self.is_trained:
            raise ModelTrainingError("Model must be trained before prediction")
            
        if future_df is None:
            raise ModelTrainingError("future_df required for ensemble predictions", model_type="ensemble")
            
        try:
            # Prepare features
            X_future = future_df[self.feature_columns].values
            
            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X_future)
                
            # Weighted ensemble prediction
            ensemble_pred = np.zeros_like(predictions['rf'])
            for i, (name, pred) in enumerate(predictions.items()):
                ensemble_pred += self.weights[i] * pred
                
            # Create result dataframe
            result_df = pd.DataFrame({
                'date': future_df['date'].values,
                'yhat': ensemble_pred,
                'yhat_lower': ensemble_pred * 0.9,  # Simple confidence interval
                'yhat_upper': ensemble_pred * 1.1
            })
            
            return result_df.tail(periods)
            
        except Exception as e:
            raise ModelTrainingError(f"Ensemble prediction failed: {str(e)}", model_type="ensemble")
            
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate ensemble model."""
        if not self.is_trained:
            raise ModelTrainingError("Model must be trained before evaluation", model_type="ensemble")
            
        try:
            X_test, y_test = self._prepare_features(test_df)
            
            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X_test)
                
            # Weighted ensemble prediction
            ensemble_pred = np.zeros_like(predictions['rf'])
            for i, (name, pred) in enumerate(predictions.items()):
                ensemble_pred += self.weights[i] * pred
                
            return self._calculate_metrics(y_test, ensemble_pred)
            
        except Exception as e:
            raise ModelTrainingError(f"Ensemble evaluation failed: {str(e)}", model_type="ensemble")
            
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }


class ForecastTrainer:
    """Main trainer class for cost forecasting models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.cur_collector = CURDataCollector(config)
        self.cloudwatch_collector = CloudWatchCollector(config)
        self.processor = CostDataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        
        # Model storage
        self.models = {}
        self.model_artifacts_path = Path(config.training.model_artifacts_path)
        self.model_artifacts_path.mkdir(parents=True, exist_ok=True)
        
    def collect_training_data(self) -> pd.DataFrame:
        """Collect and prepare training data."""
        logger.info("Collecting training data from CUR and CloudWatch")
        
        # Define date range for training
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.data_sources.training_days)
        
        # Collect CUR data
        cur_data = self.cur_collector.collect_daily_costs(start_date, end_date)
        
        # Collect CloudWatch metrics
        cloudwatch_data = self.cloudwatch_collector.collect_resource_metrics(
            start_date, 
            end_date,
            self.config.data_sources.cloudwatch_metrics
        )
        
        # Process and merge data
        processed_data = self.processor.preprocess_cost_data(cur_data)
        
        # Merge with CloudWatch metrics
        if not cloudwatch_data.empty:
            processed_data = processed_data.merge(
                cloudwatch_data, 
                on=['date', 'service'], 
                how='left'
            )
            
        # Feature engineering
        final_data = self.feature_engineer.create_features(processed_data)
        
        logger.info(f"Collected {len(final_data)} training samples")
        return final_data
        
    def train_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Train all forecasting models."""
        logger.info("Starting model training pipeline")
        
        # Validate data
        if len(df) < self.config.training.min_training_samples:
            raise DataValidationError(
                f"Insufficient training data: {len(df)} < {self.config.training.min_training_samples}"
            )
            
        # Split data for validation
        split_point = int(len(df) * 0.8)
        train_df = df.iloc[:split_point]
        val_df = df.iloc[split_point:]
        
        results = {}
        
        # Train Prophet model
        if self.config.models.prophet.enabled:
            logger.info("Training Prophet model")
            prophet_model = ProphetModel(self.config.models.prophet.dict())
            train_metrics = prophet_model.train(train_df)
            val_metrics = prophet_model.evaluate(val_df)
            
            self.models['prophet'] = prophet_model
            results['prophet'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
        # Train Ensemble model
        if self.config.models.ensemble.enabled:
            logger.info("Training Ensemble model")
            ensemble_model = EnsembleModel(self.config.models.ensemble.dict())
            train_metrics = ensemble_model.train(train_df)
            val_metrics = ensemble_model.evaluate(val_df)
            
            self.models['ensemble'] = ensemble_model
            results['ensemble'] = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
        logger.info("Model training completed")
        return results
        
    def save_models(self):
        """Save trained models to disk."""
        for name, model in self.models.items():
            model_path = self.model_artifacts_path / f"{name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
            
    def load_models(self):
        """Load models from disk."""
        for model_file in self.model_artifacts_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} model from {model_file}")
            
    def cross_validate(self, df: pd.DataFrame, n_splits: int = 3) -> Dict[str, Dict[str, List[float]]]:
        """Perform time-series cross-validation."""
        logger.info(f"Performing {n_splits}-fold time-series cross-validation")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {}
        
        for name, model_class in [('prophet', ProphetModel), ('ensemble', EnsembleModel)]:
            if name == 'prophet' and not self.config.models.prophet.enabled:
                continue
            if name == 'ensemble' and not self.config.models.ensemble.enabled:
                continue
                
            cv_scores = {'rmse': [], 'mae': [], 'r2': [], 'mape': []}
            
            for train_idx, test_idx in tscv.split(df):
                train_fold = df.iloc[train_idx]
                test_fold = df.iloc[test_idx]
                
                # Train model on fold
                if name == 'prophet':
                    fold_model = ProphetModel(self.config.models.prophet.dict())
                else:
                    fold_model = EnsembleModel(self.config.models.ensemble.dict())
                    
                fold_model.train(train_fold)
                fold_metrics = fold_model.evaluate(test_fold)
                
                for metric, value in fold_metrics.items():
                    cv_scores[metric].append(value)
                    
            cv_results[name] = cv_scores
            
        return cv_results
        
    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting Resource Forecaster training pipeline")
        
        try:
            # Collect training data
            training_data = self.collect_training_data()
            
            # Train models
            training_results = self.train_models(training_data)
            
            # Cross-validation
            cv_results = self.cross_validate(training_data)
            
            # Save models
            self.save_models()
            
            # Create summary results
            results = {
                'data_size': len(training_data),
                'training_results': training_results,
                'cv_results': cv_results,
                'models_saved': list(self.models.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise