"""
Model Factory for Resource Forecaster

Provides factory methods for creating and configuring different
forecasting models with hyperparameter optimization.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from pathlib import Path

from ..config import ForecasterConfig
from .forecaster_train import ProphetModel, EnsembleModel, ForecastModel
from ..exceptions import ModelTrainingError, ConfigurationError

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating and managing forecasting models."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.available_models = {
            'prophet': ProphetModel,
            'ensemble': EnsembleModel
        }
        
    def create_model(self, model_type: str, model_config: Optional[Dict[str, Any]] = None) -> ForecastModel:
        """Create a forecasting model of the specified type.
        
        Args:
            model_type: Type of model to create ('prophet', 'ensemble')
            model_config: Optional model-specific configuration
            
        Returns:
            Configured forecasting model
            
        Raises:
            ConfigurationError: If model type is not supported
        """
        if model_type not in self.available_models:
            raise ConfigurationError(
                f"Unsupported model type: {model_type}",
                config_key="model_type"
            )
            
        model_class = self.available_models[model_type]
        
        # Use provided config or default from main config
        if model_config is None:
            if model_type == 'prophet':
                model_config = self.config.models.prophet.dict()
            elif model_type == 'ensemble':
                model_config = self.config.models.ensemble.dict()
            else:
                model_config = {}
                
        logger.info(f"Creating {model_type} model with config: {model_config}")
        return model_class(model_config)
        
    def create_all_enabled_models(self) -> Dict[str, ForecastModel]:
        """Create all models that are enabled in configuration.
        
        Returns:
            Dictionary mapping model names to model instances
        """
        models = {}
        
        if self.config.models.prophet.enabled:
            models['prophet'] = self.create_model('prophet')
            
        if self.config.models.ensemble.enabled:
            models['ensemble'] = self.create_model('ensemble')
            
        logger.info(f"Created {len(models)} enabled models: {list(models.keys())}")
        return models
        
    def get_best_model_config(self, model_type: str, metric: str = 'rmse') -> Dict[str, Any]:
        """Get optimized configuration for a model type.
        
        This method provides predefined optimized configurations
        based on common cost forecasting patterns.
        
        Args:
            model_type: Type of model ('prophet', 'ensemble')
            metric: Optimization metric ('rmse', 'mae', 'mape')
            
        Returns:
            Optimized model configuration
        """
        if model_type == 'prophet':
            return self._get_prophet_best_config(metric)
        elif model_type == 'ensemble':
            return self._get_ensemble_best_config(metric)
        else:
            raise ConfigurationError(
                f"No best config available for model type: {model_type}",
                config_key="model_type"
            )
            
    def _get_prophet_best_config(self, metric: str) -> Dict[str, Any]:
        """Get optimized Prophet configuration."""
        base_config = {
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'multiplicative',
            'interval_width': 0.95,
            'uncertainty_samples': 1000
        }
        
        # Metric-specific optimizations
        if metric == 'rmse':
            base_config.update({
                'changepoint_prior_scale': 0.01,  # Lower for stability
                'seasonality_prior_scale': 1.0,   # Lower for less overfitting
                'holidays_prior_scale': 1.0
            })
        elif metric == 'mae':
            base_config.update({
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 5.0,
                'holidays_prior_scale': 5.0
            })
        elif metric == 'mape':
            base_config.update({
                'changepoint_prior_scale': 0.1,   # Higher for trend flexibility
                'seasonality_prior_scale': 10.0,  # Higher for seasonal patterns
                'holidays_prior_scale': 10.0
            })
            
        return base_config
        
    def _get_ensemble_best_config(self, metric: str) -> Dict[str, Any]:
        """Get optimized ensemble configuration."""
        base_config = {
            'weights': [0.4, 0.4, 0.2]  # RF, GBM, Linear
        }
        
        # Metric-specific optimizations
        if metric == 'rmse':
            base_config.update({
                'rf_n_estimators': 200,
                'rf_max_depth': 15,
                'gbm_n_estimators': 150,
                'gbm_learning_rate': 0.05,
                'gbm_max_depth': 8
            })
        elif metric == 'mae':
            base_config.update({
                'rf_n_estimators': 150,
                'rf_max_depth': 12,
                'gbm_n_estimators': 100,
                'gbm_learning_rate': 0.1,
                'gbm_max_depth': 6
            })
        elif metric == 'mape':
            base_config.update({
                'rf_n_estimators': 100,
                'rf_max_depth': 10,
                'gbm_n_estimators': 80,
                'gbm_learning_rate': 0.15,
                'gbm_max_depth': 5
            })
            
        return base_config


class ModelRegistry:
    """Registry for managing trained models and their metadata."""
    
    def __init__(self, registry_path: Union[str, Path]):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
    def register_model(
        self, 
        model: ForecastModel, 
        model_name: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Register a trained model in the registry.
        
        Args:
            model: Trained forecasting model
            model_name: Name for the model
            metadata: Model metadata (metrics, config, etc.)
            
        Returns:
            Model version ID
        """
        import joblib
        import json
        from datetime import datetime
        
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_name}_{timestamp}"
        
        # Save model
        model_path = self.models_path / f"{version_id}.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = self.models_path / f"{version_id}_metadata.json"
        metadata_with_info = {
            'version_id': version_id,
            'model_name': model_name,
            'model_type': model.model_name,
            'registered_at': datetime.now().isoformat(),
            'model_path': str(model_path),
            **metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_with_info, f, indent=2)
            
        logger.info(f"Registered model {model_name} as version {version_id}")
        return version_id
        
    def load_model(self, version_id: str) -> ForecastModel:
        """Load a model from the registry.
        
        Args:
            version_id: Model version ID
            
        Returns:
            Loaded forecasting model
        """
        import joblib
        
        model_path = self.models_path / f"{version_id}.pkl"
        if not model_path.exists():
            raise ModelTrainingError(
                f"Model version {version_id} not found",
                model_type="unknown"
            )
            
        model = joblib.load(model_path)
        logger.info(f"Loaded model version {version_id}")
        return model
        
    def get_model_metadata(self, version_id: str) -> Dict[str, Any]:
        """Get metadata for a registered model.
        
        Args:
            version_id: Model version ID
            
        Returns:
            Model metadata
        """
        import json
        
        metadata_path = self.models_path / f"{version_id}_metadata.json"
        if not metadata_path.exists():
            raise ModelTrainingError(
                f"Metadata for model version {version_id} not found",
                model_type="unknown"
            )
            
        with open(metadata_path, 'r') as f:
            return json.load(f)
            
    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models.
        
        Args:
            model_name: Optional filter by model name
            
        Returns:
            List of model metadata
        """
        import json
        
        models = []
        for metadata_file in self.models_path.glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            if model_name is None or metadata.get('model_name') == model_name:
                models.append(metadata)
                
        # Sort by registration time (newest first)
        models.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        return models
        
    def get_best_model(
        self, 
        model_name: Optional[str] = None,
        metric: str = 'rmse'
    ) -> Optional[str]:
        """Get the best performing model version.
        
        Args:
            model_name: Optional filter by model name
            metric: Metric to optimize for
            
        Returns:
            Version ID of best model, or None if no models found
        """
        models = self.list_models(model_name)
        if not models:
            return None
            
        # Find model with best validation metric
        best_model = None
        best_score = float('inf') if metric in ['rmse', 'mae', 'mape'] else float('-inf')
        
        for model in models:
            val_metrics = model.get('val_metrics', {})
            if metric in val_metrics:
                score = val_metrics[metric]
                
                # Lower is better for error metrics
                if metric in ['rmse', 'mae', 'mape'] and score < best_score:
                    best_score = score
                    best_model = model['version_id']
                # Higher is better for R²
                elif metric == 'r2' and score > best_score:
                    best_score = score
                    best_model = model['version_id']
                    
        return best_model
        
    def cleanup_old_models(self, keep_count: int = 5):
        """Remove old model versions, keeping only the most recent.
        
        Args:
            keep_count: Number of recent models to keep per model name
        """
        import os
        
        # Group models by name
        models_by_name = {}
        for model in self.list_models():
            name = model['model_name']
            if name not in models_by_name:
                models_by_name[name] = []
            models_by_name[name].append(model)
            
        # Remove old versions
        removed_count = 0
        for name, models in models_by_name.items():
            if len(models) > keep_count:
                for model in models[keep_count:]:
                    version_id = model['version_id']
                    
                    # Remove model file
                    model_path = self.models_path / f"{version_id}.pkl"
                    if model_path.exists():
                        os.remove(model_path)
                        
                    # Remove metadata file
                    metadata_path = self.models_path / f"{version_id}_metadata.json"
                    if metadata_path.exists():
                        os.remove(metadata_path)
                        
                    removed_count += 1
                    
        logger.info(f"Cleaned up {removed_count} old model versions")