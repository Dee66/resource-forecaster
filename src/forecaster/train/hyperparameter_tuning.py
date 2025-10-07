"""
Hyperparameter Tuning for Resource Forecaster

Provides automated hyperparameter optimization for Prophet and
ensemble models using grid search and Bayesian optimization.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings

from ..config import ForecasterConfig
from .forecaster_train import ProphetModel, EnsembleModel
from .model_factory import ModelFactory
from ..exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning for forecasting models."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.model_factory = ModelFactory(config)
        
    def tune_prophet(
        self, 
        train_df: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv_folds: int = 3,
        metric: str = 'rmse'
    ) -> Tuple[Dict[str, Any], float]:
        """Tune Prophet hyperparameters using time-series cross-validation.
        
        Args:
            train_df: Training data
            param_grid: Parameter grid for tuning
            cv_folds: Number of CV folds
            metric: Optimization metric
            
        Returns:
            Tuple of (best_params, best_score)
        """
        if param_grid is None:
            param_grid = self._get_prophet_param_grid()
            
        logger.info(f"Tuning Prophet with {len(list(ParameterGrid(param_grid)))} parameter combinations")
        
        best_params = None
        best_score = float('inf') if metric in ['rmse', 'mae', 'mape'] else float('-inf')
        best_model = None
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for params in ParameterGrid(param_grid):
            logger.debug(f"Testing Prophet params: {params}")
            
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(train_df):
                try:
                    # Split data
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    # Create and train model
                    model = ProphetModel(params)
                    
                    # Suppress Prophet's logging
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.train(fold_train)
                        
                    # Evaluate on validation fold
                    val_metrics = model.evaluate(fold_val)
                    cv_scores.append(val_metrics[metric])
                    
                except Exception as e:
                    logger.warning(f"Prophet tuning failed for params {params}: {str(e)}")
                    cv_scores.append(float('inf'))
                    
            # Calculate mean CV score
            mean_score = np.mean(cv_scores)
            
            # Update best if improved
            if metric in ['rmse', 'mae', 'mape'] and mean_score < best_score:
                best_score = mean_score
                best_params = params
            elif metric == 'r2' and mean_score > best_score:
                best_score = mean_score
                best_params = params
                
        logger.info(f"Best Prophet params: {best_params}, score: {best_score:.4f}")
        return best_params, best_score
        
    def tune_ensemble(
        self,
        train_df: pd.DataFrame,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv_folds: int = 3,
        metric: str = 'rmse'
    ) -> Tuple[Dict[str, Any], float]:
        """Tune ensemble hyperparameters using time-series cross-validation.
        
        Args:
            train_df: Training data
            param_grid: Parameter grid for tuning
            cv_folds: Number of CV folds
            metric: Optimization metric
            
        Returns:
            Tuple of (best_params, best_score)
        """
        if param_grid is None:
            param_grid = self._get_ensemble_param_grid()
            
        logger.info(f"Tuning Ensemble with {len(list(ParameterGrid(param_grid)))} parameter combinations")
        
        best_params = None
        best_score = float('inf') if metric in ['rmse', 'mae', 'mape'] else float('-inf')
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for params in ParameterGrid(param_grid):
            logger.debug(f"Testing Ensemble params: {params}")
            
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(train_df):
                try:
                    # Split data
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    # Create and train model
                    model = EnsembleModel(params)
                    model.train(fold_train)
                    
                    # Evaluate on validation fold
                    val_metrics = model.evaluate(fold_val)
                    cv_scores.append(val_metrics[metric])
                    
                except Exception as e:
                    logger.warning(f"Ensemble tuning failed for params {params}: {str(e)}")
                    cv_scores.append(float('inf'))
                    
            # Calculate mean CV score
            mean_score = np.mean(cv_scores)
            
            # Update best if improved
            if metric in ['rmse', 'mae', 'mape'] and mean_score < best_score:
                best_score = mean_score
                best_params = params
            elif metric == 'r2' and mean_score > best_score:
                best_score = mean_score
                best_params = params
                
        logger.info(f"Best Ensemble params: {best_params}, score: {best_score:.4f}")
        return best_params, best_score
        
    def tune_all_models(
        self,
        train_df: pd.DataFrame,
        cv_folds: int = 3,
        metric: str = 'rmse'
    ) -> Dict[str, Tuple[Dict[str, Any], float]]:
        """Tune hyperparameters for all enabled models.
        
        Args:
            train_df: Training data
            cv_folds: Number of CV folds
            metric: Optimization metric
            
        Returns:
            Dictionary mapping model names to (best_params, best_score)
        """
        results = {}
        
        if self.config.models.prophet.enabled:
            logger.info("Tuning Prophet hyperparameters")
            results['prophet'] = self.tune_prophet(train_df, cv_folds=cv_folds, metric=metric)
            
        if self.config.models.ensemble.enabled:
            logger.info("Tuning Ensemble hyperparameters")
            results['ensemble'] = self.tune_ensemble(train_df, cv_folds=cv_folds, metric=metric)
            
        return results
        
    def _get_prophet_param_grid(self) -> Dict[str, List[Any]]:
        """Get default parameter grid for Prophet tuning."""
        return {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 5.0, 10.0, 15.0],
            'holidays_prior_scale': [0.1, 1.0, 5.0, 10.0, 15.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False]
        }
        
    def _get_ensemble_param_grid(self) -> Dict[str, List[Any]]:
        """Get default parameter grid for ensemble tuning."""
        return {
            'rf_n_estimators': [50, 100, 150, 200],
            'rf_max_depth': [8, 10, 12, 15, None],
            'gbm_n_estimators': [50, 100, 150],
            'gbm_learning_rate': [0.05, 0.1, 0.15, 0.2],
            'gbm_max_depth': [4, 6, 8, 10],
            'weights': [
                [0.5, 0.3, 0.2],
                [0.4, 0.4, 0.2],
                [0.3, 0.5, 0.2],
                [0.45, 0.35, 0.2]
            ]
        }


class BayesianTuner:
    """Bayesian optimization for hyperparameter tuning.
    
    Uses scikit-optimize for more efficient parameter search.
    """
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.model_factory = ModelFactory(config)
        
    def tune_prophet_bayesian(
        self,
        train_df: pd.DataFrame,
        n_calls: int = 50,
        cv_folds: int = 3,
        metric: str = 'rmse',
        random_state: int = 42
    ) -> Tuple[Dict[str, Any], float]:
        """Tune Prophet using Bayesian optimization.
        
        Args:
            train_df: Training data
            n_calls: Number of optimization calls
            cv_folds: Number of CV folds
            metric: Optimization metric
            random_state: Random seed
            
        Returns:
            Tuple of (best_params, best_score)
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to grid search")
            return self.tune_prophet(train_df, cv_folds=cv_folds, metric=metric)
            
        # Define search space
        space = [
            Real(0.001, 0.5, name='changepoint_prior_scale', prior='log-uniform'),
            Real(0.1, 15.0, name='seasonality_prior_scale', prior='log-uniform'),
            Real(0.1, 15.0, name='holidays_prior_scale', prior='log-uniform'),
            Categorical(['additive', 'multiplicative'], name='seasonality_mode'),
            Categorical([True, False], name='yearly_seasonality'),
            Categorical([True, False], name='weekly_seasonality')
        ]
        
        @use_named_args(space)
        def objective(**params):
            """Objective function for Bayesian optimization."""
            try:
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(train_df):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    model = ProphetModel(params)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.train(fold_train)
                        
                    val_metrics = model.evaluate(fold_val)
                    cv_scores.append(val_metrics[metric])
                    
                mean_score = np.mean(cv_scores)
                
                # Bayesian optimization minimizes, so negate for R²
                if metric == 'r2':
                    return -mean_score
                else:
                    return mean_score
                    
            except Exception as e:
                logger.warning(f"Bayesian optimization failed for params {params}: {str(e)}")
                return float('inf')
                
        logger.info(f"Starting Bayesian optimization for Prophet with {n_calls} calls")
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=random_state,
            acq_func='EI'  # Expected Improvement
        )
        
        # Extract best parameters
        best_params = dict(zip([dim.name for dim in space], result.x))
        best_score = result.fun if metric != 'r2' else -result.fun
        
        logger.info(f"Bayesian optimization completed. Best score: {best_score:.4f}")
        return best_params, best_score
        
    def tune_ensemble_bayesian(
        self,
        train_df: pd.DataFrame,
        n_calls: int = 50,
        cv_folds: int = 3,
        metric: str = 'rmse',
        random_state: int = 42
    ) -> Tuple[Dict[str, Any], float]:
        """Tune ensemble using Bayesian optimization.
        
        Args:
            train_df: Training data
            n_calls: Number of optimization calls
            cv_folds: Number of CV folds
            metric: Optimization metric
            random_state: Random seed
            
        Returns:
            Tuple of (best_params, best_score)
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Integer, Real
            from skopt.utils import use_named_args
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to grid search")
            tuner = HyperparameterTuner(self.config)
            return tuner.tune_ensemble(train_df, cv_folds=cv_folds, metric=metric)
            
        # Define search space
        space = [
            Integer(50, 300, name='rf_n_estimators'),
            Integer(8, 20, name='rf_max_depth'),
            Integer(50, 200, name='gbm_n_estimators'),
            Real(0.01, 0.3, name='gbm_learning_rate'),
            Integer(4, 12, name='gbm_max_depth'),
            Real(0.2, 0.6, name='weight_rf'),
            Real(0.2, 0.6, name='weight_gbm')
        ]
        
        @use_named_args(space)
        def objective(**params):
            """Objective function for Bayesian optimization."""
            try:
                # Normalize weights
                weight_rf = params.pop('weight_rf')
                weight_gbm = params.pop('weight_gbm')
                weight_linear = 1.0 - weight_rf - weight_gbm
                
                if weight_linear < 0.1:  # Ensure minimum weight for linear model
                    return float('inf')
                    
                params['weights'] = [weight_rf, weight_gbm, weight_linear]
                
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(train_df):
                    fold_train = train_df.iloc[train_idx]
                    fold_val = train_df.iloc[val_idx]
                    
                    model = EnsembleModel(params)
                    model.train(fold_train)
                    
                    val_metrics = model.evaluate(fold_val)
                    cv_scores.append(val_metrics[metric])
                    
                mean_score = np.mean(cv_scores)
                
                # Bayesian optimization minimizes, so negate for R²
                if metric == 'r2':
                    return -mean_score
                else:
                    return mean_score
                    
            except Exception as e:
                logger.warning(f"Bayesian optimization failed for params {params}: {str(e)}")
                return float('inf')
                
        logger.info(f"Starting Bayesian optimization for Ensemble with {n_calls} calls")
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=random_state,
            acq_func='EI'
        )
        
        # Extract best parameters
        param_names = [dim.name for dim in space]
        best_params = dict(zip(param_names, result.x))
        
        # Fix weights
        weight_rf = best_params.pop('weight_rf')
        weight_gbm = best_params.pop('weight_gbm')
        weight_linear = 1.0 - weight_rf - weight_gbm
        best_params['weights'] = [weight_rf, weight_gbm, weight_linear]
        
        best_score = result.fun if metric != 'r2' else -result.fun
        
        logger.info(f"Bayesian optimization completed. Best score: {best_score:.4f}")
        return best_params, best_score


class AutoMLTuner:
    """Automated machine learning for cost forecasting.
    
    Automatically selects the best model and hyperparameters.
    """
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.grid_tuner = HyperparameterTuner(config)
        self.bayesian_tuner = BayesianTuner(config)
        
    def auto_tune(
        self,
        train_df: pd.DataFrame,
        method: str = 'bayesian',
        metric: str = 'rmse',
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """Automatically tune all models and select the best.
        
        Args:
            train_df: Training data
            method: Tuning method ('grid', 'bayesian')
            metric: Optimization metric
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with best model info and all results
        """
        logger.info(f"Starting AutoML tuning with {method} method")
        
        results = {}
        
        if method == 'bayesian':
            tuner = self.bayesian_tuner
            # Use Bayesian optimization methods
            if self.config.models.prophet.enabled:
                results['prophet'] = tuner.tune_prophet_bayesian(
                    train_df, cv_folds=cv_folds, metric=metric
                )
            if self.config.models.ensemble.enabled:
                results['ensemble'] = tuner.tune_ensemble_bayesian(
                    train_df, cv_folds=cv_folds, metric=metric
                )
        else:
            # Use grid search methods
            if self.config.models.prophet.enabled:
                results['prophet'] = self.grid_tuner.tune_prophet(
                    train_df, cv_folds=cv_folds, metric=metric
                )
            if self.config.models.ensemble.enabled:
                results['ensemble'] = self.grid_tuner.tune_ensemble(
                    train_df, cv_folds=cv_folds, metric=metric
                )
                
        # Find the best model overall
        best_model = None
        best_score = float('inf') if metric in ['rmse', 'mae', 'mape'] else float('-inf')
        
        for model_name, (params, score) in results.items():
            if metric in ['rmse', 'mae', 'mape'] and score < best_score:
                best_score = score
                best_model = model_name
            elif metric == 'r2' and score > best_score:
                best_score = score
                best_model = model_name
                
        automl_results = {
            'best_model': best_model,
            'best_score': best_score,
            'best_params': results[best_model][0] if best_model else None,
            'all_results': results,
            'tuning_method': method,
            'metric': metric
        }
        
        logger.info(f"AutoML completed. Best model: {best_model} with {metric}={best_score:.4f}")
        return automl_results