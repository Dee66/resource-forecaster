"""Custom exceptions for Resource Forecaster.

Defines exception hierarchy for different types of forecasting errors.
"""

from __future__ import annotations

from typing import Any


class ForecasterError(Exception):
    """Base exception for all forecaster errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ForecastValidationError(ForecasterError, ValueError):
    """Raised when forecast validation fails."""

    def __init__(
        self, message: str, metric_name: str, actual_value: float, threshold: float, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.metric_name = metric_name
        self.actual_value = actual_value
        self.threshold = threshold

        # Add validation details
        self.details.update(
            {
                "metric": metric_name,
                "actual": actual_value,
                "threshold": threshold,
                "exceeded_by": actual_value - threshold,
            }
        )


class DataSourceError(ForecasterError):
    """Raised when data source operations fail."""

    def __init__(self, message: str, source_type: str, **kwargs):
        super().__init__(message, **kwargs)
        self.source_type = source_type
        self.details["source_type"] = source_type


class DataValidationError(ForecasterError):
    """Raised when data validation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class ModelTrainingError(ForecasterError):
    """Raised when model training fails."""

    def __init__(self, message: str, model_type: str = "unknown", **kwargs):
        super().__init__(message, **kwargs)
        self.model_type = model_type
        self.details["model_type"] = model_type


class ModelInferenceError(ForecasterError):
    """Raised when model inference fails."""

    def __init__(self, message: str, model_path: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model_path = model_path
        if model_path:
            self.details["model_path"] = model_path


class ConfigurationError(ForecasterError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


class InfrastructureError(ForecasterError):
    """Raised when infrastructure operations fail."""

    def __init__(self, message: str, resource_type: str, **kwargs):
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.details["resource_type"] = resource_type


class FinOpsError(ForecasterError):
    """Raised when FinOps operations fail."""

    def __init__(self, message: str, operation: str, **kwargs):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.details["operation"] = operation


class DataProcessingError(ForecasterError):
    """Raised when data processing or preparation fails."""

    def __init__(self, message: str, step: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        if step:
            self.details["step"] = step


class PredictionError(ForecasterError):
    """Raised when prediction or forecasting fails."""

    def __init__(self, message: str, model_name: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        if model_name:
            self.details["model_name"] = model_name


class ModelLoadingError(ForecasterError):
    """Raised when model artifacts cannot be loaded or saved."""

    def __init__(self, message: str, model_name: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        if model_name:
            self.details["model_name"] = model_name


class RecommendationError(ForecasterError):
    """Raised when generating recommendations fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


class MetricsError(ForecasterError):
    """Raised when metrics collection or processing fails."""

    def __init__(self, message: str, metric_name: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        if metric_name:
            self.details["metric_name"] = metric_name
