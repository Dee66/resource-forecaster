"""Resource Forecaster - FinOps & Capacity Planning MLOps Platform.

This package provides enterprise-grade time-series forecasting for AWS cost
optimization and capacity planning.
"""

__version__ = "0.1.0"
__author__ = "ShieldCraft AI"
__email__ = "engineering@shieldcraft.ai"

from .config import ForecasterConfig
from .exceptions import ForecasterError, ForecastValidationError

__all__ = [
    "ForecasterConfig",
    "ForecasterError",
    "ForecastValidationError",
]
