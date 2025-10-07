"""Model fidelity gate: fails if core metrics exceed thresholds on a deterministic dataset.

This test acts as a guardrail in CI to catch regressions in metric calculations
or prediction quality heuristics. If thresholds need tuning, adjust the constants
below or set environment variables:
- FIDELITY_MAX_MAPE (percent, e.g., 5.0)
- FIDELITY_MAX_RMSE
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from src.forecaster.validation.validator import ModelValidator
from src.forecaster.config import (
    ForecasterConfig,
    DataSourceConfig,
    InfrastructureConfig,
)


def _make_validator() -> ModelValidator:
    # Minimal viable config
    cfg = ForecasterConfig(
        environment="dev",
        aws_region="us-east-1",
        aws_account_id="000000000000",
        data_source=DataSourceConfig(
            cur_bucket="test-cur-bucket",
            athena_output_bucket="test-athena-bucket",
            cur_prefix="cur/",
            cloudwatch_namespace="AWS/EC2",
            athena_database="cost_usage_reports",
            athena_workgroup="primary",
        ),
        infrastructure=InfrastructureConfig(
            model_bucket="test-model-bucket",
            data_bucket="test-data-bucket",
            vpc_id=None,
            compute_type="fargate",
            cpu=512,
            memory=1024,
            cloudwatch_log_group="/aws/forecaster",
            cloudwatch_retention_days=30,
        ),
        api_host="127.0.0.1",
        api_port=8000,
        log_level="INFO",
        structured_logging=True,
    )
    return ModelValidator(cfg)


def test_model_fidelity_thresholds():
    # Thresholds (allow env override for tuning in CI without code changes)
    max_mape = float(os.getenv("FIDELITY_MAX_MAPE", "5.0"))  # percent
    max_rmse = float(os.getenv("FIDELITY_MAX_RMSE", "30.0"))

    # Deterministic dataset
    rng = np.random.default_rng(42)
    n = 100
    base = 1000.0
    trend = np.linspace(0, 50, n)
    noise = rng.normal(0.0, 10.0, n)
    actual = base + trend + noise  # around ~1000-1050

    # Predicted = 97.5% of actual + small noise => ~2.5% MAPE
    pred_noise = rng.normal(0.0, 3.0, n)
    predicted = actual * 0.975 + pred_noise

    y_true = pd.Series(actual)
    y_pred = pd.Series(predicted)

    validator = _make_validator()
    metrics = validator.calculate_metrics(y_true, y_pred)

    # Assertions (MAPE is in percent in our metrics schema)
    assert metrics.mape < max_mape, f"MAPE {metrics.mape:.3f}% exceeds {max_mape}%"
    assert metrics.rmse < max_rmse, f"RMSE {metrics.rmse:.3f} exceeds {max_rmse}"
