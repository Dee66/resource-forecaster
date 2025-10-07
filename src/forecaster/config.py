"""Configuration management for Resource Forecaster.

Provides environment-specific configuration loading and validation
for the FinOps cost prediction system.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""

    # Cost and Usage Report (CUR) settings
    cur_bucket: str = Field(..., description="S3 bucket containing CUR data")
    cur_prefix: str = Field("cur/", description="S3 prefix for CUR data")

    # CloudWatch metrics settings
    cloudwatch_namespace: str = Field("AWS/EC2", description="CloudWatch namespace")
    cloudwatch_metrics: list[str] = Field(
        default_factory=lambda: ["CPUUtilization", "NetworkIn", "NetworkOut"],
        description="CloudWatch metrics to collect",
    )

    # Athena settings for CUR queries
    athena_database: str = Field("cost_usage_reports", description="Athena database")
    athena_workgroup: str = Field("primary", description="Athena workgroup")
    athena_output_bucket: str = Field(..., description="S3 bucket for Athena results")


class ModelConfig(BaseModel):
    """Configuration for forecasting models."""

    # Model selection
    primary_model: str = Field("prophet", description="Primary model type")
    ensemble_models: list[str] = Field(
        default_factory=lambda: ["prophet", "arima", "linear"],
        description="Models to use in ensemble",
    )

    # Training parameters
    horizon_days: int = Field(30, description="Default forecast horizon in days")
    training_window_days: int = Field(365, description="Training data window in days")
    validation_split: float = Field(0.2, description="Validation data split ratio")

    # Prophet-specific settings
    prophet_seasonality_mode: str = Field("multiplicative", description="Prophet seasonality mode")
    prophet_yearly_seasonality: bool = Field(True, description="Enable yearly seasonality")
    prophet_weekly_seasonality: bool = Field(True, description="Enable weekly seasonality")
    prophet_daily_seasonality: bool = Field(False, description="Enable daily seasonality")

    # Quality thresholds
    max_rmse: float = Field(0.05, description="Maximum allowed RMSE (5%)")
    max_mape: float = Field(0.10, description="Maximum allowed MAPE (10%)")
    min_r2: float = Field(0.8, description="Minimum required R² score")


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering."""

    # Time-based features
    include_day_of_week: bool = Field(True, description="Include day of week features")
    include_month: bool = Field(True, description="Include month features")
    include_quarter: bool = Field(True, description="Include quarter features")
    include_holidays: bool = Field(True, description="Include holiday features")

    # FinOps tag features
    required_tags: list[str] = Field(
        default_factory=lambda: ["CostCenter", "Project", "Environment", "Owner"],
        description="Required FinOps tags to include as features",
    )

    # Lag features
    lag_days: list[int] = Field(
        default_factory=lambda: [1, 7, 14, 30], description="Lag days for cost features"
    )

    # Rolling window features
    rolling_windows: list[int] = Field(
        default_factory=lambda: [7, 14, 30], description="Rolling window sizes for statistics"
    )


class InfrastructureConfig(BaseModel):
    """Configuration for AWS infrastructure."""

    # Deployment settings
    vpc_id: str | None = Field(None, description="VPC ID for deployment")
    private_subnet_ids: list[str] = Field(default_factory=list, description="Private subnet IDs")

    # Compute settings
    compute_type: str = Field("fargate", description="Compute type (fargate/lambda)")
    cpu: int = Field(512, description="CPU units for Fargate")
    memory: int = Field(1024, description="Memory MB for Fargate")

    # Storage settings
    model_bucket: str = Field(..., description="S3 bucket for model artifacts")
    data_bucket: str = Field(..., description="S3 bucket for forecast data")

    # Monitoring
    cloudwatch_log_group: str = Field("/aws/forecaster", description="CloudWatch log group")
    cloudwatch_retention_days: int = Field(30, description="Log retention days")


class FinOpsConfig(BaseModel):
    """Configuration for FinOps automation."""

    # Budget settings
    budget_alert_threshold: float = Field(0.8, description="Budget alert threshold (80%)")
    cost_anomaly_threshold: float = Field(0.2, description="Cost anomaly threshold (20%)")

    # Rightsizing settings
    rightsizing_enabled: bool = Field(True, description="Enable automated rightsizing")
    utilization_threshold: float = Field(0.7, description="Utilization threshold for rightsizing")

    # Resource lifecycle
    auto_shutdown_enabled: bool = Field(True, description="Enable auto-shutdown")
    idle_threshold_hours: int = Field(72, description="Hours before auto-shutdown")

    # Savings recommendations
    savings_plan_enabled: bool = Field(True, description="Enable Savings Plan recommendations")
    reserved_instance_enabled: bool = Field(True, description="Enable RI recommendations")


class ForecasterConfig(BaseModel):
    """Main configuration class for Resource Forecaster."""

    # Environment
    environment: str = Field(..., description="Deployment environment")
    aws_region: str = Field("us-east-1", description="AWS region")
    aws_account_id: str | None = Field(None, description="AWS account ID")

    # Sub-configurations
    data_source: DataSourceConfig
    model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            primary_model="prophet",
            ensemble_models=["prophet", "arima", "linear"],
            horizon_days=30,
            training_window_days=365,
            validation_split=0.2,
            prophet_seasonality_mode="multiplicative",
            prophet_yearly_seasonality=True,
            prophet_weekly_seasonality=True,
            prophet_daily_seasonality=False,
            max_rmse=0.05,
            max_mape=0.10,
            min_r2=0.8,
        )
    )
    features: FeatureEngineeringConfig = Field(
        default_factory=lambda: FeatureEngineeringConfig(
            include_day_of_week=True,
            include_month=True,
            include_quarter=True,
            include_holidays=True,
            required_tags=["CostCenter", "Project", "Environment", "Owner"],
            lag_days=[1, 7, 14, 30],
            rolling_windows=[7, 14, 30],
        )
    )
    infrastructure: InfrastructureConfig
    finops: FinOpsConfig = Field(
        default_factory=lambda: FinOpsConfig(
            budget_alert_threshold=0.8,
            cost_anomaly_threshold=0.2,
            rightsizing_enabled=True,
            utilization_threshold=0.7,
            auto_shutdown_enabled=True,
            idle_threshold_hours=72,
            savings_plan_enabled=True,
            reserved_instance_enabled=True,
        )
    )

    # API settings
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, description="API port")

    # Logging
    log_level: str = Field("INFO", description="Logging level")
    structured_logging: bool = Field(True, description="Enable structured logging")

    @classmethod
    @field_validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment values."""
        allowed_envs = ["dev", "staging", "prod"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v

    @classmethod
    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level values."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()

    @classmethod
    def from_env(cls) -> "ForecasterConfig":
        """Load configuration from environment variables."""
        environment = os.environ.get("ENVIRONMENT", "dev")

        config_data = {
            "environment": environment,
            "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
            "data_source": {
                "cur_bucket": os.environ.get("CUR_BUCKET", f"finops-cur-{environment}"),
                "athena_output_bucket": os.environ.get("ATHENA_OUTPUT_BUCKET", f"finops-athena-{environment}"),
                "athena_database": os.environ.get("ATHENA_DATABASE", "cost_usage_reports"),
                "athena_workgroup": os.environ.get("ATHENA_WORKGROUP", "primary"),
            },
            "infrastructure": {
                "model_bucket": os.environ.get("MODEL_BUCKET", f"forecaster-models-{environment}"),
                "data_bucket": os.environ.get("DATA_BUCKET", f"forecaster-data-{environment}"),
            },
            "model": {
                "primary_model": os.environ.get("PRIMARY_MODEL", "prophet"),
                "horizon_days": int(os.environ.get("HORIZON_DAYS", "30")),
                "training_window_days": int(os.environ.get("TRAINING_WINDOW_DAYS", "365")),
            },
            "finops": {
                "budget_alert_threshold": float(os.environ.get("BUDGET_ALERT_THRESHOLD", "0.85")),
                "auto_shutdown_enabled": os.environ.get("AUTO_SHUTDOWN_ENABLED", "false").lower()
                == "true",
            },
        }

        return cls(**config_data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ForecasterConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            ForecasterConfig instance loaded from the file
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Try to infer environment from filename if not provided
        if "environment" not in data:
            stem = config_path.stem.lower()
            if stem in {"dev", "staging", "prod"}:
                data["environment"] = stem
            else:
                data["environment"] = os.environ.get("ENVIRONMENT", "dev")

        # Backfill AWS account ID from environment if missing
        data.setdefault("aws_account_id", os.getenv("AWS_ACCOUNT_ID"))

        return cls(**data)


def load_config(environment: str, config_path: Path | None = None) -> ForecasterConfig:
    """Load configuration for the specified environment.

    Args:
        environment: Target environment (dev/staging/prod)
        config_path: Optional custom config file path

    Returns:
        Loaded configuration object

    Raises:
        FileNotFoundError: If config file is not found
        ValueError: If configuration is invalid
    """
    if config_path is None:
        # Use default config path
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_path = config_dir / f"{environment}.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML configuration
    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Add environment to config data
    config_data["environment"] = environment

    # Load AWS account ID from environment if not in config
    if "aws_account_id" not in config_data:
        config_data["aws_account_id"] = os.getenv("AWS_ACCOUNT_ID")

    # Create and validate configuration
    return ForecasterConfig(**config_data)


def get_default_config(environment: str) -> dict[str, Any]:
    """Get default configuration for an environment.

    Args:
        environment: Target environment

    Returns:
        Default configuration dictionary
    """
    base_config = {
        "environment": environment,
        "aws_region": "us-east-1",
        "data_source": {
            "cur_bucket": f"finops-cur-{environment}",
            "athena_output_bucket": f"finops-athena-{environment}",
        },
        "infrastructure": {
            "model_bucket": f"forecaster-models-{environment}",
            "data_bucket": f"forecaster-data-{environment}",
        },
    }

    # Environment-specific overrides
    if environment == "prod":
        base_config["finops"] = {
            "budget_alert_threshold": 0.85,
            "auto_shutdown_enabled": False,  # Disable in prod
        }
    elif environment == "dev":
        base_config["finops"] = {
            "budget_alert_threshold": 0.5,
            "auto_shutdown_enabled": True,
            "idle_threshold_hours": 24,  # More aggressive in dev
        }

    return base_config
