import os
import yaml
import pytest

from pathlib import Path

from src.forecaster.config import (
    ForecasterConfig,
    get_default_config,
    load_config,
)


def make_minimal_config_dict(environment: str = "dev") -> dict:
    return {
        "environment": environment,
        "aws_region": "us-east-1",
        "data_source": {
            "cur_bucket": f"finops-cur-{environment}",
            "athena_output_bucket": f"finops-athena-{environment}",
            "athena_database": "cost_usage_reports",
            "athena_workgroup": "primary",
        },
        "infrastructure": {
            "model_bucket": f"forecaster-models-{environment}",
            "data_bucket": f"forecaster-data-{environment}",
        },
    }


def test_get_default_config_overrides():
    prod = get_default_config("prod")
    dev = get_default_config("dev")

    # Prod should disable auto_shutdown in defaults
    assert prod["finops"]["auto_shutdown_enabled"] is False

    # Dev should have a more aggressive idle threshold
    assert dev["finops"]["idle_threshold_hours"] == 24


def test_load_config_from_yaml(tmp_path: Path):
    cfg_dict = make_minimal_config_dict("staging")
    p = tmp_path / "staging.yml"
    p.write_text(yaml.safe_dump(cfg_dict))

    cfg = load_config("staging", config_path=p)

    assert isinstance(cfg, ForecasterConfig)
    assert cfg.environment == "staging"
    assert cfg.infrastructure.model_bucket == cfg_dict["infrastructure"]["model_bucket"]


def test_from_env_respects_env_vars(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "dev")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("CUR_BUCKET", "cur-bucket-from-env")
    monkeypatch.setenv("ATHENA_OUTPUT_BUCKET", "athena-bucket-from-env")
    monkeypatch.setenv("MODEL_BUCKET", "model-bucket-from-env")

    cfg = ForecasterConfig.from_env()

    assert cfg.environment == "dev"
    assert cfg.aws_region == "eu-west-1"
    assert cfg.data_source.cur_bucket == "cur-bucket-from-env"
    assert cfg.infrastructure.model_bucket == "model-bucket-from-env"


def test_invalid_environment_raises():
    data = make_minimal_config_dict("badenv")
    data["environment"] = "not-a-real-env"
    # Directly call the class-level validator to ensure invalid envs are rejected
    with pytest.raises(ValueError):
        ForecasterConfig.validate_environment("not-a-real-env")
