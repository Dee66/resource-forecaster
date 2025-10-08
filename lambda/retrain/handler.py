"""Retrain Lambda handler triggered by RMSE SNS alarm.

Starts the Step Functions state machine to kick off a retrain workflow.
Accepts 'clients' injection for unit tests.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict

logger = logging.getLogger("retrain_lambda")
logger.setLevel(logging.INFO)


def lambda_handler(event: Dict[str, Any], context: Any = None, *, clients: Dict[str, Any] | None = None) -> Dict[str, Any]:
    clients = clients or {}

    sf_client = clients.get("stepfunctions")
    if sf_client is None:
        import boto3

        sf_client = boto3.client("stepfunctions")

    try:
        from forecaster.config import load_config

        cfg = load_config(os.environ.get("ENVIRONMENT", "dev"))
        state_machine_arn = os.environ.get("STATE_MACHINE_ARN") or cfg.infrastructure.data_bucket
    except Exception:
        state_machine_arn = os.getenv("STATE_MACHINE_ARN")
    if not state_machine_arn:
        logger.error("No STATE_MACHINE_ARN configured")
        return {"status": "error", "message": "no state machine configured"}

    # Use timestamp or uuid-based name; simple input
    try:
        response = sf_client.start_execution(
            stateMachineArn=state_machine_arn,
            input=json.dumps({"trigger": "rmse_alarm", "detail": event})
        )
        return {"status": "started", "executionArn": response.get("executionArn")}
    except Exception as exc:
        logger.exception("Failed to start retrain state machine: %s", exc)
        return {"status": "error", "message": str(exc)}
