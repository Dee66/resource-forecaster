"""Shutdown Lambda handler for non-prod resource recommendations.

Conservative, testable implementation. Supports:
- EC2 stop by instance id (i-...)
- RDS stop by instance identifier (db-... or arn ending in db:name)
- ECS service scale-to-zero when ARN contains /service/

Accepts `clients` dict for injection of boto3-like clients in unit tests.
"""
from __future__ import annotations

import os
import logging
from typing import Any, Dict, List
import time
import uuid

logger = logging.getLogger("shutdown_lambda")
logger.setLevel(logging.INFO)


def _split_resource_id(resource: str) -> str:
    if ":instance/" in resource:
        return resource.split("instance/")[-1]
    if resource.startswith("i-"):
        return resource
    if ":db:" in resource or ":rds:" in resource:
        return resource.split(":")[-1]
    if "/service/" in resource:
        return resource.split("/service/")[-1]
    return resource


def stop_ec2_instances(instance_ids: List[str], dry_run: bool = True, ec2_client: Any | None = None) -> Dict[str, Any]:
    if dry_run:
        logger.info("DRY RUN: stop EC2 instances %s", instance_ids)
        return {i: "dry-run" for i in instance_ids}

    if ec2_client is None:
        import boto3

        ec2_client = boto3.client("ec2")

    return ec2_client.stop_instances(InstanceIds=instance_ids)


def stop_rds_instance(db_identifier: str, dry_run: bool = True, rds_client: Any | None = None) -> Dict[str, Any]:
    if dry_run:
        logger.info("DRY RUN: stop RDS instance %s", db_identifier)
        return {db_identifier: "dry-run"}

    if rds_client is None:
        import boto3

        rds_client = boto3.client("rds")

    return rds_client.stop_db_instance(DBInstanceIdentifier=db_identifier)


def scale_ecs_service_to_zero(service_name: str, cluster: str | None = None, dry_run: bool = True, ecs_client: Any | None = None) -> Dict[str, Any]:
    if dry_run:
        logger.info("DRY RUN: scale ECS service %s to 0 (cluster=%s)", service_name, cluster)
        return {service_name: "dry-run"}

    if ecs_client is None:
        import boto3

        ecs_client = boto3.client("ecs")

    params = {"service": service_name, "desiredCount": 0}
    if cluster:
        params["cluster"] = cluster

    return ecs_client.update_service(**params)


def lambda_handler(event: Dict[str, Any], context: Any = None, *, dry_run_override: bool | None = None, clients: Dict[str, Any] | None = None) -> Dict[str, Any]:
    dry_run_env = os.getenv("DRY_RUN", "true").lower()
    dry_run = True if dry_run_env in ("1", "true", "yes") else False
    if dry_run_override is not None:
        dry_run = dry_run_override

    clients = clients or {}

    resources = event.get("resources") or []
    if not isinstance(resources, list):
        return {"status": "error", "message": "resources must be a list"}

    results: Dict[str, Any] = {}
    ec2_ids: List[str] = []
    rds_ids: List[str] = []
    ecs_services: List[str] = []

    for r in resources:
        rid = _split_resource_id(r)
        if rid.startswith("i-"):
            ec2_ids.append(rid)
        elif rid.startswith("db-") or rid.startswith("rds-") or ("db" in rid and not rid.startswith("arn:")):
            rds_ids.append(rid)
        elif "/" in r and "service" in r:
            ecs_services.append(rid)
        else:
            results[r] = {"status": "unhandled", "id": rid}

    if ec2_ids:
        ec2_client = clients.get("ec2")
        for i in ec2_ids:
            results[i] = stop_ec2_instances([i], dry_run=dry_run, ec2_client=ec2_client)

    if rds_ids:
        rds_client = clients.get("rds")
        for db in rds_ids:
            results[db] = stop_rds_instance(db, dry_run=dry_run, rds_client=rds_client)

    if ecs_services:
        ecs_client = clients.get("ecs")
        for svc in ecs_services:
            results[svc] = scale_ecs_service_to_zero(svc, dry_run=dry_run, ecs_client=ecs_client)

    # Audit persistence
    # Prefer repository config for audit table and flags when available
    try:
        from forecaster.config import load_config

        cfg = load_config(os.environ.get("ENVIRONMENT", "dev"))
        audit_enabled = True if getattr(cfg.finops, "rightsizing_enabled", True) else False
        audit_force = False
        audit_table = os.environ.get("AUDIT_TABLE_NAME") or cfg.infrastructure.data_bucket
    except Exception:
        audit_enabled = os.getenv("AUDIT_ENABLED", "true").lower() in ("1", "true", "yes")
        audit_force = os.getenv("AUDIT_FORCE", "false").lower() in ("1", "true", "yes")
        audit_table = os.getenv("AUDIT_TABLE_NAME")

    # Build basic audit event for this invocation
    audit_event = {
        "recommendation_id": str(uuid.uuid4()),
        "timestamp": int(time.time()),
        "env": os.getenv("ENV", os.environ.get("ENVIRONMENT", "dev")),
        "resources": resources,
        "dry_run": dry_run,
        "results": results,
    }

    if audit_enabled and (not dry_run or audit_force) and audit_table:
        dynamodb_client = clients.get("dynamodb")
        try:
            write_audit_event(audit_table, audit_event, dynamodb_client)
            results["_audit"] = {"status": "stored", "id": audit_event["recommendation_id"]}
        except Exception as exc:  # keep generic to let tests mock exceptions
            logger.exception("Failed to write audit event: %s", exc)
            results["_audit"] = {"status": "error", "error": str(exc)}
    return {"status": "success", "dry_run": dry_run, "results": results}


if __name__ == "__main__":
    print(lambda_handler({"resources": ["i-0123456789abcdef0", "arn:aws:rds:us-east-1:123456789012:db:mydb"]}, dry_run_override=True))


def write_audit_event(table_name: str, event: Dict[str, Any], dynamodb_client: Any | None = None, retries: int = 3) -> None:
    """Write audit event to DynamoDB table. Retries on transient errors."""
    if dynamodb_client is None:
        import boto3

        dynamodb_client = boto3.client("dynamodb")

    # Prepare item for DynamoDB PutItem (stringify numbers as needed)
    item = {
        "recommendation_id": {"S": event["recommendation_id"]},
        "timestamp": {"N": str(event["timestamp"])},
        "env": {"S": event.get("env", "")},
        "payload": {"S": str(event)},
    }

    attempt = 0
    while attempt < retries:
        try:
            dynamodb_client.put_item(TableName=table_name, Item=item)
            return
        except Exception:
            attempt += 1
            if attempt >= retries:
                raise
            time.sleep(0.1 * attempt)