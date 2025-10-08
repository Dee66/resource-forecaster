import json
import logging
import os
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def lambda_handler(event: Dict[str, Any], context: Any = None, clients: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Nightly reconciliation Lambda.

    Scans the audit DynamoDB table and checks whether recommended resources were implemented.
    If missing implementations are found, publishes a summary to the alerts SNS topic.

    The function supports dependency injection of `clients` for unit tests. Expected keys: 'dynamodb', 'sns', 'ec2'.
    """
    try:
        from forecaster.config import load_config

        cfg = load_config(os.environ.get("ENVIRONMENT", "dev"))
        table_name = os.environ.get("AUDIT_TABLE_NAME") or cfg.infrastructure.data_bucket
        alerts_topic = os.environ.get("ALERTS_TOPIC_ARN") or None
    except Exception:
        table_name = os.environ.get("AUDIT_TABLE_NAME")
        alerts_topic = os.environ.get("ALERTS_TOPIC_ARN")

    dynamodb = clients.get('dynamodb') if clients and 'dynamodb' in clients else boto3.client('dynamodb')
    sns = clients.get('sns') if clients and 'sns' in clients else boto3.client('sns')
    ec2 = clients.get('ec2') if clients and 'ec2' in clients else boto3.client('ec2')
    rds = clients.get('rds') if clients and 'rds' in clients else boto3.client('rds')
    asg = clients.get('autoscaling') if clients and 'autoscaling' in clients else boto3.client('autoscaling')
    cloudwatch = clients.get('cloudwatch') if clients and 'cloudwatch' in clients else boto3.client('cloudwatch')

    if not table_name:
        logger.error("AUDIT_TABLE_NAME not configured")
        return {"status": "error", "reason": "no_audit_table"}

    # Scan the table (lightweight scan for nightly job)
    try:
        resp = dynamodb.scan(TableName=table_name, Limit=1000)
        items = resp.get('Items', [])
    except ClientError as e:
        logger.exception("Failed to scan audit table: %s", e)
        return {"status": "error", "reason": "scan_failed"}

    missing = []

    for it in items:
        # extract resource_id
        resource_id = None
        if isinstance(it.get('resource_id'), dict):
            resource_id = it.get('resource_id').get('S')
        else:
            resource_id = it.get('resource_id')

        if not resource_id:
            continue

        # Heuristics by resource id pattern
        try:
            if resource_id.startswith('i-'):
                # EC2 instance
                try:
                    r = ec2.describe_instances(InstanceIds=[resource_id])
                    reservations = r.get('Reservations', [])
                    if not reservations:
                        missing.append(resource_id)
                except ClientError:
                    missing.append(resource_id)
            elif resource_id.startswith('db-') or resource_id.startswith('rds-'):
                # RDS instance identifiers often start with 'db-' or custom names; try describe_db_instances
                try:
                    r = rds.describe_db_instances(DBInstanceIdentifier=resource_id)
                    instances = r.get('DBInstances', [])
                    if not instances:
                        missing.append(resource_id)
                except ClientError:
                    missing.append(resource_id)
            elif resource_id.startswith('asg-') or 'auto-scaling' in resource_id.lower() or 'autoscaling' in resource_id.lower():
                # Auto Scaling Group
                try:
                    r = asg.describe_auto_scaling_groups(AutoScalingGroupNames=[resource_id])
                    groups = r.get('AutoScalingGroups', [])
                    if not groups:
                        missing.append(resource_id)
                except ClientError:
                    missing.append(resource_id)
            else:
                # Unknown resource type: treat conservatively as missing
                missing.append(resource_id)
        except Exception:
            missing.append(resource_id)

    # Emit a CloudWatch metric with missing count
    try:
        cloudwatch.put_metric_data(
            Namespace='ResourceForecaster/Monitoring',
            MetricData=[
                {
                    'MetricName': 'ReconciliationMissingCount',
                    'Dimensions': [{'Name': 'Environment', 'Value': os.environ.get('ENVIRONMENT', 'dev')}],
                    'Value': len(missing),
                    'Unit': 'Count'
                }
            ]
        )
    except ClientError:
        logger.exception("Failed to put CloudWatch metric")

    # Publish alert if missing items exist
    if missing and alerts_topic:
        message = {
            "missing_count": len(missing),
            "missing": missing[:25]  # cap details
        }
        try:
            sns.publish(TopicArn=alerts_topic, Subject="Reconciliation Alert: Missing Implementations", Message=json.dumps(message))
        except ClientError:
            logger.exception("Failed to publish reconciliation alert")
            return {"status": "error", "reason": "publish_failed", "missing_count": len(missing)}

    return {"status": "ok", "total": len(items), "missing_count": len(missing)}
