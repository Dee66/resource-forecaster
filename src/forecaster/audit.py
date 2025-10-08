"""Audit helpers for writing recommendation events to DynamoDB."""
from typing import Any, Dict, Optional
import time
import logging

import boto3

logger = logging.getLogger(__name__)


def write_audit_event(table_name: str, event: Dict[str, Any], dynamodb_client: Optional[Any] = None, retries: int = 3) -> bool:
    """Write an audit event to DynamoDB with simple retry logic.

    Args:
        table_name: DynamoDB table name
        event: Dict representing the audit event. Must include 'recommendation_id' and 'timestamp'
        dynamodb_client: Optional boto3 DynamoDB client (for DI/testing)
        retries: Number of retries on failure

    Returns:
        True on success, False otherwise
    """
    if dynamodb_client is None:
        dynamodb_client = boto3.client('dynamodb')

    item = {k: {'S': str(v)} if not isinstance(v, dict) and not isinstance(v, list) else {'S': str(v)} for k, v in event.items()}

    attempt = 0
    while attempt < retries:
        try:
            dynamodb_client.put_item(TableName=table_name, Item=item)
            logger.info(f"Wrote audit event to {table_name}: {event.get('recommendation_id')}")
            return True
        except Exception as e:
            logger.warning(f"Failed to write audit event (attempt {attempt+1}): {e}")
            attempt += 1
            time.sleep(0.1 * attempt)

    logger.error(f"Failed to write audit event after {retries} attempts: {event.get('recommendation_id')}")
    return False
