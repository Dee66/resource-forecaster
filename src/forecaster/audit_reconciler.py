"""Reconciliation utilities for audit events vs actual resource state.

The reconciler compares audit records stored in DynamoDB against the
actual AWS resource state (via an injected checker function). This file
provides a small, testable reconciliation function that can be extended
later to run as a scheduled job or Lambda.
"""
from typing import Callable, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def reconcile_audit_with_resources(dynamodb_client: Any, table_name: str, resource_checker: Callable[[str], bool], limit: int = 1000) -> Dict[str, Any]:
    """Scan the audit table and check whether recommended resources exist.

    Args:
        dynamodb_client: boto3 DynamoDB client or compatible mock with .scan and .get_item
        table_name: name of the audit table
        resource_checker: callable that accepts resource_id and returns True if resource exists/applied
        limit: maximum number of items to scan (safety)

    Returns:
        summary dict with counts and list of missing resources
    """
    # Simple scan (not paginated for this lightweight tool).
    resp = dynamodb_client.scan(TableName=table_name, Limit=limit)
    items = resp.get('Items', [])

    total = len(items)
    implemented = 0
    missing: List[Dict[str, Any]] = []

    for it in items:
        # Expecting item shape with 'resource_id' and 'recommendation_id'
        resource_id = None
        if isinstance(it.get('resource_id'), dict):
            resource_id = it.get('resource_id').get('S')
        else:
            resource_id = it.get('resource_id')

        if not resource_id:
            logger.debug('Skipping item without resource_id: %s', it)
            continue

        try:
            exists = resource_checker(resource_id)
            if exists:
                implemented += 1
            else:
                missing.append({'resource_id': resource_id, 'item': it})
        except Exception as e:
            logger.warning('Resource checker failed for %s: %s', resource_id, str(e))
            missing.append({'resource_id': resource_id, 'error': str(e)})

    return {
        'total': total,
        'implemented': implemented,
        'missing_count': len(missing),
        'missing': missing
    }
