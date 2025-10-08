"""Sample Lambda handler to shutdown non-prod resources based on forecaster recommendations.

This is a template lambda showing how permissions and logic should be structured.
It expects an event containing a list of resource ARNs to stop/terminate.
"""
from __future__ import annotations

import json
import os
from typing import Any


def stop_resources(resource_arns: list[str]) -> dict[str, Any]:
    # Minimal example: call AWS SDK to stop instances or scale down services.
    # This function intentionally does not include actual boto3 calls to avoid
    # accidental destruction when run locally. Replace with calls like ec2.stop_instances
    results = {arn: "stopped-simulated" for arn in resource_arns}
    return results


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    # Example event: { "resources": ["arn:aws:ec2:...instance/i-...", "arn:aws:rds:..." ] }
    resources = event.get("resources", [])
    if not resources:
        return {"status": "no-resources-provided"}

    results = stop_resources(resources)
    return {"status": "success", "results": results}


if __name__ == "__main__":
    # quick local smoke test
    event = {"resources": ["arn:aws:ec2:example:instance/i-12345"]}
    print(json.dumps(lambda_handler(event, None), indent=2))
