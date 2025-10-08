#!/usr/bin/env python3
"""Create CloudWatch alarms, SNS topics and CloudWatch Events for Section 11 demo.

This script can generate CloudWatch alarm JSON for RMSE drift and optionally
create the alarm and an SNS topic in AWS if --apply is passed.

Usage examples:
  python scripts/section11_create_alarms_and_guardrails.py --out-dir dashboards/section11
  python scripts/section11_create_alarms_and_guardrails.py --apply --sns-email ops@example.com
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def make_rmse_drift_alarm(env: str = "prod", threshold: float = 0.05) -> Dict[str, Any]:
    """Return a CloudWatch alarm definition (CloudWatch PutMetricAlarm parameters).

    The alarm watches a custom metric ResourceForecaster/Metrics RMSE for the
    given environment and triggers when the average RMSE over 3 datapoints
    (5min period) exceeds threshold.
    """
    alarm_name = f"ResourceForecaster-RMSE-Drift-{env}"
    return {
        "AlarmName": alarm_name,
        "MetricName": "RMSE",
        "Namespace": "ResourceForecaster/Metrics",
        "Dimensions": [{"Name": "Environment", "Value": env}],
        "Statistic": "Average",
        "Period": 300,
        "EvaluationPeriods": 3,
        "Threshold": threshold,
        "ComparisonOperator": "GreaterThanThreshold",
        "TreatMissingData": "breaching",
        "ActionsEnabled": True,
    }


def save_alarm_definition(out_dir: str, alarm_def: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{alarm_def['AlarmName']}.json")
    with open(path, "w") as fh:
        json.dump(alarm_def, fh, indent=2)
    return path


def create_alarm_in_aws(alarm_def: Dict[str, Any], sns_arn: str | None = None, region: str = "us-east-1") -> None:
    try:
        import boto3
    except Exception as e:
        raise RuntimeError("boto3 is required to create AWS resources") from e

    cw = boto3.client("cloudwatch", region_name=region)
    params = dict(alarmName=alarm_def["AlarmName"],
                  metricName=alarm_def["MetricName"],
                  namespace=alarm_def["Namespace"],
                  statistic=alarm_def.get("Statistic", "Average"),
                  period=alarm_def.get("Period", 300),
                  evaluationPeriods=alarm_def.get("EvaluationPeriods", 3),
                  threshold=alarm_def.get("Threshold", 0.05),
                  comparisonOperator=alarm_def.get("ComparisonOperator", "GreaterThanThreshold"),
                  treatMissingData=alarm_def.get("TreatMissingData", "breaching"),
                  actionsEnabled=alarm_def.get("ActionsEnabled", True),
                  dimensions=alarm_def.get("Dimensions", []))

    response = cw.put_metric_alarm(**params)
    print(f"✅ Created/Updated alarm: {alarm_def['AlarmName']}")
    if sns_arn:
        # Attach alarm actions
        cw.put_metric_alarm(
            **{**params, "AlarmActions": [sns_arn]}
        )
        print(f"🔔 Attached SNS action: {sns_arn}")


def create_sns_topic_and_subscribe(email: str, region: str = "us-east-1") -> str:
    try:
        import boto3
    except Exception as e:
        raise RuntimeError("boto3 is required to create AWS resources") from e

    sns = boto3.client("sns", region_name=region)
    resp = sns.create_topic(Name="ResourceForecaster-Alerts")
    arn = resp["TopicArn"]
    if email:
        sns.subscribe(TopicArn=arn, Protocol="email", Endpoint=email)
        print(f"✉️  Subscribed {email} to SNS topic {arn}. Confirm subscription in email.")
    return arn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="prod")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--out-dir", default="dashboards/section11")
    parser.add_argument("--apply", action="store_true", help="Apply to AWS (requires creds)")
    parser.add_argument("--sns-email", help="Email to subscribe to alert topic when --apply")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args(argv)

    alarm_def = make_rmse_drift_alarm(env=args.env, threshold=args.threshold)
    path = save_alarm_definition(args.out_dir, alarm_def)
    print(f"📄 Alarm definition saved: {path}")

    if args.apply:
        sns_arn = None
        if args.sns_email:
            sns_arn = create_sns_topic_and_subscribe(args.sns_email, region=args.region)
        create_alarm_in_aws(alarm_def, sns_arn, region=args.region)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
