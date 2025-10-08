#!/usr/bin/env python3
"""Generate CloudWatch dashboards for Resource Forecaster demo.

Creates dashboards for:
- RMSE trending over time by environment
- Cost-per-environment with forecast vs actual
- Model performance metrics
- Budget utilization alerts
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def create_rmse_dashboard(region: str = "us-east-1") -> Dict[str, Any]:
    """Create RMSE trending dashboard configuration."""
    return {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["ResourceForecaster/Metrics", "RMSE", "Environment", "dev"],
                        ["...", "staging"],
                        ["...", "prod"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Forecast RMSE by Environment",
                    "period": 300,
                    "yAxis": {
                        "left": {
                            "min": 0
                        }
                    }
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["ResourceForecaster/Metrics", "MAPE", "Environment", "dev"],
                        ["...", "staging"],
                        ["...", "prod"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Forecast MAPE by Environment (%)",
                    "period": 300
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 24,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["ResourceForecaster/Metrics", "ModelAccuracy", "Environment", "dev"],
                        ["...", "staging"],
                        ["...", "prod"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Model Accuracy Trending (R²)",
                    "period": 300,
                    "yAxis": {
                        "left": {
                            "min": 0,
                            "max": 1
                        }
                    }
                }
            }
        ]
    }


def create_cost_dashboard(region: str = "us-east-1") -> Dict[str, Any]:
    """Create cost tracking dashboard configuration."""
    return {
        "widgets": [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["AWS/Billing", "EstimatedCharges", "Currency", "USD", {"stat": "Maximum"}],
                        ["ResourceForecaster/Cost", "PredictedDailyCost", "Environment", "dev"],
                        ["...", "staging"],
                        ["...", "prod"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Cost: Actual vs Predicted",
                    "period": 86400,
                    "annotations": {
                        "horizontal": [
                            {
                                "label": "Budget Alert Threshold",
                                "value": 1000
                            }
                        ]
                    }
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["ResourceForecaster/Cost", "DailyVariance", "Environment", "dev"],
                        ["...", "staging"],
                        ["...", "prod"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Daily Cost Variance (%)",
                    "period": 86400
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 24,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["ResourceForecaster/Recommendations", "PotentialSavings", "Type", "Rightsizing"],
                        ["...", "ReservedInstances"],
                        ["...", "SavingsPlans"]
                    ],
                    "view": "timeSeries",
                    "stacked": True,
                    "region": region,
                    "title": "Cost Optimization Opportunities ($)",
                    "period": 86400
                }
            }
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CloudWatch dashboards")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--output-dir", default="dashboards", help="Output directory")
    parser.add_argument("--create-dashboards", action="store_true", help="Actually create dashboards in AWS")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate dashboard configurations
    rmse_config = create_rmse_dashboard(args.region)
    cost_config = create_cost_dashboard(args.region)

    # Save configurations
    with open(f"{args.output_dir}/rmse_dashboard.json", "w") as f:
        json.dump(rmse_config, f, indent=2)
    
    with open(f"{args.output_dir}/cost_dashboard.json", "w") as f:
        json.dump(cost_config, f, indent=2)

    print(f"📊 Dashboard configurations saved to {args.output_dir}/")

    # Optionally create actual dashboards
    if args.create_dashboards:
        try:
            import boto3
            cw = boto3.client("cloudwatch", region_name=args.region)
            
            # Create RMSE dashboard
            cw.put_dashboard(
                DashboardName="ResourceForecaster-RMSE-Trending",
                DashboardBody=json.dumps(rmse_config)
            )
            
            # Create cost dashboard  
            cw.put_dashboard(
                DashboardName="ResourceForecaster-Cost-Analysis",
                DashboardBody=json.dumps(cost_config)
            )
            
            print("✅ Created dashboards in CloudWatch")
            print(f"🔗 View at: https://{args.region}.console.aws.amazon.com/cloudwatch/home?region={args.region}#dashboards:")
            
        except Exception as e:
            print(f"⚠️  Failed to create dashboards: {e}")
            print("💡 Tip: Ensure AWS credentials are configured")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
