#!/usr/bin/env python3
"""Create sample dashboard images for demo purposes when selenium is not available."""
from __future__ import annotations

import os
from pathlib import Path


def create_sample_dashboard_image(filename: str, dashboard_type: str) -> None:
    """Create a simple text-based representation of dashboard for demo."""
    # For now, create a simple text placeholder
    # In a real scenario, this could generate matplotlib-based sample charts
    content = f"""
# {dashboard_type} Dashboard Sample

This is a placeholder for the {dashboard_type} dashboard screenshot.

## Key Metrics (Sample Data):

### RMSE Trending:
- Dev Environment: 2.3% (Target: <5%)  
- Staging Environment: 1.8% (Target: <5%)
- Production Environment: 2.7% (Target: <5%)

### Cost Analysis:
- Daily Variance: 2.1% (Target: <5%)
- Forecast Accuracy: 97.2%
- Monthly Savings Potential: $20,600

## Dashboard URL:
https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:ResourceForecaster-{dashboard_type.replace(' ', '-')}

Note: Run `nox -s demo_screenshots` to capture live screenshots from CloudWatch.
"""
    
    with open(filename, 'w') as f:
        f.write(content)


def main() -> None:
    """Create sample dashboard files for demo."""
    output_dir = "demo/screenshots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample files
    create_sample_dashboard_image(
        f"{output_dir}/rmse_trending_sample.txt", 
        "RMSE Trending"
    )
    
    create_sample_dashboard_image(
        f"{output_dir}/cost_analysis_sample.txt",
        "Cost Analysis"  
    )
    
    print("📊 Sample dashboard files created in demo/screenshots/")
    print("📸 For live screenshots, run: nox -s demo_screenshots")
    print("🔗 Live dashboards: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:")


if __name__ == "__main__":
    main()