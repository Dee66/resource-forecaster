# CloudWatch Dashboard Screenshots

This directory contains demo screenshots of the Resource Forecaster CloudWatch dashboards.

## Dashboard Overview

### RMSE Trending Dashboard (`rmse_trending_dashboard.png`)
- **Purpose**: Monitor forecast model accuracy over time
- **Key Metrics**:
  - RMSE (Root Mean Square Error) by environment
  - MAPE (Mean Absolute Percentage Error) trending
  - Model accuracy (R²) over time
- **Business Value**: Early detection of model drift requiring retraining

### Cost Analysis Dashboard (`cost_analysis_dashboard.png`)  
- **Purpose**: Track cost forecasting accuracy and optimization opportunities
- **Key Metrics**:
  - Actual vs predicted daily costs
  - Cost variance percentage by environment
  - Potential savings from rightsizing, RIs, and Savings Plans
- **Business Value**: Demonstrates 40% cost reduction potential and forecast reliability

## Usage Notes

- Screenshots are captured automatically via `nox -s demo-screenshots`
- Dashboards auto-refresh every 5 minutes in the AWS console
- Best viewing: 1920x1080 resolution for presentation clarity
- Time range: Last 7 days provides good trending visibility

## Demo Talking Points

1. **Model Reliability**: "Our RMSE consistently stays below 5% threshold"
2. **Cross-Environment**: "Accuracy maintained across dev, staging, and production"
3. **Cost Impact**: "Forecasting enables proactive 40% cost optimization"
4. **Operational Excellence**: "Automated alerting when model accuracy degrades"

## Dashboard URLs

- RMSE Trending: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:ResourceForecaster-RMSE-Trending
- Cost Analysis: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:ResourceForecaster-Cost-Analysis

## Manual Screenshot Instructions

If automated capture fails, manually capture screenshots:

1. Navigate to the CloudWatch console dashboard URLs above
2. Set time range to "Last 7 days" for best trending visibility
3. Ensure browser window is 1920x1080 for consistent presentation
4. Save screenshots as PNG files in this directory

## Automation

The screenshot capture process is automated through:
- `scripts/capture_dashboard_screenshots.py` - Selenium-based screenshot capture
- `nox -s demo_screenshots` - Automated execution with proper dependencies
- Requires chromedriver installation for headless browser automation