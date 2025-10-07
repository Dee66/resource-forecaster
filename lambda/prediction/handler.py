"""
Lambda handler for real-time predictions.

Handles API Gateway requests for cost forecasting and recommendations.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Set up logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Lambda handler for prediction requests.
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        # Log the incoming request
        logger.info(f"Received prediction request: {json.dumps(event, default=str)}")
        
        # Extract request data
        if "body" in event and event["body"]:
            request_data = json.loads(event["body"])
        else:
            request_data = event.get("input", {})
            
        # Handle different actions
        action = request_data.get("action", "predict")
        
        if action == "predict":
            response = handle_prediction(request_data)
        elif action == "collect_data":
            response = handle_data_collection(request_data)
        elif action == "generate_forecast":
            response = handle_forecast_generation(request_data)
        elif action == "send_alert":
            response = handle_alert_sending(request_data)
        else:
            raise ValueError(f"Unknown action: {action}")
            
        # Return successful response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization"
            },
            "body": json.dumps(response, default=str)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        }


def handle_prediction(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle prediction requests."""
    try:
        # Import here to avoid cold start penalty
        from forecaster.inference import ForecasterHandler
        from forecaster.config import ForecasterConfig
        
        # Load configuration
        config = ForecasterConfig.from_env()
        
        # Initialize handler
        handler = ForecasterHandler(config)
        
        # Process prediction
        result = handler.predict(request_data)
        
        logger.info(f"Prediction completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def handle_data_collection(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle data collection for Step Functions."""
    try:
        # Import here to avoid cold start penalty
        from forecaster.data import CURDataCollector, CloudWatchCollector
        from forecaster.config import ForecasterConfig
        from datetime import datetime, timedelta
        
        # Load configuration
        config = ForecasterConfig.from_env()
        
        # Initialize collectors
        cur_collector = CURDataCollector(config)
        cw_collector = CloudWatchCollector(config)
        
        # Collect data for the last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # Collect CUR data
        logger.info("Collecting CUR data")
        cur_data = cur_collector.collect_cost_data(start_date, end_date)
        
        # Collect CloudWatch metrics
        logger.info("Collecting CloudWatch metrics")
        cw_data = cw_collector.collect_cost_metrics(start_date, end_date)
        
        result = {
            "status": "completed",
            "cur_records": len(cur_data) if not cur_data.empty else 0,
            "cloudwatch_records": len(cw_data) if not cw_data.empty else 0,
            "collection_date": datetime.now().isoformat()
        }
        
        logger.info(f"Data collection completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        raise


def handle_forecast_generation(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle forecast generation for Step Functions."""
    try:
        # Import here to avoid cold start penalty
        from forecaster.inference import ForecasterHandler
        from forecaster.config import ForecasterConfig
        
        # Load configuration
        config = ForecasterConfig.from_env()
        
        # Initialize handler
        handler = ForecasterHandler(config)
        
        # Generate forecast
        forecast_request = {
            "forecast_horizon_days": request_data.get("horizon_days", 30),
            "model_name": "prophet",
            "include_recommendations": True
        }
        
        result = handler.predict(forecast_request)
        
        # Check for anomalies
        anomaly_score = 0.0
        if result.get("recommendations", {}).get("cost_anomalies"):
            anomaly_count = len(result["recommendations"]["cost_anomalies"])
            anomaly_score = min(1.0, anomaly_count / 10.0)  # Normalize to 0-1
            
        result["anomaly_score"] = anomaly_score
        
        logger.info(f"Forecast generation completed. Anomaly score: {anomaly_score}")
        return result
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        raise


def handle_alert_sending(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle alert sending for Step Functions."""
    try:
        import boto3
        
        # Initialize SNS client
        sns = boto3.client('sns')
        
        # Prepare alert message
        anomaly_score = request_data.get("anomaly_score", 0.0)
        forecast_data = request_data.get("forecast", {})
        
        message = {
            "alert_type": "cost_anomaly",
            "anomaly_score": anomaly_score,
            "forecast_summary": forecast_data.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send SNS notification (would need SNS topic ARN from environment)
        topic_arn = os.environ.get("ALERT_TOPIC_ARN")
        if topic_arn:
            sns.publish(
                TopicArn=topic_arn,
                Message=json.dumps(message, indent=2),
                Subject=f"Cost Anomaly Alert - Score: {anomaly_score:.2f}"
            )
            
            logger.info(f"Alert sent to SNS topic: {topic_arn}")
        else:
            logger.warning("No SNS topic configured for alerts")
            
        return {
            "status": "alert_sent",
            "anomaly_score": anomaly_score,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alert sending failed: {str(e)}")
        raise