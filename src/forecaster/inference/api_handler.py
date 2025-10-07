"""
API Handler for Resource Forecasting Service

Provides HTTP/REST API endpoints for cost forecasting and recommendations.
Supports FastAPI framework with async endpoints and proper error handling.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import uvicorn

from ..config import ForecasterConfig
from .forecaster_handler import ForecasterHandler
from .batch_predictor import BatchPredictor
from ..exceptions import PredictionError, DataProcessingError, ModelLoadingError

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Request/Response Models
class PredictionRequest(BaseModel):
    """Request model for forecast predictions."""
    forecast_horizon_days: int = Field(default=30, ge=1, le=365, description="Number of days to forecast")
    model_name: str = Field(default="prophet", description="Model to use for prediction")
    include_recommendations: bool = Field(default=True, description="Include cost optimization recommendations")
    account_id: Optional[str] = Field(default=None, description="AWS account ID filter")
    service_filter: Optional[List[str]] = Field(default=None, description="List of services to include")
    confidence_level: float = Field(default=0.8, ge=0.5, le=0.95, description="Confidence level for predictions")
    
    @field_validator('model_name')
    def validate_model_name(cls, v):
        allowed_models = ['prophet', 'ensemble', 'linear']
        if v not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    requests: List[PredictionRequest] = Field(description="List of prediction requests")
    async_processing: bool = Field(default=True, description="Process asynchronously")
    callback_url: Optional[str] = Field(default=None, description="URL to call when batch job completes")
    
    @field_validator('requests')
    def validate_requests(cls, v):
        if len(v) == 0:
            raise ValueError("At least one prediction request is required")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 requests per batch")
        return v


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    checks: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    request_id: str
    forecast: Dict[str, Any]
    recommendations: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class BatchJobResponse(BaseModel):
    """Response model for batch job submission."""
    job_id: str
    status: str
    total_requests: int
    estimated_completion_time: Optional[str]


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    progress: float
    total_requests: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    has_results: bool


# API Application
def create_app(config: ForecasterConfig) -> FastAPI:
    """Create FastAPI application with all endpoints."""
    
    app = FastAPI(
        title="Resource Forecaster API",
        description="Cost forecasting and optimization recommendations for AWS resources",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize handlers
    forecaster_handler = ForecasterHandler(config)
    batch_predictor = BatchPredictor(config)
    
    # Dependency for authentication
    async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify API token."""
        # Implement your authentication logic here
        # For now, just check if token exists
        if not credentials.credentials:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return credentials.credentials
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        try:
            health_data = forecaster_handler.health_check()
            return HealthResponse(**health_data)
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Health check failed")
    
    @app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
    async def predict(
        request: PredictionRequest,
        token: str = Depends(verify_token)
    ):
        """Generate cost forecast and recommendations."""
        try:
            # Convert Pydantic model to dict
            request_dict = request.model_dump()
            request_dict['request_id'] = f"api_{datetime.now().timestamp()}"
            
            logger.info(f"Processing prediction request: {request_dict['request_id']}")
            
            # Generate prediction
            result = forecaster_handler.predict(request_dict)
            
            return PredictionResponse(**result)
            
        except PredictionError as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in prediction: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.post("/predict/batch", response_model=BatchJobResponse, tags=["Batch Processing"])
    async def submit_batch_prediction(
        request: BatchPredictionRequest,
        background_tasks: BackgroundTasks,
        token: str = Depends(verify_token)
    ):
        """Submit batch prediction job."""
        try:
            # Convert requests to dicts
            request_dicts = [req.model_dump() for req in request.requests]
            
            if request.async_processing:
                # Submit async job
                job_id = batch_predictor.submit_batch_job(request_dicts)
                
                # Estimate completion time (rough estimate: 2 seconds per request)
                estimated_seconds = len(request_dicts) * 2
                estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
                
                return BatchJobResponse(
                    job_id=job_id,
                    status="submitted",
                    total_requests=len(request_dicts),
                    estimated_completion_time=estimated_completion.isoformat()
                )
            else:
                # Process synchronously in background
                job_id = f"sync_{datetime.now().timestamp()}"
                background_tasks.add_task(
                    _process_sync_batch, 
                    batch_predictor, 
                    request_dicts, 
                    job_id
                )
                
                return BatchJobResponse(
                    job_id=job_id,
                    status="processing",
                    total_requests=len(request_dicts),
                    estimated_completion_time=None
                )
                
        except Exception as e:
            logger.error(f"Batch submission error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/jobs/{job_id}/status", response_model=JobStatusResponse, tags=["Batch Processing"])
    async def get_job_status(
        job_id: str,
        token: str = Depends(verify_token)
    ):
        """Get status of a batch job."""
        try:
            status = batch_predictor.get_job_status(job_id)
            return JobStatusResponse(**status)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/jobs/{job_id}/results", tags=["Batch Processing"])
    async def get_job_results(
        job_id: str,
        token: str = Depends(verify_token)
    ):
        """Get results of a completed batch job."""
        try:
            results = batch_predictor.get_job_results(job_id)
            return {"job_id": job_id, "results": results}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting job results: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.post("/jobs/{job_id}/save", tags=["Batch Processing"])
    async def save_job_results(
        job_id: str,
        token: str = Depends(verify_token)
    ):
        """Save job results to S3."""
        try:
            s3_uri = batch_predictor.save_job_results(job_id)
            return {"job_id": job_id, "s3_uri": s3_uri, "status": "saved"}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error saving job results: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/jobs", tags=["Batch Processing"])
    async def list_jobs(
        status: Optional[str] = Query(None, description="Filter by job status"),
        limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
        token: str = Depends(verify_token)
    ):
        """List batch jobs."""
        try:
            jobs = batch_predictor.list_jobs(status_filter=status)
            return {"jobs": jobs[:limit]}
        except Exception as e:
            logger.error(f"Error listing jobs: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.delete("/jobs/{job_id}", tags=["Batch Processing"])
    async def cancel_job(
        job_id: str,
        token: str = Depends(verify_token)
    ):
        """Cancel a batch job."""
        try:
            success = batch_predictor.cancel_job(job_id)
            return {"job_id": job_id, "cancelled": success}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error cancelling job: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/models", tags=["Models"])
    async def list_available_models(token: str = Depends(verify_token)):
        """List available forecasting models."""
        return {
            "models": [
                {
                    "name": "prophet",
                    "description": "Facebook Prophet time-series forecasting",
                    "suitable_for": ["daily", "weekly", "monthly forecasts"],
                    "features": ["seasonality", "holidays", "trend changes"]
                },
                {
                    "name": "ensemble",
                    "description": "Ensemble of Random Forest, Gradient Boosting, and Linear models",
                    "suitable_for": ["complex patterns", "feature-rich data"],
                    "features": ["non-linear patterns", "feature importance", "robust predictions"]
                },
                {
                    "name": "linear",
                    "description": "Linear regression with time-series features",
                    "suitable_for": ["simple trends", "baseline comparisons"],
                    "features": ["interpretable", "fast training", "baseline model"]
                }
            ]
        }
    
    @app.get("/metrics", tags=["Monitoring"])
    async def get_metrics(token: str = Depends(verify_token)):
        """Get service metrics and statistics."""
        try:
            # Get job statistics
            all_jobs = batch_predictor.list_jobs()
            job_stats: Dict[str, Any] = {
                "total_jobs": int(len(all_jobs)),
                "completed_jobs": int(len([j for j in all_jobs if j["status"] == "completed"])),
                "failed_jobs": int(len([j for j in all_jobs if j["status"] == "failed"])),
                "running_jobs": int(len([j for j in all_jobs if j["status"] == "running"]))
            }
            
            # Calculate success rate
            success_rate: float = (
                job_stats["completed_jobs"] / job_stats["total_jobs"]
                if job_stats["total_jobs"] > 0
                else 0.0
            )
            job_stats["success_rate"] = success_rate
            
            return {
                "service_uptime": str(datetime.now()),
                "job_statistics": job_stats,
                "service_version": "1.0.0",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    # Exception handlers
    @app.exception_handler(PredictionError)
    async def prediction_error_handler(request, exc: PredictionError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "type": "PredictionError"}
        )
    
    @app.exception_handler(DataProcessingError)
    async def data_processing_error_handler(request, exc: DataProcessingError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "type": "DataProcessingError"}
        )
    
    @app.exception_handler(ModelLoadingError)
    async def model_loading_error_handler(request, exc: ModelLoadingError):
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "type": "ModelLoadingError"}
        )
    
    return app


async def _process_sync_batch(
    batch_predictor: BatchPredictor, 
    requests: List[Dict[str, Any]], 
    job_id: str
):
    """Process batch synchronously in background."""
    try:
        logger.info(f"Processing sync batch job {job_id}")
        results = batch_predictor.process_batch_sync(requests)
        
        # Store results (in a real implementation, you'd store in database/cache)
        # For now, just log completion
        logger.info(f"Sync batch job {job_id} completed with {len(results)} results")
        
    except Exception as e:
        logger.error(f"Sync batch job {job_id} failed: {str(e)}")


class APIServer:
    """API Server wrapper for easier deployment."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.app = create_app(config)
        
    def run(
        self, 
        host: str = "0.0.0.0", 
        port: int = 8000, 
        reload: bool = False
    ):
        """Run the API server."""
        logger.info(f"Starting Resource Forecaster API server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    def run_async(
        self, 
        host: str = "0.0.0.0", 
        port: int = 8000
    ):
        """Run the API server asynchronously."""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        return server.serve()


# CLI entry point for API server
def main():
    """Main entry point for API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Resource Forecaster API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", default="config/dev.yml", help="Config file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ForecasterConfig.from_yaml(args.config)
    
    # Create and run server
    server = APIServer(config)
    server.run(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()