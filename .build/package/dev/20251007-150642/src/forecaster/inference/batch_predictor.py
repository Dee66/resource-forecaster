"""
Batch Predictor for Resource Forecasting

Handles large-scale batch prediction jobs for cost forecasting.
Supports parallel processing, job scheduling, and result aggregation.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError

from ..config import ForecasterConfig
from .forecaster_handler import ForecasterHandler
from ..exceptions import PredictionError, DataProcessingError

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a batch prediction job."""
    job_id: str
    requests: List[Dict[str, Any]]
    status: str = 'pending'
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BatchJobManager:
    """Manages batch prediction jobs and their lifecycle."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.jobs: Dict[str, BatchJob] = {}
        self.s3_client = boto3.client('s3')
        
    def create_job(self, requests: List[Dict[str, Any]]) -> str:
        """Create a new batch job.
        
        Args:
            requests: List of prediction requests
            
        Returns:
            Job ID
        """
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.jobs)}"
        
        job = BatchJob(
            job_id=job_id,
            requests=requests
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created batch job {job_id} with {len(requests)} requests")
        
        return job_id
        
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job = self.jobs[job_id]
        
        return {
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'total_requests': len(job.requests),
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'error_message': job.error_message,
            'has_results': job.results is not None
        }
        
    def get_job_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Get the results of a completed batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of prediction results
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job = self.jobs[job_id]
        
        if job.status != 'completed':
            raise ValueError(f"Job {job_id} is not completed (status: {job.status})")
            
        return job.results or []
        
    def save_job_results(self, job_id: str) -> str:
        """Save job results to S3.
        
        Args:
            job_id: Job identifier
            
        Returns:
            S3 URI of saved results
        """
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job = self.jobs[job_id]
        
        if job.results is None:
            raise ValueError(f"Job {job_id} has no results to save")
            
        try:
            # Prepare results data
            results_data = {
                'job_id': job_id,
                'status': job.status,
                'created_at': job.created_at.isoformat(),
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'total_requests': len(job.requests),
                'results': job.results
            }
            
            # Save to S3
            key = f"batch_results/{job_id}/results.json"
            
            self.s3_client.put_object(
                Bucket=self.config.storage.model_bucket,
                Key=key,
                Body=json.dumps(results_data, indent=2),
                ContentType='application/json',
                Metadata={
                    'job_id': job_id,
                    'total_requests': str(len(job.requests)),
                    'created_at': job.created_at.isoformat()
                }
            )
            
            s3_uri = f"s3://{self.config.storage.model_bucket}/{key}"
            logger.info(f"Job results saved to {s3_uri}")
            
            return s3_uri
            
        except Exception as e:
            raise PredictionError(f"Failed to save job results: {str(e)}")


class BatchPredictor:
    """Handles batch prediction operations with parallel processing."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.job_manager = BatchJobManager(config)
        self.forecaster_handler = ForecasterHandler(config)
        self.max_workers = config.batch.max_workers if hasattr(config, 'batch') else 4
        
    def submit_batch_job(
        self, 
        requests: List[Dict[str, Any]],
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Submit a batch prediction job.
        
        Args:
            requests: List of prediction requests
            callback: Optional callback function called when job completes
            
        Returns:
            Job ID
        """
        job_id = self.job_manager.create_job(requests)
        
        # Start job processing asynchronously
        asyncio.create_task(self._process_job_async(job_id, callback))
        
        return job_id
        
    async def _process_job_async(
        self, 
        job_id: str,
        callback: Optional[Callable[[str], None]] = None
    ):
        """Process batch job asynchronously.
        
        Args:
            job_id: Job identifier
            callback: Optional callback function
        """
        try:
            job = self.job_manager.jobs[job_id]
            job.status = 'running'
            job.started_at = datetime.now()
            
            logger.info(f"Starting batch job {job_id}")
            
            # Process requests in parallel
            results = await self._process_requests_parallel(job.requests, job_id)
            
            # Update job with results
            job.results = results
            job.status = 'completed'
            job.completed_at = datetime.now()
            job.progress = 1.0
            
            logger.info(f"Batch job {job_id} completed successfully")
            
            # Call callback if provided
            if callback:
                callback(job_id)
                
        except Exception as e:
            job = self.job_manager.jobs[job_id]
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"Batch job {job_id} failed: {str(e)}")
            
    async def _process_requests_parallel(
        self, 
        requests: List[Dict[str, Any]],
        job_id: str
    ) -> List[Dict[str, Any]]:
        """Process prediction requests in parallel.
        
        Args:
            requests: List of prediction requests
            job_id: Job identifier for progress tracking
            
        Returns:
            List of prediction results
        """
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all requests
            future_to_request = {
                executor.submit(self._process_single_request, request, i): (request, i)
                for i, request in enumerate(requests)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_request):
                request, request_index = future_to_request[future]
                
                try:
                    result = future.result()
                    result['request_index'] = request_index
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Request {request_index} failed: {str(e)}")
                    results.append({
                        'request_index': request_index,
                        'error': str(e),
                        'status': 'failed',
                        'request_id': request.get('request_id', f'req_{request_index}')
                    })
                    
                # Update progress
                completed += 1
                progress = completed / len(requests)
                self.job_manager.jobs[job_id].progress = progress
                
                if completed % 10 == 0:  # Log progress every 10 completions
                    logger.info(f"Job {job_id} progress: {completed}/{len(requests)} ({progress:.1%})")
                    
        # Sort results by request index to maintain order
        results.sort(key=lambda x: x.get('request_index', 0))
        
        return results
        
    def _process_single_request(self, request: Dict[str, Any], request_index: int) -> Dict[str, Any]:
        """Process a single prediction request.
        
        Args:
            request: Prediction request
            request_index: Index of the request in the batch
            
        Returns:
            Prediction result
        """
        try:
            # Add request metadata
            request['request_id'] = request.get('request_id', f'batch_req_{request_index}')
            request['batch_index'] = request_index
            
            # Process request
            result = self.forecaster_handler.predict(request)
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process request {request_index}: {str(e)}")
            return {
                'request_id': request.get('request_id', f'batch_req_{request_index}'),
                'batch_index': request_index,
                'error': str(e),
                'status': 'failed'
            }
            
    def process_batch_sync(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch requests synchronously.
        
        Args:
            requests: List of prediction requests
            
        Returns:
            List of prediction results
        """
        logger.info(f"Processing batch synchronously with {len(requests)} requests")
        
        results = []
        
        for i, request in enumerate(requests):
            try:
                request['request_id'] = request.get('request_id', f'sync_req_{i}')
                result = self.forecaster_handler.predict(request)
                result['batch_index'] = i
                result['status'] = 'success'
                results.append(result)
                
            except Exception as e:
                logger.error(f"Request {i} failed: {str(e)}")
                results.append({
                    'request_id': request.get('request_id', f'sync_req_{i}'),
                    'batch_index': i,
                    'error': str(e),
                    'status': 'failed'
                })
                
        logger.info(f"Batch processing completed. {len([r for r in results if r.get('status') == 'success'])} successful")
        
        return results
        
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        return self.job_manager.get_job_status(job_id)
        
    def get_job_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Get the results of a completed batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of prediction results
        """
        return self.job_manager.get_job_results(job_id)
        
    def save_job_results(self, job_id: str) -> str:
        """Save job results to S3.
        
        Args:
            job_id: Job identifier
            
        Returns:
            S3 URI of saved results
        """
        return self.job_manager.save_job_results(job_id)
        
    def list_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all batch jobs.
        
        Args:
            status_filter: Optional status filter ('pending', 'running', 'completed', 'failed')
            
        Returns:
            List of job information
        """
        jobs = []
        
        for job_id, job in self.job_manager.jobs.items():
            if status_filter is None or job.status == status_filter:
                jobs.append({
                    'job_id': job_id,
                    'status': job.status,
                    'progress': job.progress,
                    'total_requests': len(job.requests),
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None
                })
                
        return sorted(jobs, key=lambda x: x['created_at'], reverse=True)
        
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled successfully
        """
        if job_id not in self.job_manager.jobs:
            raise ValueError(f"Job {job_id} not found")
            
        job = self.job_manager.jobs[job_id]
        
        if job.status in ['completed', 'failed']:
            raise ValueError(f"Cannot cancel job {job_id} with status {job.status}")
            
        job.status = 'cancelled'
        job.completed_at = datetime.now()
        
        logger.info(f"Job {job_id} cancelled")
        return True
        
    def cleanup_old_jobs(self, days_old: int = 7) -> int:
        """Clean up old completed jobs.
        
        Args:
            days_old: Number of days after which to clean up jobs
            
        Returns:
            Number of jobs cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        jobs_to_remove = []
        
        for job_id, job in self.job_manager.jobs.items():
            if (job.status in ['completed', 'failed', 'cancelled'] and 
                job.completed_at and job.completed_at < cutoff_date):
                jobs_to_remove.append(job_id)
                
        for job_id in jobs_to_remove:
            del self.job_manager.jobs[job_id]
            
        logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
        return len(jobs_to_remove)


class ScheduledBatchProcessor:
    """Handles scheduled batch processing jobs."""
    
    def __init__(self, config: ForecasterConfig):
        self.config = config
        self.batch_predictor = BatchPredictor(config)
        self.scheduler_active = False
        
    async def start_scheduler(self):
        """Start the job scheduler."""
        self.scheduler_active = True
        logger.info("Batch job scheduler started")
        
        while self.scheduler_active:
            try:
                await self._process_scheduled_jobs()
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Scheduler error: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
                
    def stop_scheduler(self):
        """Stop the job scheduler."""
        self.scheduler_active = False
        logger.info("Batch job scheduler stopped")
        
    async def _process_scheduled_jobs(self):
        """Process any scheduled jobs."""
        # This would typically check a job queue (SQS, database, etc.)
        # For now, just log that we're checking
        logger.debug("Checking for scheduled batch jobs")
        
        # Example: Daily cost forecasting for all accounts
        current_hour = datetime.now().hour
        if current_hour == 6:  # Run at 6 AM daily
            await self._run_daily_forecasts()
            
    async def _run_daily_forecasts(self):
        """Run daily forecasts for all configured accounts."""
        logger.info("Running daily forecasts")
        
        # Example requests for different accounts/services
        daily_requests = [
            {
                'forecast_horizon_days': 30,
                'model_name': 'prophet',
                'include_recommendations': True,
                'account_id': 'all',
                'service_filter': None
            },
            {
                'forecast_horizon_days': 7,
                'model_name': 'ensemble',
                'include_recommendations': True,
                'account_id': 'all',
                'service_filter': ['Amazon Elastic Compute Cloud - Compute']
            }
        ]
        
        job_id = self.batch_predictor.submit_batch_job(
            daily_requests,
            callback=self._daily_forecast_callback
        )
        
        logger.info(f"Daily forecast job submitted: {job_id}")
        
    def _daily_forecast_callback(self, job_id: str):
        """Callback for daily forecast completion."""
        try:
            # Save results to S3
            s3_uri = self.batch_predictor.save_job_results(job_id)
            
            # Could send notifications, update dashboards, etc.
            logger.info(f"Daily forecast {job_id} completed and saved to {s3_uri}")
            
        except Exception as e:
            logger.error(f"Daily forecast callback failed: {str(e)}")