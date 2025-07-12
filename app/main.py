from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import time
import logging
import os
from datetime import datetime
from contextlib import asynccontextmanager

# Import our components (we'll create minimal versions)
from app.ai_layer.candidate_generator import CandidateGenerator
from app.ml_layer.ranker import PredictionRanker
from app.utils.openapi_parser import OpenAPIParser
from app.utils.cache import CacheManager
from app.utils.performance import (
    performance_monitor, AsyncPerformanceTracker, 
    optimize_for_performance, with_timeout
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
cache_manager = CacheManager()
openapi_parser = OpenAPIParser(cache_manager)
candidate_generator = CandidateGenerator()
prediction_ranker = PredictionRanker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting API Predictor service...")
    optimize_for_performance()
    await cache_manager.init()
    await prediction_ranker.load_model()
    logger.info("Service initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Predictor service...")
    await cache_manager.close()

app = FastAPI(
    title="API Predictor",
    description="Predicts next API calls for SaaS users",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input/Output Models
class APIEvent(BaseModel):
    ts: str = Field(..., description="Timestamp in ISO format")
    endpoint: str = Field(..., description="API endpoint like 'GET /invoices'")
    params: Dict[str, Any] = Field(default_factory=dict, description="Request parameters")
    
    @validator('ts')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format')

class PredictionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    events: List[APIEvent] = Field(..., description="Recent API events")
    prompt: Optional[str] = Field(None, description="Optional natural language hint")
    spec_url: str = Field(..., description="OpenAPI specification URL")
    k: int = Field(5, ge=1, le=20, description="Number of predictions to return")

class Prediction(BaseModel):
    endpoint: str = Field(..., description="Predicted API endpoint")
    params: Dict[str, Any] = Field(..., description="Suggested parameters")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    why: str = Field(..., description="Human-readable explanation")

class PredictionResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    predictions: List[Prediction]
    processing_time_ms: int
    model_version: str = "1.0.0"

@app.get("/health")
async def health_check():
    """Health check endpoint with performance status"""
    stats = performance_monitor.get_stats(minutes=10)
    is_healthy = performance_monitor.is_healthy()
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "performance": {
            "median_response_ms": stats["median_ms"],
            "p95_response_ms": stats["p95_ms"],
            "total_requests_10min": stats["total_requests"],
            "cache_hit_rate": stats["cache_hit_rate"],
            "error_rate": stats["error_rate"]
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get detailed performance metrics"""
    return {
        "last_hour": performance_monitor.get_stats(minutes=60),
        "last_10_minutes": performance_monitor.get_stats(minutes=10),
        "predict_endpoint": performance_monitor.get_endpoint_stats("/predict", minutes=60)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_api_call(request: PredictionRequest):
    """
    Predict the next API call a user is likely to make
    """
    async with AsyncPerformanceTracker("/predict", request.user_id) as tracker:
        try:
            logger.info(f"Processing prediction request for user {request.user_id}")
            logger.info(f"Events: {len(request.events)}, Prompt: '{request.prompt}', Spec: {request.spec_url}")
            
            # Validate input
            if len(request.events) == 0:
                logger.error("No events provided")
                raise HTTPException(status_code=400, detail="At least one event is required")
            
            logger.info("Step 1: Parsing OpenAPI spec...")
            # Parse OpenAPI spec with timeout
            try:
                spec_data = await with_timeout(
                    openapi_parser.parse_spec(request.spec_url),
                    timeout_seconds=10.0
                )
                if spec_data is None:
                    raise Exception("OpenAPI parsing timed out")
                
                # Check if cache was used (simplified check)
                cache_key = f"openapi_spec:{request.spec_url}"
                cached = await cache_manager.get(cache_key.replace('openapi_spec:', 'openapi_spec:')[:50])
                tracker.cache_hit = cached is not None
                
                logger.info(f"OpenAPI parsed: {spec_data.get('title', 'Unknown')} with {len(spec_data.get('endpoints', []))} endpoints")
            except Exception as e:
                logger.error(f"OpenAPI parsing failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to parse OpenAPI spec: {str(e)}")
            
            logger.info("Step 2: Generating candidates...")
            # Generate candidates using AI layer with timeout
            try:
                candidates = await with_timeout(
                    candidate_generator.generate_candidates(
                        events=request.events,
                        prompt=request.prompt,
                        spec_data=spec_data,
                        k=request.k
                    ),
                    timeout_seconds=15.0,
                    default=[]
                )
                if not candidates:
                    raise Exception("No candidates generated")
                logger.info(f"Generated {len(candidates)} candidates")
            except Exception as e:
                logger.error(f"Candidate generation failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate candidates: {str(e)}")
            
            logger.info("Step 3: Ranking candidates...")
            # Rank candidates using ML layer with timeout
            try:
                ranked_predictions = await with_timeout(
                    prediction_ranker.rank_candidates(
                        candidates=candidates,
                        events=request.events,
                        prompt=request.prompt,
                        user_id=request.user_id
                    ),
                    timeout_seconds=5.0,
                    default=[]
                )
                if not ranked_predictions:
                    raise Exception("No predictions ranked")
                
                # Track if ML was used
                tracker.ml_used = prediction_ranker.is_trained
                
                logger.info(f"Ranked {len(ranked_predictions)} predictions")
            except Exception as e:
                logger.error(f"Ranking failed: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to rank candidates: {str(e)}")
            
            # Return top k predictions
            top_predictions = ranked_predictions[:request.k]
            
            processing_time = int((time.time() - tracker.start_time) * 1000)
            
            logger.info(f"Prediction completed successfully in {processing_time}ms with {len(top_predictions)} predictions")
            
            return PredictionResponse(
                predictions=top_predictions,
                processing_time_ms=processing_time
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            processing_time = int((time.time() - tracker.start_time) * 1000) if tracker.start_time else 0
            logger.error(f"Unexpected error in prediction after {processing_time}ms: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "API Predictor",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)