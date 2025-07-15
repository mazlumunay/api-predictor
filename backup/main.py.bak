from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import time
import logging
import os
import asyncio
import hashlib
from datetime import datetime
from contextlib import asynccontextmanager

# Import our components
from app.ai_layer.candidate_generator import CandidateGenerator
from app.ml_layer.ranker import PredictionRanker
from app.utils.openapi_parser import OpenAPIParser
from app.utils.cache import CacheManager
from app.utils.performance import (
    performance_monitor, AsyncPerformanceTracker, 
    optimize_for_performance, with_timeout
)
from app.utils.validation import InputValidator, ParameterGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
cache_manager = CacheManager()
openapi_parser = OpenAPIParser(cache_manager)
candidate_generator = CandidateGenerator()
prediction_ranker = PredictionRanker()
input_validator = InputValidator()
parameter_generator = ParameterGenerator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting optimized API Predictor service...")
    optimize_for_performance()
    await cache_manager.init()
    await prediction_ranker.load_model()
    
    # Start background cleanup task
    asyncio.create_task(periodic_cleanup())
    
    logger.info("Service initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Predictor service...")
    await cache_manager.close()

async def periodic_cleanup():
    """Background task to clean up caches periodically"""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            await cache_manager.cleanup_expired()
            logger.debug("Performed periodic cache cleanup")
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

app = FastAPI(
    title="API Predictor",
    description="Predicts next API calls for SaaS users",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse  # Faster JSON serialization
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance middleware
@app.middleware("http")
async def performance_middleware(request, call_next):
    # Skip detailed processing for health checks and metrics
    if request.url.path in ["/health", "/metrics", "/"]:
        return await call_next(request)
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Only log slow requests to reduce overhead
    if process_time > 1.0:
        logger.warning(f"Slow request: {request.url.path} took {process_time:.2f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

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
    cached: Optional[bool] = False

@app.get("/health")
async def health_check():
    """Optimized health check endpoint"""
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
async def predict_next_api_call(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Optimized prediction endpoint with aggressive caching and performance improvements
    """
    async with AsyncPerformanceTracker("/predict", request.user_id) as tracker:
        try:
            start_time = time.time()
            logger.info(f"Processing prediction request for user {request.user_id}")
            logger.info(f"Events: {len(request.events)}, Prompt: '{request.prompt}', Spec: {request.spec_url}")
            
            # Step 1: Fast input validation
            request_dict = request.dict()
            is_valid, validation_errors = input_validator.validate_request(request_dict)
            if not is_valid:
                logger.error(f"Validation failed: {validation_errors}")
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "message": "Invalid request data",
                        "errors": validation_errors
                    }
                )
            
            # Step 2: Check prediction cache first (fastest path)
            events_hash = cache_manager.generate_events_hash(request.events)
            prompt_hash = cache_manager.generate_prompt_hash(request.prompt)
            
            cached_prediction = await cache_manager.get_cached_prediction(
                request.user_id, events_hash, prompt_hash
            )
            
            if cached_prediction:
                processing_time = int((time.time() - start_time) * 1000)
                logger.info(f"Returning cached prediction in {processing_time}ms")
                tracker.cache_hit = True
                
                return PredictionResponse(
                    predictions=cached_prediction["predictions"][:request.k],
                    processing_time_ms=processing_time,
                    cached=True
                )
            
            logger.info("Step 1: Parsing OpenAPI spec...")
            # Step 3: Parse OpenAPI spec with optimized timeout
            try:
                spec_data = await with_timeout(
                    openapi_parser.parse_spec(request.spec_url),
                    timeout_seconds=8.0  # Reduced from 10
                )
                if spec_data is None:
                    raise Exception("OpenAPI parsing timed out")
                
                # Check if cache was used (simplified check)
                cache_key = f"openapi_spec:{hashlib.md5(request.spec_url.encode()).hexdigest()}"
                cached = await cache_manager.get(cache_key)
                tracker.cache_hit = cached is not None
                
                logger.info(f"OpenAPI parsed: {spec_data.get('title', 'Unknown')} with {len(spec_data.get('endpoints', []))} endpoints")
            except Exception as e:
                logger.error(f"OpenAPI parsing failed: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail={
                        "message": "Failed to parse OpenAPI specification",
                        "error": str(e),
                        "suggestion": "Please check that the spec_url is valid and accessible"
                    }
                )
            
            logger.info("Step 2: Generating candidates...")
            # Step 4: Generate candidates with optimized timeout
            try:
                candidates = await with_timeout(
                    candidate_generator.generate_candidates(
                        events=request.events,
                        prompt=request.prompt,
                        spec_data=spec_data,
                        k=request.k
                    ),
                    timeout_seconds=10.0,  # Reduced from 15
                    default=[]
                )
                if not candidates:
                    raise Exception("No candidates generated - this may indicate an issue with the AI layer or spec parsing")
                logger.info(f"Generated {len(candidates)} candidates")
            except Exception as e:
                logger.error(f"Candidate generation failed: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail={
                        "message": "Failed to generate prediction candidates",
                        "error": str(e),
                        "suggestion": "Try simplifying your prompt or check if OpenAI API key is configured"
                    }
                )
            
            logger.info("Step 3: Ranking candidates...")
            # Step 5: Rank candidates with optimized timeout
            try:
                ranked_predictions = await with_timeout(
                    prediction_ranker.rank_candidates(
                        candidates=candidates,
                        events=request.events,
                        prompt=request.prompt,
                        user_id=request.user_id,
                        spec_data=spec_data
                    ),
                    timeout_seconds=4.0,  # Reduced from 5
                    default=[]
                )
                if not ranked_predictions:
                    raise Exception("No predictions ranked - this may indicate an issue with the ML layer")
                
                # Track if ML was used
                tracker.ml_used = prediction_ranker.is_trained
                
                logger.info(f"Ranked {len(ranked_predictions)} predictions")
            except Exception as e:
                logger.error(f"Ranking failed: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail={
                        "message": "Failed to rank prediction candidates",
                        "error": str(e),
                        "suggestion": "The system will retry with fallback ranking methods"
                    }
                )
            
            logger.info("Step 4: Enhancing predictions with parameters...")
            # Step 6: Parameter enhancement (background task for speed)
            background_tasks.add_task(
                enhance_predictions_background,
                ranked_predictions,
                spec_data,
                request.events
            )
            
            # Return top k predictions
            top_predictions = ranked_predictions[:request.k]
            processing_time = int((time.time() - start_time) * 1000)
            
            # Cache result for future requests (background task)
            prediction_result = {
                "predictions": top_predictions,
                "processing_time_ms": processing_time
            }
            
            background_tasks.add_task(
                cache_manager.cache_prediction,
                request.user_id,
                events_hash,
                prompt_hash,
                prediction_result
            )
            
            logger.info(f"Prediction completed successfully in {processing_time}ms with {len(top_predictions)} predictions")
            
            return PredictionResponse(
                predictions=top_predictions,
                processing_time_ms=processing_time,
                cached=False
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            processing_time = int((time.time() - tracker.start_time) * 1000) if tracker.start_time else 0
            logger.error(f"Unexpected error in prediction after {processing_time}ms: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail={
                    "message": "Internal server error occurred during prediction",
                    "error": str(e),
                    "processing_time_ms": processing_time
                }
            )

async def enhance_predictions_background(predictions: List[Dict], spec_data: Dict, events: List):
    """Background task to enhance predictions with parameters"""
    try:
        user_context = parameter_generator.extract_user_context(events)
        
        for prediction in predictions:
            # Find matching endpoint data
            endpoint_data = None
            for ep in spec_data.get('endpoints', []):
                if ep['endpoint'] == prediction['endpoint']:
                    endpoint_data = ep
                    break
            
            if endpoint_data:
                # Generate enhanced parameters
                enhanced_params = parameter_generator.generate_parameters(
                    endpoint_data, user_context
                )
                # Merge with existing params, prioritizing existing ones
                prediction['params'] = {**enhanced_params, **prediction.get('params', {})}
        
        logger.debug("Enhanced predictions with generated parameters")
    except Exception as e:
        logger.warning(f"Background parameter enhancement failed: {e}")

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "API Predictor (Optimized)",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "performance_optimizations": [
            "Aggressive caching (2-hour TTL)",
            "Optimized OpenAI calls", 
            "Background processing",
            "Reduced timeouts",
            "Fast JSON serialization",
            "Feature caching",
            "Prediction result caching"
        ],
        "endpoints": {
            "predict": "/predict",
            "health": "/health", 
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/cache/status")
async def cache_status():
    """Get cache status and statistics"""
    try:
        # Get cache statistics
        in_memory_count = len(cache_manager.in_memory_cache)
        prediction_cache_count = len(getattr(cache_manager, 'prediction_cache', {}))
        
        # Try to get Redis info
        redis_status = "unavailable"
        redis_memory = "N/A"
        
        if cache_manager.redis_client:
            try:
                await cache_manager.redis_client.ping()
                redis_status = "connected"
                info = await cache_manager.redis_client.info("memory")
                redis_memory = f"{info.get('used_memory_human', 'N/A')}"
            except:
                redis_status = "error"
        
        return {
            "in_memory_cache": {
                "entries": in_memory_count,
                "status": "active"
            },
            "prediction_cache": {
                "entries": prediction_cache_count,
                "status": "active"
            },
            "redis": {
                "status": redis_status,
                "memory_usage": redis_memory
            },
            "optimizations": [
                "2-hour TTL for OpenAPI specs",
                "10-minute TTL for predictions",
                "In-memory fallback caching",
                "Automatic cache cleanup"
            ]
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches (for testing/debugging)"""
    try:
        # Clear in-memory caches
        cache_manager.in_memory_cache.clear()
        if hasattr(cache_manager, 'prediction_cache'):
            cache_manager.prediction_cache.clear()
        
        # Clear Redis cache
        if cache_manager.redis_client:
            await cache_manager.redis_client.flushdb()
        
        # Clear component caches
        if hasattr(candidate_generator, 'request_cache'):
            candidate_generator.request_cache.clear()
        
        if hasattr(prediction_ranker, 'feature_cache'):
            prediction_ranker.feature_cache.clear()
        
        if hasattr(prediction_ranker, 'heuristic_cache'):
            prediction_ranker.heuristic_cache.clear()
        
        logger.info("All caches cleared")
        
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "cleared": [
                "OpenAPI spec cache",
                "Prediction result cache", 
                "In-memory cache",
                "Redis cache",
                "Feature cache",
                "Heuristic cache"
            ]
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=False  # Disable access logs for better performance
    )