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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_api_call(request: PredictionRequest):
    """
    Predict the next API call a user is likely to make
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing prediction request for user {request.user_id}")
        
        # Validate input
        if len(request.events) == 0:
            raise HTTPException(status_code=400, detail="At least one event is required")
        
        # Parse OpenAPI spec
        spec_data = await openapi_parser.parse_spec(request.spec_url)
        
        # Generate candidates using AI layer
        candidates = await candidate_generator.generate_candidates(
            events=request.events,
            prompt=request.prompt,
            spec_data=spec_data,
            k=request.k
        )
        
        # Rank candidates using ML layer
        ranked_predictions = await prediction_ranker.rank_candidates(
            candidates=candidates,
            events=request.events,
            prompt=request.prompt,
            user_id=request.user_id
        )
        
        # Return top k predictions
        top_predictions = ranked_predictions[:request.k]
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"Prediction completed in {processing_time}ms")
        
        return PredictionResponse(
            predictions=top_predictions,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
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