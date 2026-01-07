"""
FastAPI REST API for MP-RDS
Provides endpoints for risk assessment and monitoring.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np

# Import MP-RDS modules
from ..data import TimelineProcessor
from ..features import FeatureExtractor
from ..models import DriftTransformer, RDSEngine, RiskLevel
from ..explainability import SHAPExplainer


# Pydantic models for API
class PostInput(BaseModel):
    """Single post input"""
    timestamp: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class TimelineInput(BaseModel):
    """Timeline analysis request"""
    user_id: str
    posts: List[PostInput]


class RDSResponse(BaseModel):
    """RDS prediction response"""
    user_id: str
    rds_score: float
    risk_level: str
    direction: str
    velocity: float
    confidence: float
    contributing_factors: List[str]
    alert: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_loaded: bool


class TrajectoryResponse(BaseModel):
    """Trajectory analysis response"""
    user_id: str
    current_rds: float
    trend: str
    velocity: float
    acceleration: float
    is_accelerating: bool
    days_to_threshold: Optional[int]
    volatility: float


# Initialize FastAPI app
app = FastAPI(
    title="MP-RDS API",
    description="Micro-Pattern Risk Drift Scoring Model API for Early Mental Health Detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
timeline_processor = TimelineProcessor()
feature_extractor = FeatureExtractor()
rds_engine = RDSEngine()
explainer = SHAPExplainer()

# Model placeholder (load actual model in production)
model = None

# In-memory user history (use database in production)
user_histories: Dict[str, List[float]] = {}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model is not None
    )


@app.post("/analyze", response_model=RDSResponse)
async def analyze_timeline(request: TimelineInput):
    """
    Analyze a user's post timeline and compute RDS.
    
    This is the main endpoint for risk assessment.
    """
    try:
        # Convert posts to internal format
        posts = [
            {
                'timestamp': p.timestamp,
                'text': p.text,
                'metadata': p.metadata
            }
            for p in request.posts
        ]
        
        if not posts:
            raise HTTPException(status_code=400, detail="No posts provided")
        
        # Process timeline
        windows = timeline_processor.process_timeline(posts)
        
        if not windows:
            raise HTTPException(status_code=400, detail="Could not create time windows")
        
        # Extract features for each window
        feature_sequence = []
        for window in windows:
            if window.posts:
                combined_text = " ".join([p.text for p in window.posts])
                timestamps = [p.timestamp for p in window.posts]
                
                features = feature_extractor.extract_all(
                    combined_text,
                    timestamps,
                    window.start_time,
                    window.end_time
                )
                feature_sequence.append(features)
        
        if not feature_sequence:
            raise HTTPException(status_code=400, detail="No features extracted")
        
        # Compute RDS (simplified without actual model for demo)
        # In production, this would use the trained transformer model
        rds_score = _compute_heuristic_rds(feature_sequence)
        
        # Create model output dict
        model_output = {
            'rds': rds_score,
            'direction': _compute_direction(feature_sequence),
            'velocity': _compute_velocity(feature_sequence),
            'stability': 0.7
        }
        
        # Get RDS result with interpretation
        rds_result = rds_engine.compute_rds(model_output)
        
        # Update user history
        if request.user_id not in user_histories:
            user_histories[request.user_id] = []
        user_histories[request.user_id].append(rds_score)
        
        # Analyze trajectory
        trajectory = rds_engine.analyze_trajectory(user_histories[request.user_id])
        
        # Generate alert if needed
        alert = rds_engine.generate_alert(rds_result, trajectory)
        
        # Generate explanation
        explanations = explainer.explain_prediction(
            feature_sequence[-1],
            list(feature_sequence[-1].keys()),
            rds_score
        )
        explanation_text = explainer.generate_text_explanation(explanations, rds_score)
        
        return RDSResponse(
            user_id=request.user_id,
            rds_score=rds_result.score,
            risk_level=rds_result.level.value,
            direction=rds_result.direction,
            velocity=rds_result.velocity,
            confidence=rds_result.confidence,
            contributing_factors=rds_result.contributing_factors,
            alert=alert,
            explanation=explanation_text,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trajectory/{user_id}", response_model=TrajectoryResponse)
async def get_trajectory(user_id: str):
    """Get trajectory analysis for a user"""
    if user_id not in user_histories or not user_histories[user_id]:
        raise HTTPException(status_code=404, detail="User not found or no history")
    
    trajectory = rds_engine.analyze_trajectory(user_histories[user_id])
    
    return TrajectoryResponse(
        user_id=user_id,
        current_rds=trajectory['current_rds'],
        trend=trajectory['trend'],
        velocity=trajectory['velocity'],
        acceleration=trajectory['acceleration'],
        is_accelerating=trajectory['is_accelerating'],
        days_to_threshold=trajectory['days_to_threshold'],
        volatility=trajectory['volatility']
    )


@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Get RDS history for a user"""
    if user_id not in user_histories:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": user_id,
        "rds_history": user_histories[user_id],
        "count": len(user_histories[user_id])
    }


@app.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear history for a user (for testing)"""
    if user_id in user_histories:
        del user_histories[user_id]
    return {"status": "cleared"}


# Helper functions for heuristic computation (demo mode)
def _compute_heuristic_rds(feature_sequence: List[Dict[str, float]]) -> float:
    """Compute RDS without model (for demo)"""
    if not feature_sequence:
        return 0.0
    
    # Use latest features
    latest = feature_sequence[-1]
    
    # Risk factors with weights
    risk_score = 0.0
    risk_score += latest.get('emotion_sadness', 0) * 0.15
    risk_score += latest.get('emotion_hopelessness', 0) * 0.25
    risk_score += latest.get('emotion_anxiety', 0) * 0.10
    risk_score += latest.get('self_focus_score', 0) * 0.10
    risk_score += latest.get('absolutist_ratio', 0) * 0.10
    risk_score += latest.get('night_posting_ratio', 0) * 0.10
    risk_score += latest.get('negative_emotion_sum', 0) * 0.10
    
    # Protective factors
    risk_score -= latest.get('emotion_joy', 0) * 0.10
    risk_score -= latest.get('future_orientation', 0) * 0.05
    
    # Clamp to 0-1
    return max(0.0, min(1.0, risk_score))


def _compute_direction(feature_sequence: List[Dict[str, float]]) -> float:
    """Compute trend direction"""
    if len(feature_sequence) < 2:
        return 0.0
    
    neg_emotions = [f.get('negative_emotion_sum', 0) for f in feature_sequence]
    if len(neg_emotions) >= 2:
        return float(np.polyfit(range(len(neg_emotions)), neg_emotions, 1)[0])
    return 0.0


def _compute_velocity(feature_sequence: List[Dict[str, float]]) -> float:
    """Compute rate of change"""
    if len(feature_sequence) < 2:
        return 0.0
    
    neg_emotions = [f.get('negative_emotion_sum', 0) for f in feature_sequence]
    velocity = np.diff(neg_emotions)
    return float(np.mean(np.abs(velocity)))


# Run with: uvicorn mp_rds.api.server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
