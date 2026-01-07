"""
Risk Drift Score (RDS) Engine
Computes and interprets risk scores from model outputs.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import RDS_CONFIG


class RiskLevel(Enum):
    """Risk level categories"""
    STABLE = "stable"
    LOW_CONCERN = "low_concern"
    MODERATE_RISK = "moderate_risk"
    ELEVATED_RISK = "elevated_risk"
    HIGH_RISK = "high_risk"


@dataclass
class RDSResult:
    """Risk Drift Score result with interpretation"""
    score: float
    level: RiskLevel
    direction: str  # "improving", "stable", "deteriorating"
    velocity: float  # Rate of change
    acceleration: str  # "accelerating", "stable", "decelerating"
    confidence: float
    contributing_factors: List[str]


class RDSEngine:
    """
    Risk Drift Score computation and interpretation engine.
    
    Provides:
    - Score computation from model outputs
    - Risk level classification
    - Trend analysis (direction, velocity, acceleration)
    - Interpretation for clinical use
    """
    
    def __init__(self, config=RDS_CONFIG):
        self.config = config
        
    def compute_rds(
        self,
        model_output: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> RDSResult:
        """
        Compute RDS from model output.
        
        Args:
            model_output: Dictionary with 'rds', 'direction', 'velocity', 'stability'
            feature_importance: Optional SHAP values for contributing factors
            
        Returns:
            RDSResult with full interpretation
        """
        score = model_output.get('rds', 0.0)
        direction_val = model_output.get('direction', 0.0)
        velocity_val = model_output.get('velocity', 0.0)
        stability_val = model_output.get('stability', 0.5)
        
        # Classify risk level
        level = self._classify_level(score)
        
        # Interpret direction
        direction = self._interpret_direction(direction_val)
        
        # Calculate confidence (based on stability)
        confidence = self._calculate_confidence(stability_val, score)
        
        # Get contributing factors
        factors = self._get_contributing_factors(feature_importance)
        
        return RDSResult(
            score=score,
            level=level,
            direction=direction,
            velocity=velocity_val,
            acceleration="stable",  # Will be updated by trajectory analysis
            confidence=confidence,
            contributing_factors=factors
        )
    
    def analyze_trajectory(
        self,
        rds_history: List[float],
        window_size: int = None
    ) -> Dict[str, any]:
        """
        Analyze RDS trajectory over time.
        
        Args:
            rds_history: List of historical RDS values
            window_size: Optional window for velocity calculation
            
        Returns:
            Trajectory analysis results
        """
        if len(rds_history) < 2:
            return {
                'current_rds': rds_history[-1] if rds_history else 0.0,
                'trend': 'insufficient_data',
                'velocity': 0.0,
                'acceleration': 0.0,
                'is_accelerating': False,
                'days_to_threshold': None
            }
            
        window = window_size or self.config.velocity_window
        
        # Recent window
        recent = rds_history[-window:] if len(rds_history) >= window else rds_history
        
        # Calculate velocity (first derivative)
        velocity = np.diff(recent)
        mean_velocity = np.mean(velocity)
        
        # Calculate acceleration (second derivative)
        if len(velocity) >= 2:
            acceleration = np.diff(velocity)
            mean_acceleration = np.mean(acceleration)
        else:
            mean_acceleration = 0.0
            
        # Detect if accelerating toward risk
        is_accelerating = (
            mean_velocity > 0 and 
            mean_acceleration > self.config.acceleration_threshold
        )
        
        # Estimate days to threshold (if deteriorating)
        current = rds_history[-1]
        if mean_velocity > 0:
            threshold = self.config.moderate_risk_max
            if current < threshold:
                days_to_threshold = (threshold - current) / (mean_velocity + 1e-6)
            else:
                days_to_threshold = 0
        else:
            days_to_threshold = None
            
        # Determine overall trend
        if mean_velocity > 0.02:
            trend = "deteriorating"
        elif mean_velocity < -0.02:
            trend = "improving"
        else:
            trend = "stable"
            
        return {
            'current_rds': current,
            'trend': trend,
            'velocity': float(mean_velocity),
            'acceleration': float(mean_acceleration),
            'is_accelerating': is_accelerating,
            'days_to_threshold': int(days_to_threshold) if days_to_threshold else None,
            'volatility': float(np.std(recent))
        }
    
    def generate_alert(
        self,
        rds_result: RDSResult,
        trajectory: Dict[str, any]
    ) -> Optional[Dict[str, any]]:
        """
        Generate alert based on RDS and trajectory.
        
        Returns:
            Alert dictionary or None if no alert needed
        """
        # High risk - immediate alert
        if rds_result.level == RiskLevel.HIGH_RISK:
            return {
                'severity': 'critical',
                'message': 'High risk detected - immediate review recommended',
                'rds': rds_result.score,
                'factors': rds_result.contributing_factors,
                'action': 'escalate_immediately'
            }
            
        # Elevated risk - priority alert
        if rds_result.level == RiskLevel.ELEVATED_RISK:
            return {
                'severity': 'high',
                'message': 'Elevated risk detected - priority review needed',
                'rds': rds_result.score,
                'factors': rds_result.contributing_factors,
                'action': 'review_within_24h'
            }
            
        # Accelerating toward risk - early warning
        if trajectory.get('is_accelerating') and rds_result.score > 0.3:
            days = trajectory.get('days_to_threshold')
            return {
                'severity': 'warning',
                'message': f'Risk trajectory accelerating - may reach threshold in ~{days} periods',
                'rds': rds_result.score,
                'velocity': trajectory['velocity'],
                'action': 'monitor_closely'
            }
            
        # Moderate risk - monitor
        if rds_result.level == RiskLevel.MODERATE_RISK:
            return {
                'severity': 'moderate',
                'message': 'Moderate risk level - continued monitoring advised',
                'rds': rds_result.score,
                'factors': rds_result.contributing_factors,
                'action': 'weekly_review'
            }
            
        return None
    
    def _classify_level(self, score: float) -> RiskLevel:
        """Classify score into risk level"""
        if score <= self.config.stable_max:
            return RiskLevel.STABLE
        elif score <= self.config.low_concern_max:
            return RiskLevel.LOW_CONCERN
        elif score <= self.config.moderate_risk_max:
            return RiskLevel.MODERATE_RISK
        elif score <= self.config.elevated_risk_max:
            return RiskLevel.ELEVATED_RISK
        else:
            return RiskLevel.HIGH_RISK
    
    def _interpret_direction(self, direction_val: float) -> str:
        """Interpret direction value"""
        if direction_val > 0.1:
            return "deteriorating"
        elif direction_val < -0.1:
            return "improving"
        else:
            return "stable"
    
    def _calculate_confidence(self, stability: float, score: float) -> float:
        """Calculate confidence in the score"""
        # Higher stability = higher confidence
        # Extreme scores (near 0 or 1) have higher confidence
        base_confidence = stability
        extremity_bonus = abs(score - 0.5) * 0.2
        return min(base_confidence + extremity_bonus, 1.0)
    
    def _get_contributing_factors(
        self,
        feature_importance: Optional[Dict[str, float]]
    ) -> List[str]:
        """Get top contributing factors from SHAP values"""
        if not feature_importance:
            return []
            
        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Return top 5 factor names
        return [name for name, _ in sorted_features[:5]]
    
    def get_risk_summary(self, level: RiskLevel) -> str:
        """Get human-readable risk summary"""
        summaries = {
            RiskLevel.STABLE: "No significant risk indicators detected. Continue routine monitoring.",
            RiskLevel.LOW_CONCERN: "Minor patterns of concern observed. Enhanced awareness recommended.",
            RiskLevel.MODERATE_RISK: "Moderate risk patterns present. Professional review advised.",
            RiskLevel.ELEVATED_RISK: "Elevated risk detected. Priority clinical attention needed.",
            RiskLevel.HIGH_RISK: "High risk indicators present. Immediate professional intervention recommended."
        }
        return summaries.get(level, "Unknown risk level")
