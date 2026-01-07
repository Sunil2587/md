"""
SHAP-based Explainability for MP-RDS
Provides feature attribution and interpretation for clinical use.
"""
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class Explanation:
    """Feature explanation result"""
    feature_name: str
    importance: float
    direction: str  # "increases_risk" or "decreases_risk"
    value: float
    baseline: float


class SHAPExplainer:
    """
    SHAP-based explainability for MP-RDS predictions.
    
    Provides:
    - Feature attribution using SHAP
    - Natural language explanations
    - Temporal importance analysis
    """
    
    # Feature descriptions for natural language explanations
    FEATURE_DESCRIPTIONS = {
        # Linguistic
        'ttr': 'vocabulary richness',
        'i_ratio': 'self-focused language (I, me, my)',
        'self_focus_score': 'self-focus vs social focus',
        'absolutist_ratio': 'absolutist thinking patterns',
        'repetition_ratio': 'repetitive language',
        
        # Emotional
        'emotion_sadness': 'sadness expressions',
        'emotion_hopelessness': 'hopelessness indicators',
        'emotion_anxiety': 'anxiety expressions',
        'emotion_joy': 'positive emotions',
        'vader_compound': 'overall sentiment',
        'negative_emotion_sum': 'total negative emotions',
        
        # Behavioral
        'night_posting_ratio': 'late-night posting',
        'circadian_entropy': 'posting time variability',
        'frequency_cv': 'posting frequency irregularity',
        'burst_score': 'posting burst patterns'
    }
    
    def __init__(self, model=None, background_data=None):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (optional, can be set later)
            background_data: Background dataset for SHAP
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
        if SHAP_AVAILABLE and model is not None:
            self._init_explainer()
    
    def _init_explainer(self):
        """Initialize SHAP explainer"""
        if not SHAP_AVAILABLE:
            return
            
        if self.background_data is not None:
            self.explainer = shap.DeepExplainer(
                self.model, 
                self.background_data
            )
    
    def explain_prediction(
        self,
        features: Dict[str, float],
        feature_names: List[str],
        prediction: float
    ) -> List[Explanation]:
        """
        Generate explanations for a prediction.
        
        Args:
            features: Feature dictionary
            feature_names: List of feature names in order
            prediction: Model prediction (RDS)
            
        Returns:
            List of Explanation objects sorted by importance
        """
        if SHAP_AVAILABLE and self.explainer is not None:
            return self._shap_explain(features, feature_names)
        else:
            return self._heuristic_explain(features, prediction)
    
    def _shap_explain(
        self,
        features: Dict[str, float],
        feature_names: List[str]
    ) -> List[Explanation]:
        """Use SHAP for explanation"""
        import torch
        
        # Convert to tensor
        feature_values = [features.get(name, 0) for name in feature_names]
        x = torch.tensor([feature_values], dtype=torch.float32)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(x)
        
        # Create explanations
        explanations = []
        for i, name in enumerate(feature_names):
            importance = float(shap_values[0][i])
            value = features.get(name, 0)
            
            explanations.append(Explanation(
                feature_name=name,
                importance=abs(importance),
                direction="increases_risk" if importance > 0 else "decreases_risk",
                value=value,
                baseline=0.0
            ))
            
        # Sort by importance
        explanations.sort(key=lambda x: x.importance, reverse=True)
        return explanations
    
    def _heuristic_explain(
        self,
        features: Dict[str, float],
        prediction: float
    ) -> List[Explanation]:
        """Heuristic explanation when SHAP is not available"""
        # Risk-increasing features and their weights
        risk_features = {
            'emotion_sadness': 1.0,
            'emotion_hopelessness': 1.5,
            'emotion_anxiety': 0.8,
            'self_focus_score': 0.7,
            'absolutist_ratio': 0.9,
            'night_posting_ratio': 0.8,
            'negative_emotion_sum': 1.2,
            'burst_score': 0.6
        }
        
        # Risk-decreasing features
        protective_features = {
            'emotion_joy': -1.0,
            'ttr': -0.5,
            'future_orientation': -0.7
        }
        
        explanations = []
        
        # Calculate importance scores
        for name, value in features.items():
            if name in risk_features:
                weight = risk_features[name]
                importance = value * weight
                direction = "increases_risk"
            elif name in protective_features:
                weight = protective_features[name]
                importance = value * abs(weight)
                direction = "decreases_risk"
            else:
                importance = value * 0.3
                direction = "increases_risk" if value > 0.5 else "decreases_risk"
                
            explanations.append(Explanation(
                feature_name=name,
                importance=importance,
                direction=direction,
                value=value,
                baseline=0.0
            ))
            
        # Sort by importance
        explanations.sort(key=lambda x: x.importance, reverse=True)
        return explanations[:10]  # Top 10
    
    def generate_text_explanation(
        self,
        explanations: List[Explanation],
        rds_score: float,
        n_top: int = 5
    ) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            explanations: List of Explanation objects
            rds_score: The RDS score
            n_top: Number of top factors to include
            
        Returns:
            Natural language explanation
        """
        lines = []
        
        # Risk level summary
        if rds_score < 0.2:
            lines.append("**Risk Assessment: Low**")
        elif rds_score < 0.4:
            lines.append("**Risk Assessment: Low-Moderate**")
        elif rds_score < 0.6:
            lines.append("**Risk Assessment: Moderate**")
        elif rds_score < 0.8:
            lines.append("**Risk Assessment: Elevated**")
        else:
            lines.append("**Risk Assessment: High**")
            
        lines.append(f"\nRisk Drift Score: {rds_score:.2f}")
        lines.append("\n**Key Contributing Factors:**\n")
        
        # Top contributing factors
        for i, exp in enumerate(explanations[:n_top], 1):
            desc = self.FEATURE_DESCRIPTIONS.get(
                exp.feature_name, 
                exp.feature_name.replace('_', ' ')
            )
            
            if exp.direction == "increases_risk":
                lines.append(f"{i}. ↑ **{desc.title()}** ({exp.value:.2f})")
            else:
                lines.append(f"{i}. ↓ {desc.title()} ({exp.value:.2f})")
                
        return "\n".join(lines)
    
    def get_temporal_importance(
        self,
        feature_sequence: List[Dict[str, float]],
        attention_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze which time windows contributed most to the prediction.
        
        Args:
            feature_sequence: Features over time
            attention_weights: Optional attention weights from model
            
        Returns:
            Temporal importance analysis
        """
        n_windows = len(feature_sequence)
        
        if attention_weights is not None:
            # Use attention weights
            window_importance = attention_weights.mean(axis=-1)
        else:
            # Heuristic: recent windows more important
            window_importance = np.exp(np.linspace(-2, 0, n_windows))
            window_importance /= window_importance.sum()
            
        # Find peak importance window
        peak_idx = int(np.argmax(window_importance))
        
        # Analyze trend
        negative_trend = []
        for i, features in enumerate(feature_sequence):
            neg_score = features.get('negative_emotion_sum', 0)
            negative_trend.append(neg_score)
            
        return {
            'window_importance': window_importance.tolist(),
            'peak_window_index': peak_idx,
            'recent_weight': float(window_importance[-1]),
            'negative_emotion_trend': negative_trend
        }


# Export
__all__ = ['SHAPExplainer', 'Explanation', 'SHAP_AVAILABLE']
