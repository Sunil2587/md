"""
MP-RDS Configuration Settings
"""
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Temporal Transformer configuration"""
    d_features: int = 20          # Number of input features
    d_model: int = 128            # Model dimension
    n_heads: int = 4              # Attention heads
    n_layers: int = 3             # Transformer layers
    dropout: float = 0.1
    max_sequence_length: int = 52  # Max weeks to analyze


@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    # Linguistic features
    min_words_for_ttr: int = 10
    
    # Emotional features
    emotion_categories: tuple = (
        "sadness", "fear", "guilt", "hopelessness", 
        "anger", "joy", "surprise"
    )
    
    # Behavioral features
    night_hours: tuple = (1, 5)   # 1 AM to 5 AM
    silence_threshold_days: int = 14
    burst_multiplier: float = 3.0


@dataclass
class RDSConfig:
    """Risk Drift Score configuration"""
    # Score thresholds
    stable_max: float = 0.2
    low_concern_max: float = 0.4
    moderate_risk_max: float = 0.6
    elevated_risk_max: float = 0.8
    
    # Acceleration detection
    acceleration_threshold: float = 0.1
    velocity_window: int = 3


@dataclass
class TimelineConfig:
    """Timeline processing configuration"""
    window_type: str = "weekly"   # daily, weekly, sliding
    sliding_window_days: int = 7
    sliding_overlap_days: int = 3
    min_posts_per_window: int = 1
    interpolation_method: str = "linear"
    silence_threshold_days: int = 14
    burst_multiplier: float = 3.0



# Global configuration instances
MODEL_CONFIG = ModelConfig()
FEATURE_CONFIG = FeatureConfig()
RDS_CONFIG = RDSConfig()
TIMELINE_CONFIG = TimelineConfig()
