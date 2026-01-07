# Features module
from .linguistic import LinguisticFeatureExtractor
from .emotional import EmotionalFeatureExtractor
from .behavioral import BehavioralFeatureExtractor


class FeatureExtractor:
    """
    Combined feature extractor for all micro-patterns.
    """
    
    def __init__(self):
        self.linguistic = LinguisticFeatureExtractor()
        self.emotional = EmotionalFeatureExtractor()
        self.behavioral = BehavioralFeatureExtractor()
        
    def extract_all(self, text: str, timestamps=None, window_start=None, window_end=None):
        """Extract all features from text and timestamps"""
        features = {}
        
        # Linguistic features
        features.update(self.linguistic.extract_features(text))
        
        # Emotional features
        features.update(self.emotional.extract_features(text))
        
        # Behavioral features (if timestamps provided)
        if timestamps and window_start and window_end:
            features.update(
                self.behavioral.extract_features(timestamps, window_start, window_end)
            )
            
        return features
    
    def get_all_feature_names(self):
        """Get all feature names"""
        names = []
        names.extend(self.linguistic.get_feature_names())
        names.extend(self.emotional.get_feature_names())
        names.extend(self.behavioral.get_feature_names())
        return names
