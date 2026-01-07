"""
Emotional Micro-Pattern Features
Extracts emotion intensity, volatility, and decay patterns.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

# Try to import sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


@dataclass
class EmotionVector:
    """Emotion intensity vector for a single text"""
    sadness: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    joy: float = 0.0
    hopelessness: float = 0.0
    guilt: float = 0.0
    anxiety: float = 0.0


class EmotionalFeatureExtractor:
    """
    Extracts emotional micro-patterns from text.
    
    Features:
    - Emotion intensity scores
    - Sentiment polarity and subjectivity
    - Hopelessness markers
    - Positive emotion decay tracking
    """
    
    # Emotion lexicons (simplified)
    SADNESS_WORDS = {
        'sad', 'depressed', 'unhappy', 'miserable', 'hopeless', 'empty',
        'lonely', 'hurt', 'crying', 'tears', 'grief', 'sorrow', 'gloomy',
        'heartbroken', 'devastated', 'melancholy', 'dejected', 'down'
    }
    
    FEAR_WORDS = {
        'afraid', 'scared', 'terrified', 'anxious', 'worried', 'panic',
        'nervous', 'frightened', 'dread', 'horror', 'phobia', 'terror',
        'alarmed', 'uneasy', 'apprehensive', 'paranoid'
    }
    
    ANGER_WORDS = {
        'angry', 'furious', 'mad', 'rage', 'hate', 'frustrated', 'annoyed',
        'irritated', 'resentful', 'bitter', 'hostile', 'outraged', 'livid'
    }
    
    JOY_WORDS = {
        'happy', 'joy', 'excited', 'glad', 'cheerful', 'delighted', 'pleased',
        'wonderful', 'amazing', 'great', 'love', 'blessed', 'grateful',
        'thankful', 'hopeful', 'optimistic', 'elated'
    }
    
    HOPELESSNESS_WORDS = {
        'hopeless', 'pointless', 'worthless', 'useless', 'meaningless',
        'nothing matters', 'give up', 'no point', 'why bother', 'end it',
        'no future', 'no hope', 'trapped', 'burden', 'no way out'
    }
    
    GUILT_WORDS = {
        'guilty', 'shame', 'ashamed', 'fault', 'blame', 'sorry', 'regret',
        'apologize', 'wrong', 'mistake', 'failed', 'failure', 'disappoint'
    }
    
    ANXIETY_WORDS = {
        'anxious', 'worried', 'stress', 'stressed', 'overwhelmed', 'panic',
        'nervous', 'tense', 'restless', 'uneasy', 'on edge', 'racing thoughts'
    }
    
    def __init__(self):
        self.vader = None
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract all emotional features from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of feature names to values
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        features = {}
        
        # Lexicon-based emotion scores
        features.update(self._lexicon_emotions(words))
        
        # VADER sentiment (if available)
        features.update(self._vader_sentiment(text))
        
        # Hopelessness markers
        features.update(self._hopelessness_features(text_lower))
        
        return features
    
    def _lexicon_emotions(self, words: List[str]) -> Dict[str, float]:
        """Calculate emotion intensities from lexicons"""
        total = len(words) if words else 1
        
        sadness = sum(1 for w in words if w in self.SADNESS_WORDS)
        fear = sum(1 for w in words if w in self.FEAR_WORDS)
        anger = sum(1 for w in words if w in self.ANGER_WORDS)
        joy = sum(1 for w in words if w in self.JOY_WORDS)
        hopelessness = sum(1 for w in words if w in self.HOPELESSNESS_WORDS)
        guilt = sum(1 for w in words if w in self.GUILT_WORDS)
        anxiety = sum(1 for w in words if w in self.ANXIETY_WORDS)
        
        # Normalize by text length
        return {
            'emotion_sadness': sadness / total,
            'emotion_fear': fear / total,
            'emotion_anger': anger / total,
            'emotion_joy': joy / total,
            'emotion_hopelessness': hopelessness / total,
            'emotion_guilt': guilt / total,
            'emotion_anxiety': anxiety / total,
            'negative_emotion_sum': (sadness + fear + anger + hopelessness + guilt + anxiety) / total,
            'emotion_valence': (joy - sadness - hopelessness) / total
        }
    
    def _vader_sentiment(self, text: str) -> Dict[str, float]:
        """Extract VADER sentiment scores"""
        if self.vader is None:
            return {
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 0.0,
                'vader_compound': 0.0
            }
            
        scores = self.vader.polarity_scores(text)
        return {
            'vader_positive': scores['pos'],
            'vader_negative': scores['neg'],
            'vader_neutral': scores['neu'],
            'vader_compound': scores['compound']
        }
    
    def _hopelessness_features(self, text: str) -> Dict[str, float]:
        """Detect hopelessness markers in text"""
        # Check for multi-word phrases
        hopelessness_phrases = [
            'no point', 'give up', 'what\'s the point', 'why bother',
            'nothing matters', 'end it all', 'no hope', 'no future',
            'no way out', 'can\'t go on', 'want to die', 'better off dead'
        ]
        
        phrase_count = sum(1 for phrase in hopelessness_phrases if phrase in text)
        
        # Future orientation (lack of future words = risk)
        future_words = {'tomorrow', 'future', 'will', 'going to', 'plan', 'hope', 'want to'}
        future_count = sum(1 for w in text.split() if w in future_words)
        
        return {
            'hopelessness_phrases': phrase_count,
            'future_orientation': future_count / (len(text.split()) + 1),
            'has_crisis_language': 1.0 if phrase_count > 0 else 0.0
        }
    
    def calculate_volatility(
        self, 
        emotion_sequence: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate emotional volatility across a sequence of time windows.
        
        Args:
            emotion_sequence: List of emotion feature dicts over time
            
        Returns:
            Volatility metrics
        """
        if len(emotion_sequence) < 2:
            return {
                'sadness_volatility': 0.0,
                'joy_volatility': 0.0,
                'overall_volatility': 0.0,
                'positive_emotion_decay': 0.0
            }
            
        # Extract time series for key emotions
        sadness = [e.get('emotion_sadness', 0) for e in emotion_sequence]
        joy = [e.get('emotion_joy', 0) for e in emotion_sequence]
        compound = [e.get('vader_compound', 0) for e in emotion_sequence]
        
        # Calculate volatility (std of differences)
        sadness_vol = np.std(np.diff(sadness)) if len(sadness) > 1 else 0
        joy_vol = np.std(np.diff(joy)) if len(joy) > 1 else 0
        compound_vol = np.std(np.diff(compound)) if len(compound) > 1 else 0
        
        # Positive emotion decay (trend)
        if len(joy) >= 2:
            joy_trend = np.polyfit(range(len(joy)), joy, 1)[0]
        else:
            joy_trend = 0
            
        return {
            'sadness_volatility': float(sadness_vol),
            'joy_volatility': float(joy_vol),
            'overall_volatility': float(compound_vol),
            'positive_emotion_decay': float(-joy_trend)  # Negative trend = decay
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return [
            'emotion_sadness', 'emotion_fear', 'emotion_anger', 'emotion_joy',
            'emotion_hopelessness', 'emotion_guilt', 'emotion_anxiety',
            'negative_emotion_sum', 'emotion_valence',
            'vader_positive', 'vader_negative', 'vader_neutral', 'vader_compound',
            'hopelessness_phrases', 'future_orientation', 'has_crisis_language'
        ]
