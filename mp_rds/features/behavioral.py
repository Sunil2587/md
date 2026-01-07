"""
Behavioral Micro-Pattern Features
Extracts posting time, frequency, and activity pattern features.
"""
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import Counter

from ..config import FEATURE_CONFIG


class BehavioralFeatureExtractor:
    """
    Extracts behavioral micro-patterns from posting activity.
    
    Features:
    - Circadian rhythm disruption
    - Posting frequency patterns
    - Silence and burst detection
    - Activity regularity
    """
    
    def __init__(self, config=FEATURE_CONFIG):
        self.config = config
        self.night_start, self.night_end = config.night_hours
        
    def extract_features(
        self, 
        timestamps: List[datetime],
        window_start: datetime,
        window_end: datetime
    ) -> Dict[str, float]:
        """
        Extract behavioral features from posting timestamps.
        
        Args:
            timestamps: List of post timestamps
            window_start: Start of analysis window
            window_end: End of analysis window
            
        Returns:
            Dictionary of feature names to values
        """
        if not timestamps:
            return self._empty_features()
            
        features = {}
        
        # Circadian features
        features.update(self._circadian_features(timestamps))
        
        # Frequency features
        features.update(self._frequency_features(timestamps, window_start, window_end))
        
        # Regularity features
        features.update(self._regularity_features(timestamps))
        
        return features
    
    def _circadian_features(self, timestamps: List[datetime]) -> Dict[str, float]:
        """
        Analyze circadian rhythm patterns.
        Night posting correlates with sleep disruption and depression.
        """
        hours = [ts.hour for ts in timestamps]
        
        # Night posting (1 AM - 5 AM by default)
        night_posts = sum(
            1 for h in hours 
            if self.night_start <= h <= self.night_end
        )
        night_ratio = night_posts / len(timestamps)
        
        # Hour distribution entropy
        hour_counts = Counter(hours)
        probs = np.array(list(hour_counts.values())) / len(timestamps)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(24)
        normalized_entropy = entropy / max_entropy
        
        # Peak hour
        peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 12
        
        # Day vs night ratio
        day_posts = sum(1 for h in hours if 6 <= h <= 22)
        day_night_ratio = day_posts / (night_posts + 1)
        
        return {
            'night_posting_ratio': night_ratio,
            'circadian_entropy': normalized_entropy,
            'peak_posting_hour': peak_hour / 24,  # Normalize to 0-1
            'day_night_ratio': min(day_night_ratio, 10) / 10,  # Cap at 10
            'late_night_count': night_posts
        }
    
    def _frequency_features(
        self,
        timestamps: List[datetime],
        window_start: datetime,
        window_end: datetime
    ) -> Dict[str, float]:
        """
        Analyze posting frequency patterns.
        """
        # Posts per day
        window_days = (window_end - window_start).days + 1
        posts_per_day = len(timestamps) / max(window_days, 1)
        
        # Daily post counts
        daily_counts = Counter()
        for ts in timestamps:
            daily_counts[ts.date()] += 1
            
        counts = list(daily_counts.values())
        
        if counts:
            mean_daily = np.mean(counts)
            std_daily = np.std(counts)
            max_daily = max(counts)
            min_daily = min(counts)
        else:
            mean_daily = std_daily = max_daily = min_daily = 0
            
        # Coefficient of variation
        cv = std_daily / mean_daily if mean_daily > 0 else 0
        
        return {
            'posts_per_day': posts_per_day,
            'daily_post_mean': mean_daily,
            'daily_post_std': std_daily,
            'daily_post_max': max_daily,
            'frequency_cv': min(cv, 3) / 3,  # Normalize, cap at 3
            'total_posts': len(timestamps)
        }
    
    def _regularity_features(self, timestamps: List[datetime]) -> Dict[str, float]:
        """
        Analyze posting regularity and intervals.
        Irregular patterns may indicate instability.
        """
        if len(timestamps) < 2:
            return {
                'interval_mean_hours': 0.0,
                'interval_std_hours': 0.0,
                'interval_irregularity': 0.0,
                'burst_score': 0.0
            }
            
        # Sort timestamps
        sorted_ts = sorted(timestamps)
        
        # Calculate inter-post intervals
        intervals = []
        for i in range(1, len(sorted_ts)):
            delta = (sorted_ts[i] - sorted_ts[i-1]).total_seconds() / 3600
            intervals.append(delta)
            
        intervals = np.array(intervals)
        
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Irregularity: coefficient of variation of intervals
        irregularity = std_interval / mean_interval if mean_interval > 0 else 0
        
        # Burst score: ratio of short intervals
        short_threshold = mean_interval / 4
        short_intervals = sum(1 for i in intervals if i < short_threshold)
        burst_score = short_intervals / len(intervals)
        
        return {
            'interval_mean_hours': min(mean_interval, 168) / 168,  # Cap at 1 week
            'interval_std_hours': min(std_interval, 168) / 168,
            'interval_irregularity': min(irregularity, 3) / 3,
            'burst_score': burst_score
        }
    
    def calculate_temporal_drift(
        self,
        behavioral_sequence: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate behavioral drift across time windows.
        
        Args:
            behavioral_sequence: List of behavioral features over time
            
        Returns:
            Drift metrics
        """
        if len(behavioral_sequence) < 2:
            return {
                'circadian_drift': 0.0,
                'frequency_drift': 0.0,
                'regularity_drift': 0.0
            }
            
        # Extract time series
        night_ratios = [b.get('night_posting_ratio', 0) for b in behavioral_sequence]
        frequencies = [b.get('posts_per_day', 0) for b in behavioral_sequence]
        irregularities = [b.get('interval_irregularity', 0) for b in behavioral_sequence]
        
        # Calculate trends (slope)
        def get_trend(values):
            if len(values) < 2:
                return 0
            return np.polyfit(range(len(values)), values, 1)[0]
            
        return {
            'circadian_drift': float(get_trend(night_ratios)),
            'frequency_drift': float(get_trend(frequencies)),
            'regularity_drift': float(get_trend(irregularities))
        }
    
    def detect_anomalies(
        self,
        current_features: Dict[str, float],
        historical_features: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Detect anomalies in current behavior vs historical baseline.
        """
        if len(historical_features) < 3:
            return {
                'circadian_anomaly': 0.0,
                'frequency_anomaly': 0.0,
                'is_anomalous': 0.0
            }
            
        # Calculate historical baselines
        hist_night = [h.get('night_posting_ratio', 0) for h in historical_features]
        hist_freq = [h.get('posts_per_day', 0) for h in historical_features]
        
        night_mean, night_std = np.mean(hist_night), np.std(hist_night)
        freq_mean, freq_std = np.mean(hist_freq), np.std(hist_freq)
        
        # Z-scores
        curr_night = current_features.get('night_posting_ratio', 0)
        curr_freq = current_features.get('posts_per_day', 0)
        
        night_z = abs(curr_night - night_mean) / (night_std + 1e-6)
        freq_z = abs(curr_freq - freq_mean) / (freq_std + 1e-6)
        
        is_anomalous = 1.0 if night_z > 2 or freq_z > 2 else 0.0
        
        return {
            'circadian_anomaly': min(night_z, 5) / 5,
            'frequency_anomaly': min(freq_z, 5) / 5,
            'is_anomalous': is_anomalous
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict"""
        return {
            'night_posting_ratio': 0.0,
            'circadian_entropy': 0.0,
            'peak_posting_hour': 0.5,
            'day_night_ratio': 1.0,
            'late_night_count': 0.0,
            'posts_per_day': 0.0,
            'daily_post_mean': 0.0,
            'daily_post_std': 0.0,
            'daily_post_max': 0.0,
            'frequency_cv': 0.0,
            'total_posts': 0.0,
            'interval_mean_hours': 0.0,
            'interval_std_hours': 0.0,
            'interval_irregularity': 0.0,
            'burst_score': 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return list(self._empty_features().keys())
