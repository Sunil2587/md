"""
Timeline Processor - Data ingestion and preprocessing for MP-RDS
Handles timeline segmentation, missing data, and irregular activity patterns.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..config import TIMELINE_CONFIG


@dataclass
class Post:
    """Single user post"""
    timestamp: datetime
    text: str
    metadata: Optional[Dict] = None


@dataclass
class TimeWindow:
    """Aggregated time window"""
    start_time: datetime
    end_time: datetime
    posts: List[Post]
    silence_flag: bool = False
    burst_flag: bool = False


class TimelineProcessor:
    """
    Processes user post timelines into structured windows for analysis.
    Handles irregular posting patterns, silence periods, and bursts.
    """
    
    def __init__(self, config=TIMELINE_CONFIG):
        self.config = config
        
    def process_timeline(self, posts: List[Dict]) -> List[TimeWindow]:
        """
        Main entry point: Convert raw posts to time windows.
        
        Args:
            posts: List of post dictionaries with 'timestamp' and 'text'
            
        Returns:
            List of TimeWindow objects
        """
        # Parse posts
        parsed_posts = self._parse_posts(posts)
        if not parsed_posts:
            return []
            
        # Sort by timestamp
        parsed_posts.sort(key=lambda x: x.timestamp)
        
        # Create windows based on configuration
        if self.config.window_type == "daily":
            windows = self._create_daily_windows(parsed_posts)
        elif self.config.window_type == "weekly":
            windows = self._create_weekly_windows(parsed_posts)
        else:  # sliding
            windows = self._create_sliding_windows(parsed_posts)
            
        # Detect anomalies
        windows = self._detect_silence_periods(windows)
        windows = self._detect_burst_patterns(windows)
        
        return windows
    
    def _parse_posts(self, posts: List[Dict]) -> List[Post]:
        """Parse raw post dictionaries into Post objects"""
        parsed = []
        for p in posts:
            try:
                ts = p.get('timestamp')
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts)
                    
                parsed.append(Post(
                    timestamp=ts,
                    text=p.get('text', ''),
                    metadata=p.get('metadata')
                ))
            except Exception:
                continue
        return parsed
    
    def _create_daily_windows(self, posts: List[Post]) -> List[TimeWindow]:
        """Create daily time windows"""
        if not posts:
            return []
            
        windows = []
        grouped = defaultdict(list)
        
        for post in posts:
            day_key = post.timestamp.date()
            grouped[day_key].append(post)
            
        # Create windows for each day in range
        start_date = min(grouped.keys())
        end_date = max(grouped.keys())
        current = start_date
        
        while current <= end_date:
            window_posts = grouped.get(current, [])
            windows.append(TimeWindow(
                start_time=datetime.combine(current, datetime.min.time()),
                end_time=datetime.combine(current, datetime.max.time()),
                posts=window_posts
            ))
            current += timedelta(days=1)
            
        return windows
    
    def _create_weekly_windows(self, posts: List[Post]) -> List[TimeWindow]:
        """Create weekly time windows"""
        if not posts:
            return []
            
        windows = []
        start_date = posts[0].timestamp.date()
        end_date = posts[-1].timestamp.date()
        
        # Align to week start (Monday)
        start_date = start_date - timedelta(days=start_date.weekday())
        
        current = start_date
        while current <= end_date:
            week_end = current + timedelta(days=6)
            window_posts = [
                p for p in posts 
                if current <= p.timestamp.date() <= week_end
            ]
            windows.append(TimeWindow(
                start_time=datetime.combine(current, datetime.min.time()),
                end_time=datetime.combine(week_end, datetime.max.time()),
                posts=window_posts
            ))
            current += timedelta(days=7)
            
        return windows
    
    def _create_sliding_windows(self, posts: List[Post]) -> List[TimeWindow]:
        """Create sliding time windows with overlap"""
        if not posts:
            return []
            
        windows = []
        start_date = posts[0].timestamp.date()
        end_date = posts[-1].timestamp.date()
        
        window_size = self.config.sliding_window_days
        step = window_size - self.config.sliding_overlap_days
        
        current = start_date
        while current <= end_date:
            window_end = current + timedelta(days=window_size - 1)
            window_posts = [
                p for p in posts 
                if current <= p.timestamp.date() <= window_end
            ]
            windows.append(TimeWindow(
                start_time=datetime.combine(current, datetime.min.time()),
                end_time=datetime.combine(window_end, datetime.max.time()),
                posts=window_posts
            ))
            current += timedelta(days=step)
            
        return windows
    
    def _detect_silence_periods(self, windows: List[TimeWindow]) -> List[TimeWindow]:
        """Flag windows that follow extended silence periods"""
        threshold = self.config.silence_threshold_days
        
        for i, window in enumerate(windows):
            if i == 0:
                continue
                
            # Find last window with posts
            days_since_post = 0
            for j in range(i - 1, -1, -1):
                if windows[j].posts:
                    break
                days_since_post += (
                    windows[j].end_time - windows[j].start_time
                ).days + 1
                
            if days_since_post >= threshold:
                window.silence_flag = True
                
        return windows
    
    def _detect_burst_patterns(self, windows: List[TimeWindow]) -> List[TimeWindow]:
        """Flag windows with abnormal posting bursts"""
        post_counts = [len(w.posts) for w in windows if w.posts]
        if not post_counts:
            return windows
            
        mean_posts = np.mean(post_counts)
        threshold = mean_posts * self.config.burst_multiplier
        
        for window in windows:
            if len(window.posts) > threshold:
                window.burst_flag = True
                
        return windows
    
    def interpolate_features(
        self, 
        feature_sequence: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate missing features in a sequence.
        
        Args:
            feature_sequence: Shape (time_steps, features)
            mask: Boolean mask where True = valid data
            
        Returns:
            Interpolated feature sequence
        """
        result = feature_sequence.copy()
        
        for feat_idx in range(feature_sequence.shape[1]):
            feat_values = feature_sequence[:, feat_idx]
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) < 2:
                continue
                
            # Linear interpolation
            interp_values = np.interp(
                np.arange(len(feat_values)),
                valid_indices,
                feat_values[valid_indices]
            )
            result[:, feat_idx] = interp_values
            
        return result
    
    def get_timeline_stats(self, windows: List[TimeWindow]) -> Dict:
        """Get summary statistics for a timeline"""
        total_posts = sum(len(w.posts) for w in windows)
        active_windows = sum(1 for w in windows if w.posts)
        silence_windows = sum(1 for w in windows if w.silence_flag)
        burst_windows = sum(1 for w in windows if w.burst_flag)
        
        post_counts = [len(w.posts) for w in windows]
        
        return {
            "total_windows": len(windows),
            "total_posts": total_posts,
            "active_windows": active_windows,
            "silence_windows": silence_windows,
            "burst_windows": burst_windows,
            "mean_posts_per_window": np.mean(post_counts) if post_counts else 0,
            "std_posts_per_window": np.std(post_counts) if post_counts else 0,
        }
