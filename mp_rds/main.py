"""
MP-RDS: Micro-Pattern Risk Drift Scoring Model
Main entry point and orchestration.
"""
import argparse
import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from .config import MODEL_CONFIG, FEATURE_CONFIG, RDS_CONFIG
from .data import TimelineProcessor, TimeWindow
from .features import FeatureExtractor
from .models import DriftTransformer, RDSEngine, create_model
from .explainability import SHAPExplainer


class MPRDS:
    """
    Main MP-RDS system class.
    Orchestrates the full pipeline from data to risk score.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize MP-RDS system.
        
        Args:
            model_path: Optional path to trained model weights
        """
        self.timeline_processor = TimelineProcessor()
        self.feature_extractor = FeatureExtractor()
        self.model = create_model()
        self.rds_engine = RDSEngine()
        self.explainer = SHAPExplainer()
        
        # Load model weights if provided
        if model_path:
            self._load_model(model_path)
            
        # Feature names for consistent ordering
        self.feature_names = self.feature_extractor.get_all_feature_names()
        
    def _load_model(self, path: str):
        """Load model weights from file"""
        import torch
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        
    def analyze(
        self,
        posts: List[Dict],
        user_id: str = "anonymous"
    ) -> Dict:
        """
        Analyze a user's post timeline.
        
        Args:
            posts: List of post dictionaries with 'timestamp' and 'text'
            user_id: User identifier
            
        Returns:
            Analysis results including RDS and interpretation
        """
        # Process timeline into windows
        windows = self.timeline_processor.process_timeline(posts)
        
        if not windows:
            return {"error": "Could not process timeline", "user_id": user_id}
            
        # Extract features for each window
        feature_sequence = []
        for window in windows:
            if window.posts:
                combined_text = " ".join([p.text for p in window.posts])
                timestamps = [p.timestamp for p in window.posts]
                
                features = self.feature_extractor.extract_all(
                    combined_text,
                    timestamps,
                    window.start_time,
                    window.end_time
                )
                feature_sequence.append(features)
                
        if not feature_sequence:
            return {"error": "No features extracted", "user_id": user_id}
            
        # Convert to numpy array
        feature_matrix = self._features_to_matrix(feature_sequence)
        
        # Run model (or heuristic if model not loaded)
        model_output = self._run_inference(feature_matrix)
        
        # Compute RDS with interpretation
        rds_result = self.rds_engine.compute_rds(
            model_output,
            feature_importance=self._get_feature_importance(feature_sequence[-1])
        )
        
        # Generate explanation
        explanations = self.explainer.explain_prediction(
            feature_sequence[-1],
            self.feature_names,
            rds_result.score
        )
        
        explanation_text = self.explainer.generate_text_explanation(
            explanations, 
            rds_result.score
        )
        
        return {
            "user_id": user_id,
            "rds_score": rds_result.score,
            "risk_level": rds_result.level.value,
            "direction": rds_result.direction,
            "velocity": rds_result.velocity,
            "confidence": rds_result.confidence,
            "contributing_factors": rds_result.contributing_factors,
            "explanation": explanation_text,
            "timeline_stats": self.timeline_processor.get_timeline_stats(windows),
            "n_windows": len(windows),
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    def _features_to_matrix(self, feature_sequence: List[Dict]) -> np.ndarray:
        """Convert feature sequence to numpy matrix"""
        matrix = []
        for features in feature_sequence:
            row = [features.get(name, 0) for name in self.feature_names]
            matrix.append(row)
        return np.array(matrix)
    
    def _run_inference(self, feature_matrix: np.ndarray) -> Dict:
        """Run model inference or heuristic"""
        import torch
        
        # Convert to tensor
        x = torch.tensor(feature_matrix, dtype=torch.float32).unsqueeze(0)
        
        # Pad if needed
        if x.shape[1] < MODEL_CONFIG.max_sequence_length:
            padding = torch.zeros(
                1, 
                MODEL_CONFIG.max_sequence_length - x.shape[1], 
                x.shape[2]
            )
            x = torch.cat([x, padding], dim=1)
            
        # Truncate if needed
        x = x[:, :MODEL_CONFIG.max_sequence_length, :]
        
        try:
            with torch.no_grad():
                output = self.model(x)
                
            return {
                'rds': float(output['rds'][0]),
                'direction': float(output['direction'][0]),
                'velocity': float(output['velocity'][0]),
                'stability': float(output['stability'][0])
            }
        except Exception:
            # Fallback to heuristic
            return self._heuristic_inference(feature_matrix)
            
    def _heuristic_inference(self, feature_matrix: np.ndarray) -> Dict:
        """Heuristic RDS computation without trained model"""
        # Use latest features
        latest = feature_matrix[-1] if len(feature_matrix) > 0 else np.zeros(len(self.feature_names))
        
        # Simple weighted sum
        feature_dict = dict(zip(self.feature_names, latest))
        
        risk_score = 0.0
        risk_score += feature_dict.get('emotion_sadness', 0) * 0.15
        risk_score += feature_dict.get('emotion_hopelessness', 0) * 0.25
        risk_score += feature_dict.get('emotion_anxiety', 0) * 0.10
        risk_score += feature_dict.get('self_focus_score', 0) * 0.10
        risk_score += feature_dict.get('absolutist_ratio', 0) * 0.10
        risk_score += feature_dict.get('night_posting_ratio', 0) * 0.10
        risk_score += feature_dict.get('negative_emotion_sum', 0) * 0.10
        risk_score -= feature_dict.get('emotion_joy', 0) * 0.10
        
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Compute direction from trend
        if len(feature_matrix) >= 2:
            neg_emotions = [f[self.feature_names.index('negative_emotion_sum')] 
                           for f in feature_matrix if 'negative_emotion_sum' in self.feature_names]
            direction = np.polyfit(range(len(neg_emotions)), neg_emotions, 1)[0] if neg_emotions else 0
        else:
            direction = 0
            
        return {
            'rds': risk_score,
            'direction': float(direction),
            'velocity': 0.1,
            'stability': 0.7
        }
        
    def _get_feature_importance(self, features: Dict) -> Dict[str, float]:
        """Get feature importance (simplified)"""
        # In production, use SHAP
        importance = {}
        risk_weights = {
            'emotion_sadness': 0.15,
            'emotion_hopelessness': 0.25,
            'emotion_anxiety': 0.10,
            'self_focus_score': 0.10,
            'night_posting_ratio': 0.10
        }
        
        for name, value in features.items():
            if name in risk_weights:
                importance[name] = value * risk_weights[name]
            else:
                importance[name] = value * 0.05
                
        return importance


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='MP-RDS Mental Health Risk Detection')
    parser.add_argument('--mode', choices=['analyze', 'api', 'dashboard'], default='analyze')
    parser.add_argument('--input', type=str, help='Input JSON file with posts')
    parser.add_argument('--model', type=str, help='Model weights file')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        if not args.input:
            print("Please provide --input file")
            return
            
        with open(args.input) as f:
            data = json.load(f)
            
        mprds = MPRDS(model_path=args.model)
        result = mprds.analyze(data['posts'], data.get('user_id', 'anonymous'))
        print(json.dumps(result, indent=2))
        
    elif args.mode == 'api':
        import uvicorn
        uvicorn.run("mp_rds.api.server:app", host="0.0.0.0", port=args.port, reload=True)
        
    elif args.mode == 'dashboard':
        import subprocess
        subprocess.run(["streamlit", "run", "mp_rds/dashboard/app.py"])


if __name__ == "__main__":
    main()
