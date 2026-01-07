"""
Test script for MP-RDS system
"""
import sys
sys.path.insert(0, '.')

import json

# Test individual modules
print("=" * 60)
print("MP-RDS System Test")
print("=" * 60)

# 1. Test Timeline Processor
print("\n[1] Testing Timeline Processor...")
from mp_rds.data import TimelineProcessor

posts = [
    {"timestamp": "2024-01-01T10:00:00Z", "text": "Had a good day today"},
    {"timestamp": "2024-01-08T14:00:00Z", "text": "Feeling neutral about things"},
    {"timestamp": "2024-01-15T23:00:00Z", "text": "Can't sleep. Mind racing."},
]

processor = TimelineProcessor()
windows = processor.process_timeline(posts)
print(f"   ✓ Created {len(windows)} time windows")
print(f"   ✓ Timeline stats: {processor.get_timeline_stats(windows)}")

# 2. Test Feature Extractors
print("\n[2] Testing Feature Extractors...")
from mp_rds.features import FeatureExtractor

extractor = FeatureExtractor()

test_text = """I feel so hopeless and alone. Nothing matters anymore. 
I can't stop thinking about my failures. Everything I do is wrong. 
I'm always sad and I never feel happy about anything."""

features = extractor.linguistic.extract_features(test_text)
print(f"   ✓ Linguistic features: {len(features)} extracted")
print(f"     - TTR: {features['ttr']:.3f}")
print(f"     - Self-focus: {features['self_focus_score']:.3f}")
print(f"     - Absolutist ratio: {features['absolutist_ratio']:.3f}")

emo_features = extractor.emotional.extract_features(test_text)
print(f"   ✓ Emotional features: {len(emo_features)} extracted")
print(f"     - Sadness: {emo_features['emotion_sadness']:.3f}")
print(f"     - Hopelessness: {emo_features['emotion_hopelessness']:.3f}")

# 3. Test Model
print("\n[3] Testing Temporal Transformer Model...")
from mp_rds.models import DriftTransformer
import torch

model = DriftTransformer(d_features=20, d_model=64, n_heads=2, n_layers=2)
print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
x = torch.randn(2, 10, 20)  # batch=2, seq=10, features=20
output = model(x)
print(f"   ✓ Forward pass successful")
print(f"     - RDS output shape: {output['rds'].shape}")
print(f"     - RDS values: {output['rds'].tolist()}")

# 4. Test RDS Engine
print("\n[4] Testing RDS Engine...")
from mp_rds.models import RDSEngine, RiskLevel

engine = RDSEngine()

model_output = {'rds': 0.65, 'direction': 0.15, 'velocity': 0.1, 'stability': 0.7}
result = engine.compute_rds(model_output)

print(f"   ✓ RDS Result:")
print(f"     - Score: {result.score:.2f}")
print(f"     - Level: {result.level.value}")
print(f"     - Direction: {result.direction}")
print(f"     - Confidence: {result.confidence:.2f}")

# Test trajectory
history = [0.15, 0.22, 0.28, 0.35, 0.45, 0.52, 0.65]
trajectory = engine.analyze_trajectory(history)
print(f"   ✓ Trajectory Analysis:")
print(f"     - Trend: {trajectory['trend']}")
print(f"     - Velocity: {trajectory['velocity']:.3f}")
print(f"     - Accelerating: {trajectory['is_accelerating']}")

# 5. Test Explainability
print("\n[5] Testing Explainability...")
from mp_rds.explainability import SHAPExplainer

explainer = SHAPExplainer()

sample_features = {
    'emotion_sadness': 0.4,
    'emotion_hopelessness': 0.35,
    'self_focus_score': 0.5,
    'night_posting_ratio': 0.3,
    'emotion_joy': 0.05
}

explanations = explainer.explain_prediction(sample_features, list(sample_features.keys()), 0.65)
print(f"   ✓ Generated {len(explanations)} feature explanations")

explanation_text = explainer.generate_text_explanation(explanations, 0.65)
print(f"   ✓ Text explanation generated:")
print("-" * 40)
print(explanation_text)
print("-" * 40)

# 6. Test Full Pipeline
print("\n[6] Testing Full Pipeline (MPRDS)...")
from mp_rds import MPRDS

with open('examples/sample_input.json') as f:
    sample_data = json.load(f)

mprds = MPRDS()
result = mprds.analyze(sample_data['posts'], sample_data['user_id'])

print(f"   ✓ Full analysis completed:")
print(f"     - User: {result['user_id']}")
print(f"     - RDS Score: {result['rds_score']:.2f}")
print(f"     - Risk Level: {result['risk_level']}")
print(f"     - Direction: {result['direction']}")
print(f"     - Windows analyzed: {result['n_windows']}")

print("\n" + "=" * 60)
print("✓ All tests passed! MP-RDS system is functional.")
print("=" * 60)
print("\nTo run the API server:")
print("  python -m mp_rds.main --mode api")
print("\nTo run the dashboard:")
print("  streamlit run mp_rds/dashboard/app.py")
