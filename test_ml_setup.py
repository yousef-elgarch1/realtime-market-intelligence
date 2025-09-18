#!/usr/bin/env python3
"""
Test ML infrastructure setup
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow
import torch
import transformers
from src.ml.utils.data_preprocessor import data_preprocessor

def test_ml_infrastructure():
    """Test all ML components"""
    
    print("🧪 Testing ML Infrastructure Setup")
    print("=" * 50)
    
    # Test MLflow
    try:
        mlflow.set_experiment("test-experiment")
        with mlflow.start_run():
            mlflow.log_param("test", "infrastructure")
            mlflow.log_metric("accuracy", 0.95)
        print("✅ MLflow: Working")
    except Exception as e:
        print(f"❌ MLflow: {e}")
    
    # Test PyTorch
    try:
        tensor = torch.tensor([1, 2, 3])
        print(f"✅ PyTorch: Working (tensor: {tensor})")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
    
    # Test Transformers
    try:
        from transformers import pipeline
        print("✅ Transformers: Library imported successfully")
    except Exception as e:
        print(f"❌ Transformers: {e}")
    
    # Test Data Preprocessor
    try:
        test_data = [{'content': 'BTC to the moon!', 'crypto_symbol': 'BTC', 
                     'timestamp': '2024-01-01T12:00:00', 'like_count': 100,
                     'retweet_count': 50, 'follower_count': 1000}]
        
        df = data_preprocessor.prepare_sentiment_features(test_data)
        print(f"✅ Data Preprocessor: Working (features: {len(df.columns)})")
    except Exception as e:
        print(f"❌ Data Preprocessor: {e}")
    
    print("\n🚀 ML Infrastructure Test Complete!")

if __name__ == "__main__":
    test_ml_infrastructure()