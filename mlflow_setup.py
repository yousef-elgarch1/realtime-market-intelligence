#!/usr/bin/env python3
"""
MLflow setup for Windows - BCG X Market Intelligence
"""
import mlflow
import os
from pathlib import Path

def setup_mlflow_windows():
    """Initialize MLflow for Windows"""
    
    # Create mlflow directory
    mlflow_dir = Path("mlflow")
    mlflow_dir.mkdir(exist_ok=True)
    
    # Set tracking URI using relative path (Windows compatible)
    mlflow.set_tracking_uri("./mlflow")
    
    # Create experiments with error handling
    experiments = [
        "sentiment-analysis",
        "price-prediction", 
        "market-intelligence"
    ]
    
    for exp_name in experiments:
        try:
            # Check if experiment already exists
            existing = mlflow.get_experiment_by_name(exp_name)
            if existing is None:
                experiment_id = mlflow.create_experiment(exp_name)
                print(f"✅ Created MLflow experiment: {exp_name}")
            else:
                print(f"ℹ️ MLflow experiment already exists: {exp_name}")
        except Exception as e:
            print(f"⚠️ Could not create experiment {exp_name}: {e}")
    
    print("\n✅ MLflow setup completed!")
    print("📊 Tracking URI: ./mlflow")
    print("💡 You can run 'mlflow ui' to view the dashboard")

if __name__ == "__main__":
    setup_mlflow_windows()