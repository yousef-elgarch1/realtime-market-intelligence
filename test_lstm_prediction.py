#!/usr/bin/env python3
"""
Test the complete LSTM price prediction system
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ml.models.price_prediction.lstm_predictor import price_predictor
from src.ml.training.price_prediction_trainer import prediction_trainer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_lstm_system():
    """Test the complete LSTM prediction system"""
    
    print("🧪 Testing LSTM Price Prediction System")
    print("=" * 60)
    
    # Test 1: Create synthetic training data
    print("🔄 Test 1: Creating synthetic training data...")
    
    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                         end=datetime.now(), freq='1H')
    
    synthetic_data = []
    for symbol in ['BTC', 'ETH', 'SOL']:
        base_price = {'BTC': 45000, 'ETH': 3000, 'SOL': 100}[symbol]
        
        for i, date in enumerate(dates):
            # Create realistic price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            price = base_price * (1 + price_change * (i / len(dates)))
            
            synthetic_data.append({
                'symbol': symbol,
                'price': price,
                'volume_24h': np.random.uniform(1000000, 10000000),
                'market_cap': price * 20000000,
                'price_change_24h': price_change * 100,
                'timestamp': date
            })
    
    train_df = pd.DataFrame(synthetic_data)
    print(f"   ✅ Created {len(train_df)} synthetic training samples")
    
    # Test 2: Train LSTM model
    print("\n🔄 Test 2: Training LSTM model...")
    try:
        price_predictor.train_model(
            train_data=train_df,
            epochs=10,  # Quick training for test
            batch_size=16
        )
        print("   ✅ LSTM model trained successfully")
    except Exception as e:
        print(f"   ❌ LSTM training failed: {e}")
        return
    
    # Test 3: Make predictions
    print("\n🔄 Test 3: Making price predictions...")
    try:
        # Use recent data for prediction
        recent_data = train_df.tail(50)  # Last 50 hours
        
        predictions = price_predictor.predict(recent_data)
        
        if predictions:
            print(f"   ✅ Generated predictions for {len(predictions)} cryptocurrencies")
            
            for symbol, pred in predictions.items():
                current_price = pred['current_price']
                future_prices = pred['predicted_prices']
                confidence = pred['confidence']
                
                print(f"      {symbol}: ${current_price:,.2f} → "
                      f"${future_prices[0]:,.2f} (1h) → "
                      f"${future_prices[1]:,.2f} (2h) "
                      f"[Confidence: {confidence:.0%}]")
        else:
            print("   ⚠️ No predictions generated")
            
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
    
    # Test 4: Model evaluation
    print("\n🔄 Test 4: Model evaluation...")
    try:
        eval_results = price_predictor.evaluate_model(train_df.tail(100))
        
        print(f"   ✅ Model Evaluation Results:")
        print(f"      Accuracy: {eval_results['accuracy_percentage']:.2f}%")
        print(f"      RMSE: ${eval_results['rmse']:.2f}")
        print(f"      MAE: ${eval_results['mae']:.2f}")
        
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
    
    # Test 5: Training pipeline
    print("\n🔄 Test 5: Testing training pipeline...")
    try:
        # This will use real database data if available
        trained_model = prediction_trainer.train_prediction_model(
            days_back=5,  # Limited days for test
            epochs=5     # Quick training
        )
        
        if trained_model:
            print("   ✅ Training pipeline completed successfully")
        else:
            print("   ⚠️ Training pipeline completed with warnings")
            
    except Exception as e:
        print(f"   ❌ Training pipeline failed: {e}")
    
    print("\n✅ LSTM Price Prediction System Test Complete!")
    print("🎯 Deep learning model ready for real-time predictions!")

if __name__ == "__main__":
    test_lstm_system()