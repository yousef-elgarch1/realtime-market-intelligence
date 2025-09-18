import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.ml.prediction.real_time_predictor import prediction_service
from src.database.models import SessionLocal, MarketPredictions, CryptoPrices
import pandas as pd
from datetime import datetime, timedelta
import time

def display_prediction_dashboard():
    """Display real-time price prediction dashboard"""
    
    print("🎯 LSTM PRICE PREDICTION DASHBOARD")
    print("=" * 70)
    print("🧠 Deep Learning Cryptocurrency Price Forecasting")
    print("📈 Neural Network: LSTM with Attention Mechanism")
    print("=" * 70)
    
    try:
        while True:
            db = SessionLocal()
            
            try:
                # Get latest predictions
                latest_predictions = prediction_service.get_latest_predictions(hours=2)
                
                # Get total prediction count
                total_predictions = db.query(MarketPredictions).count()
                
                # Get recent predictions count
                recent_cutoff = datetime.utcnow() - timedelta(hours=1)
                recent_predictions = db.query(MarketPredictions).filter(
                    MarketPredictions.created_at >= recent_cutoff
                ).count()
                
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - LSTM PREDICTION METRICS")
                print(f"🧠 Total AI Predictions: {total_predictions:,}")
                print(f"🔥 Recent Predictions (1h): {recent_predictions}")
                print(f"🎯 Model: LSTM + Attention + Sentiment Integration")
                
                if latest_predictions:
                    print(f"\n💰 CRYPTOCURRENCY PRICE FORECASTS:")
                    
                    # Sort by latest predictions
                    sorted_predictions = sorted(
                        latest_predictions.items(), 
                        key=lambda x: x[1]['created_at'], 
                        reverse=True
                    )
                    
                    for symbol, pred_data in sorted_predictions:
                        predictions = pred_data['predictions']
                        confidence = pred_data['confidence']
                        
                        if predictions:
                            # Get 1h and 2h predictions
                            pred_1h = next((p for p in predictions if p['horizon_hours'] == 1), None)
                            pred_2h = next((p for p in predictions if p['horizon_hours'] == 2), None)
                            
                            # Get current price for comparison
                            current_price_query = db.query(CryptoPrices).filter(
                                CryptoPrices.symbol == symbol
                            ).order_by(CryptoPrices.timestamp.desc()).first()
                            
                            if current_price_query:
                                current_price = float(current_price_query.price)
                                
                                # Calculate percentage changes
                                change_1h = ((pred_1h['predicted_price'] - current_price) / current_price * 100) if pred_1h else 0
                                change_2h = ((pred_2h['predicted_price'] - current_price) / current_price * 100) if pred_2h else 0
                                
                                # Determine signal strength
                                if abs(change_2h) > 3 and confidence > 0.8:
                                    signal_strength = "🔥 STRONG"
                                elif abs(change_2h) > 2 and confidence > 0.7:
                                    signal_strength = "⚡ MEDIUM"
                                elif abs(change_2h) > 1:
                                    signal_strength = "📊 WEAK"
                                else:
                                    signal_strength = "⚪ NEUTRAL"
                                
                                # Direction indicator
                                direction = "🟢 UP" if change_2h > 0 else "🔴 DOWN" if change_2h < 0 else "⚪ FLAT"
                                
                                print(f"   {signal_strength} {symbol}: ${current_price:,.2f} → "
                                      f"${pred_1h['predicted_price'] if pred_1h else current_price:,.2f} (1h) → "
                                      f"${pred_2h['predicted_price'] if pred_2h else current_price:,.2f} (2h)")
                                print(f"      {direction} Changes: {change_1h:+.2f}% (1h), {change_2h:+.2f}% (2h) "
                                      f"| Confidence: {confidence:.0%}")
                
                # Show trading signals
                if latest_predictions:
                    trading_opportunities = []
                    
                    for symbol, pred_data in latest_predictions.items():
                        predictions = pred_data['predictions']
                        confidence = pred_data['confidence']
                        
                        if predictions and confidence > 0.7:
                            pred_2h = next((p for p in predictions if p['horizon_hours'] == 2), None)
                            if pred_2h:
                                current_price_query = db.query(CryptoPrices).filter(
                                    CryptoPrices.symbol == symbol
                                ).order_by(CryptoPrices.timestamp.desc()).first()
                                
                                if current_price_query:
                                    current_price = float(current_price_query.price)
                                    change_2h = ((pred_2h['predicted_price'] - current_price) / current_price * 100)
                                    
                                    if abs(change_2h) > 2:  # Significant movement
                                        trading_opportunities.append((symbol, change_2h, confidence))
                    
                    if trading_opportunities:
                        trading_opportunities.sort(key=lambda x: abs(x[1]), reverse=True)
                        print(f"\n🎯 AI TRADING SIGNALS (High Confidence):")
                        
                        for symbol, change, conf in trading_opportunities[:5]:
                            if change > 2:
                                signal = f"📈 LONG {symbol}"
                                action = "BUY"
                            elif change < -2:
                                signal = f"📉 SHORT {symbol}"
                                action = "SELL"
                            else:
                                continue
                            
                            print(f"   {signal}: {change:+.2f}% predicted | "
                                  f"Action: {action} | Confidence: {conf:.0%}")
                
                # Model performance summary
                print(f"\n🧠 LSTM Model Performance:")
                print(f"   • Architecture: LSTM + Attention Mechanism")
                print(f"   • Features: Price + Volume + Technical + Sentiment")
                print(f"   • Prediction Horizon: 1-2 hours ahead")
                print(f"   • Update Frequency: Every 15 minutes")
                print(f"   • Training Data: Real market prices + social sentiment")
                
                print(f"\n💡 BCG X Value: AI-powered trading signal generation")
                print(f"🎯 Business Impact: Automated decision support for traders")
                
            finally:
                db.close()
            
            time.sleep(60)  # Update every minute
            
    except KeyboardInterrupt:
        print("\n🛑 Prediction dashboard stopped")
        print("🎯 LSTM price prediction system ready for BCG X demo!")

if __name__ == "__main__":
    display_prediction_dashboard()