import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.ml.models.price_prediction.lstm_predictor import price_predictor
from src.ml.training.price_prediction_trainer import prediction_trainer
from src.database.models import SessionLocal, MarketPredictions
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict
import schedule

logger = logging.getLogger(__name__)

class RealTimePredictionService:
    """Real-time cryptocurrency price prediction service"""
    
    def __init__(self):
        self.prediction_interval = 15  # minutes
        self.is_running = False
        self.total_predictions_made = 0
        
    def ensure_model_trained(self):
        """Ensure the prediction model is trained and ready"""
        
        if not price_predictor.is_trained:
            logger.info("🧠 Training price prediction model...")
            try:
                trained_model = prediction_trainer.train_prediction_model(
                    days_back=10, 
                    epochs=30
                )
                
                if trained_model:
                    logger.info("✅ Model training completed successfully")
                else:
                    logger.error("❌ Model training failed")
                    return False
            except Exception as e:
                logger.error(f"❌ Error during model training: {e}")
                return False
        
        return True
    
    def generate_and_store_predictions(self):
        """Generate predictions and store them in database"""
        
        try:
            # Generate predictions
            predictions = prediction_trainer.create_predictions_for_all_cryptos()
            
            if not predictions:
                logger.warning("⚠️ No predictions generated")
                return
            
            # Store predictions in database
            db = SessionLocal()
            try:
                for symbol, pred_data in predictions.items():
                    
                    # Store each prediction horizon
                    for i, (predicted_price, price_change, horizon) in enumerate(zip(
                        pred_data['predicted_prices'],
                        pred_data['price_changes_pct'], 
                        pred_data['prediction_horizons']
                    )):
                        
                        prediction_record = MarketPredictions(
                            crypto_symbol=symbol,
                            predicted_price=predicted_price,
                            prediction_horizon=int(horizon.replace('h', '')),  # Extract hours
                            confidence=pred_data['confidence'],
                            model_version='lstm-v1',
                            created_at=datetime.utcnow()
                        )
                        
                        db.add(prediction_record)
                
                db.commit()
                self.total_predictions_made += len(predictions) * 2  # 2 horizons per crypto
                
                logger.info(f"✅ Stored predictions for {len(predictions)} cryptocurrencies")
                
                # Log best predictions
                best_opportunities = []
                for symbol, pred in predictions.items():
                    max_change = max(pred['price_changes_pct'])
                    if abs(max_change) > 2 and pred['confidence'] > 0.7:  # Significant change with high confidence
                        best_opportunities.append((symbol, max_change, pred['confidence']))
                
                if best_opportunities:
                    best_opportunities.sort(key=lambda x: abs(x[1]), reverse=True)
                    logger.info("🎯 Top Trading Opportunities:")
                    for symbol, change, conf in best_opportunities[:3]:
                        direction = "📈 LONG" if change > 0 else "📉 SHORT"
                        logger.info(f"   {direction} {symbol}: {change:+.2f}% (Confidence: {conf:.0%})")
                
            except Exception as e:
                logger.error(f"❌ Error storing predictions: {e}")
                db.rollback()
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"❌ Error in prediction generation: {e}")
    
    def get_latest_predictions(self, hours: int = 1) -> Dict:
        """Get latest predictions from database"""
        
        db = SessionLocal()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            predictions = db.query(MarketPredictions).filter(
                MarketPredictions.created_at >= cutoff_time
            ).order_by(MarketPredictions.created_at.desc()).all()
            
            # Organize by crypto symbol
            latest_predictions = {}
            for pred in predictions:
                symbol = pred.crypto_symbol
                
                if symbol not in latest_predictions:
                    latest_predictions[symbol] = {
                        'symbol': symbol,
                        'predictions': [],
                        'confidence': pred.confidence,
                        'model_version': pred.model_version,
                        'created_at': pred.created_at.isoformat()
                    }
                
                latest_predictions[symbol]['predictions'].append({
                    'predicted_price': float(pred.predicted_price),
                    'horizon_hours': pred.prediction_horizon,
                    'confidence': float(pred.confidence)
                })
            
            return latest_predictions
            
        except Exception as e:
            logger.error(f"❌ Error getting latest predictions: {e}")
            return {}
        finally:
            db.close()
    
    def start_prediction_service(self):
        """Start the real-time prediction service"""
        
        logger.info("🚀 Starting Real-time Price Prediction Service")
        logger.info(f"⏰ Predictions will be generated every {self.prediction_interval} minutes")
        
        # Ensure model is trained
        if not self.ensure_model_trained():
            logger.error("❌ Cannot start service - model training failed")
            return
        
        # Schedule predictions
        schedule.every(self.prediction_interval).minutes.do(self.generate_and_store_predictions)
        
        # Generate initial predictions
        logger.info("🎯 Generating initial predictions...")
        self.generate_and_store_predictions()
        
        # Start service loop
        self.is_running = True
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping Real-time Prediction Service")
            self.is_running = False
        
        logger.info(f"📊 Service stopped. Total predictions made: {self.total_predictions_made}")
    
    def stop_service(self):
        """Stop the prediction service"""
        self.is_running = False

# Global prediction service
prediction_service = RealTimePredictionService()