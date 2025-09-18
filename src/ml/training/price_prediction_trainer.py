import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.database.models import SessionLocal, CryptoPrices, SentimentAnalysis
from src.ml.models.price_prediction.lstm_predictor import price_predictor
from src.ml.training.sentiment_processor import sentiment_processor
import pandas as pd
from datetime import datetime, timedelta
import logging
import mlflow
from typing import Dict, List, Optional




logger = logging.getLogger(__name__)

class PricePredictionTrainer:
    """Training pipeline for cryptocurrency price prediction"""
    
    def __init__(self):
        self.min_training_samples = 100
        
    def get_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """Get historical price data for training"""
        
        db = SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            price_data = db.query(CryptoPrices).filter(
                CryptoPrices.timestamp >= cutoff_date
            ).order_by(CryptoPrices.timestamp.desc()).all()
            
            if not price_data:
                logger.warning("No price data found for training")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'symbol': price.symbol,
                    'price': float(price.price),
                    'volume_24h': float(price.volume_24h),
                    'market_cap': float(price.market_cap or 0),
                    'price_change_24h': float(price.price_change_24h),
                    'timestamp': price.timestamp
                }
                for price in price_data
            ])
            
            logger.info(f"📊 Retrieved {len(df)} price records for training")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting training data: {e}")
            return pd.DataFrame()
        finally:
            db.close()
    
    def get_sentiment_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """Get sentiment analysis data for training"""
        
        try:
            sentiment_summary = sentiment_processor.get_crypto_sentiment_summary(hours=days_back * 24)
            
            if not sentiment_summary.empty:
                # Convert to format expected by price predictor
                sentiment_df = sentiment_summary.rename(columns={
                    'crypto_symbol': 'symbol',
                    'avg_sentiment': 'sentiment_score',
                    'post_count': 'post_volume',
                    'total_engagement': 'engagement_rate'
                })
                
                # Add timestamp (use current time as approximation)
                sentiment_df['timestamp'] = datetime.utcnow()
                
                logger.info(f"📊 Retrieved sentiment data for {len(sentiment_df)} cryptocurrencies")
                return sentiment_df
            else:
                logger.info("ℹ️ No sentiment data available for training")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error getting sentiment training data: {e}")
            return pd.DataFrame()
    
    def train_prediction_model(self, days_back: int = 30, epochs: int = 50):
        """Train the complete price prediction model"""
        
        logger.info("🚀 Starting price prediction model training...")
        
        # Get training data
        price_data = self.get_training_data(days_back)
        sentiment_data = self.get_sentiment_training_data(days_back)
        
        if price_data.empty:
            logger.error("❌ No price data available for training")
            return None
        
        # Check minimum samples
        if len(price_data) < self.min_training_samples:
            logger.warning(f"⚠️ Limited training data: {len(price_data)} samples (minimum: {self.min_training_samples})")
        
        try:
            # Train the model
            price_predictor.train_model(
                train_data=price_data,
                sentiment_data=sentiment_data if not sentiment_data.empty else None,
                epochs=epochs,
                batch_size=16,  # Smaller batch size for limited data
                learning_rate=0.001
            )
            
            logger.info("✅ Price prediction model training completed!")
            
            # Evaluate on training data (basic check)
            if len(price_data) > 50:
                eval_results = price_predictor.evaluate_model(
                    price_data.tail(50),  # Use recent data for evaluation
                    sentiment_data if not sentiment_data.empty else None
                )
                
                logger.info(f"📊 Training Evaluation: {eval_results['accuracy_percentage']:.1f}% accuracy")
            
            return price_predictor
            
        except Exception as e:
            logger.error(f"❌ Error training price prediction model: {e}")
            return None
    
    def create_predictions_for_all_cryptos(self) -> Dict:
        """Generate predictions for all available cryptocurrencies"""
        
        if not price_predictor.is_trained:
            logger.warning("⚠️ Model not trained. Training with available data...")
            self.train_prediction_model(days_back=7, epochs=20)  # Quick training
        
        # Get recent data
        recent_price_data = self.get_training_data(days_back=3)  # Last 3 days
        recent_sentiment_data = self.get_sentiment_training_data(days_back=1)  # Last 24 hours
        
        if recent_price_data.empty:
            logger.error("❌ No recent price data for predictions")
            return {}
        
        try:
            predictions = price_predictor.predict(
                recent_price_data,
                recent_sentiment_data if not recent_sentiment_data.empty else None
            )
            
            if predictions:
                logger.info(f"🎯 Generated predictions for {len(predictions)} cryptocurrencies")
                
                # Log prediction summary
                for symbol, pred in predictions.items():
                    changes = pred['price_changes_pct']
                    confidence = pred['confidence']
                    logger.info(f"   {symbol}: {changes[0]:+.2f}% (1h), {changes[1]:+.2f}% (2h) "
                              f"[Confidence: {confidence:.0%}]")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error generating predictions: {e}")
            return {}

# Global trainer instance
prediction_trainer = PricePredictionTrainer()