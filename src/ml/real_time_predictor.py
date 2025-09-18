import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
from kafka import KafkaProducer, KafkaConsumer
import threading
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.sentiment_analyzer import financial_sentiment_analyzer
from src.ml.price_predictor import crypto_price_predictor
from src.ml.mlflow_tracking import ml_tracker
from src.database.models import SessionLocal, SocialMediaPost, CryptoPrices, SentimentAnalysis, MarketPredictions
from src.data.real_market_data import real_market_service

logger = logging.getLogger(__name__)

class RealTimePredictionEngine:
    """Real-time ML prediction engine for market intelligence"""
    
    def __init__(self, kafka_servers: str = "localhost:9092"):
        self.kafka_servers = kafka_servers
        self.producer = None
        self.consumer = None
        self.running = False
        
        # Prediction cache to avoid duplicate predictions
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.predictions_made = 0
        self.sentiment_analyses = 0
        self.start_time = datetime.utcnow()
        
        self._setup_kafka()
        
    def _setup_kafka(self):
        """Setup Kafka producer and consumer for real-time processing"""
        try:
            # Producer for sending predictions
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            # Consumer for listening to new data
            self.consumer = KafkaConsumer(
                'crypto-prices', 'social-media-posts',
                bootstrap_servers=self.kafka_servers,
                group_id='ml_prediction_engine',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            logger.info("✅ Real-time prediction engine Kafka setup complete")
            
        except Exception as e:
            logger.error(f"❌ Kafka setup failed: {e}")
    
    def start_real_time_processing(self):
        """Start real-time ML processing"""
        logger.info("🚀 Starting real-time ML prediction engine")
        self.running = True
        
        # Start background threads for different processing tasks
        threads = [
            threading.Thread(target=self._process_incoming_data, daemon=True),
            threading.Thread(target=self._periodic_predictions, daemon=True),
            threading.Thread(target=self._performance_monitoring, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        try:
            # Main monitoring loop
            while self.running:
                self._health_check()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 models(self, training_sets: Dict[int, Tuple], epochs: int = 50) -> Dict:
        """Train LSTM models for all prediction horizons"""
        training_results = {}
        
        for horizon, (X_train, X_val, y_train, y_val) in training_sets.items():
            logger.info(f"🎯 Training {horizon}h prediction model...")
            
            # Callbacks for training
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            # Train the model
            history = self.models[horizon].fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            train_loss = self.models[horizon].evaluate(X_train, y_train, verbose=0)
            val_loss = self.models[horizon].evaluate(X_val, y_val, verbose=0)
            
            # Save model and scaler
            model_path = os.path.join(self.model_dir, f"lstm_model_{horizon}h.h5")
            scaler_path = os.path.join(self.model_dir, f"scaler_{horizon}h.pkl")
            
            self.models[horizon].save(model_path)
            joblib.dump(self.scalers[horizon], scaler_path)
            
            training_results[horizon] = {
                'final_train_loss': train_loss[0],
                'final_val_loss': val_loss[0],
                'epochs_trained': len(history.history['loss']),
                'best_val_loss': min(history.history['val_loss'])
            }
            
            logger.info(f"✅ {horizon}h model trained - Val Loss: {val_loss[0]:.6f}")
        
        return training_results
    
    def predict_price(self, recent_data: pd.DataFrame, 
                     crypto_symbol: str = "BTC",
                     include_features: Dict = None) -> Dict:
        """Make price predictions for all horizons"""
        try:
            if len(recent_data) < self.sequence_length:
                logger.warning(f"Insufficient data for prediction: {len(recent_data)} < {self.sequence_length}")
                return self._get_default_prediction(crypto_symbol)
            
            # Prepare features
            feature_data = self._prepare_prediction_features(recent_data, include_features)
            
            predictions = {}
            
            for horizon in self.prediction_horizons:
                if horizon not in self.models or horizon not in self.scalers:
                    continue
                
                # Scale features
                scaled_data = self.scalers[horizon].transform(feature_data)
                
                # Create sequence for prediction
                sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, len(self.feature_columns))
                
                # Make prediction
                scaled_prediction = self.models[horizon].predict(sequence, verbose=0)[0][0]
                
                # Inverse transform to get actual price
                # Create dummy array for inverse transform
                dummy_array = np.zeros((1, len(self.feature_columns)))
                dummy_array[0][0] = scaled_prediction  # price is first column
                
                actual_prediction = self.scalers[horizon].inverse_transform(dummy_array)[0][0]
                
                # Calculate prediction confidence (simplified)
                current_price = recent_data['price'].iloc[-1]
                price_change_pct = ((actual_prediction - current_price) / current_price) * 100
                
                # Confidence based on model certainty and market conditions
                confidence = self._calculate_prediction_confidence(
                    recent_data, price_change_pct, horizon
                )
                
                predictions[f"{horizon}h"] = {
                    'predicted_price': float(actual_prediction),
                    'current_price': float(current_price),
                    'price_change_pct': float(price_change_pct),
                    'confidence': float(confidence),
                    'horizon_hours': horizon,
                    'prediction_timestamp': datetime.utcnow().isoformat()
                }
            
            return {
                'symbol': crypto_symbol,
                'predictions': predictions,
                'model_version': '1.0.0',
                'features_used': self.feature_columns,
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in price prediction: {e}")
            return self._get_default_prediction(crypto_symbol)
    
    def _prepare_prediction_features(self, recent_data: pd.DataFrame, 
                                   include_features: Dict = None) -> np.ndarray:
        """Prepare features for prediction"""
        # Start with price data
        features = recent_data[['price', 'volume_24h', 'market_cap', 'price_change_24h']].copy()
        
        # Add additional features if provided
        if include_features:
            for feature_name, feature_value in include_features.items():
                if feature_name in self.feature_columns:
                    features[feature_name] = feature_value
        
        # Fill missing features with defaults
        for col in self.feature_columns:
            if col not in features.columns:
                if 'sentiment' in col:
                    features[col] = 0
                elif 'fear_greed' in col:
                    features[col] = 50
                else:
                    features[col] = 0
        
        return features[self.feature_columns].values
    
    def _calculate_prediction_confidence(self, recent_data: pd.DataFrame, 
                                       price_change_pct: float, horizon: int) -> float:
        """Calculate prediction confidence based on market conditions"""
        
        # Base confidence starts at 0.7
        confidence = 0.7
        
        # Adjust based on recent volatility
        recent_volatility = recent_data['price_change_24h'].std()
        if recent_volatility < 2:  # Low volatility = higher confidence
            confidence += 0.1
        elif recent_volatility > 10:  # High volatility = lower confidence
            confidence -= 0.2
        
        # Adjust based on prediction magnitude
        abs_change = abs(price_change_pct)
        if abs_change > 20:  # Extreme predictions = lower confidence
            confidence -= 0.3
        elif abs_change < 2:  # Conservative predictions = higher confidence
            confidence += 0.1
        
        # Adjust based on horizon (longer predictions = lower confidence)
        confidence -= (horizon - 1) * 0.05
        
        # Ensure confidence is between 0 and 1
        return max(0.1, min(0.95, confidence))
    
    def _get_default_prediction(self, crypto_symbol: str) -> Dict:
        """Return default prediction when models fail"""
        return {
            'symbol': crypto_symbol,
            'predictions': {},
            'error': 'Insufficient data or model unavailable',
            'model_version': '1.0.0',
            'prediction_timestamp': datetime.utcnow().isoformat()
        }
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all trained models"""
        performance = {}
        
        for horizon in self.prediction_horizons:
            if horizon in self.models:
                # This would require validation data to calculate properly
                # For now, return basic info
                performance[f"{horizon}h"] = {
                    'model_exists': True,
                    'parameters': self.models[horizon].count_params(),
                    'last_updated': datetime.utcnow().isoformat()
                }
            else:
                performance[f"{horizon}h"] = {
                    'model_exists': False,
                    'status': 'Not trained'
                }
        
        return performance

# Global instance
crypto_price_predictor = CryptoPricePredictor()