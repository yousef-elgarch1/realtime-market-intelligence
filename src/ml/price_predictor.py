import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CryptoPricePredictor:
    """Advanced cryptocurrency price prediction using LSTM neural networks"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.sequence_length = 60  # Use 60 time periods for prediction
        self.prediction_horizons = [1, 6, 24]  # 1h, 6h, 24h predictions
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info("📈 Initializing Crypto Price Predictor with LSTM")
        
        # Feature columns for enhanced prediction
        self.feature_columns = [
            'price', 'volume_24h', 'market_cap', 'price_change_24h',
            'sentiment_score', 'fear_greed_index', 'sp500_change', 'nasdaq_change'
        ]
        
        self._setup_models()
    
    def _setup_models(self):
        """Setup LSTM models for different prediction horizons"""
        for horizon in self.prediction_horizons:
            model_name = f"lstm_model_{horizon}h"
            
            # Try to load existing model
            model_path = os.path.join(self.model_dir, f"{model_name}.h5")
            scaler_path = os.path.join(self.model_dir, f"scaler_{horizon}h.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[horizon] = load_model(model_path)
                    self.scalers[horizon] = joblib.load(scaler_path)
                    logger.info(f"✅ Loaded existing {horizon}h prediction model")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {horizon}h model: {e}")
                    self._create_new_model(horizon)
            else:
                self._create_new_model(horizon)
    
    def _create_new_model(self, horizon: int):
        """Create new LSTM model for specific prediction horizon"""
        logger.info(f"🏗️ Creating new LSTM model for {horizon}h prediction")
        
        # Create LSTM model architecture
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(25, activation='relu'),
            Dense(1)  # Single output for price prediction
        ])
        
        # Compile model with appropriate optimizer and loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        self.models[horizon] = model
        self.scalers[horizon] = MinMaxScaler(feature_range=(0, 1))
        
        logger.info(f"✅ Created {horizon}h LSTM model with {model.count_params():,} parameters")
    
    def prepare_training_data(self, price_data: pd.DataFrame, 
                            sentiment_data: pd.DataFrame = None,
                            market_data: pd.DataFrame = None) -> Dict[int, Tuple]:
        """Prepare training data for all prediction horizons"""
        logger.info("📊 Preparing training data for LSTM models")
        
        # Merge all data sources
        combined_data = self._merge_data_sources(price_data, sentiment_data, market_data)
        
        training_sets = {}
        
        for horizon in self.prediction_horizons:
            X, y = self._create_sequences(combined_data, horizon)
            
            if len(X) > 0:
                # Split into train/validation sets
                split_idx = int(len(X) * 0.8)
                
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                training_sets[horizon] = (X_train, X_val, y_train, y_val)
                
                logger.info(f"✅ Prepared {horizon}h data: {len(X_train)} train, {len(X_val)} validation samples")
            else:
                logger.warning(f"⚠️ Insufficient data for {horizon}h model")
        
        return training_sets
    
    def _merge_data_sources(self, price_data: pd.DataFrame,
                          sentiment_data: pd.DataFrame = None,
                          market_data: pd.DataFrame = None) -> pd.DataFrame:
        """Merge price, sentiment, and market data"""
        
        # Start with price data
        combined = price_data.copy()
        
        # Add sentiment features if available
        if sentiment_data is not None and not sentiment_data.empty:
            # Aggregate sentiment by time period
            sentiment_agg = sentiment_data.groupby('timestamp').agg({
                'sentiment_score': 'mean',
                'confidence': 'mean'
            }).reset_index()
            
            combined = pd.merge(combined, sentiment_agg, on='timestamp', how='left')
            combined['sentiment_score'] = combined['sentiment_score'].fillna(0)
        else:
            combined['sentiment_score'] = 0
        
        # Add market data if available
        if market_data is not None and not market_data.empty:
            combined = pd.merge(combined, market_data, on='timestamp', how='left')
            combined['fear_greed_index'] = combined['fear_greed_index'].fillna(50)
            combined['sp500_change'] = combined['sp500_change'].fillna(0)
            combined['nasdaq_change'] = combined['nasdaq_change'].fillna(0)
        else:
            combined['fear_greed_index'] = 50
            combined['sp500_change'] = 0
            combined['nasdaq_change'] = 0
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in combined.columns:
                combined[col] = 0
        
        # Sort by timestamp and reset index
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        return combined[self.feature_columns + ['timestamp']]
    
    def _create_sequences(self, data: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        if len(data) < self.sequence_length + horizon:
            logger.warning(f"Insufficient data for {horizon}h sequences")
            return np.array([]), np.array([])
        
        # Scale the features
        feature_data = data[self.feature_columns].values
        scaled_data = self.scalers[horizon].fit_transform(feature_data)
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_data) - horizon):
            # Features: sequence of past data
            X.append(scaled_data[i-self.sequence_length:i])
            
            # Target: future price (scaled)
            future_price_idx = 0  # price is first column
            y.append(scaled_data[i + horizon][future_price_idx])
        
        return np.array(X), np.array(y)
    
    def train_ml_models(**context):
    """Train sentiment analysis and price prediction models"""
    logger.info("🧠 Training ML models with latest data")
    
    # Load extracted data
    price_df = pd.read_parquet('/tmp/price_data.parquet')
    sentiment_df = pd.read_parquet('/tmp/sentiment_data.parquet')
    
    results = {}
    
    try:
        # Import ML modules
        from src.ml.sentiment_analyzer import financial_sentiment_analyzer
        from src.ml.price_predictor import crypto_price_predictor
        
        # Train sentiment models if we have social media data
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        social_query = """
        SELECT content, crypto_symbol, timestamp, like_count, retweet_count
        FROM social_media_posts 
        WHERE created_at >= NOW() - INTERVAL '7 days'
        ORDER BY timestamp
        """
        
        social_df = postgres_hook.get_pandas_df(social_query)
        
        if len(social_df) > 100:  # Only train if sufficient data
            logger.info("🧠 Training sentiment analysis models...")
            
            # Analyze sentiment for recent posts
            sample_texts = social_df['content'].head(50).tolist()
            sentiment_results = financial_sentiment_analyzer.analyze_batch(sample_texts)
            
            results['sentiment_analysis'] = {
                'samples_analyzed': len(sentiment_results),
                'avg_confidence': sum(r['confidence'] for r in sentiment_results) / len(sentiment_results)
            }
        
        # Train price prediction models
        if len(price_df) > 1000:  # Need sufficient historical data
            logger.info("📈 Training price prediction models...")
            
            # Group by symbol and train individual models
            symbols_trained = []
            for symbol in price_df['symbol'].unique():
                symbol_data = price_df[price_df['symbol'] == symbol].copy()
                
                if len(symbol_data) >= 200:  # Minimum data for LSTM
                    symbol_sentiment = sentiment_df[sentiment_df['symbol'] == symbol] if not sentiment_df.empty else None
                    
                    # Prepare training data
                    training_sets = crypto_price_predictor.prepare_training_data(
                        symbol_data, symbol_sentiment
                    )
                    
                    if training_sets:
                        # Train models
                        training_results = crypto_price_predictor.train_models(training_sets, epochs=20)
                        symbols_trained.append(symbol)
                        logger.info(f"✅ Trained models for {symbol}")
            
            results['price_prediction'] = {
                'symbols_trained': symbols_trained,
                'models_updated': len(symbols_trained)
            }
        
        logger.info(f"✅ ML model training completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in ML model training: {e}")
        return {'error': str(e)}

def generate_predictions(**context):
    """Generate predictions for all cryptocurrencies"""
    logger.info("🔮 Generating price predictions")
    
    try:
        from src.ml.price_predictor import crypto_price_predictor
        from src.ml.sentiment_analyzer import financial_sentiment_analyzer
        
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Get latest data for each symbol
        latest_data_query = """
        WITH latest_prices AS (
            SELECT symbol, 
                   MAX(timestamp) as latest_timestamp
            FROM crypto_prices 
            GROUP BY symbol
        )
        SELECT cp.symbol, cp.price, cp.volume_24h, cp.market_cap, 
               cp.price_change_24h, cp.timestamp
        FROM crypto_prices cp
        INNER JOIN latest_prices lp ON cp.symbol = lp.symbol 
                                   AND cp.timestamp = lp.latest_timestamp
        """
        
        latest_df = postgres_hook.get_pandas_df(latest_data_query)
        
        predictions_to_store = []
        
        for _, row in latest_df.iterrows():
            symbol = row['symbol']
            
            # Get recent price history for this symbol
            history_query = f"""
            SELECT price, volume_24h, market_cap, price_change_24h, timestamp
            FROM crypto_prices 
            WHERE symbol = '{symbol}'
            ORDER BY timestamp DESC
            LIMIT 100
            """
            
            history_df = postgres_hook.get_pandas_df(history_query)
            history_df = history_df.sort_values('timestamp').reset_index(drop=True)
            
            if len(history_df) >= 60:  # Minimum for LSTM sequence
                # Get recent sentiment
                sentiment_query = f"""
                SELECT sentiment_score, confidence
                FROM sentiment_analysis 
                WHERE crypto_symbol = '{symbol}'
                  AND processed_at >= NOW() - INTERVAL '24 hours'
                ORDER BY processed_at DESC
                LIMIT 10
                """
                
                try:
                    sentiment_df = postgres_hook.get_pandas_df(sentiment_query)
                    avg_sentiment = sentiment_df['sentiment_score'].mean() if not sentiment_df.empty else 0
                except:
                    avg_sentiment = 0
                
                # Include additional features
                include_features = {
                    'sentiment_score': avg_sentiment,
                    'fear_greed_index': 50,  # Default, can be updated with real data
                    'sp500_change': 0,  # Default, can be updated with real data
                    'nasdaq_change': 0   # Default, can be updated with real data
                }
                
                # Generate predictions
                prediction_result = crypto_price_predictor.predict_price(
                    history_df, symbol, include_features
                )
                
                if 'predictions' in prediction_result and prediction_result['predictions']:
                    for horizon, pred_data in prediction_result['predictions'].items():
                        predictions_to_store.append({
                            'crypto_symbol': symbol,
                            'predicted_price': pred_data['predicted_price'],
                            'prediction_horizon': pred_data['horizon_hours'],
                            'confidence': pred_data['confidence'],
                            'model_version': prediction_result['model_version'],
                            'created_at': datetime.utcnow()
                        })
        
        # Store predictions in database
        if predictions_to_store:
            pred_df = pd.DataFrame(predictions_to_store)
            pred_df.to_sql('market_predictions', postgres_hook.get_sqlalchemy_engine(), 
                          if_exists='append', index=False)
            
            logger.info(f"✅ Generated and stored {len(predictions_to_store)} predictions")
        
        return {'predictions_generated': len(predictions_to_store)}
        
    except Exception as e:
        logger.error(f"❌ Error generating predictions: {e}")
        return {'error': str(e)}

def update_sentiment_analysis(**context):
    """Analyze sentiment for recent social media posts"""
    logger.info("📱 Updating sentiment analysis")
    
    try:
        from src.ml.sentiment_analyzer import financial_sentiment_analyzer
        
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Get recent posts without sentiment analysis
        posts_query = """
        SELECT smp.id, smp.post_id, smp.content, smp.crypto_symbol,
               smp.like_count, smp.retweet_count, smp.follower_count
        FROM social_media_posts smp
        LEFT JOIN sentiment_analysis sa ON smp.post_id = sa.post_id
        WHERE sa.id IS NULL
          AND smp.created_at >= NOW() - INTERVAL '6 hours'
        ORDER BY smp.created_at DESC
        LIMIT 100
        """
        
        posts_df = postgres_hook.get_pandas_df(posts_query)
        
        sentiment_results = []
        
        for _, post in posts_df.iterrows():
            # Analyze sentiment
            sentiment_result = financial_sentiment_analyzer.analyze_text_sentiment(post['content'])
            
            # Calculate influence weight based on engagement
            engagement_score = (post['like_count'] + post['retweet_count'] * 2) / max(post['follower_count'], 1)
            influence_weight = min(engagement_score * 100, 1.0)
            
            sentiment_results.append({
                'post_id': post['post_id'],
                'crypto_symbol': post['crypto_symbol'],
                'sentiment_score': sentiment_result['sentiment_score'],
                'sentiment_label': sentiment_result['sentiment_label'],
                'confidence': sentiment_result['confidence'],
                'influence_weight': influence_weight,
                'processed_at': datetime.utcnow()
            })
        
        # Store sentiment analysis results
        if sentiment_results:
            sentiment_df = pd.DataFrame(sentiment_results)
            sentiment_df.to_sql('sentiment_analysis', postgres_hook.get_sqlalchemy_engine(),
                              if_exists='append', index=False)
            
            logger.info(f"✅ Analyzed sentiment for {len(sentiment_results)} posts")
        
        return {'posts_analyzed': len(sentiment_results)}
        
    except Exception as e:
        logger.error(f"❌ Error in sentiment analysis: {e}")
        return {'error': str(e)}

def validate_predictions(**context):
    """Validate previous predictions against actual prices"""
    logger.info("✅ Validating prediction accuracy")
    
    try:
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Get predictions from 1 hour ago to validate
        validation_query = """
        SELECT mp.crypto_symbol, mp.predicted_price, mp.prediction_horizon,
               mp.confidence, mp.created_at
        FROM market_predictions mp
        WHERE mp.created_at <= NOW() - INTERVAL '1 hour'
          AND mp.created_at >= NOW() - INTERVAL '2 hours'
          AND mp.prediction_horizon = 1
        """
        
        predictions_df = postgres_hook.get_pandas_df(validation_query)
        
        validation_results = []
        
        for _, pred in predictions_df.iterrows():
            # Get actual price at prediction time + horizon
            target_time = pred['created_at'] + timedelta(hours=pred['prediction_horizon'])
            
            actual_price_query = f"""
            SELECT price
            FROM crypto_prices
            WHERE symbol = '{pred['crypto_symbol']}'
              AND timestamp >= '{target_time - timedelta(minutes=30)}'
              AND timestamp <= '{target_time + timedelta(minutes=30)}'
            ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - '{target_time}')))
            LIMIT 1
            """
            
            actual_df = postgres_hook.get_pandas_df(actual_price_query)
            
            if not actual_df.empty:
                actual_price = actual_df['price'].iloc[0]
                predicted_price = pred['predicted_price']
                
                # Calculate accuracy metrics
                absolute_error = abs(actual_price - predicted_price)
                percentage_error = (absolute_error / actual_price) * 100
                
                validation_results.append({
                    'symbol': pred['crypto_symbol'],
                    'predicted_price': predicted_price,
                    'actual_price': actual_price,
                    'absolute_error': absolute_error,
                    'percentage_error': percentage_error,
                    'accuracy': max(0, 100 - percentage_error),
                    'horizon': pred['prediction_horizon']
                })
        
        if validation_results:
            avg_accuracy = sum(r['accuracy'] for r in validation_results) / len(validation_results)
            logger.info(f"✅ Validation completed: {avg_accuracy:.2f}% average accuracy")
            
            return {
                'predictions_validated': len(validation_results),
                'average_accuracy': avg_accuracy,
                'validation_details': validation_results[:5]  # Sample results
            }
        
        return {'predictions_validated': 0}
        
    except Exception as e:
        logger.error(f"❌ Error in prediction validation: {e}")
        return {'error': str(e)}

# ML Training DAG
ml_training_dag = DAG(
    'ml_model_training',
    default_args=default_args,
    description='Train ML models for market intelligence',
    schedule_interval=timedelta(hours=6),  # Retrain every 6 hours
    catchup=False,
    tags=['ml', 'training', 'bcg-x']
)

# ML Prediction DAG
ml_prediction_dag = DAG(
    'ml_predictions',
    default_args=default_args,
    description='Generate real-time predictions',
    schedule_interval=timedelta(minutes=30),  # Generate predictions every 30 minutes
    catchup=False,
    tags=['ml', 'predictions', 'bcg-x']
)

# Tasks for Training DAG
extract_data_task = PythonOperator(
    task_id='extract_training_data',
    python_callable=extract_training_data,
    dag=ml_training_dag
)

train_models_task = PythonOperator(
    task_id='train_ml_models',
    python_callable=train_ml_models,
    dag=ml_training_dag
)

validate_models_task = PythonOperator(
    task_id='validate_predictions',
    python_callable=validate_predictions,
    dag=ml_training_dag
)

# Tasks for Prediction DAG
sentiment_analysis_task = PythonOperator(
    task_id='update_sentiment_analysis',
    python_callable=update_sentiment_analysis,
    dag=ml_prediction_dag
)

generate_predictions_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=ml_prediction_dag
)

# Define task dependencies
extract_data_task >> train_models_task >> validate_models_task
sentiment_analysis_task >> generate_predictions_task