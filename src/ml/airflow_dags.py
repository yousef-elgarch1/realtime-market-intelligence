from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, '/opt/airflow/dags')

logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'bcg-x-market-intelligence',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def extract_training_data(**context):
    """Extract data from PostgreSQL for model training"""
    logger.info("📊 Extracting training data from database")
    
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Extract price data
    price_query = """
    SELECT symbol, price, volume_24h, market_cap, price_change_24h, timestamp
    FROM crypto_prices 
    WHERE timestamp >= NOW() - INTERVAL '30 days'
    ORDER BY symbol, timestamp
    """
    
    price_df = postgres_hook.get_pandas_df(price_query)
    
    # Extract sentiment data
    sentiment_query = """
    SELECT sa.crypto_symbol as symbol, sa.sentiment_score, sa.confidence, sa.processed_at as timestamp
    FROM sentiment_analysis sa
    WHERE sa.processed_at >= NOW() - INTERVAL '30 days'
    ORDER BY sa.crypto_symbol, sa.processed_at
    """
    
    try:
        sentiment_df = postgres_hook.get_pandas_df(sentiment_query)
    except:
        # Create empty sentiment dataframe if table doesn't exist
        sentiment_df = pd.DataFrame(columns=['symbol', 'sentiment_score', 'confidence', 'timestamp'])
    
    # Save to temporary location for next task
    price_df.to_parquet('/tmp/price_data.parquet')
    sentiment_df.to_parquet('/tmp/sentiment_data.parquet')
    
    logger.info(f"✅ Extracted {len(price_df)} price records, {len(sentiment_df)} sentiment records")
    
    return {
        'price_records': len(price_df),
        'sentiment_records': len(sentiment_df)
    }

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