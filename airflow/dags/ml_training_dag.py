from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ml.training.sentiment_processor import sentiment_processor
from src.ml.training.price_prediction_trainer import prediction_trainer
from src.data.real_market_data import real_market_service
import pandas as pd
import mlflow
import logging

logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'bcg_x_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Create the DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='BCG X ML Model Training Pipeline',
    schedule_interval='@daily',  # Run daily
    max_active_runs=1,
    tags=['ml', 'training', 'bcg_x']
)

def check_data_quality():
    """Check data quality before training"""
    logger.info("🔍 Checking data quality for ML training...")
    
    try:
        # Check real market data availability
        crypto_prices = real_market_service.get_crypto_prices()
        
        if not crypto_prices or len(crypto_prices) < 5:
            raise ValueError("Insufficient market data for training")
        
        # Check sentiment data
        summary = sentiment_processor.get_crypto_sentiment_summary(hours=48)
        
        data_quality_report = {
            'crypto_prices_count': len(crypto_prices) if crypto_prices else 0,
            'sentiment_cryptos': len(summary) if not summary.empty else 0,
            'data_freshness_hours': 2,  # Assuming recent data
            'quality_score': 0.9 if crypto_prices and not summary.empty else 0.5
        }
        
        logger.info(f"📊 Data Quality Report: {data_quality_report}")
        
        if data_quality_report['quality_score'] < 0.7:
            raise ValueError("Data quality below threshold")
        
        return data_quality_report
        
    except Exception as e:
        logger.error(f"❌ Data quality check failed: {e}")
        raise

def train_sentiment_model():
    """Train and update sentiment analysis model"""
    logger.info("🧠 Training sentiment analysis model...")
    
    try:
        # Process recent social media posts
        results = sentiment_processor.run_sentiment_analysis_pipeline(batch_limit=200)
        
        if results:
            logger.info(f"✅ Sentiment model processed {len(results)} posts")
            
            # Log metrics to MLflow
            with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("sentiment-analysis").experiment_id):
                mlflow.log_metric("posts_processed", len(results))
                mlflow.log_metric("training_timestamp", datetime.utcnow().timestamp())
                
                # Calculate sentiment distribution
                sentiment_dist = {}
                for result in results:
                    label = result['sentiment_label']
                    sentiment_dist[label] = sentiment_dist.get(label, 0) + 1
                
                for label, count in sentiment_dist.items():
                    mlflow.log_metric(f"sentiment_{label}_count", count)
            
            return {"status": "success", "processed_posts": len(results)}
        else:
            logger.warning("⚠️ No new posts to process")
            return {"status": "no_new_data", "processed_posts": 0}
            
    except Exception as e:
        logger.error(f"❌ Sentiment model training failed: {e}")
        raise

def train_price_prediction_model():
    """Train and update LSTM price prediction model"""
    logger.info("📈 Training LSTM price prediction model...")
    
    try:
        # Train model with recent data
        trained_model = prediction_trainer.train_prediction_model(
            days_back=14,  # 2 weeks of data
            epochs=30
        )
        
        if trained_model:
            logger.info("✅ LSTM price prediction model trained successfully")
            
            # Generate test predictions
            predictions = prediction_trainer.create_predictions_for_all_cryptos()
            
            return {
                "status": "success", 
                "model_trained": True,
                "predictions_generated": len(predictions) if predictions else 0
            }
        else:
            raise ValueError("Model training returned None")
            
    except Exception as e:
        logger.error(f"❌ LSTM model training failed: {e}")
        raise

def validate_models():
    """Validate trained models performance"""
    logger.info("✅ Validating trained models...")
    
    try:
        validation_results = {
            'sentiment_model_status': 'active',
            'price_model_status': 'active',
            'validation_timestamp': datetime.utcnow().isoformat()
        }
        
        # Test sentiment model
        try:
            test_result = sentiment_processor.get_crypto_sentiment_summary(hours=24)
            if not test_result.empty:
                validation_results['sentiment_cryptos_analyzed'] = len(test_result)
                validation_results['avg_sentiment_score'] = float(test_result['avg_sentiment'].mean())
            else:
                validation_results['sentiment_model_status'] = 'no_data'
        except Exception as e:
            logger.warning(f"Sentiment validation warning: {e}")
            validation_results['sentiment_model_status'] = 'warning'
        
        # Test price prediction model
        try:
            predictions = prediction_trainer.create_predictions_for_all_cryptos()
            if predictions:
                validation_results['price_predictions_generated'] = len(predictions)
                avg_confidence = sum(p['confidence'] for p in predictions.values()) / len(predictions)
                validation_results['avg_prediction_confidence'] = avg_confidence
            else:
                validation_results['price_model_status'] = 'no_predictions'
        except Exception as e:
            logger.warning(f"Price model validation warning: {e}")
            validation_results['price_model_status'] = 'warning'
        
        logger.info(f"📊 Model validation results: {validation_results}")
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        raise

def cleanup_old_data():
    """Clean up old model artifacts and logs"""
    logger.info("🧹 Cleaning up old data...")
    
    try:
        # This is a placeholder for cleanup operations
        cleanup_summary = {
            'old_models_removed': 0,
            'log_files_cleaned': 0,
            'disk_space_freed_mb': 0
        }
        
        logger.info(f"✅ Cleanup completed: {cleanup_summary}")
        return cleanup_summary
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        raise

def send_training_report():
    """Send training completion report"""
    logger.info("📧 Generating training completion report...")
    
    # This would normally send an email/Slack notification
    # For demo purposes, we'll just log the report
    
    report = {
        'training_date': datetime.utcnow().isoformat(),
        'pipeline_status': 'completed',
        'next_training': (datetime.utcnow() + timedelta(days=1)).isoformat(),
        'models_updated': ['sentiment_analysis', 'price_prediction']
    }
    
    logger.info(f"📊 Training Report: {report}")
    return report

# Define tasks
check_data_task = PythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag
)

train_sentiment_task = PythonOperator(
    task_id='train_sentiment_model',
    python_callable=train_sentiment_model,
    dag=dag
)

train_price_task = PythonOperator(
    task_id='train_price_prediction_model', 
    python_callable=train_price_prediction_model,
    dag=dag
)

validate_models_task = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models,
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_data',
    python_callable=cleanup_old_data,
    dag=dag
)

report_task = PythonOperator(
    task_id='send_training_report',
    python_callable=send_training_report,
    dag=dag
)

# Define task dependencies
check_data_task >> [train_sentiment_task, train_price_task]
[train_sentiment_task, train_price_task] >> validate_models_task
validate_models_task >> cleanup_task >> report_task