from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ml.prediction.real_time_predictor import prediction_service
from src.ml.training.price_prediction_trainer import prediction_trainer
import logging

logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'bcg_x_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'catchup': False
}

# Create DAG for frequent predictions
dag = DAG(
    'real_time_predictions',
    default_args=default_args,
    description='BCG X Real-time Price Predictions',
    schedule_interval=timedelta(minutes=15),  # Every 15 minutes
    max_active_runs=1,
    tags=['ml', 'predictions', 'real_time', 'bcg_x']
)

def ensure_model_ready():
    """Ensure prediction models are trained and ready"""
    logger.info("🔍 Checking model readiness...")
    
    try:
        # Ensure prediction service has trained model
        if not prediction_service.ensure_model_trained():
            raise ValueError("Prediction model is not ready")
        
        logger.info("✅ Models are ready for predictions")
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"❌ Model readiness check failed: {e}")
        raise

def generate_predictions():
    """Generate new price predictions"""
    logger.info("🎯 Generating real-time price predictions...")
    
    try:
        # Generate predictions for all cryptocurrencies
        prediction_service.generate_and_store_predictions()
        
        # Get latest predictions for validation
        latest_predictions = prediction_service.get_latest_predictions(hours=1)
        
        result = {
            "status": "success",
            "predictions_generated": len(latest_predictions),
            "cryptos_analyzed": list(latest_predictions.keys()) if latest_predictions else [],
            "generation_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"✅ Generated predictions for {len(latest_predictions)} cryptocurrencies")
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction generation failed: {e}")
        raise

def validate_predictions():
    """Validate the quality of generated predictions"""
    logger.info("✅ Validating prediction quality...")
    
    try:
        # Get recent predictions
        latest_predictions = prediction_service.get_latest_predictions(hours=1)
        
        if not latest_predictions:
            raise ValueError("No recent predictions found")
        
        # Validate prediction quality
        validation_results = {
            'total_predictions': len(latest_predictions),
            'avg_confidence': 0,
            'high_confidence_count': 0,
            'valid_predictions': 0
        }
        
        for symbol, pred_data in latest_predictions.items():
            confidence = pred_data.get('confidence', 0)
            predictions = pred_data.get('predictions', [])
            
            if predictions and confidence > 0:
                validation_results['valid_predictions'] += 1
                validation_results['avg_confidence'] += confidence
                
                if confidence > 0.7:
                    validation_results['high_confidence_count'] += 1
        
        if validation_results['valid_predictions'] > 0:
            validation_results['avg_confidence'] /= validation_results['valid_predictions']
        
        logger.info(f"📊 Prediction validation: {validation_results}")
        
        if validation_results['valid_predictions'] == 0:
            raise ValueError("No valid predictions generated")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Prediction validation failed: {e}")
        raise

def monitor_prediction_performance():
    """Monitor ongoing prediction performance"""
    logger.info("📊 Monitoring prediction performance...")
    
    try:
        # Get predictions from last 24 hours for performance analysis
        latest_predictions = prediction_service.get_latest_predictions(hours=24)
        
        performance_metrics = {
            'active_predictions': len(latest_predictions),
            'monitoring_timestamp': datetime.utcnow().isoformat(),
            'system_status': 'operational'
        }
        
        # Check for any concerning patterns
        if len(latest_predictions) == 0:
            performance_metrics['system_status'] = 'no_predictions'
        elif len(latest_predictions) < 3:
            performance_metrics['system_status'] = 'limited_data'
        
        logger.info(f"📈 Performance monitoring: {performance_metrics}")
        return performance_metrics
        
    except Exception as e:
        logger.error(f"❌ Performance monitoring failed: {e}")
        raise

# Define tasks
model_ready_task = PythonOperator(
    task_id='ensure_model_ready',
    python_callable=ensure_model_ready,
    dag=dag
)

generate_predictions_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag
)

validate_predictions_task = PythonOperator(
    task_id='validate_predictions',
    python_callable=validate_predictions,
    dag=dag
)

monitor_performance_task = PythonOperator(
    task_id='monitor_prediction_performance',
    python_callable=monitor_prediction_performance,
    dag=dag
)

# Define task dependencies
model_ready_task >> generate_predictions_task >> validate_predictions_task >> monitor_performance_task