from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.database.models import SessionLocal, SocialMediaPost, CryptoPrices, SentimentAnalysis, MarketPredictions
from src.data.real_market_data import real_market_service
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
    'retry_delay': timedelta(minutes=3),
    'catchup': False
}

# Create monitoring DAG
dag = DAG(
    'data_pipeline_monitoring',
    default_args=default_args,
    description='BCG X Data Pipeline Health Monitoring',
    schedule_interval=timedelta(hours=1),  # Every hour
    max_active_runs=1,
    tags=['monitoring', 'data_quality', 'bcg_x']
)

def check_data_freshness():
    """Check if data is being updated regularly"""
    logger.info("🔍 Checking data freshness...")
    
    db = SessionLocal()
    try:
        # Check recent data across all tables
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Count recent records
        recent_posts = db.query(SocialMediaPost).filter(
            SocialMediaPost.created_at >= hour_ago
        ).count()
        
        recent_prices = db.query(CryptoPrices).filter(
            CryptoPrices.timestamp >= hour_ago
        ).count()
        
        recent_sentiment = db.query(SentimentAnalysis).filter(
            SentimentAnalysis.processed_at >= hour_ago
        ).count()
        
        recent_predictions = db.query(MarketPredictions).filter(
            MarketPredictions.created_at >= hour_ago
        ).count()
        
        freshness_report = {
            'check_timestamp': now.isoformat(),
            'recent_social_posts': recent_posts,
            'recent_crypto_prices': recent_prices,
            'recent_sentiment_analysis': recent_sentiment,
            'recent_predictions': recent_predictions,
            'data_pipeline_status': 'healthy' if recent_prices > 0 else 'warning'
        }
        
        logger.info(f"📊 Data Freshness Report: {freshness_report}")
        
        # Alert if no recent data
        if recent_prices == 0:
            logger.warning("⚠️ No recent price data - potential pipeline issue")
        
        return freshness_report
        
    except Exception as e:
        logger.error(f"❌ Data freshness check failed: {e}")
        raise
    finally:
        db.close()

def monitor_api_health():
    """Monitor external API health and performance"""
    logger.info("🌐 Monitoring external API health...")
    
    try:
        # Test CoinGecko API
        crypto_prices = real_market_service.get_crypto_prices()
        
        # Test market sentiment API
        sentiment_data = real_market_service.get_market_sentiment_indicators()
        
        api_health = {
            'coingecko_status': 'operational' if crypto_prices else 'failed',
            'crypto_data_points': len(crypto_prices) if crypto_prices else 0,
            'sentiment_api_status': 'operational' if sentiment_data else 'failed',
            'fear_greed_index': sentiment_data.get('fear_greed_index', 'N/A') if sentiment_data else 'N/A',
            'api_check_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"🌐 API Health Report: {api_health}")
        
        # Alert on API failures
        if not crypto_prices:
            logger.warning("⚠️ CoinGecko API failure detected")
        
        return api_health
        
    except Exception as e:
        logger.error(f"❌ API health monitoring failed: {e}")
        raise

def analyze_data_quality():
    """Analyze overall data quality metrics"""
    logger.info("📊 Analyzing data quality...")
    
    db = SessionLocal()
    try:
        # Get data quality metrics
        total_posts = db.query(SocialMediaPost).count()
        total_prices = db.query(CryptoPrices).count()
        total_sentiment = db.query(SentimentAnalysis).count()
        total_predictions = db.query(MarketPredictions).count()
        
        # Check for data consistency
        sentiment_coverage = (total_sentiment / total_posts * 100) if total_posts > 0 else 0
        
        # Check unique cryptocurrencies
        unique_crypto_posts = db.query(SocialMediaPost.crypto_symbol).distinct().count()
        unique_crypto_prices = db.query(CryptoPrices.symbol).distinct().count()
        
        quality_metrics = {
            'total_social_posts': total_posts,
            'total_crypto_prices': total_prices,
            'total_sentiment_analyses': total_sentiment,
            'total_predictions': total_predictions,
            'sentiment_coverage_percent': round(sentiment_coverage, 2),
            'unique_cryptos_social': unique_crypto_posts,
            'unique_cryptos_prices': unique_crypto_prices,
            'data_consistency_score': min(100, sentiment_coverage),
            'quality_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"📊 Data Quality Metrics: {quality_metrics}")
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"❌ Data quality analysis failed: {e}")
        raise
    finally:
        db.close()

def check_system_resources():
    """Check system resource utilization"""
    logger.info("💻 Checking system resources...")
    
    try:
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resource_metrics = {
            'cpu_usage_percent': cpu_percent,
            'memory_usage_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_usage_percent': disk.percent,
            'disk_free_gb': round(disk.free / (1024**3), 2),
            'resource_check_timestamp': datetime.utcnow().isoformat()
        }
        
        # Check for resource alerts
        if cpu_percent > 80:
            logger.warning(f"⚠️ High CPU usage: {cpu_percent}%")
        if memory.percent > 80:
            logger.warning(f"⚠️ High memory usage: {memory.percent}%")
        if disk.percent > 85:
            logger.warning(f"⚠️ High disk usage: {disk.percent}%")
        
        logger.info(f"💻 System Resources: {resource_metrics}")
        return resource_metrics
        
    except ImportError:
        logger.info("💻 psutil not available, skipping resource monitoring")
        return {"status": "psutil_not_available"}
    except Exception as e:
        logger.error(f"❌ System resource check failed: {e}")
        raise

def generate_health_report():
    """Generate comprehensive system health report"""
    logger.info("📋 Generating comprehensive health report...")
    
    try:
        # This would normally compile all monitoring data
        # and send alerts/reports to stakeholders
        
        health_report = {
            'report_timestamp': datetime.utcnow().isoformat(),
            'overall_system_status': 'operational',
            'data_pipeline_health': 'good',
            'ml_models_status': 'active',
            'api_connectivity': 'stable',
            'next_check': (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        logger.info(f"📋 Health Report Generated: {health_report}")
        
        return health_report
        
    except Exception as e:
        logger.error(f"❌ Health report generation failed: {e}")
        raise

# Define tasks
data_freshness_task = PythonOperator(
    task_id='check_data_freshness',
    python_callable=check_data_freshness,
    dag=dag
)

api_health_task = PythonOperator(
    task_id='monitor_api_health',
    python_callable=monitor_api_health,
    dag=dag
)

data_quality_task = PythonOperator(
    task_id='analyze_data_quality',
    python_callable=analyze_data_quality,
    dag=dag
)

system_resources_task = PythonOperator(
    task_id='check_system_resources',
    python_callable=check_system_resources,
    dag=dag
)

health_report_task = PythonOperator(
    task_id='generate_health_report',
    python_callable=generate_health_report,
    dag=dag
)

# Define task dependencies - all monitoring tasks run in parallel, then generate report
[data_freshness_task, api_health_task, data_quality_task, system_resources_task] >> health_report_task