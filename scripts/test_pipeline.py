#!/usr/bin/env python3
"""
Test the complete data pipeline
"""
import sys
import os
import time
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import SessionLocal, SocialMediaPost, CryptoPrices
from src.ingestion.kafka_producer import MarketDataProducer
from src.processing.kafka_consumer import MarketDataConsumer

def test_database_connection():
    """Test database connectivity"""
    print("🔄 Testing database connection...")
    
    try:
        db = SessionLocal()
        
        # Test query
        post_count = db.query(SocialMediaPost).count()
        price_count = db.query(CryptoPrices).count()
        
        print(f"✅ Database connected! Posts: {post_count}, Prices: {price_count}")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
    finally:
        db.close()

def test_producer():
    """Test Kafka producer"""
    print("🔄 Testing Kafka producer...")
    
    try:
        producer = MarketDataProducer()
        
        # Send test data
        producer.send_social_media_data()
        producer.send_crypto_prices()
        
        print("✅ Producer test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Producer test failed: {e}")
        return False

def run_pipeline_test(duration_seconds: int = 30):
    """Run full pipeline test"""
    print(f"🚀 Running full pipeline test for {duration_seconds} seconds...")
    
    # Start consumer in background
    consumer = MarketDataConsumer()
    consumer_thread = threading.Thread(target=consumer.start_consuming)
    consumer_thread.daemon = True
    consumer_thread.start()
    
    # Wait a bit for consumer to initialize
    time.sleep(2)
    
    # Start producer
    producer = MarketDataProducer()
    
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        producer.send_social_media_data()
        if time.time() - start_time > duration_seconds / 2:
            producer.send_crypto_prices()
        
        time.sleep(5)
    
    print("✅ Pipeline test completed!")
    
    # Check results
    db = SessionLocal()
    try:
        post_count = db.query(SocialMediaPost).count()
        price_count = db.query(CryptoPrices).count()
        print(f"📊 Final counts - Posts: {post_count}, Prices: {price_count}")
    finally:
        db.close()

if __name__ == "__main__":
    print("🧪 Market Intelligence Pipeline Test Suite")
    print("=" * 50)
    
    # Run tests
    if test_database_connection():
        if test_producer():
            run_pipeline_test(30)
        else:
            print("❌ Cannot proceed - producer test failed")
    else:
        print("❌ Cannot proceed - database connection failed")
    
    print("🏁 Test suite completed!")