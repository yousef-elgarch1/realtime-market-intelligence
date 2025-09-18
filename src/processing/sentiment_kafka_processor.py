import json
import logging
from kafka import KafkaConsumer
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.ml.training.sentiment_processor import sentiment_processor
from src.ml.models.sentiment.finbert_analyzer import sentiment_analyzer
from src.database.models import SessionLocal, SentimentAnalysis
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class RealTimeSentimentProcessor:
    """Process social media posts for sentiment analysis in real-time"""
    
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.processed_count = 0
        
        # Initialize sentiment analyzer
        sentiment_analyzer.load_model()
    
    def create_consumer(self):
        """Create Kafka consumer for social media posts"""
        return KafkaConsumer(
            'social-media-posts',
            bootstrap_servers=self.bootstrap_servers,
            group_id='sentiment-analysis-group',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
    
    def process_post_sentiment(self, post_data: dict):
        """Process sentiment for a single social media post"""
        try:
            # Extract text content
            content = post_data.get('content', '')
            post_id = post_data.get('post_id', '')
            crypto_symbol = post_data.get('crypto_symbol', '')
            
            if not content or not post_id:
                logger.warning(f"⚠️ Invalid post data: {post_data}")
                return
            
            # Check if already processed
            db = SessionLocal()
            try:
                existing = db.query(SentimentAnalysis).filter(
                    SentimentAnalysis.post_id == post_id
                ).first()
                
                if existing:
                    logger.debug(f"⏭️ Post {post_id} already processed")
                    return
                
                # Analyze sentiment
                sentiment_result = sentiment_analyzer.analyze_text(content)
                
                # Save to database
                sentiment_record = SentimentAnalysis(
                    post_id=post_id,
                    crypto_symbol=crypto_symbol,
                    sentiment_score=sentiment_result['sentiment_score'],
                    sentiment_label=sentiment_result['sentiment_label'],
                    confidence=sentiment_result['confidence'],
                    processed_at=datetime.utcnow()
                )
                
                db.add(sentiment_record)
                db.commit()
                
                self.processed_count += 1
                
                logger.info(f"✅ Sentiment processed: {crypto_symbol} - {sentiment_result['sentiment_score']:.1f} "
                          f"({sentiment_result['sentiment_label']}) - {content[:40]}...")
                
                # Log every 10 processed posts
                if self.processed_count % 10 == 0:
                    logger.info(f"📊 Total sentiment analyses completed: {self.processed_count}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"❌ Error processing sentiment for post {post_data.get('post_id', 'unknown')}: {e}")
    
    def start_real_time_processing(self):
        """Start real-time sentiment analysis of social media posts"""
        
        consumer = self.create_consumer()
        logger.info("🧠 Starting real-time sentiment analysis processor...")
        logger.info("📊 Analyzing social media posts with FinBERT")
        
        try:
            for message in consumer:
                try:
                    post_data = message.value
                    self.process_post_sentiment(post_data)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing Kafka message: {e}")
                    
        except KeyboardInterrupt:
            logger.info("🛑 Stopping real-time sentiment processor")
        finally:
            consumer.close()
            logger.info(f"📊 Final stats: {self.processed_count} posts analyzed")

if __name__ == "__main__":
    processor = RealTimeSentimentProcessor()
    processor.start_real_time_processing()