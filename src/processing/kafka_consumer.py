import json
import logging
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from sqlalchemy.orm import Session
from datetime import datetime
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.database.models import SessionLocal, SocialMediaPost, CryptoPrices, SentimentAnalysis, MarketPredictions

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealMarketDataConsumer:
    """Kafka consumer that processes and stores REAL market data"""
    
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.processed_count = 0
        self.real_price_count = 0
        self.social_post_count = 0
        
    def create_consumer(self, topics: list, group_id: str = 'real-market-intelligence'):
        """Create Kafka consumer for real data processing"""
        return KafkaConsumer(
            *topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
    
    def process_real_social_media_post(self, post_data: dict, db: Session):
        """Process and store REAL social media post with enhanced data"""
        try:
            # Check if post already exists
            existing = db.query(SocialMediaPost).filter(
                SocialMediaPost.post_id == post_data['post_id']
            ).first()
            
            if existing:
                logger.debug(f"⏭️ Post {post_data['post_id']} already exists, skipping")
                return
            
            # Create new post record with real data
            post = SocialMediaPost(
                post_id=post_data['post_id'],
                platform=post_data['platform'],
                content=post_data['content'],
                author=post_data['author'],
                crypto_symbol=post_data['crypto_symbol'],
                timestamp=datetime.fromisoformat(post_data['timestamp']),
                follower_count=post_data.get('follower_count', 0),
                retweet_count=post_data.get('retweet_count', 0),
                like_count=post_data.get('like_count', 0)
            )
            
            db.add(post)
            db.commit()
            
            self.social_post_count += 1
            self.processed_count += 1
            
            # Enhanced logging for real data
            data_source = post_data.get('data_source', 'unknown')
            sentiment = post_data.get('sentiment_type', 'neutral')
            real_price_info = ""
            
            if 'real_price_data' in post_data:
                price_change = post_data['real_price_data']['price_change_24h']
                real_price_info = f" (REAL price: {price_change:+.1f}%)"
            
            logger.info(f"✅ Stored REAL social post #{self.social_post_count}: {post_data['crypto_symbol']} "
                       f"[{sentiment}]{real_price_info} - {post_data['content'][:40]}...")
            
        except Exception as e:
            logger.error(f"❌ Error processing real social media post: {e}")
            db.rollback()
    
    def process_real_crypto_price(self, price_data: dict, db: Session):
        """Process and store REAL crypto price data"""
        try:
            price = CryptoPrices(
                symbol=price_data['symbol'],
                price=price_data['price'],
                volume_24h=price_data.get('volume_24h', 0),
                market_cap=price_data.get('market_cap', 0),
                price_change_24h=price_data.get('price_change_24h', 0),
                timestamp=datetime.fromisoformat(price_data['timestamp'])
            )
            
            db.add(price)
            db.commit()
            
            self.real_price_count += 1
            self.processed_count += 1
            
            # Enhanced logging for real price data
            change_emoji = "🟢" if price_data['price_change_24h'] > 0 else "🔴" if price_data['price_change_24h'] < 0 else "⚪"
            data_source = price_data.get('data_source', 'unknown')
            
            logger.info(f"💰 Stored REAL price #{self.real_price_count}: {price_data['symbol']} "
                       f"${price_data['price']:,.2f} {change_emoji}{price_data['price_change_24h']:+.2f}% "
                       f"[{data_source}]")
            
        except Exception as e:
            logger.error(f"❌ Error processing real crypto price: {e}")
            db.rollback()
    
    def process_market_sentiment(self, sentiment_data: dict, db: Session):
        """Process and store market sentiment data"""
        try:
            # You can extend the database model to include sentiment data
            # For now, we'll log it
            fear_greed = sentiment_data.get('fear_greed_index', 50)
            trend = sentiment_data.get('market_trend', 'neutral')
            
            logger.info(f"📊 Market Sentiment: {trend.upper()} (Fear & Greed: {fear_greed}/100)")
            
        except Exception as e:
            logger.error(f"❌ Error processing market sentiment: {e}")
    
    def process_stock_market_data(self, stock_data: dict, db: Session):
        """Process and store stock market correlation data"""
        try:
            indices = stock_data.get('indices', {})
            
            if indices:
                sp500 = indices.get('GSPC', {})
                nasdaq = indices.get('IXIC', {})
                
                logger.info(f"📈 Stock Market: S&P 500 {sp500.get('change_24h', 0):+.2f}%, "
                          f"NASDAQ {nasdaq.get('change_24h', 0):+.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error processing stock market data: {e}")
    
    def start_consuming_real_data(self):
        """Start consuming REAL market data from Kafka topics"""
        topics = ['social-media-posts', 'crypto-prices', 'market-sentiment', 'stock-market-data']
        consumer = self.create_consumer(topics)
        
        logger.info(f"🔄 Starting REAL market data consumer for topics: {topics}")
        logger.info("📊 Processing live cryptocurrency and social media data...")
        
        try:
            for message in consumer:
                db = SessionLocal()
                
                try:
                    topic = message.topic
                    data = message.value
                    
                    if topic == 'social-media-posts':
                        self.process_real_social_media_post(data, db)
                    elif topic == 'crypto-prices':
                        self.process_real_crypto_price(data, db)
                    elif topic == 'market-sentiment':
                        self.process_market_sentiment(data, db)
                    elif topic == 'stock-market-data':
                        self.process_stock_market_data(data, db)
                    
                    # Log progress every 25 messages
                    if self.processed_count % 25 == 0:
                        logger.info(f"📊 REAL DATA PROCESSED: {self.processed_count} total "
                                  f"({self.social_post_count} posts, {self.real_price_count} prices)")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing real market message: {e}")
                finally:
                    db.close()
                    
        except KeyboardInterrupt:
            logger.info("🛑 Stopping real market data consumer")
        finally:
            consumer.close()
            logger.info(f"📊 Final Stats: {self.processed_count} total messages processed "
                       f"({self.social_post_count} social posts, {self.real_price_count} crypto prices)")

if __name__ == "__main__":
    consumer = RealMarketDataConsumer()
    consumer.start_consuming_real_data()