import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError
from typing import Dict, List
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.real_market_data import real_market_service
from src.ingestion.social_media_generator import RealSocialMediaGenerator

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealMarketDataProducer:
    """Kafka producer for REAL market intelligence data"""
    
    def __init__(self):
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.social_generator = RealSocialMediaGenerator()
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            retries=3,
            acks='all',
            compression_type='gzip'
        )
        
        logger.info("✅ Real Market Data Kafka producer initialized")

    def send_real_crypto_prices(self, topic: str = 'crypto-prices'):
        """Send REAL cryptocurrency price data"""
        try:
            # Get real price data
            real_prices = real_market_service.get_crypto_prices()
            
            if real_prices and len(real_prices) > 0:
                for price_data in real_prices:
                    # Send to Kafka
                    future = self.producer.send(
                        topic,
                        key=price_data['symbol'],
                        value=price_data
                    )
                    
                    # Log with real price information
                    change_emoji = "🟢" if price_data['price_change_24h'] > 0 else "🔴" if price_data['price_change_24h'] < 0 else "⚪"
                    logger.info(f"💰 Sent REAL {price_data['symbol']} price: ${price_data['price']:,.2f} "
                              f"({change_emoji}{price_data['price_change_24h']:+.2f}% 24h) "
                              f"Vol: ${price_data['volume_24h']:,.0f}")
                
                self.producer.flush()
                logger.info(f"✅ Successfully sent REAL price data for {len(real_prices)} cryptocurrencies")
                return real_prices
            else:
                logger.warning("⚠️ No real price data available")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error sending real price data: {e}")
            return None

    def send_real_social_media_data(self, topic: str = 'social-media-posts'):
        """Send social media posts based on REAL market data"""
        try:
            # Get current real price data for context
            real_prices = real_market_service.get_crypto_prices()
            
            if not real_prices:
                logger.warning("⚠️ No real price data for social media generation")
                return
            
            # Get market sentiment
            sentiment_data = real_market_service.get_market_sentiment_indicators()
            market_sentiment = sentiment_data.get('market_trend', 'neutral')
            
            # Generate posts based on real data
            posts = self.social_generator.generate_batch_from_real_data(
                real_prices, 
                market_sentiment, 
                count=5
            )
            
            for post in posts:
                # Send to Kafka
                future = self.producer.send(
                    topic,
                    key=post['crypto_symbol'],
                    value=post
                )
                
                # Log the message with real price context
                price_info = ""
                if 'real_price_data' in post:
                    price_change = post['real_price_data']['price_change_24h']
                    price_info = f" (Real: {price_change:+.1f}%)"
                
                logger.info(f"📤 Sent REAL-DATA {post['crypto_symbol']} post{price_info}: {post['content'][:60]}...")
                
            self.producer.flush()
            logger.info(f"✅ Successfully sent {len(posts)} real-data social media posts")
            
        except Exception as e:
            logger.error(f"❌ Error sending real social media data: {e}")

    def send_market_sentiment_data(self, topic: str = 'market-sentiment'):
        """Send real market sentiment indicators"""
        try:
            sentiment_data = real_market_service.get_market_sentiment_indicators()
            
            if sentiment_data:
                future = self.producer.send(
                    topic,
                    key="market_sentiment",
                    value=sentiment_data
                )
                
                logger.info(f"📊 Sent market sentiment: {sentiment_data['market_trend']} "
                          f"(F&G Index: {sentiment_data['fear_greed_index']})")
                
                self.producer.flush()
                
        except Exception as e:
            logger.error(f"❌ Error sending market sentiment data: {e}")

    def send_stock_market_correlation(self, topic: str = 'stock-market-data'):
        """Send stock market data for crypto correlation analysis"""
        try:
            stock_data = real_market_service.get_stock_market_data()
            
            if stock_data:
                future = self.producer.send(
                    topic,
                    key="stock_indices",
                    value=stock_data
                )
                
                # Log major indices changes
                indices = stock_data.get('indices', {})
                if indices:
                    sp500_change = indices.get('GSPC', {}).get('change_24h', 0)
                    nasdaq_change = indices.get('IXIC', {}).get('change_24h', 0)
                    logger.info(f"📈 Sent stock data: S&P {sp500_change:+.2f}%, NASDAQ {nasdaq_change:+.2f}%")
                
                self.producer.flush()
                
        except Exception as e:
            logger.error(f"❌ Error sending stock market data: {e}")

    def start_real_streaming(self, interval_seconds: int = 15):
        """Start continuous REAL data streaming"""
        logger.info(f"🚀 Starting REAL market data streaming (every {interval_seconds} seconds)")
        logger.info("📊 Data sources: CoinGecko (crypto), Yahoo Finance (stocks), Alternative.me (sentiment)")
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                logger.info(f"🔄 Cycle {cycle_count} - Fetching real market data...")
                
                # Send real crypto prices (every cycle)
                real_prices = self.send_real_crypto_prices()
                
                # Send social media data based on real prices (every cycle)
                if real_prices:
                    self.send_real_social_media_data()
                
                # Send market sentiment (every 3rd cycle)
                if cycle_count % 3 == 0:
                    self.send_market_sentiment_data()
                
                # Send stock market data (every 5th cycle)
                if cycle_count % 5 == 0:
                    self.send_stock_market_correlation()
                
                logger.info(f"✅ Cycle {cycle_count} completed. Next update in {interval_seconds} seconds")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping real data streaming")
        finally:
            self.producer.close()

if __name__ == "__main__":
    producer = RealMarketDataProducer()
    producer.start_real_streaming()