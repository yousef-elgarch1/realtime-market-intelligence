from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class SocialMediaPost(Base):
    """Social media posts (tweets, reddit posts, etc.)"""
    __tablename__ = "social_media_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String(100), unique=True, index=True)
    platform = Column(String(50))  # twitter, reddit, telegram
    content = Column(Text)
    author = Column(String(100))
    crypto_symbol = Column(String(10), index=True)  # BTC, ETH, etc.
    timestamp = Column(DateTime, default=datetime.utcnow)
    follower_count = Column(Integer, default=0)
    retweet_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class CryptoPrices(Base):
    """Cryptocurrency price data"""
    __tablename__ = "crypto_prices"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True)
    price = Column(Float)
    volume_24h = Column(Float)
    market_cap = Column(Float)
    price_change_24h = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class SentimentAnalysis(Base):
    """Sentiment analysis results"""
    __tablename__ = "sentiment_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String(100), index=True)
    crypto_symbol = Column(String(10), index=True)
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive, negative, neutral
    confidence = Column(Float)  # 0 to 1
    processed_at = Column(DateTime, default=datetime.utcnow)

class MarketPredictions(Base):
    """ML model predictions"""
    __tablename__ = "market_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    crypto_symbol = Column(String(10), index=True)
    predicted_price = Column(Float)
    prediction_horizon = Column(Integer)  # hours ahead
    confidence = Column(Float)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

# Database connection
# Load environment variables
load_dotenv()

# Database connection with better error handling
def get_database_url():
    """Get database URL with proper error handling"""
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    db = os.getenv('POSTGRES_DB', 'market_intelligence')
    user = os.getenv('POSTGRES_USER', 'admin')
    password = os.getenv('POSTGRES_PASSWORD', 'password123')
    
    # Validate port is a number
    try:
        port = int(port)
    except (ValueError, TypeError):
        port = 5432
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()