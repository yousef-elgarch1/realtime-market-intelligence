#!/usr/bin/env python3
"""
Database setup script for Market Intelligence Platform
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import create_tables, SessionLocal, SocialMediaPost
from datetime import datetime
import random

def setup_database():
    """Initialize database with tables and sample data"""
    print("🔄 Setting up database...")
    
    # Create tables
    create_tables()
    
    # Add sample data for testing
    db = SessionLocal()
    try:
        # Check if data already exists
        if db.query(SocialMediaPost).count() == 0:
            print("📊 Adding sample data...")
            
            sample_posts = [
                {
                    "post_id": "tweet_001",
                    "platform": "twitter",
                    "content": "Bitcoin is looking bullish! 🚀 New ATH incoming?",
                    "author": "crypto_expert",
                    "crypto_symbol": "BTC",
                    "follower_count": 10000,
                    "retweet_count": 50,
                    "like_count": 200
                },
                {
                    "post_id": "tweet_002", 
                    "platform": "twitter",
                    "content": "Ethereum merge was a success, but price is struggling",
                    "author": "eth_analyst",
                    "crypto_symbol": "ETH",
                    "follower_count": 5000,
                    "retweet_count": 25,
                    "like_count": 100
                }
            ]
            
            for post_data in sample_posts:
                post = SocialMediaPost(**post_data)
                db.add(post)
            
            db.commit()
            print("✅ Sample data added successfully!")
        else:
            print("ℹ️ Database already has data")
            
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    setup_database()