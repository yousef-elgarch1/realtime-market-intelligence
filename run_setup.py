#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🔄 Setting up database tables...")
    try:
        from src.database.models import create_tables
        create_tables()
        print("✅ Database setup completed!")
        print("📊 Tables created: social_media_posts, crypto_prices, sentiment_analysis, market_predictions")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        import traceback
        traceback.print_exc()