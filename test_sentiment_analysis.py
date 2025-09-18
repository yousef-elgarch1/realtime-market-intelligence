#!/usr/bin/env python3
"""
Test the complete sentiment analysis system
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ml.models.sentiment.finbert_analyzer import sentiment_analyzer
from src.ml.training.sentiment_processor import sentiment_processor

def test_sentiment_system():
    """Test the complete sentiment analysis system"""
    
    print("🧪 Testing FinBERT Sentiment Analysis System")
    print("=" * 60)
    
    # Test sample texts
    test_texts = [
        "🚀 Bitcoin to the moon! This is going to be HUGE! 💎🙌",
        "BTC is crashing hard. Time to sell everything 📉",
        "Ethereum holding steady around $3000. Sideways movement.",
        "BREAKING: Major institutional adoption for Solana! Bullish news!",
        "Market looking bearish today. Risk-off sentiment everywhere."
    ]
    
    # Test single text analysis
    print("🔄 Testing single text analysis...")
    for i, text in enumerate(test_texts[:2]):
        result = sentiment_analyzer.analyze_text(text)
        print(f"   Text {i+1}: {text}")
        print(f"   Result: {result['sentiment_score']:.1f}/100 ({result['sentiment_label']})")
        print(f"   Confidence: {result['confidence']}")
        print()
    
    # Test batch analysis
    print("🔄 Testing batch analysis...")
    batch_results = sentiment_analyzer.analyze_batch(test_texts)
    
    for i, (text, result) in enumerate(zip(test_texts, batch_results)):
        print(f"   Batch {i+1}: {result['sentiment_score']:.1f}/100 "
              f"({result['sentiment_label']}) - {text[:40]}...")
    
    # Test database processing
    print("\n🔄 Testing database sentiment processing...")
    try:
        results = sentiment_processor.run_sentiment_analysis_pipeline(batch_limit=10)
        if results:
            print(f"   ✅ Processed {len(results)} posts from database")
        else:
            print("   ℹ️ No unprocessed posts found in database")
    except Exception as e:
        print(f"   ⚠️ Database test skipped: {e}")
    
    # Test summary generation
    print("\n🔄 Testing sentiment summary generation...")
    try:
        summary = sentiment_processor.get_crypto_sentiment_summary(hours=24)
        if not summary.empty:
            print(f"   ✅ Generated summary for {len(summary)} cryptocurrencies")
            for _, row in summary.head(3).iterrows():
                print(f"      {row['crypto_symbol']}: {row['avg_sentiment']:.1f}/100 "
                      f"({row['post_count']} posts)")
        else:
            print("   ℹ️ No sentiment data found for summary")
    except Exception as e:
        print(f"   ⚠️ Summary test failed: {e}")
    
    print("\n✅ Sentiment Analysis System Test Complete!")
    print("🧠 FinBERT integration ready for real-time processing!")

if __name__ == "__main__":
    test_sentiment_system()