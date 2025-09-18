import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.ml.training.sentiment_processor import sentiment_processor
from src.database.models import SessionLocal, SentimentAnalysis, SocialMediaPost, CryptoPrices
import pandas as pd
from datetime import datetime, timedelta
import time

def display_sentiment_dashboard():
    """Display real-time sentiment analysis dashboard"""
    
    print("🧠 FINBERT SENTIMENT ANALYSIS DASHBOARD")
    print("=" * 70)
    print("📊 Real-time Social Media Sentiment Analysis")
    print("🎯 Financial Language Processing with FinBERT")
    print("=" * 70)
    
    try:
        while True:
            db = SessionLocal()
            
            try:
                # Get sentiment summary
                sentiment_summary = sentiment_processor.get_crypto_sentiment_summary(hours=24)
                
                # Get recent analysis count
                recent_cutoff = datetime.utcnow() - timedelta(minutes=30)
                recent_analyses = db.query(SentimentAnalysis).filter(
                    SentimentAnalysis.processed_at >= recent_cutoff
                ).count()
                
                total_analyses = db.query(SentimentAnalysis).count()
                
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - SENTIMENT ANALYSIS METRICS")
                print(f"🧠 Total Sentiment Analyses: {total_analyses:,}")
                print(f"🔥 Recent Analyses (30min): {recent_analyses}")
                print(f"📊 Model: FinBERT (Financial Language Processing)")
                
                if not sentiment_summary.empty:
                    print(f"\n💰 CRYPTOCURRENCY SENTIMENT ANALYSIS:")
                    
                    for _, crypto in sentiment_summary.iterrows():
                        symbol = crypto['crypto_symbol']
                        avg_sentiment = crypto['avg_sentiment']
                        post_count = crypto['post_count']
                        signal = crypto['sentiment_signal']
                        
                        # Sentiment indicator
                        if avg_sentiment >= 70:
                            sentiment_emoji = "🟢 VERY BULLISH"
                        elif avg_sentiment >= 60:
                            sentiment_emoji = "💚 BULLISH"
                        elif avg_sentiment >= 40:
                            sentiment_emoji = "⚪ NEUTRAL"
                        elif avg_sentiment >= 30:
                            sentiment_emoji = "🔴 BEARISH"
                        else:
                            sentiment_emoji = "💀 VERY BEARISH"
                        
                        bullish_pct = crypto['bullish_ratio'] * 100
                        bearish_pct = crypto['bearish_ratio'] * 100
                        
                        print(f"   {sentiment_emoji} {symbol}: {avg_sentiment:.1f}/100 "
                              f"({post_count} posts) - Signal: {signal}")
                        print(f"      📈 Bullish: {bullish_pct:.1f}% | "
                              f"📉 Bearish: {bearish_pct:.1f}% | "
                              f"⚖️ Volatility: {crypto['sentiment_volatility']:.1f}")
                
                # Show recent sentiment analyses
                recent_analyses_data = db.query(SentimentAnalysis).order_by(
                    SentimentAnalysis.processed_at.desc()
                ).limit(5).all()
                
                if recent_analyses_data:
                    print(f"\n📱 Latest Sentiment Analyses:")
                    for analysis in recent_analyses_data:
                        age = (datetime.utcnow() - analysis.processed_at).total_seconds()
                        time_str = f"{age:.0f}s" if age < 60 else f"{age/60:.0f}m"
                        
                        confidence_indicator = "🎯" if analysis.confidence == 'high' else "📊" if analysis.confidence == 'medium' else "🤔"
                        
                        print(f"   {confidence_indicator} [{time_str}] {analysis.crypto_symbol}: "
                              f"{analysis.sentiment_score:.1f} ({analysis.sentiment_label.upper()})")
                
                # Business intelligence
                if not sentiment_summary.empty:
                    top_bullish = sentiment_summary.nlargest(1, 'avg_sentiment')
                    top_bearish = sentiment_summary.nsmallest(1, 'avg_sentiment')
                    
                    if not top_bullish.empty:
                        top_bull = top_bullish.iloc[0]
                        print(f"\n📈 Most Bullish: {top_bull['crypto_symbol']} "
                              f"({top_bull['avg_sentiment']:.1f}/100)")
                    
                    if not top_bearish.empty:
                        top_bear = top_bearish.iloc[0]
                        print(f"📉 Most Bearish: {top_bear['crypto_symbol']} "
                              f"({top_bear['avg_sentiment']:.1f}/100)")
                
                print(f"\n🎯 AI-Powered Business Intelligence:")
                print(f"   • Real-time sentiment scoring with FinBERT")
                print(f"   • Financial language understanding")
                print(f"   • Automated trading signal generation")
                print(f"   • Risk sentiment monitoring")
                
                print(f"\n💡 BCG X Value: AI-driven market sentiment prediction")
                
            finally:
                db.close()
            
            time.sleep(60)  # Update every minute
            
    except KeyboardInterrupt:
        print("\n🛑 Sentiment dashboard stopped")
        print("🧠 FinBERT sentiment analysis system ready for BCG X demo!")

if __name__ == "__main__":
    display_sentiment_dashboard()