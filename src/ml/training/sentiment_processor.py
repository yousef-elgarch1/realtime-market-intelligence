import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.database.models import SessionLocal, SocialMediaPost, SentimentAnalysis
from src.ml.models.sentiment.finbert_analyzer import sentiment_analyzer
from datetime import datetime, timedelta
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)

class SentimentProcessor:
    """Process social media posts through sentiment analysis pipeline"""
    
    def __init__(self):
        self.processed_count = 0
        self.batch_size = 50
    
    def get_unprocessed_posts(self, limit: int = 100) -> pd.DataFrame:
        """Get social media posts that haven't been sentiment analyzed"""
        
        db = SessionLocal()
        try:
            # Query posts that don't have sentiment analysis
            query = """
            SELECT sp.* FROM social_media_posts sp
            LEFT JOIN sentiment_analysis sa ON sp.post_id = sa.post_id
            WHERE sa.post_id IS NULL
            ORDER BY sp.created_at DESC
            LIMIT :limit
            """
            
            posts = db.execute(query, {'limit': limit}).fetchall()
            
            if posts:
                # Convert to DataFrame
                posts_df = pd.DataFrame([
                    {
                        'id': post.id,
                        'post_id': post.post_id,
                        'content': post.content,
                        'crypto_symbol': post.crypto_symbol,
                        'author': post.author,
                        'platform': post.platform,
                        'like_count': post.like_count,
                        'retweet_count': post.retweet_count,
                        'follower_count': post.follower_count,
                        'timestamp': post.timestamp,
                        'created_at': post.created_at
                    }
                    for post in posts
                ])
                
                logger.info(f"📊 Found {len(posts_df)} unprocessed posts")
                return posts_df
            else:
                logger.info("ℹ️ No unprocessed posts found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error getting unprocessed posts: {e}")
            return pd.DataFrame()
        finally:
            db.close()
    
    def process_sentiment_batch(self, posts_df: pd.DataFrame) -> List[Dict]:
        """Process a batch of posts through sentiment analysis"""
        
        if posts_df.empty:
            return []
        
        try:
            logger.info(f"🧠 Processing {len(posts_df)} posts with FinBERT...")
            
            # Analyze sentiment
            texts = posts_df['content'].tolist()
            sentiment_results = sentiment_analyzer.analyze_batch(texts)
            
            # Combine with post data
            processed_results = []
            for i, (_, post) in enumerate(posts_df.iterrows()):
                if i < len(sentiment_results):
                    result = sentiment_results[i]
                    
                    processed_results.append({
                        'post_id': post['post_id'],
                        'crypto_symbol': post['crypto_symbol'],
                        'sentiment_score': result['sentiment_score'],
                        'sentiment_label': result['sentiment_label'],
                        'confidence': result['confidence'],
                        'bullish_keywords': result.get('keyword_analysis', {}).get('bullish_keywords', 0),
                        'bearish_keywords': result.get('keyword_analysis', {}).get('bearish_keywords', 0),
                        'model_version': 'finbert-v1',
                        'processed_at': datetime.utcnow()
                    })
            
            logger.info(f"✅ Sentiment analysis completed for {len(processed_results)} posts")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Error in sentiment processing: {e}")
            return []
    
    def save_sentiment_results(self, sentiment_results: List[Dict]):
        """Save sentiment analysis results to database"""
        
        if not sentiment_results:
            return
        
        db = SessionLocal()
        try:
            for result in sentiment_results:
                # Check if already exists
                existing = db.query(SentimentAnalysis).filter(
                    SentimentAnalysis.post_id == result['post_id']
                ).first()
                
                if not existing:
                    sentiment_record = SentimentAnalysis(
                        post_id=result['post_id'],
                        crypto_symbol=result['crypto_symbol'],
                        sentiment_score=result['sentiment_score'],
                        sentiment_label=result['sentiment_label'],
                        confidence=1.0 if result['confidence'] == 'high' else 0.7 if result['confidence'] == 'medium' else 0.3,
                        processed_at=result['processed_at']
                    )
                    
                    db.add(sentiment_record)
            
            db.commit()
            logger.info(f"✅ Saved {len(sentiment_results)} sentiment analysis results")
            
        except Exception as e:
            logger.error(f"❌ Error saving sentiment results: {e}")
            db.rollback()
        finally:
            db.close()
    
    def run_sentiment_analysis_pipeline(self, batch_limit: int = 100):
        """Run the complete sentiment analysis pipeline"""
        
        mlflow.set_tracking_uri('./mlflow')

        
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("sentiment-analysis").experiment_id):
            
            # Get unprocessed posts
            posts_df = self.get_unprocessed_posts(batch_limit)
            
            if posts_df.empty:
                logger.info("ℹ️ No posts to process")
                return
            
            # Process in batches
            all_results = []
            
            for i in range(0, len(posts_df), self.batch_size):
                batch_df = posts_df.iloc[i:i+self.batch_size]
                
                logger.info(f"🔄 Processing batch {i//self.batch_size + 1} ({len(batch_df)} posts)")
                
                batch_results = self.process_sentiment_batch(batch_df)
                all_results.extend(batch_results)
                
                # Save batch results immediately
                self.save_sentiment_results(batch_results)
                self.processed_count += len(batch_results)
            
            # Log MLflow metrics
            if all_results:
                avg_sentiment = np.mean([r['sentiment_score'] for r in all_results])
                sentiment_distribution = {
                    'bullish': sum(1 for r in all_results if r['sentiment_label'] == 'bullish'),
                    'bearish': sum(1 for r in all_results if r['sentiment_label'] == 'bearish'),
                    'neutral': sum(1 for r in all_results if r['sentiment_label'] == 'neutral')
                }
                
                mlflow.log_metric("avg_sentiment_score", avg_sentiment)
                mlflow.log_metric("total_processed", len(all_results))
                mlflow.log_metric("bullish_count", sentiment_distribution['bullish'])
                mlflow.log_metric("bearish_count", sentiment_distribution['bearish'])
                mlflow.log_metric("neutral_count", sentiment_distribution['neutral'])
                
                logger.info(f"📊 Pipeline Complete: {len(all_results)} posts processed")
                logger.info(f"📈 Average sentiment: {avg_sentiment:.2f}")
                logger.info(f"📊 Distribution: {sentiment_distribution}")
            
            return all_results
    
    def get_crypto_sentiment_summary(self, hours: int = 24) -> pd.DataFrame:
        """Get sentiment summary for each cryptocurrency"""
        
        db = SessionLocal()
        try:
            # Get recent sentiment data
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            query = """
            SELECT 
                sa.crypto_symbol,
                AVG(sa.sentiment_score) as avg_sentiment,
                STDDEV(sa.sentiment_score) as sentiment_volatility,
                COUNT(*) as post_count,
                SUM(CASE WHEN sa.sentiment_label = 'bullish' THEN 1 ELSE 0 END) as bullish_count,
                SUM(CASE WHEN sa.sentiment_label = 'bearish' THEN 1 ELSE 0 END) as bearish_count,
                SUM(CASE WHEN sa.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
            FROM sentiment_analysis sa
            WHERE sa.processed_at >= :cutoff_time
            GROUP BY sa.crypto_symbol
            ORDER BY post_count DESC
            """
            
            result = db.execute(query, {'cutoff_time': cutoff_time})
            
            summary_df = pd.DataFrame([
                {
                    'crypto_symbol': row[0],
                    'avg_sentiment': float(row[1]) if row[1] else 50.0,
                    'sentiment_volatility': float(row[2]) if row[2] else 0.0,
                    'post_count': int(row[3]),
                    'bullish_count': int(row[4]),
                    'bearish_count': int(row[5]),
                    'neutral_count': int(row[6])
                }
                for row in result
            ])
            
            if not summary_df.empty:
                # Calculate additional metrics
                summary_df['bullish_ratio'] = summary_df['bullish_count'] / summary_df['post_count']
                summary_df['bearish_ratio'] = summary_df['bearish_count'] / summary_df['post_count']
                summary_df['neutral_ratio'] = summary_df['neutral_count'] / summary_df['post_count']
                
                # Generate trading signals
                summary_df['sentiment_signal'] = summary_df['avg_sentiment'].apply(
                    lambda x: 'STRONG_BUY' if x >= 80 else 
                             'BUY' if x >= 65 else
                             'NEUTRAL' if x >= 35 else
                             'SELL' if x >= 20 else 'STRONG_SELL'
                )
            
            return summary_df
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment summary: {e}")
            return pd.DataFrame()
        finally:
            db.close()

# Global processor instance
sentiment_processor = SentimentProcessor()