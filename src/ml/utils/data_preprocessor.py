import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class MarketDataPreprocessor:
    """Preprocess market data for ML models"""
    
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def prepare_sentiment_features(self, social_posts: List[Dict]) -> pd.DataFrame:
        """Prepare social media posts for sentiment analysis"""
        
        if not social_posts:
            return pd.DataFrame()
        
        df = pd.DataFrame(social_posts)
        
        # Basic feature engineering
        df['content_length'] = df['content'].str.len()
        df['word_count'] = df['content'].str.split().str.len()
        df['exclamation_count'] = df['content'].str.count('!')
        df['question_count'] = df['content'].str.count('?')
        df['emoji_count'] = df['content'].str.count('🚀|💰|📈|📉|💎|🙌')
        
        # Engagement features
        df['engagement_rate'] = (df['like_count'] + df['retweet_count']) / df['follower_count'].clip(lower=1)
        df['influence_weighted_engagement'] = df['engagement_rate'] * df.get('influence_score', 0.5)
        
        # Time features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
    
    def prepare_price_features(self, price_data: List[Dict], 
                             lookback_periods: int = 24) -> pd.DataFrame:
        """Prepare price data for prediction models"""
        
        if not price_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(price_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Technical indicators
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) >= 20:  # Minimum data for indicators
                # Moving averages
                symbol_data['sma_5'] = symbol_data['price'].rolling(5).mean()
                symbol_data['sma_20'] = symbol_data['price'].rolling(20).mean()
                symbol_data['ema_12'] = symbol_data['price'].ewm(span=12).mean()
                
                # Price momentum
                symbol_data['price_momentum'] = symbol_data['price'].pct_change(5)
                symbol_data['volume_momentum'] = symbol_data['volume_24h'].pct_change(5)
                
                # Volatility
                symbol_data['price_volatility'] = symbol_data['price'].rolling(10).std()
                
                # RSI (simplified)
                delta = symbol_data['price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.clip(lower=0.001)
                symbol_data['rsi'] = 100 - (100 / (1 + rs))
                
                # Update main dataframe
                df.loc[df['symbol'] == symbol, symbol_data.columns] = symbol_data
        
        return df
    
    def create_sequence_data(self, df: pd.DataFrame, 
                           target_col: str, 
                           sequence_length: int = 24,
                           prediction_horizon: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM model"""
        
        sequences = []
        targets = []
        
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'symbol', target_col]]
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].sort_values('timestamp')
            
            if len(symbol_data) < sequence_length + prediction_horizon:
                continue
            
            # Scale features
            feature_data = symbol_data[feature_cols].fillna(method='ffill').fillna(0)
            
            if not self.is_fitted:
                feature_data_scaled = self.price_scaler.fit_transform(feature_data)
            else:
                feature_data_scaled = self.price_scaler.transform(feature_data)
            
            # Create sequences
            for i in range(len(feature_data_scaled) - sequence_length - prediction_horizon + 1):
                # Input sequence
                seq = feature_data_scaled[i:i+sequence_length]
                sequences.append(seq)
                
                # Target (price after prediction_horizon)
                target_idx = i + sequence_length + prediction_horizon - 1
                target = symbol_data.iloc[target_idx][target_col]
                targets.append(target)
        
        self.is_fitted = True
        return np.array(sequences), np.array(targets)
    
    def aggregate_sentiment_by_crypto(self, sentiment_df: pd.DataFrame, 
                                    time_window: str = '1H') -> pd.DataFrame:
        """Aggregate sentiment scores by cryptocurrency and time window"""
        
        if sentiment_df.empty:
            return pd.DataFrame()
        
        # Group by symbol and time window
        sentiment_agg = sentiment_df.groupby(['crypto_symbol', 
                                            pd.Grouper(key='timestamp', freq=time_window)]).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'influence_weighted_engagement': 'sum',
            'content_length': 'mean',
            'emoji_count': 'sum'
        }).round(4)
        
        # Flatten column names
        sentiment_agg.columns = ['_'.join(col).strip() for col in sentiment_agg.columns]
        sentiment_agg = sentiment_agg.reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'sentiment_score_mean': 'avg_sentiment',
            'sentiment_score_std': 'sentiment_volatility', 
            'sentiment_score_count': 'post_volume',
            'influence_weighted_engagement_sum': 'total_engagement',
            'content_length_mean': 'avg_content_length',
            'emoji_count_sum': 'total_emojis'
        }
        
        sentiment_agg = sentiment_agg.rename(columns=column_mapping)
        
        return sentiment_agg

# Global preprocessor instance
data_preprocessor = MarketDataPreprocessor()