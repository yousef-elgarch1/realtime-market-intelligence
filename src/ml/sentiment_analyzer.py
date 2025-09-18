import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import numpy as np
from typing import Dict, List, Tuple
import re
from datetime import datetime
import joblib
import os

logger = logging.getLogger(__name__)

class FinancialSentimentAnalyzer:
    """Advanced financial sentiment analysis using FinBERT and custom models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🧠 Initializing ML models on device: {self.device}")
        
        # Initialize FinBERT for financial sentiment
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.sentiment_pipeline = None
        
        # Custom sentiment patterns for crypto
        self.crypto_patterns = {
            'extremely_bullish': [
                r'moon\w*', r'rocket', r'🚀+', r'lambo', r'diamond\s*hands', r'💎+', 
                r'ath\s*incoming', r'breakout', r'pump\w*', r'bull\w*run'
            ],
            'bullish': [
                r'bull\w*', r'green', r'up\s*\d+%', r'gains?', r'profit', r'hodl',
                r'buy\s*the\s*dip', r'accumulate', r'support\s*broken'
            ],
            'bearish': [
                r'bear\w*', r'red', r'down\s*\d+%', r'dump\w*', r'crash\w*', r'sell',
                r'resistance', r'exit', r'stop\s*loss', r'support\s*fail\w*'
            ],
            'extremely_bearish': [
                r'rekt', r'rip', r'dead', r'💀+', r'blood\w*', r'massacre', 
                r'panic', r'capitulation', r'worthless'
            ]
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained financial sentiment models"""
        try:
            # Load FinBERT model for financial text analysis
            model_name = "ProsusAI/finbert"
            
            logger.info("📥 Loading FinBERT model...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("✅ FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading FinBERT model: {e}")
            # Fallback to simpler model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback sentiment model if FinBERT fails"""
        try:
            logger.info("🔄 Loading fallback sentiment model...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Fallback sentiment model loaded")
        except Exception as e:
            logger.error(f"❌ Fallback model failed: {e}")
            self.sentiment_pipeline = None
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of financial text using FinBERT"""
        try:
            if not self.sentiment_pipeline:
                return self._pattern_based_sentiment(text)
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Get FinBERT prediction
            result = self.sentiment_pipeline(cleaned_text)
            
            if isinstance(result, list):
                result = result[0]
            
            # Convert to standardized format
            label = result['label'].lower()
            confidence = result['score']
            
            # Map FinBERT labels to our format
            if 'positive' in label or 'bullish' in label:
                sentiment = 'bullish'
                score = confidence
            elif 'negative' in label or 'bearish' in label:
                sentiment = 'bearish'
                score = -confidence
            else:
                sentiment = 'neutral'
                score = 0.0
            
            # Enhance with pattern-based analysis
            pattern_result = self._pattern_based_sentiment(text)
            
            # Combine results with weighted average
            final_score = (score * 0.7) + (pattern_result['sentiment_score'] * 0.3)
            final_sentiment = self._score_to_label(final_score)
            
            return {
                'sentiment_label': final_sentiment,
                'sentiment_score': final_score,
                'confidence': confidence,
                'finbert_prediction': {
                    'label': label,
                    'score': confidence
                },
                'pattern_analysis': pattern_result,
                'processed_text': cleaned_text,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error in sentiment analysis: {e}")
            return self._pattern_based_sentiment(text)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep the content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length for model input
        if len(text) > 512:
            text = text[:512]
        
        return text
    
    def _pattern_based_sentiment(self, text: str) -> Dict:
        """Pattern-based sentiment analysis for crypto-specific language"""
        text_lower = text.lower()
        
        scores = {
            'extremely_bullish': 0,
            'bullish': 0,
            'bearish': 0,
            'extremely_bearish': 0
        }
        
        # Count pattern matches
        for sentiment_type, patterns in self.crypto_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                scores[sentiment_type] += matches
        
        # Calculate overall sentiment score (-1 to 1)
        positive_score = (scores['bullish'] * 0.5) + (scores['extremely_bullish'] * 1.0)
        negative_score = (scores['bearish'] * 0.5) + (scores['extremely_bearish'] * 1.0)
        
        total_signals = positive_score + negative_score
        if total_signals == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_score - negative_score) / max(total_signals, 1)
        
        # Determine label
        sentiment_label = self._score_to_label(sentiment_score)
        
        return {
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score,
            'confidence': min(total_signals / 5.0, 1.0),  # Confidence based on signal strength
            'pattern_matches': scores,
            'method': 'pattern_based'
        }
    
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score >= 0.6:
            return 'extremely_bullish'
        elif score >= 0.2:
            return 'bullish'
        elif score <= -0.6:
            return 'extremely_bearish'
        elif score <= -0.2:
            return 'bearish'
        else:
            return 'neutral'
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for multiple texts"""
        results = []
        
        for text in texts:
            result = self.analyze_text_sentiment(text)
            results.append(result)
        
        return results
    
    def get_aggregated_sentiment(self, sentiment_results: List[Dict], 
                               weights: List[float] = None) -> Dict:
        """Aggregate sentiment from multiple sources with optional weighting"""
        if not sentiment_results:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'sample_size': 0
            }
        
        if weights is None:
            weights = [1.0] * len(sentiment_results)
        
        # Weighted average of sentiment scores
        total_weight = sum(weights)
        weighted_score = sum(result['sentiment_score'] * weight 
                           for result, weight in zip(sentiment_results, weights))
        avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Average confidence
        avg_confidence = sum(result['confidence'] for result in sentiment_results) / len(sentiment_results)
        
        # Overall sentiment label
        overall_sentiment = self._score_to_label(avg_score)
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': avg_score,
            'confidence': avg_confidence,
            'sample_size': len(sentiment_results),
            'sentiment_distribution': self._calculate_distribution(sentiment_results),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def _calculate_distribution(self, sentiment_results: List[Dict]) -> Dict:
        """Calculate distribution of sentiment labels"""
        labels = [result['sentiment_label'] for result in sentiment_results]
        unique_labels = set(labels)
        
        distribution = {}
        for label in unique_labels:
            count = labels.count(label)
            distribution[label] = {
                'count': count,
                'percentage': (count / len(labels)) * 100
            }
        
        return distribution

# Global instance
financial_sentiment_analyzer = FinancialSentimentAnalyzer()