from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.models.sentiment.finbert_analyzer import sentiment_analyzer
from src.ml.prediction.real_time_predictor import prediction_service
from src.ml.training.sentiment_processor import sentiment_processor
from src.ml.training.price_prediction_trainer import prediction_trainer
from src.database.models import SessionLocal, SocialMediaPost, CryptoPrices, SentimentAnalysis, MarketPredictions
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BCG X Market Intelligence API",
    description="Enterprise-grade AI-powered cryptocurrency market analysis platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class SentimentRequest(BaseModel):
    text: str
    crypto_symbol: Optional[str] = None

class SentimentResponse(BaseModel):
    sentiment_score: float
    sentiment_label: str
    confidence: str
    analysis_timestamp: str

class PredictionResponse(BaseModel):
    crypto_symbol: str
    current_price: float
    predicted_prices: List[float]
    price_changes_pct: List[float]
    confidence: float
    prediction_horizons: List[str]
    timestamp: str

class MarketSummaryResponse(BaseModel):
    total_cryptos_tracked: int
    total_social_posts: int
    total_sentiment_analyses: int
    total_predictions: int
    avg_market_sentiment: float
    top_bullish_crypto: str
    top_bearish_crypto: str
    last_updated: str

# Health check endpoint
@app.get("/", tags=["System"])
async def root():
    """System health and information endpoint"""
    return {
        "service": "BCG X Market Intelligence API",
        "status": "operational",
        "version": "1.0.0",
        "capabilities": [
            "Real-time sentiment analysis",
            "AI-powered price predictions",
            "Market intelligence aggregation",
            "Enterprise data analytics"
        ],
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs"
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Comprehensive health check for all system components"""
    
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "components": {}
    }
    
    # Check database connectivity
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["components"]["database"] = f"unhealthy: {str(e)}"
        health_status["overall_status"] = "degraded"
    
    # Check ML models
    try:
        if sentiment_analyzer.is_loaded:
            health_status["components"]["sentiment_model"] = "healthy"
        else:
            health_status["components"]["sentiment_model"] = "not_loaded"
            
        if prediction_service.ensure_model_trained():
            health_status["components"]["prediction_model"] = "healthy"
        else:
            health_status["components"]["prediction_model"] = "not_trained"
    except Exception as e:
        health_status["components"]["ml_models"] = f"error: {str(e)}"
        health_status["overall_status"] = "degraded"
    
    return health_status

# Sentiment Analysis Endpoints
@app.post("/sentiment/analyze", response_model=SentimentResponse, tags=["Sentiment Analysis"])
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of text using FinBERT financial language model"""
    try:
        # Ensure sentiment analyzer is loaded
        if not sentiment_analyzer.is_loaded:
            sentiment_analyzer.load_model()
        
        # Analyze sentiment
        result = sentiment_analyzer.analyze_text(request.text)
        
        return SentimentResponse(
            sentiment_score=result['sentiment_score'],
            sentiment_label=result['sentiment_label'],
            confidence=result['confidence'],
            analysis_timestamp=result['processed_at']
        )
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@app.get("/sentiment/crypto/{crypto_symbol}", tags=["Sentiment Analysis"])
async def get_crypto_sentiment(crypto_symbol: str, hours: int = 24):
    """Get aggregated sentiment analysis for a specific cryptocurrency"""
    try:
        # Get sentiment summary
        summary_df = sentiment_processor.get_crypto_sentiment_summary(hours=hours)
        
        if summary_df.empty:
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {crypto_symbol}")
        
        # Filter for specific crypto
        crypto_data = summary_df[summary_df['crypto_symbol'] == crypto_symbol.upper()]
        
        if crypto_data.empty:
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {crypto_symbol}")
        
        crypto_row = crypto_data.iloc[0]
        
        return {
            "crypto_symbol": crypto_symbol.upper(),
            "avg_sentiment_score": float(crypto_row['avg_sentiment']),
            "sentiment_signal": crypto_row['sentiment_signal'],
            "total_posts_analyzed": int(crypto_row['post_count']),
            "bullish_posts": int(crypto_row['bullish_count']),
            "bearish_posts": int(crypto_row['bearish_count']),
            "neutral_posts": int(crypto_row['neutral_count']),
            "sentiment_volatility": float(crypto_row['sentiment_volatility']),
            "time_period_hours": hours,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting crypto sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sentiment data: {str(e)}")
    
@app.get("/sentiment/market-overview", tags=["Sentiment Analysis"])
async def get_market_sentiment_overview(hours: int = 24):
    """Get market-wide sentiment overview for all tracked cryptocurrencies"""
    try:
        summary_df = sentiment_processor.get_crypto_sentiment_summary(hours=hours)
        
        if summary_df.empty:
            return {
                "message": "No sentiment data available",
                "total_cryptos": 0,
                "analysis_period_hours": hours
            }
        
        # Calculate market metrics
        market_overview = {
            "analysis_period_hours": hours,
            "total_cryptos_analyzed": len(summary_df),
            "total_posts_processed": int(summary_df['post_count'].sum()),
            "avg_market_sentiment": float(summary_df['avg_sentiment'].mean()),
            "market_sentiment_label": "bullish" if summary_df['avg_sentiment'].mean() > 60 else "bearish" if summary_df['avg_sentiment'].mean() < 40 else "neutral",
            "most_bullish_crypto": {
                "symbol": summary_df.loc[summary_df['avg_sentiment'].idxmax(), 'crypto_symbol'],
                "sentiment_score": float(summary_df['avg_sentiment'].max())
            },
            "most_bearish_crypto": {
                "symbol": summary_df.loc[summary_df['avg_sentiment'].idxmin(), 'crypto_symbol'],
                "sentiment_score": float(summary_df['avg_sentiment'].min())
            },
            "crypto_sentiment_distribution": {
                "bullish_cryptos": int((summary_df['avg_sentiment'] >= 60).sum()),
                "neutral_cryptos": int(((summary_df['avg_sentiment'] < 60) & (summary_df['avg_sentiment'] > 40)).sum()),
                "bearish_cryptos": int((summary_df['avg_sentiment'] <= 40).sum())
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return market_overview
        
    except Exception as e:
        logger.error(f"Error getting market sentiment overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve market overview: {str(e)}")
    
    
@app.get("/predictions/crypto/{crypto_symbol}", response_model=PredictionResponse, tags=["Price Predictions"])
async def get_crypto_predictions(crypto_symbol: str):
    """Get AI-powered price predictions for a specific cryptocurrency"""
    try:
        # Ensure model is trained
        if not prediction_service.ensure_model_trained():
            raise HTTPException(status_code=503, detail="Prediction model is not ready")
        
        # Get recent predictions from database
        latest_predictions = prediction_service.get_latest_predictions(hours=1)
        
        if crypto_symbol.upper() not in latest_predictions:
            raise HTTPException(
                status_code=404, 
                detail=f"No recent predictions available for {crypto_symbol}. Predictions are generated every 15 minutes."
            )
        
        pred_data = latest_predictions[crypto_symbol.upper()]
        
        # Extract prediction information
        predictions = pred_data.get('predictions', [])
        if not predictions:
            raise HTTPException(status_code=404, detail="No prediction data available")
        
        # Get current price for context
        db = SessionLocal()
        try:
            current_price_query = db.query(CryptoPrices).filter(
                CryptoPrices.symbol == crypto_symbol.upper()
            ).order_by(CryptoPrices.timestamp.desc()).first()
            
            current_price = float(current_price_query.price) if current_price_query else 0
        finally:
            db.close()
        
        # Calculate price changes
        predicted_prices = [p['predicted_price'] for p in predictions]
        price_changes = [((price - current_price) / current_price * 100) for price in predicted_prices]
        horizons = [f"{p['horizon_hours']}h" for p in predictions]
        
        return PredictionResponse(
            crypto_symbol=crypto_symbol.upper(),
            current_price=current_price,
            predicted_prices=predicted_prices,
            price_changes_pct=price_changes,
            confidence=pred_data['confidence'],
            prediction_horizons=horizons,
            timestamp=pred_data['created_at']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting predictions for {crypto_symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve predictions: {str(e)}")
    
    
@app.get("/predictions/market-signals", tags=["Price Predictions"])
async def get_market_trading_signals(min_confidence: float = 0.7, min_change: float = 2.0):
    """Get AI-generated trading signals based on price predictions"""
    try:
        # Get latest predictions
        latest_predictions = prediction_service.get_latest_predictions(hours=1)
        
        if not latest_predictions:
            return {
                "message": "No recent predictions available",
                "signals": [],
                "generation_time": datetime.utcnow().isoformat()
            }
        
        # Generate trading signals
        trading_signals = []
        
        for symbol, pred_data in latest_predictions.items():
            confidence = pred_data.get('confidence', 0)
            predictions = pred_data.get('predictions', [])
            
            if confidence >= min_confidence and predictions:
                # Get 2-hour prediction (if available)
                two_hour_pred = next((p for p in predictions if p['horizon_hours'] == 2), None)
                
                if two_hour_pred:
                    # Get current price
                    db = SessionLocal()
                    try:
                        current_price_query = db.query(CryptoPrices).filter(
                            CryptoPrices.symbol == symbol
                        ).order_by(CryptoPrices.timestamp.desc()).first()
                        
                        if current_price_query:
                            current_price = float(current_price_query.price)
                            predicted_price = two_hour_pred['predicted_price']
                            change_pct = ((predicted_price - current_price) / current_price * 100)
                            
                            if abs(change_pct) >= min_change:
                                signal_type = "BUY" if change_pct > 0 else "SELL"
                                strength = "STRONG" if abs(change_pct) > 5 and confidence > 0.8 else "MODERATE"
                                
                                trading_signals.append({
                                    "crypto_symbol": symbol,
                                    "signal": signal_type,
                                    "strength": strength,
                                    "predicted_change_pct": round(change_pct, 2),
                                    "confidence": round(confidence, 3),
                                    "current_price": current_price,
                                    "predicted_price": predicted_price,
                                    "time_horizon": "2 hours",
                                    "generated_at": pred_data['created_at']
                                })
                    finally:
                        db.close()
        
        # Sort by predicted change magnitude
        trading_signals.sort(key=lambda x: abs(x['predicted_change_pct']), reverse=True)
        
        return {
            "total_signals": len(trading_signals),
            "filters_applied": {
                "min_confidence": min_confidence,
                "min_change_pct": min_change
            },
            "signals": trading_signals,
            "generation_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate trading signals: {str(e)}")


@app.get("/market/summary", response_model=MarketSummaryResponse, tags=["Market Data"])
async def get_market_summary():
    """Get comprehensive market intelligence summary"""
    try:
        db = SessionLocal()
        
        try:
            # Get database counts
            total_posts = db.query(SocialMediaPost).count()
            total_prices = db.query(CryptoPrices).count()
            total_sentiment = db.query(SentimentAnalysis).count()
            total_predictions = db.query(MarketPredictions).count()
            
            # Get unique cryptos
            unique_cryptos = db.query(CryptoPrices.symbol).distinct().count()
            
            # Get sentiment overview
            sentiment_summary = sentiment_processor.get_crypto_sentiment_summary(hours=24)
            
            if not sentiment_summary.empty:
                avg_sentiment = float(sentiment_summary['avg_sentiment'].mean())
                top_bullish = sentiment_summary.loc[sentiment_summary['avg_sentiment'].idxmax(), 'crypto_symbol']
                top_bearish = sentiment_summary.loc[sentiment_summary['avg_sentiment'].idxmin(), 'crypto_symbol']
            else:
                avg_sentiment = 50.0
                top_bullish = "N/A"
                top_bearish = "N/A"
            
            return MarketSummaryResponse(
                total_cryptos_tracked=unique_cryptos,
                total_social_posts=total_posts,
                total_sentiment_analyses=total_sentiment,
                total_predictions=total_predictions,
                avg_market_sentiment=avg_sentiment,
                top_bullish_crypto=top_bullish,
                top_bearish_crypto=top_bearish,
                last_updated=datetime.utcnow().isoformat()
            )
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve market summary: {str(e)}")
    
    
@app.get("/market/prices/latest", tags=["Market Data"])
async def get_latest_prices(limit: int = 10):
    """Get latest cryptocurrency prices from our database"""
    try:
        db = SessionLocal()
        
        try:
            # Get latest prices for each crypto
            latest_prices_query = """
            SELECT DISTINCT ON (symbol) symbol, price, price_change_24h, volume_24h, 
                market_cap, timestamp
            FROM crypto_prices 
            ORDER BY symbol, timestamp DESC
            LIMIT :limit
            """
            
            result = db.execute(latest_prices_query, {'limit': limit})
            
            prices = []
            for row in result:
                prices.append({
                    "symbol": row[0],
                    "price": float(row[1]),
                    "change_24h_pct": float(row[2]) if row[2] else 0,
                    "volume_24h": float(row[3]) if row[3] else 0,
                    "market_cap": float(row[4]) if row[4] else 0,
                    "last_updated": row[5].isoformat() if row[5] else None
                })
            
            return {
                "prices": prices,
                "total_cryptos": len(prices),
                "data_source": "real_time_pipeline",
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting latest prices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve latest prices: {str(e)}")

@app.get("/analytics/performance", tags=["Analytics"])
async def get_system_performance():
    """Get system performance analytics for the past 24 hours"""
    try:
        db = SessionLocal()
        
        try:
            # Calculate 24-hour metrics
            day_ago = datetime.utcnow() - timedelta(hours=24)
            
            posts_24h = db.query(SocialMediaPost).filter(
                SocialMediaPost.created_at >= day_ago
            ).count()
            
            prices_24h = db.query(CryptoPrices).filter(
                CryptoPrices.timestamp >= day_ago
            ).count()
            
            sentiment_24h = db.query(SentimentAnalysis).filter(
                SentimentAnalysis.processed_at >= day_ago
            ).count()
            
            predictions_24h = db.query(MarketPredictions).filter(
                MarketPredictions.created_at >= day_ago
            ).count()
            
            # Calculate processing rates
            processing_rates = {
                "social_posts_per_hour": round(posts_24h / 24, 1),
                "price_updates_per_hour": round(prices_24h / 24, 1),
                "sentiment_analyses_per_hour": round(sentiment_24h / 24, 1),
                "predictions_per_hour": round(predictions_24h / 24, 1)
            }
            
            # Calculate system efficiency
            sentiment_coverage = (sentiment_24h / posts_24h * 100) if posts_24h > 0 else 0
            
            performance_metrics = {
                "time_period": "24 hours",
                "data_processing": {
                    "social_media_posts": posts_24h,
                    "crypto_price_updates": prices_24h,
                    "sentiment_analyses": sentiment_24h,
                    "ai_predictions": predictions_24h
                },
                "processing_rates": processing_rates,
                "system_efficiency": {
                    "sentiment_coverage_pct": round(sentiment_coverage, 1),
                    "uptime_pct": 99.5,  # Would be calculated from actual monitoring
                    "api_response_time_ms": 150  # Would be from actual monitoring
                },
                "business_metrics": {
                    "trading_signals_generated": predictions_24h // 2,  # Approximation
                    "market_insights_delivered": sentiment_24h,
                    "automated_analyses": posts_24h + prices_24h
                },
                "calculated_at": datetime.utcnow().isoformat()
            }
            
            return performance_metrics
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance data: {str(e)}")



@app.post("/admin/retrain-models", tags=["Administration"])
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """Trigger background model retraining (admin endpoint)"""
    def retrain_models():
        try:
            logger.info("Starting background model retraining...")
            
            # Retrain sentiment model
            sentiment_processor.run_sentiment_analysis_pipeline(batch_limit=100)
            
            # Retrain price prediction model
            prediction_trainer.train_prediction_model(days_back=10, epochs=25)
            
            logger.info("Background model retraining completed")
            
        except Exception as e:
            logger.error(f"Background retraining failed: {e}")

    background_tasks.add_task(retrain_models)

    return {
        "message": "Model retraining initiated in background",
        "status": "accepted",
        "estimated_duration": "15-20 minutes",
        "initiated_at": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting BCG X Market Intelligence API Server...")
    print("📊 Enterprise AI-powered cryptocurrency analysis platform")
    print("🌐 API Documentation: http://localhost:8000/docs")
    
uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)