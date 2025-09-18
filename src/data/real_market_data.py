import requests
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import pandas as pd

logger = logging.getLogger(__name__)

class RealMarketDataService:
    """Service to fetch real cryptocurrency and market data"""
    
    def __init__(self):
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.crypto_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'ADA': 'cardano',
            'SOL': 'solana',
            'MATIC': 'polygon',
            'LINK': 'chainlink',
            'DOT': 'polkadot',
            'AVAX': 'avalanche-2'
        }
        self.last_request_time = 0
        self.min_request_interval = 12  # 12 seconds between requests (API friendly)
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def _rate_limit_check(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.info(f"⏱️ Rate limiting: waiting {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_crypto_prices(self) -> Optional[List[Dict]]:
        """Get real crypto prices from CoinGecko API"""
        try:
            # Check cache first
            cache_key = "crypto_prices"
            current_time = time.time()
            
            if (cache_key in self.cache and 
                current_time - self.cache[cache_key]['timestamp'] < self.cache_duration):
                logger.info("📋 Using cached crypto prices")
                return self.cache[cache_key]['data']
            
            self._rate_limit_check()
            
            # Get coin IDs for our symbols
            coin_ids = ','.join(self.crypto_mapping.values())
            
            url = f"{self.coingecko_base}/simple/price"
            params = {
                'ids': coin_ids,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            logger.info("🌐 Fetching REAL crypto prices from CoinGecko...")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to our format
            formatted_data = []
            for symbol, coin_id in self.crypto_mapping.items():
                if coin_id in data:
                    coin_data = data[coin_id]
                    formatted_data.append({
                        'symbol': symbol,
                        'price': float(coin_data.get('usd', 0)),
                        'market_cap': float(coin_data.get('usd_market_cap', 0)),
                        'volume_24h': float(coin_data.get('usd_24h_vol', 0)),
                        'price_change_24h': float(coin_data.get('usd_24h_change', 0)),
                        'timestamp': datetime.utcnow().isoformat(),
                        'data_source': 'coingecko_real',
                        'last_updated': coin_data.get('last_updated_at', int(current_time))
                    })
            
            # Cache the results
            self.cache[cache_key] = {
                'data': formatted_data,
                'timestamp': current_time
            }
            
            logger.info(f"✅ Retrieved REAL prices for {len(formatted_data)} cryptocurrencies")
            return formatted_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching real crypto prices: {e}")
            return None
    
    def get_market_sentiment_indicators(self) -> Dict:
        """Get market sentiment indicators from various sources"""
        try:
            self._rate_limit_check()
            
            # Get Fear & Greed Index (if available)
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            sentiment_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'fear_greed_index': 50,  # Default neutral
                'market_trend': 'neutral'
            }
            
            if response.status_code == 200:
                fng_data = response.json()
                if 'data' in fng_data and len(fng_data['data']) > 0:
                    fng_value = int(fng_data['data'][0]['value'])
                    sentiment_data['fear_greed_index'] = fng_value
                    
                    if fng_value >= 75:
                        sentiment_data['market_trend'] = 'extreme_greed'
                    elif fng_value >= 55:
                        sentiment_data['market_trend'] = 'greed'
                    elif fng_value >= 45:
                        sentiment_data['market_trend'] = 'neutral'
                    elif fng_value >= 25:
                        sentiment_data['market_trend'] = 'fear'
                    else:
                        sentiment_data['market_trend'] = 'extreme_fear'
            
            logger.info(f"📊 Market sentiment: {sentiment_data['market_trend']} (F&G: {sentiment_data['fear_greed_index']})")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching market sentiment: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'fear_greed_index': 50,
                'market_trend': 'neutral'
            }
    
    def get_stock_market_data(self) -> Optional[Dict]:
        """Get real stock market data for correlation"""
        try:
            # Get major indices
            symbols = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
            
            stock_data = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="2d")
                    
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        prev_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        
                        stock_data[symbol.replace('^', '')] = {
                            'price': current_price,
                            'change_24h': change_pct,
                            'volume': float(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
                        }
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get data for {symbol}: {e}")
                    continue
            
            if stock_data:
                logger.info(f"📈 Retrieved stock market data for {len(stock_data)} indices")
                return {
                    'timestamp': datetime.utcnow().isoformat(),
                    'indices': stock_data,
                    'data_source': 'yahoo_finance'
                }
            
        except Exception as e:
            logger.error(f"❌ Error fetching stock market data: {e}")
        
        return None

# Global instance
real_market_service = RealMarketDataService()