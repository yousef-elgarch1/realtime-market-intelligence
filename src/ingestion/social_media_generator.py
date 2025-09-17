import random
import json
from datetime import datetime, timedelta
from typing import List, Dict
import uuid
import logging

logger = logging.getLogger(__name__)

class RealSocialMediaGenerator:
    """Generate realistic social media posts based on REAL market data"""
    
    def __init__(self):
        self.crypto_symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'MATIC', 'LINK', 'DOT', 'AVAX']
        
        # User profiles with realistic follower counts and influence
        self.user_profiles = [
            {"username": "crypto_whale", "followers": 150000, "influence": 0.95},
            {"username": "blockchain_expert", "followers": 89000, "influence": 0.90},
            {"username": "defi_analyst", "followers": 45000, "influence": 0.85},
            {"username": "trading_guru", "followers": 32000, "influence": 0.80},
            {"username": "crypto_researcher", "followers": 28000, "influence": 0.75},
            {"username": "hodl_master", "followers": 15000, "influence": 0.70},
            {"username": "altcoin_daily", "followers": 12000, "influence": 0.65},
            {"username": "market_analyst", "followers": 8500, "influence": 0.60},
            {"username": "crypto_trader", "followers": 5200, "influence": 0.50},
            {"username": "diamond_hands", "followers": 3100, "influence": 0.40},
        ]
        
        # News and event templates
        self.news_templates = [
            "BREAKING: {symbol} institutional adoption news!",
            "Major {symbol} partnership announced with Fortune 500 company",
            "{symbol} technical analysis: Key resistance at ${price:.0f}",
            "Whale alert: Large {symbol} transaction detected",
            "{symbol} network upgrade successful, prices react",
            "SEC news impacts {symbol} trading sentiment",
            "DeFi protocol integrates {symbol}, bullish signal",
            "{symbol} Layer 2 solution gains traction"
        ]
    
    def generate_price_reactive_post(self, symbol: str, current_price: float, 
                                   price_change_24h: float, volume_24h: float = 0,
                                   market_sentiment: str = "neutral") -> Dict:
        """Generate social media post that reacts to REAL price movements"""
        
        # Determine post sentiment based on real price action
        abs_change = abs(price_change_24h)
        
        if price_change_24h > 8:
            sentiment_type = 'extremely_bullish'
            intensity = 'extreme'
        elif price_change_24h > 3:
            sentiment_type = 'bullish'
            intensity = 'high'
        elif price_change_24h > 1:
            sentiment_type = 'slightly_bullish'
            intensity = 'medium'
        elif price_change_24h < -8:
            sentiment_type = 'extremely_bearish'
            intensity = 'extreme'
        elif price_change_24h < -3:
            sentiment_type = 'bearish'
            intensity = 'high'
        elif price_change_24h < -1:
            sentiment_type = 'slightly_bearish'
            intensity = 'medium'
        else:
            sentiment_type = 'neutral'
            intensity = 'low'
        
        # Generate content based on sentiment and real data
        content = self._generate_content_by_sentiment(
            symbol, current_price, price_change_24h, sentiment_type, market_sentiment
        )
        
        # Select user profile based on post importance
        if intensity == 'extreme':
            user = random.choice([u for u in self.user_profiles if u['influence'] > 0.80])
        elif intensity == 'high':
            user = random.choice([u for u in self.user_profiles if u['influence'] > 0.60])
        else:
            user = random.choice(self.user_profiles)
        
        # Calculate engagement based on price movement and user influence
        base_engagement = int(user['followers'] * user['influence'] * random.uniform(0.005, 0.03))
        
        # Boost engagement for significant price movements
        movement_multiplier = 1 + (abs_change / 10)  # 10% change = 2x engagement
        if volume_24h > 1000000000:  # High volume boost
            movement_multiplier *= 1.5
        
        final_engagement = int(base_engagement * movement_multiplier)
        
        post = {
            "post_id": f"real_{uuid.uuid4().hex[:10]}",
            "platform": random.choice(["twitter", "reddit", "telegram", "discord"]),
            "content": content,
            "author": user['username'],
            "crypto_symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "follower_count": user['followers'],
            "retweet_count": max(1, int(final_engagement * random.uniform(0.08, 0.25))),
            "like_count": max(1, int(final_engagement * random.uniform(0.4, 1.5))),
            "reply_count": max(0, int(final_engagement * random.uniform(0.05, 0.15))),
            "sentiment_type": sentiment_type,
            "influence_score": user['influence'],
            "real_price_data": {
                "current_price": current_price,
                "price_change_24h": price_change_24h,
                "volume_24h": volume_24h
            },
            "market_sentiment": market_sentiment,
            "engagement_score": final_engagement,
            "data_source": "real_market_reactive"
        }
        
        return post
    
    def _generate_content_by_sentiment(self, symbol: str, price: float, change: float, 
                                     sentiment: str, market_sentiment: str) -> str:
        """Generate realistic content based on sentiment and real market data"""
        
        templates = {
            'extremely_bullish': [
                f"🚀🚀 {symbol} ABSOLUTELY MOONING! +{change:.1f}% in 24h! WHO'S STILL SLEEPING ON THIS?! 💎🙌",
                f"HOLY MOLLY! {symbol} just broke ${price:,.0f}! Up {change:.1f}%! This is INSANE! 🔥🔥🔥",
                f"I TOLD YOU ALL! {symbol} +{change:.1f}%! Next stop: ANDROMEDA! 🌌🚀 #crypto #moon",
                f"BREAKING: {symbol} surges {change:.1f}%! This pump is UNREAL! Where are the bears now? 🐂💪",
                f"🚨 WHALE ALERT 🚨 {symbol} up {change:.1f}%! Smart money is moving! Follow the whales! 🐋"
            ],
            
            'bullish': [
                f"{symbol} looking STRONG! 💪 Up {change:.1f}% today. Resistance at ${price:,.0f} broken! 📈",
                f"Green candles everywhere! {symbol} +{change:.1f}% in 24h. Bulls are in control! 🐂",
                f"Beautiful breakout for {symbol}! +{change:.1f}% and climbing. Target: ${price*1.2:,.0f} 🎯",
                f"Love to see it! {symbol} pumping {change:.1f}%! DCA paying off 💚 #HODL",
                f"Technical analysis was right! {symbol} broke resistance, now +{change:.1f}%. More upside coming! 📊"
            ],
            
            'slightly_bullish': [
                f"{symbol} steady gains today +{change:.1f}%. Slow and steady wins the race 🐢💚",
                f"Nice little pump for {symbol}! +{change:.1f}% in 24h. Building momentum 📈",
                f"{symbol} holding above ${price:,.0f} nicely. +{change:.1f}% today. Bullish structure intact 📊",
                f"Small wins count! {symbol} up {change:.1f}%. Patience is key in this market 🧘‍♂️",
                f"Consolidation paying off for {symbol}. +{change:.1f}% move. Next leg up soon? 🤔"
            ],
            
            'extremely_bearish': [
                f"💀 {symbol} ABSOLUTELY REKT! Down {abs(change):.1f}%! This is BRUTAL! 😱🩸",
                f"RIP {symbol} holders... -{abs(change):.1f}% bloodbath today. Support DECIMATED! 📉💔",
                f"PANIC MODE: {symbol} crashing {abs(change):.1f}%! Who's buying this knife? 🔪😰",
                f"MASSACRE! {symbol} down {abs(change):.1f}%! Bears feasting today! 🐻💀",
                f"🚨 DUMP ALERT 🚨 {symbol} -{abs(change):.1f}%! Stop losses triggered everywhere! 💥"
            ],
            
            'bearish': [
                f"Ouch! {symbol} taking a hit today -{abs(change):.1f}%. Support at ${price:,.0f} broken 📉",
                f"Red day for {symbol}. Down {abs(change):.1f}% in 24h. Where's the bottom? 😬",
                f"{symbol} sellers stepping in. -{abs(change):.1f}% drop. Time to buy the dip? 🤷‍♂️",
                f"Not looking good for {symbol}. Down {abs(change):.1f}%. Bears gaining control 🐻",
                f"Bearish momentum for {symbol}. -{abs(change):.1f}% today. Support levels failing 📊"
            ],
            
            'slightly_bearish': [
                f"{symbol} pulling back slightly -{abs(change):.1f}%. Healthy correction or start of dump? 🤔",
                f"Minor red for {symbol} today -{abs(change):.1f}%. Nothing to panic about... yet 😅",
                f"{symbol} down {abs(change):.1f}%. Profit taking or trend reversal? Watching closely 👀",
                f"Small dip for {symbol} -{abs(change):.1f}%. Good entry point for DCA? 💭",
                f"{symbol} cooling off a bit -{abs(change):.1f}%. Consolidation before next move? 📈"
            ],
            
            'neutral': [
                f"{symbol} trading sideways around ${price:,.0f}. Only {change:+.1f}% today. Calm before storm? ⛈️",
                f"Boring day for {symbol}. {change:+.1f}% movement. Accumulation phase? 🤷‍♂️",
                f"{symbol} consolidating at ${price:,.0f}. {change:+.1f}% in 24h. Waiting for direction 📊",
                f"Not much happening with {symbol} today {change:+.1f}%. Perfect for accumulation? 💰",
                f"{symbol} stable around ${price:,.0f}. {change:+.1f}% change. Bulls vs bears stalemate 🤝"
            ]
        }
        
        # Add market context if extreme conditions
        content = random.choice(templates.get(sentiment, templates['neutral']))
        
        if market_sentiment == 'extreme_fear' and sentiment in ['bearish', 'extremely_bearish']:
            content += " Market fear is REAL! 😨"
        elif market_sentiment == 'extreme_greed' and sentiment in ['bullish', 'extremely_bullish']:
            content += " Market greed at peak! 🤑"
        
        return content
    
    def generate_news_based_post(self, symbol: str, current_price: float) -> Dict:
        """Generate news-based social media post"""
        
        user = random.choice([u for u in self.user_profiles if u['influence'] > 0.70])
        template = random.choice(self.news_templates)
        content = template.format(symbol=symbol, price=current_price)
        
        base_engagement = int(user['followers'] * user['influence'] * random.uniform(0.01, 0.04))
        
        post = {
            "post_id": f"news_{uuid.uuid4().hex[:10]}",
            "platform": random.choice(["twitter", "reddit", "linkedin"]),
            "content": content,
            "author": user['username'],
            "crypto_symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "follower_count": user['followers'],
            "retweet_count": max(1, int(base_engagement * random.uniform(0.1, 0.3))),
            "like_count": max(1, int(base_engagement * random.uniform(0.5, 1.2))),
            "reply_count": max(0, int(base_engagement * random.uniform(0.08, 0.2))),
            "sentiment_type": "news_based",
            "influence_score": user['influence'],
            "data_source": "news_reactive"
        }
        
        return post
    
    def generate_batch_from_real_data(self, crypto_prices: List[Dict], 
                                    market_sentiment: str = "neutral", 
                                    count: int = 5) -> List[Dict]:
        """Generate a batch of posts based on real market data"""
        
        if not crypto_prices:
            logger.warning("No real crypto prices provided, cannot generate posts")
            return []
        
        posts = []
        
        for _ in range(count):
            # Select random crypto from real data
            crypto_data = random.choice(crypto_prices)
            
            # 80% price-reactive posts, 20% news-based posts
            if random.random() < 0.8:
                post = self.generate_price_reactive_post(
                    crypto_data['symbol'],
                    crypto_data['price'],
                    crypto_data['price_change_24h'],
                    crypto_data.get('volume_24h', 0),
                    market_sentiment
                )
            else:
                post = self.generate_news_based_post(
                    crypto_data['symbol'],
                    crypto_data['price']
                )
            
            posts.append(post)
        
        return posts