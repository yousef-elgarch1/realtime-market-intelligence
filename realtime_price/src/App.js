import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, CandlestickChart,
  ComposedChart
} from 'recharts';
import { 
  TrendingUp, TrendingDown, Activity, DollarSign, BarChart3, 
  Brain, AlertTriangle, RefreshCw, Target, Zap, Eye, Users,
  MessageSquare, Award, Shield, Globe, ArrowUpRight, ArrowDownRight
} from 'lucide-react';
import { 
  SiBitcoin, SiEthereum, SiLitecoin, SiCardano, SiSolana,
  SiPolygon, SiChainlink, SiPolkadot
} from 'react-icons/si';
import { FaCoins } from 'react-icons/fa';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

// Cryptocurrency icon mapping
const CryptoIcons = {
  BTC: SiBitcoin,
  ETH: SiEthereum,
  ADA: SiCardano,
  SOL: SiSolana,
  MATIC: SiPolygon,
  LINK: SiChainlink,
  DOT: SiPolkadot,
  AVAX: FaCoins,
  LTC: SiLitecoin
};

function App() {
  const [marketData, setMarketData] = useState(null);
  const [prices, setPrices] = useState([]);
  const [sentiment, setSentiment] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedCrypto, setSelectedCrypto] = useState('BTC');
  const [realTimeData, setRealTimeData] = useState([]);
  const [candlestickData, setCandlestickData] = useState([]);
  const [tradingSignals, setTradingSignals] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchAllData = async () => {
    try {
      const [marketRes, pricesRes, sentimentRes, perfRes] = await Promise.all([
        axios.get(`${API_BASE}/market/summary`),
        axios.get(`${API_BASE}/market/prices/latest?limit=15`),
        axios.get(`${API_BASE}/sentiment/market-overview`),
        axios.get(`${API_BASE}/analytics/performance`)
      ]);

      setMarketData(marketRes.data);
      setPrices(pricesRes.data.prices);
      setSentiment(sentimentRes.data);
      setPerformance(perfRes.data);

      generateRealisticChartData();
      generateTradingSignals();
      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  const generateRealisticChartData = () => {
    const data = [];
    const candleData = [];
    let basePrice = 45000;
    
    for (let i = 23; i >= 0; i--) {
      const change = (Math.random() - 0.5) * 0.05;
      const volatility = Math.random() * 0.02;
      
      const open = basePrice;
      const close = basePrice * (1 + change);
      const high = Math.max(open, close) * (1 + volatility);
      const low = Math.min(open, close) * (1 - volatility);
      
      basePrice = close;
      
      const time = new Date();
      time.setHours(time.getHours() - i);
      
      data.push({
        time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        price: Math.round(close),
        volume: Math.random() * 2000000 + 500000,
        sentiment: Math.random() * 40 + 30 + (change > 0 ? 20 : -10),
        rsi: Math.random() * 40 + 30,
        macd: Math.random() * 200 - 100
      });

      candleData.push({
        time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        open: Math.round(open),
        high: Math.round(high),
        low: Math.round(low),
        close: Math.round(close),
        volume: Math.random() * 2000000 + 500000
      });
    }
    
    setRealTimeData(data);
    setCandlestickData(candleData);
  };

  const generateTradingSignals = () => {
    const signals = [
      { crypto: 'BTC', signal: 'STRONG BUY', confidence: 87, prediction: '+4.2%', timeframe: '2h', type: 'buy', strength: 'strong' },
      { crypto: 'ETH', signal: 'BUY', confidence: 74, prediction: '+2.8%', timeframe: '4h', type: 'buy', strength: 'medium' },
      { crypto: 'SOL', signal: 'HOLD', confidence: 68, prediction: '+0.5%', timeframe: '6h', type: 'hold', strength: 'weak' },
      { crypto: 'ADA', signal: 'SELL', confidence: 82, prediction: '-3.1%', timeframe: '1h', type: 'sell', strength: 'strong' },
      { crypto: 'MATIC', signal: 'STRONG BUY', confidence: 91, prediction: '+5.7%', timeframe: '3h', type: 'buy', strength: 'strong' },
      { crypto: 'LINK', signal: 'BUY', confidence: 76, prediction: '+1.9%', timeframe: '5h', type: 'buy', strength: 'medium' }
    ];
    setTradingSignals(signals);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center text-white">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-xl font-medium">Loading Market Intelligence Platform...</p>
          <p className="text-gray-400 mt-2">Connecting to real-time data feeds</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Enhanced Header */}
      <header className="bg-black/20 backdrop-blur-lg border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-8xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                <BarChart3 className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                  CryptoTrader Pro
                </h1>
                <p className="text-gray-400 text-sm">AI-Powered Market Intelligence Platform</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2 bg-green-500/20 px-3 py-2 rounded-full">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium">Live Data</span>
              </div>
              
              <div className="flex items-center space-x-4 text-sm">
                <div className="text-center">
                  <div className="text-green-400 font-bold">{marketData?.total_cryptos_tracked || '0'}</div>
                  <div className="text-gray-400">Assets</div>
                </div>
                <div className="text-center">
                  <div className="text-blue-400 font-bold">{(marketData?.total_social_posts + marketData?.total_crypto_prices || 0).toLocaleString()}</div>
                  <div className="text-gray-400">Data Points</div>
                </div>
              </div>
              
              <button 
                onClick={fetchAllData}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 px-6 py-3 rounded-xl flex items-center space-x-2 font-medium transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Refresh</span>
              </button>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex space-x-8 mt-6">
            {[
              { id: 'overview', label: 'Market Overview', icon: Globe },
              { id: 'trading', label: 'Trading Signals', icon: Target },
              { id: 'sentiment', label: 'AI Sentiment', icon: Brain },
              { id: 'analytics', label: 'Analytics', icon: BarChart3 }
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    activeTab === tab.id 
                      ? 'bg-white/10 text-white' 
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </header>

      <div className="max-w-8xl mx-auto px-6 py-8">
        {activeTab === 'overview' && (
          <>
            {/* Enhanced Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              <MetricCard
                title="Portfolio Value"
                value="$2,847,392"
                icon={<DollarSign className="w-6 h-6" />}
                trend="+15.3%"
                trendUp={true}
                gradient="from-green-400 to-green-600"
              />
              <MetricCard
                title="AI Predictions"
                value={marketData?.total_sentiment_analyses?.toLocaleString() || '0'}
                icon={<Brain className="w-6 h-6" />}
                trend="+24.7%"
                trendUp={true}
                gradient="from-purple-400 to-purple-600"
              />
              <MetricCard
                title="Success Rate"
                value="87.3%"
                icon={<Award className="w-6 h-6" />}
                trend="+3.2%"
                trendUp={true}
                gradient="from-blue-400 to-blue-600"
              />
              <MetricCard
                title="Risk Score"
                value="Medium"
                icon={<Shield className="w-6 h-6" />}
                trend="Stable"
                trendUp={true}
                gradient="from-orange-400 to-orange-600"
              />
            </div>

            {/* Main Trading Dashboard */}
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-8 mb-8">
              {/* Advanced Price Chart */}
              <div className="xl:col-span-3 bg-black/20 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
                <div className="flex justify-between items-center mb-6">
                  <div className="flex items-center space-x-4">
                    <h2 className="text-2xl font-bold">Advanced Chart Analysis</h2>
                    <div className="flex items-center space-x-2">
                      {Object.keys(CryptoIcons).slice(0, 8).map(crypto => {
                        const Icon = CryptoIcons[crypto];
                        return (
                          <button
                            key={crypto}
                            onClick={() => setSelectedCrypto(crypto)}
                            className={`p-2 rounded-lg transition-all duration-200 ${
                              selectedCrypto === crypto 
                                ? 'bg-white/20 text-white' 
                                : 'text-gray-400 hover:text-white hover:bg-white/10'
                            }`}
                          >
                            <Icon className="w-5 h-5" />
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <select 
                      value={selectedCrypto}
                      onChange={(e) => setSelectedCrypto(e.target.value)}
                      className="bg-black/30 border border-white/20 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500"
                    >
                      {Object.keys(CryptoIcons).map(crypto => (
                        <option key={crypto} value={crypto} className="bg-slate-800">{crypto}</option>
                      ))}
                    </select>
                  </div>
                </div>
                
                <div className="h-96 mb-6">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={candlestickData}>
                      <defs>
                        <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="time" stroke="#666" fontSize={12} />
                      <YAxis stroke="#666" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'rgba(0,0,0,0.8)', 
                          border: '1px solid rgba(255,255,255,0.2)',
                          borderRadius: '12px',
                          backdropFilter: 'blur(10px)'
                        }} 
                      />
                      <Area 
                        type="monotone" 
                        dataKey="close" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        fillOpacity={1} 
                        fill="url(#priceGradient)" 
                      />
                      <Bar dataKey="volume" fill="rgba(59, 130, 246, 0.3)" yAxisId="volume" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>

                {/* Technical Indicators */}
                <div className="grid grid-cols-3 gap-4">
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">RSI (14)</div>
                    <div className="text-xl font-bold text-yellow-400">64.2</div>
                    <div className="text-xs text-gray-400">Neutral</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">MACD</div>
                    <div className="text-xl font-bold text-green-400">+127.3</div>
                    <div className="text-xs text-gray-400">Bullish</div>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-1">Volume</div>
                    <div className="text-xl font-bold text-blue-400">1.2M</div>
                    <div className="text-xs text-gray-400">Above Average</div>
                  </div>
                </div>
              </div>

              {/* Enhanced Live Prices */}
              <div className="bg-black/20 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
                <h2 className="text-xl font-bold mb-6 flex items-center">
                  <Activity className="w-5 h-5 mr-2" />
                  Live Market
                </h2>
                <div className="space-y-3">
                  {prices.slice(0, 8).map((crypto, index) => {
                    const Icon = CryptoIcons[crypto.symbol] || FaCoins;
                    const isPositive = crypto.change_24h_pct >= 0;
                    
                    return (
                      <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-xl hover:bg-white/10 transition-all duration-200 cursor-pointer">
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded-lg ${isPositive ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
                            <Icon className={`w-5 h-5 ${isPositive ? 'text-green-400' : 'text-red-400'}`} />
                          </div>
                          <div>
                            <div className="font-semibold">{crypto.symbol}</div>
                            <div className="text-2xl font-bold">${crypto.price?.toLocaleString()}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`flex items-center space-x-1 font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                            {isPositive ? 
                              <ArrowUpRight className="w-4 h-4" /> : 
                              <ArrowDownRight className="w-4 h-4" />
                            }
                            <span>{crypto.change_24h_pct >= 0 ? '+' : ''}{crypto.change_24h_pct?.toFixed(2)}%</span>
                          </div>
                          <div className="text-sm text-gray-400">
                            ${(crypto.volume_24h / 1000000)?.toFixed(0)}M Vol
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'trading' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* AI Trading Signals */}
            <div className="bg-black/20 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <h2 className="text-2xl font-bold mb-6 flex items-center">
                <Target className="w-6 h-6 mr-3" />
                AI Trading Signals
              </h2>
              <div className="space-y-4">
                {tradingSignals.map((signal, index) => {
                  const Icon = CryptoIcons[signal.crypto] || FaCoins;
                  return (
                    <TradingSignalCard key={index} signal={signal} Icon={Icon} />
                  );
                })}
              </div>
            </div>

            {/* Market Sentiment */}
            <div className="bg-black/20 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <h2 className="text-2xl font-bold mb-6 flex items-center">
                <Brain className="w-6 h-6 mr-3" />
                Sentiment Analysis
              </h2>
              <div className="h-64 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={realTimeData}>
                    <defs>
                      <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="time" stroke="#666" fontSize={12} />
                    <YAxis stroke="#666" fontSize={12} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(0,0,0,0.8)', 
                        border: '1px solid rgba(255,255,255,0.2)',
                        borderRadius: '12px'
                      }} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="sentiment" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      fillOpacity={1} 
                      fill="url(#sentimentGradient)" 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center p-4 bg-green-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-green-400">67%</div>
                  <div className="text-sm text-gray-400">Bullish</div>
                </div>
                <div className="text-center p-4 bg-gray-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-gray-400">21%</div>
                  <div className="text-sm text-gray-400">Neutral</div>
                </div>
                <div className="text-center p-4 bg-red-500/20 rounded-lg">
                  <div className="text-2xl font-bold text-red-400">12%</div>
                  <div className="text-sm text-gray-400">Bearish</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'sentiment' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 bg-black/20 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <h2 className="text-2xl font-bold mb-6">Market Sentiment Breakdown</h2>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Very Bullish', value: 35, color: '#10b981' },
                        { name: 'Bullish', value: 32, color: '#34d399' },
                        { name: 'Neutral', value: 21, color: '#6b7280' },
                        { name: 'Bearish', value: 8, color: '#f87171' },
                        { name: 'Very Bearish', value: 4, color: '#ef4444' }
                      ]}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {[
                        { name: 'Very Bullish', value: 35, color: '#10b981' },
                        { name: 'Bullish', value: 32, color: '#34d399' },
                        { name: 'Neutral', value: 21, color: '#6b7280' },
                        { name: 'Bearish', value: 8, color: '#f87171' },
                        { name: 'Very Bearish', value: 4, color: '#ef4444' }
                      ].map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            <div className="bg-black/20 backdrop-blur-lg rounded-2xl p-6 border border-white/10">
              <h2 className="text-xl font-bold mb-6">Social Mentions</h2>
              <div className="space-y-4">
                {Object.keys(CryptoIcons).slice(0, 6).map((crypto, index) => {
                  const Icon = CryptoIcons[crypto];
                  const mentions = Math.floor(Math.random() * 1000) + 100;
                  return (
                    <div key={crypto} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <Icon className="w-6 h-6 text-blue-400" />
                        <span className="font-medium">{crypto}</span>
                      </div>
                      <div className="text-right">
                        <div className="font-bold">{mentions.toLocaleString()}</div>
                        <div className="text-xs text-gray-400">mentions</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Enhanced Components
const MetricCard = ({ title, value, icon, trend, trendUp, gradient }) => (
  <div className="bg-black/20 backdrop-blur-lg rounded-2xl p-6 border border-white/10 hover:bg-black/30 transition-all duration-300">
    <div className="flex justify-between items-start mb-4">
      <div className={`p-3 rounded-xl bg-gradient-to-r ${gradient}`}>
        {icon}
      </div>
      <div className={`flex items-center space-x-1 text-sm font-medium ${trendUp ? 'text-green-400' : 'text-red-400'}`}>
        {trendUp ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
        <span>{trend}</span>
      </div>
    </div>
    <div className="text-3xl font-bold mb-2 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
      {value}
    </div>
    <div className="text-gray-400 text-sm">{title}</div>
  </div>
);

const TradingSignalCard = ({ signal, Icon }) => {
  const getSignalStyles = () => {
    switch(signal.type) {
      case 'buy':
        return signal.strength === 'strong' 
          ? 'bg-green-500/20 text-green-400 border-green-400/30' 
          : 'bg-green-500/10 text-green-400 border-green-400/20';
      case 'sell':
        return signal.strength === 'strong'
          ? 'bg-red-500/20 text-red-400 border-red-400/30'
          : 'bg-red-500/10 text-red-400 border-red-400/20';
      default:
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-400/30';
    }
  };

  return (
    <div className="bg-white/5 rounded-xl p-4 hover:bg-white/10 transition-all duration-200">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <Icon className="w-8 h-8 text-blue-400" />
          <div>
            <div className="font-bold text-lg">{signal.crypto}</div>
            <div className={`inline-block px-3 py-1 rounded-full text-sm font-bold border ${getSignalStyles()}`}>
              {signal.signal}
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold">{signal.prediction}</div>
          <div className="text-sm text-gray-400">{signal.timeframe}</div>
        </div>
      </div>
      <div className="flex justify-between items-center">
        <div className="text-sm text-gray-400">
          Confidence: <span className="text-white font-bold">{signal.confidence}%</span>
        </div>
        <div className="w-20 bg-gray-700 rounded-full h-2">
          <div 
            className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${signal.confidence}%` }}
          ></div>
          </div>
      </div>
    </div>
  );
};

export default App;