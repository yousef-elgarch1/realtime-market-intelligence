#!/usr/bin/env python3
"""
Comprehensive API testing for BCG X Market Intelligence Platform
"""
import requests
import json
import time
from datetime import datetime

def test_production_api():
    """Test all API endpoints comprehensively"""
    
    print("🧪 BCG X MARKET INTELLIGENCE API TEST SUITE")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test results storage
    test_results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "test_details": []
    }
    
    def run_test(test_name: str, endpoint: str, method: str = "GET", data: dict = None):
        """Run individual API test"""
        test_results["total_tests"] += 1
        
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
            elif method == "POST":
                response = requests.post(f"{base_url}{endpoint}", json=data, timeout=10)
            
            if response.status_code == 200:
                test_results["passed_tests"] += 1
                print(f"   ✅ {test_name}: PASSED ({response.status_code})")
                
                # Show sample response for important endpoints
                if endpoint in ["/market/summary", "/predictions/market-signals"]:
                    sample_data = response.json()
                    print(f"      📄 Sample: {str(sample_data)[:100]}...")
                
                test_results["test_details"].append({
                    "test": test_name,
                    "status": "PASSED",
                    "response_time": response.elapsed.total_seconds()
                })
                
            else:
                test_results["failed_tests"] += 1
                print(f"   ❌ {test_name}: FAILED ({response.status_code})")
                test_results["test_details"].append({
                    "test": test_name,
                    "status": "FAILED",
                    "error": response.text[:200]
                })
                
        except Exception as e:
            test_results["failed_tests"] += 1
            print(f"   ❌ {test_name}: ERROR ({str(e)})")
            test_results["test_details"].append({
                "test": test_name,
                "status": "ERROR",
                "error": str(e)
            })
    
    print("🔄 Testing Core Endpoints...")
    
    # Test 1: System Health
    run_test("System Health Check", "/health")
    run_test("Root Endpoint", "/")
    
    # Test 2: Sentiment Analysis
    print("\n🧠 Testing Sentiment Analysis...")
    run_test("Sentiment Analysis", "/sentiment/analyze", "POST", {
        "text": "Bitcoin is looking very bullish today! 🚀 To the moon!",
        "crypto_symbol": "BTC"
    })
    run_test("Market Sentiment Overview", "/sentiment/market-overview")
    
    # Test 3: Price Predictions (might fail if no data)
    print("\n📈 Testing Price Predictions...")
    run_test("Market Trading Signals", "/predictions/market-signals")
    
    # Test 4: Market Data
    print("\n💰 Testing Market Data...")
    run_test("Market Summary", "/market/summary")
    run_test("Latest Prices", "/market/prices/latest")
    
    # Test 5: Analytics
    print("\n📊 Testing Analytics...")
    run_test("System Performance", "/analytics/performance")
    
    # Test 6: Specific crypto endpoints (might fail if no data)
    print("\n🪙 Testing Crypto-Specific Endpoints...")
    for crypto in ["BTC", "ETH", "SOL"]:
        run_test(f"Sentiment for {crypto}", f"/sentiment/crypto/{crypto}")
        run_test(f"Predictions for {crypto}", f"/predictions/crypto/{crypto}")
    
    # Print test summary
    print("\n" + "=" * 60)
    print("📊 API TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"✅ Passed: {test_results['passed_tests']}")
    print(f"❌ Failed: {test_results['failed_tests']}")
    
    success_rate = (test_results['passed_tests'] / test_results['total_tests']) * 100
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 API READY FOR BCG X DEMO!")
    elif success_rate >= 60:
        print("⚠️ API partially functional - some endpoints need data")
    else:
        print("❌ API needs troubleshooting")
    
    return test_results

if __name__ == "__main__":
    test_production_api()