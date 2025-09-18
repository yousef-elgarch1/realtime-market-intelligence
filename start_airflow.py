#!/usr/bin/env python3
"""
Start Apache Airflow services for BCG X platform
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def start_airflow_services():
    """Start Airflow webserver and scheduler"""
    
    # Set environment variables
    airflow_home = Path("airflow").absolute()
    os.environ['AIRFLOW_HOME'] = str(airflow_home)
    
    print("🚀 Starting Apache Airflow Services...")
    print("=" * 50)
    
    try:
        # Start webserver in background
        print("🌐 Starting Airflow Webserver...")
        webserver_process = subprocess.Popen([
            sys.executable, "-m", "airflow", "webserver", 
            "--port", "8080", "--daemon"
        ])
        
        time.sleep(3)  # Wait for webserver to start
        
        # Start scheduler in background  
        print("⚙️ Starting Airflow Scheduler...")
        scheduler_process = subprocess.Popen([
            sys.executable, "-m", "airflow", "scheduler", "--daemon"
        ])
        
        time.sleep(5)  # Wait for scheduler to start
        
        print("✅ Airflow Services Started!")
        print("🌐 Webserver: http://localhost:8080")
        print("🔑 Login: admin / admin123")
        print("📊 DAGs Available:")
        print("   • ml_training_pipeline (Daily)")
        print("   • real_time_predictions (Every 15 min)")
        print("   • data_pipeline_monitoring (Hourly)")
        
        return webserver_process, scheduler_process
        
    except Exception as e:
        print(f"❌ Failed to start Airflow: {e}")
        return None, None

if __name__ == "__main__":
    webserver, scheduler = start_airflow_services()
    
    if webserver and scheduler:
        try:
            print("\n⏰ Airflow is running. Press Ctrl+C to stop...")
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Stopping Airflow services...")
            webserver.terminate()
            scheduler.terminate()