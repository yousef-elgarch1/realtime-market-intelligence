#!/usr/bin/env python3
"""
Apache Airflow setup for BCG X Market Intelligence Platform
"""
import os
from pathlib import Path
import subprocess
import sys

def setup_airflow():
    """Initialize Apache Airflow for ML orchestration"""
    
    print("🔄 Setting up Apache Airflow for ML Orchestration...")
    
    # Set Airflow environment variables
    airflow_home = Path("airflow").absolute()
    airflow_home.mkdir(exist_ok=True)
    
    os.environ['AIRFLOW_HOME'] = str(airflow_home)
    os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = str(airflow_home / "dags")
    os.environ['AIRFLOW__CORE__PLUGINS_FOLDER'] = str(airflow_home / "plugins")
    os.environ['AIRFLOW__CORE__LOGS_FOLDER'] = str(airflow_home / "logs")
    os.environ['AIRFLOW__CORE__EXECUTOR'] = 'LocalExecutor'
    os.environ['AIRFLOW__DATABASE__SQL_ALCHEMY_CONN'] = 'sqlite:///airflow/airflow.db'
    os.environ['AIRFLOW__WEBSERVER__SECRET_KEY'] = 'bcg_x_market_intelligence'
    os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
    
    # Create necessary directories
    (airflow_home / "dags").mkdir(exist_ok=True)
    (airflow_home / "logs").mkdir(exist_ok=True)
    (airflow_home / "plugins").mkdir(exist_ok=True)
    
    try:
        # Initialize Airflow database
        print("📊 Initializing Airflow database...")
        subprocess.run([sys.executable, "-m", "airflow", "db", "init"], check=True)
        
        # Create admin user
        print("👤 Creating Airflow admin user...")
        subprocess.run([
            sys.executable, "-m", "airflow", "users", "create",
            "--username", "admin",
            "--firstname", "BCG",
            "--lastname", "Admin", 
            "--email", "admin@bcg.com",
            "--role", "Admin",
            "--password", "admin123"
        ], check=True)
        
        print("✅ Airflow setup completed!")
        print("🌐 Start Airflow with: 'airflow webserver --port 8080'")
        print("⚙️ Start scheduler with: 'airflow scheduler'")
        print("🔑 Login: admin / admin123")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Airflow setup failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    setup_airflow()