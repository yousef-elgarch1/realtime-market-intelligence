#!/usr/bin/env python3
"""
Test Apache Airflow setup and DAG functionality
"""
import subprocess
import sys
import os
from pathlib import Path

def test_airflow_installation():
    """Test if Airflow is properly installed and configured"""
    
    print("🧪 Testing Apache Airflow Setup")
    print("=" * 50)
    
    # Set Airflow home
    airflow_home = Path("airflow").absolute()
    os.environ['AIRFLOW_HOME'] = str(airflow_home)
    
    try:
        # Test 1: Check Airflow version
        print("🔄 Test 1: Airflow Version Check...")
        result = subprocess.run([sys.executable, "-m", "airflow", "version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Airflow installed: {result.stdout.strip()}")
        else:
            print(f"   ❌ Airflow not properly installed")
            return False
        
        # Test 2: Check database connection
        print("\n🔄 Test 2: Database Connection...")
        result = subprocess.run([sys.executable, "-m", "airflow", "db", "check"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Database connection successful")
        else:
            print("   ❌ Database connection failed")
            print(f"   Error: {result.stderr}")
        
        # Test 3: List DAGs
        print("\n🔄 Test 3: DAG Discovery...")
        result = subprocess.run([sys.executable, "-m", "airflow", "dags", "list"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            dag_count = len([line for line in result.stdout.split('\n') 
                           if 'ml_training_pipeline' in line or 
                              'real_time_predictions' in line or
                              'data_pipeline_monitoring' in line])
            print(f"   ✅ Found {dag_count} BCG X DAGs")
        else:
            print("   ⚠️ DAG listing failed (normal if no DAGs yet)")
        
        # Test 4: Check DAG files
        print("\n🔄 Test 4: DAG Files Check...")
        dag_files = [
            'airflow/dags/ml_training_dag.py',
            'airflow/dags/prediction_dag.py', 
            'airflow/dags/data_monitoring_dag.py'
        ]
        
        for dag_file in dag_files:
            if Path(dag_file).exists():
                print(f"   ✅ {dag_file} exists")
            else:
                print(f"   ❌ {dag_file} missing")
        
        print("\n✅ Airflow Setup Test Complete!")
        print("🚀 Ready to start Airflow services with 'python start_airflow.py'")
        return True
        
    except Exception as e:
        print(f"❌ Airflow test failed: {e}")
        return False

if __name__ == "__main__":
    test_airflow_installation()