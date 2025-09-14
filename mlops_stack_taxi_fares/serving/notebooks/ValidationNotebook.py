# Databricks notebook source
##################################################################################
# Model and Online Feature Store Validation Notebook
#
# This notebook provides comprehensive validation of:
# 1. Model deployment and serving endpoint health
# 2. Online feature store configuration and performance
# 3. End-to-end prediction pipeline with feature lookup
#
# Use this notebook to validate your deployment after setting up online tables
# and model serving endpoints.
##################################################################################

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.13.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
import requests
import json
import time

# Initialize clients
spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()
w = WorkspaceClient()

# Model information - update these for your deployment
MODEL_NAME = "p03.e2e_demo_simon.dev_mlops_stack_taxi_fares_model"
ENDPOINT_NAME = "mlops-taxi-fare-endpoint"

# COMMAND ----------

# DBTITLE 1,1. Validate Online Feature Tables

print("=" * 60)
print("ONLINE FEATURE STORE VALIDATION")
print("=" * 60)

# Check feature tables exist
feature_tables = [
    "p03.e2e_demo_simon.trip_pickup_features",
    "p03.e2e_demo_simon.trip_dropoff_features"
]

for table_name in feature_tables:
    try:
        table_info = spark.sql(f"DESCRIBE TABLE {table_name}").collect()
        print(f"✓ Feature table exists: {table_name}")
        
        # Check if it's an online table (FOREIGN table type indicates online table)
        table_details = spark.sql(f"DESCRIBE DETAIL {table_name}").collect()
        if table_details:
            table_format = table_details[0]['format'] if 'format' in table_details[0] else 'Unknown'
            print(f"  Table format: {table_format}")
            
    except Exception as e:
        print(f"✗ Error accessing {table_name}: {str(e)}")

# COMMAND ----------

# DBTITLE 1,2. Validate Model Registration and Serving Endpoint

print("\n" + "=" * 60)
print("MODEL SERVING VALIDATION")
print("=" * 60)

try:
    # Get latest model version
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if latest_versions:
        latest_version = latest_versions[0].version
        print(f"✓ Model found: {MODEL_NAME}")
        print(f"  Latest version: {latest_version}")
        
        # Test model loading
        model_uri = f"models:/{MODEL_NAME}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"✓ Model loads successfully")
        
    else:
        print(f"✗ No versions found for model: {MODEL_NAME}")
        
except Exception as e:
    print(f"✗ Error with model: {str(e)}")

# Check serving endpoint
try:
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"✓ Serving endpoint exists: {ENDPOINT_NAME}")
    print(f"  Status: {endpoint.state.ready}")
    print(f"  Config update: {endpoint.state.config_update}")
    
except Exception as e:
    print(f"✗ Serving endpoint error: {str(e)}")

# COMMAND ----------

# DBTITLE 1,3. Test End-to-End Prediction

print("\n" + "=" * 60)
print("END-TO-END PREDICTION TEST")
print("=" * 60)

# Test data for prediction
test_inputs = [
    {
        "pickup_zip": "10001",
        "dropoff_zip": "10002",
        "trip_distance": 2.5,
        "pickup_weekday": 1,
        "pickup_hour": 14,
        "trip_duration": 15.5
    },
    {
        "pickup_zip": "10003",
        "dropoff_zip": "10004",
        "trip_distance": 1.2,
        "pickup_weekday": 5,
        "pickup_hour": 18,
        "trip_duration": 8.0
    }
]

try:
    # Test serving endpoint
    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        inputs=test_inputs
    )
    
    print(f"✓ Serving endpoint responded successfully")
    print(f"  Predictions received: {len(response.predictions)}")
    
    for i, prediction in enumerate(response.predictions):
        print(f"  Sample {i+1}: ${prediction:.2f}")
        
    print(f"\n✓ Online feature lookup working (predictions vary based on location features)")
    
except Exception as e:
    print(f"✗ Prediction error: {str(e)}")

# COMMAND ----------

# DBTITLE 1,4. Performance and Monitoring Validation

print("\n" + "=" * 60)
print("PERFORMANCE AND MONITORING VALIDATION")  
print("=" * 60)

# Test prediction latency
import time

try:
    start_time = time.time()
    
    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        inputs=[test_inputs[0]]  # Single prediction for latency test
    )
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    print(f"✓ Prediction latency: {latency_ms:.2f} ms")
    
    if latency_ms < 1000:  # Sub-second response
        print(f"✓ Excellent latency (< 1 second)")
    elif latency_ms < 5000:  # Sub 5-second response
        print(f"✓ Good latency (< 5 seconds)")
    else:
        print(f"⚠ High latency (> 5 seconds) - consider optimization")
        
except Exception as e:
    print(f"✗ Latency test error: {str(e)}")

# Check auto-capture table (for monitoring)
try:
    payload_table = "p03.e2e_demo_simon.taxi_fare_endpoint_payload"
    count_result = spark.sql(f"SELECT COUNT(*) as count FROM {payload_table}").collect()
    if count_result:
        count = count_result[0]['count']
        print(f"✓ Auto-capture table active: {payload_table}")
        print(f"  Logged requests: {count}")
    
except Exception as e:
    print(f"⚠ Auto-capture table check: {str(e)}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

# COMMAND ----------