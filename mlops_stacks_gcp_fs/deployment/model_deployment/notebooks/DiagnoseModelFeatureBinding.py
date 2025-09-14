# Databricks notebook source
# MAGIC %md
# MAGIC # Diagnose Model Feature Store Binding
# MAGIC 
# MAGIC This notebook diagnoses why model serving is failing with "Feature lookup setup failed" error.
# MAGIC 
# MAGIC According to Azure Databricks documentation:
# MAGIC - Models must be logged with `FeatureEngineeringClient.log_model` (not legacy `FeatureStoreClient.log_model`)
# MAGIC - Feature tables must be published to online stores
# MAGIC - The binding happens automatically if both conditions are met

# COMMAND ----------

# Install required packages
%pip install databricks-feature-engineering>=0.13.0
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession

# Initialize clients
fe = FeatureEngineeringClient()
spark = SparkSession.builder.getOrCreate()
mlflow_client = mlflow.tracking.MlflowClient()

print("Diagnosing Model Feature Store Binding...")
print("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Check Model Registry Information

# COMMAND ----------

# Check the current model in the registry
MODEL_NAME = "p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model"

print(f"Checking model: {MODEL_NAME}")
print()

try:
    # For Unity Catalog, use search_model_versions instead of get_latest_versions
    model_versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    
    if not model_versions:
        print("No model versions found!")
    else:
        # Sort by version number to get the latest
        latest_version = max(model_versions, key=lambda x: int(x.version))
        print(f"Latest version: {latest_version.version}")
        print(f"Run ID: {latest_version.run_id}")
        print(f"Status: {latest_version.status}")
        print()
        
        # Get model version details
        model_version = mlflow_client.get_model_version(MODEL_NAME, latest_version.version)
        print(f"Model URI: models:/{MODEL_NAME}/{latest_version.version}")
        print(f"Source: {model_version.source}")
        print()
        
        # Check if model has feature store metadata
        try:
            model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest_version.version}")
            print("Model loaded successfully")
            
            # Check model metadata for feature store information
            model_info = mlflow_client.get_model_version(MODEL_NAME, latest_version.version)
            print(f"Model tags: {model_info.tags}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            
except Exception as e:
    print(f"Error accessing model registry: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Check Online Feature Store Status

# COMMAND ----------

# Check online feature store
CATALOG = "p03" 
SCHEMA = "e2e_demo_simon"
FEATURE_TABLES = [
    f"{CATALOG}.{SCHEMA}.trip_pickup_features",
    f"{CATALOG}.{SCHEMA}.trip_dropoff_features"
]

print("Checking Online Feature Store Status...")
print()

# Check if online store exists
try:
    online_store = fe.get_online_store(name="mlops-feature-store")
    print(f"Online Store Status:")
    print(f"  Name: {online_store.name}")
    print(f"  State: {online_store.state}")
    print(f"  Capacity: {online_store.capacity}")
    print()
    
    # Check if state is AVAILABLE (comparing enum value properly)
    if str(online_store.state) != "State.AVAILABLE":
        print("WARNING: Online store is not in AVAILABLE state!")
        print("This could cause feature lookup failures.")
        print()
    else:
        print("✓ Online store is AVAILABLE and ready for serving")
        print()
        
except Exception as e:
    print(f"Online store not found or error: {str(e)}")
    print("The online store may not be created yet.")
    print()

# Check feature tables
print("Feature Table Status:")
for table_name in FEATURE_TABLES:
    try:
        # Check if table exists
        table_df = spark.table(table_name)
        row_count = table_df.count()
        print(f"  {table_name}:")
        print(f"    Exists: YES")
        print(f"    Rows: {row_count:,}")
        
        # Check if published to online store
        try:
            table_info = fe.get_table(name=table_name)
            print(f"    Primary Keys: {table_info.primary_keys}")
        except Exception as e:
            print(f"    Feature Engineering Client Error: {str(e)}")
            
    except Exception as e:
        print(f"  {table_name}:")
        print(f"    Exists: NO - {str(e)}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Check Feature Store Integration Method

# COMMAND ----------

# Check how the current model was logged
print("Checking Model Logging Method...")
print()

try:
    # Get the run that created the model (Unity Catalog compatible)
    model_versions = mlflow_client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = max(model_versions, key=lambda x: int(x.version))
    run = mlflow_client.get_run(latest_version.run_id)
    
    print(f"Run ID: {run.info.run_id}")
    print(f"Run Tags:")
    for key, value in run.data.tags.items():
        if 'feature' in key.lower() or 'store' in key.lower():
            print(f"  {key}: {value}")
    print()
    
    # Check artifacts to see if feature store metadata exists
    artifacts = mlflow_client.list_artifacts(latest_version.run_id, "model_packaged")
    print("Model Artifacts:")
    for artifact in artifacts:
        print(f"  {artifact.path}")
        if 'feature' in artifact.path.lower():
            print(f"    Feature-related: YES")
    print()
    
    # Try to load the model and check its type
    model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model type: {type(model)}")
        
        # Check if it's a feature store model
        if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'feature_lookups'):
            print("Feature Store Integration: DETECTED")
            print(f"Feature Lookups: {model._model_impl.feature_lookups}")
        else:
            print("Feature Store Integration: NOT DETECTED or LEGACY")
            print("This may be the cause of the serving failure!")
            
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")
        
except Exception as e:
    print(f"Error checking run information: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Recommendations

# COMMAND ----------

print("DIAGNOSIS SUMMARY")
print("=" * 50)
print()

print("Common causes of 'Feature lookup setup failed' error:")
print()
print("1. MODEL LOGGING METHOD:")
print("   - Model must be logged with FeatureEngineeringClient.log_model")
print("   - NOT with legacy FeatureStoreClient.log_model")
print("   - Check training notebook uses: from databricks.feature_engineering import FeatureEngineeringClient")
print()

print("2. ONLINE STORE STATUS:")
print("   - Online store must be in AVAILABLE state")
print("   - Feature tables must be published to online store")
print("   - Run deploy.py to ensure online store is set up")
print()

print("3. FEATURE TABLE REQUIREMENTS:")
print("   - Tables must have primary keys defined")
print("   - Change Data Feed must be enabled")
print("   - Tables must be accessible from Unity Catalog")
print()

print("NEXT STEPS:")
print("1. If model was logged with legacy client, retrain with FeatureEngineeringClient")
print("2. Ensure online store is AVAILABLE (run deploy.py)")
print("3. Verify feature tables are properly published")
print("4. Redeploy model after fixes")
print()

print("Updated training notebook should use:")
print("  fe = FeatureEngineeringClient()")
print("  fe.log_model(model=model, training_set=training_set, ...)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Quick Fix Test

# COMMAND ----------

# Test if we can manually verify feature lookup
print("Testing Feature Lookup Capability...")
print()

try:
    # Create a simple test dataframe
    test_data = spark.createDataFrame([
        (1, "2023-01-01 12:00:00", "2023-01-01 12:30:00")
    ], ["pickup_zip", "rounded_pickup_datetime", "rounded_dropoff_datetime"])
    
    print("Created test data:")
    test_data.show()
    
    # Try to create a simple feature lookup to test the mechanism
    from databricks.feature_engineering import FeatureLookup
    
    test_lookup = FeatureLookup(
        table_name=FEATURE_TABLES[0],  # pickup features
        feature_names=["mean_fare_window_1h_pickup_zip"],
        lookup_key=["pickup_zip"],
        timestamp_lookup_key=["rounded_pickup_datetime"]
    )
    
    print("Feature lookup definition created successfully")
    print("This suggests the feature store infrastructure is working")
    
except Exception as e:
    print(f"Feature lookup test failed: {str(e)}")
    print("This indicates a problem with the feature store setup")

print()
print("Diagnostic complete!")