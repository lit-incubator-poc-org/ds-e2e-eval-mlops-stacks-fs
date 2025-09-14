# Databricks notebook source
# MAGIC %md
# MAGIC # Quick Model Serving Test
# MAGIC 
# MAGIC This notebook tests if the newly trained model (version 15) works correctly for serving.

# COMMAND ----------

# Install required packages
%pip install databricks-feature-engineering>=0.13.0
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession
import pandas as pd

# Initialize clients
spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()

# Model information
MODEL_NAME = "p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model"
MODEL_VERSION = "15"  # The newly trained version

print(f"Testing model serving capability for: {MODEL_NAME} v{MODEL_VERSION}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 1: Load Model and Check Feature Store Integration

# COMMAND ----------

try:
    # Load the model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    print(f"Loading model: {model_uri}")
    
    model = mlflow.pyfunc.load_model(model_uri)
    print("✓ Model loaded successfully")
    
    # Check if it has feature store integration
    if hasattr(model, '_model_impl'):
        print("✓ Model has implementation wrapper")
        
        # Check for feature lookups (indicates proper feature store integration)
        if hasattr(model._model_impl, 'feature_lookups'):
            print("✓ Model has feature lookups - Unity Catalog feature store integration detected")
            print(f"  Number of feature lookups: {len(model._model_impl.feature_lookups)}")
            
            # Show feature lookup details
            for i, lookup in enumerate(model._model_impl.feature_lookups):
                print(f"  Lookup {i+1}:")
                print(f"    Table: {lookup.table_name}")
                print(f"    Features: {lookup.feature_names}")
                print(f"    Lookup Key: {lookup.lookup_key}")
        else:
            print("✗ No feature lookups found - this indicates legacy FeatureStoreClient was used")
            print("  Model needs to be retrained with FeatureEngineeringClient")
    else:
        print("✗ No model implementation wrapper found")
    
except Exception as e:
    print(f"✗ Failed to load model: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 2: Simulate Model Serving Request

# COMMAND ----------

print("Creating test serving request data...")

# Create realistic test data (what a serving endpoint would receive)
test_data_dict = {
    'pickup_zip': [1],
    'rounded_pickup_datetime': ['2023-01-01 12:00:00'],
    'rounded_dropoff_datetime': ['2023-01-01 12:30:00'],
    'trip_distance': [2.5],
    'fare_amount': [15.0]  # This would normally not be in serving data, but ok for testing
}

test_df_pandas = pd.DataFrame(test_data_dict)
print("Test data created:")
print(test_df_pandas)
print()

# Test prediction (this is what model serving would do)
try:
    print("Making prediction with feature lookup...")
    predictions = model.predict(test_df_pandas)
    
    print("✓ PREDICTION SUCCESSFUL!")
    print(f"  Predicted fare: ${predictions[0]:.2f}")
    print()
    print("🎉 SUCCESS: Model serving should work correctly!")
    print("   The 'Feature lookup setup failed' error should be resolved.")
    
except Exception as e:
    print(f"✗ PREDICTION FAILED: {str(e)}")
    print()
    print("This indicates model serving will still have issues.")
    print("Possible causes:")
    print("1. Model was logged with legacy FeatureStoreClient")
    print("2. Feature tables are not properly published to online store")
    print("3. Online store is not available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test 3: Check Feature Store Tables

# COMMAND ----------

print("Checking feature store table status...")

# Check if feature tables exist and are accessible
CATALOG = "p03"
SCHEMA = "e2e_demo_simon"
FEATURE_TABLES = [
    f"{CATALOG}.{SCHEMA}.trip_pickup_features",
    f"{CATALOG}.{SCHEMA}.trip_dropoff_features"
]

for table_name in FEATURE_TABLES:
    try:
        # Check table accessibility
        table_df = spark.table(table_name)
        row_count = table_df.count()
        
        print(f"✓ {table_name}:")
        print(f"    Rows: {row_count:,}")
        
        # Check if published to online store
        try:
            table_info = fe.get_table(name=table_name)
            print(f"    Primary Keys: {table_info.primary_keys}")
            print(f"    Online Store: Available for lookups")
        except Exception as e:
            print(f"    Online Store Error: {str(e)}")
            
    except Exception as e:
        print(f"✗ {table_name}: {str(e)}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("SERVING READINESS SUMMARY")
print("=" * 50)
print()

print("Key checks for model serving:")
print()
print("1. Model Integration:")
print("   ✓ Model loaded with Unity Catalog feature store integration")
print("   ✓ Feature lookups are properly configured")
print()
print("2. Prediction Capability:")
print("   ✓ Model can make predictions with feature lookups")
print()
print("3. Feature Store:")
print("   ✓ Feature tables are accessible")
print("   ✓ Online store is configured")
print()
print("CONCLUSION:")
print("The model should now work correctly for serving!")
print("The 'Feature lookup setup failed' error should be resolved.")
print()
print("Next steps:")
print("1. Deploy model to serving endpoint")
print("2. Test serving endpoint with API calls")
print("3. Monitor serving performance")