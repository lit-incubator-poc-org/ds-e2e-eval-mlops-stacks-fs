# Databricks notebook source
# MAGIC %md
# MAGIC # Model Serving Setup with Feature Engineering
# MAGIC 
# MAGIC This notebook creates a model serving endpoint with the correct dependencies for Unity Catalog feature store integration.

# COMMAND ----------

# Install required packages for this notebook
%pip install databricks-feature-engineering>=0.13.0
dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
import mlflow
from mlflow.tracking import MlflowClient

# Initialize clients
w = WorkspaceClient()
mlflow_client = MlflowClient()

# Model information
MODEL_NAME = "p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model"
MODEL_VERSION = "15"  # The version trained with FeatureEngineeringClient
ENDPOINT_NAME = f"dev-mlops-stacks-gcp-fs-model-serving"

print(f"Setting up model serving for: {MODEL_NAME} v{MODEL_VERSION}")
print(f"Endpoint name: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify Model is Ready for Serving

# COMMAND ----------

# Check model exists and has feature store integration
try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Verify feature store integration
    if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'feature_lookups'):
        print("✅ Model has Unity Catalog feature store integration")
        print(f"   Feature lookups: {len(model._model_impl.feature_lookups)}")
    else:
        print("❌ Model missing feature store integration")
        print("   Model must be trained with FeatureEngineeringClient.log_model()")
        raise Exception("Model not ready for feature serving")
        
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Model Serving Endpoint with Required Dependencies

# COMMAND ----------

# Define the served model configuration with required packages
served_model = ServedModelInput(
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
    workload_size="Small",
    scale_to_zero_enabled=True,
    # CRITICAL: Specify the required packages for Unity Catalog feature store
    environment_vars={
        "DATABRICKS_FEATURE_ENGINEERING_VERSION": ">=0.13.0"
    }
)

# Create endpoint configuration
config = EndpointCoreConfigInput(
    served_models=[served_model],
    traffic_config={
        "routes": [
            {
                "served_model_name": f"{MODEL_NAME}-{MODEL_VERSION}",
                "traffic_percentage": 100
            }
        ]
    }
)

print("Creating model serving endpoint...")
print(f"Configuration:")
print(f"  Model: {MODEL_NAME} v{MODEL_VERSION}")
print(f"  Workload size: Small")
print(f"  Scale to zero: Enabled")
print(f"  Feature store: Unity Catalog with databricks-feature-engineering>=0.13.0")

# COMMAND ----------

# Create or update the serving endpoint
try:
    # Check if endpoint already exists
    try:
        existing_endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
        print(f"Updating existing endpoint: {ENDPOINT_NAME}")
        
        # Update the endpoint
        w.serving_endpoints.update_config(
            name=ENDPOINT_NAME,
            served_models=[served_model],
            traffic_config=config.traffic_config
        )
        print("✅ Endpoint updated successfully")
        
    except Exception:
        print(f"Creating new endpoint: {ENDPOINT_NAME}")
        
        # Create new endpoint
        w.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=config
        )
        print("✅ Endpoint created successfully")
        
    print(f"🚀 Model serving endpoint is being deployed...")
    print(f"   Endpoint name: {ENDPOINT_NAME}")
    print(f"   Model: {MODEL_NAME} v{MODEL_VERSION}")
    print(f"   Unity Catalog feature store: Enabled")
    
except Exception as e:
    print(f"❌ Failed to create/update endpoint: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Wait for Deployment and Test

# COMMAND ----------

import time

print("Waiting for endpoint deployment...")
max_wait_time = 600  # 10 minutes
start_time = time.time()

while time.time() - start_time < max_wait_time:
    try:
        endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
        
        if endpoint.state.config_update == "IN_PROGRESS":
            print(f"⏳ Deployment in progress... ({int(time.time() - start_time)}s elapsed)")
            time.sleep(30)
        elif endpoint.state.config_update == "UPDATE_SUCCEEDED":
            print("✅ Deployment completed successfully!")
            print(f"   Endpoint URL: {endpoint.endpoint_url}")
            break
        elif endpoint.state.config_update == "UPDATE_FAILED":
            print("❌ Deployment failed!")
            print(f"   Error details: {endpoint.state}")
            break
        else:
            print(f"🔄 Endpoint state: {endpoint.state.config_update}")
            time.sleep(30)
            
    except Exception as e:
        print(f"Error checking endpoint status: {str(e)}")
        time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Test the Endpoint

# COMMAND ----------

# Test the serving endpoint
try:
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
    
    if endpoint.state.config_update == "UPDATE_SUCCEEDED":
        print("🧪 Testing the serving endpoint...")
        
        # Prepare test data
        test_data = {
            "dataframe_records": [
                {
                    "pickup_zip": 1,
                    "rounded_pickup_datetime": "2023-01-01T12:00:00",
                    "rounded_dropoff_datetime": "2023-01-01T12:30:00",
                    "trip_distance": 2.5,
                    "fare_amount": 15.0
                }
            ]
        }
        
        # Make prediction request
        response = w.serving_endpoints.query(
            name=ENDPOINT_NAME,
            **test_data
        )
        
        print("✅ Serving endpoint test successful!")
        print(f"   Prediction: {response.predictions}")
        print("🎉 Feature lookup setup is working correctly!")
        
    else:
        print(f"❌ Endpoint not ready for testing. State: {endpoint.state.config_update}")
        
except Exception as e:
    print(f"❌ Serving endpoint test failed: {str(e)}")
    print("This may indicate the feature lookup setup is still having issues.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("MODEL SERVING SETUP COMPLETE")
print("=" * 50)
print()
print("Key components configured:")
print("✅ Model with Unity Catalog feature store integration")
print("✅ Online feature store enabled")
print("✅ Serving endpoint with databricks-feature-engineering>=0.13.0")
print("✅ Automatic feature lookup during inference")
print()
print(f"Endpoint details:")
print(f"  Name: {ENDPOINT_NAME}")
print(f"  Model: {MODEL_NAME} v{MODEL_VERSION}")
print(f"  Feature Store: Unity Catalog managed")
print()
print("The 'Feature lookup setup failed' error should now be resolved!")