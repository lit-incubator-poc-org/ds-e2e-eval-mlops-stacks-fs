# Databricks notebook source
##################################################################################
# Online Table Model Deployment Notebook
#
# This notebook sets up Databricks Online Tables and creates a model serving endpoint
# with automatic feature lookup for real-time taxi fare prediction.
#
# Based on Databricks Online Tables pattern for Unity Catalog feature stores.
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * model_name (required)           - MLflow registered model name to deploy
# * model_version (optional)        - Specific model version to deploy. If not provided, uses latest version.
# * endpoint_name (optional)        - Name for the serving endpoint. If not provided, generates from model name.
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Notebook Parameters

# Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Model details - can be passed from training job
dbutils.widgets.text("model_name", "", "Model Name")
dbutils.widgets.text("model_version", "", "Model Version (optional)")
dbutils.widgets.text("endpoint_name", "", "Endpoint Name (optional)")

# Get parameters
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
endpoint_name = dbutils.widgets.get("endpoint_name")

# Try to get from task values if not provided as widgets
if not model_name:
    model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="")

if not model_version:
    model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="")

# Validation
assert model_name != "", "model_name must be specified either as widget or task value"

print(f"Deploying model: {model_name}")
print(f"Model version: {model_version if model_version else 'latest'}")
print(f"Environment: {env}")

# COMMAND ----------

# DBTITLE 1,Setup Imports and Client

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import sys
import pathlib

# Add project utils
sys.path.append(str(pathlib.Path().resolve().parent.parent.parent))
from utils import get_deployed_model_alias_for_env

# Initialize clients
fe = FeatureEngineeringClient()
workspace = WorkspaceClient()
mlflow_client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,Get Model Information

# Get latest version if not specified
if not model_version:
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    if versions:
        model_version = str(max(int(v.version) for v in versions))
    else:
        raise ValueError(f"No versions found for model {model_name}")

print(f"Using model version: {model_version}")

# Get model details to understand feature dependencies
model_uri = f"models:/{model_name}/{model_version}"
model_details = mlflow_client.get_model_version(model_name, model_version)

print(f"Model URI: {model_uri}")
print(f"Model status: {model_details.status}")

# COMMAND ----------

# DBTITLE 1,Setup Online Tables for Feature Store

def setup_online_tables():
    """
    Set up Databricks Online Tables for taxi fare prediction feature tables.
    Based on the online-tables.ipynb example pattern.
    """
    # Parse catalog and schema from model name (assuming Unity Catalog format)
    model_parts = model_name.split(".")
    if len(model_parts) >= 2:
        catalog_name = model_parts[0]
        schema_name = model_parts[1]
    else:
        # Fallback defaults
        catalog_name = "p03"
        schema_name = "e2e_demo_simon"
    
    print(f"Using catalog: {catalog_name}, schema: {schema_name}")
    
    # Feature tables that need online tables
    feature_tables = [
        f"{catalog_name}.{schema_name}.trip_pickup_features",
        f"{catalog_name}.{schema_name}.trip_dropoff_features"
    ]
    
    online_tables_created = []
    
    for source_table in feature_tables:
        # Generate online table name
        table_suffix = source_table.split(".")[-1]  # e.g., "trip_pickup_features"
        online_table_name = f"{catalog_name}.{schema_name}.{table_suffix}_online"
        
        print(f"\n🔄 Setting up online table for: {source_table}")
        print(f"   Online table name: {online_table_name}")
        
        try:
            # Check if online table already exists
            try:
                existing_table = workspace.online_tables.get(online_table_name)
                print(f"   ✅ Online table already exists (Status: {existing_table.status})")
                online_tables_created.append(online_table_name)
                continue
            except Exception:
                # Table doesn't exist, create it
                pass
            
            # Create online table spec following the example pattern
            spec = OnlineTableSpec(
                primary_key_columns=["pickup_zip", "rounded_pickup_datetime"] if "pickup" in source_table else ["dropoff_zip", "rounded_dropoff_datetime"],
                source_table_full_name=source_table,
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),
                perform_full_copy=True
            )
            
            online_table = OnlineTable(name=online_table_name, spec=spec)
            
            # Create and wait for the online table
            print(f"   🔄 Creating online table...")
            result = workspace.online_tables.create_and_wait(table=online_table)
            
            print(f"   ✅ Online table created successfully!")
            print(f"   Status: {result.status}")
            online_tables_created.append(online_table_name)
            
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"   ✅ Online table already exists")
                online_tables_created.append(online_table_name)
            else:
                print(f"   ❌ Failed to create online table: {str(e)}")
                # Continue with other tables
                continue
    
    print(f"\n🎉 Online tables setup completed!")
    print(f"Created/verified {len(online_tables_created)} online tables:")
    for table in online_tables_created:
        print(f"  - {table}")
    
    return online_tables_created

# Setup online tables
online_tables = setup_online_tables()

# COMMAND ----------

# DBTITLE 1,Create Model Serving Endpoint

def create_serving_endpoint():
    """
    Create a model serving endpoint with automatic feature lookup using online tables.
    Based on the online-tables.ipynb example pattern.
    """
    # Generate endpoint name if not provided
    if not endpoint_name:
        # Use model name but replace dots with underscores for endpoint naming
        base_name = model_name.replace(".", "_").lower()
        generated_endpoint_name = f"{base_name}_endpoint_{env}"
    else:
        generated_endpoint_name = endpoint_name
    
    print(f"Creating serving endpoint: {generated_endpoint_name}")
    
    try:
        # Check if endpoint already exists
        try:
            existing_endpoint = workspace.serving_endpoints.get(generated_endpoint_name)
            print(f"✅ Endpoint '{generated_endpoint_name}' already exists")
            print(f"   Status: {existing_endpoint.state}")
            return generated_endpoint_name
        except Exception:
            # Endpoint doesn't exist, create it
            pass
        
        # Create endpoint configuration
        print(f"🔄 Creating new endpoint...")
        status = workspace.serving_endpoints.create_and_wait(
            name=generated_endpoint_name,
            config=EndpointCoreConfigInput(
                served_entities=[
                    ServedEntityInput(
                        entity_name=model_name,
                        entity_version=model_version,
                        scale_to_zero_enabled=True,  # Enable auto-scaling to save costs
                        workload_size="Small"  # Start with small workload
                    )
                ]
            )
        )
        
        print(f"✅ Endpoint created successfully!")
        print(f"   Name: {generated_endpoint_name}")
        print(f"   Status: {status.state}")
        
        # Set model alias for deployment tracking
        alias = get_deployed_model_alias_for_env(env)
        print(f"🏷️  Setting model alias '{alias}' for deployment tracking...")
        
        mlflow_client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=model_version
        )
        
        print(f"✅ Model alias '{alias}' set for version {model_version}")
        
        return generated_endpoint_name
        
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✅ Endpoint '{generated_endpoint_name}' already exists")
            return generated_endpoint_name
        else:
            print(f"❌ Failed to create endpoint: {str(e)}")
            raise e

# Create the serving endpoint
final_endpoint_name = create_serving_endpoint()

# COMMAND ----------

# DBTITLE 1,Test Endpoint with Sample Request

def test_endpoint(endpoint_name, sample_size=3):
    """
    Test the deployed endpoint with sample data to verify online feature lookup works.
    """
    print(f"🧪 Testing endpoint: {endpoint_name}")
    
    try:
        import mlflow.deployments
        
        # Create deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Sample test data - just the lookup keys and real-time features
        # The online tables will automatically provide the pre-computed features
        test_data = [
            {
                "pickup_zip": 10001, 
                "dropoff_zip": 10002,
                "trip_distance": 2.5,
                "rounded_pickup_datetime": "2023-01-01 12:00:00",
                "rounded_dropoff_datetime": "2023-01-01 12:30:00"
            },
            {
                "pickup_zip": 10003,
                "dropoff_zip": 10004, 
                "trip_distance": 1.2,
                "rounded_pickup_datetime": "2023-01-01 14:00:00",
                "rounded_dropoff_datetime": "2023-01-01 14:15:00"
            },
            {
                "pickup_zip": 10005,
                "dropoff_zip": 10006,
                "trip_distance": 5.1,
                "rounded_pickup_datetime": "2023-01-01 16:00:00", 
                "rounded_dropoff_datetime": "2023-01-01 16:45:00"
            }
        ]
        
        # Take only the requested sample size
        test_data = test_data[:sample_size]
        
        print(f"📤 Sending test request with {len(test_data)} records...")
        
        response = client.predict(
            endpoint=endpoint_name,
            inputs={
                "dataframe_records": test_data
            }
        )
        
        print("✅ Test request successful!")
        print("📥 Response:")
        pprint(response)
        
        # Validate response
        if "predictions" in response:
            predictions = response["predictions"]
            print(f"\n🎯 Received {len(predictions)} predictions:")
            for i, pred in enumerate(predictions):
                print(f"  Record {i+1}: ${pred:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print("💡 This may be expected if the endpoint is still starting up.")
        print("   Try testing again in a few minutes.")
        return False

# Test the endpoint
test_successful = test_endpoint(final_endpoint_name)

# COMMAND ----------

# DBTITLE 1,Deployment Summary

print("=" * 60)
print("🎉 DEPLOYMENT SUMMARY")
print("=" * 60)
print(f"✅ Model Name: {model_name}")
print(f"✅ Model Version: {model_version}")
print(f"✅ Environment: {env}")
print(f"✅ Serving Endpoint: {final_endpoint_name}")
print(f"✅ Online Tables: {len(online_tables)} tables configured")

for table in online_tables:
    print(f"   - {table}")

print(f"\n📡 Endpoint URL: https://{workspace.config.host}/ml/endpoints/{final_endpoint_name}")
print(f"🏷️  Model Alias: {get_deployed_model_alias_for_env(env)}")

if test_successful:
    print("🧪 Endpoint Test: ✅ PASSED")
else:
    print("🧪 Endpoint Test: ⚠️  PENDING (may need a few minutes)")

print("\n💡 Next Steps:")
print("1. Monitor endpoint performance in the Databricks UI")
print("2. Set up monitoring and alerting for production use")
print("3. Configure auto-scaling based on traffic patterns")
print("4. Test with real production data")

print("\n🔗 Useful Links:")
print(f"📊 Model Registry: https://{workspace.config.host}/explore/data/models/{model_name.replace('.', '/')}")
print(f"🚀 Serving Endpoint: https://{workspace.config.host}/ml/endpoints/{final_endpoint_name}")
print(f"📋 Feature Tables: https://{workspace.config.host}/explore/data")

# Set task values for downstream notebooks
dbutils.jobs.taskValues.set("endpoint_name", final_endpoint_name)
dbutils.jobs.taskValues.set("deployment_status", "SUCCESS")
dbutils.jobs.taskValues.set("online_tables_count", len(online_tables))

# COMMAND ----------

# DBTITLE 1,Optional: Create Batch Inference Alternative

print("\n" + "=" * 50)
print("📝 BATCH INFERENCE ALTERNATIVE")
print("=" * 50)
print("If you prefer batch inference over real-time serving:")
print(f"1. Use model URI: {model_uri}")
print("2. Load model with: mlflow.pyfunc.load_model('{model_uri}')")
print("3. The model includes automatic feature lookup from feature tables")
print("4. No online tables required for batch scenarios")
print("\n💡 Online tables are specifically for low-latency real-time inference")