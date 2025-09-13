import sys
import pathlib
import os

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from utils import get_deployed_model_alias_for_env
from mlflow.tracking import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient


def enable_online_feature_store(catalog_name="p03", schema_name="e2e_demo_simon"):
    """
    Enable Databricks-managed online feature store for feature tables required by the model.
    Uses the new databricks-feature-engineering package for both AWS and Azure.
    
    :param catalog_name: Unity Catalog name
    :param schema_name: Schema name containing feature tables
    """
    fe = FeatureEngineeringClient()
    
    # Feature tables used by the model
    feature_tables = [
        f"{catalog_name}.{schema_name}.trip_pickup_features",
        f"{catalog_name}.{schema_name}.trip_dropoff_features"
    ]
    
    print("🚀 Setting up Databricks-managed Online Feature Store...")
    
    # Step 1: Create or get the online store
    store_name = "mlops-feature-store"
    try:
        # Check if store already exists
        online_store = fe.get_online_store(name=store_name)
        print(f"✅ Online store '{store_name}' already exists (State: {online_store.state})")
    except:
        print(f"🔄 Creating new online store '{store_name}'...")
        try:
            online_store = fe.create_online_store(
                name=store_name,
                capacity="CU_1"  # Start with minimal capacity
            )
            print(f"✅ Created online store '{store_name}' (State: {online_store.state})")
        except Exception as e:
            print(f"❌ Failed to create online store: {str(e)}")
            return
    
    # Step 2: Prepare and publish feature tables
    for table_name in feature_tables:
        try:
            print(f"📋 Processing {table_name}...")
            
            # Check if table exists and get info
            try:
                table_info = fe.get_table(name=table_name)
                print(f"  ✅ Table exists with primary keys: {table_info.primary_keys}")
            except Exception as e:
                print(f"  ❌ Table not found: {str(e)}")
                continue
            
            # Enable Change Data Feed if not already enabled
            try:
                from pyspark.sql import SparkSession
                spark = SparkSession.builder.getOrCreate()
                
                spark.sql(f"""
                    ALTER TABLE {table_name}
                    SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
                """)
                print(f"  ✅ Change Data Feed enabled")
            except Exception as e:
                print(f"  ⚠️ Could not enable Change Data Feed: {str(e)}")
            
            # Publish to online store
            try:
                online_table_name = table_name.replace("_features", "_online_features")
                
                fe.publish_table(
                    online_store=online_store,
                    source_table_name=table_name,
                    online_table_name=online_table_name
                )
                print(f"  ✅ Successfully published to online store as {online_table_name}")
                
            except Exception as e:
                print(f"  ❌ Failed to publish {table_name}: {str(e)}")
                # Continue with other tables
                continue
                
        except Exception as e:
            print(f"❌ Error processing {table_name}: {str(e)}")
            continue
    
    print("🎉 Databricks Online Feature Store setup completed!")
    print(f"💡 Store state: {online_store.state}")
    if online_store.state != "AVAILABLE":
        print("⏳ Note: Online store may take a few minutes to become fully available")


def deploy(model_uri, env):
    """
    Deploy the model to Unity Catalog registry by setting an alias
    Also ensures online feature store is enabled for real-time inference
    
    :param model_uri: URI of the model in format "models://<name>/<version>" or just "<name>"
    :param env: Environment (dev, staging, prod)
    """
    # Step 1: Enable online feature store for required tables
    print("Step 1: Enabling online feature store...")
    enable_online_feature_store()
    
    # Step 2: Set model alias for deployment
    print("Step 2: Setting model alias for deployment...")
    alias = get_deployed_model_alias_for_env(env)
    
    client = MlflowClient()
    
    # Parse model URI to extract model name and version
    if model_uri.startswith("models:/"):
        # Format: "models://<name>/<version>"
        uri_parts = model_uri.replace("models:/", "").split("/")
        model_name = uri_parts[0]
        model_version = uri_parts[1] if len(uri_parts) > 1 else "1"
    else:
        # Assume it's just the model name
        model_name = model_uri
        model_version = "1"
    
    print(f"Setting alias '{alias}' for model '{model_name}' version {model_version}")
    
    # Set the alias for model serving
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version
    )
    
    print(f"Successfully deployed model {model_name} (version {model_version}) with alias '{alias}' for environment '{env}'")
    print(f"Model is now ready for real-time inference with online feature store enabled.")


if __name__ == "__main__":
    deploy(model_uri=sys.argv[1], env=sys.argv[2])
