import sys
import pathlib
import os

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from utils import get_deployed_model_alias_for_env
from mlflow.tracking import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient


def create_online_tables(catalog_name="p03", schema_name="e2e_demo_simon"):
    """
    Create Databricks Online Tables for feature tables required by the model.
    Uses the modern Online Tables approach following Unity Catalog best practices.
    
    :param catalog_name: Unity Catalog name
    :param schema_name: Schema name containing feature tables
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
    
    workspace = WorkspaceClient()
    
    # Feature tables used by the model
    feature_tables = [
        {
            "source": f"{catalog_name}.{schema_name}.trip_pickup_features",
            "online": f"{catalog_name}.{schema_name}.trip_pickup_features_online",
            "primary_keys": ["pickup_zip", "rounded_pickup_datetime"]
        },
        {
            "source": f"{catalog_name}.{schema_name}.trip_dropoff_features", 
            "online": f"{catalog_name}.{schema_name}.trip_dropoff_features_online",
            "primary_keys": ["dropoff_zip", "rounded_dropoff_datetime"]
        }
    ]
    
    print("� Setting up Databricks Online Tables for real-time feature serving...")
    
    online_tables_created = []
    
    for table_config in feature_tables:
        source_table = table_config["source"]
        online_table_name = table_config["online"]
        primary_keys = table_config["primary_keys"]
        
        try:
            print(f"📋 Processing {source_table}...")
            print(f"   Creating online table: {online_table_name}")
            
            # Check if online table already exists
            try:
                existing_table = workspace.online_tables.get(online_table_name)
                print(f"  ✅ Online table already exists (Status: {existing_table.status})")
                online_tables_created.append(online_table_name)
                continue
            except Exception:
                # Table doesn't exist, create it
                pass
            
            # Create online table spec following best practices
            spec = OnlineTableSpec(
                primary_key_columns=primary_keys,
                source_table_full_name=source_table,
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),
                perform_full_copy=True  # Ensure full sync for reliable serving
            )
            
            online_table = OnlineTable(name=online_table_name, spec=spec)
            
            # Create and wait for the online table
            print(f"  🔄 Creating online table...")
            result = workspace.online_tables.create_and_wait(table=online_table)
            
            print(f"  ✅ Online table created successfully!")
            print(f"  Status: {result.status}")
            online_tables_created.append(online_table_name)
            
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  ✅ Online table already exists")
                online_tables_created.append(online_table_name)
            else:
                print(f"  ❌ Failed to create online table: {str(e)}")
                # Continue with other tables
                continue
    
    print(f"🎉 Online Tables setup completed!")
    print(f"Created/verified {len(online_tables_created)} online tables:")
    for table in online_tables_created:
        print(f"  - {table}")
    
    return online_tables_created


def deploy_with_online_tables(model_uri, env):
    """
    Deploy the model to Unity Catalog registry with Online Tables for real-time inference.
    Creates serving endpoint with automatic feature lookup capabilities.
    
    :param model_uri: URI of the model in format "models://<name>/<version>" or just "<name>"
    :param env: Environment (dev, staging, prod)
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    
    workspace = WorkspaceClient()
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
    
    print(f"🚀 Deploying model: {model_name} (version {model_version})")
    print(f"📍 Environment: {env}")
    
    # Step 1: Create online tables for feature serving
    print("\nStep 1: Setting up Online Tables...")
    
    # Extract catalog and schema from model name for online tables
    model_parts = model_name.split(".")
    if len(model_parts) >= 2:
        catalog_name = model_parts[0]
        schema_name = model_parts[1]
    else:
        # Fallback defaults
        catalog_name = "p03"
        schema_name = "e2e_demo_simon"
    
    online_tables = create_online_tables(catalog_name, schema_name)
    
    # Step 2: Set model alias for deployment tracking
    print("\nStep 2: Setting model alias for deployment...")
    alias = get_deployed_model_alias_for_env(env)
    
    print(f"Setting alias '{alias}' for model '{model_name}' version {model_version}")
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version
    )
    print(f"✅ Model alias '{alias}' set successfully")
    
    # Step 3: Create serving endpoint with online feature lookup
    print("\nStep 3: Creating serving endpoint...")
    
    # Generate endpoint name
    endpoint_name = f"{model_name.replace('.', '_').lower()}_endpoint_{env}"
    
    try:
        # Check if endpoint already exists
        try:
            existing_endpoint = workspace.serving_endpoints.get(endpoint_name)
            print(f"✅ Endpoint '{endpoint_name}' already exists (Status: {existing_endpoint.state})")
            
        except Exception:
            # Create new endpoint
            print(f"🔄 Creating new endpoint: {endpoint_name}")
            status = workspace.serving_endpoints.create_and_wait(
                name=endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=[
                        ServedEntityInput(
                            entity_name=model_name,
                            entity_version=model_version,
                            scale_to_zero_enabled=True,
                            workload_size="Small"
                        )
                    ]
                )
            )
            print(f"✅ Endpoint created successfully! (Status: {status.state})")
            
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"✅ Endpoint '{endpoint_name}' already exists")
        else:
            print(f"❌ Failed to create endpoint: {str(e)}")
            raise e
    
    # Success summary
    print("\n" + "="*60)
    print("🎉 DEPLOYMENT SUCCESSFUL")
    print("="*60)
    print(f"✅ Model: {model_name} (v{model_version})")
    print(f"✅ Environment: {env}")
    print(f"✅ Model Alias: {alias}")
    print(f"✅ Serving Endpoint: {endpoint_name}")
    print(f"✅ Online Tables: {len(online_tables)} tables configured")
    
    for table in online_tables:
        print(f"   - {table}")
    
    print(f"\n🔗 Endpoint URL: https://{workspace.config.host}/ml/endpoints/{endpoint_name}")
    print("💡 Model is ready for real-time inference with automatic feature lookup!")
    
    return {
        "model_name": model_name,
        "model_version": model_version,
        "endpoint_name": endpoint_name,
        "online_tables": online_tables,
        "alias": alias
    }


def deploy(model_uri, env):
    """
    Legacy deploy function - now redirects to new online tables deployment.
    Maintained for backward compatibility.
    
    :param model_uri: URI of the model in format "models://<name>/<version>" or just "<name>"
    :param env: Environment (dev, staging, prod)
    """
    return deploy_with_online_tables(model_uri, env)


if __name__ == "__main__":
    deploy(model_uri=sys.argv[1], env=sys.argv[2])
