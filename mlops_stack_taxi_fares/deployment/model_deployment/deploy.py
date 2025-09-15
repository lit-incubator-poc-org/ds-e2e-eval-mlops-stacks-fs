import sys
import pathlib
import os
from typing import Dict, List, Tuple, Optional, Any

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from utils import get_deployed_model_alias_for_env
from mlflow.tracking import MlflowClient
from databricks.feature_engineering import FeatureEngineeringClient


# ============================================================================
# PUBLIC FUNCTIONS (API Interface)
# ============================================================================

def create_online_tables(catalog_name: str = "p03", schema_name: str = "e2e_demo_simon") -> List[str]:
    """
    Create Databricks Online Tables for feature tables required by the model.
    Uses the modern Online Tables approach following Unity Catalog best practices.
    
    :param catalog_name: Unity Catalog name
    :param schema_name: Schema name containing feature tables
    :return: List of created online table names
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
    
    # Initialize WorkspaceClient using Databricks CLI configuration
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
    
    print("INFO: Setting up Databricks Online Tables for real-time feature serving...")
    
    online_tables_created = []
    
    for table_config in feature_tables:
        source_table = table_config["source"]
        online_table_name = table_config["online"]
        primary_keys = table_config["primary_keys"]
        
        try:
            print(f"INFO: Processing {source_table}...")
            print(f"   Creating online table: {online_table_name}")
            
            # Check if online table already exists
            try:
                existing_table = workspace.online_tables.get(online_table_name)
                print(f"  SUCCESS: Online table already exists (Status: {existing_table.status})")
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
            print(f"  PROGRESS: Creating online table...")
            result = workspace.online_tables.create_and_wait(table=online_table)
            
            print(f"  SUCCESS: Online table created successfully!")
            print(f"  Status: {result.status}")
            online_tables_created.append(online_table_name)
            
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  SUCCESS: Online table already exists")
                online_tables_created.append(online_table_name)
            else:
                print(f"  ERROR: Failed to create online table: {str(e)}")
                # Continue with other tables
                continue
    
    print(f"SUCCESS: Online Tables setup completed!")
    print(f"Created/verified {len(online_tables_created)} online tables:")
    for table in online_tables_created:
        print(f"  - {table}")
    
    return online_tables_created


def get_latest_model_version(client: MlflowClient, model_name: str) -> str:
    """Get the latest version of the model"""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_version = str(max(int(v.version) for v in versions))
            return latest_version
        else:
            raise ValueError(f"No versions found for model {model_name}")
    except Exception as e:
        print(f"WARNING: Could not determine latest model version: {str(e)}")
        return "1"  # Fallback to version 1


def deploy_with_online_tables(model_uri: str, env: str) -> Dict[str, Any]:
    """
    Deploy the model to Unity Catalog registry with Online Tables for real-time inference.
    Creates serving endpoint with automatic feature lookup capabilities.
    
    :param model_uri: URI of the model in format "models://<name>/<version>" or just "<name>"
    :param env: Environment (dev, staging, prod)
    :return: Deployment summary dictionary
    """
    from databricks.sdk import WorkspaceClient
    
    # Initialize clients
    workspace = WorkspaceClient()
    
    import mlflow
    mlflow.set_tracking_uri("databricks")
    client = MlflowClient()
    
    # Parse model URI and extract components
    model_name, model_version = _parse_model_uri(model_uri, client)
    catalog_name, schema_name = _extract_catalog_schema(model_name)
    
    print(f"INFO: Deploying model: {model_name} (version {model_version})")
    print(f"INFO: Environment: {env}")
    
    # Step 1: Create online tables for feature serving
    print("\nStep 1: Setting up Online Tables...")
    online_tables = create_online_tables(catalog_name, schema_name)
    
    # Step 2: Set model alias for deployment tracking
    alias = _set_model_alias(client, model_name, model_version, env)
    
    # Step 3: Create/Update serving endpoint
    endpoint_name = "nytaxifares"
    endpoint_config = _create_endpoint_config(model_name, model_version, catalog_name, schema_name)
    _create_or_update_endpoint(workspace, endpoint_name, endpoint_config, model_version, env)
    
    # Step 4: Verify endpoint status
    _verify_endpoint_status(workspace, endpoint_name)
    
    # Step 5: Wait for endpoint to be fully ready
    _wait_for_endpoint_ready(workspace, endpoint_name)
    
    # Print deployment summary
    _print_deployment_summary(workspace, model_name, model_version, env, alias, endpoint_name, online_tables)
    
    return {
        "model_name": model_name,
        "model_version": model_version,
        "endpoint_name": endpoint_name,
        "online_tables": online_tables,
        "alias": alias
    }


def deploy(model_uri: str, env: str) -> Dict[str, Any]:
    """
    Legacy deploy function - now redirects to new online tables deployment.
    Maintained for backward compatibility.
    
    :param model_uri: URI of the model in format "models://<name>/<version>" or just "<name>"
    :param env: Environment (dev, staging, prod)
    :return: Deployment summary dictionary
    """
    return deploy_with_online_tables(model_uri, env)


# ============================================================================
# PRIVATE UTILITY FUNCTIONS (Implementation Details - in call order)
# ============================================================================

def _parse_model_uri(model_uri: str, client: MlflowClient) -> Tuple[str, str]:
    """
    Parse model URI and return model name and version.
    
    :param model_uri: URI of the model in format "models://<name>/<version>" or just "<name>"
    :param client: MLflow client for version lookup
    :return: tuple of (model_name, model_version)
    """
    if model_uri.startswith("models:/"):
        # Format: "models://<name>/<version>"
        uri_parts = model_uri.replace("models:/", "").split("/")
        model_name = uri_parts[0]
        model_version = uri_parts[1] if len(uri_parts) > 1 else get_latest_model_version(client, model_name)
    else:
        # Assume it's just the model name
        model_name = model_uri
        model_version = get_latest_model_version(client, model_name)
    
    return model_name, model_version


def _extract_catalog_schema(model_name: str) -> Tuple[str, str]:
    """
    Extract catalog and schema names from model name.
    
    :param model_name: Full model name like "catalog.schema.model_name"
    :return: tuple of (catalog_name, schema_name)
    """
    model_parts = model_name.split(".")
    if len(model_parts) >= 2:
        catalog_name = model_parts[0]
        schema_name = model_parts[1]
    else:
        # Fallback defaults
        catalog_name = "p03"
        schema_name = "e2e_demo_simon"
    
    return catalog_name, schema_name


def _set_model_alias(client: MlflowClient, model_name: str, model_version: str, env: str) -> str:
    """
    Set model alias for deployment tracking.
    
    :param client: MLflow client
    :param model_name: Model name
    :param model_version: Model version
    :param env: Environment (dev, staging, prod)
    :return: alias name that was set
    """
    print("\nStep 2: Setting model alias for deployment...")
    alias = get_deployed_model_alias_for_env(env)
    
    print(f"Setting alias '{alias}' for model '{model_name}' version {model_version}")
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version
    )
    print(f"SUCCESS: Model alias '{alias}' set successfully")
    
    return alias


def _create_endpoint_config(model_name: str, model_version: str, catalog_name: str, schema_name: str) -> Any:
    """
    Create endpoint configuration for serving.
    
    :param model_name: Model name
    :param model_version: Model version
    :param catalog_name: Unity Catalog name
    :param schema_name: Schema name
    :return: EndpointCoreConfigInput object
    """
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput, 
        ServedEntityInput, 
        AutoCaptureConfigInput, 
        ServedModelInputWorkloadType
    )
    
    return EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=model_name,
                entity_version=model_version,
                scale_to_zero_enabled=True,
                workload_size="Small",
                workload_type=ServedModelInputWorkloadType.CPU
            )
        ],
        auto_capture_config=AutoCaptureConfigInput(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name_prefix="nytaxifares_endpoint"
        )
    )


def _create_or_update_endpoint(workspace: Any, endpoint_name: str, endpoint_config: Any, model_version: str, env: str) -> None:
    """
    Create or update serving endpoint.
    
    :param workspace: Databricks WorkspaceClient
    :param endpoint_name: Name of the endpoint
    :param endpoint_config: Endpoint configuration
    :param model_version: Model version for tagging
    :param env: Environment for tagging
    """
    print("\nStep 3: Creating/Updating serving endpoint...")
    
    try:
        # Check if endpoint already exists
        try:
            existing_endpoint = workspace.serving_endpoints.get(endpoint_name)
            ready_status = existing_endpoint.state.ready.value if hasattr(existing_endpoint.state.ready, 'value') else str(existing_endpoint.state.ready)
            config_status = existing_endpoint.state.config_update.value if hasattr(existing_endpoint.state.config_update, 'value') else str(existing_endpoint.state.config_update)
            print(f"INFO: Endpoint '{endpoint_name}' exists (Ready: {ready_status}, Config: {config_status})")
            
            # Update existing endpoint with new model version
            print(f"PROGRESS: Updating endpoint with model version {model_version}...")
            update_status = workspace.serving_endpoints.update_config_and_wait(
                name=endpoint_name,
                served_entities=endpoint_config.served_entities,
                auto_capture_config=endpoint_config.auto_capture_config
            )
            update_ready = update_status.state.ready.value if hasattr(update_status.state.ready, 'value') else str(update_status.state.ready)
            update_config = update_status.state.config_update.value if hasattr(update_status.state.config_update, 'value') else str(update_status.state.config_update)
            print(f"SUCCESS: Endpoint updated successfully! (Ready: {update_ready}, Config: {update_config})")
            
        except Exception as get_error:
            # Endpoint doesn't exist, create new one
            print(f"PROGRESS: Creating new endpoint: {endpoint_name}")
            
            # Add tags for the new endpoint
            endpoint_tags = [
                {"key": "environment", "value": env},
                {"key": "project", "value": "mlops-taxi-fare"},
                {"key": "model_name", "value": endpoint_config.served_entities[0].entity_name.split(".")[-1]},
                {"key": "model_version", "value": model_version}
            ]
            
            create_status = workspace.serving_endpoints.create_and_wait(
                name=endpoint_name,
                config=endpoint_config,
                tags=endpoint_tags
            )
            create_ready = create_status.state.ready.value if hasattr(create_status.state.ready, 'value') else str(create_status.state.ready)
            create_config = create_status.state.config_update.value if hasattr(create_status.state.config_update, 'value') else str(create_status.state.config_update)
            print(f"SUCCESS: Endpoint created successfully! (Ready: {create_ready}, Config: {create_config})")
            
    except Exception as e:
        print(f"ERROR: Failed to create/update endpoint: {str(e)}")
        raise e


def _verify_endpoint_status(workspace: Any, endpoint_name: str) -> None:
    """
    Verify endpoint status and readiness.
    
    :param workspace: Databricks WorkspaceClient
    :param endpoint_name: Name of the endpoint
    """
    print("\nStep 4: Verifying endpoint status...")
    try:
        final_endpoint = workspace.serving_endpoints.get(endpoint_name)
        final_ready = final_endpoint.state.ready.value if hasattr(final_endpoint.state.ready, 'value') else str(final_endpoint.state.ready)
        final_config = final_endpoint.state.config_update.value if hasattr(final_endpoint.state.config_update, 'value') else str(final_endpoint.state.config_update)
        print(f"SUCCESS: Endpoint Status - Ready: {final_ready}, Config: {final_config}")
        
        # If endpoint is ready, perform a test inference
        ready_check = final_endpoint.state.ready.value if hasattr(final_endpoint.state.ready, 'value') else str(final_endpoint.state.ready)
        if ready_check == "READY":
            print("SUCCESS: Endpoint is ready for inference!")
        else:
            print("WARNING: Endpoint is still initializing. Monitor status in Databricks UI.")
            
    except Exception as e:
        print(f"WARNING: Could not verify endpoint status: {str(e)}")


def _wait_for_endpoint_ready(workspace: Any, endpoint_name: str, max_wait_time: int = 300) -> None:
    """
    Wait for endpoint to be fully ready with timeout.
    
    :param workspace: Databricks WorkspaceClient
    :param endpoint_name: Name of the endpoint
    :param max_wait_time: Maximum wait time in seconds (default: 5 minutes)
    """
    print("\nStep 5: Waiting for endpoint to be fully ready...")
    import time
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            endpoint_status = workspace.serving_endpoints.get(endpoint_name)
            status_ready = endpoint_status.state.ready.value if hasattr(endpoint_status.state.ready, 'value') else str(endpoint_status.state.ready)
            status_config = endpoint_status.state.config_update.value if hasattr(endpoint_status.state.config_update, 'value') else str(endpoint_status.state.config_update)
            
            if (status_ready == "READY" and status_config == "NOT_UPDATING"):
                print("SUCCESS: Endpoint is fully ready for serving!")
                break
            else:
                print(f"PROGRESS: Endpoint status - Ready: {status_ready}, Config: {status_config}")
                time.sleep(30)  # Wait 30 seconds before checking again
        except Exception as e:
            print(f"WARNING: Error checking endpoint status: {str(e)}")
            time.sleep(30)
    else:
        print("WARNING: Endpoint may still be initializing after 5 minutes. Check Databricks UI for status.")


def _print_deployment_summary(workspace: Any, model_name: str, model_version: str, env: str, alias: str, endpoint_name: str, online_tables: List[str]) -> None:
    """
    Print deployment success summary.
    
    :param workspace: Databricks WorkspaceClient
    :param model_name: Model name
    :param model_version: Model version
    :param env: Environment
    :param alias: Model alias
    :param endpoint_name: Endpoint name
    :param online_tables: List of online tables created
    """
    print("\n" + "="*60)
    print("SUCCESS: DEPLOYMENT SUCCESSFUL")
    print("="*60)
    print(f"SUCCESS: Model: {model_name} (v{model_version})")
    print(f"SUCCESS: Environment: {env}")
    print(f"SUCCESS: Model Alias: {alias}")
    print(f"SUCCESS: Serving Endpoint: {endpoint_name}")
    print(f"SUCCESS: Online Tables: {len(online_tables)} tables configured")
    
    for table in online_tables:
        print(f"   - {table}")
    
    print(f"\nINFO: Endpoint URL: https://{workspace.config.host}/ml/endpoints/{endpoint_name}")
    print("TIP: Model is ready for real-time inference with automatic feature lookup!")


if __name__ == "__main__":
    deploy(model_uri=sys.argv[1], env=sys.argv[2])
