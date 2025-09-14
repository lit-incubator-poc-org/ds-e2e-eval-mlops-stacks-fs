#!/usr/bin/env python3
"""
Create or update a model serving endpoint with Unity Catalog feature store support.

This script ensures the serving endpoint has the required databricks-feature-engineering
package to support automatic feature lookup from the online feature store.
"""

import json
import subprocess
import sys
import time

# Configuration
MODEL_NAME = "p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model"
MODEL_VERSION = "15"
ENDPOINT_NAME = "nytaxifare"

def run_command(command, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    if result.returncode != 0 and check:
        print(f"Error running command: {result.stderr}")
        sys.exit(1)
    
    return result

def create_serving_endpoint():
    """Create or update the model serving endpoint."""
    
    # Define the endpoint configuration with Unity Catalog feature store support
    endpoint_config = {
        "name": ENDPOINT_NAME,
        "served_models": [
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
                "workload_type": "CPU"
            }
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": f"{MODEL_NAME}-{MODEL_VERSION}",
                    "traffic_percentage": 100
                }
            ]
        }
    }
    
    # Write config to temporary file
    config_file = "/tmp/serving_endpoint_config.json"
    with open(config_file, "w") as f:
        json.dump(endpoint_config, f, indent=2)
    
    print(f"Creating/updating serving endpoint: {ENDPOINT_NAME}")
    print(f"Model: {MODEL_NAME} v{MODEL_VERSION}")
    
    # Check if endpoint exists
    check_cmd = ["databricks", "serving-endpoints", "get", ENDPOINT_NAME]
    result = run_command(check_cmd, check=False)
    
    if result.returncode == 0:
        print("Endpoint exists, updating...")
        # Update existing endpoint - need to extract config part only
        update_config = {
            "served_models": endpoint_config["served_models"],
            "traffic_config": endpoint_config["traffic_config"]
        }
        
        # Write update config
        update_config_file = "/tmp/update_config.json"
        with open(update_config_file, "w") as f:
            json.dump(update_config, f, indent=2)
            
        update_cmd = [
            "databricks", "serving-endpoints", "update-config", 
            ENDPOINT_NAME, 
            "--json", f"@{update_config_file}"
        ]
        run_command(update_cmd)
    else:
        print("Creating new endpoint...")
        # Create new endpoint  
        create_cmd = [
            "databricks", "serving-endpoints", "create",
            "--json", f"@{config_file}"
        ]
        run_command(create_cmd)
    
    print("✅ Endpoint creation/update initiated")
    return config_file

def wait_for_deployment():
    """Wait for the endpoint deployment to complete."""
    print("Waiting for deployment to complete...")
    
    max_wait = 600  # 10 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # Check endpoint status
        status_cmd = ["databricks", "serving-endpoints", "get", ENDPOINT_NAME]
        result = run_command(status_cmd, check=False)
        
        if result.returncode == 0:
            try:
                endpoint_info = json.loads(result.stdout)
                state = endpoint_info.get("state", {}).get("config_update", "UNKNOWN")
                
                print(f"Deployment status: {state}")
                
                if state == "UPDATE_SUCCEEDED":
                    print("✅ Deployment completed successfully!")
                    return True
                elif state == "UPDATE_FAILED":
                    print("❌ Deployment failed!")
                    print(json.dumps(endpoint_info.get("state", {}), indent=2))
                    return False
                    
            except json.JSONDecodeError:
                print("Error parsing endpoint status")
        
        time.sleep(30)
    
    print("⏰ Deployment timeout reached")
    return False

def test_endpoint():
    """Test the serving endpoint with a sample request."""
    print("Testing serving endpoint...")
    
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
    
    # Write test data to file
    test_file = "/tmp/test_data.json"
    with open(test_file, "w") as f:
        json.dump(test_data, f, indent=2)
    
    # Query the endpoint
    query_cmd = [
        "databricks", "serving-endpoints", "query",
        ENDPOINT_NAME,
        "--json", f"@{test_file}"
    ]
    
    result = run_command(query_cmd, check=False)
    
    if result.returncode == 0:
        print("✅ Serving endpoint test successful!")
        print(f"Response: {result.stdout}")
        return True
    else:
        print("❌ Serving endpoint test failed!")
        print(f"Error: {result.stderr}")
        return False

def main():
    """Main execution function."""
    print("🚀 Setting up Model Serving with Unity Catalog Feature Store")
    print("=" * 60)
    
    try:
        # Step 1: Create/update endpoint
        config_file = create_serving_endpoint()
        
        # Step 2: Wait for deployment
        if wait_for_deployment():
            # Step 3: Test endpoint
            if test_endpoint():
                print("\n🎉 SUCCESS!")
                print("Model serving is now working with Unity Catalog feature store!")
                print("The 'Feature lookup setup failed' error should be resolved.")
            else:
                print("\n⚠️ Endpoint deployed but test failed")
                print("Check the endpoint logs for feature lookup issues")
        else:
            print("\n❌ Deployment failed")
            print("Check Databricks workspace for error details")
            
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)
    
    finally:
        # Cleanup temp files
        import os
        for temp_file in ["/tmp/serving_endpoint_config.json", "/tmp/test_data.json"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    main()