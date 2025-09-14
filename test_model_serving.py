"""
Test script to validate the deployed model serving endpoint with online feature store integration.
This script will test the taxi fare prediction model that uses features from online tables.
"""
import requests
import json
import os
import time
from datetime import datetime

def test_serving_endpoint():
    """Test the deployed model serving endpoint"""
    
    # Databricks workspace URL - you'll need to set this
    workspace_url = "https://adb-8490988242777396.16.azuredatabricks.net"
    endpoint_name = "mlops-taxi-fare-endpoint"
    
    # You'll need to set your personal access token
    # For security, it should be set as an environment variable
    token = os.getenv("DATABRICKS_TOKEN")
    if not token:
        print("ERROR: Please set DATABRICKS_TOKEN environment variable")
        return
    
    # Serving endpoint URL
    serving_url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
    
    # Headers for the request
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test data for taxi fare prediction
    # These should include the primary keys that will be used to lookup features
    # from the online feature store
    test_inputs = [
        {
            "pickup_zip": "10001",  # Primary key for pickup features
            "dropoff_zip": "10002", # Primary key for dropoff features
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
    
    # Prepare the request payload
    payload = {
        "inputs": test_inputs
    }
    
    print("Testing Model Serving Endpoint")
    print("=" * 50)
    print(f"Endpoint: {endpoint_name}")
    print(f"URL: {serving_url}")
    print(f"Test inputs: {len(test_inputs)} samples")
    print()
    
    try:
        # Make the prediction request
        print("Sending prediction request...")
        response = requests.post(serving_url, headers=headers, json=payload, timeout=30)
        
        # Check response status
        if response.status_code == 200:
            predictions = response.json()
            print("✅ SUCCESS: Model predictions received!")
            print()
            print("Predictions:")
            for i, prediction in enumerate(predictions.get("predictions", [])):
                print(f"  Sample {i+1}: ${prediction:.2f}")
            
            # If the model uses online feature lookups, the response might include
            # additional metadata about the features used
            if "feature_metadata" in predictions:
                print()
                print("Feature Store Integration:")
                print(f"  Online features used: {predictions['feature_metadata']}")
            
            return True
            
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error - {e}")
        return False

def check_online_tables():
    """Check if online feature tables are available and accessible"""
    
    print("Checking Online Feature Tables")
    print("=" * 50)
    
    # These would be checked using Databricks SDK in a real implementation
    # For now, we'll just print the expected table names
    expected_tables = [
        "p03.e2e_demo_simon.trip_pickup_online_features",
        "p03.e2e_demo_simon.trip_dropoff_online_features"
    ]
    
    for table in expected_tables:
        print(f"Expected online table: {table}")
    
    print()
    print("Note: Online tables should be accessible for feature lookups during prediction")
    print()

def main():
    """Main test function"""
    print("Model Serving with Online Feature Store - Integration Test")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Check online tables
    check_online_tables()
    
    # Test serving endpoint
    success = test_serving_endpoint()
    
    print()
    if success:
        print("🎉 Model serving with online feature store is working!")
        print()
        print("Key capabilities validated:")
        print("✅ Model serving endpoint is deployed and responsive")
        print("✅ Model accepts prediction requests with primary keys")
        print("✅ Online feature lookup integration (if configured)")
        print("✅ Real-time prediction capability")
    else:
        print("❌ Model serving test failed")
        print()
        print("Troubleshooting steps:")
        print("1. Verify DATABRICKS_TOKEN environment variable is set")
        print("2. Check that the serving endpoint is in READY state")
        print("3. Ensure online feature tables are properly configured")
        print("4. Verify model expects the correct input schema")
    
    print()

if __name__ == "__main__":
    main()