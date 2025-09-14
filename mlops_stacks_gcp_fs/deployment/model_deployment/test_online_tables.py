"""
Test script for Online Tables deployment functionality.
This script validates that the online tables and serving endpoint work correctly.
"""

import os
import sys
import json
from datetime import datetime

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..", "..")
sys.path.append(project_root)

def test_online_tables_deployment():
    """
    Test the online tables deployment functionality.
    """
    try:
        from deploy import deploy_with_online_tables
        
        print("🧪 Testing Online Tables Deployment")
        print("=" * 50)
        
        # Test parameters
        test_model_uri = "p03.e2e_demo_simon.taxi_fare_regressor/15"  # Use your latest model
        test_env = "staging"
        
        print(f"📋 Test Parameters:")
        print(f"   Model URI: {test_model_uri}")
        print(f"   Environment: {test_env}")
        print()
        
        # Run deployment
        result = deploy_with_online_tables(test_model_uri, test_env)
        
        print("\n✅ Deployment test completed!")
        print("📄 Result:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment test failed: {str(e)}")
        return False


def test_endpoint_inference():
    """
    Test inference on the deployed endpoint.
    """
    try:
        import mlflow.deployments
        
        print("\n🧪 Testing Endpoint Inference")
        print("=" * 40)
        
        # Test endpoint name - adjust based on your deployment
        endpoint_name = "p03_e2e_demo_simon_taxi_fare_regressor_endpoint_staging"
        
        print(f"📡 Testing endpoint: {endpoint_name}")
        
        # Create deployment client
        client = mlflow.deployments.get_deploy_client("databricks")
        
        # Sample test data with lookup keys and real-time features
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
            }
        ]
        
        print(f"📤 Sending test request...")
        
        response = client.predict(
            endpoint=endpoint_name,
            inputs={
                "dataframe_records": test_data
            }
        )
        
        print("✅ Inference test successful!")
        print("📥 Response:")
        print(json.dumps(response, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {str(e)}")
        print("💡 This may be expected if the endpoint is still starting up.")
        return False


def main():
    """
    Run all tests.
    """
    print("🚀 Online Tables Deployment Test Suite")
    print("=" * 60)
    print(f"🕒 Started at: {datetime.now()}")
    print()
    
    # Test 1: Deployment
    deployment_success = test_online_tables_deployment()
    
    # Test 2: Inference (only if deployment succeeded)
    if deployment_success:
        inference_success = test_endpoint_inference()
    else:
        inference_success = False
        print("⏭️  Skipping inference test due to deployment failure")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"🏗️  Deployment Test: {'✅ PASSED' if deployment_success else '❌ FAILED'}")
    print(f"🧪 Inference Test: {'✅ PASSED' if inference_success else '❌ FAILED'}")
    
    if deployment_success and inference_success:
        print("\n🎉 All tests passed! Online tables deployment is working correctly.")
    elif deployment_success:
        print("\n⚠️  Deployment successful but inference test failed.")
        print("💡 The endpoint may need a few minutes to become ready.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
    
    print(f"\n🕒 Completed at: {datetime.now()}")
    return deployment_success and inference_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)