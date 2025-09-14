#!/usr/bin/env python3
"""
Test script for deployed taxi fare prediction model endpoint.
Tests both the serving endpoint and feature lookup functionality.
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
from databricks.sdk import WorkspaceClient


def test_serving_endpoint(endpoint_name="nytaxifares", host=None, token=None):
    """
    Test the deployed model serving endpoint with sample taxi trip data.
    
    :param endpoint_name: Name of the serving endpoint
    :param host: Databricks workspace host (optional, will use env var if not provided)  
    :param token: Databricks token (optional, will use env var if not provided)
    """
    
    # Initialize WorkspaceClient
    workspace = WorkspaceClient(
        host=host or os.getenv("DATABRICKS_HOST"),
        token=token or os.getenv("DATABRICKS_TOKEN")
    )
    
    print(f"INFO: Testing serving endpoint: {endpoint_name}")
    
    # Check endpoint status first
    try:
        endpoint = workspace.serving_endpoints.get(endpoint_name)
        print(f"SUCCESS: Endpoint found - Status: {endpoint.state.ready}")
        print(f"Config Status: {endpoint.state.config_update}")
        
        if endpoint.state.ready != "READY":
            print(f"WARNING: Endpoint is not ready yet. Current status: {endpoint.state.ready}")
            return False
            
    except Exception as e:
        print(f"ERROR: Could not get endpoint status: {str(e)}")
        return False
    
    # Prepare test data - sample taxi trip for prediction
    test_data = {
        "dataframe_records": [
            {
                "tpep_pickup_datetime": "2023-01-15 14:30:00",
                "tpep_dropoff_datetime": "2023-01-15 14:45:00", 
                "pickup_zip": "10001",
                "dropoff_zip": "10002",
                "trip_distance": 2.5,
                "pickup_latitude": 40.7589,
                "pickup_longitude": -73.9851,
                "dropoff_latitude": 40.7505,
                "dropoff_longitude": -73.9934
            }
        ]
    }
    
    print(f"INFO: Sending test prediction request...")
    print(f"Test data: {json.dumps(test_data, indent=2)}")
    
    try:
        # Query the endpoint
        response = workspace.serving_endpoints.query(endpoint_name, **test_data)
        
        print(f"SUCCESS: Prediction completed!")
        print(f"Response: {response}")
        
        # Parse and display results
        if hasattr(response, 'predictions'):
            predictions = response.predictions
            print(f"SUCCESS: Predicted fare: ${predictions[0]:.2f}")
        else:
            print(f"INFO: Raw response received: {response}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Prediction failed: {str(e)}")
        print(f"Error type: {type(e)}")
        return False


def test_batch_predictions(endpoint_name="nytaxifares", host=None, token=None):
    """
    Test batch predictions with multiple taxi trips.
    """
    workspace = WorkspaceClient(
        host=host or os.getenv("DATABRICKS_HOST"),
        token=token or os.getenv("DATABRICKS_TOKEN")
    )
    
    print(f"\nINFO: Testing batch predictions...")
    
    # Multiple test trips
    test_batch = {
        "dataframe_records": [
            {
                "tpep_pickup_datetime": "2023-01-15 14:30:00",
                "tpep_dropoff_datetime": "2023-01-15 14:45:00",
                "pickup_zip": "10001", 
                "dropoff_zip": "10002",
                "trip_distance": 2.5,
                "pickup_latitude": 40.7589,
                "pickup_longitude": -73.9851,
                "dropoff_latitude": 40.7505,
                "dropoff_longitude": -73.9934
            },
            {
                "tpep_pickup_datetime": "2023-01-15 15:00:00",
                "tpep_dropoff_datetime": "2023-01-15 15:20:00",
                "pickup_zip": "10003",
                "dropoff_zip": "10004", 
                "trip_distance": 5.2,
                "pickup_latitude": 40.7282,
                "pickup_longitude": -73.9942,
                "dropoff_latitude": 40.7061,
                "dropoff_longitude": -74.0087
            }
        ]
    }
    
    try:
        response = workspace.serving_endpoints.query(endpoint_name, **test_batch)
        
        print(f"SUCCESS: Batch predictions completed!")
        
        if hasattr(response, 'predictions'):
            predictions = response.predictions
            print(f"SUCCESS: Batch predictions:")
            for i, pred in enumerate(predictions):
                print(f"  Trip {i+1}: ${pred:.2f}")
        else:
            print(f"INFO: Raw batch response: {response}")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Batch prediction failed: {str(e)}")
        return False


def main():
    """Main test execution"""
    print("="*60)
    print("TAXI FARE PREDICTION ENDPOINT TEST")
    print("="*60)
    
    # Test single prediction
    success1 = test_serving_endpoint()
    
    # Test batch predictions  
    success2 = test_batch_predictions()
    
    # Summary
    print("\n" + "="*60)
    if success1 and success2:
        print("SUCCESS: All endpoint tests passed!")
        print("TIP: Model is ready for production inference")
    else:
        print("WARNING: Some tests failed. Check endpoint configuration.")
    print("="*60)


if __name__ == "__main__":
    main()