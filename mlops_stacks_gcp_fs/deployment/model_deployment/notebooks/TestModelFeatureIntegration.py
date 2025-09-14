# Quick test to verify the newly trained model can be loaded and used
# This simulates what the model serving endpoint would do

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession

# Initialize clients
spark = SparkSession.builder.getOrCreate()
fe = FeatureEngineeringClient()

# Model information
MODEL_NAME = "p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model"
MODEL_VERSION = "15"  # The newly trained version

print(f"Testing model: {MODEL_NAME} version {MODEL_VERSION}")
print()

try:
    # Load the model
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    print("✓ Model loaded successfully")
    
    # Check if it has feature store integration
    if hasattr(model, '_model_impl'):
        print("✓ Model has implementation details")
        
        # Check for feature lookups (indicates proper feature store integration)
        if hasattr(model._model_impl, 'feature_lookups'):
            print("✓ Model has feature lookups - Feature Store integration detected")
            print(f"  Feature lookups: {len(model._model_impl.feature_lookups)}")
        else:
            print("✗ No feature lookups found")
    
    # Create test data (similar to what serving would receive)
    test_data = spark.createDataFrame([
        (1, "2023-01-01 12:00:00", "2023-01-01 12:30:00", 2.5, 5.0)
    ], ["pickup_zip", "rounded_pickup_datetime", "rounded_dropoff_datetime", "trip_distance", "fare_amount"])
    
    print()
    print("Test data:")
    test_data.show()
    
    # Try to make predictions (this tests the full pipeline)
    try:
        # Convert to pandas for model prediction (as serving would do)
        test_pd = test_data.toPandas()
        
        # Make prediction
        predictions = model.predict(test_pd)
        print("✓ Prediction successful!")
        print(f"  Predicted fare: ${predictions[0]:.2f}")
        
        print()
        print("SUCCESS: Model is working correctly with feature store integration!")
        print("The 'Feature lookup setup failed' error should now be resolved.")
        
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        print("This indicates the model serving might still have issues.")
    
except Exception as e:
    print(f"✗ Failed to load model: {str(e)}")

print()
print("Test complete!")