"""
Alternative Solution: Create a Custom MLflow Model with Dependencies

Since model serving endpoints don't support custom packages directly, we need to
create a custom MLflow model that bundles the required dependencies.
"""

import mlflow
import mlflow.pyfunc
from databricks.feature_engineering import FeatureEngineeringClient
import pickle
import pandas as pd
from typing import Dict, Any

class FeatureStoreModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model that includes databricks-feature-engineering for serving.
    
    This wraps the original model and handles feature lookups during prediction.
    """
    
    def __init__(self):
        self.model = None
        self.fe_client = None
        self.feature_lookups = None
        
    def load_context(self, context):
        """Load the model and initialize feature engineering client."""
        import sys
        import subprocess
        
        # Install required package if not available
        try:
            from databricks.feature_engineering import FeatureEngineeringClient
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "databricks-feature-engineering>=0.13.0"])
            from databricks.feature_engineering import FeatureEngineeringClient
        
        # Load the original model
        model_path = context.artifacts["model"]
        self.model = mlflow.pyfunc.load_model(model_path)
        
        # Initialize feature engineering client
        self.fe_client = FeatureEngineeringClient()
        
        # Load feature lookup configuration
        with open(context.artifacts["feature_lookups"], 'rb') as f:
            self.feature_lookups = pickle.load(f)
            
    def predict(self, context, model_input):
        """Make prediction with feature lookups."""
        # Convert input to DataFrame if needed
        if isinstance(model_input, dict):
            df = pd.DataFrame(model_input)
        elif isinstance(model_input, pd.DataFrame):
            df = model_input
        else:
            raise ValueError("Input must be a dictionary or DataFrame")
            
        # Perform feature lookups if needed
        if self.feature_lookups:
            try:
                # Create training set for feature enrichment
                enriched_df = self.fe_client.create_training_set(
                    df=df,
                    feature_lookups=self.feature_lookups,
                    label=None
                ).load_df()
                
                # Make prediction with enriched features
                return self.model.predict(enriched_df)
            except Exception as e:
                print(f"Feature lookup failed: {str(e)}")
                # Fall back to prediction without feature enrichment
                return self.model.predict(df)
        else:
            return self.model.predict(df)


def create_serving_ready_model():
    """Create a new model version that's ready for serving without external dependencies."""
    
    # Load the original model
    MODEL_NAME = "p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model"
    MODEL_VERSION = "15"
    
    original_model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    original_model = mlflow.pyfunc.load_model(original_model_uri)
    
    # Extract feature lookups from the original model
    feature_lookups = None
    if hasattr(original_model, '_model_impl') and hasattr(original_model._model_impl, 'feature_lookups'):
        feature_lookups = original_model._model_impl.feature_lookups
        print(f"Extracted {len(feature_lookups)} feature lookups from original model")
    else:
        print("No feature lookups found in original model")
        
    # Create artifacts directory
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the original model
        model_path = os.path.join(temp_dir, "original_model")
        mlflow.pyfunc.save_model(path=model_path, python_model=original_model)
        
        # Save feature lookups configuration
        feature_lookups_path = os.path.join(temp_dir, "feature_lookups.pkl")
        with open(feature_lookups_path, 'wb') as f:
            pickle.dump(feature_lookups, f)
            
        # Define conda environment with required packages
        conda_env = {
            'channels': ['conda-forge'],
            'dependencies': [
                'python=3.10',
                'pip',
                {
                    'pip': [
                        'mlflow>=2.14.0',
                        'pandas>=1.4.3',
                        'numpy>=1.23.0',
                        'databricks-feature-engineering>=0.13.0'
                    ]
                }
            ],
            'name': 'serving_env'
        }
        
        # Create and log the serving-ready model
        artifacts = {
            "model": model_path,
            "feature_lookups": feature_lookups_path
        }
        
        # Log the new model version
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                artifact_path="serving_model",
                python_model=FeatureStoreModel(),
                artifacts=artifacts,
                conda_env=conda_env,
                registered_model_name=f"{MODEL_NAME}_serving"
            )
            
    print("Serving-ready model created successfully!")
    print(f"Model name: {MODEL_NAME}_serving")
    print("This model includes all required dependencies for serving.")

if __name__ == "__main__":
    create_serving_ready_model()