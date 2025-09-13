"""
Training utility functions extracted from TrainWithFeatureStore notebook.
This allows for proper type checking and linting of the core functions.
"""

from mlflow.tracking import MlflowClient


def get_latest_model_version(model_name: str) -> int:
    """
    Get the latest version number for a given model.

    Args:
        model_name: Name of the MLflow registered model

    Returns:
        Latest version number of the model
    """
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        latest_version = max(latest_version, version_int)
    return latest_version


def create_model_uri(model_name: str, model_version: int) -> str:
    """
    Create a model URI for MLflow model loading.

    Args:
        model_name: Name of the registered model
        model_version: Version number of the model

    Returns:
        Model URI string in the format "models://{name}/{version}"
    """
    return f"models:/{model_name}/{model_version}"
