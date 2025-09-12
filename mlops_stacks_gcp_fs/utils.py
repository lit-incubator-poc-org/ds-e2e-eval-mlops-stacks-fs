"""This module contains utils shared between different notebooks"""
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from datetime import timedelta, timezone
import math


def get_deployed_model_stage_for_env(env):
    """
    Get the model version stage under which the latest deployed model version can be found
    for the current environment (Legacy workspace model registry)
    :param env: Current environment
    :return: Model version stage
    """
    # For a registered model version to be served, it needs to be in either the Staging or Production
    # model registry stage
    # (https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#transition-a-model-stage).
    # For models in dev and staging environments, we deploy the model to the "Staging" stage, and in prod we deploy to the
    # "Production" stage
    _MODEL_STAGE_FOR_ENV = {
        "dev": "Staging",
        "staging": "Staging",
        "prod": "Production",
        "test": "Production",
    }
    return _MODEL_STAGE_FOR_ENV[env]


def get_deployed_model_alias_for_env(env):
    """
    Get the model alias for Unity Catalog model deployment for the current environment
    :param env: Current environment
    :return: Model alias for Unity Catalog
    """
    # For Unity Catalog, we use aliases instead of stages for model deployment
    # Use environment-specific aliases for better deployment management
    _MODEL_ALIAS_FOR_ENV = {
        "dev": "dev",
        "staging": "staging", 
        "prod": "prod",
        "test": "test",
    }
    return _MODEL_ALIAS_FOR_ENV[env]


# Shared data preprocessing utilities
def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    
    Args:
        dt: datetime object
        num_minutes: interval in minutes to round to
    
    Returns:
        Unix timestamp as integer
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


# Create UDF for use in Spark
rounded_unix_timestamp_udf = F.udf(rounded_unix_timestamp, IntegerType())


def add_rounded_timestamps(df, pickup_minutes=15, dropoff_minutes=30):
    """
    Add rounded timestamp columns to taxi data for feature store lookups.
    
    Args:
        df: Input DataFrame with tpep_pickup_datetime and tpep_dropoff_datetime columns
        pickup_minutes: Minutes to round pickup timestamp to (default: 15)
        dropoff_minutes: Minutes to round dropoff timestamp to (default: 30)
    
    Returns:
        DataFrame with added rounded_pickup_datetime and rounded_dropoff_datetime columns
    """
    return (
        df.withColumn(
            "rounded_pickup_datetime",
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    df["tpep_pickup_datetime"], F.lit(pickup_minutes)
                )
            ),
        )
        .withColumn(
            "rounded_dropoff_datetime", 
            F.to_timestamp(
                rounded_unix_timestamp_udf(
                    df["tpep_dropoff_datetime"], F.lit(dropoff_minutes)
                )
            ),
        )
    )
