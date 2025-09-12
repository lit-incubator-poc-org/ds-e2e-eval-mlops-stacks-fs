"""
Shared feature engineering utilities for MLOps pipeline.
Contains common timestamp rounding logic used across training and inference.

Note: UDF creation must be done in the calling notebooks to avoid 
Spark serialization issues in distributed environments.
"""
from datetime import timedelta, timezone
import math
import pyspark.sql.functions as F


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    
    Args:
        dt: datetime object
        num_minutes: interval in minutes to round to (default: 15)
    
    Returns:
        Unix timestamp as integer
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())


def create_add_rounded_timestamps_function(rounded_unix_timestamp_udf):
    """
    Factory function that creates the add_rounded_timestamps function with a given UDF.
    
    This pattern allows notebooks to create their own UDFs locally and pass them to this function,
    avoiding Spark serialization issues with UDFs defined in imported modules.
    
    Args:
        rounded_unix_timestamp_udf: A Spark UDF created from rounded_unix_timestamp function
    
    Returns:
        Function that adds rounded timestamp columns to a DataFrame
    """
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
    
    return add_rounded_timestamps


def add_rounded_timestamps_direct(df, rounded_unix_timestamp_udf, pickup_minutes=15, dropoff_minutes=30):
    """
    Direct function to add rounded timestamp columns to taxi data for feature store lookups.
    
    This is an alternative to the factory pattern above - use whichever feels more natural.
    
    Args:
        df: Input DataFrame with tpep_pickup_datetime and tpep_dropoff_datetime columns
        rounded_unix_timestamp_udf: A Spark UDF created from rounded_unix_timestamp function
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