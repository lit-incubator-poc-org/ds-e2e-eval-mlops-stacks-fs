"""
Shared feature engineering utilities for MLOps pipeline.
Uses only Spark built-in functions to avoid UDF serialization issues.
"""

from typing import Union
from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import col, ceil, unix_timestamp, from_unixtime


def add_rounded_timestamps(df: DataFrame, pickup_minutes: int = 15, dropoff_minutes: int = 30) -> DataFrame:
    """
    Add rounded timestamp columns using only Spark built-in functions (no UDFs).

    This approach uses Spark's built-in date/time functions to round timestamps
    to the nearest interval, avoiding the need for custom UDFs entirely.

    Args:
        df: Input DataFrame with tpep_pickup_datetime and tpep_dropoff_datetime columns
        pickup_minutes: Minutes to round pickup timestamp to (default: 15)
        dropoff_minutes: Minutes to round dropoff timestamp to (default: 30)

    Returns:
        DataFrame with added rounded_pickup_datetime and
        rounded_dropoff_datetime columns
        
    """

    def round_timestamp_to_interval(timestamp_col: Union[str, Column], minutes: int) -> Column:
        """
        Round a timestamp column to the nearest interval using built-in Spark functions.

        The logic:
        1. Convert timestamp to Unix timestamp (seconds)
        2. Divide by interval seconds to get fractional intervals
        3. Use ceil() to round up to next interval
        4. Multiply back and convert to timestamp
        """
        interval_seconds = minutes * 60

        # Get Unix timestamp, divide by interval, ceil to round up, multiply back
        rounded_unix = ceil(unix_timestamp(timestamp_col) / interval_seconds) * interval_seconds

        # Convert back to timestamp
        return from_unixtime(rounded_unix).cast("timestamp")

    return df.withColumn(
        "rounded_pickup_datetime",
        round_timestamp_to_interval(col("tpep_pickup_datetime"), pickup_minutes),
    ).withColumn(
        "rounded_dropoff_datetime",
        round_timestamp_to_interval(col("tpep_dropoff_datetime"), dropoff_minutes),
    )
