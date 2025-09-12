import mlflow
from pyspark.sql.functions import struct, lit, to_timestamp
from databricks.feature_store import FeatureStoreClient


def predict_batch_with_preprocessed_data(
    model_uri, preprocessed_table, output_table_name, model_version, ts
):
    """
    Apply the model at the specified URI for batch inference on preprocessed data,
    writing results to the table with name output_table_name.
    
    This function expects the input table to already have the rounded timestamp columns
    (rounded_pickup_datetime, rounded_dropoff_datetime) added for feature store lookups.
    """
    
    fs_client = FeatureStoreClient()

    prediction_df = fs_client.score_batch(
        model_uri,
        preprocessed_table
    )
    output_df = (
        prediction_df.withColumn("prediction", prediction_df["prediction"])
        .withColumn("model_version", lit(model_version))
        .withColumn("inference_timestamp", to_timestamp(lit(ts)))
    )
    
    output_df.display()
    # Model predictions are written to the Delta table provided as input.
    # Delta is the default format in Databricks Runtime 8.0 and above.
    output_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)
    
    
def load_input_table(spark_session, input_table_name):
    """
    Load input data from either a file path or table name.
    """
    if input_table_name.startswith("/"):
        # It's a file path, read as parquet/delta
        return spark_session.read.format("delta").load(input_table_name)
    else:
        # It's a table name
        return spark_session.table(input_table_name)