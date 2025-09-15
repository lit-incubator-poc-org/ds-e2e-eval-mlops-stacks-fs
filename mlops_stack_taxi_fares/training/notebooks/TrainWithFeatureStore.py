# Databricks notebook source
##################################################################################
# Model Training Notebook using Databricks Feature Store
#
# This notebook shows an example of a Model Training pipeline using Databricks Feature Store tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``mlops_stack_taxi_fares/assets/model-workflow-asset.yml``
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * training_data_path (required)   - Path to the training data.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - MLflow registered model name to use for the trained model. Will be created if it
# *                                   doesn't exist.
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# DBTITLE 1, Notebook arguments

# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text(
    "training_data_path",
    "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
    label="Path to the training data",
)

# Schema name.
dbutils.widgets.text(
    "schema_name",
    "default-schema",
    label="Schema name",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    "default-experiment",
    label="MLflow experiment name",
)

# MLflow registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "default_model", label="Model Name"
)

# Pickup features table name
dbutils.widgets.text(
    "pickup_features_table",
    "p03.e2e_demo_simon.trip_pickup_features",
    label="Pickup Features Table",
)

# Dropoff features table name
dbutils.widgets.text(
    "dropoff_features_table",
    "p03.e2e_demo_simon.trip_dropoff_features",
    label="Dropoff Features Table",
)

# COMMAND ----------
# DBTITLE 1,Define input and output variables

input_table_path = dbutils.widgets.get("training_data_path")
schema_name = dbutils.widgets.get("schema_name")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------
# DBTITLE 1, Set experiment

import mlflow

mlflow.set_experiment(experiment_name)


# COMMAND ----------
# DBTITLE 1, Load raw data

raw_data = spark.read.format("delta").load(input_table_path)
raw_data.display()

# COMMAND ----------
# DBTITLE 1, Helper functions

import pyspark.sql.functions as F
import sys
# Add directories to the Python path (relative to notebook location)
sys.path.append("../../feature_engineering")
sys.path.append("..")

# Import consolidated feature engineering and training functions
from feature_engineering_utils import add_rounded_timestamps
from training_utils import get_latest_model_version
import mlflow.pyfunc


# COMMAND ----------
# DBTITLE 1, Read taxi data for training

# Add rounded timestamps for feature store lookups, then drop original timestamps for training
taxi_data = (
    add_rounded_timestamps(raw_data)
    .drop("tpep_pickup_datetime")
    .drop("tpep_dropoff_datetime")
)
taxi_data.createOrReplaceTempView("taxi_data")
taxi_data.display()

# COMMAND ----------
# DBTITLE 1, Create FeatureLookups

from databricks.feature_engineering import FeatureLookup, FeatureEngineeringClient
import mlflow

pickup_features_table = dbutils.widgets.get("pickup_features_table")
dropoff_features_table = dbutils.widgets.get("dropoff_features_table")

pickup_feature_lookups = [
    FeatureLookup(
        table_name=pickup_features_table,
        feature_names=[
            "mean_fare_window_1h_pickup_zip",
            "count_trips_window_1h_pickup_zip",
        ],
        lookup_key=["pickup_zip"],
        timestamp_lookup_key=["rounded_pickup_datetime"],
    ),
]

dropoff_feature_lookups = [
    FeatureLookup(
        table_name=dropoff_features_table,
        feature_names=["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
        lookup_key=["dropoff_zip"],
        timestamp_lookup_key=["rounded_dropoff_datetime"],
    ),
]

# COMMAND ----------
# DBTITLE 1, Create Training Dataset

# Initialize the new Feature Engineering Client for Unity Catalog
fe = FeatureEngineeringClient()

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run()

# Since the rounded timestamp columns would likely cause the model to overfit the data
# unless additional feature engineering was performed, exclude them to avoid training on them.
exclude_columns = ["rounded_pickup_datetime", "rounded_dropoff_datetime"]

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
# Using the new FeatureEngineeringClient for Unity Catalog compatibility
training_set = fe.create_training_set(
    df=taxi_data,
    feature_lookups=pickup_feature_lookups + dropoff_feature_lookups,
    label="fare_amount",
    exclude_columns=exclude_columns,
)

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe, and note that it contains both the raw input data and the features from the Feature Store, like `dropoff_is_weekend`
training_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Train a LightGBM model on the data returned by `TrainingSet.to_df`, then log the model with `FeatureStoreClient.log_model`. The model will be packaged with feature metadata.

# COMMAND ----------
# DBTITLE 1, Train model

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import mlflow.lightgbm
from mlflow.tracking import MlflowClient


features_and_label = training_df.columns

# Collect data into a Pandas array for training
data = training_df.toPandas()[features_and_label]

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)
y_train = train.fare_amount
y_test = test.fare_amount

mlflow.lightgbm.autolog()
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
num_rounds = 100

# Train a lightGBM model
model = lgb.train(param, train_lgb_dataset, num_rounds)

# COMMAND ----------
# DBTITLE 1, Log model and return output.

# Log the trained model with MLflow and package it with feature lookup information.
# CRITICAL: Use FeatureEngineeringClient.log_model for Unity Catalog and online feature store compatibility
fe.log_model(
    model=model,
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name=model_name,  # Model name already contains the full Unity Catalog name
)

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
