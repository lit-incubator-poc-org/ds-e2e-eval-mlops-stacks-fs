# Databricks notebook source
##################################################################################
# Batch Inference Notebook
#
# This notebook is an example of applying a model for batch inference against an input delta table,
# It is configured and can be executed as the batch_inference_job in the batch_inference_job workflow defined under
# ``mlops_stack_taxi_fares/assets/batch-inference-workflow-asset.yml``
#
# Parameters:
#
#  * env (optional)  - String name of the current environment (dev, staging, or prod). Defaults to "dev"
#  * input_table_name (required)  - Delta table name containing your input data.
#  * output_table_name (required) - Delta table name where the predictions will be written to.
#                                   Note that this will create a new version of the Delta table if
#                                   the table already exists
#  * model_name (required) - The name of the model to be used in batch inference.
##################################################################################


# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
# A Hive-registered Delta table containing the input features.
dbutils.widgets.text("input_table_name", "", label="Input Table Name")
# Delta table to store the output predictions.
dbutils.widgets.text("output_table_name", "", label="Output Table Name")
# Batch inference model name
dbutils.widgets.text(
    "model_name", "dev-mlops_stack_taxi_fares-model", label="Model Name"
)


# COMMAND ----------

import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

# COMMAND ----------

# DBTITLE 1,Define input and output variables
from utils import get_deployed_model_alias_for_env

env = dbutils.widgets.get("env")
input_table_name = dbutils.widgets.get("input_table_name")
output_table_name = dbutils.widgets.get("output_table_name")
model_name = dbutils.widgets.get("model_name")
assert input_table_name != "", "input_table_name notebook parameter must be specified"
assert output_table_name != "", "output_table_name notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
alias = get_deployed_model_alias_for_env(env)
model_uri = f"models:/{model_name}@{alias}"

# COMMAND ----------

from mlflow import MlflowClient

# Get model version from alias
client = MlflowClient()
try:
    # Get the model version by alias for Unity Catalog
    model_version_by_alias = client.get_model_version_by_alias(model_name, alias)
    model_version = int(model_version_by_alias.version)
except Exception:
    # Fallback: get the latest version if alias doesn't exist
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    if model_version_infos:
        model_version = max(int(version.version) for version in model_version_infos)
    else:
        raise ValueError(f"No model versions found for model {model_name}")

# COMMAND ----------

# Get datetime
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------
# DBTITLE 1,Data preprocessing and inference

# Import consolidated feature engineering and prediction utilities
from feature_engineering.feature_engineering_utils import add_rounded_timestamps
from predict import load_input_table, predict_batch_with_preprocessed_data

# Load input data using shared utility
raw_table = load_input_table(spark, input_table_name)

# Preprocess the data to match the training data schema using inline UDF logic
# For inference, we just need to add rounded timestamps (keeping originals for potential downstream use)
table = add_rounded_timestamps(raw_table)

# COMMAND ----------
# DBTITLE 1,Run inference

# Use the centralized prediction logic (UDF-free)
predict_batch_with_preprocessed_data(
    model_uri=model_uri,
    preprocessed_table=table,
    output_table_name=output_table_name,
    model_version=model_version,
    ts=ts
)

dbutils.notebook.exit(output_table_name)
