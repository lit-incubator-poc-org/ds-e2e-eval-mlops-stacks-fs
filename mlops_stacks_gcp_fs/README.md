# ## 📋 Table of ContentsYC Taxi Fare Prediction MLOps Pipeline

## � Table of Contents

- [🚕 Overview](#-overview)
- [🎯 Business Problem](#-business-problem)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [🔄 MLOps Pipeline Workflows](#-mlops-pipeline-workflows)
- [📁 Project Structure](#-project-structure)
- [🔧 Model Details](#-model-details)
  - [Algorithm & Features](#algorithm-lightgbm-gradient-boosting)
- [📊 Raw Data Overview](#-raw-data-overview)
- [🏪 Feature Store Architecture](#-feature-store-architecture)
  - [Why Feature Store is Needed](#why-feature-store-is-needed)
  - [Feature Tables](#feature-tables)
  - [Sample Feature Data](#-sample-feature-table-data)
- [⚙️ Feature Engineering Pipeline](#️-feature-engineering-pipeline)
- [🚀 Training Pipeline](#-training-pipeline)
- [📊 Deployment & Inference](#-deployment--inference)
  - [Model Deployment](#1-model-deployment)
  - [Feature Importance](#2-why-features-are-critical-during-inference)
  - [Batch Inference](#3-batch-inference)
  - [Real-time API Inference](#4-real-time-api-inference)
- [🔄 What Happens During Inference](#-what-happens-during-inference)
  - [Step-by-Step Flow](#step-by-step-inference-flow)
  - [Performance & Error Handling](#-performance-characteristics)
- [🛠️ Development Tools](#️-development-tools)
- [🚀 Getting Started](#-getting-started)

## �🚕 Overview

This project implements a complete MLOps pipeline for predicting NYC taxi fare amounts using a **LightGBM regression model**. The system leverages **Databricks Feature Store** for feature management, **MLflow** for experiment tracking and model registry, and **Databricks Asset Bundles** for deployment automation.

## 🎯 Business Problem

The goal is to predict taxi fare amounts based on:
- **Pickup and dropoff locations** (zip codes)
- **Trip timestamps** (pickup/dropoff times)
- **Historical patterns** from feature store
- **Temporal features** (weekend detection, time-based aggregations)

This enables taxi companies and ride-sharing services to:
- Provide accurate fare estimates to customers
- Optimize pricing strategies
- Improve route planning and demand forecasting

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │    │  Feature Store   │    │  Model Training │
│   (NYC Taxi)    │───▶│  Engineering     │───▶│   (LightGBM)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Real-time API   │    │   Batch          │    │  Model Registry │
│  Inference      │◀───┤  Inference       │◀───┤   (MLflow)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔄 MLOps Pipeline Workflows

### **1. Feature Engineering Job** (`feature-engineering-workflow-asset.yml`)
- Computes and updates feature store tables
- Runs on schedule or triggered by new data
- Maintains feature freshness and data quality

### **2. Model Training Job** (`model-workflow-asset.yml`)  
- Trains new model versions
- Validates model performance
- Registers approved models
- Deploys to appropriate environment

### **3. Batch Inference Job** (`batch-inference-workflow-asset.yml`)
- Processes large datasets for predictions
- Writes results to Delta tables
- Scheduled for regular batch processing

## 📁 Project Structure

```
mlops_stacks_gcp_fs/
├── feature_engineering/
│   ├── feature_engineering_utils.py      # Consolidated timestamp utilities
│   ├── features/
│   │   ├── pickup_features.py           # Pickup location features
│   │   └── dropoff_features.py          # Dropoff location features
│   └── notebooks/
│       └── GenerateAndWriteFeatures.py  # Feature store pipeline
├── training/
│   ├── training_utils.py                # Model utilities (type-safe)
│   └── notebooks/
│       └── TrainWithFeatureStore.py     # Training pipeline
├── deployment/
│   ├── model_deployment/                # Model deployment logic
│   └── batch_inference/                 # Batch prediction pipeline
├── assets/                             # Databricks Asset Bundle configs
├── requirements.txt                    # Dependencies + linting tools
├── pyproject.toml                     # Black configuration  
├── .pylintrc                          # Pylint configuration
└── mypy.ini                           # MyPy type checking
```

## 🔧 Model Details

### **Algorithm**: LightGBM Gradient Boosting
- **Type**: Regression model
- **Target**: `fare_amount` (continuous variable)
- **Framework**: LightGBM with MLflow integration
- **Parameters**:
  ```python
  {
    "num_leaves": 32,
    "objective": "regression", 
    "metric": "rmse"
  }
  ```

### **Input Features**:
1. **Raw Trip Data**:
   - `pickup_zip`: Pickup location zip code
   - `dropoff_zip`: Dropoff location zip code
   - `tpep_pickup_datetime`: Trip start time
   - `tpep_dropoff_datetime`: Trip end time

2. **Engineered Features from Feature Store**:
   - `mean_fare_window_1h_pickup_zip`: Average fare in pickup area (1-hour window)
   - `count_trips_window_1h_pickup_zip`: Trip volume in pickup area (1-hour window)
   - `count_trips_window_30m_dropoff_zip`: Trip volume in dropoff area (30-min window)
   - `dropoff_is_weekend`: Weekend indicator for dropoff time

## 📊 Raw Data Overview

### **NYC Taxi Raw Dataset (`/databricks-datasets/nyctaxi-with-zipcodes/subsampled`)**
```sql
-- Query: SELECT * FROM delta.`/databricks-datasets/nyctaxi-with-zipcodes/subsampled` LIMIT 10;
```

| tpep_pickup_datetime | tpep_dropoff_datetime | trip_distance | fare_amount | pickup_zip | dropoff_zip |
|---------------------|----------------------|---------------|-------------|------------|-------------|
| 2016-02-14 16:52:13 | 2016-02-14 17:16:04  | 4.94          | 19.0        | 10282      | 10171       |
| 2016-02-04 18:44:19 | 2016-02-04 18:46:00  | 0.28          | 3.5         | 10110      | 10110       |
| 2016-02-17 17:13:57 | 2016-02-17 17:17:55  | 0.7           | 5.0         | 10103      | 10023       |
| 2016-02-18 10:36:07 | 2016-02-18 10:41:45  | 0.8           | 6.0         | 10022      | 10017       |
| 2016-02-22 14:14:41 | 2016-02-22 14:31:52  | 4.51          | 17.0        | 10110      | 10282       |
| 2016-02-05 06:45:02 | 2016-02-05 06:50:26  | 1.8           | 7.0         | 10009      | 10065       |
| 2016-02-15 15:03:28 | 2016-02-15 15:18:45  | 2.58          | 12.0        | 10153      | 10199       |
| 2016-02-25 19:09:26 | 2016-02-25 19:24:50  | 1.4           | 11.0        | 10112      | 10069       |
| 2016-02-13 16:28:18 | 2016-02-13 16:36:36  | 1.21          | 7.5         | 10023      | 10153       |
| 2016-02-14 00:03:48 | 2016-02-14 00:10:24  | 0.6           | 6.0         | 10012      | 10003       |

**Raw Data Schema**:
- `tpep_pickup_datetime`: Trip start timestamp (used for pickup feature lookups)
- `tpep_dropoff_datetime`: Trip end timestamp (used for dropoff feature lookups)  
- `trip_distance`: Distance traveled in miles (not used in current model)
- `fare_amount`: **Target variable** for prediction
- `pickup_zip`: **Primary key** for pickup feature table lookups
- `dropoff_zip`: **Primary key** for dropoff feature table lookups

**Key Observations**:
- **Fare range**: $3.50 to $19.00 showing natural price variation
- **Trip patterns**: Short trips (0.28 miles, 2 minutes) to longer trips (4.94 miles, 24 minutes)
- **NYC coverage**: Various Manhattan zip codes (10009-10282) representing different neighborhoods
- **Time diversity**: Morning (06:45), afternoon (16:52), and late night (00:03) trips
- **February 2016**: All data from winter month, consistent with feature table samples

## 🏪 Feature Store Architecture

### **Why Feature Store is Needed**

The **Databricks Feature Store** solves several critical challenges:

1. **Feature Reusability**: Store computed features once, use across training and inference
2. **Data Consistency**: Ensure training and serving use identical feature definitions
3. **Point-in-Time Correctness**: Automatically handle temporal lookups to avoid data leakage
4. **Feature Discovery**: Centralized catalog of available features for different teams
5. **Lineage Tracking**: Track feature dependencies and transformations

### **Feature Tables**

#### **1. Pickup Features (`p03.e2e_demo_simon.trip_pickup_features`)** ([See sample data](#pickup-features-table-sample))
```python
# Time-windowed aggregations for pickup locations
features = df.groupBy("zip", window("tpep_pickup_datetime", "1 hour", "15 minutes")).agg(
    mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
    count("*").alias("count_trips_window_1h_pickup_zip")
)
```
- **Primary Key**: `zip`
- **Timestamp Key**: `tpep_pickup_datetime` (rounded to 15-minute intervals)
- **Window**: 1-hour sliding window, updated every 15 minutes
- **Purpose**: Capture demand patterns and fare trends by pickup location

#### **2. Dropoff Features (`p03.e2e_demo_simon.trip_dropoff_features`)** ([See sample data](#dropoff-features-table-sample))
```python
# Trip volume and temporal features for dropoff locations  
features = df.groupBy("zip", window("tpep_dropoff_datetime", "30 minutes")).agg(
    count("*").alias("count_trips_window_30m_dropoff_zip")
).withColumn("dropoff_is_weekend", is_weekend_udf("tpep_dropoff_datetime"))
```
- **Primary Key**: `zip`  
- **Timestamp Key**: `tpep_dropoff_datetime` (rounded to 30-minute intervals)
- **Window**: 30-minute sliding window
- **Purpose**: Capture destination demand and day-of-week patterns

### **📊 Sample Feature Table Data**

#### **Pickup Features Table Sample**
```sql
-- Query: SELECT * FROM p03.e2e_demo_simon.trip_pickup_features ORDER BY zip, tpep_pickup_datetime DESC LIMIT 10;
```

| zip   | tpep_pickup_datetime | yyyy_mm | mean_fare_window_1h_pickup_zip | count_trips_window_1h_pickup_zip |
|-------|---------------------|---------|-------------------------------|----------------------------------|
| 10001 | 2016-02-13 15:15:00 | 2016-02 | 12.25                         | 4                                |
| 10014 | 2016-02-03 05:15:00 | 2016-02 | 6.5                           | 1                                |
| 10018 | 2016-02-21 01:30:00 | 2016-02 | 6.75                          | 2                                |
| 10018 | 2016-02-26 23:00:00 | 2016-02 | 9                             | 2                                |
| 10019 | 2016-02-06 22:45:00 | 2016-02 | 15.33333302                   | 3                                |
| 10019 | 2016-02-23 18:30:00 | 2016-02 | 26.5                          | 2                                |
| 10103 | 2016-02-18 23:15:00 | 2016-02 | 23.5                          | 1                                |
| 10110 | 2016-02-12 19:00:00 | 2016-02 | 8.5                           | 3                                |
| 10162 | 2016-02-24 00:15:00 | 2016-02 | 20                            | 1                                |

**Key Observations**:
- **High-value areas**: Zip 10019 shows premium fares ($26.50, $15.33) and zip 10103 ($23.50) → Business/entertainment districts
- **Low-demand periods**: Early morning (05:15, 01:30) and late night (23:00, 22:45) show lower volumes (1-3 trips)
- **Fare variations**: Wide range from $6.50 (10014 early morning) to $26.50 (10019 evening) demonstrates time-of-day pricing
- **15-minute intervals**: All timestamps end in :00, :15, :30, :45 showing consistent rounding implementation
- **yyyy_mm partitioning**: All data from 2016-02, enabling efficient temporal queries and data lifecycle management

#### **Dropoff Features Table Sample**
```sql
-- Query: SELECT * FROM p03.e2e_demo_simon.trip_dropoff_features ORDER BY zip, tpep_dropoff_datetime DESC LIMIT 10;
```

| zip  | tpep_dropoff_datetime | yyyy_mm | count_trips_window_30m_dropoff_zip | dropoff_is_weekend |
|------|----------------------|---------|-----------------------------------|--------------------|
| 7002 | 2016-02-26 11:30:00  | 2016-02 | 1                                 | 0                  |
| 7002 | 2016-01-22 14:00:00  | 2016-01 | 1                                 | 0                  |
| 7008 | 2016-02-12 22:00:00  | 2016-02 | 1                                 | 0                  |
| 7024 | 2016-02-22 19:00:00  | 2016-02 | 1                                 | 0                  |
| 7024 | 2016-02-09 00:30:00  | 2016-02 | 1                                 | 0                  |
| 7030 | 2016-02-28 21:30:00  | 2016-02 | 1                                 | 1                  |
| 7030 | 2016-02-21 22:30:00  | 2016-02 | 1                                 | 1                  |
| 7030 | 2016-02-11 11:30:00  | 2016-02 | 1                                 | 0                  |
| 7030 | 2016-01-31 01:30:00  | 2016-01 | 1                                 | 1                  |
| 7086 | 2016-01-11 01:00:00  | 2016-01 | 1                                 | 1                  |

**Key Observations**:
- **Weekend detection**: Mixed weekend (1) and weekday (0) patterns across different zip codes and times
- **Low-activity areas**: All zip codes (7002, 7008, 7024, 7030, 7086) show single trip counts indicating residential/suburban areas  
- **30-minute intervals**: All timestamps properly rounded to :00 and :30 minute intervals
- **Cross-month data**: Spans January (2016-01) and February (2016-02) showing temporal partitioning
- **Time diversity**: Late night (01:30), evening (21:30, 22:30), and daytime (11:30, 14:00) dropoffs captured

#### **Feature Table Schema**

##### **Pickup Features Schema**
```sql
CREATE TABLE p03.e2e_demo_simon.trip_pickup_features (
  zip STRING NOT NULL,                                 -- Primary key
  tpep_pickup_datetime TIMESTAMP NOT NULL,             -- Timestamp key (rounded to 15min)
  yyyy_mm STRING,                                      -- Year-month partition
  mean_fare_window_1h_pickup_zip DOUBLE,              -- Average fare in 1-hour window
  count_trips_window_1h_pickup_zip BIGINT,            -- Trip count in 1-hour window
  
  PRIMARY KEY (zip, tpep_pickup_datetime)
) USING DELTA
TBLPROPERTIES (
  'delta.feature.allowColumnDefaults' = 'supported',
  'delta.columnMapping.mode' = 'name'
);
```

##### **Dropoff Features Schema**
```sql
CREATE TABLE p03.e2e_demo_simon.trip_dropoff_features (
  zip STRING NOT NULL,                                 -- Primary key
  tpep_dropoff_datetime TIMESTAMP NOT NULL,            -- Timestamp key (rounded to 30min)
  yyyy_mm STRING,                                      -- Year-month partition
  count_trips_window_30m_dropoff_zip BIGINT,          -- Trip count in 30-minute window
  dropoff_is_weekend INT,                              -- Weekend flag (0=weekday, 1=weekend)
  
  PRIMARY KEY (zip, tpep_dropoff_datetime)
) USING DELTA
TBLPROPERTIES (
  'delta.feature.allowColumnDefaults' = 'supported',
  'delta.columnMapping.mode' = 'name'
);
```

## ⚙️ Feature Engineering Pipeline

### **1. Timestamp Rounding**
```python
def add_rounded_timestamps(df: DataFrame, pickup_minutes: int = 15, dropoff_minutes: int = 30) -> DataFrame:
    """
    Rounds timestamps to enable consistent feature store lookups.
    - Pickup: 15-minute intervals (e.g., 14:23 → 14:30)
    - Dropoff: 30-minute intervals (e.g., 14:43 → 15:00)
    """
```
**Why needed**: Feature Store requires consistent timestamp keys for point-in-time lookups.

### **2. Aggregation Features**
- **Mean fare by pickup zone**: Historical pricing trends
- **Trip counts**: Demand intensity indicators  
- **Weekend detection**: Temporal pattern capture

### **3. UDF-Free Implementation** 
```python
# Uses Spark built-in functions instead of UDFs for better performance
rounded_unix = ceil(unix_timestamp(timestamp_col) / interval_seconds) * interval_seconds
return from_unixtime(rounded_unix).cast("timestamp")
```
**Benefits**: 
- No serialization issues
- Better Spark optimization
- Improved performance

## 🚀 Training Pipeline

### **Process Flow**:

1. **Data Loading**: Read raw NYC taxi data from Delta Lake
2. **Feature Engineering**: Add rounded timestamps for feature store lookups
3. **Feature Store Integration**:
   ```python
   training_set = fs.create_training_set(
       taxi_data,
       feature_lookups=[pickup_features, dropoff_features],
       label="fare_amount",
       exclude_columns=["rounded_pickup_datetime", "rounded_dropoff_datetime"]
   )
   ```
4. **Model Training**: LightGBM with MLflow autologging
5. **Model Registration**: Automatic versioning in Unity Catalog
6. **Model Deployment**: Alias-based deployment (dev/staging/prod)

### **Feature Lookup Configuration**:
```python
pickup_feature_lookups = [
    FeatureLookup(
        table_name="p03.e2e_demo_simon.trip_pickup_features",
        feature_names=["mean_fare_window_1h_pickup_zip", "count_trips_window_1h_pickup_zip"],
        lookup_key=["pickup_zip"],
        timestamp_lookup_key=["rounded_pickup_datetime"]
    )
]
```

## 📊 Deployment & Inference

### **1. Model Deployment**
- **Registry**: MLflow Model Registry with Unity Catalog integration
- **Aliases**: Environment-based aliases (dev, staging, prod)
- **Versioning**: Automatic model versioning and lineage tracking

### **2. Why Features Are Critical During Inference**

The model **cannot make accurate predictions** without the same engineered features it was trained on. Here's why each feature type is essential:

#### **🕐 Temporal Context Features**

##### **Feature 1: `mean_fare_window_1h_pickup_zip`**
```python
# Calculation: Average fare amount for trips starting in the same pickup zone within 1-hour window
SELECT pickup_zip as zip, 
       window(tpep_pickup_datetime, "1 hour", "15 minutes") as time_window,
       AVG(fare_amount) as mean_fare_window_1h_pickup_zip
FROM taxi_trips 
GROUP BY pickup_zip, time_window
```

**Input Data Required**:
- `pickup_zip`: "10282", "10110", "10103" (pickup location zip codes from raw data)
- `tpep_pickup_datetime`: "2016-02-14 16:52:13" (trip start timestamp)
- `fare_amount`: 3.5, 19.0, 17.0 (historical fare amounts from raw data)

**Sample Feature Values**:
```json
{
  "pickup_zip": "10001",
  "timestamp": "2025-09-13T14:30:00",
  "mean_fare_window_1h_pickup_zip": 16.85  // Average of last hour's fares from this zip
}
```

**Business Logic**: 
- **Rush Hour (8 AM)**: Mean fare = $22.50 (high demand)
- **Late Night (2 AM)**: Mean fare = $8.75 (low demand) 
- **Airport Zone**: Mean fare = $35.00 (premium location)

##### **Feature 2: `count_trips_window_1h_pickup_zip`**
```python
# Calculation: Number of trips started from the same pickup zone within 1-hour window
SELECT pickup_zip as zip,
       window(tpep_pickup_datetime, "1 hour", "15 minutes") as time_window,
       COUNT(*) as count_trips_window_1h_pickup_zip
FROM taxi_trips
GROUP BY pickup_zip, time_window
```

**Input Data Required**:
- `pickup_zip`: Location identifiers for trip origins (from raw data)
- `tpep_pickup_datetime`: Trip start timestamps (from raw data)
- Trip records (each row = one trip from raw dataset)

**Sample Feature Values**:
```json
{
  "pickup_zip": "10001", 
  "timestamp": "2025-09-13T14:30:00",
  "count_trips_window_1h_pickup_zip": 45  // 45 trips started from this zip in last hour
}
```

**Business Logic**:
- **High Volume (>50 trips/hour)**: Surge pricing indicator → +30% fare premium
- **Medium Volume (20-50 trips/hour)**: Normal pricing
- **Low Volume (<20 trips/hour)**: Potential discounts to stimulate demand

**Purpose**: Capture **real-time market conditions**
- **Dynamic Pricing**: Fare patterns change throughout the day (rush hour vs. late night)
- **Demand Signals**: High trip volume = surge pricing opportunities  
- **Location Hotspots**: Popular pickup areas command premium fares
- **Without these**: Model would miss 30-50% of fare variation due to time-of-day effects

#### **🌍 Geographic Context Features**

##### **Feature 3: `count_trips_window_30m_dropoff_zip`**
```python
# Calculation: Number of trips ending in the same dropoff zone within 30-minute window
SELECT dropoff_zip as zip,
       window(tpep_dropoff_datetime, "30 minutes") as time_window,
       COUNT(*) as count_trips_window_30m_dropoff_zip  
FROM taxi_trips
GROUP BY dropoff_zip, time_window
```

**Input Data Required**:
- `dropoff_zip`: "10171", "10110", "10023" (destination zip codes from raw data)
- `tpep_dropoff_datetime`: "2016-02-14 17:16:04" (trip end timestamp from raw data)
- Trip records for counting destination activity

**Sample Feature Values**:
```json
{
  "dropoff_zip": "10019",
  "timestamp": "2025-09-13T14:45:00", 
  "count_trips_window_30m_dropoff_zip": 28  // 28 trips ended here in last 30 minutes
}
```

**Business Logic**:
- **High Dropoff Volume (>40 trips/30min)**: Popular destination → event happening → +25% fare premium
- **Business District Weekday**: 35-50 trips/30min during business hours
- **Entertainment District Weekend**: 60+ trips/30min during nightlife hours
- **Residential Areas**: <15 trips/30min typically

##### **Feature 4: `dropoff_is_weekend`** 
```python
# Calculation: Boolean flag indicating if dropoff occurs on weekend
SELECT dropoff_zip as zip,
       tpep_dropoff_datetime,
       CASE WHEN dayofweek(tpep_dropoff_datetime) IN (1, 7) THEN 1 ELSE 0 END as dropoff_is_weekend
FROM taxi_trips
```

**Input Data Required**:
- `tpep_dropoff_datetime`: Trip end timestamp to extract day of week
- No additional historical data needed (computed from timestamp)

**Sample Feature Values**:
```json
{
  "dropoff_zip": "10019",
  "timestamp": "2025-09-13T22:30:00",  // Saturday night
  "dropoff_is_weekend": 1  // Weekend = 1, Weekday = 0
}
```

**Business Logic**:
- **Weekend = 1**: Entertainment destinations, nightlife, higher fares expected
- **Weekday = 0**: Business destinations, commuter patterns, standard fares
- **Weekend Premium Examples**:
  - Entertainment District: +40% fare premium on weekends
  - Business District: -20% fare discount on weekends (less demand)
  - Airport: Similar fares regardless of day type

**Purpose**: Capture **destination-specific patterns**
- **Destination Premium**: Business districts vs. residential areas have different fare profiles
- **Event Detection**: Sudden dropoff volume spikes indicate concerts/events → higher fares
- **Day-of-Week Effects**: Weekend destinations (entertainment) vs. weekday (business) have different pricing
- **Without these**: Model would treat all destinations equally, missing 20-30% of geographic premium

#### **⚡ Real-Time Feature Requirements**

**Critical Point**: Features must be computed **at prediction time** using **current data**:

```python
# ❌ WRONG: Using stale/cached features from training time
cached_features = {"mean_fare_pickup": 12.50}  # From last week

# ✅ CORRECT: Fresh features from current timestamp  
current_time = "2025-09-13T14:30:00"
live_features = fs.lookup_features(
    pickup_zip="10001", 
    timestamp=current_time  # Gets features from CURRENT 1-hour window
)
```

**Real-World Feature Calculation Example**:
```python
# For a trip request at 2025-09-13 14:30:00 from zip 10001 to 10019
prediction_request = {
    "pickup_zip": "10001",
    "dropoff_zip": "10019", 
    "tpep_pickup_datetime": "2025-09-13T14:30:00",
    "tpep_dropoff_datetime": "2025-09-13T14:45:00"
}

# Feature Store automatically computes:
features_computed = {
    # Pickup features (1-hour lookback from 14:30:00)
    "mean_fare_window_1h_pickup_zip": 18.75,      # Avg fare 13:30-14:30 from zip 10001
    "count_trips_window_1h_pickup_zip": 42,       # Trip count 13:30-14:30 from zip 10001
    
    # Dropoff features (30-min lookback from 14:45:00)  
    "count_trips_window_30m_dropoff_zip": 15,     # Trip count 14:15-14:45 to zip 10019
    "dropoff_is_weekend": 0                       # Saturday = weekend = 1, but this is Friday
}

# Model input combines raw data + computed features:
model_input = {**prediction_request, **features_computed}
```

#### **🎯 Business Impact Without Features**

| **Scenario** | **Without Features** | **With Features** | **Impact** |
|--------------|---------------------|-------------------|------------|
| **Rush Hour (8 AM)** | Predicts $12 | Predicts $18 | +50% accuracy |  
| **Airport Dropoff** | Predicts $15 | Predicts $25 | +67% accuracy |
| **Weekend Night** | Predicts $20 | Predicts $35 | +75% accuracy |
| **Low Demand Period** | Predicts $12 | Predicts $8 | +33% accuracy |

**Result**: Without features, the model becomes a **simple distance calculator**, losing all contextual intelligence that makes it valuable for dynamic pricing.

### **3. Batch Inference**

#### **📦 Overview**
Batch inference processes large datasets (thousands to millions of trips) for scenarios like:
- **Historical Analysis**: Analyzing fare patterns over past months
- **Business Intelligence**: Daily/weekly fare reports
- **Model Validation**: Backtesting model performance
- **Bulk Pricing**: Pre-computing fares for route optimization

#### **🔧 Implementation**

##### **Step 1: Data Preparation**
```python
from databricks import feature_store
from feature_engineering.feature_engineering_utils import add_rounded_timestamps
import mlflow

# Load raw trip data (Delta table or Parquet files)
raw_data = spark.read.format("delta").load("/path/to/trip/data")

# Sample raw data structure:
# +-----------------+-------------------+-------------+-----------+------------+-------------+
# |pickup_zip       |dropoff_zip        |trip_distance|fare_amount|pickup_time |dropoff_time |
# +-----------------+-------------------+-------------+-----------+------------+-------------+
# |10001            |10019              |2.5          |12.5       |2025-09-13..|2025-09-13..|
# |10110            |10282              |4.8          |19.0       |2025-09-13..|2025-09-13..|
# +-----------------+-------------------+-------------+-----------+------------+-------------+

# Add rounded timestamps for feature store lookups
preprocessed_data = add_rounded_timestamps(
    raw_data,
    pickup_minutes=15,   # Round to 15-minute intervals for pickup features
    dropoff_minutes=30   # Round to 30-minute intervals for dropoff features
)
```

##### **Step 2: Feature Store Batch Scoring**
```python
# Initialize Feature Store client
fs_client = feature_store.FeatureStoreClient()

# Model URI (from MLflow Model Registry)
model_uri = "models:/p03.e2e_demo_simon.taxi_fare_model/9"

# Batch scoring with automatic feature lookups
batch_predictions = fs_client.score_batch(
    model_uri=model_uri,
    df=preprocessed_data,
    result_type="float"  # Return predictions as float values
)

# Result DataFrame includes:
# - Original columns (pickup_zip, dropoff_zip, timestamps)
# - Feature Store lookups (mean_fare_1h, count_trips, etc.)  
# - Model prediction column
batch_predictions.display()
```

##### **Step 3: Results Processing & Storage**
```python
# Add metadata and business logic
enriched_results = (
    batch_predictions
    .withColumn("prediction_date", current_date())
    .withColumn("model_version", lit("9"))
    .withColumn("batch_id", lit("batch_2025_09_13"))
    .withColumn("fare_category", 
        when(col("prediction") < 10, "Economy")
        .when(col("prediction") < 20, "Standard") 
        .otherwise("Premium")
    )
)

# Save results to Delta table for business analytics
enriched_results.write.format("delta").mode("overwrite").saveAsTable(
    "p03.e2e_demo_simon.taxi_fare_predictions_batch"
)

# Show sample results
enriched_results.select(
    "pickup_zip", "dropoff_zip", "prediction", "fare_category", "batch_id"
).show(10)
```

#### **📊 Sample Batch Output**
```
+----------+-----------+----------+-------------+------------------+
|pickup_zip|dropoff_zip|prediction|fare_category|batch_id          |
+----------+-----------+----------+-------------+------------------+
|10001     |10019      |15.75     |Standard     |batch_2025_09_13  |
|10110     |10282      |22.40     |Premium      |batch_2025_09_13  |
|10103     |10023      |8.90      |Economy      |batch_2025_09_13  |
|10017     |10065      |18.25     |Standard     |batch_2025_09_13  |
|10009     |10153      |12.60     |Standard     |batch_2025_09_13  |
+----------+-----------+----------+-------------+------------------+
```


### **4. Real-time API Inference**

#### **🔄 Feature Store Integration During Inference**

The model is packaged with **Feature Store integration**, which creates a two-step process:

```
Raw API Input → Feature Store Lookups → Model Prediction
     ↓                    ↓                   ↓
pickup_zip: "10001"  →  mean_fare_1h: 18.75  →  $15.75
dropoff_zip: "10019" →  count_trips: 42      →
timestamps           →  is_weekend: 0        →
```

#### **📥 API Input Schema** (What you send):
```json
{
  "dataframe_records": [
    {
      "pickup_zip": "10001",           // String: Pickup location identifier  
      "dropoff_zip": "10019",          // String: Dropoff location identifier
      "tpep_pickup_datetime": "2025-09-13T14:30:00",    // ISO timestamp
      "tpep_dropoff_datetime": "2025-09-13T14:45:00"    // ISO timestamp
    }
  ]
}
```

#### **🧮 Model Signature Schema** (What the model receives after feature store):
| Name | Type | Source |
|------|------|--------|
| `trip_distance` | double | Raw input (added by feature store) |
| `pickup_zip` | integer | Raw input (converted to int) |
| `dropoff_zip` | integer | Raw input (converted to int) |
| `mean_fare_window_1h_pickup_zip` | float | **Feature Store Lookup** |
| `count_trips_window_1h_pickup_zip` | integer | **Feature Store Lookup** |
| `count_trips_window_30m_dropoff_zip` | integer | **Feature Store Lookup** |  
| `dropoff_is_weekend` | integer | **Feature Store Lookup** |

**🔑 Key Point**: You only send **4 fields** in the API call, but the model receives **7 fields** after Feature Store automatically adds the engineered features.

#### **Expected Response**:
```json
{
  "predictions": [
    {
      "prediction": 15.75,
      "model_version": "9",
      "pickup_zip": "10001",
      "dropoff_zip": "10019"
    }
  ]
}
```

#### **API Call Example**:
```bash
curl -X POST "https://<databricks-instance>/serving-endpoints/taxi-fare-model/invocations" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "pickup_zip": "10001",
        "dropoff_zip": "10019",
        "tpep_pickup_datetime": "2025-09-13T14:30:00", 
        "tpep_dropoff_datetime": "2025-09-13T14:45:00"
      }
    ]
  }'
```

#### **🚨 Important Notes**:
1. **`trip_distance`**: Not required in API input - Feature Store adds default value
2. **Timestamp Format**: Must be ISO 8601 format (`YYYY-MM-DDTHH:MM:SS`)
3. **Zip Codes**: Send as strings, automatically converted to integers
4. **Feature Lookups**: Happen automatically - no need to pre-compute features

## 🔄 What Happens During Inference

### **Step-by-Step Inference Flow**

When a prediction request is made, here's exactly what happens behind the scenes:

#### **Step 1: API Request Received** 
```json
POST /serving-endpoints/taxi-fare-model/invocations
{
  "dataframe_records": [{
    "pickup_zip": "10001",
    "dropoff_zip": "10019", 
    "tpep_pickup_datetime": "2025-09-13T14:30:00",
    "tpep_dropoff_datetime": "2025-09-13T14:45:00"
  }]
}
```

#### **Step 2: Input Validation & Preprocessing**
```python
# Databricks automatically handles:
validated_input = {
    "pickup_zip": 10001,                           # String → Integer conversion
    "dropoff_zip": 10019,                          # String → Integer conversion  
    "tpep_pickup_datetime": "2025-09-13T14:30:00", # ISO timestamp validation
    "tpep_dropoff_datetime": "2025-09-13T14:45:00" # ISO timestamp validation
}

# Add missing required fields with defaults
enhanced_input = {
    **validated_input,
    "trip_distance": 0.0,                          # Default value (not used in model)
    "rounded_pickup_datetime": "2025-09-13T14:30:00",   # For feature lookup
    "rounded_dropoff_datetime": "2025-09-13T14:45:00"   # For feature lookup  
}
```

#### **Step 3: Feature Store Lookups**
```python
# Pickup Features Lookup (1-hour window from 13:30-14:30)
pickup_features = feature_store.lookup(
    table="p03.e2e_demo_simon.trip_pickup_features",
    keys={"zip": 10001},
    timestamp="2025-09-13T14:30:00"
)
# Returns:
# mean_fare_window_1h_pickup_zip: 18.75
# count_trips_window_1h_pickup_zip: 42

# Dropoff Features Lookup (30-min window from 14:15-14:45)  
dropoff_features = feature_store.lookup(
    table="p03.e2e_demo_simon.trip_dropoff_features", 
    keys={"zip": 10019},
    timestamp="2025-09-13T14:45:00"
)
# Returns:
# count_trips_window_30m_dropoff_zip: 15
# dropoff_is_weekend: 0  (Friday = weekday)
```

#### **Step 4: Feature Vector Assembly**
```python
# Combine raw input + feature store lookups
model_input_vector = {
    # From API request (preprocessed)
    "pickup_zip": 10001,
    "dropoff_zip": 10019,
    "trip_distance": 0.0,
    
    # From pickup feature store lookup
    "mean_fare_window_1h_pickup_zip": 18.75,     # Avg fare in pickup area (last hour)
    "count_trips_window_1h_pickup_zip": 42,      # Trip volume in pickup area (last hour)
    
    # From dropoff feature store lookup
    "count_trips_window_30m_dropoff_zip": 15,    # Trip volume in dropoff area (last 30min)
    "dropoff_is_weekend": 0                      # Weekend flag (0 = weekday)
}
```

#### **Step 5: Model Prediction**
```python
# LightGBM model processes the complete feature vector
prediction = lightgbm_model.predict([
    [10001, 10019, 0.0, 18.75, 42, 15, 0]  # Feature vector as array
])
# prediction = 15.75
```

#### **Step 6: Response Assembly**
```json
{
  "predictions": [
    {
      "prediction": 15.75,
      "model_version": "9", 
      "pickup_zip": "10001",
      "dropoff_zip": "10019"
    }
  ]
}
```

### **🕐 Real-Time Feature Computation Example**

**Scenario**: Prediction request at `2025-09-13T14:30:00`

#### **Pickup Features (zip: 10001)**
```sql
-- Query executed by Feature Store automatically:
SELECT 
    mean_fare_window_1h_pickup_zip,
    count_trips_window_1h_pickup_zip
FROM p03.e2e_demo_simon.trip_pickup_features  
WHERE zip = '10001'
  AND tpep_pickup_datetime = '2025-09-13T14:30:00'  -- Rounded timestamp
```

**Result**: `mean_fare: $18.75, trip_count: 42` (from 13:30-14:30 window)

#### **Dropoff Features (zip: 10019)**
```sql
-- Query executed by Feature Store automatically:
SELECT 
    count_trips_window_30m_dropoff_zip,
    dropoff_is_weekend
FROM p03.e2e_demo_simon.trip_dropoff_features
WHERE zip = '10019' 
  AND tpep_dropoff_datetime = '2025-09-13T14:45:00'  -- Rounded timestamp
```

**Result**: `trip_count: 15, is_weekend: 0` (from 14:15-14:45 window)

### **⚡ Performance Characteristics**

- **Latency**: ~50-150ms per prediction
- **Feature Store**: ~20-30ms for lookups
- **Model Inference**: ~10-20ms for LightGBM
- **Network/Serialization**: ~20-100ms

### **🚨 Error Handling**

**Common Issues & Responses**:

1. **Missing Features**:
   ```json
   {
     "error": "Feature not found for zip=99999 at timestamp=2025-09-13T14:30:00",
     "fallback": "Uses historical average for missing features"
   }
   ```

2. **Invalid Timestamps**:
   ```json
   {
     "error": "Invalid timestamp format. Expected ISO 8601: YYYY-MM-DDTHH:MM:SS"
   }
   ```

3. **Model Version Issues**:
   ```json
   {
     "error": "Model version 9 not found",
     "fallback": "Using latest available version"
   }
   ```

## 🛠️ Development Tools

### **Code Quality**:
- **Black**: Code formatting (88 char line length)
- **Pylint**: Linting (10.00/10 score achieved)
- **MyPy**: Static type checking (fully typed)

### **Type Safety**:
All functions include complete type annotations:
```python
def add_rounded_timestamps(
    df: DataFrame, pickup_minutes: int = 15, dropoff_minutes: int = 30
) -> DataFrame:
```

### **Testing**:
```bash
# Run all quality checks
.venv/bin/black feature_engineering/
.venv/bin/mypy feature_engineering/  
.venv/bin/pylint feature_engineering/
```

## 🚀 Getting Started

1. **Setup Environment**: `./scripts/setup_environment.sh`
2. **Deploy Bundle**: `databricks bundle deploy --force-lock`
3. **Run Training**: `databricks bundle run model_training_job`
4. **Run Inference**: `databricks bundle run batch_inference_job`

See the [Project overview](../docs/project-overview.md) for additional details on code structure.