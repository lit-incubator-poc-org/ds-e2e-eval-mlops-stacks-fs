# Feature Engineering in MLOps Pipeline

This document provides a comprehensive guide to feature engineering in the taxi fare prediction MLOps pipeline, covering how features are computed, stored, and used across different stages of the machine learning lifecycle.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Feature Engineering Architecture](#feature-engineering-architecture)
3. [Feature Development & Computation](#feature-development--computation)
4. [Feature Storage & Versioning](#feature-storage--versioning)
5. [Features in Model Training](#features-in-model-training)
6. [Features in Real-time Serving](#features-in-real-time-serving)
7. [Features in Batch Inference](#features-in-batch-inference)
8. [Feature Monitoring & Observability](#feature-monitoring--observability)
9. [Best Practices](#best-practices)

## 🎯 Overview

The feature engineering system in this MLOps pipeline is built on **Databricks Unity Catalog Feature Store**, providing:

- **Centralized Feature Management**: All features stored in Unity Catalog with governance
- **Time-series Features**: Sliding window aggregations for taxi trip patterns
- **Real-time Serving**: Sub-millisecond feature lookup via online tables
- **Feature Reuse**: Same features used in training and inference
- **Data Lineage**: Complete traceability from raw data to model predictions

### Key Technologies
- **Databricks Feature Store** (Unity Catalog-based)
- **Delta Lake** for feature table storage
- **Online Tables** for real-time serving
- **Apache Spark** for large-scale feature computation
- **Feature Engineering Client** for Unity Catalog integration

## 🏗️ Feature Engineering Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING LIFECYCLE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RAW DATA                FEATURE COMPUTATION         STORAGE    │
│  ┌─────────────┐        ┌─────────────────────┐    ┌─────────── │
│  │ NYC Taxi    │   ──>  │ pickup_features.py  │──> │ Unity      │
│  │ Trip Data   │        │ - Window aggregates │    │ Catalog    │
│  │ (Delta Lake)│        │ - Time-based        │    │ Feature    │
│  └─────────────┘        │   computations      │    │ Store      │
│                         │                     │    │            │
│                         │ dropoff_features.py │    │ ┌────────  │
│                         │ - Location stats    │    │ │ Offline  │
│                         │ - Weekend detection │    │ │ Tables   │
│                         └─────────────────────┘    │ │ (Delta)  │
│                                                    │ └────────  │
│  FEATURE SERVING                                   │            │
│  ┌─────────────────────────────────────────────┐  │ ┌────────  │
│  │              TRAINING                       │  │ │ Online   │
│  │  ┌─────────────────────────────────────┐    │  │ │ Tables   │
│  │  │ FeatureLookup + TrainingSet         │<───┼──┤ │ (Real-   │
│  │  │ - Automatic feature joining         │    │  │ │  time)   │
│  │  │ - Point-in-time correctness         │    │  │ └────────  │
│  │  └─────────────────────────────────────┘    │  │            │
│  └─────────────────────────────────────────────┘  └─────────── │
│                                                                 │
│  ┌─────────────────────────────────────────────┐               │
│  │            REAL-TIME SERVING                │               │
│  │  ┌─────────────────────────────────────┐    │               │
│  │  │ Serving Endpoint + Online Tables    │<───┼───────────────┤
│  │  │ - Sub-millisecond feature lookup    │    │               │
│  │  │ - Automatic feature enrichment      │    │               │
│  │  └─────────────────────────────────────┘    │               │
│  └─────────────────────────────────────────────┘               │
│                                                                 │
│  ┌─────────────────────────────────────────────┐               │
│  │             BATCH INFERENCE                 │               │
│  │  ┌─────────────────────────────────────┐    │               │
│  │  │ Batch Feature Lookup + Scoring      │<───┼───────────────┤
│  │  │ - Large-scale feature joining       │    │               │
│  │  │ - Historical feature access         │    │               │
│  │  └─────────────────────────────────────┘    │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Feature Development & Computation

### Feature Modules Structure

The feature engineering logic is organized into modular components:

```
feature_engineering/
├── features/
│   ├── pickup_features.py      # Pickup location feature logic
│   └── dropoff_features.py     # Dropoff location feature logic
├── notebooks/
│   └── GenerateAndWriteFeatures.py  # Feature pipeline execution
└── feature_engineering_utils.py     # Shared utilities
```

### Pickup Location Features

**File**: `feature_engineering/features/pickup_features.py`

**Features Computed**:
- `mean_fare_window_1h_pickup_zip`: Average fare amount in pickup location over 1-hour sliding window
- `count_trips_window_1h_pickup_zip`: Number of trips from pickup location in 1-hour window

**Implementation Details**:
```python
def compute_features_fn(input_df, timestamp_column, start_date, end_date):
    """
    Computes pickup location features using sliding window aggregations.
    
    Window Configuration:
    - Window Size: 1 hour
    - Slide Interval: 15 minutes (overlapping windows)
    - Aggregations: Mean fare, Trip count
    """
    pickupzip_features = (
        df.groupBy(
            "pickup_zip", 
            F.window(timestamp_column, "1 hour", "15 minutes")
        )
        .agg(
            F.mean("fare_amount").alias("mean_fare_window_1h_pickup_zip"),
            F.count("*").alias("count_trips_window_1h_pickup_zip"),
        )
        # Transform to feature store format with primary keys
        .select(
            F.col("pickup_zip").alias("zip"),
            F.unix_timestamp(F.col("window.end"))
            .alias(timestamp_column)
            .cast(TimestampType()),
            F.col("mean_fare_window_1h_pickup_zip").cast(FloatType()),
            F.col("count_trips_window_1h_pickup_zip").cast(IntegerType()),
        )
    )
    return pickupzip_features
```

**Primary Keys**: `["pickup_zip", "rounded_pickup_datetime"]`

### Dropoff Location Features  

**File**: `feature_engineering/features/dropoff_features.py`

**Features Computed**:
- `count_trips_window_30m_dropoff_zip`: Number of trips to dropoff location in 30-minute window
- `dropoff_is_weekend`: Boolean indicating if dropoff occurred on weekend

**Implementation Details**:
```python
@F.udf(returnType=IntegerType())
def _is_weekend(dt):
    """Detect weekend dropoffs using New York timezone"""
    tz = "America/New_York"
    return int(dt.astimezone(timezone(tz)).weekday() >= 5)

def compute_features_fn(input_df, timestamp_column, start_date, end_date):
    """
    Computes dropoff location features with time-based aggregations.
    
    Window Configuration:
    - Window Size: 30 minutes (non-overlapping)
    - Aggregations: Trip count, Weekend detection
    """
    dropoffzip_features = (
        df.groupBy("dropoff_zip", F.window(timestamp_column, "30 minute"))
        .agg(F.count("*").alias("count_trips_window_30m_dropoff_zip"))
        .select(
            F.col("dropoff_zip").alias("zip"),
            F.unix_timestamp(F.col("window.end")).alias(timestamp_column),
            F.col("count_trips_window_30m_dropoff_zip").cast(IntegerType()),
            _is_weekend(F.col("window.end")).alias("dropoff_is_weekend"),
        )
    )
    return dropoffzip_features
```

**Primary Keys**: `["dropoff_zip", "rounded_dropoff_datetime"]`

### Feature Pipeline Execution

**File**: `feature_engineering/notebooks/GenerateAndWriteFeatures.py`

**Purpose**: Orchestrates feature computation and writes to Feature Store

**Process**:
1. **Load Raw Data**: Read NYC taxi trip data from Delta Lake
2. **Dynamic Module Loading**: Import feature computation modules
3. **Execute Feature Functions**: Call `compute_features_fn` for each module
4. **Write to Feature Store**: Persist features with proper schema and partitioning
5. **Enable Change Data Feed**: Configure for online table synchronization

**Key Parameters**:
- `input_table_path`: Source data location  
- `output_table_name`: Target feature table name
- `primary_keys`: Primary key columns for feature store
- `timestamp_column`: Timestamp for time-series features
- `features_transform_module`: Feature computation module to use

## 📊 Feature Storage & Versioning

### Unity Catalog Feature Tables

Features are stored as Delta Lake tables in Unity Catalog with the following structure:

#### **Pickup Features Table**
**Table Name**: `p03.e2e_demo_simon.trip_pickup_features`

| Column | Data Type | Description |
|--------|-----------|-------------|
| `pickup_zip` | String | Pickup location ZIP code (Primary Key) |
| `rounded_pickup_datetime` | Timestamp | Rounded pickup time (Primary Key) |
| `mean_fare_window_1h_pickup_zip` | Float | Average fare in 1-hour window |
| `count_trips_window_1h_pickup_zip` | Integer | Trip count in 1-hour window |
| `yyyy_mm` | String | Partition key (YYYY-MM format) |

#### **Dropoff Features Table**  
**Table Name**: `p03.e2e_demo_simon.trip_dropoff_features`

| Column | Data Type | Description |
|--------|-----------|-------------|
| `dropoff_zip` | String | Dropoff location ZIP code (Primary Key) |
| `rounded_dropoff_datetime` | Timestamp | Rounded dropoff time (Primary Key) |
| `count_trips_window_30m_dropoff_zip` | Integer | Trip count in 30-minute window |
| `dropoff_is_weekend` | Integer | Weekend indicator (0/1) |
| `yyyy_mm` | String | Partition key (YYYY-MM format) |

### Data Partitioning Strategy

Features are partitioned by `yyyy_mm` for optimal query performance:
- **Partition Granularity**: Monthly partitions
- **Query Optimization**: Time-range queries leverage partition pruning
- **Storage Efficiency**: Old partitions can be archived or deleted
- **Parallel Processing**: Multiple partitions can be processed concurrently

### Change Data Feed (CDC)

**Purpose**: Enables incremental updates to online tables

**Configuration**:
```sql
-- Enable CDC on feature tables
ALTER TABLE p03.e2e_demo_simon.trip_pickup_features 
SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

ALTER TABLE p03.e2e_demo_simon.trip_dropoff_features 
SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
```

**Benefits**:
- **Real-time Sync**: Changes propagate to online tables automatically
- **Efficient Updates**: Only changed records are synchronized
- **Audit Trail**: Complete history of feature changes is maintained

## 🎯 Features in Model Training

### Feature Lookup Configuration

During training, features are automatically joined with training data using **FeatureLookup** specifications:

```python
from databricks.feature_engineering import FeatureLookup, FeatureEngineeringClient

# Define feature lookups with point-in-time correctness
pickup_feature_lookups = [
    FeatureLookup(
        table_name="p03.e2e_demo_simon.trip_pickup_features",
        feature_names=[
            "mean_fare_window_1h_pickup_zip",
            "count_trips_window_1h_pickup_zip",
        ],
        lookup_key=["pickup_zip"],                    # Join key
        timestamp_lookup_key=["rounded_pickup_datetime"],  # Temporal join
    ),
]

dropoff_feature_lookups = [
    FeatureLookup(
        table_name="p03.e2e_demo_simon.trip_dropoff_features", 
        feature_names=["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
        lookup_key=["dropoff_zip"],
        timestamp_lookup_key=["rounded_dropoff_datetime"],
    ),
]
```

### Training Set Creation

**Point-in-Time Correctness**: Features are joined based on temporal relationships to prevent data leakage:

```python
# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Create training set with automatic feature joining
training_set = fe.create_training_set(
    df=taxi_data,                                          # Base training data
    feature_lookups=pickup_feature_lookups + dropoff_feature_lookups,  # Feature specs
    label="fare_amount",                                   # Target variable
    exclude_columns=["rounded_pickup_datetime", "rounded_dropoff_datetime"],  # Exclude timestamp keys
)

# Load complete dataset with features
training_df = training_set.load_df()
```

### Feature Lineage & Metadata

The Feature Store automatically tracks:
- **Data Lineage**: Source tables → Feature tables → Training datasets → Models
- **Feature Metadata**: Computation logic, data types, update frequency
- **Usage Tracking**: Which models use which features
- **Version History**: Feature schema evolution over time

### Training Data Schema

After feature joining, the training dataset contains:

| Column | Source | Type | Description |
|--------|--------|------|-------------|
| `pickup_zip` | Raw data | String | Pickup location |
| `dropoff_zip` | Raw data | String | Dropoff location |
| `trip_distance` | Raw data | Double | Trip distance in miles |
| `fare_amount` | Raw data | Double | **Target variable** |
| `mean_fare_window_1h_pickup_zip` | **Feature Store** | Float | Avg fare at pickup location |
| `count_trips_window_1h_pickup_zip` | **Feature Store** | Integer | Trip count at pickup |
| `count_trips_window_30m_dropoff_zip` | **Feature Store** | Integer | Trip count at dropoff |
| `dropoff_is_weekend` | **Feature Store** | Integer | Weekend indicator |

## 🚀 Features in Real-time Serving

### Online Tables Architecture

For real-time serving, feature tables are synchronized to **Online Tables** that provide sub-millisecond lookup:

```python
def create_online_tables(catalog_name: str = "p03", schema_name: str = "e2e_demo_simon") -> List[str]:
    """
    Create Databricks Online Tables for real-time feature serving.
    """
    feature_tables = [
        {
            "source": f"{catalog_name}.{schema_name}.trip_pickup_features",
            "online": f"{catalog_name}.{schema_name}.trip_pickup_features_online",
            "primary_keys": ["pickup_zip", "rounded_pickup_datetime"]
        },
        {
            "source": f"{catalog_name}.{schema_name}.trip_dropoff_features", 
            "online": f"{catalog_name}.{schema_name}.trip_dropoff_features_online",
            "primary_keys": ["dropoff_zip", "rounded_dropoff_datetime"]
        }
    ]
    
    for table_config in feature_tables:
        # Create online table with triggered scheduling
        spec = OnlineTableSpec(
            primary_key_columns=table_config["primary_keys"],
            source_table_full_name=table_config["source"],
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),
            perform_full_copy=True
        )
        
        online_table = OnlineTable(name=table_config["online"], spec=spec)
        workspace.online_tables.create_and_wait(table=online_table)
```

### Real-time Feature Lookup

During inference, the serving endpoint automatically performs feature lookups:

#### **API Request** (Input)
```json
{
  "dataframe_records": [{
    "pickup_zip": "10001",
    "dropoff_zip": "10002", 
    "trip_distance": 2.5,
    "pickup_datetime": "2023-01-01T12:00:00"
  }]
}
```

#### **Automatic Feature Enrichment**

The serving endpoint automatically:
1. **Rounds Timestamps**: Convert `pickup_datetime` to `rounded_pickup_datetime`
2. **Lookup Pickup Features**: Query `trip_pickup_features_online` using `pickup_zip` + `rounded_pickup_datetime`
3. **Lookup Dropoff Features**: Query `trip_dropoff_features_online` using `dropoff_zip` + `rounded_dropoff_datetime`  
4. **Enrich Request**: Combine input data with retrieved features
5. **Model Inference**: Score enriched data with trained model

#### **Enriched Feature Vector** (Internal)
```python
{
    "pickup_zip": "10001",
    "dropoff_zip": "10002", 
    "trip_distance": 2.5,
    # Automatically retrieved features:
    "mean_fare_window_1h_pickup_zip": 12.45,
    "count_trips_window_1h_pickup_zip": 23,
    "count_trips_window_30m_dropoff_zip": 18,
    "dropoff_is_weekend": 0
}
```

#### **API Response** (Output)
```json
{
  "predictions": [4.97]
}
```

### Performance Characteristics

**Online Tables provide**:
- **Latency**: Sub-millisecond feature lookup
- **Throughput**: High concurrent request handling
- **Consistency**: Eventual consistency with source tables
- **Availability**: High availability with automatic failover

## 📈 Features in Batch Inference

### Batch Feature Processing

For large-scale batch inference, features are retrieved directly from offline feature tables:

```python
# Batch inference with feature store integration
def batch_inference_with_features(input_df, model_uri, feature_lookups):
    """
    Perform batch inference with automatic feature joining.
    """
    fe = FeatureEngineeringClient()
    
    # Create inference set (similar to training set but without labels)
    inference_set = fe.create_training_set(
        df=input_df,
        feature_lookups=feature_lookups,
        exclude_columns=["rounded_pickup_datetime", "rounded_dropoff_datetime"]
    )
    
    # Load features and make predictions
    inference_df = inference_set.load_df()
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(inference_df)
    
    return predictions
```

### Batch Processing Benefits

- **Scalability**: Process millions of records using Spark
- **Cost Efficiency**: Use offline tables (no online table costs)
- **Historical Features**: Access to complete historical feature data
- **Point-in-Time Correctness**: Maintain temporal accuracy for batch scenarios

## 📊 Feature Monitoring & Observability

### Feature Quality Monitoring

**Data Quality Checks**:
```python
# Example feature validation logic
def validate_features(feature_df):
    """Validate feature quality and completeness."""
    
    # Check for null values
    null_counts = feature_df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_nulls") 
        for c in feature_df.columns
    ]).collect()[0]
    
    # Check feature distributions
    feature_stats = feature_df.describe().collect()
    
    # Validate primary key uniqueness
    total_rows = feature_df.count()
    unique_keys = feature_df.select("pickup_zip", "rounded_pickup_datetime").distinct().count()
    
    return {
        "null_counts": null_counts,
        "feature_stats": feature_stats, 
        "key_uniqueness": unique_keys == total_rows
    }
```

### Feature Drift Detection

**Monitoring Approach**:
- **Statistical Tests**: Compare feature distributions over time
- **Threshold Alerts**: Alert on significant distribution changes  
- **Visualization**: Dashboard showing feature evolution
- **Automated Retraining**: Trigger model retraining on drift detection

### Usage Analytics

**Feature Store provides**:
- **Usage Tracking**: Which models use which features
- **Performance Metrics**: Feature lookup latency and error rates
- **Cost Analytics**: Storage and compute costs per feature table
- **Lineage Visualization**: End-to-end data flow diagrams

## 🎯 Best Practices

### Feature Engineering Best Practices

#### **1. Temporal Correctness**
```python
# ✅ Good: Use point-in-time lookups
FeatureLookup(
    table_name="features_table",
    lookup_key=["entity_id"],
    timestamp_lookup_key=["timestamp"]  # Ensures no data leakage
)

# ❌ Bad: Static lookups without temporal constraints
FeatureLookup(
    table_name="features_table", 
    lookup_key=["entity_id"]
    # Missing timestamp - could cause data leakage
)
```

#### **2. Feature Naming Conventions**
```python
# ✅ Good: Descriptive names with context
"mean_fare_window_1h_pickup_zip"     # Clear aggregation, window, and entity
"count_trips_window_30m_dropoff_zip" # Explicit time window and metric

# ❌ Bad: Generic or unclear names  
"feature_1"                          # No semantic meaning
"avg_fare"                          # Missing context (window, entity)
```

#### **3. Primary Key Design**
```python
# ✅ Good: Compound primary keys for time-series features
primary_keys = ["pickup_zip", "rounded_pickup_datetime"]

# ❌ Bad: Missing temporal component
primary_keys = ["pickup_zip"]  # Could lead to duplicates or overwrites
```

#### **4. Data Type Optimization**
```python
# ✅ Good: Appropriate data types for efficiency
F.col("count_trips").cast(IntegerType())    # Integer for counts
F.col("mean_fare").cast(FloatType())        # Float for averages
F.col("is_weekend").cast(IntegerType())     # Integer for booleans (0/1)

# ❌ Bad: Using overly large data types
F.col("count_trips").cast(DoubleType())     # Unnecessary precision
```

### Production Deployment Best Practices

#### **1. Feature Store Governance**
- **Schema Evolution**: Use backward-compatible schema changes
- **Access Control**: Implement proper permissions and governance
- **Documentation**: Maintain feature documentation and examples
- **Testing**: Implement comprehensive feature validation tests

#### **2. Online Table Management**
- **Sync Strategy**: Use triggered scheduling for controlled updates
- **Performance Monitoring**: Track lookup latency and error rates
- **Capacity Planning**: Monitor storage and compute usage
- **Disaster Recovery**: Implement backup and recovery procedures

#### **3. Feature Pipeline Monitoring**
- **Data Quality**: Implement automated data quality checks
- **Pipeline Health**: Monitor feature computation job success rates
- **Alert Configuration**: Set up alerts for pipeline failures
- **Performance Optimization**: Regular performance tuning and optimization

### Development Workflow

#### **1. Feature Development Cycle**
1. **Explore**: Analyze raw data and identify potential features
2. **Implement**: Code feature computation logic with proper testing
3. **Validate**: Test features in development environment
4. **Deploy**: Deploy to feature store using CI/CD pipeline
5. **Monitor**: Set up monitoring and quality checks

#### **2. Testing Strategy**  
```python
# Unit tests for feature computation
def test_pickup_features():
    # Create test data
    test_df = spark.createDataFrame([...])
    
    # Compute features
    result = compute_features_fn(test_df, "timestamp", None, None)
    
    # Validate output schema and values
    assert result.schema == expected_schema
    assert result.count() > 0
```

#### **3. CI/CD Integration**
- **Automated Testing**: Run feature tests on every commit
- **Staging Deployment**: Deploy to staging environment first
- **Production Promotion**: Promote features after validation
- **Rollback Strategy**: Implement safe rollback procedures

---

This feature engineering system provides a robust, scalable foundation for machine learning in production, ensuring consistent feature computation across training and serving while maintaining data quality and operational excellence.