# MLOps Workflow Documentation

This document provides a comprehensive overview of the MLOps workflow, including all files, configurations, and steps involved in the end-to-end machine learning lifecycle.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Workflow Steps](#workflow-steps)
4. [File Descriptions](#file-descriptions)
5. [Asset Bundle Configuration](#asset-bundle-configuration)
6. [Configuration Files](#configuration-files)
7. [Execution Commands](#execution-commands)
8. [Monitoring & Validation](#monitoring--validation)

## 🎯 Overview

This MLOps pipeline implements a complete machine learning workflow using Azure Databricks, Unity Catalog, and Feature Store integration. The pipeline covers:

- **Feature Engineering**: Create and manage features in Unity Catalog Feature Store
- **Model Training**: Train models with feature store integration and log to MLflow
- **Model Deployment**: Deploy models to serving endpoints with online feature lookup
- **Testing & Validation**: Comprehensive testing of real-time and batch inference

### Key Technologies
- **Azure Databricks** with Unity Catalog
- **Databricks Feature Store** (Unity Catalog-based)
- **MLflow** for experiment tracking and model registry  
- **Databricks Asset Bundles** for deployment automation
- **Delta Lake** for data storage
- **Apache Spark** for data processing

## Automation Scripts

**Location**: `scripts/run_e2e_mlops_pipeline.sh`

**Usage Examples**:
```bash
# Quick Start

## Option 1: From Project Root Directory

```bash
# Full pipeline (recommended for first run)
./scripts/run_e2e_mlops_pipeline.sh

# Skip feature engineering (if features already exist)
./scripts/run_e2e_mlops_pipeline.sh --skip-features

# Force feature refresh
./scripts/run_e2e_mlops_pipeline.sh --force-features
```

## Option 2: Using Convenience Launcher (from anywhere)

```bash
# Can be run from any directory within the project
./run_mlops.sh

# With options
./run_mlops.sh --skip-features
./run_mlops.sh --force-features
```
```

**Key Features**:
- ✅ Uses existing configuration files from `serving/config/`
- ✅ Dynamic model version detection from training output
- ✅ Automatic directory navigation (works from project root or scripts folder)
- ✅ Comprehensive error handling and status reporting

## 📁 Directory Structure

```
mlops_stack_taxi_fares/
├── 🤖 AUTOMATION SCRIPTS
│   └── scripts/
│       └── run_e2e_mlops_pipeline.sh   # 🚀 Complete end-to-end automation
│
├── 🔧 CONFIGURATION
│   ├── databricks.yml                  # 🔧 Main Databricks Asset Bundle config
│   ├── README_WORKFLOW.md              # 📖 This workflow documentation
│   ├── requirements.txt                # 📦 Python dependencies
│   ├── pyproject.toml                  # 🎨 Code formatting (Black)
│   ├── .pylintrc                       # 📏 Code quality (Pylint)
│   └── mypy.ini                        # 🔍 Type checking (MyPy)
│
├── 🏗️ ASSET BUNDLES (Workflow Definitions)
│   └── assets/
│       ├── feature-engineering-workflow-asset.yml  # Feature pipeline definition
│       ├── model-workflow-asset.yml               # Training pipeline definition
│       ├── batch-inference-workflow-asset.yml     # Batch inference definition
│       ├── monitoring-workflow-asset.yml          # Monitoring pipeline definition
│       └── ml-artifacts-asset.yml                 # ML artifacts management
│
├── 🔬 FEATURE ENGINEERING
│   └── feature_engineering/
│       ├── features/
│       │   ├── pickup_features.py      # Pickup location feature computations
│       │   └── dropoff_features.py     # Dropoff location feature computations
│       └── notebooks/
│           └── GenerateAndWriteFeatures.py  # Feature store pipeline execution
│
├── 🎯 MODEL TRAINING  
│   └── training/
│       ├── steps/                     # Training pipeline components
│       │   ├── ingest.py              # Data ingestion logic
│       │   ├── split.py               # Data splitting logic
│       │   ├── transform.py           # Data transformation logic
│       │   ├── train.py               # Model training logic
│       │   └── custom_metrics.py      # Custom evaluation metrics
│       ├── data/
│       │   └── sample.parquet         # Sample data for testing
│       └── notebooks/
│           └── TrainWithFeatureStore.py    # Main training notebook
│
├── 🚀 MODEL SERVING
│   └── serving/
│       ├── notebooks/
│       │   ├── OnlineTableDeployment.py     # Online tables & serving setup
│       │   └── ValidationNotebook.py        # End-to-end validation
│       ├── config/
│       │   ├── serving_endpoint_config.json # Active serving configuration
│       │   ├── test_single_prediction.json  # Single prediction test
│       │   └── test_batch_predictions.json  # Batch prediction test
│       └── README.md                        # Serving-specific documentation
│
├── 🔄 DEPLOYMENT (Legacy/Traditional)
│   └── deployment/
│       ├── model_deployment/           # Traditional deployment methods
│       └── batch_inference/            # Batch prediction pipeline
│           ├── predict.py              # Batch inference logic
│           └── notebooks/
│               └── BatchInference.py   # Batch inference notebook
│
├── ✅ VALIDATION & MONITORING
│   ├── validation/
│   │   ├── validation.py              # Model validation utilities
│   │   └── notebooks/
│   │       └── ModelValidation.py     # Model validation notebook
│   └── monitoring/
│       └── README.md                  # Monitoring documentation
│
└── 🧪 TESTING
    └── tests/
        ├── feature_engineering/         # Feature engineering tests
        │   ├── pickup_features_test.py
        │   └── dropoff_features_test.py
        └── training/                    # Training pipeline tests
            └── test_notebooks.py
```

## 🔄 Workflow Steps

### Step 1: Feature Engineering Pipeline

**Purpose**: Create and manage features in Unity Catalog Feature Store

**Files Involved**:
- `assets/feature-engineering-workflow-asset.yml` - Workflow definition
- `feature_engineering/notebooks/GenerateAndWriteFeatures.py` - Main execution
- `feature_engineering/features/pickup_features.py` - Pickup feature logic
- `feature_engineering/features/dropoff_features.py` - Dropoff feature logic

**Process**:
1. Load raw taxi trip data from Delta Lake
2. Compute pickup location features (aggregations, encodings)
3. Compute dropoff location features (aggregations, encodings)  
4. Write features to Unity Catalog Feature Store with primary keys
5. Enable Change Data Feed for online tables

**Outputs**:
- Feature tables: `p03.e2e_demo_simon.pickup_features`
- Feature tables: `p03.e2e_demo_simon.dropoff_features`
- Delta tables with CDC enabled for online serving

### Step 2: Model Training Pipeline

**Purpose**: Train ML models with feature store integration and log to MLflow

**Files Involved**:
- `assets/model-workflow-asset.yml` - Workflow definition
- `training/notebooks/TrainWithFeatureStore.py` - Main training notebook
- `training/steps/ingest.py` - Data ingestion
- `training/steps/split.py` - Train/validation split
- `training/steps/transform.py` - Data transformations
- `training/steps/train.py` - Model training
- `training/steps/custom_metrics.py` - Evaluation metrics

**Process**:
1. Load training data with feature store lookups
2. Split data into train/validation sets
3. Apply feature transformations and preprocessing
4. Train RandomForest regression model
5. Evaluate model performance with custom metrics
6. Log model, metrics, and artifacts to MLflow
7. Register model in Unity Catalog Model Registry

**Outputs**:
- Trained model: `p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model`
- MLflow experiment with metrics and artifacts
- Model registered with version number in Unity Catalog

### Step 3: Model Deployment & Serving

**Purpose**: Deploy model to serving endpoint with online feature lookup

**Files Involved**:
- `serving/notebooks/OnlineTableDeployment.py` - Deployment orchestration
- `serving/config/serving_endpoint_config.json` - Endpoint configuration
- Unity Catalog online tables for feature serving

**Process**:
1. Create or update online tables from feature store
2. Configure serving endpoint with Unity Catalog integration
3. Deploy model version to serving endpoint
4. Configure traffic routing and auto-scaling
5. Enable auto-capture for payload logging

**Outputs**:
- Serving endpoint: `mlops-taxi-fare-endpoint`
- Online tables for sub-millisecond feature lookup
- Auto-capture enabled for monitoring

### Step 4: Testing & Validation

**Purpose**: Comprehensive testing of serving functionality

**Files Involved**:
- `serving/notebooks/ValidationNotebook.py` - End-to-end validation
- `serving/config/test_single_prediction.json` - Single prediction test
- `serving/config/test_batch_predictions.json` - Batch prediction test
- `deployment/batch_inference/notebooks/BatchInference.py` - Batch testing

**Process**:
1. Test real-time single predictions via REST API
2. Test real-time batch predictions via REST API
3. Validate feature store lookup functionality
4. Run batch inference job for large-scale predictions
5. Verify endpoint health and performance metrics

**Outputs**:
- Validation results and performance metrics
- Test prediction outputs
- Endpoint health status

## 📄 File Descriptions

### Core Configuration Files

#### `databricks.yml` - Main Asset Bundle Configuration
```yaml
bundle:
  name: mlops_stacks_gcp_fs
  
targets:
  dev:
    default: true
    workspace:
      host: https://adb-8490988242777396.16.azuredatabricks.net
    
resources:
  jobs:
    feature_engineering_job: # References assets/feature-engineering-workflow-asset.yml
    model_training_job:      # References assets/model-workflow-asset.yml  
    batch_inference_job:     # References assets/batch-inference-workflow-asset.yml
    monitoring_job:          # References assets/monitoring-workflow-asset.yml
```

**Purpose**: Central configuration for Databricks Asset Bundles, defining all jobs, resources, and deployment targets.

#### `requirements.txt` - Python Dependencies
```
databricks-feature-engineering>=0.13.0
mlflow>=2.4.1
pandas>=1.3.0
scikit-learn>=1.0.0
# ... other dependencies
```

**Purpose**: Specifies all Python packages required for the MLOps pipeline.

### Asset Bundle Workflow Definitions

#### `assets/feature-engineering-workflow-asset.yml`
```yaml
resources:
  jobs:
    feature_engineering_job:
      name: "dev-${bundle.name}-feature-engineering-job"
      job_clusters:
        - job_cluster_key: feature_engineering_cluster
          new_cluster:
            spark_version: "17.1.x-cpu-ml-scala2.12"
            node_type_id: "Standard_D4s_v3"
            num_workers: 2
      tasks:
        - task_key: GenerateAndWriteFeatures
          job_cluster_key: feature_engineering_cluster
          notebook_task:
            notebook_path: ./feature_engineering/notebooks/GenerateAndWriteFeatures.py
```

**Purpose**: Defines the Databricks job configuration for feature engineering pipeline.

#### `assets/model-workflow-asset.yml`
```yaml
resources:
  jobs:
    model_training_job:
      name: "dev-${bundle.name}-model-training-job"
      tasks:
        - task_key: Train
          notebook_task:
            notebook_path: ./training/notebooks/TrainWithFeatureStore.py
        - task_key: ModelValidation
          depends_on:
            - task_key: Train
          notebook_task:
            notebook_path: ./validation/notebooks/ModelValidation.py
```

**Purpose**: Defines multi-task job for model training and validation with dependencies.

### Feature Engineering Files

#### `feature_engineering/features/pickup_features.py`
**Purpose**: Implements pickup location feature computations including:
- Location aggregations and statistics
- Time-based feature engineering  
- Categorical encodings
- Primary key management for feature store

**Key Functions**:
- `compute_pickup_features()` - Main feature computation
- `create_feature_table()` - Feature store table creation
- Feature schema definitions and data types

#### `feature_engineering/features/dropoff_features.py`  
**Purpose**: Similar to pickup features but for dropoff locations:
- Dropoff location aggregations
- Cross-location feature engineering
- Feature store integration

### Training Pipeline Files

#### `training/notebooks/TrainWithFeatureStore.py`
**Purpose**: Main training orchestration notebook that:
- Loads data with feature store lookups
- Orchestrates training pipeline steps
- Logs models and metrics to MLflow
- Registers models in Unity Catalog

**Key Sections**:
1. Feature store client initialization
2. Data loading with feature lookups  
3. Training pipeline execution
4. Model evaluation and logging
5. Model registration

#### `training/steps/train.py`
**Purpose**: Core model training logic:
- RandomForest model implementation
- Hyperparameter configuration
- Model training with feature store data
- Performance evaluation

### Serving Configuration Files

#### `serving/config/serving_endpoint_config.json`
```json
{
  "name": "mlops-taxi-fare-endpoint",
  "config": {
    "served_entities": [
      {
        "entity_name": "p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model",
        "entity_version": "18",
        "workload_size": "Small",
        "scale_to_zero_enabled": true,
        "workload_type": "CPU"
      }
    ],
    "auto_capture_config": {
      "catalog_name": "p03",
      "schema_name": "e2e_demo_simon", 
      "table_name_prefix": "taxi_fare_endpoint"
    }
  }
}
```

**Purpose**: Configuration for serving endpoint creation and updates.

#### `serving/config/test_single_prediction.json`
```json
{
  "dataframe_records": [
    {
      "pickup_location_id": 161,
      "dropoff_location_id": 141,
      "trip_distance": 1.0,
      "fare_amount": 7.0,
      "pickup_datetime": "2023-01-01 12:00:00"
    }
  ]
}
```

**Purpose**: Test input for single real-time predictions.

## 🚀 Asset Bundle Configuration

Asset bundles provide declarative deployment automation for Databricks resources. Each asset file defines:

### Resource Types
- **Jobs**: Databricks workflows with tasks, clusters, and dependencies
- **Clusters**: Compute configurations for different workloads  
- **Notebooks**: Execution paths and parameters
- **Permissions**: Access controls and security settings

### Environment Management
- **Development**: `targets.dev` for development and testing
- **Staging**: `targets.staging` for pre-production validation
- **Production**: `targets.prod` for production deployments

### Deployment Process
1. `databricks bundle validate` - Validate configuration syntax
2. `databricks bundle deploy` - Deploy resources to Databricks workspace
3. `databricks bundle run <job_name>` - Execute specific workflows

## ⚙️ Configuration Files

### Code Quality & Formatting

#### `pyproject.toml` - Black Configuration
```toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'
```

#### `.pylintrc` - Code Quality Rules  
```ini
[MESSAGES CONTROL]
disable=missing-docstring,invalid-name,too-few-public-methods
```

#### `mypy.ini` - Type Checking
```ini
[mypy]
python_version = 3.8
warn_return_any = True
strict_optional = True
```

## 🔧 Execution Commands

### Manual Execution Steps

```bash
# 1. Deploy Asset Bundles
databricks bundle deploy --target dev

# 2. Run Feature Engineering
databricks bundle run feature_engineering_job --target dev

# 3. Run Model Training  
databricks bundle run model_training_job --target dev

# 4. Deploy Model Serving
# Run serving/notebooks/OnlineTableDeployment.py in Databricks workspace

# 5. Test Serving
databricks serving-endpoints query mlops-taxi-fare-endpoint \
  --json @serving/config/test_single_prediction.json

# 6. Run Batch Inference
databricks bundle run batch_inference_job --target dev
```

### Automated Execution

```bash
# Full end-to-end pipeline
./run_e2e_mlops_pipeline.sh

# Skip specific steps
./run_e2e_mlops_pipeline.sh --skip-features --skip-training

# Help and options
./run_e2e_mlops_pipeline.sh --help
```

## 📊 Monitoring & Validation

### Endpoint Monitoring
- **Health Checks**: `databricks serving-endpoints get mlops-taxi-fare-endpoint`
- **Logs**: Available in Databricks workspace under Serving > Endpoints
- **Metrics**: Auto-capture enabled for payload logging
- **Performance**: Latency and throughput monitoring

### Model Performance
- **MLflow Tracking**: Experiment metrics and artifacts
- **Model Registry**: Version management and stage transitions
- **Feature Store**: Feature lineage and data quality
- **Auto-capture**: Request/response logging for drift detection

### Data Quality
- **Feature Validation**: Schema and data type checks
- **Data Drift**: Monitor feature distributions over time
- **Model Drift**: Compare prediction distributions
- **Alert Configuration**: Set up notifications for anomalies

## 🔗 Integration Points

### Unity Catalog Integration
- **Feature Store**: Centralized feature management with governance
- **Model Registry**: Versioned model storage with lineage tracking  
- **Online Tables**: Sub-millisecond feature serving
- **Permissions**: Fine-grained access control

### MLflow Integration  
- **Experiment Tracking**: Metrics, parameters, and artifacts
- **Model Registry**: Model versioning and stage management
- **Model Serving**: Automatic deployment from registry
- **Model Lineage**: Track data and code dependencies

### Databricks Integration
- **Asset Bundles**: Infrastructure as code deployment
- **Jobs**: Workflow orchestration and scheduling
- **Notebooks**: Interactive development and execution
- **Compute**: Auto-scaling clusters for different workloads

## 🎯 Best Practices

### Development Workflow
1. **Feature Development**: Start with feature engineering and validation
2. **Model Development**: Use feature store for training data
3. **Testing**: Validate in development environment first
4. **Deployment**: Use asset bundles for consistent deployments
5. **Monitoring**: Set up observability from day one

### Code Organization
- **Modular Design**: Separate feature, training, and serving code
- **Configuration Management**: Use external config files
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Keep documentation updated with code changes

### Operational Excellence  
- **Version Control**: Track all code and configuration changes
- **CI/CD**: Automate testing and deployment pipelines
- **Monitoring**: Implement comprehensive observability
- **Security**: Follow principle of least privilege
- **Disaster Recovery**: Regular backups and tested recovery procedures

---

For questions or support, refer to the main project documentation or Databricks workspace resources.