# Online Tables Deployment Guide

This guide explains how to deploy your MLOps stack with Databricks Online Tables for real-time feature serving and model inference.

## Overview

The online tables deployment approach provides:
- **Real-time feature serving** with sub-millisecond latency
- **Automatic feature lookup** during model inference
- **Unity Catalog integration** for governance and security
- **Serverless endpoints** with auto-scaling capabilities

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Feature       │    │   Online        │    │   Serving       │
│   Tables        │───▶│   Tables        │───▶│   Endpoint      │
│ (Delta Lake)    │    │ (Real-time)     │    │ (Auto Feature   │
│                 │    │                 │    │  Lookup)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
  Batch Updates          Real-time Sync         Inference API
```

## Files Created

### 1. OnlineTableDeployment.py
**Location**: `deployment/model_deployment/notebooks/OnlineTableDeployment.py`
**Purpose**: Complete notebook for setting up online tables and serving endpoints

**Key Features**:
- Automated online table creation from feature tables
- Serving endpoint deployment with feature lookup
- Validation and testing of deployed endpoints
- Comprehensive error handling and status reporting

### 2. Updated deploy.py
**Location**: `deployment/model_deployment/deploy.py`
**Purpose**: Updated deployment functions using Online Tables API

**Key Changes**:
- Replaced old `FeatureEngineeringClient.publish_table()` with Online Tables API
- Added `create_online_tables()` function following Unity Catalog patterns
- Enhanced `deploy_with_online_tables()` for end-to-end deployment
- Backward compatibility maintained with legacy `deploy()` function

### 3. test_online_tables.py
**Location**: `deployment/model_deployment/test_online_tables.py`
**Purpose**: Test suite for validating online tables deployment

## Usage Instructions

### Option 1: Using the Databricks Notebook (Recommended)

1. **Upload the notebook**:
   ```bash
   # Upload to your Databricks workspace
   databricks workspace import deployment/model_deployment/notebooks/OnlineTableDeployment.py \
     /Workspace/Users/your-user/mlops-stack/OnlineTableDeployment --language PYTHON
   ```

2. **Run the notebook** with parameters:
   - `env`: "staging" or "prod"
   - `model_name`: Your Unity Catalog model name (e.g., "p03.e2e_demo_simon.taxi_fare_regressor")
   - `model_version`: Specific version or leave empty for latest

3. **Monitor the output** for:
   - Online table creation status
   - Serving endpoint deployment
   - Validation test results

### Option 2: Using the Python Script

1. **Run deployment**:
   ```python
   from deployment.model_deployment.deploy import deploy_with_online_tables
   
   result = deploy_with_online_tables(
       model_uri="p03.e2e_demo_simon.taxi_fare_regressor/15",
       env="staging"
   )
   ```

2. **Test the deployment**:
   ```bash
   cd deployment/model_deployment
   python test_online_tables.py
   ```

### Option 3: Integration with Existing Workflow

The updated `deploy.py` maintains backward compatibility. Your existing workflow will automatically use online tables:

```python
from deployment.model_deployment.deploy import deploy

# This now uses online tables automatically
deploy(model_uri="models:/your_model/1", env="staging")
```

## Key Differences from Previous Approach

| Aspect | Old Approach | New Online Tables Approach |
|--------|-------------|---------------------------|
| **API** | `FeatureEngineeringClient.publish_table()` | `WorkspaceClient.online_tables.create()` |
| **Architecture** | Online Feature Store with separate store | Unity Catalog Online Tables |
| **Serving** | Manual feature lookup configuration | Automatic feature lookup from model metadata |
| **Latency** | ~10-50ms | <5ms (sub-millisecond possible) |
| **Scaling** | Manual capacity management | Serverless auto-scaling |
| **Integration** | Requires separate store setup | Native Unity Catalog integration |

## Configuration

### Primary Keys Configuration
The online tables are configured with appropriate primary keys for taxi data:

```python
# Pickup features table
primary_keys = ["pickup_zip", "rounded_pickup_datetime"]

# Dropoff features table  
primary_keys = ["dropoff_zip", "rounded_dropoff_datetime"]
```

### Feature Tables Expected
```
<catalog>.<schema>.trip_pickup_features
<catalog>.<schema>.trip_dropoff_features
```

### Generated Online Tables
```
<catalog>.<schema>.trip_pickup_features_online
<catalog>.<schema>.trip_dropoff_features_online
```

## Troubleshooting

### Common Issues

1. **"Feature lookup setup failed"** during serving:
   - **Cause**: Using standard runtime instead of ML runtime
   - **Solution**: Switch cluster to Databricks Runtime ML (e.g., 17.1.x ML)

2. **Online table creation fails**:
   - **Cause**: Insufficient permissions or missing source tables
   - **Solution**: Ensure `CREATE ONLINE TABLE` permissions and verify feature tables exist

3. **Endpoint creation timeout**:
   - **Cause**: Resource provisioning delays
   - **Solution**: Check endpoint status in UI, may take 5-10 minutes

4. **Inference fails with missing features**:
   - **Cause**: Online table sync delay or missing primary key data
   - **Solution**: Verify online table status and ensure test data uses valid keys

### Monitoring and Validation

1. **Check online table status**:
   ```python
   from databricks.sdk import WorkspaceClient
   
   workspace = WorkspaceClient()
   status = workspace.online_tables.get("catalog.schema.table_name_online")
   print(f"Status: {status.status}")
   ```

2. **Test endpoint manually**:
   ```python
   import mlflow.deployments
   
   client = mlflow.deployments.get_deploy_client("databricks")
   response = client.predict(
       endpoint="your_endpoint_name",
       inputs={"dataframe_records": [test_data]}
   )
   ```

3. **Monitor endpoint performance**:
   - Check Databricks serving UI for latency metrics
   - Monitor auto-scaling behavior
   - Review error logs for failed requests

## Best Practices

1. **Development Workflow**:
   - Test with small datasets first
   - Use staging environment for validation
   - Monitor costs during auto-scaling

2. **Production Deployment**:
   - Enable monitoring and alerting
   - Set up proper access controls
   - Document endpoint URLs and usage

3. **Feature Table Management**:
   - Ensure consistent primary key formats
   - Monitor feature freshness
   - Plan for schema evolution

4. **Performance Optimization**:
   - Use appropriate workload sizes
   - Enable scale-to-zero for cost savings
   - Monitor and tune auto-scaling policies

## Next Steps

After successful deployment:

1. **Integration**: Update your applications to use the serving endpoint
2. **Monitoring**: Set up alerts for endpoint availability and performance
3. **Scaling**: Monitor usage and adjust workload size as needed
4. **Governance**: Implement proper access controls and audit logging

## Support

For issues or questions:
1. Check the Databricks documentation on Online Tables
2. Review the test output for specific error messages
3. Verify Unity Catalog permissions and feature table accessibility
4. Ensure using ML runtime for serving endpoints with feature lookup