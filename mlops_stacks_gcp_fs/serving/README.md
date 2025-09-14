# Model Serving Directory

This directory contains all files related to model serving with online feature store integration.

## 📁 Directory Structure

```
serving/
├── notebooks/
│   ├── OnlineTableDeployment.py     # 🎯 Complete online table and serving deployment
│   └── ValidationNotebook.py        # ✅ Comprehensive deployment validation
├── config/
│   ├── serving_endpoint_config.json # 🔧 Active serving endpoint configuration  
│   ├── test_single_prediction.json  # 📝 Single prediction test input
│   └── test_batch_predictions.json  # 📊 Batch prediction test inputs
└── README.md                        # 📖 This documentation
```

## 🚀 Quick Start

### 1. Deploy Online Tables and Serving Endpoint
```bash
# Option A: Use Databricks notebook (RECOMMENDED)
# Navigate to: serving/notebooks/OnlineTableDeployment.py
# Open in Databricks workspace and run all cells

# Option B: Use CLI with configuration file
databricks serving-endpoints create --json @serving/config/serving_endpoint_config.json
```

### 2. Test the Serving Endpoint
```bash
# Single prediction test
databricks serving-endpoints query mlops-taxi-fare-endpoint --json @serving/config/test_single_prediction.json

# Batch predictions test  
databricks serving-endpoints query mlops-taxi-fare-endpoint --json @serving/config/test_batch_predictions.json
```

### 3. Validate Deployment
```bash
# Run validation notebook
# Navigate to: serving/notebooks/ValidationNotebook.py
# Open in Databricks workspace and run all cells
```

## 📋 Configuration Files

### **Current Configurations**
- `serving_endpoint_config.json` - **Active serving endpoint configuration (mlops-taxi-fare-endpoint)**
- `test_single_prediction.json` - Single prediction test input
- `test_batch_predictions.json` - Batch prediction test inputs

### **Current Deployment Status**
- **Endpoint**: `mlops-taxi-fare-endpoint` ✅ READY  
- **Model**: `p03.e2e_demo_simon.dev_mlops_stacks_gcp_fs_model` version `18` 🆕
- **Features**: Unity Catalog online tables with sub-millisecond lookup
- **Auto-capture**: Enabled for payload logging
- **Last Updated**: Model version 18 deployed successfully (September 14, 2025)

### **Legacy Configurations** (for reference only)
- `nytaxifare_endpoint.json` - Old endpoint configuration format
- `nytaxifare_update_config.json` - Old update configuration format  
- `deployment_job.json` - Databricks job configuration for notebook execution
- `simple_deployment_job.json` - Simplified job configuration

## 🔧 Utilities

### **Active Utilities**
- `setup_serving.py` - Serving endpoint setup and management utilities
- `test_online_tables.py` - Online table testing and validation utilities
- `create_serving_model.py` - Model serving creation utilities

### **Legacy Utilities** (for reference only)
- `direct_deploy.py` - Direct deployment script (replaced by notebooks)
- `deploy_online_tables.py` - Online table deployment script (replaced by notebooks)

## 🧪 Testing Your Deployment

### Check Endpoint Status
```bash
databricks serving-endpoints get mlops-taxi-fare-endpoint
```

### Monitor Predictions
```sql
-- View recent predictions
SELECT * FROM p03.e2e_demo_simon.taxi_fare_endpoint_payload 
ORDER BY timestamp DESC LIMIT 10;
```

### Validate Online Tables
```sql
-- Check online feature tables exist
SHOW TABLES IN p03.e2e_demo_simon LIKE '*online*features*';
```

## 🚨 Troubleshooting

### Feature Lookup Setup Failed
1. Run `OnlineTableDeployment.py` notebook first
2. Check Unity Catalog permissions
3. Verify online tables exist

### Endpoint Not Responding
1. Check endpoint status: `databricks serving-endpoints get mlops-taxi-fare-endpoint`
2. View logs in Databricks workspace: Serving > mlops-taxi-fare-endpoint > Logs
3. Verify model version exists

## 🧹 Recent Cleanup Summary

**Removed Old/Unused Files:**
- ❌ `direct_deploy.py` - Used non-existent deployment paths
- ❌ `deploy_online_tables.py` - Hardcoded old model names  
- ❌ `create_serving_model.py` - Alternative approach no longer needed
- ❌ `setup_serving.py` - Used wrong endpoint name "nytaxifare"
- ❌ `test_online_tables.py` - Referenced removed functions

**Removed Legacy Configs:**
- ❌ `config/deployment_job.json` - Old model name references
- ❌ `config/simple_deployment_job.json` - Hardcoded cluster IDs
- ❌ `config/nytaxifare_*.json` - Wrong endpoint configurations

**Kept Essential Files:**
- ✅ `notebooks/OnlineTableDeployment.py` - Primary deployment notebook
- ✅ `notebooks/ValidationNotebook.py` - Deployment validation  
- ✅ `config/serving_endpoint_config.json` - Current active configuration
- ✅ `config/test_*.json` - Current test inputs

---

For complete deployment instructions, see the main [README.md](../README.md).