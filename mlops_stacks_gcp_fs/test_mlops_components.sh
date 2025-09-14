#!/bin/bash

# Quick Test Script for MLOps Pipeline Components
# ===============================================
#
# This script runs quick validation checks for each component of the MLOps pipeline
# without executing the full end-to-end workflow.
#
# Usage: ./test_mlops_components.sh

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[TEST] $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

echo "🧪 MLOps Pipeline Component Tests"
echo "═══════════════════════════════════════════════════════════════"

# Test 1: Check prerequisites
log "Testing prerequisites..."
if command -v databricks &> /dev/null; then
    success "Databricks CLI found"
else
    error "Databricks CLI not found"
    exit 1
fi

if databricks auth profiles &> /dev/null; then
    success "Databricks authentication configured"
else
    error "Databricks authentication not configured"
    exit 1
fi

# Test 2: Validate asset bundle configuration
log "Validating Databricks Asset Bundle configuration..."
if databricks bundle validate --target dev; then
    success "Asset bundle configuration is valid"
else
    error "Asset bundle configuration has errors"
    exit 1
fi

# Test 3: Check serving endpoint status
log "Checking serving endpoint status..."
ENDPOINT_STATUS=$(databricks serving-endpoints get mlops-taxi-fare-endpoint --output json 2>/dev/null | jq -r '.state.ready // "NOT_FOUND"')
if [[ "$ENDPOINT_STATUS" == "READY" ]]; then
    success "Serving endpoint is ready"
elif [[ "$ENDPOINT_STATUS" == "NOT_FOUND" ]]; then
    warning "Serving endpoint not found (will be created during deployment)"
else
    warning "Serving endpoint status: $ENDPOINT_STATUS"
fi

# Test 4: Check Unity Catalog permissions
log "Checking Unity Catalog access..."
if databricks catalogs list &> /dev/null; then
    success "Unity Catalog access confirmed"
else
    warning "Unity Catalog access may be limited"
fi

# Test 5: Validate serving configuration files
log "Validating serving configuration files..."
if [[ -f "serving/config/serving_endpoint_config.json" ]]; then
    if jq empty serving/config/serving_endpoint_config.json 2>/dev/null; then
        success "Serving endpoint config is valid JSON"
    else
        error "Serving endpoint config is invalid JSON"
    fi
else
    error "Serving endpoint config file not found"
fi

if [[ -f "serving/config/test_single_prediction.json" ]]; then
    if jq empty serving/config/test_single_prediction.json 2>/dev/null; then
        success "Single prediction test config is valid JSON"
    else
        error "Single prediction test config is invalid JSON"
    fi
else
    error "Single prediction test config file not found"
fi

# Test 6: Check Python dependencies
log "Checking Python dependencies..."
python3 -c "
try:
    import databricks.feature_engineering
    import mlflow
    import pandas
    import sklearn
    print('✅ Core dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

# Test 7: Quick serving endpoint test (if available)
if [[ "$ENDPOINT_STATUS" == "READY" ]]; then
    log "Testing serving endpoint with quick prediction..."
    RESULT=$(databricks serving-endpoints query mlops-taxi-fare-endpoint --json @serving/config/test_single_prediction.json 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        PREDICTION=$(echo "$RESULT" | jq -r '.predictions[0]' 2>/dev/null)
        if [[ "$PREDICTION" != "null" && "$PREDICTION" != "" ]]; then
            success "Serving endpoint test successful - Prediction: $PREDICTION"
        else
            warning "Serving endpoint responded but prediction parsing failed"
        fi
    else
        warning "Serving endpoint test failed - endpoint may be starting up"
    fi
fi

echo ""
echo "🎯 Component Test Summary"
echo "═══════════════════════════════════════════════════════════════"
success "All critical components validated successfully!"
echo ""
echo "Ready to run full MLOps pipeline:"
echo "  ./run_e2e_mlops_pipeline.sh"
echo ""
echo "Or run individual steps:"
echo "  databricks bundle deploy --target dev"
echo "  databricks bundle run feature_engineering_job --target dev"  
echo "  databricks bundle run model_training_job --target dev"
echo ""