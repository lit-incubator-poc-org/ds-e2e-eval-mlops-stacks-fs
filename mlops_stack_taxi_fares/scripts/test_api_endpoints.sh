#!/bin/bash

# Simple Taxi Fare Prediction API Test
# ====================================
# Tests the inference endpoint for taxi fare predictions

# Colors for output
GREEN='\033[0;32m'  
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing Taxi Fare Prediction API${NC}"
echo "================================"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${RED}ERROR: jq not found. Install with: brew install jq${NC}"
    exit 1
fi

# Check if DATABRICKS_TOKEN is set
if [ -z "$DATABRICKS_TOKEN" ]; then
    echo -e "${RED}ERROR: DATABRICKS_TOKEN environment variable not set${NC}"
    echo "TIP: Set it with: export DATABRICKS_TOKEN=your_token"
    exit 1
fi

# Serving endpoint configuration
ENDPOINT_NAME="nytaxifares"
ENDPOINT_URL="https://adb-8490988242777396.16.azuredatabricks.net/serving-endpoints/nytaxifares/invocations"

echo "Checking serving endpoint..."
echo "  Endpoint: $ENDPOINT_NAME"
echo "  URL: $ENDPOINT_URL"

# Skip endpoint status check - we'll test directly with HTTP call

echo -e "${BLUE}Testing endpoint: $ENDPOINT_NAME${NC}"

# Sample taxi trip data - format for model inference
SAMPLE_DATA='{
  "dataframe_records": [{
    "pickup_zip": "10001",
    "dropoff_zip": "10002", 
    "trip_distance": 2.5,
    "pickup_datetime": "2023-01-01T12:00:00"
  }]
}'

echo "Sending test request..."
echo "Sample data: $(echo "$SAMPLE_DATA" | jq -c .)"

# Make direct HTTP API call using curl with token authentication
RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
  -u "token:$DATABRICKS_TOKEN" \
  -X POST \
  -H "Content-Type: application/json" \
  -d "$SAMPLE_DATA" \
  "$ENDPOINT_URL" 2>&1)

# Extract HTTP status code
HTTP_CODE=$(echo "$RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$RESPONSE" | sed '/HTTP_CODE:/d')

echo "HTTP Status: $HTTP_CODE"
# Show raw response for debugging
echo "Raw Response:"
echo "$RESPONSE_BODY"
echo ""

# Check if response is valid JSON first
if ! echo "$RESPONSE_BODY" | jq empty >/dev/null 2>&1; then
  echo "ERROR: Response is not valid JSON"
  echo "Response content: $RESPONSE_BODY"
  exit 1
fi

# Check if response contains an error
if echo "$RESPONSE_BODY" | jq -e '.error_code' >/dev/null 2>&1; then
  ERROR_CODE=$(echo "$RESPONSE_BODY" | jq -r '.error_code')
  ERROR_MESSAGE=$(echo "$RESPONSE_BODY" | jq -r '.message')
  echo "ERROR: API Error: $ERROR_CODE - $ERROR_MESSAGE"
  exit 1
elif echo "$RESPONSE_BODY" | jq -e '.predictions' >/dev/null 2>&1; then
  PREDICTION=$(echo "$RESPONSE_BODY" | jq -r '.predictions[0]')
  echo "SUCCESS: Prediction successful: $PREDICTION"
else
  echo "ERROR: Unexpected response format"
  exit 1
fi

echo ""
echo -e "${GREEN}Test completed successfully!${NC}"