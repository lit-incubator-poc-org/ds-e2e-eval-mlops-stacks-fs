#!/bin/bash

# Setup script for creating virtual environment and installing dependencies
# This script creates a repeatable process for setting up the development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found in current directory. Please run this script from the project root."
    exit 1
fi

print_status "Starting environment setup..."

# Check Python version
PYTHON_VERSION=$(python3.12 --version 2>&1 | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    print_warning "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment using venv
print_status "Creating virtual environment with Python 3.12..."
python3.12 -m venv .venv

print_success "Virtual environment created!"

# Activate virtual environment and install dependencies
print_status "Installing dependencies..."
print_status "This may take a few minutes..."

# Upgrade pip first
.venv/bin/pip install --upgrade pip

# Install dependencies from requirements.txt
.venv/bin/pip install -r requirements.txt

print_success "Dependencies installed successfully!"

# Verify installation
print_status "Verifying installation..."

# Check if we can import key packages
.venv/bin/python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import pandas as pd
    print(f'pandas: {pd.__version__}')
except ImportError as e:
    print(f'Error importing pandas: {e}')
    sys.exit(1)

try:
    import numpy as np
    print(f'numpy: {np.__version__}')
except ImportError as e:
    print(f'Error importing numpy: {e}')
    sys.exit(1)

try:
    import sklearn
    print(f'scikit-learn: {sklearn.__version__}')
except ImportError as e:
    print(f'Error importing scikit-learn: {e}')
    sys.exit(1)

try:
    import mlflow
    print(f'mlflow: {mlflow.__version__}')
except ImportError as e:
    print(f'Error importing mlflow: {e}')
    sys.exit(1)

try:
    import pyarrow
    print(f'pyarrow: {pyarrow.__version__}')
except ImportError as e:
    print(f'Error importing pyarrow: {e}')
    sys.exit(1)

print('All key dependencies imported successfully!')
"

print_success "Environment setup completed successfully!"

# Display usage instructions
echo ""
echo "=================================================================="
echo "Environment Setup Complete!"
echo "=================================================================="
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run commands in the virtual environment:"
echo "  .venv/bin/python <script>"
echo "  .venv/bin/pip install <package>"
echo ""
echo "Example usage:"
echo "  source .venv/bin/activate"
echo "  python training/notebooks/TrainWithFeatureStore.py"
echo "  pytest"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""
echo "=================================================================="
