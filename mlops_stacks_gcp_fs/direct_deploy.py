#!/usr/bin/env python3
"""
Direct deployment using the deploy.py functions.
This script will deploy the model with online tables using the existing infrastructure.
"""

import sys
import os

# Add the deployment module to path
deployment_path = os.path.join(os.path.dirname(__file__), "deployment", "model_deployment")
sys.path.append(deployment_path)

try:
    from deploy import deploy_with_online_tables
    print("✅ Successfully imported deploy_with_online_tables function")
except ImportError as e:
    print(f"❌ Failed to import deployment functions: {e}")
    print("💡 Make sure you're running from the project root directory")
    sys.exit(1)

def main():
    """Deploy the model using online tables."""
    print("🚀 Direct Online Tables Deployment")
    print("=" * 50)
    
    # Model configuration
    model_name = "p03.e2e_demo_simon.taxi_fare_regressor"
    model_version = "15"  # Use known version
    model_uri = f"{model_name}/{model_version}"
    env = "dev"
    
    print(f"📋 Deployment Configuration:")
    print(f"   Model URI: {model_uri}")
    print(f"   Environment: {env}")
    print()
    
    try:
        print("🔄 Starting deployment...")
        result = deploy_with_online_tables(model_uri, env)
        
        print("\n🎉 Deployment completed successfully!")
        print("📄 Result Summary:")
        for key, value in result.items():
            print(f"   {key}: {value}")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {str(e)}")
        print("💡 This might be due to authentication or permission issues")
        print("💡 Ensure you have proper Databricks authentication configured")
        return 1

if __name__ == "__main__":
    exit(main())