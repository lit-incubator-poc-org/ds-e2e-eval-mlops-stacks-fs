#!/usr/bin/env python3
"""
Simple deployment script for online tables deployment.
Uses the databricks CLI authentication.
"""

import subprocess
import sys
import os

def run_databricks_command(cmd):
    """Run a databricks CLI command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Exception running command: {e}")
        return None

def deploy_model():
    """Deploy the model using online tables."""
    print("🚀 Starting Online Tables Model Deployment")
    print("=" * 60)
    
    # Model details - using the latest trained model
    model_name = "p03.e2e_demo_simon.taxi_fare_regressor"
    model_version = "15"  # Use the version we know exists
    env = "dev"
    
    print(f"📋 Deployment Parameters:")
    print(f"   Model Name: {model_name}")
    print(f"   Model Version: {model_version}")
    print(f"   Environment: {env}")
    print()
    
    # Step 1: Run the OnlineTableDeployment notebook
    notebook_path = "/Workspace/Users/simon.curran@liberty-it.co.uk/OnlineTableDeployment"
    
    print("📓 Running OnlineTableDeployment notebook...")
    
    # Create parameters for the notebook
    params = {
        "env": env,
        "model_name": model_name,
        "model_version": model_version
    }
    
    # Build the databricks run command
    param_args = []
    for key, value in params.items():
        param_args.extend(["--python-named-params", f"{key}={value}"])
    
    cmd_parts = [
        "databricks", "runs", "submit",
        "--run-name", f"OnlineTableDeployment-{model_name}-v{model_version}",
        "--notebook-task", f"notebook_path={notebook_path}",
    ]
    
    # Add parameters
    if param_args:
        cmd_parts.extend(param_args)
    
    cmd = " ".join(cmd_parts)
    
    print(f"🔄 Running command: {cmd}")
    result = run_databricks_command(cmd)
    
    if result:
        print("✅ Notebook submission successful!")
        print(f"📄 Result: {result}")
        
        # Try to extract run ID if possible
        try:
            import json
            run_info = json.loads(result)
            run_id = run_info.get("run_id")
            if run_id:
                print(f"📊 Run ID: {run_id}")
                print(f"🔗 Monitor at: https://your-workspace.cloud.databricks.com/#job/{run_id}")
        except:
            pass
            
        return True
    else:
        print("❌ Notebook submission failed!")
        return False

def main():
    """Main deployment function."""
    print("🎯 MLOps Online Tables Deployment")
    print("🕒 Starting deployment process...")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("databricks.yml"):
        print("❌ Error: databricks.yml not found. Please run from the project root.")
        return 1
    
    # Run deployment
    success = deploy_model()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT INITIATED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Online tables deployment notebook has been submitted")
        print("📊 Check the Databricks workspace for execution status")
        print("💡 The deployment creates:")
        print("   - Online tables for real-time feature serving")
        print("   - Model serving endpoint with automatic feature lookup")
        print("   - Validation tests for the deployed model")
        print()
        print("🔗 Next Steps:")
        print("1. Monitor the notebook execution in Databricks UI")
        print("2. Check endpoint status once deployment completes")
        print("3. Test the endpoint with sample predictions")
        return 0
    else:
        print("\n❌ Deployment failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())