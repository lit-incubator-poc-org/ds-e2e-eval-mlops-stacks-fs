# Databricks notebook source
# MAGIC %md
# MAGIC # Check Databricks Runtime Version for Online Feature Store Support
# MAGIC 
# MAGIC This notebook checks if your Databricks Runtime supports Unity Catalog online feature stores.
# MAGIC 
# MAGIC **Requirement**: Databricks Runtime 16.4 LTS ML or above

# COMMAND ----------

import sys
import os

print("🔍 DATABRICKS RUNTIME VERSION CHECK")
print("=" * 50)

# Method 1: Check spark.conf for runtime version
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    # Get Databricks Runtime version from Spark configuration
    runtime_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "Unknown")
    cluster_name = spark.conf.get("spark.databricks.clusterUsageTags.clusterName", "Unknown")
    
    print(f"📊 Cluster Information:")
    print(f"   Cluster Name: {cluster_name}")
    print(f"   Runtime Version: {runtime_version}")
    
except Exception as e:
    print(f"Error getting Spark config: {str(e)}")

# COMMAND ----------

# Method 2: Check environment variables
print("\n🔧 ENVIRONMENT INFORMATION:")
print("-" * 30)

# Check DBR version from environment
dbr_version = os.environ.get('DATABRICKS_RUNTIME_VERSION', 'Not found')
print(f"DBR Environment Variable: {dbr_version}")

# Check if it's ML runtime
ml_indicator = os.environ.get('DATABRICKS_ML_RUNTIME', 'false')
print(f"ML Runtime: {ml_indicator}")

# COMMAND ----------

# Method 3: Check using dbutils (if available)
try:
    import dbutils
    notebook_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    
    # Try to get cluster information
    cluster_id = notebook_context.clusterId().get()
    print(f"\n🏗️  CLUSTER DETAILS:")
    print(f"   Cluster ID: {cluster_id}")
    
    # Get additional context information
    tags = notebook_context.tags()
    if tags:
        print("   Cluster Tags:")
        for tag in tags:
            tag_key = tag._1() if hasattr(tag, '_1') else 'unknown'
            tag_value = tag._2() if hasattr(tag, '_2') else 'unknown'
            if 'version' in str(tag_key).lower() or 'runtime' in str(tag_key).lower():
                print(f"     {tag_key}: {tag_value}")
                
except Exception as e:
    print(f"\ndbutils not available or error: {str(e)}")

# COMMAND ----------

# Method 4: Check Python and package versions
print(f"\n🐍 PYTHON ENVIRONMENT:")
print(f"   Python Version: {sys.version}")
print(f"   Python Executable: {sys.executable}")

# Check for ML-specific packages that indicate ML runtime
ml_packages = ['mlflow', 'tensorflow', 'torch', 'sklearn']
installed_ml_packages = []

for package in ml_packages:
    try:
        __import__(package)
        try:
            version = __import__(package).__version__
            installed_ml_packages.append(f"{package}=={version}")
        except:
            installed_ml_packages.append(f"{package} (version unknown)")
    except ImportError:
        pass

if installed_ml_packages:
    print(f"\n📦 ML PACKAGES DETECTED:")
    for pkg in installed_ml_packages:
        print(f"   ✓ {pkg}")
else:
    print(f"\n📦 No common ML packages detected")

# COMMAND ----------

# Method 5: Check databricks-feature-engineering package availability and version
print(f"\n🎯 FEATURE ENGINEERING PACKAGE CHECK:")
print("-" * 40)

try:
    import databricks.feature_engineering
    fe_version = getattr(databricks.feature_engineering, '__version__', 'Version not available')
    print(f"✅ databricks-feature-engineering: {fe_version}")
    
    # Try to initialize the client to test functionality
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print(f"✅ FeatureEngineeringClient initialized successfully")
    
    # Test online feature store functionality
    try:
        # This will succeed if the runtime supports online feature stores
        online_stores = fe.list_online_stores()
        print(f"✅ Online feature store API accessible")
        print(f"   Found {len(online_stores)} online store(s)")
    except Exception as e:
        print(f"⚠️  Online feature store API issue: {str(e)}")
        
except ImportError as e:
    print(f"❌ databricks-feature-engineering not available: {str(e)}")
    print("   This package is required for Unity Catalog feature stores")

# COMMAND ----------

# Method 6: Version comparison and recommendations
print(f"\n🎯 COMPATIBILITY CHECK:")
print("=" * 30)

def parse_runtime_version(version_str):
    """Parse Databricks Runtime version string."""
    if not version_str or version_str == "Unknown":
        return None
    
    try:
        # Handle formats like "16.4.x-scala2.12" or "16.4 LTS ML"
        version_str = version_str.replace(" LTS", "").replace(" ML", "")
        parts = version_str.split(".")
        if len(parts) >= 2:
            major = int(parts[0])
            minor = int(parts[1])
            return (major, minor)
    except:
        pass
    return None

# Check if runtime meets requirements
min_version = (16, 4)  # DBR 16.4 LTS ML minimum

runtime_info = []
try:
    runtime_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "Unknown")
    runtime_info.append(("Spark Config", runtime_version))
except:
    pass

if dbr_version != 'Not found':
    runtime_info.append(("Environment", dbr_version))

print("Runtime Version Sources:")
compatible_versions = []

for source, version in runtime_info:
    print(f"  {source}: {version}")
    parsed = parse_runtime_version(version)
    if parsed and parsed >= min_version:
        compatible_versions.append((source, version, parsed))

print(f"\nRequirement: Databricks Runtime 16.4 LTS ML or above")

if compatible_versions:
    print(f"✅ COMPATIBLE RUNTIME DETECTED:")
    for source, version, parsed in compatible_versions:
        print(f"   {source}: {version} (parsed as {parsed[0]}.{parsed[1]})")
    print(f"\n🎉 Your runtime should support Unity Catalog online feature stores!")
else:
    print(f"❌ RUNTIME COMPATIBILITY UNCLEAR OR INSUFFICIENT:")
    print(f"   Could not determine if runtime meets minimum requirements")
    print(f"   This may explain the 'Feature lookup setup failed' error")

# COMMAND ----------

# Method 7: Test online feature store creation
print(f"\n🧪 ONLINE FEATURE STORE TEST:")
print("-" * 35)

try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    
    # Test if we can check for existing online stores
    try:
        existing_stores = fe.list_online_stores()
        print(f"✅ Successfully listed online stores: {len(existing_stores)} found")
        
        for store in existing_stores:
            print(f"   Store: {store.name} (State: {store.state})")
            
    except Exception as e:
        print(f"❌ Cannot list online stores: {str(e)}")
        if "not supported" in str(e).lower() or "runtime" in str(e).lower():
            print(f"   This suggests your runtime version may be too old")
        
except Exception as e:
    print(f"❌ Feature engineering client error: {str(e)}")

# COMMAND ----------

print(f"\n📋 SUMMARY AND RECOMMENDATIONS:")
print("=" * 40)
print()

print("To resolve 'Feature lookup setup failed' errors:")
print()
print("1. ✅ CHECK RUNTIME VERSION:")
print("   - Ensure you're using DBR 16.4 LTS ML or newer")
print("   - ML runtime is required (not just standard runtime)")
print()
print("2. ✅ VERIFY CLUSTER CONFIGURATION:")
print("   - Use a cluster with ML capabilities")
print("   - Ensure Unity Catalog is enabled")
print()
print("3. ✅ PACKAGE REQUIREMENTS:")
print("   - databricks-feature-engineering>=0.13.0")
print("   - Unity Catalog permissions")
print()
print("4. 🔧 IF RUNTIME IS TOO OLD:")
print("   - Upgrade to DBR 16.4 LTS ML or newer")
print("   - Contact your Databricks admin")
print("   - Consider using batch inference as workaround")
print()
print("5. 💡 ALTERNATIVE SOLUTIONS:")
print("   - Use the TaxiFarePredictionService notebook")
print("   - Deploy as Databricks job instead of serving endpoint")
print("   - Batch inference works with older runtimes")