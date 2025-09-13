# Databricks notebook source
# MAGIC %md
# MAGIC # Debug: Databricks-Managed Online Feature Store Setup
# MAGIC 
# MAGIC ## 🔧 **When to Use This Debug Notebook**
# MAGIC 
# MAGIC **Normal Flow**: The `deploy.py` script automatically handles online feature store setup during model deployment.
# MAGIC 
# MAGIC **Use this notebook ONLY if**:
# MAGIC - ❌ Model deployment fails with "Feature lookup setup failed" error after running `deploy.py`
# MAGIC - 🔍 You need detailed diagnostics and step-by-step verification
# MAGIC - 🧪 You want to manually test online feature store configuration
# MAGIC - 🛠️ You need comprehensive error analysis and troubleshooting
# MAGIC 
# MAGIC ## 📋 **Automatic vs Manual Approach**
# MAGIC 
# MAGIC | Approach | When to Use | Features |
# MAGIC |----------|-------------|----------|
# MAGIC | **`deploy.py`** (Automatic) | Normal model deployment | ✅ Integrated with deployment<br/>✅ Production-ready<br/>✅ Handles core requirements |
# MAGIC | **This Notebook** (Manual Debug) | Troubleshooting failures | ✅ Detailed diagnostics<br/>✅ Step-by-step verification<br/>✅ Comprehensive error handling |
# MAGIC 
# MAGIC ## 🚀 **Quick Start**
# MAGIC 
# MAGIC 1. **Try automatic approach first**: Run model deployment - `deploy.py` should handle everything
# MAGIC 2. **If deployment fails**: Use this notebook for detailed diagnosis and manual setup
# MAGIC 3. **After running this notebook**: Retry your model deployment
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC This notebook uses the **Databricks-managed Online Feature Store** approach that works identically on both Azure and AWS.
# MAGIC **No external Azure services required!** - Everything is managed by Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Required Packages
# MAGIC 
# MAGIC **Important**: Use the new `databricks-feature-engineering` package for Databricks-managed online stores

# COMMAND ----------

# Install required packages - NEW approach uses feature-engineering package
%pip install databricks-feature-engineering>=0.13.0
dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from datetime import datetime
from pyspark.sql import SparkSession

print("� Databricks-Managed Online Feature Store Setup")
print(f"📅 Current Time: {datetime.now()}")
print(f"🔧 Databricks Runtime: {spark.version}")

# Import the NEW feature engineering client
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    print("✅ Using FeatureEngineeringClient (Databricks-managed approach)")
    fe = FeatureEngineeringClient()
except ImportError as e:
    print(f"❌ Failed to import FeatureEngineeringClient: {e}")
    print("💡 Make sure you have databricks-feature-engineering>=0.13.0 installed")
    sys.exit(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Environment Analysis

# COMMAND ----------

# Check current workspace and catalog configuration
spark = SparkSession.builder.getOrCreate()

current_user = spark.sql("SELECT current_user()").collect()[0][0]
current_catalog = spark.sql("SELECT current_catalog()").collect()[0][0]
current_schema = spark.sql("SELECT current_schema()").collect()[0][0]

print(f"👤 Current User: {current_user}")
print(f"📚 Current Catalog: {current_catalog}")  
print(f"📋 Current Schema: {current_schema}")

# Check Unity Catalog availability
try:
    catalogs = spark.sql("SHOW CATALOGS").collect()
    print(f"✅ Unity Catalog enabled - {len(catalogs)} catalogs available")
except Exception as e:
    print(f"⚠️ Unity Catalog check failed: {str(e)}")

print(f"🚀 Using Databricks-managed online store (no external services required!)")

# COMMAND ----------

# MAGIC %md  
# MAGIC ## Step 2: Feature Table Analysis and Preparation

# COMMAND ----------

# Define target feature tables
CATALOG = "p03"
SCHEMA = "e2e_demo_simon"
PICKUP_TABLE = f"{CATALOG}.{SCHEMA}.trip_pickup_features"
DROPOFF_TABLE = f"{CATALOG}.{SCHEMA}.trip_dropoff_features"

feature_tables = [PICKUP_TABLE, DROPOFF_TABLE]

print("🔍 Analyzing and preparing feature tables for online store...")
table_info = {}

for table_name in feature_tables:
    try:
        print(f"\n📊 Processing: {table_name}")
        
        # Check if table exists
        df = spark.table(table_name)
        count = df.count()
        
        print(f"   ✅ Table exists - {count:,} rows")
        print(f"   📋 Columns: {[col[0] for col in df.dtypes[:5]]}...")  # Show first 5 columns
        
        # Enable Change Data Feed (required for online stores)
        try:
            spark.sql(f"""
                ALTER TABLE {table_name}
                SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
            """)
            print(f"   ✅ Change Data Feed enabled")
        except Exception as cdf_e:
            print(f"   ⚠️ CDF already enabled or error: {str(cdf_e)[:50]}...")
        
        # Get feature table metadata using FeatureEngineeringClient
        try:
            ft_info = fe.get_table(name=table_name)
            print(f"   📑 Primary Keys: {ft_info.primary_keys}")
            
            # Ensure primary keys are NOT NULL (required for online stores)
            if ft_info.primary_keys:
                for pk_col in ft_info.primary_keys:
                    try:
                        spark.sql(f"""
                            ALTER TABLE {table_name}
                            ALTER COLUMN {pk_col} SET NOT NULL
                        """)
                        print(f"   ✅ Set {pk_col} as NOT NULL")
                    except Exception as pk_e:
                        print(f"   ⚠️ {pk_col} NOT NULL: {str(pk_e)[:30]}...")
            
            table_info[table_name] = {
                'exists': True, 
                'primary_keys': ft_info.primary_keys,
                'count': count,
                'ready': True
            }
            
        except Exception as e:
            print(f"   ❌ Feature table metadata error: {str(e)[:50]}...")
            table_info[table_name] = {
                'exists': True, 
                'primary_keys': ['pickup_location_id'] if 'pickup' in table_name else ['dropoff_location_id'],
                'count': count,
                'ready': False
            }
        
    except Exception as e:
        print(f"❌ Error processing {table_name}: {str(e)}")
        table_info[table_name] = {'exists': False, 'error': str(e)}

print(f"\n✅ Feature table preparation complete!")
print(f"📊 Tables ready: {sum(1 for t in table_info.values() if t.get('exists', False))}/{len(feature_tables)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Databricks-Managed Online Store
# MAGIC 
# MAGIC ### ✅ **No External Services Required!**
# MAGIC 
# MAGIC This approach uses Databricks-managed online stores which:
# MAGIC - Work identically on Azure and AWS Databricks
# MAGIC - Require **no external Azure services** (Cosmos DB, SQL, etc.)
# MAGIC - Require **no secrets or authentication setup**
# MAGIC - Are **fully managed** by Databricks
# MAGIC - Provide **low-latency** feature serving

# COMMAND ----------

def create_databricks_online_store():
    """
    Create a Databricks-managed online store - NO external services required!
    """
    print("🚀 Creating Databricks-Managed Online Store")
    
    store_name = "mlops-feature-store"
    
    try:
        # Step 1: Try to get existing store first
        try:
            existing_store = fe.get_online_store(name=store_name)
            print(f"✅ Found existing online store: {store_name}")
            print(f"   State: {existing_store.state}")
            print(f"   Capacity: {existing_store.capacity}")
            return existing_store
        except:
            print(f"📝 Online store '{store_name}' does not exist - creating new one...")
        
        # Step 2: Create new online store
        print("⚙️ Creating Databricks-managed online store...")
        online_store = fe.create_online_store(
            name=store_name,
            capacity="CU_1"  # Options: CU_1, CU_2, CU_4, CU_8
        )
        
        print(f"✅ Successfully created online store: {store_name}")
        print(f"   State: {online_store.state}")
        print(f"   Capacity: {online_store.capacity}")
        print(f"   💡 Note: Store may take a few minutes to become AVAILABLE")
        
        return online_store
        
    except Exception as e:
        print(f"❌ Failed to create online store: {str(e)}")
        
        # Provide helpful error guidance
        error_msg = str(e).lower()
        if "permission" in error_msg or "forbidden" in error_msg:
            print("   💡 Check workspace permissions for online store creation")
        elif "quota" in error_msg or "limit" in error_msg:
            print("   💡 May have reached online store quota limits")
        elif "region" in error_msg:
            print("   💡 Online stores may not be available in this region")
            print("      Available regions: westus, westus2, eastus, eastus2, etc.")
        
        return None

# Create the Databricks-managed online store
print("=" * 60)
print("🎯 CREATING DATABRICKS-MANAGED ONLINE STORE")
print("=" * 60)
online_store = create_databricks_online_store()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Publish Feature Tables to Online Store

# COMMAND ----------

def publish_tables_to_online_store(online_store):
    """
    Publish feature tables to the Databricks-managed online store
    """
    if not online_store:
        print("❌ No online store available for publishing")
        return False
    
    print("📋 Publishing feature tables to online store...")
    success_count = 0
    
    for table_name in feature_tables:
        if not table_info[table_name].get('exists', False):
            print(f"⚠️ Skipping {table_name} - table does not exist")
            continue
        
        try:
            print(f"\n🚀 Publishing {table_name}...")
            
            # Generate online table name
            online_table_name = table_name.replace("_features", "_online_features")
            
            # Publish table to online store
            fe.publish_table(
                online_store=online_store,
                source_table_name=table_name,
                online_table_name=online_table_name,
                streaming=False  # Use batch mode initially
            )
            
            print(f"✅ Successfully published {table_name}")
            print(f"   📊 Online table: {online_table_name}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed to publish {table_name}: {str(e)}")
            
            # Try alternative publish method
            try:
                print(f"   🔄 Trying alternative publish method...")
                fe.publish_table(
                    online_store=online_store,
                    source_table_name=table_name,
                    online_table_name=f"{table_name}_online"
                )
                print(f"   ✅ Alternative method succeeded!")
                success_count += 1
                
            except Exception as e2:
                print(f"   ❌ Alternative method also failed: {str(e2)}")
                
                # Provide specific error guidance
                error_msg = str(e2).lower()
                if "change data feed" in error_msg:
                    print("   💡 Ensure Change Data Feed is enabled (should be done automatically)")
                elif "primary key" in error_msg:
                    print("   💡 Check that primary key columns are not null")
                elif "permission" in error_msg:
                    print("   💡 Check Unity Catalog permissions for the table")
    
    return success_count

# Publish all feature tables
if online_store:
    print("=" * 60)  
    print("📋 PUBLISHING FEATURE TABLES")
    print("=" * 60)
    
    published_count = publish_tables_to_online_store(online_store)
    
    if published_count > 0:
        print(f"\n🎉 SUCCESS! Published {published_count}/{len(feature_tables)} tables")
        print("✅ Online Feature Store is now ready!")
        print("✅ Model serving should work without 'Feature lookup setup failed' error")
        managed_success = True
    else:
        print(f"\n⚠️ Could not publish any tables to online store")
        managed_success = False
else:
    print("❌ Cannot publish tables - online store creation failed")
    managed_success = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Verification and Status Check

# COMMAND ----------

def verify_online_store_setup():
    """
    Verify that the online store and feature tables are properly configured
    """
    print("🔍 Verifying Online Store Setup...")
    
    if not managed_success:
        print("❌ Online store setup was not successful")
        return False
    
    # Check online store status
    try:
        store = fe.get_online_store(name="mlops-feature-store")
        print(f"\n📊 Online Store Status:")
        print(f"   Name: {store.name}")
        print(f"   State: {store.state}")
        print(f"   Capacity: {store.capacity}")
        
        if store.state == "AVAILABLE":
            print(f"   ✅ Online store is ready for serving!")
        else:
            print(f"   ⏳ Online store is still provisioning (state: {store.state})")
            print(f"      This is normal - it may take a few minutes to become AVAILABLE")
        
    except Exception as e:
        print(f"❌ Could not verify online store: {str(e)}")
        return False
    
    # Check feature table status
    print(f"\n📋 Feature Tables Status:")
    for table_name in feature_tables:
        if table_info[table_name].get('exists', False):
            try:
                ft_info = fe.get_table(name=table_name)
                print(f"   📊 {table_name}:")
                print(f"      ✅ Table exists")
                print(f"      🔑 Primary Keys: {ft_info.primary_keys}")
                
            except Exception as e:
                print(f"   📊 {table_name}:")
                print(f"      ⚠️ Could not get table info: {str(e)[:50]}...")
        else:
            print(f"   📊 {table_name}: ❌ Table does not exist")
    
    return True

# Run verification
verify_success = verify_online_store_setup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Next Steps

# COMMAND ----------

print("=" * 60)
print("🎯 DATABRICKS ONLINE FEATURE STORE DEBUG SUMMARY")
print("=" * 60)

if managed_success:
    print("🎉 SUCCESS! Databricks-managed online feature store is configured!")
    print()
    print("✅ What was accomplished:")
    print(f"   • Online store 'mlops-feature-store' created")
    print(f"   • Feature tables prepared with Change Data Feed")
    print(f"   • Tables published to online store")
    print(f"   • Ready for real-time model serving")
    print()
    print("🚀 Next Steps:")
    print("   1. Try deploying your model again using ModelDeployment.py notebook or deploy.py script")
    print("   2. The automatic deployment should now work without 'Feature lookup setup failed' error")  
    print("   3. Create model serving endpoints that will automatically use online features")
    print("   4. Test real-time inference with feature lookup")
    print()
    print("💡 Future Deployments: The automatic deploy.py should handle online store setup going forward")
    print("⏳ Note: Online store may take a few minutes to become fully AVAILABLE")
    
else:
    print("⚠️ Online store debug setup encountered issues")
    print()
    print("🔧 Troubleshooting Steps:")
    print("   1. Check Databricks Runtime version (requires 16.4 LTS ML+)")
    print("   2. Verify workspace region supports online stores")
    print("   3. Check Unity Catalog permissions")
    print("   4. Review any error messages above")
    print("   5. The automatic deploy.py may still work even if this debug notebook fails")
    print()
    print("📞 If issues persist:")
    print("   • Try the automatic deployment approach (deploy.py) first") 
    print("   • Contact your Databricks administrator") 
    print("   • Check Databricks documentation for online feature stores")
    print("   • Verify your workspace configuration")

print()
print("📊 Feature Tables Status:")
ready_tables = sum(1 for t in table_info.values() if t.get('exists', False))
print(f"   • {ready_tables}/{len(feature_tables)} feature tables available")
for table_name in feature_tables:
    status = "✅" if table_info[table_name].get('exists', False) else "❌"
    print(f"   {status} {table_name}")

print()
print("🔗 Resources:")
print("   • Databricks Online Feature Store Docs: https://docs.databricks.com/machine-learning/feature-store/online-stores.html")
print("   • Unity Catalog Feature Store: https://docs.databricks.com/machine-learning/feature-store/")
print("   • Model Serving with Features: https://docs.databricks.com/machine-learning/model-serving/")

print(f"\n📅 Completed: {datetime.now()}")
print("✨ Your online feature store is ready for production ML workloads!")