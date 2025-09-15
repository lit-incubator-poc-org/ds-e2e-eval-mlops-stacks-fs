## Project Context
This is a **MLOps Pipeline** is deployed on Azure Databricks. The project implements end-to-end MLOps workflows with Databricks Feature Store integration.

## Technology Stack
- **Azure Databricks** with Unity Catalog
- **Databricks Feature Store** (Unity Catalog-based)
- **Databricks Feature Engineering** (`databricks-feature-engineering>=0.13.0`)
- **MLflow** for experiment tracking and model registry
- **Apache Spark** for data processing
- **Python** with PySpark
- **Databricks Asset Bundles (DABs)** for deployment automation
- **Delta Lake** for data storage

## Code Generation Guidelines

### ⚠️ CRITICAL CODE REQUIREMENTS ⚠️
**MANDATORY REQUIREMENTS FOR ALL GENERATED CODE:**

1. **NO EMOJIS OR UNICODE ICONS ANYWHERE**
   - Do NOT use ANY emojis/icons
   - DO use plain text: "ERROR:", "SUCCESS:", "INFO:", "WARNING:", "TIP:"
   - This applies to ALL code: Python, bash scripts, notebooks, documentation, comments
   - Use descriptive text labels instead of visual symbols
   - Example: Use "ERROR: Connection failed" instead of "❌ Connection failed"

2. **MANDATORY TYPE HINTS FOR ALL FUNCTIONS**
   - ALWAYS add type hints to ALL function parameters and return values
   - Import necessary types: `from typing import Dict, List, Tuple, Optional, Any, Union`
   - Use specific types when possible, `Any` only when necessary
   - Example: `def process_data(input_data: List[Dict[str, Any]], batch_size: int = 100) -> Tuple[bool, Optional[str]]:`
   - This is CRITICAL for code maintainability and is NEVER optional
   - **FAILURE TO INCLUDE TYPE HINTS WILL RESULT IN CODE REJECTION**

3. **MANDATORY FUNCTION ORGANIZATION**
   - Public functions MUST be placed at the TOP of the file
   - Private functions (prefixed with `_`) MUST be placed at the BOTTOM of the file
   - Private functions MUST be ordered in the sequence they are called
   - This improves code readability and maintainability
   - Example organization:
     ```python
     # Public functions first
     def deploy_model(model_uri: str, env: str) -> Dict[str, Any]:
         pass
     
     def create_online_tables(catalog: str, schema: str) -> List[str]:
         pass
     
     # Private functions last, in call order
     def _parse_model_uri(model_uri: str, client: Any) -> Tuple[str, str]:
         pass
     
     def _create_endpoint_config(name: str, version: str) -> Any:
         pass
     ```

### General Principles
- Follow PEP 8 Python coding standards
- **MANDATORY: Always use type hints for ALL function parameters and return values** - This is a critical requirement
- **MANDATORY: Organize functions with public functions first, private functions last in call order**
- Include comprehensive docstrings for all functions and classes
- Implement proper error handling with try-catch blocks
- Use logging instead of print statements for production code
- Follow secure coding practices, especially for secrets management
- Use meaningful variable names that reflect the taxi fare prediction domain

### Databricks-Specific Guidelines
- Use Databricks utilities (dbutils) for file system operations and secrets
- Implement Delta Lake best practices for data versioning and ACID transactions
- Use Databricks widgets for parameterization in notebooks
- Follow Databricks naming conventions for databases, tables, and jobs
- Use cluster policies and instance pools for cost optimization

### MLOps Patterns
- Use **Databricks Feature Engineering Client** (`databricks-feature-engineering>=0.13.0`) for Unity Catalog feature stores
- Implement **online feature stores** using Databricks-managed approach (no external services)
- Use **MLflow** for experiment tracking, model registry, and deployment with Unity Catalog integration
- Create reproducible pipelines with clear data lineage for taxi data processing
- Implement **Feature Lookups** for pickup/dropoff location features and temporal aggregations
- Follow **Databricks Asset Bundles** patterns for automated deployment
- Implement data quality checks and model validation for taxi fare predictions

### Azure Integration
- Use Azure Active Directory for authentication
- Integrate with Azure Key Vault for secrets management
- Use Azure Data Lake Storage for data persistence
- Implement Azure Monitor for logging and alerting
- Follow Azure resource naming conventions

### File Structure Patterns
- Follow the established structure: `feature_engineering/`, `training/`, `deployment/`, `validation/`, `monitoring/`
- Place feature computation logic in `feature_engineering/features/` modules
- Use `notebooks/` subdirectories for Databricks notebook files
- Maintain `assets/` directory for Databricks Asset Bundle workflow definitions
- Keep environment-specific configurations in `databricks.yml` with proper target separation
- Separate debug notebooks (prefix with `Debug`) from production pipeline code

### Data Engineering Best Practices
- Use **Delta Lake** format for all data storage with **Change Data Feed** enabled for online stores
- Implement proper data partitioning strategies for taxi trip data by date/location
- Use **Unity Catalog** for data governance and feature table management
- Follow medallion architecture (bronze, silver, gold layers) for data processing
- Implement proper **primary key** constraints for feature tables

### Security and Compliance
- Never hardcode credentials or sensitive information
- Use service principals for automation
- Implement proper access controls and permissions
- Follow data privacy and compliance requirements
- Use encryption for data at rest and in transit

### Testing Strategies
- Write unit tests for data transformation functions
- Implement integration tests for pipeline components
- Use data validation frameworks like Great Expectations
- Create model performance tests and benchmarks
- Implement smoke tests for deployment validation

### Type Hints Best Practices
**CRITICAL: Type hints are mandatory for all functions**
- Always import required types: `from typing import Dict, List, Tuple, Optional, Any, Union`
- Use specific types whenever possible:
  - `List[str]` instead of `List`
  - `Dict[str, Any]` for dictionaries with string keys
  - `Optional[str]` for nullable parameters
  - `Tuple[str, int]` for fixed-size tuples
- For Databricks/MLflow objects use `Any` with descriptive docstrings:
  - `workspace: Any  # Databricks WorkspaceClient`
  - `client: MlflowClient` (when imported)
- Always specify return types, use `None` for functions with no return value
- Example of proper function signature:
  ```python
  def process_features(
      feature_data: List[Dict[str, Any]], 
      catalog_name: str, 
      batch_size: Optional[int] = None
  ) -> Tuple[bool, Optional[str]]:
  ```

### Code Comments and Documentation
- Add inline comments for complex business logic
- Document data schemas and transformation logic
- Include performance considerations in comments
- Document known limitations and assumptions
- Provide clear setup and deployment instructions

### Performance Optimization
- Use appropriate Spark configurations for workloads
- Implement proper caching strategies for repeated computations
- Use broadcast joins where appropriate
- Optimize file sizes and formats for query performance
- Monitor and tune resource utilization

### Output Messages and Logging Standards
**MANDATORY: All output messages must use plain text labels**
- Use "ERROR:" prefix for error messages
- Use "SUCCESS:" prefix for successful operations  
- Use "INFO:" prefix for informational messages
- Use "WARNING:" prefix for warnings
- Use "TIP:" prefix for helpful suggestions
- Use "PROGRESS:" prefix for status updates
- Never use emojis, icons, or Unicode symbols in any output

### Specific Implementation Patterns

When generating code, prioritize maintainability, scalability, and adherence to MLOps best practices while ensuring compatibility with Azure Databricks services and APIs.