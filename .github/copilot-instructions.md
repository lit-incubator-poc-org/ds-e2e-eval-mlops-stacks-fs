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

### General Principles
- Follow PEP 8 Python coding standards
- Use type hints for all function parameters and return values
- Include comprehensive docstrings for all functions and classes
- Implement proper error handling with try-catch blocks
- Use logging instead of print statements for production code
- **Never use emojis in generated code** - use plain text for messages
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

### Specific Implementation Patterns

When generating code, prioritize maintainability, scalability, and adherence to MLOps best practices while ensuring compatibility with Azure Databricks services and APIs.