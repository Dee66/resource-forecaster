# AWS Secrets Manager Integration

## Overview

Resource Forecaster **mandatorily** uses AWS Secrets Manager for all database and API credentials to ensure security compliance and centralized secret management.

## Required Secrets

### 1. Database Credentials

Create secrets for any external databases:

```bash
# Create database secret
aws secretsmanager create-secret \
  --name "forecaster/database/prod" \
  --description "Production database credentials for Resource Forecaster" \
  --secret-string '{
    "host": "your-db-host.rds.amazonaws.com",
    "port": "5432",
    "username": "forecaster_user",
    "password": "secure-password-here",
    "database": "forecaster_db",
    "ssl_mode": "require"
  }'
```

### 2. API Keys and External Service Credentials

```bash
# Create API keys secret
aws secretsmanager create-secret \
  --name "forecaster/api-keys/prod" \
  --description "API keys for external services" \
  --secret-string '{
    "openai_api_key": "sk-...",
    "slack_webhook_url": "https://hooks.slack.com/...",
    "datadog_api_key": "dd_api_key_here"
  }'
```

### 3. Authentication Tokens

```bash
# Create authentication secret
aws secretsmanager create-secret \
  --name "forecaster/auth/prod" \
  --description "Authentication tokens and JWT secrets" \
  --secret-string '{
    "jwt_secret_key": "your-jwt-secret-256-bit-key",
    "api_token_salt": "random-salt-for-api-tokens",
    "encryption_key": "32-byte-encryption-key-here"
  }'
```

## Secret Naming Convention

All secrets must follow this naming pattern:
```
forecaster/{component}/{environment}
```

Examples:
- `forecaster/database/dev`
- `forecaster/database/staging`
- `forecaster/database/prod`
- `forecaster/api-keys/prod`
- `forecaster/auth/prod`

## Application Integration

### 1. Configuration Updates

Update your configuration files to reference secrets:

```yaml
# config/prod.yml
database:
  secret_name: "forecaster/database/prod"
  secret_region: "us-east-1"

api_keys:
  secret_name: "forecaster/api-keys/prod"
  secret_region: "us-east-1"

auth:
  secret_name: "forecaster/auth/prod"
  secret_region: "us-east-1"
```

### 2. Python Code Integration

The forecaster uses this pattern to retrieve secrets:

```python
import boto3
import json
from typing import Dict, Any

class SecretsManager:
    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client('secretsmanager', region_name=region_name)
        self._cache = {}
    
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve and cache secret values."""
        if secret_name in self._cache:
            return self._cache[secret_name]
            
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_value = json.loads(response['SecretString'])
            self._cache[secret_name] = secret_value
            return secret_value
        except Exception as e:
            raise ValueError(f"Failed to retrieve secret {secret_name}: {str(e)}")
    
    def get_database_config(self, environment: str) -> Dict[str, str]:
        """Get database configuration from secrets."""
        secret_name = f"forecaster/database/{environment}"
        return self.get_secret(secret_name)
    
    def get_api_keys(self, environment: str) -> Dict[str, str]:
        """Get API keys from secrets."""
        secret_name = f"forecaster/api-keys/{environment}"
        return self.get_secret(secret_name)

# Usage in application
secrets = SecretsManager()
db_config = secrets.get_database_config("prod")
api_keys = secrets.get_api_keys("prod")
```

### 3. Environment Variables

Never store secrets in environment variables. Instead, store secret names:

```bash
# Good - reference to secret
export DB_SECRET_NAME="forecaster/database/prod"
export API_SECRET_NAME="forecaster/api-keys/prod"

# Bad - actual credentials
export DB_PASSWORD="actual-password"  # Never do this!
```

## Security Best Practices

### 1. Secret Rotation

```bash
# Enable automatic rotation for database secrets
aws secretsmanager rotate-secret \
  --secret-id "forecaster/database/prod" \
  --rotation-lambda-arn "arn:aws:lambda:us-east-1:123456789012:function:SecretsManagerRotation"
```

### 2. Access Control

Restrict access using IAM policies:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "arn:aws:secretsmanager:*:*:secret:forecaster/*"
      ],
      "Condition": {
        "StringEquals": {
          "secretsmanager:ResourceTag/Application": "ResourceForecaster"
        }
      }
    }
  ]
}
```

### 3. Tagging

Tag all secrets for organization and access control:

```bash
aws secretsmanager tag-resource \
  --secret-id "forecaster/database/prod" \
  --tags Key=Application,Value=ResourceForecaster \
         Key=Environment,Value=prod \
         Key=CostCenter,Value=MLOps \
         Key=Owner,Value=DataScience
```

### 4. Monitoring

Set up CloudWatch alarms for secret access:

```bash
# Monitor failed secret retrievals
aws logs create-log-group --log-group-name "/aws/secretsmanager/forecaster"
```

## Lambda Integration

For Lambda functions, use execution role permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:123456789012:secret:forecaster/*"
      ]
    }
  ]
}
```

## ECS/Fargate Integration

Use secrets in ECS task definitions:

```json
{
  "secrets": [
    {
      "name": "DB_CONFIG",
      "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:forecaster/database/prod"
    }
  ]
}
```

## Local Development

### 1. Development Secrets

Create separate secrets for development:

```bash
aws secretsmanager create-secret \
  --name "forecaster/database/dev" \
  --secret-string '{
    "host": "localhost",
    "port": "5432",
    "username": "dev_user",
    "password": "dev_password",
    "database": "forecaster_dev"
  }'
```

### 2. Local Configuration

```python
# For local development, you can override with environment variables
import os

class Config:
    def __init__(self, environment: str):
        self.environment = environment
        
        if environment == "local":
            # Use environment variables for local dev
            self.db_config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "username": os.getenv("DB_USER", "dev_user"),
                "password": os.getenv("DB_PASSWORD", "dev_password")
            }
        else:
            # Use Secrets Manager for all other environments
            secrets = SecretsManager()
            self.db_config = secrets.get_database_config(environment)
```

## Troubleshooting

### Common Issues

1. **Access Denied**:
   - Check IAM permissions for `secretsmanager:GetSecretValue`
   - Verify secret exists and name is correct
   - Check resource-based policies on secret

2. **Secret Not Found**:
   - Verify secret name spelling
   - Check if secret is in correct region
   - Ensure secret hasn't been deleted

3. **JSON Parse Error**:
   - Verify secret value is valid JSON
   - Check for trailing commas or syntax errors
   - Use AWS Console to validate secret format

### Debugging Commands

```bash
# List all forecaster secrets
aws secretsmanager list-secrets \
  --filters Key=tag-key,Values=Application \
  --query 'SecretList[?Tags[?Key==`Application` && Value==`ResourceForecaster`]]'

# Get secret metadata (without value)
aws secretsmanager describe-secret --secret-id "forecaster/database/prod"

# Test secret retrieval
aws secretsmanager get-secret-value --secret-id "forecaster/database/prod"
```

## Cost Optimization

1. **Secret Consolidation**: Group related secrets to minimize API calls
2. **Caching**: Cache secret values in application memory
3. **Regional Placement**: Store secrets in same region as application
4. **Cleanup**: Delete unused secrets to avoid charges

## Compliance

Secrets Manager helps meet compliance requirements:

- **SOC 2**: Centralized secret management and audit trails
- **GDPR**: Encryption at rest and in transit
- **HIPAA**: Access logging and encryption
- **PCI DSS**: Secure credential storage

All secret access is logged in CloudTrail for audit purposes.
