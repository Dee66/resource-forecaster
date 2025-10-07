# AWS Setup Requirements for Resource Forecaster

## AWS CLI v2.27.50+ Setup

### Installation

**macOS/Linux:**
```bash
# Download and install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version  # Should show aws-cli/2.27.50 or higher
```

**Windows:**
```powershell
# Download and run the MSI installer
# https://awscli.amazonaws.com/AWSCLIV2.msi

# Verify installation
aws --version
```

### Restricted IAM Profile Setup

#### 1. Create Dedicated IAM User

Create a dedicated IAM user for Resource Forecaster with minimal required permissions:

```bash
# Create IAM user
aws iam create-user --user-name forecaster-service

# Create access keys
aws iam create-access-key --user-name forecaster-service
```

#### 2. Required IAM Permissions

Create and attach this IAM policy to the forecaster-service user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CostExplorerAccess",
      "Effect": "Allow",
      "Action": [
        "ce:GetCostAndUsage",
        "ce:GetDimensionValues",
        "ce:GetReservationCoverage",
        "ce:GetReservationPurchaseRecommendation",
        "ce:GetReservationUtilization",
        "ce:GetSavingsPlansUtilization",
        "ce:GetSavingsPlansUtilizationDetails",
        "ce:GetSavingsPlansPurchaseRecommendation",
        "ce:GetRightsizingRecommendation"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AthenaAccess",
      "Effect": "Allow",
      "Action": [
        "athena:StartQueryExecution",
        "athena:GetQueryExecution",
        "athena:GetQueryResults",
        "athena:StopQueryExecution",
        "athena:GetWorkGroup"
      ],
      "Resource": [
        "arn:aws:athena:*:*:workgroup/primary",
        "arn:aws:athena:*:*:workgroup/forecaster-*"
      ]
    },
    {
      "Sid": "S3AccessForCURAndModels",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::*cur*",
        "arn:aws:s3:::*cur*/*",
        "arn:aws:s3:::forecaster-models-*",
        "arn:aws:s3:::forecaster-models-*/*",
        "arn:aws:s3:::forecaster-data-*",
        "arn:aws:s3:::forecaster-data-*/*",
        "arn:aws:s3:::aws-athena-query-results-*",
        "arn:aws:s3:::aws-athena-query-results-*/*"
      ]
    },
    {
      "Sid": "CloudWatchAccess",
      "Effect": "Allow",
      "Action": [
        "cloudwatch:GetMetricData",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:ListMetrics",
        "cloudwatch:PutMetricData",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    },
    {
      "Sid": "DynamoDBAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": [
        "arn:aws:dynamodb:*:*:table/forecaster-jobs-*",
        "arn:aws:dynamodb:*:*:table/forecaster-jobs-*/index/*"
      ]
    },
    {
      "Sid": "SecretsManagerAccess",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "arn:aws:secretsmanager:*:*:secret:forecaster/*"
      ]
    }
  ]
}
```

#### 3. Configure AWS Profile

```bash
# Configure AWS profile
aws configure --profile forecaster
# Enter the access key ID and secret from step 1
# Set default region (e.g., us-east-1)
# Set output format to json

# Test configuration
aws sts get-caller-identity --profile forecaster
```

#### 4. Environment Variables

Set these environment variables for the application:

```bash
export AWS_PROFILE=forecaster
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
```

### Security Best Practices

1. **Rotate Access Keys**: Rotate access keys every 90 days
2. **Use IAM Roles**: In production, use IAM roles instead of access keys
3. **Enable CloudTrail**: Monitor all API calls for audit purposes
4. **Restrict IP Access**: Add IP restrictions to IAM policies if needed
5. **Enable MFA**: Require MFA for sensitive operations

### Verification Commands

```bash
# Test Cost Explorer access
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-02 \
  --granularity DAILY \
  --metrics BlendedCost \
  --profile forecaster

# Test Athena access
aws athena list-work-groups --profile forecaster

# Test S3 access (replace with your CUR bucket)
aws s3 ls s3://your-cur-bucket --profile forecaster
```

## Troubleshooting

### Common Issues

1. **Access Denied Errors**:
   - Verify IAM policy is attached correctly
   - Check resource ARNs match your account
   - Ensure AWS CLI profile is configured

2. **Athena Query Failures**:
   - Verify CUR database and table exist
   - Check S3 permissions for query results bucket
   - Ensure workgroup permissions

3. **Cost Explorer API Limits**:
   - API has rate limits (5 requests per second)
   - Implement exponential backoff
   - Use pagination for large datasets

### Support Resources

- [AWS CLI User Guide](https://docs.aws.amazon.com/cli/latest/userguide/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Cost Explorer API Reference](https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/)
