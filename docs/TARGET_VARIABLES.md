# Target Variable Definitions for Cost Forecasting

## Overview

Resource Forecaster supports multiple target variables for different forecasting scenarios. Each target variable is optimized for specific use cases in FinOps and capacity planning.

## Primary Target Variables

### 1. Daily Cost (Primary)

**Variable**: `daily_cost`  
**Description**: Total daily AWS cost aggregated across all services and accounts  
**Unit**: USD  
**Granularity**: Daily  
**Use Case**: Overall budget forecasting and trend analysis

```sql
-- Athena query for daily cost
SELECT 
    DATE(line_item_usage_start_date) as date,
    SUM(line_item_blended_cost) as daily_cost,
    COUNT(*) as line_items
FROM cur_table
WHERE line_item_usage_start_date >= CURRENT_DATE - INTERVAL '90' DAY
GROUP BY DATE(line_item_usage_start_date)
ORDER BY date
```

**Features**:
- Seasonality: Weekly (weekday vs weekend), Monthly (billing cycles)
- Holidays: US federal holidays impact
- External factors: Business events, deployments

### 2. Service-Level Daily Cost

**Variable**: `service_daily_cost`  
**Description**: Daily cost per AWS service (EC2, S3, RDS, etc.)  
**Unit**: USD  
**Granularity**: Daily per service  
**Use Case**: Service-specific optimization and rightsizing

```sql
-- Athena query for service daily cost
SELECT 
    DATE(line_item_usage_start_date) as date,
    product_product_name as service,
    SUM(line_item_blended_cost) as service_daily_cost
FROM cur_table
WHERE line_item_usage_start_date >= CURRENT_DATE - INTERVAL '90' DAY
GROUP BY DATE(line_item_usage_start_date), product_product_name
ORDER BY date, service
```

### 3. Hourly GPU Utilization Cost

**Variable**: `hourly_gpu_cost`  
**Description**: Hourly cost for GPU-enabled instances (ml.*, p3.*, g4.*, etc.)  
**Unit**: USD  
**Granularity**: Hourly  
**Use Case**: ML workload optimization and GPU rightsizing

```sql
-- Athena query for GPU hourly cost
SELECT 
    DATE_TRUNC('hour', line_item_usage_start_date) as hour,
    product_instance_type,
    SUM(line_item_blended_cost) as hourly_gpu_cost,
    SUM(line_item_usage_amount) as usage_hours
FROM cur_table
WHERE line_item_usage_start_date >= CURRENT_DATE - INTERVAL '30' DAY
  AND (product_instance_type LIKE 'ml.%' 
       OR product_instance_type LIKE 'p3.%'
       OR product_instance_type LIKE 'p4.%'
       OR product_instance_type LIKE 'g4.%'
       OR product_instance_type LIKE 'g5.%')
GROUP BY DATE_TRUNC('hour', line_item_usage_start_date), product_instance_type
ORDER BY hour, product_instance_type
```

### 4. Account-Level Monthly Cost

**Variable**: `account_monthly_cost`  
**Description**: Monthly cost aggregated by AWS account  
**Unit**: USD  
**Granularity**: Monthly per account  
**Use Case**: Multi-account budget planning and chargeback

```sql
-- Athena query for account monthly cost
SELECT 
    DATE_TRUNC('month', line_item_usage_start_date) as month,
    line_item_usage_account_id as account_id,
    SUM(line_item_blended_cost) as account_monthly_cost
FROM cur_table
WHERE line_item_usage_start_date >= CURRENT_DATE - INTERVAL '12' MONTH
GROUP BY DATE_TRUNC('month', line_item_usage_start_date), line_item_usage_account_id
ORDER BY month, account_id
```

## Secondary Target Variables

### 5. Resource Utilization Cost

**Variable**: `utilization_weighted_cost`  
**Description**: Cost weighted by actual resource utilization  
**Unit**: USD per utilization unit  
**Use Case**: Efficiency optimization and waste identification

### 6. Reserved Instance Coverage Cost

**Variable**: `ri_coverage_cost`  
**Description**: Cost breakdown by RI coverage (covered vs on-demand)  
**Unit**: USD  
**Use Case**: RI optimization and savings plan recommendations

### 7. Spot Instance Cost

**Variable**: `spot_instance_cost`  
**Description**: Cost for spot instances with interruption risk  
**Unit**: USD  
**Use Case**: Spot strategy optimization

## Target Variable Configuration

### Python Implementation

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class TargetVariable(Enum):
    DAILY_COST = "daily_cost"
    SERVICE_DAILY_COST = "service_daily_cost" 
    HOURLY_GPU_COST = "hourly_gpu_cost"
    ACCOUNT_MONTHLY_COST = "account_monthly_cost"
    UTILIZATION_WEIGHTED_COST = "utilization_weighted_cost"
    RI_COVERAGE_COST = "ri_coverage_cost"
    SPOT_INSTANCE_COST = "spot_instance_cost"

@dataclass
class TargetConfig:
    variable: TargetVariable
    granularity: str  # 'hourly', 'daily', 'weekly', 'monthly'
    aggregation: str  # 'sum', 'mean', 'max'
    filters: Optional[List[str]] = None
    groupby: Optional[List[str]] = None
    
    def get_query_template(self) -> str:
        """Return Athena query template for this target variable."""
        templates = {
            TargetVariable.DAILY_COST: """
                SELECT 
                    DATE(line_item_usage_start_date) as date,
                    SUM(line_item_blended_cost) as {variable}
                FROM {table}
                WHERE line_item_usage_start_date >= '{start_date}'
                GROUP BY DATE(line_item_usage_start_date)
                ORDER BY date
            """,
            TargetVariable.SERVICE_DAILY_COST: """
                SELECT 
                    DATE(line_item_usage_start_date) as date,
                    product_product_name as service,
                    SUM(line_item_blended_cost) as {variable}
                FROM {table}
                WHERE line_item_usage_start_date >= '{start_date}'
                GROUP BY DATE(line_item_usage_start_date), product_product_name
                ORDER BY date, service
            """
        }
        return templates.get(self.variable, "")

# Usage in forecaster
target_config = TargetConfig(
    variable=TargetVariable.DAILY_COST,
    granularity='daily',
    aggregation='sum'
)
```

### Configuration File

```yaml
# config/targets.yml
targets:
  primary:
    variable: "daily_cost"
    granularity: "daily"
    aggregation: "sum"
    description: "Total daily AWS cost across all services"
    
  secondary:
    - variable: "service_daily_cost"
      granularity: "daily" 
      aggregation: "sum"
      groupby: ["service"]
      description: "Daily cost per AWS service"
      
    - variable: "hourly_gpu_cost"
      granularity: "hourly"
      aggregation: "sum"
      filters: ["gpu_instances_only"]
      description: "Hourly cost for GPU instances"

prediction_windows:
  short_term: 7   # days
  medium_term: 30 # days  
  long_term: 90   # days

accuracy_targets:
  daily_cost:
    mape: 0.05  # 5% MAPE target
    rmse: 100   # $100 RMSE target
  service_daily_cost:
    mape: 0.10  # 10% MAPE target
    rmse: 50    # $50 RMSE target
```

## Model-Specific Considerations

### Prophet Model
- **Best for**: Daily and weekly seasonality patterns
- **Target variables**: daily_cost, service_daily_cost
- **Features**: Automatic holiday detection, trend changes

### Ensemble Model  
- **Best for**: Complex patterns with external features
- **Target variables**: All variables with rich feature sets
- **Features**: Utilization metrics, deployment events, business calendars

### Linear Model
- **Best for**: Simple trends and baseline comparisons
- **Target variables**: account_monthly_cost, long-term trends
- **Features**: Time-based features, moving averages

## Validation Metrics

For each target variable, track these metrics:

```python
validation_metrics = {
    'mape': 'Mean Absolute Percentage Error',
    'rmse': 'Root Mean Square Error', 
    'mae': 'Mean Absolute Error',
    'r2': 'R-squared Score',
    'directional_accuracy': 'Percentage of correct trend predictions'
}

# Quality gates
quality_gates = {
    'daily_cost': {'mape': 0.05, 'rmse': 100},
    'service_daily_cost': {'mape': 0.10, 'rmse': 50},
    'hourly_gpu_cost': {'mape': 0.15, 'rmse': 25}
}
```

## Business Context

### Cost Drivers
- **Seasonality**: End-of-quarter spending, holiday shutdowns
- **Business Events**: Product launches, marketing campaigns
- **Technical Events**: Auto-scaling, deployments, incident response
- **External Factors**: AWS pricing changes, new service adoption

### Use Case Mapping
- **Budget Planning**: daily_cost, account_monthly_cost
- **Rightsizing**: service_daily_cost, utilization_weighted_cost
- **ML Optimization**: hourly_gpu_cost
- **RI Planning**: ri_coverage_cost
- **Spot Strategy**: spot_instance_cost

## Implementation Priority

1. **Phase 1**: daily_cost (primary target)
2. **Phase 2**: service_daily_cost (service-level optimization)
3. **Phase 3**: hourly_gpu_cost (ML workload optimization) 
4. **Phase 4**: Advanced targets (RI, spot, utilization)

This target variable framework provides flexibility for different forecasting scenarios while maintaining consistency in model training and evaluation.
