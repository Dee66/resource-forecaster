# Demo Script & Talking Points

Goal: Demonstrate how Resource Forecaster reduces cost and automates rightsizing.

1) Setup & context (30s)
- Explain the problem: cloud waste in non-prod and inefficient instance families.
- Show objective: predict cost and recommend rightsizing to reduce spend (~40% narrative target).

2) Synthesize infra (20s)
- Show that infra is provisioned with VPC-only deployment, Step Functions orchestration, and audit trail (DynamoDB).

3) Run a forecast (1 min)
- Trigger the daily forecast Step Function (or run locally) and show output:
  - Forecast vs actual chart (RMSE metric)
  - Top recommendations (rightsizing, RIs)

4) Show automation and audit
- Show RMSE alarm triggers: SNS -> retrain starter Lambda -> Step Function executes retrain
- Show a rightsizing recommendation executed in dry-run (Shutdown Lambda in DRY_RUN) and persisted to the audit table.

5) Metrics & outcome
- Point to dashboards (RMSE trending, cost-per-environment).
- Explain the fidelity gate (CI checks RMSE before promoting model) and runbook for approval.

6) Q&A (remaining time)
- Address accuracy, production safety, and how to tune thresholds.
